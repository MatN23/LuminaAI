# Copyright (c) 2025 Matias Nielsen. All rights reserved.
# Licensed under the Custom License below.

import os
import sys
import json
import time
import math
import logging
import traceback
import gc
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# Import our custom modules
from model_manager import ModelManager, ModelConfig, TrainingConfig, ModelMetadata
from subword_transformer import SubwordTransformer, SubwordTokenizer

# Optional dependencies
try:
    import deepspeed
    from deepspeed.ops.adam import FusedAdam
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False
    deepspeed = None

try:
    from torch.cuda.amp import GradScaler, autocast
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False

try:
    from torch.utils.checkpoint import checkpoint
    CHECKPOINT_AVAILABLE = True
except ImportError:
    CHECKPOINT_AVAILABLE = False

# Setup logging
def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup comprehensive logging with file output."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_{timestamp}.log"
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, mode='w', encoding='utf-8')
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"🔧 Logging initialized - Log file: {log_file}")
    return logger

logger = setup_logging()

@dataclass
class PrecisionConfig:
    """Configuration for mixed precision training."""
    precision_type: str = "auto"  # "auto", "bf16", "fp16", "fp32"
    use_mixed_precision: bool = True
    use_loss_scaling: bool = True
    initial_scale: float = 2.**16
    growth_factor: float = 2.0
    backoff_factor: float = 0.5
    growth_interval: int = 2000

class HardwareDetector:
    """Detect hardware capabilities and optimal settings."""
    
    @staticmethod
    def detect_device() -> torch.device:
        """Detect and configure the best available device."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name()
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            logger.info(f"🔥 Using CUDA device: {gpu_name}")
            logger.info(f"📊 GPU Memory: {total_memory:.1f}GB")
            
            # Set conservative memory fraction
            torch.cuda.set_per_process_memory_fraction(0.85)
            torch.cuda.empty_cache()
            
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("🍎 Using MPS (Apple Silicon)")
            torch.mps.empty_cache()
            
        else:
            device = torch.device("cpu")
            logger.info("💻 Using CPU")
            torch.set_num_threads(min(8, os.cpu_count() or 1))
        
        return device

    @staticmethod
    def detect_precision_capabilities(device: torch.device) -> PrecisionConfig:
        """Detect optimal precision settings for the hardware."""
        precision_config = PrecisionConfig()
        
        if device.type == 'cuda' and torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            
            # Check for BF16 support (Ampere and newer)
            if major >= 8:
                precision_config.precision_type = "bf16"
                precision_config.use_loss_scaling = False  # BF16 doesn't need loss scaling
                logger.info("✅ BF16 precision detected - Maximum stability")
                
            elif major >= 7:  # Volta/Turing - FP16 support
                precision_config.precision_type = "fp16"
                precision_config.use_loss_scaling = True
                logger.info("⚡ FP16 precision detected - High efficiency")
                
            else:  # Older GPUs
                precision_config.precision_type = "fp32"
                precision_config.use_mixed_precision = False
                logger.info("🐌 FP32 precision - Maximum compatibility")
                
        else:
            # Non-CUDA devices
            precision_config.precision_type = "fp32"
            precision_config.use_mixed_precision = False
            logger.info("💻 FP32 precision - CPU/MPS mode")
        
        return precision_config

    @staticmethod
    def get_optimal_config(device: torch.device, target_memory_gb: float = None) -> Tuple[ModelConfig, TrainingConfig]:
        """Get optimal model and training configuration based on hardware."""
        precision_config = HardwareDetector.detect_precision_capabilities(device)
        
        if device.type == 'cuda':
            # Get available memory
            props = torch.cuda.get_device_properties(0)
            total_memory_gb = props.total_memory / 1024**3
            available_memory_gb = target_memory_gb or (total_memory_gb * 0.8)
            
            if available_memory_gb >= 40:  # High-end GPUs (A100, H100)
                model_config = ModelConfig(
                    vocab_size=50000,
                    hidden_size=4096,
                    num_layers=32,
                    num_heads=32,
                    seq_length=2048,
                    dropout=0.1
                )
                training_config = TrainingConfig(
                    learning_rate=1e-4,
                    batch_size=16,
                    gradient_accumulation_steps=2,
                    max_epochs=20,
                    warmup_ratio=0.1
                )
                
            elif available_memory_gb >= 20:  # Mid-range GPUs (RTX 4090, A6000)
                model_config = ModelConfig(
                    vocab_size=32000,
                    hidden_size=2048,
                    num_layers=24,
                    num_heads=16,
                    seq_length=1024,
                    dropout=0.1
                )
                training_config = TrainingConfig(
                    learning_rate=1e-4,
                    batch_size=8,
                    gradient_accumulation_steps=4,
                    max_epochs=25,
                    warmup_ratio=0.1
                )
                
            elif available_memory_gb >= 8:  # Consumer GPUs (RTX 3080, 4070)
                model_config = ModelConfig(
                    vocab_size=16000,
                    hidden_size=1024,
                    num_layers=12,
                    num_heads=8,
                    seq_length=512,
                    dropout=0.1
                )
                training_config = TrainingConfig(
                    learning_rate=2e-4,
                    batch_size=4,
                    gradient_accumulation_steps=8,
                    max_epochs=30,
                    warmup_ratio=0.05
                )
                
            else:  # Low VRAM GPUs
                model_config = ModelConfig(
                    vocab_size=8000,
                    hidden_size=512,
                    num_layers=6,
                    num_heads=4,
                    seq_length=256,
                    dropout=0.1
                )
                training_config = TrainingConfig(
                    learning_rate=3e-4,
                    batch_size=2,
                    gradient_accumulation_steps=16,
                    max_epochs=40,
                    warmup_ratio=0.05
                )
                
        else:  # CPU or MPS
            model_config = ModelConfig(
                vocab_size=4000,
                hidden_size=256,
                num_layers=4,
                num_heads=4,
                seq_length=128,
                dropout=0.1
            )
            training_config = TrainingConfig(
                learning_rate=5e-4,
                batch_size=2,
                gradient_accumulation_steps=8,
                max_epochs=50,
                warmup_ratio=0.1
            )
        
        logger.info(f"📊 Optimal config selected:")
        logger.info(f"   Model: {model_config.hidden_size}d x {model_config.num_layers}L")
        logger.info(f"   Vocab: {model_config.vocab_size:,}")
        logger.info(f"   Sequence: {model_config.seq_length}")
        logger.info(f"   Batch: {training_config.batch_size}")
        logger.info(f"   Precision: {precision_config.precision_type}")
        
        return model_config, training_config

@contextmanager
def memory_cleanup():
    """Aggressive memory cleanup context manager."""
    try:
        yield
    finally:
        # Python garbage collection
        for _ in range(3):
            gc.collect()
        
        # Device-specific cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()

class OptimizedDataset(Dataset):
    """Memory-efficient dataset using the SubwordTokenizer."""
    
    def __init__(self, texts: List[str], tokenizer: SubwordTokenizer, 
                 seq_length: int, max_samples: int = None):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.pad_token_id = tokenizer.vocab.get("<pad>", 0)
        self.bos_token_id = tokenizer.vocab.get("<s>", 2)
        self.eos_token_id = tokenizer.vocab.get("</s>", 3)
        
        logger.info(f"📦 Creating dataset with seq_length={seq_length}")
        
        self.sequences = []
        
        # Process texts in chunks to manage memory
        chunk_size = 1000
        max_sequences = max_samples or len(texts)
        
        for i in range(0, len(texts), chunk_size):
            if len(self.sequences) >= max_sequences:
                break
                
            chunk = texts[i:i+chunk_size]
            
            for text in chunk:
                if len(self.sequences) >= max_sequences:
                    break
                    
                if not text or len(text.strip()) < 10:
                    continue
                
                try:
                    # Tokenize text
                    tokens = tokenizer.encode(text.strip())
                    if len(tokens) < 3:
                        continue
                    
                    # Create input-output sequence
                    # Add BOS at start, truncate to fit EOS
                    full_sequence = [self.bos_token_id] + tokens[:seq_length-2] + [self.eos_token_id]
                    
                    # Pad to sequence length + 1 (for input/target pairs)
                    if len(full_sequence) <= seq_length:
                        full_sequence.extend([self.pad_token_id] * (seq_length + 1 - len(full_sequence)))
                    else:
                        full_sequence = full_sequence[:seq_length + 1]
                    
                    self.sequences.append(full_sequence)
                    
                except Exception as e:
                    logger.debug(f"Failed to process text: {e}")
                    continue
            
            # Cleanup chunk and report progress
            del chunk
            if i % 5000 == 0:
                logger.info(f"   Processed {i:,}/{len(texts):,} texts, {len(self.sequences):,} sequences")
                gc.collect()
        
        if not self.sequences:
            raise ValueError("No valid sequences created from the input texts!")
        
        logger.info(f"✅ Dataset created: {len(self.sequences):,} sequences")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        # Input is sequence[:-1], target is sequence[1:]
        input_ids = torch.tensor(sequence[:-1], dtype=torch.long)
        target_ids = torch.tensor(sequence[1:], dtype=torch.long)
        return input_ids, target_ids

class DeepSpeedManager:
    """Manage DeepSpeed initialization and configuration."""
    
    @staticmethod
    def create_config(training_config: TrainingConfig, precision_config: PrecisionConfig) -> Dict[str, Any]:
        """Create DeepSpeed configuration."""
        config = {
            "train_batch_size": training_config.batch_size * training_config.gradient_accumulation_steps,
            "train_micro_batch_size_per_gpu": training_config.batch_size,
            "gradient_accumulation_steps": training_config.gradient_accumulation_steps,
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": training_config.learning_rate,
                    "betas": [training_config.beta1, training_config.beta2],
                    "eps": training_config.eps,
                    "weight_decay": training_config.weight_decay
                }
            },
            "scheduler": {
                "type": "WarmupDecayLR",
                "params": {
                    "warmup_min_lr": 0,
                    "warmup_max_lr": training_config.learning_rate,
                    "warmup_num_steps": "auto",
                    "total_num_steps": "auto"
                }
            },
            "gradient_clipping": training_config.max_grad_norm,
            "steps_per_print": 10,
            "wall_clock_breakdown": False,
            "dump_state": False
        }
        
        # Precision configuration
        if precision_config.precision_type == "bf16":
            config["bf16"] = {"enabled": True}
            config["fp16"] = {"enabled": False}
            
        elif precision_config.precision_type == "fp16":
            config["fp16"] = {
                "enabled": True,
                "loss_scale": 0,  # Dynamic
                "loss_scale_window": precision_config.growth_interval,
                "initial_scale_power": int(math.log2(precision_config.initial_scale)),
                "hysteresis": 2,
                "min_loss_scale": 1
            }
            config["bf16"] = {"enabled": False}
            
        else:  # fp32
            config["fp16"] = {"enabled": False}
            config["bf16"] = {"enabled": False}
        
        # ZeRO optimization
        config["zero_optimization"] = {
            "stage": 2,  # ZeRO-2 is usually optimal
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
            },
            "overlap_comm": True,
            "contiguous_gradients": True,
            "sub_group_size": 1e9,
            "reduce_bucket_size": "auto"
        }
        
        return config
    
    @staticmethod
    def initialize(model: nn.Module, training_config: TrainingConfig, 
                  precision_config: PrecisionConfig) -> Optional[Any]:
        """Initialize DeepSpeed engine."""
        if not DEEPSPEED_AVAILABLE:
            logger.warning("DeepSpeed not available")
            return None
        
        try:
            # Create config
            ds_config = DeepSpeedManager.create_config(training_config, precision_config)
            
            # Write config file
            config_path = "deepspeed_config.json"
            with open(config_path, 'w') as f:
                json.dump(ds_config, f, indent=2)
            
            logger.info(f"🔧 DeepSpeed config written: {config_path}")
            
            # Initialize engine
            engine, optimizer, _, scheduler = deepspeed.initialize(
                model=model,
                config=config_path,
                model_parameters=model.parameters()
            )
            
            logger.info("✅ DeepSpeed engine initialized")
            return engine
            
        except Exception as e:
            logger.error(f"❌ DeepSpeed initialization failed: {e}")
            return None

class TrainingMetrics:
    """Track and manage training metrics."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.total_loss = 0.0
        self.total_correct = 0
        self.total_tokens = 0
        self.num_batches = 0
        self.start_time = time.time()
    
    def update(self, loss: float, predictions: torch.Tensor, targets: torch.Tensor, pad_token_id: int = 0):
        """Update metrics with batch results."""
        if torch.isnan(torch.tensor(loss)) or torch.isinf(torch.tensor(loss)):
            return False
        
        # Calculate accuracy
        mask = (targets != pad_token_id)
        correct = ((predictions == targets) & mask).sum().item()
        valid_tokens = mask.sum().item()
        
        self.total_loss += loss
        self.total_correct += correct
        self.total_tokens += valid_tokens
        self.num_batches += 1
        
        return True
    
    def get_metrics(self) -> Dict[str, float]:
        """Get current averaged metrics."""
        if self.num_batches == 0:
            return {"loss": float('inf'), "accuracy": 0.0, "perplexity": float('inf')}
        
        avg_loss = self.total_loss / self.num_batches
        accuracy = self.total_correct / max(self.total_tokens, 1)
        perplexity = math.exp(min(avg_loss, 10))  # Cap to prevent overflow
        
        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "perplexity": perplexity,
            "time": time.time() - self.start_time
        }

class EnhancedTrainer:
    """Enhanced trainer using proper model management and tokenization."""
    
    def __init__(self, model_config: ModelConfig, training_config: TrainingConfig,
                 precision_config: PrecisionConfig, device: torch.device):
        self.model_config = model_config
        self.training_config = training_config
        self.precision_config = precision_config
        self.device = device
        
        # Initialize model manager
        self.model_manager = ModelManager("models")
        
        # Initialize tokenizer
        self.tokenizer = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.deepspeed_engine = None
        
        logger.info("🚀 Enhanced trainer initialized")
    
    def prepare_tokenizer(self, texts: List[str]) -> SubwordTokenizer:
        """Prepare and train the SubwordTokenizer."""
        logger.info("🔤 Training SubwordTokenizer...")
        
        # Create tokenizer
        self.tokenizer = SubwordTokenizer()
        
        # Prepare training text (sample for efficiency)
        max_training_chars = 10_000_000  # 10M chars max
        training_text = ""
        char_count = 0
        
        for text in texts:
            if char_count >= max_training_chars:
                break
            training_text += text + "\n"
            char_count += len(text)
        
        # Train tokenizer
        self.tokenizer.train_from_text(
            training_text, 
            vocab_size=self.model_config.vocab_size,
            min_freq=2
        )
        
        # Update model config with actual vocab size
        actual_vocab_size = self.tokenizer.vocab_size()
        self.model_config.vocab_size = actual_vocab_size
        
        logger.info(f"✅ Tokenizer trained: {actual_vocab_size:,} tokens")
        
        return self.tokenizer
    
    def prepare_model(self) -> SubwordTransformer:
        """Prepare the SubwordTransformer model."""
        logger.info("🧠 Creating SubwordTransformer...")
        
        # Create model
        self.model = SubwordTransformer(self.model_config)
        self.model = self.model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        model_size_mb = total_params * 4 / (1024 * 1024)  # Assume float32
        
        logger.info(f"✅ Model created:")
        logger.info(f"   Total parameters: {total_params:,}")
        logger.info(f"   Trainable parameters: {trainable_params:,}")
        logger.info(f"   Model size: {model_size_mb:.1f}MB")
        
        return self.model
    
    def prepare_training_components(self):
        """Prepare optimizer, scheduler, and other training components."""
        if self.model is None:
            raise ValueError("Model must be prepared first")
        
        # Try to initialize DeepSpeed first
        if DEEPSPEED_AVAILABLE and self.device.type == 'cuda':
            logger.info("🔧 Attempting DeepSpeed initialization...")
            self.deepspeed_engine = DeepSpeedManager.initialize(
                self.model, self.training_config, self.precision_config
            )
            
            if self.deepspeed_engine is not None:
                logger.info("✅ DeepSpeed enabled")
                return
        
        # Fallback to regular training setup
        logger.info("🔧 Setting up regular training components...")
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.training_config.learning_rate,
            betas=(self.training_config.beta1, self.training_config.beta2),
            eps=self.training_config.eps,
            weight_decay=self.training_config.weight_decay
        )
        
        # Scheduler
        total_steps = 0  # Will be set when dataloader is ready
        warmup_steps = 0
        
        from torch.optim.lr_scheduler import OneCycleLR
        # Temporary scheduler - will be recreated with proper steps
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.training_config.learning_rate,
            total_steps=1000,
            pct_start=self.training_config.warmup_ratio
        )
        
        # Mixed precision scaler
        if self.precision_config.use_mixed_precision and AMP_AVAILABLE:
            if self.precision_config.precision_type == "fp16":
                self.scaler = GradScaler(
                    init_scale=self.precision_config.initial_scale,
                    growth_factor=self.precision_config.growth_factor,
                    backoff_factor=self.precision_config.backoff_factor,
                    growth_interval=self.precision_config.growth_interval
                )
            else:  # bf16 doesn't need scaling
                self.scaler = GradScaler(enabled=False)
            
            logger.info(f"✅ Mixed precision enabled: {self.precision_config.precision_type}")
    
    def create_dataloader(self, dataset: Dataset, shuffle: bool = True) -> DataLoader:
        """Create an optimized DataLoader."""
        return DataLoader(
            dataset,
            batch_size=self.training_config.batch_size,
            shuffle=shuffle,
            num_workers=0,  # Keep simple for stability
            pin_memory=self.device.type == 'cuda',
            drop_last=True
        )
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        metrics = TrainingMetrics()
        
        # Set model to training mode
        if self.deepspeed_engine is not None:
            self.deepspeed_engine.train()
            model = self.deepspeed_engine
        else:
            self.model.train()
            model = self.model
            if self.optimizer is not None:
                self.optimizer.zero_grad()
        
        criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.vocab.get("<pad>", 0))
        accumulation_count = 0
        
        logger.info(f"🏃 Starting epoch {epoch} with {len(dataloader)} batches")
        
        for batch_idx, (input_ids, target_ids) in enumerate(dataloader):
            try:
                # Move to device
                if self.deepspeed_engine is not None:
                    input_ids = input_ids.to(self.deepspeed_engine.local_rank)
                    target_ids = target_ids.to(self.deepspeed_engine.local_rank)
                else:
                    input_ids = input_ids.to(self.device)
                    target_ids = target_ids.to(self.device)
                
                # Forward pass
                if self.deepspeed_engine is not None:
                    # DeepSpeed handles mixed precision internally
                    logits = model(input_ids)
                    loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
                    
                    # DeepSpeed backward and step
                    model.backward(loss)
                    model.step()
                    
                else:
                    # Regular training with manual mixed precision
                    if self.precision_config.use_mixed_precision and self.scaler is not None:
                        with autocast(dtype=torch.bfloat16 if self.precision_config.precision_type == "bf16" else torch.float16):
                            logits = model(input_ids)
                            loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
                        
                        # Scale loss for gradient accumulation
                        scaled_loss = loss / self.training_config.gradient_accumulation_steps
                        self.scaler.scale(scaled_loss).backward()
                        
                    else:
                        # Regular forward pass
                        logits = model(input_ids)
                        loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
                        scaled_loss = loss / self.training_config.gradient_accumulation_steps
                        scaled_loss.backward()
                    
                    accumulation_count += 1
                    
                    # Optimizer step
                    if accumulation_count >= self.training_config.gradient_accumulation_steps:
                        if self.scaler is not None:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), self.training_config.max_grad_norm)
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), self.training_config.max_grad_norm)
                            self.optimizer.step()
                        
                        if self.scheduler is not None:
                            self.scheduler.step()
                        
                        self.optimizer.zero_grad()
                        accumulation_count = 0
                
                # Update metrics
                with torch.no_grad():
                    predictions = torch.argmax(logits, dim=-1)
                    metrics.update(loss.item(), predictions, target_ids, self.tokenizer.vocab.get("<pad>", 0))
                
                # Clean up tensors
                del input_ids, target_ids, logits, loss, predictions
                
                # Periodic logging
                if batch_idx % 50 == 0:
                    current_metrics = metrics.get_metrics()
                    lr = (self.deepspeed_engine.get_lr()[0] if self.deepspeed_engine is not None 
                         else self.optimizer.param_groups[0]['lr'] if self.optimizer is not None else 0.0)
                    
                    logger.info(
                        f"Epoch {epoch} | Batch {batch_idx:4d}/{len(dataloader)} | "
                        f"Loss: {current_metrics['loss']:.4f} | "
                        f"Acc: {current_metrics['accuracy']:.3f} | "
                        f"PPL: {current_metrics['perplexity']:.2f} | "
                        f"LR: {lr:.2e}"
                    )
                
                # Memory cleanup every 100 batches
                if batch_idx % 100 == 0:
                    with memory_cleanup():
                        pass
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning(f"OOM at batch {batch_idx}, skipping...")
                    
                    # Emergency cleanup
                    if self.optimizer is not None:
                        self.optimizer.zero_grad()
                    
                    with memory_cleanup():
                        pass
                    continue
                else:
                    raise e
            
            except Exception as e:
                logger.warning(f"Error in batch {batch_idx}: {e}")
                continue
        
        # Final gradient step if needed (for non-DeepSpeed)
        if self.deepspeed_engine is None and accumulation_count > 0:
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.training_config.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.training_config.max_grad_norm)
                self.optimizer.step()
            self.optimizer.zero_grad()
        
        return metrics.get_metrics()
    
    def evaluate(self, dataloader: DataLoader, max_batches: int = 10) -> Dict[str, float]:
        """Evaluate the model."""
        metrics = TrainingMetrics()
        
        # Set to evaluation mode
        if self.deepspeed_engine is not None:
            self.deepspeed_engine.eval()
            model = self.deepspeed_engine
        else:
            self.model.eval()
            model = self.model
        
        criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.vocab.get("<pad>", 0))
        
        try:
            with torch.no_grad():
                for batch_idx, (input_ids, target_ids) in enumerate(dataloader):
                    if batch_idx >= max_batches:
                        break
                    
                    try:
                        # Move to device
                        if self.deepspeed_engine is not None:
                            input_ids = input_ids.to(self.deepspeed_engine.local_rank)
                            target_ids = target_ids.to(self.deepspeed_engine.local_rank)
                        else:
                            input_ids = input_ids.to(self.device)
                            target_ids = target_ids.to(self.device)
                        
                        # Forward pass
                        if self.precision_config.use_mixed_precision and self.deepspeed_engine is None:
                            with autocast(dtype=torch.bfloat16 if self.precision_config.precision_type == "bf16" else torch.float16):
                                logits = model(input_ids)
                                loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
                        else:
                            logits = model(input_ids)
                            loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
                        
                        # Update metrics
                        predictions = torch.argmax(logits, dim=-1)
                        metrics.update(loss.item(), predictions, target_ids, self.tokenizer.vocab.get("<pad>", 0))
                        
                        # Clean up
                        del input_ids, target_ids, logits, loss, predictions
                        
                    except Exception as e:
                        logger.debug(f"Error in eval batch {batch_idx}: {e}")
                        continue
        
        finally:
            # Return to training mode
            if self.deepspeed_engine is not None:
                self.deepspeed_engine.train()
            else:
                self.model.train()
        
        return metrics.get_metrics()
    
    def generate_sample(self, prompt: str = "<user> Hello", max_length: int = 50) -> str:
        """Generate a sample text."""
        if self.deepspeed_engine is not None:
            self.deepspeed_engine.eval()
            model = self.deepspeed_engine
        else:
            self.model.eval()
            model = self.model
        
        try:
            with torch.no_grad():
                # Encode prompt
                input_ids = torch.tensor(
                    self.tokenizer.encode(prompt), 
                    dtype=torch.long
                ).unsqueeze(0)
                
                # Move to device
                if self.deepspeed_engine is not None:
                    input_ids = input_ids.to(self.deepspeed_engine.local_rank)
                else:
                    input_ids = input_ids.to(self.device)
                
                # Generate using model's generate method
                if hasattr(model, 'generate'):
                    generated = model.generate(
                        input_ids,
                        max_length=max_length,
                        temperature=0.8,
                        top_k=50,
                        top_p=0.9,
                        do_sample=True,
                        pad_token_id=self.tokenizer.vocab.get("<pad>", 0)
                    )
                else:
                    # Simple greedy generation
                    generated = input_ids.clone()
                    for _ in range(max_length):
                        if generated.size(1) >= self.model_config.seq_length:
                            break
                        
                        logits = model(generated)
                        next_token = torch.argmax(logits[0, -1, :], dim=-1)
                        generated = torch.cat([generated, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
                        
                        # Stop on EOS
                        if next_token.item() == self.tokenizer.vocab.get("</s>", -1):
                            break
                
                # Decode response (skip the input part)
                response_ids = generated[0][input_ids.size(1):].tolist()
                response = self.tokenizer.decode(response_ids)
                
                return response.strip()
        
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return "Generation failed"
        
        finally:
            if self.deepspeed_engine is not None:
                self.deepspeed_engine.train()
            else:
                self.model.train()
    
    def save_model(self, epoch: int, metrics: Dict[str, float], is_best: bool = False) -> Optional[str]:
        """Save model using ModelManager."""
        try:
            # Prepare metadata
            metadata = ModelMetadata(
                model_name=f"SubwordTransformer_{self.precision_config.precision_type.upper()}",
                version=f"v1.0_epoch_{epoch}",
                created_at=datetime.now().isoformat(),
                last_modified=datetime.now().isoformat(),
                model_config=asdict(self.model_config),
                training_config=asdict(self.training_config),
                performance_metrics=metrics,
                model_size_mb=sum(p.numel() for p in self.model.parameters()) * 4 / (1024 * 1024),
                total_parameters=sum(p.numel() for p in self.model.parameters()),
                trainable_parameters=sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                epochs_trained=epoch,
                best_loss=metrics.get('loss', float('inf')),
                best_perplexity=metrics.get('perplexity', float('inf')),
                hardware_used=f"{self.device.type.upper()} ({torch.cuda.get_device_name() if torch.cuda.is_available() else 'N/A'})",
                pytorch_version=torch.__version__,
                cuda_version=torch.version.cuda if torch.cuda.is_available() else None,
                notes=f"Training epoch {epoch} with {self.precision_config.precision_type} precision" + (
                    " using DeepSpeed" if self.deepspeed_engine is not None else ""
                ),
                tags=[
                    f"epoch_{epoch}",
                    f"{self.precision_config.precision_type}_precision",
                    "subword_transformer"
                ] + (["best"] if is_best else []) + (["deepspeed"] if self.deepspeed_engine is not None else [])
            )
            
            # Save using ModelManager
            if self.deepspeed_engine is not None:
                # DeepSpeed saves differently
                model_id = self.model_manager.save_model(
                    self.model, 
                    self.tokenizer, 
                    metadata,
                    deepspeed_engine=self.deepspeed_engine
                )
            else:
                model_id = self.model_manager.save_model(
                    self.model, 
                    self.tokenizer, 
                    metadata,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler
                )
            
            if model_id:
                logger.info(f"💾 Model saved: {model_id}")
            
            return model_id
            
        except Exception as e:
            logger.error(f"❌ Failed to save model: {e}")
            return None

def load_oasst_data(data_path: str, max_samples: int = None) -> List[str]:
    """Load and process OASST1 dataset."""
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    logger.info(f"📂 Loading data from: {data_path}")
    
    texts = []
    processed_count = 0
    
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if max_samples and processed_count >= max_samples:
                    break
                
                try:
                    line = line.strip()
                    if not line:
                        continue
                    
                    record = json.loads(line)
                    
                    # Skip deleted or non-English records
                    if record.get("deleted", False) or record.get("lang") != "en":
                        continue
                    
                    text = record.get("text", "").strip()
                    if not text or len(text.split()) < 3:
                        continue
                    
                    # Format with role information
                    role = record.get("role", "").lower()
                    if role == "prompter":
                        formatted_text = f"<user> {text}"
                    elif role == "assistant":
                        formatted_text = f"<assistant> {text}"
                    else:
                        formatted_text = text
                    
                    texts.append(formatted_text)
                    processed_count += 1
                    
                    # Progress logging
                    if line_num % 10000 == 0:
                        logger.info(f"   Processed {line_num:,} lines, {processed_count:,} valid texts")
                    
                except (json.JSONDecodeError, KeyError) as e:
                    logger.debug(f"Skipping invalid line {line_num}: {e}")
                    continue
    
    except Exception as e:
        logger.error(f"❌ Error loading data: {e}")
        raise
    
    logger.info(f"✅ Loaded {len(texts):,} texts from {data_path}")
    return texts

def get_memory_usage() -> str:
    """Get formatted memory usage string."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        return f"GPU: {allocated:.2f}GB used, {reserved:.2f}GB reserved, {max_allocated:.2f}GB peak"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        allocated = torch.mps.current_allocated_memory() / 1024**3
        return f"MPS: {allocated:.2f}GB used"
    else:
        return "CPU mode"

def setup_scheduler_with_proper_steps(optimizer, total_steps: int, warmup_ratio: float):
    """Setup scheduler with proper total steps."""
    from torch.optim.lr_scheduler import OneCycleLR
    
    return OneCycleLR(
        optimizer,
        max_lr=optimizer.param_groups[0]['lr'],
        total_steps=total_steps,
        pct_start=warmup_ratio,
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=10000.0
    )

def main():
    """Enhanced main training function."""
    logger.info("🚀 Enhanced SubwordTransformer Training")
    logger.info("=" * 60)
    
    # Check environment
    logger.info("🔍 Environment Check:")
    logger.info(f"   Python: {sys.version}")
    logger.info(f"   PyTorch: {torch.__version__}")
    logger.info(f"   CUDA Available: {torch.cuda.is_available()}")
    logger.info(f"   AMP Available: {AMP_AVAILABLE}")
    logger.info(f"   DeepSpeed Available: {DEEPSPEED_AVAILABLE}")
    
    # Detect hardware and get optimal configuration
    device = HardwareDetector.detect_device()
    model_config, training_config = HardwareDetector.get_optimal_config(device)
    precision_config = HardwareDetector.detect_precision_capabilities(device)
    
    # Initialize trainer
    trainer = EnhancedTrainer(model_config, training_config, precision_config, device)
    
    # Load data
    data_path = "oasst1_data/oasst1_train.jsonl"
    if not Path(data_path).exists():
        logger.error(f"❌ Data file not found: {data_path}")
        logger.info("Please ensure the OASST1 dataset is available at the specified path.")
        return 1
    
    # Load with reasonable limits based on hardware
    if device.type == 'cuda':
        max_samples = min(100000, training_config.max_epochs * 1000)  # Reasonable limit
    else:
        max_samples = 5000  # Much smaller for CPU/MPS
    
    texts = load_oasst_data(data_path, max_samples)
    
    if not texts:
        logger.error("❌ No training data loaded!")
        return 1
    
    try:
        # Prepare tokenizer
        tokenizer = trainer.prepare_tokenizer(texts)
        
        # Prepare model
        model = trainer.prepare_model()
        
        # Create datasets
        logger.info("📦 Creating datasets...")
        
        # Split data (80% train, 20% eval)
        split_idx = int(0.8 * len(texts))
        train_texts = texts[:split_idx]
        eval_texts = texts[split_idx:]
        
        train_dataset = OptimizedDataset(
            train_texts, tokenizer, model_config.seq_length
        )
        eval_dataset = OptimizedDataset(
            eval_texts, tokenizer, model_config.seq_length
        )
        
        # Create dataloaders
        train_dataloader = trainer.create_dataloader(train_dataset, shuffle=True)
        eval_dataloader = trainer.create_dataloader(eval_dataset, shuffle=False)
        
        # Setup training components
        trainer.prepare_training_components()
        
        # Fix scheduler with proper steps (if not using DeepSpeed)
        if trainer.deepspeed_engine is None and trainer.scheduler is not None:
            total_steps = len(train_dataloader) * training_config.max_epochs // training_config.gradient_accumulation_steps
            trainer.scheduler = setup_scheduler_with_proper_steps(
                trainer.optimizer, total_steps, training_config.warmup_ratio
            )
            logger.info(f"📊 Training setup: {total_steps:,} total steps")
        
        # Training loop
        logger.info("🏃 Starting training...")
        logger.info(f"   Precision: {precision_config.precision_type}")
        logger.info(f"   DeepSpeed: {'Yes' if trainer.deepspeed_engine else 'No'}")
        logger.info(f"   Mixed Precision: {precision_config.use_mixed_precision}")
        logger.info(f"   Device: {device}")
        
        best_loss = float('inf')
        training_start = time.time()
        
        for epoch in range(1, training_config.max_epochs + 1):
            logger.info(f"\n🎯 Epoch {epoch}/{training_config.max_epochs}")
            logger.info(f"   Memory: {get_memory_usage()}")
            
            # Training
            train_metrics = trainer.train_epoch(train_dataloader, epoch)
            
            logger.info(f"✅ Training completed:")
            logger.info(f"   Loss: {train_metrics['loss']:.4f}")
            logger.info(f"   Accuracy: {train_metrics['accuracy']:.3f}")
            logger.info(f"   Perplexity: {train_metrics['perplexity']:.2f}")
            logger.info(f"   Time: {train_metrics['time']:.1f}s")
            
            # Evaluation (every 5 epochs)
            eval_metrics = {}
            if epoch % 5 == 0 or epoch == 1:
                logger.info("🔍 Running evaluation...")
                eval_metrics = trainer.evaluate(eval_dataloader, max_batches=20)
                
                logger.info(f"📊 Evaluation results:")
                logger.info(f"   Loss: {eval_metrics['loss']:.4f}")
                logger.info(f"   Accuracy: {eval_metrics['accuracy']:.3f}")
                logger.info(f"   Perplexity: {eval_metrics['perplexity']:.2f}")
            
            # Text generation (every 10 epochs)
            if epoch % 10 == 0:
                logger.info("🎭 Generating sample text...")
                sample = trainer.generate_sample("<user> What is machine learning?")
                logger.info(f"   Sample: {sample}")
            
            # Track best model
            current_loss = eval_metrics.get('loss', train_metrics['loss'])
            is_best = current_loss < best_loss
            if is_best:
                best_loss = current_loss
                logger.info(f"🏆 New best model! Loss: {best_loss:.4f}")
            
            # Save model
            save_metrics = {**train_metrics, **eval_metrics}
            if epoch % 5 == 0 or is_best or epoch == training_config.max_epochs:
                model_id = trainer.save_model(epoch, save_metrics, is_best)
                if model_id:
                    logger.info(f"💾 Model saved: {model_id}")
            
            # Memory cleanup
            with memory_cleanup():
                pass
        
        # Training completion
        total_time = time.time() - training_start
        
        logger.info("\n" + "=" * 60)
        logger.info("🎉 Training completed successfully!")
        logger.info(f"   Best loss: {best_loss:.4f}")
        logger.info(f"   Total time: {total_time/3600:.2f} hours")
        logger.info(f"   Final memory: {get_memory_usage()}")
        
        # Show model summary
        trainer.model_manager.print_model_summary()
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("\n⏹️ Training interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"\n❌ Training failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return 1
    finally:
        with memory_cleanup():
            pass

if __name__ == "__main__":
    exit(main())