# Enhanced Training System with Full Integration
# Copyright (c) 2025 Matias Nielsen. All rights reserved.

import os
import sys
import json
import time
import math
import logging
import traceback
import gc
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import warnings

# Core PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# Import our enhanced modules
from model_manager import (
    ModelManager, ModelConfig, TrainingConfig, PrecisionConfig, 
    DataConfig, ModelMetadata, ConfigPresets, auto_select_config
)
from subword_transformer import ModernSubwordTransformer, SubwordTokenizer

# Optional advanced dependencies
try:
    import deepspeed
    from deepspeed.ops.adam import FusedAdam
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False

try:
    from torch.cuda.amp import GradScaler, autocast
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def setup_logging(log_level: str = "INFO", log_dir: str = "logs") -> logging.Logger:
    """Setup comprehensive logging system."""
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_{timestamp}.log"
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler with color support
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    root_logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)  # Always log debug to file
    root_logger.addHandler(file_handler)
    
    logger = logging.getLogger(__name__)
    logger.info(f"🔧 Logging initialized - Log file: {log_file}")
    
    return logger

class HardwareDetector:
    """Advanced hardware detection and optimization."""
    
    @staticmethod
    def detect_device() -> torch.device:
        """Detect and configure the best available device."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            
            # GPU information
            gpu_count = torch.cuda.device_count()
            current_gpu = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(current_gpu)
            props = torch.cuda.get_device_properties(current_gpu)
            
            total_memory = props.total_memory / (1024**3)
            major, minor = props.major, props.minor
            
            logger.info(f"🔥 CUDA Device Detected:")
            logger.info(f"   GPU: {gpu_name} (Compute {major}.{minor})")
            logger.info(f"   Memory: {total_memory:.1f}GB")
            logger.info(f"   GPUs Available: {gpu_count}")
            
            # Optimize CUDA settings
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Conservative memory management
            torch.cuda.set_per_process_memory_fraction(0.85)
            torch.cuda.empty_cache()
            
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("🍎 Apple Silicon (MPS) detected")
            try:
                torch.mps.empty_cache()
            except:
                pass
            
        else:
            device = torch.device("cpu")
            logger.info("💻 Using CPU")
            # Optimize CPU usage
            torch.set_num_threads(min(8, os.cpu_count() or 1))
        
        return device

    @staticmethod
    def get_memory_info() -> Dict[str, float]:
        """Get detailed memory information."""
        info = {}
        
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            total = props.total_memory / (1024**3)
            
            info.update({
                'total_gb': total,
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'free_gb': total - reserved,
                'utilization': (allocated / total) * 100
            })
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            try:
                allocated = torch.mps.current_allocated_memory() / (1024**3)
                info.update({
                    'allocated_gb': allocated,
                    'device_type': 'mps'
                })
            except:
                info.update({'device_type': 'mps', 'allocated_gb': 0})
        else:
            info.update({'device_type': 'cpu'})
        
        return info

@contextmanager
def memory_cleanup():
    """Enhanced memory cleanup context manager."""
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
            try:
                torch.mps.empty_cache()
            except:
                pass

class OptimizedDataset(Dataset):
    """Highly optimized dataset with advanced preprocessing."""
    
    def __init__(self, texts: List[str], tokenizer: SubwordTokenizer, 
                 seq_length: int, config: DataConfig):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.config = config
        
        # Token IDs
        self.pad_token_id = tokenizer.vocab.get("<|pad|>", 0)
        self.bos_token_id = tokenizer.vocab.get("<|bos|>", 2)
        self.eos_token_id = tokenizer.vocab.get("<|eos|>", 3)
        self.user_token_id = tokenizer.vocab.get("<|user|>", 4)
        self.assistant_token_id = tokenizer.vocab.get("<|assistant|>", 5)
        self.system_token_id = tokenizer.vocab.get("<|system|>", 6)
        self.end_token_id = tokenizer.vocab.get("<|end|>", 7)
        
        logger.info(f"📦 Creating optimized dataset:")
        logger.info(f"   Sequence length: {seq_length}")
        logger.info(f"   Input texts: {len(texts):,}")
        logger.info(f"   Conversation format: {config.use_conversation_format}")
        
        self.sequences = []
        self._create_sequences(texts)
        
        if not self.sequences:
            raise ValueError("No valid sequences created from input texts!")
        
        logger.info(f"✅ Dataset created: {len(self.sequences):,} sequences")
    
    def _create_sequences(self, texts: List[str]):
        """Create training sequences with advanced preprocessing."""
        
        chunk_size = 1000
        max_sequences = self.config.max_samples_train or len(texts)
        processed = 0
        
        for i in range(0, len(texts), chunk_size):
            if len(self.sequences) >= max_sequences:
                break
            
            chunk = texts[i:i+chunk_size]
            
            for text in chunk:
                if len(self.sequences) >= max_sequences:
                    break
                
                try:
                    sequence = self._process_text(text.strip())
                    if sequence:
                        self.sequences.append(sequence)
                        
                except Exception as e:
                    logger.debug(f"Failed to process text: {e}")
                    continue
            
            processed += len(chunk)
            if processed % 5000 == 0:
                logger.info(f"   Processed {processed:,}/{len(texts):,} texts, "
                          f"{len(self.sequences):,} sequences created")
                gc.collect()
    
    def _process_text(self, text: str) -> Optional[List[int]]:
        """Process individual text with conversation formatting."""
        
        if not text or len(text.strip()) < self.config.min_text_length:
            return None
        
        # Handle conversation format
        if self.config.use_conversation_format:
            # Try to parse as conversation
            if any(token in text for token in ["<|user|>", "<|assistant|>", "<|system|>"]):
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
            else:
                # Format as single assistant message
                formatted_text = f"<|user|>Please help me with this:<|end|>\n<|assistant|>{text}<|end|>"
                tokens = self.tokenizer.encode(formatted_text, add_special_tokens=False)
        else:
            # Regular text encoding
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        if len(tokens) < 3:  # Too short
            return None
        
        # Create training sequence
        # Add BOS, truncate content, add EOS
        sequence = [self.bos_token_id] + tokens[:self.seq_length-2] + [self.eos_token_id]
        
        # Pad to sequence length + 1 (for input/target pairs)
        if len(sequence) <= self.seq_length:
            sequence.extend([self.pad_token_id] * (self.seq_length + 1 - len(sequence)))
        else:
            sequence = sequence[:self.seq_length + 1]
        
        return sequence
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        # Input is sequence[:-1], target is sequence[1:]
        input_ids = torch.tensor(sequence[:-1], dtype=torch.long)
        target_ids = torch.tensor(sequence[1:], dtype=torch.long)
        return input_ids, target_ids

class TrainingMetrics:
    """Comprehensive training metrics tracking."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.total_loss = 0.0
        self.total_correct = 0
        self.total_tokens = 0
        self.num_batches = 0
        self.start_time = time.time()
        self.loss_history = []
    
    def update(self, loss: float, predictions: torch.Tensor, targets: torch.Tensor, 
               pad_token_id: int = 0) -> bool:
        """Update metrics with batch results."""
        
        # Validate loss
        if torch.isnan(torch.tensor(loss)) or torch.isinf(torch.tensor(loss)):
            logger.warning("Invalid loss detected, skipping batch")
            return False
        
        # Calculate accuracy (excluding padding tokens)
        mask = (targets != pad_token_id)
        correct = ((predictions == targets) & mask).sum().item()
        valid_tokens = mask.sum().item()
        
        if valid_tokens == 0:
            return False
        
        self.total_loss += loss
        self.total_correct += correct
        self.total_tokens += valid_tokens
        self.num_batches += 1
        self.loss_history.append(loss)
        
        return True
    
    def get_metrics(self) -> Dict[str, float]:
        """Get comprehensive metrics."""
        if self.num_batches == 0:
            return {
                "loss": float('inf'),
                "accuracy": 0.0,
                "perplexity": float('inf'),
                "time": 0.0,
                "tokens_per_second": 0.0
            }
        
        avg_loss = self.total_loss / self.num_batches
        accuracy = self.total_correct / max(self.total_tokens, 1)
        perplexity = math.exp(min(avg_loss, 10))  # Cap to prevent overflow
        elapsed_time = time.time() - self.start_time
        tokens_per_second = self.total_tokens / max(elapsed_time, 0.001)
        
        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "perplexity": perplexity,
            "time": elapsed_time,
            "tokens_per_second": tokens_per_second
        }

class AdvancedTrainer:
    """Production-ready trainer with all integrations."""
    
    def __init__(self, model_config: ModelConfig, training_config: TrainingConfig,
                 precision_config: PrecisionConfig, data_config: DataConfig, 
                 device: torch.device, experiment_name: str = None):
        
        self.model_config = model_config
        self.training_config = training_config
        self.precision_config = precision_config
        self.data_config = data_config
        self.device = device
        self.experiment_name = experiment_name or f"experiment_{int(time.time())}"
        
        # Initialize components
        self.model_manager = ModelManager("models")
        self.tokenizer = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.deepspeed_engine = None
        
        # Tracking
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        self.training_start_time = None
        
        # Wandb integration
        self.use_wandb = WANDB_AVAILABLE and os.getenv('WANDB_PROJECT')
        if self.use_wandb:
            self._init_wandb()
        
        logger.info(f"🚀 AdvancedTrainer initialized:")
        logger.info(f"   Experiment: {self.experiment_name}")
        logger.info(f"   Device: {device}")
        logger.info(f"   Precision: {precision_config.precision_type}")
        logger.info(f"   Wandb: {self.use_wandb}")
    
    def _init_wandb(self):
        """Initialize Weights & Biases tracking."""
        try:
            wandb.init(
                project=os.getenv('WANDB_PROJECT', 'transformer-training'),
                name=self.experiment_name,
                config={
                    **asdict(self.model_config),
                    **asdict(self.training_config),
                    **asdict(self.precision_config),
                    **asdict(self.data_config)
                }
            )
            logger.info("✅ Wandb initialized")
        except Exception as e:
            logger.warning(f"Wandb initialization failed: {e}")
            self.use_wandb = False
    
    def prepare_tokenizer(self, texts: List[str]) -> SubwordTokenizer:
        """Prepare and train the tokenizer."""
        logger.info("🔤 Training SubwordTokenizer...")
        
        # Create tokenizer
        self.tokenizer = SubwordTokenizer()
        
        # Prepare training text
        max_chars = 50_000_000  # 50M chars max for training
        training_text = ""
        char_count = 0
        
        # Sample texts for tokenizer training
        sample_texts = texts[:self.data_config.tokenizer_train_size] if texts else []
        
        for text in sample_texts:
            if char_count >= max_chars:
                break
            training_text += text + "\n"
            char_count += len(text)
        
        if not training_text:
            raise ValueError("No training text available for tokenizer!")
        
        # Train with progress callback
        def progress_callback(progress, current, total):
            if int(progress) % 10 == 0:  # Log every 10%
                logger.info(f"   Tokenizer training: {progress:.1f}% ({current:,}/{total:,})")
        
        self.tokenizer.train_from_text(
            training_text,
            vocab_size=self.model_config.vocab_size,
            min_freq=self.data_config.min_frequency,
            progress_callback=progress_callback
        )
        
        # Update model config with actual vocab size
        actual_vocab_size = self.tokenizer.vocab_size()
        self.model_config.vocab_size = actual_vocab_size
        
        logger.info(f"✅ Tokenizer trained: {actual_vocab_size:,} tokens")
        return self.tokenizer
    
    def prepare_model(self) -> ModernSubwordTransformer:
        """Prepare the model."""
        logger.info("🧠 Creating ModernSubwordTransformer...")
        
        # Create model
        self.model = ModernSubwordTransformer(self.model_config)
        self.model = self.model.to(self.device)
        
        # Apply torch.compile if available and requested
        if self.precision_config.use_compile and hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(
                    self.model, 
                    mode=self.precision_config.compile_mode
                )
                logger.info("✅ Model compiled with torch.compile")
            except Exception as e:
                logger.warning(f"Failed to compile model: {e}")
        
        # Log model info
        info = self.model.get_model_info()
        logger.info(f"✅ Model created:")
        logger.info(f"   Parameters: {info['parameters']['total']:,}")
        logger.info(f"   Memory: {info['memory']['model_mb']:.1f}MB")
        
        if self.use_wandb:
            wandb.log({
                "model/total_parameters": info['parameters']['total'],
                "model/trainable_parameters": info['parameters']['trainable'],
                "model/memory_mb": info['memory']['model_mb']
            })
        
        return self.model
    
    def prepare_training_components(self):
        """Setup optimizer, scheduler, and other training components."""
        logger.info("🔧 Setting up training components...")
        
        # Try DeepSpeed first if available
        if DEEPSPEED_AVAILABLE and self.device.type == 'cuda':
            if self._init_deepspeed():
                return
        
        # Fallback to regular training
        self._init_regular_training()
    
    def _init_deepspeed(self) -> bool:
        """Initialize DeepSpeed if available."""
        try:
            # Create DeepSpeed config
            ds_config = {
                "train_batch_size": self.training_config.batch_size * self.training_config.gradient_accumulation_steps,
                "train_micro_batch_size_per_gpu": self.training_config.batch_size,
                "gradient_accumulation_steps": self.training_config.gradient_accumulation_steps,
                "optimizer": {
                    "type": "AdamW",
                    "params": {
                        "lr": self.training_config.learning_rate,
                        "betas": [self.training_config.beta1, self.training_config.beta2],
                        "eps": self.training_config.eps,
                        "weight_decay": self.training_config.weight_decay
                    }
                },
                "scheduler": {
                    "type": "WarmupDecayLR",
                    "params": {
                        "warmup_min_lr": 0,
                        "warmup_max_lr": self.training_config.learning_rate,
                        "warmup_num_steps": "auto",
                        "total_num_steps": "auto"
                    }
                },
                "gradient_clipping": self.training_config.max_grad_norm,
                "wall_clock_breakdown": False,
            }
            
            # Precision settings
            if self.precision_config.precision_type == "bf16":
                ds_config["bf16"] = {"enabled": True}
            elif self.precision_config.precision_type == "fp16":
                ds_config["fp16"] = {
                    "enabled": True,
                    "loss_scale": 0,  # Dynamic scaling
                    "initial_scale_power": int(math.log2(self.precision_config.initial_scale)),
                    "loss_scale_window": self.precision_config.growth_interval,
                }
            
            # ZeRO optimization
            ds_config["zero_optimization"] = {
                "stage": 2,
                "offload_optimizer": {"device": "cpu", "pin_memory": True},
                "overlap_comm": True,
                "contiguous_gradients": True,
            }
            
            # Write config and initialize
            config_path = "deepspeed_config.json"
            with open(config_path, 'w') as f:
                json.dump(ds_config, f, indent=2)
            
            self.deepspeed_engine, self.optimizer, _, self.scheduler = deepspeed.initialize(
                model=self.model,
                config=config_path,
                model_parameters=self.model.parameters()
            )
            
            logger.info("✅ DeepSpeed initialized")
            return True
            
        except Exception as e:
            logger.warning(f"DeepSpeed initialization failed: {e}")
            return False
    
    def _init_regular_training(self):
        """Initialize regular PyTorch training components."""
        
        # Optimizer
        if self.training_config.optimizer_type.lower() == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.training_config.learning_rate,
                betas=(self.training_config.beta1, self.training_config.beta2),
                eps=self.training_config.eps,
                weight_decay=self.training_config.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.training_config.optimizer_type}")
        
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
        
        logger.info(f"✅ Regular training setup complete")
        logger.info(f"   Optimizer: {self.training_config.optimizer_type}")
        logger.info(f"   Mixed precision: {self.precision_config.precision_type}")
    
    def create_scheduler(self, total_steps: int):
        """Create learning rate scheduler after knowing total steps."""
        if self.deepspeed_engine:
            return  # DeepSpeed handles scheduling
        
        if self.training_config.scheduler_type == "cosine_with_warmup":
            from torch.optim.lr_scheduler import OneCycleLR
            
            warmup_steps = self.training_config.warmup_steps
            if warmup_steps is None:
                warmup_steps = int(total_steps * self.training_config.warmup_ratio)
            
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.training_config.learning_rate,
                total_steps=total_steps,
                pct_start=warmup_steps / total_steps,
                anneal_strategy='cos',
                div_factor=25.0,
                final_div_factor=10000.0
            )
            
            logger.info(f"✅ OneCycleLR scheduler created:")
            logger.info(f"   Total steps: {total_steps:,}")
            logger.info(f"   Warmup steps: {warmup_steps:,}")
    
    def create_dataloader(self, dataset: Dataset, shuffle: bool = True) -> DataLoader:
        """Create optimized DataLoader."""
        return DataLoader(
            dataset,
            batch_size=self.training_config.batch_size,
            shuffle=shuffle,
            num_workers=self.training_config.num_workers if self.training_config.use_dataloader_workers else 0,
            pin_memory=self.training_config.pin_memory and self.device.type == 'cuda',
            drop_last=True,
            prefetch_factor=self.training_config.prefetch_factor if self.training_config.num_workers > 0 else 2
        )
    
    def train_epoch(self, train_dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.epoch += 1
        metrics = TrainingMetrics()
        
        # Set training mode
        if self.deepspeed_engine:
            self.deepspeed_engine.train()
            model = self.deepspeed_engine
        else:
            self.model.train()
            model = self.model
            if self.optimizer:
                self.optimizer.zero_grad()
        
        criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.vocab.get("<|pad|>", 0))
        accumulation_count = 0
        
        logger.info(f"🏃 Epoch {self.epoch}: Training on {len(train_dataloader)} batches")
        
        for batch_idx, (input_ids, target_ids) in enumerate(train_dataloader):
            try:
                self.global_step += 1
                
                # Move to device
                input_ids = input_ids.to(self.device)
                target_ids = target_ids.to(self.device)
                
                # Forward pass
                if self.deepspeed_engine:
                    # DeepSpeed handles mixed precision internally
                    outputs = model(input_ids, return_dict=True)
                    logits = outputs['logits']
                    loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
                    
                    # DeepSpeed backward and step
                    model.backward(loss)
                    model.step()
                    
                else:
                    # Regular training with optional mixed precision
                    if self.precision_config.use_mixed_precision and self.scaler:
                        with autocast(dtype=torch.bfloat16 if self.precision_config.precision_type == "bf16" else torch.float16):
                            outputs = model(input_ids, return_dict=True)
                            logits = outputs['logits']
                            loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
                        
                        # Scale loss for gradient accumulation
                        scaled_loss = loss / self.training_config.gradient_accumulation_steps
                        self.scaler.scale(scaled_loss).backward()
                        
                    else:
                        outputs = model(input_ids, return_dict=True)
                        logits = outputs['logits']
                        loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
                        scaled_loss = loss / self.training_config.gradient_accumulation_steps
                        scaled_loss.backward()
                    
                    accumulation_count += 1
                    
                    # Optimizer step
                    if accumulation_count >= self.training_config.gradient_accumulation_steps:
                        if self.scaler:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), self.training_config.max_grad_norm)
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), self.training_config.max_grad_norm)
                            self.optimizer.step()
                        
                        if self.scheduler:
                            self.scheduler.step()
                        
                        self.optimizer.zero_grad()
                        accumulation_count = 0
                
                # Update metrics
                with torch.no_grad():
                    predictions = torch.argmax(logits, dim=-1)
                    metrics.update(loss.item(), predictions, target_ids, self.tokenizer.vocab.get("<|pad|>", 0))
                
                # Logging and cleanup
                if batch_idx % 50 == 0:
                    current_metrics = metrics.get_metrics()
                    lr = (self.deepspeed_engine.get_lr()[0] if self.deepspeed_engine 
                         else self.optimizer.param_groups[0]['lr'] if self.optimizer else 0.0)
                    memory_info = HardwareDetector.get_memory_info()
                    
                    logger.info(
                        f"Epoch {self.epoch} | Step {self.global_step} | Batch {batch_idx:4d}/{len(train_dataloader)} | "
                        f"Loss: {current_metrics['loss']:.4f} | "
                        f"Acc: {current_metrics['accuracy']:.3f} | "
                        f"PPL: {current_metrics['perplexity']:.2f} | "
                        f"LR: {lr:.2e} | "
                        f"Mem: {memory_info.get('allocated_gb', 0):.1f}GB"
                    )
                    
                    # Wandb logging
                    if self.use_wandb:
                        wandb.log({
                            "train/loss": current_metrics['loss'],
                            "train/accuracy": current_metrics['accuracy'],
                            "train/perplexity": current_metrics['perplexity'],
                            "train/learning_rate": lr,
                            "train/tokens_per_second": current_metrics['tokens_per_second'],
                            "system/memory_allocated_gb": memory_info.get('allocated_gb', 0),
                            "system/memory_utilization": memory_info.get('utilization', 0),
                            "global_step": self.global_step
                        })
                
                # Cleanup
                del input_ids, target_ids, logits, loss, predictions
                if batch_idx % 100 == 0:
                    with memory_cleanup():
                        pass
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning(f"⚠️  OOM at step {self.global_step}, skipping batch...")
                    
                    # Emergency cleanup
                    if self.optimizer:
                        self.optimizer.zero_grad()
                    
                    with memory_cleanup():
                        pass
                    continue
                else:
                    raise e
            
            except Exception as e:
                logger.warning(f"Error in batch {batch_idx}: {e}")
                continue
        
        # Final gradient step for incomplete accumulation
        if not self.deepspeed_engine and accumulation_count > 0:
            if self.scaler:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.training_config.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.training_config.max_grad_norm)
                self.optimizer.step()
            self.optimizer.zero_grad()
        
        return metrics.get_metrics()
    
    def evaluate(self, eval_dataloader: DataLoader, max_batches: int = 20) -> Dict[str, float]:
        """Evaluate the model."""
        metrics = TrainingMetrics()
        
        # Set evaluation mode
        if self.deepspeed_engine:
            self.deepspeed_engine.eval()
            model = self.deepspeed_engine
        else:
            self.model.eval()
            model = self.model
        
        criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.vocab.get("<|pad|>", 0))
        
        try:
            with torch.no_grad():
                for batch_idx, (input_ids, target_ids) in enumerate(eval_dataloader):
                    if batch_idx >= max_batches:
                        break
                    
                    try:
                        input_ids = input_ids.to(self.device)
                        target_ids = target_ids.to(self.device)
                        
                        # Forward pass
                        if self.precision_config.use_mixed_precision and not self.deepspeed_engine:
                            with autocast(dtype=torch.bfloat16 if self.precision_config.precision_type == "bf16" else torch.float16):
                                outputs = model(input_ids, return_dict=True)
                                logits = outputs['logits']
                                loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
                        else:
                            outputs = model(input_ids, return_dict=True)
                            logits = outputs['logits']
                            loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
                        
                        # Update metrics
                        predictions = torch.argmax(logits, dim=-1)
                        metrics.update(loss.item(), predictions, target_ids, self.tokenizer.vocab.get("<|pad|>", 0))
                        
                        del input_ids, target_ids, logits, loss, predictions
                        
                    except Exception as e:
                        logger.debug(f"Error in eval batch {batch_idx}: {e}")
                        continue
        
        finally:
            # Return to training mode
            if self.deepspeed_engine:
                self.deepspeed_engine.train()
            else:
                self.model.train()
        
        return metrics.get_metrics()
    
    def generate_sample(self, prompt: str = "Hello, how are you?", max_length: int = 100) -> str:
        """Generate a sample text for evaluation."""
        if self.deepspeed_engine:
            self.deepspeed_engine.eval()
            model = self.deepspeed_engine
        else:
            self.model.eval()
            model = self.model
        
        try:
            with torch.no_grad():
                # Format prompt for conversation
                if self.data_config.use_conversation_format:
                    formatted_prompt = f"<|user|>{prompt}<|end|>\n<|assistant|>"
                else:
                    formatted_prompt = prompt
                
                # Encode prompt
                input_ids = torch.tensor(
                    self.tokenizer.encode(formatted_prompt, add_special_tokens=False),
                    dtype=torch.long
                ).unsqueeze(0).to(self.device)
                
                # Generate
                if hasattr(model, 'generate'):
                    generated = model.generate(
                        input_ids,
                        max_new_tokens=max_length,
                        temperature=0.8,
                        top_k=50,
                        top_p=0.9,
                        do_sample=True,
                        pad_token_id=self.tokenizer.vocab.get("<|pad|>", 0),
                        eos_token_id=self.tokenizer.vocab.get("<|eos|>", 3)
                    )
                else:
                    # Fallback generation
                    generated = input_ids.clone()
                    for _ in range(max_length):
                        if generated.size(1) >= self.model_config.seq_length:
                            break
                        
                        outputs = model(generated, return_dict=True)
                        logits = outputs['logits']
                        next_token = torch.argmax(logits[0, -1, :], dim=-1)
                        generated = torch.cat([generated, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
                        
                        if next_token.item() == self.tokenizer.vocab.get("<|eos|>", 3):
                            break
                
                # Decode only the new part
                response_ids = generated[0][input_ids.size(1):].tolist()
                response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                
                return response.strip()
        
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return "Generation failed"
        
        finally:
            if self.deepspeed_engine:
                self.deepspeed_engine.train()
            else:
                self.model.train()
    
    def save_model(self, metrics: Dict[str, float], is_best: bool = False) -> Optional[str]:
        """Save model with comprehensive metadata."""
        try:
            # Create metadata
            metadata = ModelMetadata(
                model_name=f"ModernTransformer_{self.experiment_name}",
                version=f"epoch_{self.epoch}",
                created_at=datetime.now().isoformat(),
                last_modified=datetime.now().isoformat(),
                
                # Configurations
                model_config=asdict(self.model_config),
                training_config=asdict(self.training_config),
                precision_config=asdict(self.precision_config),
                data_config=asdict(self.data_config),
                
                # Performance metrics
                performance_metrics=metrics,
                
                # Model statistics
                total_parameters=self.model.count_parameters() if hasattr(self.model, 'count_parameters') else 0,
                trainable_parameters=self.model.count_trainable_parameters() if hasattr(self.model, 'count_trainable_parameters') else 0,
                model_size_mb=self.model.estimate_memory_mb() if hasattr(self.model, 'estimate_memory_mb') else 0,
                
                # Training info
                epochs_trained=self.epoch,
                total_training_time=time.time() - self.training_start_time if self.training_start_time else 0,
                best_loss=metrics.get('loss', float('inf')),
                best_perplexity=metrics.get('perplexity', float('inf')),
                
                # Environment
                hardware_used=f"{self.device.type.upper()} ({torch.cuda.get_device_name() if torch.cuda.is_available() else 'N/A'})",
                
                # Additional info
                notes=f"Training epoch {self.epoch} with {self.precision_config.precision_type} precision" + 
                      (" using DeepSpeed" if self.deepspeed_engine else ""),
                tags=[
                    f"epoch_{self.epoch}",
                    f"{self.precision_config.precision_type}_precision",
                    "modern_transformer",
                    self.experiment_name
                ] + (["best"] if is_best else []) + (["deepspeed"] if self.deepspeed_engine else [])
            )
            
            # Save model
            model_id = self.model_manager.save_model(
                self.model,
                self.tokenizer,
                metadata,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                deepspeed_engine=self.deepspeed_engine
            )
            
            if model_id and self.use_wandb:
                wandb.log({
                    "model/saved": True,
                    "model/id": model_id,
                    "model/is_best": is_best
                })
            
            return model_id
            
        except Exception as e:
            logger.error(f"❌ Failed to save model: {e}")
            return None

def load_data(data_path: str, config: DataConfig) -> List[str]:
    """Load and preprocess training data."""
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    logger.info(f"📂 Loading data from: {data_path}")
    
    texts = []
    processed_count = 0
    max_samples = config.max_samples_train
    
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if max_samples and processed_count >= max_samples:
                    break
                
                try:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Try to parse as JSON first
                    try:
                        record = json.loads(line)
                        
                        # Handle different data formats
                        if isinstance(record, dict):
                            # OASST format
                            if "text" in record:
                                text = record["text"].strip()
                                if record.get("lang") != "en" or record.get("deleted", False):
                                    continue
                                
                                role = record.get("role", "").lower()
                                if role == "prompter":
                                    text = f"<|user|>{text}<|end|>"
                                elif role == "assistant":
                                    text = f"<|assistant|>{text}<|end|>"
                            
                            # Alpaca format
                            elif "instruction" in record:
                                instruction = record.get("instruction", "").strip()
                                input_text = record.get("input", "").strip()
                                output = record.get("output", "").strip()
                                
                                if input_text:
                                    text = f"<|user|>{instruction}\n\n{input_text}<|end|>\n<|assistant|>{output}<|end|>"
                                else:
                                    text = f"<|user|>{instruction}<|end|>\n<|assistant|>{output}<|end|>"
                            
                            # Chat format
                            elif "messages" in record:
                                messages = record["messages"]
                                text_parts = []
                                for msg in messages:
                                    role = msg.get("role", "").lower()
                                    content = msg.get("content", "").strip()
                                    if role == "user":
                                        text_parts.append(f"<|user|>{content}<|end|>")
                                    elif role == "assistant":
                                        text_parts.append(f"<|assistant|>{content}<|end|>")
                                    elif role == "system":
                                        text_parts.append(f"<|system|>{content}<|end|>")
                                text = "\n".join(text_parts)
                            
                            else:
                                # Generic dict - try to find text content
                                text_candidates = ["text", "content", "response", "output"]
                                text = None
                                for key in text_candidates:
                                    if key in record and record[key]:
                                        text = str(record[key]).strip()
                                        break
                                
                                if not text:
                                    continue
                        else:
                            # Direct string
                            text = str(record).strip()
                    
                    except json.JSONDecodeError:
                        # Not JSON, treat as plain text
                        text = line.strip()
                    
                    # Validate text
                    if not text or len(text) < config.min_text_length:
                        continue
                    
                    if config.max_text_length and len(text) > config.max_text_length:
                        text = text[:config.max_text_length]
                    
                    texts.append(text)
                    processed_count += 1
                    
                    # Progress logging
                    if line_num % 10000 == 0:
                        logger.info(f"   Processed {line_num:,} lines, {processed_count:,} valid texts")
                    
                except Exception as e:
                    logger.debug(f"Skipping line {line_num}: {e}")
                    continue
    
    except Exception as e:
        logger.error(f"❌ Error loading data: {e}")
        raise
    
    # Remove duplicates if requested
    if config.remove_duplicates:
        original_count = len(texts)
        texts = list(dict.fromkeys(texts))  # Preserve order while removing duplicates
        logger.info(f"   Removed {original_count - len(texts):,} duplicates")
    
    logger.info(f"✅ Loaded {len(texts):,} texts from {data_path}")
    return texts

def get_memory_stats() -> str:
    """Get formatted memory statistics."""
    info = HardwareDetector.get_memory_info()
    
    if 'total_gb' in info:
        return (f"GPU: {info['allocated_gb']:.1f}/{info['total_gb']:.1f}GB "
                f"({info['utilization']:.1f}% util)")
    elif info.get('device_type') == 'mps':
        return f"MPS: {info.get('allocated_gb', 0):.1f}GB used"
    else:
        return "CPU mode"

def main():
    """Enhanced main training function with full integration."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Enhanced Transformer Training")
    parser.add_argument("--data", type=str, required=True, help="Path to training data")
    parser.add_argument("--config", type=str, choices=["auto", "tiny", "debug", "7b"], 
                       default="auto", help="Configuration preset")
    parser.add_argument("--experiment", type=str, default=None, help="Experiment name")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--no_wandb", action="store_true", help="Disable Wandb logging")
    
    args = parser.parse_args()
    
    # Setup logging
    global logger
    logger = setup_logging(args.log_level)
    
    logger.info("🚀 Enhanced ModernSubwordTransformer Training")
    logger.info("=" * 80)
    
    # Environment check
    logger.info("🔍 Environment Check:")
    logger.info(f"   Python: {sys.version}")
    logger.info(f"   PyTorch: {torch.__version__}")
    logger.info(f"   CUDA Available: {torch.cuda.is_available()}")
    logger.info(f"   AMP Available: {AMP_AVAILABLE}")
    logger.info(f"   DeepSpeed Available: {DEEPSPEED_AVAILABLE}")
    logger.info(f"   Wandb Available: {WANDB_AVAILABLE and not args.no_wandb}")
    
    if args.no_wandb:
        os.environ.pop('WANDB_PROJECT', None)
    
    try:
        # Detect hardware and get configuration
        device = HardwareDetector.detect_device()
        
        # Select configuration preset
        if args.config == "auto":
            model_config, training_config, precision_config, data_config = auto_select_config()
        elif args.config == "tiny":
            model_config, training_config, precision_config, data_config = ConfigPresets.tiny_debug()
        elif args.config == "debug":
            model_config, training_config, precision_config, data_config = ConfigPresets.tiny_debug()
            data_config.max_samples_train = 1000
            data_config.max_samples_eval = 100
        elif args.config == "7b":
            model_config, training_config, precision_config, data_config = ConfigPresets.research_7b()
        else:
            raise ValueError(f"Unknown config preset: {args.config}")
        
        logger.info(f"📊 Configuration: {args.config}")
        logger.info(f"   Model: {model_config.hidden_size}d × {model_config.num_layers}L")
        logger.info(f"   Vocabulary: {model_config.vocab_size:,}")
        logger.info(f"   Sequence Length: {model_config.seq_length}")
        logger.info(f"   Precision: {precision_config.precision_type}")
        logger.info(f"   Batch Size: {training_config.batch_size}")
        logger.info(f"   Gradient Accumulation: {training_config.gradient_accumulation_steps}")
        
        # Update data config with file path
        data_config.train_data_path = args.data
        
        # Initialize trainer
        experiment_name = args.experiment or f"{args.config}_{int(time.time())}"
        trainer = AdvancedTrainer(
            model_config, training_config, precision_config, data_config, 
            device, experiment_name
        )
        
        # Load and prepare data
        logger.info("📦 Loading and preparing data...")
        texts = load_data(args.data, data_config)
        
        if not texts:
            logger.error("❌ No training data loaded!")
            return 1
        
        # Split data
        split_idx = int(data_config.train_split_ratio * len(texts))
        train_texts = texts[:split_idx]
        eval_texts = texts[split_idx:] if split_idx < len(texts) else train_texts[-100:]
        
        logger.info(f"📊 Data split:")
        logger.info(f"   Training: {len(train_texts):,} texts")
        logger.info(f"   Evaluation: {len(eval_texts):,} texts")
        
        # Prepare tokenizer
        trainer.prepare_tokenizer(train_texts)
        
        # Prepare model
        trainer.prepare_model()
        
        # Create datasets
        logger.info("📦 Creating datasets...")
        train_dataset = OptimizedDataset(train_texts, trainer.tokenizer, model_config.seq_length, data_config)
        eval_dataset = OptimizedDataset(eval_texts, trainer.tokenizer, model_config.seq_length, data_config)
        
        # Create dataloaders
        train_dataloader = trainer.create_dataloader(train_dataset, shuffle=True)
        eval_dataloader = trainer.create_dataloader(eval_dataset, shuffle=False)
        
        # Setup training components
        trainer.prepare_training_components()
        
        # Create scheduler with proper total steps
        total_steps = (len(train_dataloader) * training_config.max_epochs) // training_config.gradient_accumulation_steps
        trainer.create_scheduler(total_steps)
        
        logger.info(f"📊 Training Setup:")
        logger.info(f"   Total steps: {total_steps:,}")
        logger.info(f"   Effective batch size: {training_config.batch_size * training_config.gradient_accumulation_steps}")
        logger.info(f"   Memory: {get_memory_stats()}")
        
        # Training loop
        logger.info("\n🏃 Starting Training...")
        trainer.training_start_time = time.time()
        
        for epoch in range(1, training_config.max_epochs + 1):
            logger.info(f"\n🎯 Epoch {epoch}/{training_config.max_epochs}")
            logger.info(f"   Memory: {get_memory_stats()}")
            
            # Training
            train_metrics = trainer.train_epoch(train_dataloader)
            
            logger.info(f"✅ Training Results:")
            logger.info(f"   Loss: {train_metrics['loss']:.4f}")
            logger.info(f"   Accuracy: {train_metrics['accuracy']:.3f}")
            logger.info(f"   Perplexity: {train_metrics['perplexity']:.2f}")
            logger.info(f"   Tokens/sec: {train_metrics['tokens_per_second']:.0f}")
            logger.info(f"   Time: {train_metrics['time']:.1f}s")
            
            # Evaluation
            eval_metrics = {}
            if epoch % 3 == 0 or epoch == 1:
                logger.info("🔍 Running evaluation...")
                eval_metrics = trainer.evaluate(eval_dataloader, max_batches=30)
                
                logger.info(f"📊 Evaluation Results:")
                logger.info(f"   Loss: {eval_metrics['loss']:.4f}")
                logger.info(f"   Accuracy: {eval_metrics['accuracy']:.3f}")
                logger.info(f"   Perplexity: {eval_metrics['perplexity']:.2f}")
            
            # Text generation sample
            if epoch % 5 == 0 or epoch == 1:
                logger.info("🎭 Generating sample...")
                sample_prompts = [
                    "What is artificial intelligence?",
                    "Explain quantum computing in simple terms.",
                    "How do neural networks work?"
                ]
                
                for i, prompt in enumerate(sample_prompts[:1]):  # Just one sample to save time
                    sample = trainer.generate_sample(prompt, max_length=80)
                    logger.info(f"   Sample {i+1}: {sample[:200]}...")
            
            # Model saving
            current_loss = eval_metrics.get('loss', train_metrics['loss'])
            is_best = current_loss < trainer.best_loss
            
            if is_best:
                trainer.best_loss = current_loss
                logger.info(f"🏆 New best model! Loss: {trainer.best_loss:.4f}")
            
            # Save model
            save_metrics = {**train_metrics, **eval_metrics}
            if epoch % 5 == 0 or is_best or epoch == training_config.max_epochs:
                model_id = trainer.save_model(save_metrics, is_best)
                if model_id:
                    logger.info(f"💾 Model saved: {model_id}")
            
            # Wandb logging
            if trainer.use_wandb:
                log_dict = {
                    "epoch": epoch,
                    "train/loss": train_metrics['loss'],
                    "train/accuracy": train_metrics['accuracy'],
                    "train/perplexity": train_metrics['perplexity'],
                    "train/tokens_per_second": train_metrics['tokens_per_second'],
                    "system/memory": get_memory_stats(),
                }
                
                if eval_metrics:
                    log_dict.update({
                        "eval/loss": eval_metrics['loss'],
                        "eval/accuracy": eval_metrics['accuracy'],
                        "eval/perplexity": eval_metrics['perplexity'],
                    })
                
                wandb.log(log_dict)
            
            # Memory cleanup
            with memory_cleanup():
                pass
        
        # Training completion
        total_time = time.time() - trainer.training_start_time
        
        logger.info("\n" + "=" * 80)
        logger.info("🎉 Training Completed Successfully!")
        logger.info(f"   Best Loss: {trainer.best_loss:.4f}")
        logger.info(f"   Total Time: {total_time/3600:.2f} hours")
        logger.info(f"   Final Memory: {get_memory_stats()}")
        
        # Show model summary
        trainer.model_manager.print_model_summary()
        
        # Final generation samples
        logger.info("\n🎭 Final Generation Samples:")
        final_prompts = [
            "Hello, how are you today?",
            "What is the meaning of life?",
            "Explain machine learning to a beginner."
        ]
        
        for prompt in final_prompts:
            sample = trainer.generate_sample(prompt, max_length=100)
            logger.info(f"Prompt: {prompt}")
            logger.info(f"Response: {sample}")
            logger.info("")
        
        if trainer.use_wandb:
            wandb.finish()
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("\n⏹️ Training interrupted by user")
        if 'trainer' in locals() and trainer.use_wandb:
            wandb.finish()
        return 1
        
    except Exception as e:
        logger.error(f"\n❌ Training failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        if 'trainer' in locals() and trainer.use_wandb:
            wandb.finish()
        return 1
        
    finally:
        with memory_cleanup():
            pass

if __name__ == "__main__":
    exit(main())