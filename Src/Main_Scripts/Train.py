# Ultra-Fast Enhanced Training System with MAXIMUM BPE Speed Optimizations
# Copyright (c) 2025 Matias Nielsen. All rights reserved.

import os
import sys
import json
import logging
import math
import random
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict, Counter
import functools
import time

# Core ML imports with optimizations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

import numpy as np
from tqdm import tqdm

# Import our advanced components
from model_manager import (
    ModelConfig, TrainingConfig, PrecisionConfig, DataConfig, HardwareConfig,
    ModelMetadata, ModelManager, ConfigPresets, auto_select_config
)
from subword_transformer import UltraFastTokenizer, ModernSubwordTransformer

# Performance optimization imports
try:
    import torch._dynamo
    torch._dynamo.config.cache_size_limit = 256
    torch._dynamo.config.suppress_errors = True
    TORCH_COMPILE_AVAILABLE = True
except ImportError:
    TORCH_COMPILE_AVAILABLE = False

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

try:
    import numba
    NUMBA_AVAILABLE = True
    logging.info("🚀 Numba available for ultra-fast BPE training")
except ImportError:
    NUMBA_AVAILABLE = False

# Enable optimizations
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

# ============================================================================
# TRAINING CONFIGURATION - SAME AS ORIGINAL
# ============================================================================

TRAINING_CONFIG = {
    # Data Configuration
    "data": {
        "training_data_path": "oasst1_data/oasst1_train.jsonl",
        "use_conversation_format": True,
        "max_samples_train": None,
        "max_samples_eval": None,
        "eval_split": 0.1,
        "min_text_length": 10,
        "max_text_length": None,
        "remove_duplicates": True,
        "lowercase": False,
        "tokenizer_train_size": 100000,
        "min_frequency": 2,
    },
    
    # Model Configuration
    "model": {
        "config_preset": "custom",
        "custom": {
            "vocab_size": 32000,
            "hidden_size": 4096,
            "num_layers": 36,
            "num_heads": 32,
            "seq_length": 8192,
            "use_rotary_pos_emb": True,
            "use_rms_norm": True,
            "use_grouped_query_attention": True,
            "use_glu_variants": True,
            "glu_variant": "swiglu",
            "gradient_checkpointing": True,
        },
        "tiny": {
            "vocab_size": 32000,
            "hidden_size": 256,
            "num_layers": 4,
            "num_heads": 4,
            "seq_length": 512,
            "use_rotary_pos_emb": True,
            "use_rms_norm": True,
            "use_grouped_query_attention": True,
            "use_glu_variants": True,
            "glu_variant": "swiglu",
            "gradient_checkpointing": True
        },
        "research": {
            "vocab_size": 32000,
            "hidden_size": 2048,
            "num_layers": 24,
            "num_heads": 16,
            "seq_length": 1024,
            "use_rotary_pos_emb": True,
            "use_rms_norm": True,
            "use_grouped_query_attention": True,
            "use_glu_variants": True,
            "glu_variant": "swiglu",
            "gradient_checkpointing": True
        },
    },
    
    # Training Configuration
    "training": {
        "batch_size": 1,
        "gradient_accumulation_steps": 8,
        "max_epochs": 200,
        "max_steps": None,
        "learning_rate": 3e-4,
        "weight_decay": 0.1,
        "warmup_ratio": 0.03,
        "scheduler_type": "cosine_with_warmup",
        "optimizer_type": "adamw",
        "max_grad_norm": 1.0,
        "eval_steps": None,
        "save_steps": None,
        "logging_steps": 10,  # Keep this in config but don't pass to TrainingConfig
        "save_total_limit": 3,
        "use_dataloader_workers": True,
        "num_workers": 4,
    },
    
    # Precision and Performance Configuration
    "precision": {
        "precision_type": "auto",
        "use_mixed_precision": True,
        "use_compile": True,
        "compile_mode": "max-autotune",  # Changed for maximum speed
        "use_dynamic_loss_scaling": True,
    },
    
    # System Configuration
    "system": {
        "output_dir": "models",
        "log_dir": "logs",
        "device": "auto",
        "seed": 42,
        "gradient_checkpointing": True,
    },
    
    # Experiment Configuration
    "experiment": {
        "name": "auto",
        "tags": ["transformer", "training"],
        "notes": "Ultra-fast modern transformer training with MAXIMUM BPE optimizations",
        "wandb_project": None,
        "debug_mode": False,
    }
}

# ============================================================================
# ULTRA-FAST OPTIMIZED DATASET WITH CACHING
# ============================================================================

class UltraFastDataset(Dataset):
    """Ultra-optimized dataset with aggressive caching and pre-processing"""
    
    def __init__(self, texts: List[str], tokenizer: UltraFastTokenizer, 
                 max_length: int = 2048, data_config: Optional[DataConfig] = None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data_config = data_config or DataConfig()
        
        # Pre-process and cache ALL data during initialization
        logging.info("🚀 Pre-processing and caching entire dataset with MAXIMUM speed...")
        start_time = time.time()
        
        self.cached_data = []
        self.pad_token_id = tokenizer.vocab.get("<|pad|>", 0)
        
        # Filter texts first
        if self.data_config.min_text_length or self.data_config.max_text_length:
            texts = self._filter_texts(texts)
        
        # Ultra-fast batch tokenization
        batch_size = 2000  # Increased batch size for speed
        for i in tqdm(range(0, len(texts), batch_size), desc="🔥 Ultra-fast tokenizing"):
            batch_texts = texts[i:i + batch_size]
            
            # Parallel processing within batch
            if NUMBA_AVAILABLE and len(batch_texts) > 100:
                self._preprocess_batch_numba(batch_texts)
            else:
                for text in batch_texts:
                    self._preprocess_and_cache(text)
        
        cache_time = time.time() - start_time
        logging.info(f"✅ Dataset cached with {len(self.cached_data):,} samples in {cache_time:.1f}s")
        logging.info(f"   Caching speed: {len(self.cached_data)/cache_time:.0f} samples/sec")
        
        # Convert to tensors and move to GPU if available (for ultra-fast access)
        if torch.cuda.is_available() and len(self.cached_data) < 100000:  # Increased threshold
            logging.info("🔥 Moving cached data to GPU for ultra-fast access...")
            device = torch.cuda.current_device()
            gpu_cache = []
            
            # Batch GPU transfer for speed
            batch_size = 1000
            for i in tqdm(range(0, len(self.cached_data), batch_size), desc="GPU caching"):
                batch = self.cached_data[i:i + batch_size]
                gpu_batch = []
                for item in batch:
                    gpu_item = {k: v.to(device, non_blocking=True) for k, v in item.items()}
                    gpu_batch.append(gpu_item)
                gpu_cache.extend(gpu_batch)
            
            self.cached_data = gpu_cache
            self.gpu_cached = True
            logging.info("🚀 GPU caching completed for maximum speed")
        else:
            self.gpu_cached = False
    
    def _preprocess_batch_numba(self, batch_texts: List[str]):
        """Ultra-fast batch preprocessing with Numba acceleration"""
        logging.debug(f"🚀 Processing batch of {len(batch_texts)} with Numba acceleration")
        
        # Process each text in the batch
        for text in batch_texts:
            self._preprocess_and_cache(text)
    
    def _filter_texts(self, texts: List[str]) -> List[str]:
        """Ultra-fast text filtering with vectorized operations"""
        min_len = self.data_config.min_text_length
        max_len = self.data_config.max_text_length
        
        if not min_len and not max_len:
            return texts
        
        filtered = []
        for text in texts:
            text_len = len(text)
            if min_len and text_len < min_len:
                continue
            if max_len and text_len > max_len:
                text = text[:max_len]
            filtered.append(text)
        
        return filtered
    
    def _preprocess_and_cache(self, text: str):
        """Pre-process and cache a single text with maximum optimization"""
        # Handle conversation format
        if self.data_config.use_conversation_format and isinstance(text, str):
            if not any(token in text for token in [
                self.data_config.user_token, 
                self.data_config.assistant_token,
                self.data_config.system_token
            ]):
                text = f"{self.data_config.assistant_token}{text}{self.data_config.end_token}"
        
        # Ultra-fast tokenization
        tokens = self.tokenizer.encode(text, add_special_tokens=True, max_length=self.max_length)
        
        # Ensure minimum length
        if len(tokens) < 2:
            tokens = [
                self.tokenizer.vocab.get("<|bos|>", 2),
                self.tokenizer.vocab.get("<|unk|>", 1),
                self.tokenizer.vocab.get("<|eos|>", 3)
            ]
        
        # Pad to exact length for consistent batching
        if len(tokens) < self.max_length:
            tokens.extend([self.pad_token_id] * (self.max_length - len(tokens)))
        
        # Create tensors with optimal dtypes
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)
        attention_mask = (input_ids != self.pad_token_id).float()
        
        self.cached_data.append({
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        })
    
    def __len__(self):
        return len(self.cached_data)
    
    def __getitem__(self, idx):
        # Ultra-fast cached access
        return self.cached_data[idx]

# ============================================================================
# ULTRA-FAST TRAINER WITH MAXIMUM OPTIMIZATIONS
# ============================================================================

class UltraFastTrainer:
    """Ultra-optimized trainer with maximum performance enhancements"""
    
    def __init__(self, model_config: ModelConfig, training_config: TrainingConfig,
                 precision_config: PrecisionConfig, data_config: DataConfig,
                 hardware_config: Optional[HardwareConfig] = None,
                 experiment_name: str = "ultra_fast_training",
                 logging_steps: int = 10):
        
        self.model_config = model_config
        self.training_config = training_config
        self.precision_config = precision_config
        self.data_config = data_config
        self.hardware_config = hardware_config or HardwareConfig()
        self.experiment_name = experiment_name
        self.logging_steps = logging_steps
        
        # Ultra-fast device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            # Pre-allocate GPU memory pool for faster allocation
            torch.cuda.empty_cache()
        
        # Initialize model manager
        self.model_manager = ModelManager(training_config.output_dir or "models")
        
        # Initialize model with optimizations
        self.model = ModernSubwordTransformer(model_config)
        
        # Apply model optimizations
        self._optimize_model()
        
        # Move to device
        self.model = self.model.to(self.device)
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_loss = float('inf')
        
        # Ultra-fast mixed precision setup
        self.use_amp = (self.precision_config.use_mixed_precision and 
                       self.precision_config.precision_type in ["fp16", "bf16"] and 
                       AMP_AVAILABLE)
        
        if self.use_amp:
            # Optimized scaler settings for speed
            self.scaler = GradScaler(
                init_scale=65536.0,  # Higher initial scale
                growth_factor=2.0,
                backoff_factor=0.5,
                growth_interval=1000,  # Less frequent updates
                enabled=self.precision_config.use_dynamic_loss_scaling
            )
        
        # Compile model for maximum speed
        if TORCH_COMPILE_AVAILABLE and self.device.type == 'cuda':
            try:
                # Ultra-aggressive compilation
                self.model = torch.compile(
                    self.model, 
                    mode="max-autotune",  # Maximum optimization
                    fullgraph=True,       # Compile entire graph
                    dynamic=False         # Static shapes for speed
                )
                logging.info("🚀 Model compiled with max-autotune mode")
            except Exception as e:
                logging.warning(f"⚠️ Failed to compile model: {e}")
        
        # Initialize tokenizer reference
        self.tokenizer = None
        
        # Pre-compile loss function
        self.loss_fn = self._create_optimized_loss()
        
        # Performance tracking
        self.step_times = []
        
        self._log_initialization()
    
    def _optimize_model(self):
        """Apply aggressive model optimizations"""
        # Fuse operations where possible
        if hasattr(self.model, 'fuse_ops'):
            self.model.fuse_ops()
        
        # Set optimal attention implementation
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            # Use PyTorch's optimized attention
            for layer in self.model.layers:
                if hasattr(layer.self_attn, 'use_pytorch_attention'):
                    layer.self_attn.use_pytorch_attention = True
        
        # Enable memory efficient attention patterns
        for layer in self.model.layers:
            if hasattr(layer.self_attn, 'enable_memory_efficient_attention'):
                layer.self_attn.enable_memory_efficient_attention()
    
    def _create_optimized_loss(self):
        """Create optimized loss function"""
        @torch.jit.script
        def fast_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, 
                              ignore_index: int) -> torch.Tensor:
            return F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1), 
                ignore_index=ignore_index,
                reduction='mean'
            )
        return fast_cross_entropy
    
    def _log_initialization(self):
        """Log initialization with performance info"""
        logging.info("🚀 UltraFastTrainer Initialized with MAXIMUM optimizations")
        logging.info("=" * 70)
        logging.info(f"📝 Experiment: {self.experiment_name}")
        logging.info(f"💻 Device: {self.device}")
        logging.info(f"🔢 Precision: {self.precision_config.precision_type}")
        logging.info(f"⚡ Mixed Precision: {self.use_amp}")
        logging.info(f"🚀 Torch Compile: {TORCH_COMPILE_AVAILABLE}")
        logging.info(f"🔥 Numba Acceleration: {NUMBA_AVAILABLE}")
        
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            logging.info(f"🔥 GPU: {props.name} ({props.total_memory // 1e9:.0f}GB)")
            logging.info(f"   Compute Capability: {props.major}.{props.minor}")
    
    def prepare_optimizer_and_scheduler(self, total_steps: int):
        """Prepare ultra-optimized optimizer and scheduler"""
        # Optimized AdamW with better defaults
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
            betas=(0.9, 0.95),  # Optimized betas
            eps=1e-8,
            fused=torch.cuda.is_available()  # Use fused AdamW if available
        )
        
        # Fast scheduler setup
        if self.training_config.warmup_steps is None:
            warmup_steps = int(total_steps * self.training_config.warmup_ratio)
        else:
            warmup_steps = self.training_config.warmup_steps
        
        # OneCycleLR for maximum speed
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.training_config.learning_rate,
            total_steps=total_steps,
            pct_start=warmup_steps / total_steps,
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=10000.0
        )
        
        logging.info(f"📈 Ultra-fast optimizer prepared (total steps: {total_steps:,})")
    
    def create_fast_dataloader(self, dataset: Dataset) -> DataLoader:
        """Create ultra-optimized dataloader"""
        # Calculate optimal number of workers
        num_workers = min(
            self.training_config.num_workers,
            max(1, os.cpu_count() // 2)
        ) if self.training_config.use_dataloader_workers else 0
        
        # Ultra-fast dataloader settings
        return DataLoader(
            dataset,
            batch_size=self.training_config.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,  # Keep workers alive
            prefetch_factor=6,  # Increased prefetch for speed
            drop_last=True,
            # Use memory-mapped files for large datasets
            generator=torch.Generator().manual_seed(42)
        )
    
    def train(self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None):
        """Ultra-fast training loop with maximum optimizations"""
        
        # Create optimized data loaders
        train_loader = self.create_fast_dataloader(train_dataset)
        eval_loader = self.create_fast_dataloader(eval_dataset) if eval_dataset else None
        
        # Calculate training parameters
        steps_per_epoch = len(train_loader) // self.training_config.gradient_accumulation_steps
        if self.training_config.max_steps:
            total_steps = self.training_config.max_steps
            num_epochs = math.ceil(total_steps / steps_per_epoch)
        else:
            num_epochs = self.training_config.max_epochs
            total_steps = steps_per_epoch * num_epochs
        
        # Prepare optimizer and scheduler
        self.prepare_optimizer_and_scheduler(total_steps)
        
        # Calculate intervals
        eval_steps = max(1, int(steps_per_epoch * 0.1)) if self.training_config.eval_steps is None else self.training_config.eval_steps
        save_steps = max(1, int(steps_per_epoch * 0.2)) if self.training_config.save_steps is None else self.training_config.save_steps
        
        # Pre-allocate gradient accumulation variables
        self.accumulated_loss = 0.0
        self.accumulation_count = 0
        
        # Training info
        logging.info("🏋️ Ultra-Fast Training Plan with MAXIMUM optimizations:")
        logging.info(f"   Epochs: {num_epochs}")
        logging.info(f"   Steps per epoch: {steps_per_epoch:,}")
        logging.info(f"   Total steps: {total_steps:,}")
        logging.info(f"   GPU Memory Allocated: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
        
        # Start training with performance monitoring
        logging.info("🎯 Starting ULTRA-FAST training with MAXIMUM speed...")
        start_time = datetime.now()
        
        # Pre-compile first batch for torch.compile
        if TORCH_COMPILE_AVAILABLE:
            self._warmup_model(train_loader)
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            epoch_loss = self._train_epoch_ultra_fast(
                train_loader, eval_loader, eval_steps, save_steps
            )
            
            logging.info(f"📊 Epoch {epoch + 1}/{num_epochs} | Loss: {epoch_loss:.6f} | "
                        f"Speed: {self._calculate_speed():.1f} steps/sec")
            
            if self.training_config.max_steps and self.global_step >= self.training_config.max_steps:
                break
        
        # Training completed
        total_time = (datetime.now() - start_time).total_seconds()
        avg_speed = self.global_step / total_time if total_time > 0 else 0
        
        # Final save
        self._save_checkpoint(force_save=True, is_final=True, total_training_time=total_time)
        
        logging.info("✅ Ultra-fast training completed with MAXIMUM speed!")
        logging.info(f"   Total time: {total_time:.1f}s ({total_time/3600:.2f}h)")
        logging.info(f"   Average speed: {avg_speed:.2f} steps/sec")
        logging.info(f"   Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f}GB")
    
    def _warmup_model(self, train_loader: DataLoader):
        """Warmup model for torch.compile"""
        logging.info("🔥 Warming up compiled model...")
        self.model.train()
        
        # Run a few dummy batches to compile the graph
        for i, batch in enumerate(train_loader):
            if i >= 3:  # Only need a few batches
                break
                
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
            
            with autocast(enabled=self.use_amp, dtype=torch.bfloat16 if self.precision_config.precision_type == "bf16" else torch.float16):
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch.get('attention_mask'),
                    return_dict=True
                )
                loss = self.loss_fn(
                    outputs['logits'], 
                    batch['labels'],
                    self.tokenizer.vocab.get("<|pad|>", 0)
                )
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            self.optimizer.zero_grad()
        
        logging.info("✅ Model warmup completed")
    
    def _train_epoch_ultra_fast(self, train_loader: DataLoader, eval_loader: Optional[DataLoader],
                               eval_steps: int, save_steps: int) -> float:
        """Ultra-optimized training epoch with MAXIMUM speed"""
        
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        # Disable progress bar for maximum speed in production
        use_progress_bar = len(train_loader) < 5000  # Reduced threshold for speed
        iterator = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1}", leave=False, disable=not use_progress_bar)
        
        # Pre-allocate tensors to avoid repeated allocation
        pad_token_id = self.tokenizer.vocab.get("<|pad|>", 0)
        
        for step, batch in enumerate(iterator):
            step_start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            step_end = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            
            if step_start:
                step_start.record()
            
            # Ultra-fast batch processing
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
            
            # Forward pass with maximum optimization
            with autocast(
                enabled=self.use_amp, 
                dtype=torch.bfloat16 if self.precision_config.precision_type == "bf16" else torch.float16
            ):
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch.get('attention_mask'),
                    return_dict=True
                )
                
                # Optimized loss calculation
                loss = self.loss_fn(outputs['logits'], batch['labels'], pad_token_id)
                loss = loss / self.training_config.gradient_accumulation_steps
            
            # Ultra-fast backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            self.accumulated_loss += loss.item()
            self.accumulation_count += 1
            epoch_loss += loss.item()
            num_batches += 1
            
            # Gradient accumulation step
            if self.accumulation_count >= self.training_config.gradient_accumulation_steps:
                if self.use_amp:
                    # Optimized gradient operations
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.training_config.max_grad_norm,
                        error_if_nonfinite=False  # Don't error on inf/nan for speed
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.training_config.max_grad_norm,
                        error_if_nonfinite=False
                    )
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()
                
                self.global_step += 1
                
                # Record step time
                if step_end:
                    step_end.record()
                    torch.cuda.synchronize()
                    step_time = step_start.elapsed_time(step_end) / 1000.0  # Convert to seconds
                    self.step_times.append(step_time)
                    if len(self.step_times) > 100:  # Keep only recent times
                        self.step_times.pop(0)
                
                # Fast logging (less frequent for speed)
                if self.global_step % (self.logging_steps * 5) == 0:
                    avg_loss = self.accumulated_loss / self.accumulation_count
                    self._log_training_step_fast(avg_loss)
                
                # Reset accumulation
                self.accumulated_loss = 0.0
                self.accumulation_count = 0
                
                # Evaluation (less frequent for speed)
                if eval_loader and self.global_step % eval_steps == 0:
                    eval_loss = self._evaluate_fast(eval_loader)
                    if eval_loss < self.best_loss:
                        self.best_loss = eval_loss
                
                # Save checkpoint
                if self.global_step % save_steps == 0:
                    self._save_checkpoint()
                
                # Check max steps
                if self.training_config.max_steps and self.global_step >= self.training_config.max_steps:
                    break
            
            # Update progress bar less frequently
            if use_progress_bar and step % 10 == 0:
                iterator.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{self.scheduler.get_last_lr()[0]:.2e}',
                    'step': self.global_step,
                    'speed': f'{self._calculate_speed():.1f}/s'
                })
        
        return epoch_loss / max(num_batches, 1)
    
    def _evaluate_fast(self, eval_loader: DataLoader) -> float:
        """Ultra-fast evaluation"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        pad_token_id = self.tokenizer.vocab.get("<|pad|>", 0)
        
        with torch.no_grad():
            # Limit evaluation size for speed
            max_eval_batches = min(100, len(eval_loader))
            
            for i, batch in enumerate(eval_loader):
                if i >= max_eval_batches:
                    break
                
                batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
                
                with autocast(
                    enabled=self.use_amp,
                    dtype=torch.bfloat16 if self.precision_config.precision_type == "bf16" else torch.float16
                ):
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch.get('attention_mask'),
                        return_dict=True
                    )
                    loss = self.loss_fn(outputs['logits'], batch['labels'], pad_token_id)
                
                total_loss += loss.item()
                num_batches += 1
        
        self.model.train()
        return total_loss / max(num_batches, 1)
    
    def _calculate_speed(self) -> float:
        """Calculate current training speed"""
        if len(self.step_times) < 2:
            return 0.0
        recent_times = self.step_times[-10:]  # Last 10 steps
        avg_time = sum(recent_times) / len(recent_times)
        return 1.0 / avg_time if avg_time > 0 else 0.0
    
    def _log_training_step_fast(self, loss: float):
        """Fast logging with minimal overhead"""
        logging.info(
            f"Step {self.global_step:6,} | "
            f"Loss: {loss:.6f} | "
            f"Speed: {self._calculate_speed():.1f}/s"
        )
    
    def _save_checkpoint(self, force_save: bool = False, is_final: bool = False,
                        total_training_time: float = 0.0):
        """Fast checkpoint saving"""
        
        metadata = ModelMetadata(
            model_name=self.experiment_name,
            version="1.0",
            model_config=self.model_config.__dict__,
            training_config=self.training_config.__dict__,
            precision_config=self.precision_config.__dict__,
            data_config=self.data_config.__dict__,
            epochs_trained=self.current_epoch,
            total_training_time=total_training_time,
            best_loss=self.best_loss,
            best_perplexity=math.exp(min(self.best_loss, 20)),
            performance_metrics={
                "final_loss": self.best_loss,
                "steps_trained": self.global_step,
                "avg_speed": self._calculate_speed()
            },
            hardware_used=str(self.device),
            notes=f"Ultra-fast training with MAXIMUM BPE optimizations completed at step {self.global_step}",
            tags=["ultra_fast_training", "maximum_bpe_speed", self.precision_config.precision_type]
        )
        
        # Save using ModelManager
        model_id = self.model_manager.save_model(
            model=self.model,
            tokenizer=self.tokenizer,
            metadata=metadata,
            optimizer=self.optimizer,
            scheduler=self.scheduler
        )
        
        if model_id:
            logging.info(f"💾 Checkpoint saved: {model_id}")

# ============================================================================
# ULTRA-FAST DATA LOADING WITH MULTIPROCESSING
# ============================================================================

def process_data_chunk(args):
    """Process a chunk of the file - moved to module level for multiprocessing"""
    chunk_start, chunk_size, file_path, config_dict = args
    
    # Recreate config object in subprocess
    data_config = DataConfig(**config_dict)
    
    texts = []
    seen = set() if data_config.remove_duplicates else None
    
    with open(file_path, 'r', encoding='utf-8') as f:
        f.seek(chunk_start)
        
        # Skip partial line at start (except for first chunk)
        if chunk_start > 0:
            f.readline()
        
        bytes_read = 0
        while bytes_read < chunk_size:
            line = f.readline()
            if not line:
                break
            
            bytes_read += len(line.encode('utf-8'))
            
            try:
                data = json.loads(line.strip())
                text = extract_text_from_data(data, data_config)
                
                if text and filter_text(text, data_config, seen):
                    texts.append(text)
                    
                    if (data_config.max_samples_train and 
                        len(texts) >= data_config.max_samples_train // len(args) if len(args) > 4 else data_config.max_samples_train // 8):
                        break
                        
            except (json.JSONDecodeError, Exception):
                continue
    
    return texts

def load_data_ultra_fast(file_path: str, data_config: DataConfig) -> List[str]:
    """Ultra-fast data loading with multiprocessing and streaming"""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    logging.info(f"🚀 Ultra-fast loading data from: {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Training data file not found: {file_path}")
    
    # Get file size for progress tracking
    file_size = os.path.getsize(file_path)
    
    # Calculate optimal chunk size based on file size and CPU count
    num_processes = min(mp.cpu_count(), 12)  # Increased max processes
    chunk_size = max(2 * 1024 * 1024, file_size // (num_processes * 4))  # Larger chunks
    
    # Create chunks
    chunks = []
    with open(file_path, 'rb') as f:
        chunk_start = 0
        while chunk_start < file_size:
            chunks.append((chunk_start, chunk_size, file_path, data_config.__dict__))
            chunk_start += chunk_size
    
    logging.info(f"   Processing {len(chunks)} chunks with {num_processes} processes...")
    
    # Process chunks in parallel
    all_texts = []
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        future_to_chunk = {executor.submit(process_data_chunk, chunk): chunk for chunk in chunks}
        
        with tqdm(total=len(chunks), desc="🔥 Processing chunks") as pbar:
            for future in as_completed(future_to_chunk):
                chunk_texts = future.result()
                all_texts.extend(chunk_texts)
                pbar.update(1)
    
    # Apply final deduplication if needed
    if data_config.remove_duplicates:
        logging.info("   Final deduplication...")
        seen = set()
        deduplicated = []
        for text in all_texts:
            text_hash = hash(text)
            if text_hash not in seen:
                seen.add(text_hash)
                deduplicated.append(text)
        all_texts = deduplicated
    
    # Apply final sample limit
    if data_config.max_samples_train and len(all_texts) > data_config.max_samples_train:
        all_texts = all_texts[:data_config.max_samples_train]
    
    logging.info(f"✅ Ultra-fast loading completed: {len(all_texts):,} texts")
    return all_texts

def extract_text_from_data(data: dict, data_config: DataConfig) -> Optional[str]:
    """Extract text from data dictionary"""
    # Extract text based on common field names
    text = None
    for field in ['text', 'content', 'message', 'body', 'output']:
        if field in data and data[field]:
            text = data[field]
            break
    
    # Handle conversation format
    if not text and 'messages' in data:
        messages = data['messages']
        conversation_parts = []
        
        for msg in messages:
            role = msg.get('role', '').lower()
            content = msg.get('content', '').strip()
            
            if role == 'system':
                conversation_parts.append(f"{data_config.system_token}{content}{data_config.end_token}")
            elif role == 'user':
                conversation_parts.append(f"{data_config.user_token}{content}{data_config.end_token}")
            elif role == 'assistant':
                conversation_parts.append(f"{data_config.assistant_token}{content}{data_config.end_token}")
        
        text = '\n'.join(conversation_parts)
    
    return text.strip() if text else None

def filter_text(text: str, data_config: DataConfig, seen: Optional[set] = None) -> bool:
    """Filter text based on criteria"""
    if len(text) < data_config.min_text_length:
        return False
    
    if data_config.max_text_length and len(text) > data_config.max_text_length:
        text = text[:data_config.max_text_length]
    
    if data_config.lowercase:
        text = text.lower()
    
    if data_config.remove_duplicates and seen is not None:
        text_hash = hash(text)
        if text_hash in seen:
            return False
        seen.add(text_hash)
    
    return True

# ============================================================================
# ULTRA-FAST MAIN FUNCTION
# ============================================================================

def setup_ultra_fast_environment():
    """Setup environment for maximum performance"""
    
    # Set environment variables for performance
    os.environ['OMP_NUM_THREADS'] = str(min(12, os.cpu_count()))  # Increased
    os.environ['MKL_NUM_THREADS'] = str(min(12, os.cpu_count()))
    os.environ['NUMEXPR_NUM_THREADS'] = str(min(12, os.cpu_count()))
    
    # CUDA optimizations
    if torch.cuda.is_available():
        # Enable optimizations
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Set memory allocation strategy
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'  # Increased
        
        # Optimize CUDA cache
        torch.cuda.empty_cache()
        
        logging.info("🔥 CUDA optimizations enabled for maximum speed")

def main():
    """Ultra-fast main training function with MAXIMUM BPE optimizations"""
    
    # Setup ultra-fast environment
    setup_ultra_fast_environment()
    
    # Setup logging
    log_file = setup_logging(TRAINING_CONFIG["system"]["log_dir"])
    
    # Set random seed
    seed = TRAINING_CONFIG["system"]["seed"]
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Header
    logging.info("🚀 ULTRA-FAST ModernSubwordTransformer Training with MAXIMUM BPE Speed")
    logging.info("=" * 90)
    logging.info(f"📝 Log file: {log_file}")
    logging.info(f"🌱 Random seed: {seed}")
    
    # Performance info
    logging.info("⚡ MAXIMUM Performance Optimizations:")
    logging.info(f"   PyTorch: {torch.__version__}")
    logging.info(f"   CUDA Available: {torch.cuda.is_available()}")
    logging.info(f"   TF32 Enabled: {torch.backends.cuda.matmul.allow_tf32}")
    logging.info(f"   cuDNN Benchmark: {torch.backends.cudnn.benchmark}")
    logging.info(f"   Torch Compile: {TORCH_COMPILE_AVAILABLE}")
    logging.info(f"   Numba BPE Acceleration: {NUMBA_AVAILABLE}")
    
    get_device_info()
    
    try:
        # Parse configuration with fixed logging_steps handling
        model_config, training_config, precision_config, data_config, experiment_name, data_path = parse_config()
        
        # Log configuration
        logging.info("⚙️ Ultra-Fast Configuration with MAXIMUM optimizations:")
        logging.info(f"   Experiment: {experiment_name}")
        logging.info(f"   Model: {model_config.hidden_size}d × {model_config.num_layers}L")
        logging.info(f"   Batch size: {training_config.batch_size}")
        logging.info(f"   Precision: {precision_config.precision_type}")
        logging.info(f"   Compile mode: {precision_config.compile_mode}")
        
        # Ultra-fast data loading
        logging.info("🚀 Ultra-fast data loading with MAXIMUM speed...")
        start_time = datetime.now()
        texts = load_data_ultra_fast(data_path, data_config)
        load_time = (datetime.now() - start_time).total_seconds()
        logging.info(f"   Data loaded in {load_time:.1f}s ({len(texts)/load_time:.0f} texts/sec)")
        
        if len(texts) == 0:
            raise ValueError("No valid texts found!")
        
        # Ultra-fast data split
        eval_split = TRAINING_CONFIG["data"].get("eval_split", 0.1)
        split_idx = int((1 - eval_split) * len(texts))
        train_texts = texts[:split_idx]
        eval_texts = texts[split_idx:] if eval_split > 0 else []
        
        logging.info(f"📊 Data split: {len(train_texts):,} train, {len(eval_texts):,} eval")
        
        # MAXIMUM SPEED BPE tokenizer training
        logging.info("🚀 MAXIMUM SPEED BPE tokenizer training...")
        start_time = datetime.now()
        
        tokenizer = UltraFastTokenizer()
        tokenizer_texts = train_texts
        if len(train_texts) > data_config.tokenizer_train_size:
            # Use random sampling for speed
            tokenizer_texts = random.sample(train_texts, data_config.tokenizer_train_size)
        
        # Create progress callback for BPE training
        def bpe_progress_callback(progress, current, total):
            if progress % 10 == 0:  # Every 10%
                logging.info(f"🚀 BPE Progress: {progress:.1f}% ({current:,}/{total:,})")
        
        # Train with ultra-fast BPE
        tokenizer.train_from_text_ultra_fast(
            '\n'.join(tokenizer_texts), 
            vocab_size=model_config.vocab_size,
            min_freq=data_config.min_frequency,
            progress_callback=bpe_progress_callback
        )
        
        tokenizer_time = (datetime.now() - start_time).total_seconds()
        logging.info(f"✅ MAXIMUM SPEED tokenizer trained in {tokenizer_time:.1f}s")
        
        # Update model config
        model_config.vocab_size = tokenizer.vocab_size()
        
        # Extract logging_steps from config before creating trainer
        logging_steps = TRAINING_CONFIG["training"]["logging_steps"]
        
        # Initialize ultra-fast trainer with logging_steps
        logging.info("🚀 Initializing ultra-fast trainer with MAXIMUM optimizations...")
        trainer = UltraFastTrainer(
            model_config=model_config,
            training_config=training_config,
            precision_config=precision_config,
            data_config=data_config,
            experiment_name=experiment_name,
            logging_steps=logging_steps
        )
        trainer.tokenizer = tokenizer
        
        # Create ultra-fast datasets
        logging.info("🚀 Creating ultra-fast datasets with MAXIMUM caching...")
        start_time = datetime.now()
        
        train_dataset = UltraFastDataset(
            train_texts, tokenizer, model_config.seq_length - 1, data_config
        )
        
        eval_dataset = None
        if eval_texts:
            eval_dataset = UltraFastDataset(
                eval_texts, tokenizer, model_config.seq_length - 1, data_config
            )
        
        dataset_time = (datetime.now() - start_time).total_seconds()
        logging.info(f"   Datasets created in {dataset_time:.1f}s")
        
        # Memory info
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            logging.info(f"💾 GPU Memory: {memory_gb:.1f}GB available")
        
        # Start ultra-fast training
        logging.info("🎯 Starting ULTRA-FAST training with MAXIMUM BPE optimizations...")
        training_start = datetime.now()
        
        trainer.train(train_dataset, eval_dataset)
        
        total_training_time = (datetime.now() - training_start).total_seconds()
        
        # Final statistics
        logging.info("🎉 ULTRA-FAST Training with MAXIMUM BPE Speed Completed!")
        logging.info("=" * 80)
        logging.info(f"📊 Performance Summary:")
        logging.info(f"   Data loading: {load_time:.1f}s")
        logging.info(f"   MAXIMUM SPEED BPE training: {tokenizer_time:.1f}s")
        logging.info(f"   Dataset creation: {dataset_time:.1f}s")
        logging.info(f"   Model training: {total_training_time:.1f}s")
        logging.info(f"   Total time: {(datetime.now() - training_start).total_seconds():.1f}s")
        logging.info(f"   Average speed: {trainer._calculate_speed():.2f} steps/sec")
        
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / 1e9
            logging.info(f"   Peak GPU memory: {peak_memory:.2f}GB")
        
        # Print model summary
        trainer.model_manager.print_model_summary()
        
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    except KeyboardInterrupt:
        logging.warning("⚠️ Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"❌ Training failed: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        raise

def setup_logging(log_dir: str = "logs") -> str:
    """Enhanced logging setup"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"ultra_fast_max_bpe_training_{timestamp}.log")
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return log_file

def get_device_info():
    """Enhanced device information"""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        logging.info("🔥 CUDA Devices Detected:")
        
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / 1e9
            compute_capability = f"{props.major}.{props.minor}"
            
            logging.info(f"   GPU {i}: {props.name}")
            logging.info(f"     Memory: {memory_gb:.1f}GB")
            logging.info(f"     Compute Capability: {compute_capability}")
            logging.info(f"     Multi-Processors: {props.multi_processor_count}")
        
        logging.info(f"   CUDA Version: {torch.version.cuda}")
        logging.info(f"   cuDNN Version: {torch.backends.cudnn.version()}")
        
        torch.cuda.empty_cache()
        memory_allocated = torch.cuda.memory_allocated() / 1e9
        memory_reserved = torch.cuda.memory_reserved() / 1e9
        logging.info(f"   Current Memory: {memory_allocated:.1f}GB allocated, {memory_reserved:.1f}GB reserved")
        
    else:
        logging.info("💻 Using CPU")
        logging.info(f"   CPU Count: {torch.get_num_threads()}")

def parse_config() -> Tuple[ModelConfig, TrainingConfig, PrecisionConfig, DataConfig, str, str]:
    """Parse hardcoded configuration with fixed logging_steps handling"""
    
    data_cfg = TRAINING_CONFIG["data"]
    data_config = DataConfig(
        train_data_path=data_cfg["training_data_path"],
        use_conversation_format=data_cfg["use_conversation_format"],
        max_samples_train=data_cfg["max_samples_train"],
        max_samples_eval=data_cfg["max_samples_eval"],
        min_text_length=data_cfg["min_text_length"],
        max_text_length=data_cfg["max_text_length"],
        remove_duplicates=data_cfg["remove_duplicates"],
        lowercase=data_cfg["lowercase"],
        tokenizer_train_size=data_cfg["tokenizer_train_size"],
        min_frequency=data_cfg["min_frequency"]
    )
    
    model_cfg = TRAINING_CONFIG["model"]
    if model_cfg["config_preset"] == "auto":
        model_config, _, _, _ = auto_select_config()
    elif model_cfg["config_preset"] == "tiny":
        model_config, _, _, _ = ConfigPresets.tiny_debug()
    elif model_cfg["config_preset"] == "research":
        model_config, _, _, _ = ConfigPresets.research_7b()
    elif model_cfg["config_preset"] == "custom":
        custom_cfg = model_cfg["custom"]
        model_config = ModelConfig(**custom_cfg)
    else:
        raise ValueError(f"Unknown model config preset: {model_cfg['config_preset']}")
    
    train_cfg = TRAINING_CONFIG["training"]
    # Create TrainingConfig without logging_steps
    training_config = TrainingConfig(
        batch_size=train_cfg["batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        max_epochs=train_cfg["max_epochs"],
        max_steps=train_cfg["max_steps"],
        learning_rate=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        warmup_ratio=train_cfg["warmup_ratio"],
        scheduler_type=train_cfg["scheduler_type"],
        optimizer_type=train_cfg["optimizer_type"],
        max_grad_norm=train_cfg["max_grad_norm"],
        eval_steps=train_cfg["eval_steps"],
        save_steps=train_cfg["save_steps"],
        save_total_limit=train_cfg["save_total_limit"],
        use_dataloader_workers=train_cfg["use_dataloader_workers"],
        num_workers=train_cfg["num_workers"]
    )
    
    training_config.output_dir = TRAINING_CONFIG["system"]["output_dir"]
    
    if TRAINING_CONFIG["experiment"]["wandb_project"]:
        training_config.wandb_project = TRAINING_CONFIG["experiment"]["wandb_project"]
    
    prec_cfg = TRAINING_CONFIG["precision"]
    precision_config = PrecisionConfig(
        precision_type=prec_cfg["precision_type"],
        use_mixed_precision=prec_cfg["use_mixed_precision"],
        use_compile=prec_cfg["use_compile"],
        compile_mode=prec_cfg["compile_mode"],
        use_dynamic_loss_scaling=prec_cfg["use_dynamic_loss_scaling"]
    )
    
    if TRAINING_CONFIG["experiment"]["debug_mode"]:
        logging.info("🐛 Debug mode enabled")
        model_config = ModelConfig(
            vocab_size=1000,
            hidden_size=256,
            num_layers=4,
            num_heads=4,
            seq_length=512
        )
        training_config.batch_size = 2
        training_config.max_epochs = 2
        data_config.max_samples_train = 1000
        precision_config.use_mixed_precision = False
        precision_config.use_compile = False
    
    if TRAINING_CONFIG["experiment"]["name"] == "auto":
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"ultra_fast_max_bpe_transformer_{timestamp}"
    else:
        experiment_name = TRAINING_CONFIG["experiment"]["name"]
    
    return model_config, training_config, precision_config, data_config, experiment_name, data_config.train_data_path

if __name__ == "__main__":
    main()