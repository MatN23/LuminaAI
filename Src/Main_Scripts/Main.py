# main.py - Complete Training System (Fixed)
# Copyright (c) 2025 Matias Nielsen. All rights reserved.

import os
import sys
import json
import logging
import argparse
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from datetime import datetime
import time
import psutil
import gc

# Import our modules (fixed imports)
from dataset import ConversationDataset, create_dataloader
from model import TransformerModel, estimate_parameters


# Missing tokenizer implementation (simplified for compatibility)
@dataclass
class TokenizerConfig:
    """Tokenizer configuration."""
    model: str = "gpt-4"
    max_sequence_length: int = 2048
    vocab_size: int = 50257


class ConversationTokenizer:
    """Simplified tokenizer for compatibility."""
    
    def __init__(self, config: TokenizerConfig = None):
        self.config = config or TokenizerConfig()
        self.vocab_size = self.config.vocab_size
        self.pad_token_id = 0
        
        # Special tokens
        self.special_tokens = {
            "<|pad|>": 0,
            "<|im_start|>": 1,
            "<|im_end|>": 2,
            "<|assistant|>": 3,
            "<|user|>": 4,
            "<|system|>": 5,
        }
    
    def encode_conversation(self, conversation: Dict) -> List[int]:
        """Encode conversation to token IDs."""
        tokens = []
        
        for message in conversation.get('messages', []):
            role = message.get('role', 'user')
            content = message.get('content', '')
            
            # Add role tokens
            tokens.append(self.special_tokens.get(f"<|{role}|>", 4))
            tokens.append(self.special_tokens["<|im_start|>"])
            
            # Simple word-based tokenization (replace with proper tokenizer)
            words = content.split()
            for i, word in enumerate(words[:100]):  # Limit to prevent overflow
                tokens.append(hash(word) % (self.vocab_size - 10) + 10)
            
            tokens.append(self.special_tokens["<|im_end|>"])
        
        return tokens
    
    def decode(self, tokens: List[int], skip_special_tokens: bool = True) -> str:
        """Decode tokens to text."""
        return f"Generated response with {len(tokens)} tokens"
    
    def get_role_token(self, role: str) -> int:
        """Get token ID for role."""
        return self.special_tokens.get(f"<|{role}|>", 4)


# Missing logger implementations
class ProductionLogger:
    """Production logger implementation."""
    
    def __init__(self, log_level: str, experiment_name: str):
        self.log_level = log_level
        self.experiment_name = experiment_name
        self.metrics_history = []
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
        )
    
    def log_metrics(self, metrics: Dict[str, Any], step: int, prefix: str = ""):
        """Log metrics."""
        self.metrics_history.append((step, metrics))
        
        if prefix:
            log_str = f"{prefix.upper()} - Step {step}: "
        else:
            log_str = f"Step {step}: "
        
        for key, value in metrics.items():
            if isinstance(value, float):
                log_str += f"{key}={value:.6f} "
            else:
                log_str += f"{key}={value} "
        
        logging.info(log_str.strip())
    
    def log_system_stats(self, step: int):
        """Log system statistics."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            logging.info(f"Step {step} - GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
        
        memory = psutil.virtual_memory()
        logging.info(f"Step {step} - System Memory: {memory.percent:.1f}% used")
    
    def close(self):
        """Close logger."""
        pass


class TrainingHealthMonitor:
    """Training health monitor."""
    
    def __init__(self):
        self.loss_history = []
        self.grad_norm_history = []
        self.nan_count = 0
        self.inf_count = 0
    
    def update(self, loss: float, grad_norm: float):
        """Update with new metrics."""
        self.loss_history.append(loss)
        self.grad_norm_history.append(grad_norm)
        
        if np.isnan(loss):
            self.nan_count += 1
        if np.isinf(loss):
            self.inf_count += 1
    
    def get_status(self) -> str:
        """Get health status."""
        if self.nan_count > 5 or self.inf_count > 5:
            return "CRITICAL"
        elif self.nan_count > 0 or self.inf_count > 0:
            return "WARNING"
        else:
            return "HEALTHY"
    
    def get_summary(self) -> Dict[str, Any]:
        """Get health summary."""
        return {
            'nan_count': self.nan_count,
            'inf_count': self.inf_count,
            'avg_loss': np.mean(self.loss_history) if self.loss_history else 0,
            'avg_grad_norm': np.mean(self.grad_norm_history) if self.grad_norm_history else 0
        }


@dataclass
class TrainingConfig:
    """Comprehensive training configuration (Fixed)."""
    # Model architecture
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    num_kv_heads: int = 4
    intermediate_size: int = 3072
    seq_length: int = 2048
    vocab_size: int = 50257
    dropout: float = 0.1
    
    # Stability settings
    init_std: float = 0.02
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    use_stable_embedding: bool = True
    
    # Training parameters
    batch_size: int = 2
    gradient_accumulation_steps: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_steps: int = 1000
    max_steps: int = 50000
    eval_steps: int = 1000
    save_steps: int = 2500
    
    # Loss configuration
    assistant_loss_weight: float = 2.0
    
    # Optimization
    optimizer_type: str = "adamw"
    scheduler_type: str = "cosine"
    use_amp: bool = True
    gradient_checkpointing: bool = True
    mixed_precision: str = "fp16"
    precision: str = "fp16"  # Add for compatibility
    
    # Data
    train_data_path: str = "data/train.jsonl"
    val_data_path: str = "data/val.jsonl"
    num_workers: int = 2
    
    # System
    device: str = "auto"
    compile_model: bool = False
    compile: bool = False  # Add for compatibility
    
    # Monitoring
    log_level: str = "INFO"
    experiment_name: Optional[str] = None
    use_wandb: bool = False
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    resume_from: Optional[str] = None
    keep_best: int = 3
    
    # Add missing attributes for compatibility
    num_epochs: int = 10
    effective_batch_size: int = 32
    eval_every_n_batches: int = 1000
    save_every_n_batches: int = 2500
    early_stopping_patience: Optional[int] = None
    health_check_interval: int = 100
    backup_every_n_hours: int = 6
    save_total_limit: int = 5
    lr_scheduler: str = "cosine"
    min_lr: float = 1e-6
    warmup_ratio: float = 0.02
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    max_new_tokens: int = 512
    max_retries: int = 3
    auto_resume: bool = True
    
    def __post_init__(self):
        if self.experiment_name is None:
            self.experiment_name = f"transformer_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Calculate effective batch size
        self.effective_batch_size = self.batch_size * self.gradient_accumulation_steps
        
        # Validation
        assert self.hidden_size % self.num_heads == 0
        assert self.num_heads % self.num_kv_heads == 0
        assert self.seq_length > 0
        assert self.batch_size > 0
    
    def save(self, path: str):
        """Save configuration to file."""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'TrainingConfig':
        """Load configuration from file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)
    
    @classmethod
    def from_yaml(cls, path: str) -> 'TrainingConfig':
        """Load from YAML with nested structure support."""
        try:
            import yaml
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
            
            # Flatten nested structure
            flat_data = {}
            if 'model' in data:
                flat_data.update(data['model'])
            if 'training' in data:
                flat_data.update(data['training'])
            if 'loss' in data:
                for k, v in data['loss'].items():
                    flat_data[k] = v
            if 'data' in data:
                flat_data['train_data_path'] = data['data'].get('train_data_path', 'data/train.jsonl')
                flat_data['val_data_path'] = data['data'].get('val_data_path', 'data/val.jsonl')
                flat_data['num_workers'] = data['data'].get('num_workers', 2)
            if 'system' in data:
                flat_data.update(data['system'])
            if 'monitoring' in data:
                flat_data.update(data['monitoring'])
            if 'checkpoints' in data:
                flat_data['checkpoint_dir'] = data['checkpoints'].get('checkpoint_dir', 'checkpoints')
                flat_data['keep_best'] = data['checkpoints'].get('keep_best', 3)
                flat_data['resume_from'] = data['checkpoints'].get('resume_from')
            
            return cls(**flat_data)
        except ImportError:
            raise ImportError("PyYAML required for YAML config loading")


class CheckpointManager:
    """Fixed checkpoint manager."""
    
    def __init__(self, checkpoint_dir: str, keep_best: int = 3):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_best = keep_best
        self.best_checkpoints = []
        
        logging.info(f"CheckpointManager initialized: {checkpoint_dir}")
    
    def save_checkpoint(self, model: nn.Module, optimizer, scheduler, 
                       step: int, loss: float, config: TrainingConfig,
                       is_best: bool = False, emergency: bool = False) -> str:
        """Save model checkpoint."""
        if emergency:
            filename = f"checkpoint_emergency_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        elif is_best:
            filename = f"checkpoint_best.pt"
        else:
            filename = f"checkpoint_step_{step}.pt"
        
        checkpoint_path = self.checkpoint_dir / filename
        
        checkpoint = {
            'step': step,
            'loss': loss,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'config': asdict(config),
            'timestamp': datetime.now().isoformat(),
            'torch_version': torch.__version__,
        }
        
        if torch.cuda.is_available():
            checkpoint['gpu_info'] = {
                'device_name': torch.cuda.get_device_name(),
                'memory_allocated': torch.cuda.memory_allocated(),
                'memory_reserved': torch.cuda.memory_reserved(),
            }
        
        temp_path = checkpoint_path.with_suffix('.tmp')
        torch.save(checkpoint, temp_path)
        temp_path.replace(checkpoint_path)
        
        logging.info(f"Checkpoint saved: {checkpoint_path}")
        
        if not emergency:
            self._update_best_checkpoints(loss, str(checkpoint_path))
        
        return str(checkpoint_path)
    
    def _update_best_checkpoints(self, loss: float, checkpoint_path: str):
        """Update best checkpoints list."""
        self.best_checkpoints.append((loss, checkpoint_path))
        self.best_checkpoints.sort(key=lambda x: x[0])
        
        if len(self.best_checkpoints) > self.keep_best:
            _, old_path = self.best_checkpoints.pop()
            if os.path.exists(old_path) and not old_path.endswith('best.pt'):
                try:
                    os.remove(old_path)
                    logging.info(f"Removed old checkpoint: {old_path}")
                except Exception as e:
                    logging.warning(f"Failed to remove checkpoint {old_path}: {e}")
    
    def load_checkpoint(self, checkpoint_path: str, model: nn.Module, 
                       optimizer=None, scheduler=None) -> Dict[str, Any]:
        """Load checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        logging.info(f"Loading checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler and checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logging.info(f"Checkpoint loaded: step {checkpoint.get('step', 0)}")
        
        return checkpoint


class Trainer:
    """Fixed trainer implementation."""
    
    def __init__(self, config: TrainingConfig, tokenizer: ConversationTokenizer):
        self.config = config
        self.tokenizer = tokenizer
        
        # Setup logging and monitoring
        self.logger = ProductionLogger(config.log_level, config.experiment_name)
        self.health_monitor = TrainingHealthMonitor()
        
        # Device setup
        self.device = torch.device(config.device)
        self.use_amp = config.use_amp and self.device.type == 'cuda'
        
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Initialize model
        self._initialize_model()
        
        # Setup optimization
        self._setup_optimization()
        
        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(config.checkpoint_dir, config.keep_best)
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        
        logging.info(f"Trainer initialized on {self.device}")
        self._log_memory_usage("Initial")
    
    def _initialize_model(self):
        """Initialize model."""
        self.config.vocab_size = self.tokenizer.vocab_size
        
        self.model = TransformerModel(self.config)
        self.model.to(self.device)
        
        if self.config.gradient_checkpointing:
            for layer in self.model.layers:
                if hasattr(layer, 'gradient_checkpointing'):
                    layer.gradient_checkpointing = True
        
        if self.config.compile_model and hasattr(torch, 'compile'):
            try:
                logging.info("Compiling model...")
                self.model = torch.compile(self.model, mode='default')
                logging.info("Model compiled successfully")
            except Exception as e:
                logging.warning(f"Model compilation failed: {e}")
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logging.info(f"Model: {total_params:,} total params ({trainable_params:,} trainable)")
    
    def _setup_optimization(self):
        """Setup optimizer and scheduler."""
        if self.config.optimizer_type.lower() == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=(0.9, 0.95),
                eps=1e-8
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer_type}")
        
        if self.config.scheduler_type.lower() == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.max_steps,
                eta_min=self.config.learning_rate * 0.1
            )
        elif self.config.scheduler_type.lower() == "linear":
            self.scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=self.config.max_steps
            )
        else:
            self.scheduler = None
    
    def _log_memory_usage(self, prefix: str = ""):
        """Log memory usage."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            max_memory = torch.cuda.max_memory_allocated() / 1e9
            
            logging.info(f"{prefix} - GPU Memory: {allocated:.2f}GB allocated, "
                        f"{reserved:.2f}GB reserved, {max_memory:.2f}GB max")
        
        memory = psutil.virtual_memory()
        logging.info(f"{prefix} - System Memory: {memory.percent:.1f}% used, "
                    f"{memory.available / 1e9:.1f}GB available")
    
    def train(self, train_dataloader: DataLoader, val_dataloader: Optional[DataLoader] = None):
        """Main training loop."""
        logging.info("Starting training...")
        
        try:
            self.model.train()
            
            steps_per_epoch = len(train_dataloader)
            total_epochs = self.config.max_steps // steps_per_epoch + 1
            effective_batch_size = self.config.batch_size * self.config.gradient_accumulation_steps
            
            logging.info(f"Training for {total_epochs} epochs, {self.config.max_steps:,} total steps")
            logging.info(f"Effective batch size: {effective_batch_size}")
            
            for epoch in range(total_epochs):
                if self.global_step >= self.config.max_steps:
                    break
                
                self.epoch = epoch
                logging.info(f"\n{'='*60}")
                logging.info(f"EPOCH {epoch + 1}/{total_epochs}")
                logging.info(f"{'='*60}")
                
                epoch_loss = 0.0
                epoch_steps = 0
                
                for batch_idx, batch in enumerate(train_dataloader):
                    if self.global_step >= self.config.max_steps:
                        break
                    
                    loss = self._training_step(batch, batch_idx)
                    epoch_loss += loss
                    epoch_steps += 1
                    
                    if self.global_step % 100 == 0:
                        self._log_training_step(loss)
                    
                    if val_dataloader and self.global_step % self.config.eval_steps == 0:
                        val_loss = self._validation_step(val_dataloader)
                        self._log_validation(val_loss)
                        
                        if val_loss < self.best_loss:
                            self.best_loss = val_loss
                            self.checkpoint_manager.save_checkpoint(
                                self.model, self.optimizer, self.scheduler,
                                self.global_step, val_loss, self.config, is_best=True
                            )
                    
                    if self.global_step % self.config.save_steps == 0:
                        self.checkpoint_manager.save_checkpoint(
                            self.model, self.optimizer, self.scheduler,
                            self.global_step, loss, self.config
                        )
                    
                    self.global_step += 1
                
                avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
                logging.info(f"Epoch {epoch + 1} completed - Average loss: {avg_epoch_loss:.4f}")
                
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            logging.info("Training completed successfully!")
            
        except Exception as e:
            logging.error(f"Training failed: {e}")
            logging.error(traceback.format_exc())
            
            try:
                self.checkpoint_manager.save_checkpoint(
                    self.model, self.optimizer, self.scheduler,
                    self.global_step, float('inf'), self.config, emergency=True
                )
            except Exception as save_error:
                logging.error(f"Failed to save emergency checkpoint: {save_error}")
            
            raise
    
    def _training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> float:
        """Training step."""
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        with torch.amp.autocast('cuda', enabled=self.use_amp):
            logits = self.model(batch['input_ids'], batch['attention_mask'])
            loss = self._calculate_loss(logits, batch['labels'], batch.get('loss_weights'))
            loss = loss / self.config.gradient_accumulation_steps
        
        if self.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
            if self.use_amp:
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
                self.optimizer.step()
            
            self.optimizer.zero_grad()
            
            if self.scheduler:
                self.scheduler.step()
            
            self.health_monitor.update(loss.item() * self.config.gradient_accumulation_steps, grad_norm.item())
        
        return loss.item() * self.config.gradient_accumulation_steps
    
    def _calculate_loss(self, logits: torch.Tensor, labels: torch.Tensor, 
                       loss_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Calculate loss."""
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_labels = shift_labels.view(-1)
        
        loss = nn.functional.cross_entropy(flat_logits, flat_labels, reduction='none')
        
        if loss_weights is not None:
            shift_weights = loss_weights[..., 1:].contiguous().view(-1)
            loss = loss * shift_weights
        
        mask = (flat_labels != 0).float()
        loss = (loss * mask).sum() / mask.sum().clamp(min=1.0)
        
        return loss
    
    def _validation_step(self, val_dataloader: DataLoader) -> float:
        """Validation step."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    logits = self.model(batch['input_ids'], batch['attention_mask'])
                    loss = self._calculate_loss(logits, batch['labels'], batch.get('loss_weights'))
                
                total_loss += loss.item()
                num_batches += 1
                
                if num_batches >= 100:
                    break
        
        self.model.train()
        return total_loss / max(num_batches, 1)
    
    def _log_training_step(self, loss: float):
        """Log training step."""
        lr = self.optimizer.param_groups[0]['lr']
        grad_norm = self.health_monitor.grad_norm_history[-1] if self.health_monitor.grad_norm_history else 0
        
        if hasattr(self, '_last_log_time'):
            time_diff = time.time() - self._last_log_time
            throughput = 100 / time_diff if time_diff > 0 else 0
        else:
            throughput = 0
        
        self._last_log_time = time.time()
        
        logging.info(f"Step {self.global_step:7d} | Loss: {loss:.4f} | "
                    f"LR: {lr:.2e} | Grad: {grad_norm:.3f} | "
                    f"Throughput: {throughput:.1f} steps/s")
        
        metrics = {
            'train/loss': loss,
            'train/learning_rate': lr,
            'train/grad_norm': grad_norm,
            'train/throughput': throughput
        }
        
        self.logger.log_metrics(metrics, self.global_step)
        
        if self.global_step % 500 == 0:
            self.logger.log_system_stats(self.global_step)
    
    def _log_validation(self, val_loss: float):
        """Log validation."""
        logging.info(f"Validation - Step {self.global_step:7d} | Loss: {val_loss:.4f}")
        
        metrics = {'val/loss': val_loss}
        self.logger.log_metrics(metrics, self.global_step)


def setup_datasets(config: TrainingConfig, tokenizer: ConversationTokenizer):
    """Setup datasets."""
    logging.info("Setting up datasets...")
    
    train_dataset = ConversationDataset(
        config.train_data_path, 
        tokenizer, 
        config,
        split="train"
    )
    
    val_dataset = None
    if config.val_data_path and Path(config.val_data_path).exists():
        val_dataset = ConversationDataset(
            config.val_data_path,
            tokenizer,
            config,
            split="val"
        )
    
    train_dataloader = create_dataloader(train_dataset, config, shuffle=True)
    val_dataloader = create_dataloader(val_dataset, config, shuffle=False) if val_dataset else None
    
    return train_dataloader, val_dataloader


def check_dependencies():
    """Check required dependencies."""
    required = ['torch', 'numpy', 'psutil']
    missing = []
    
    for module in required:
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    
    if missing:
        raise ImportError(f"Missing required modules: {missing}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train Conversational Transformer")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--train_data", type=str, help="Path to training data")
    parser.add_argument("--val_data", type=str, help="Path to validation data")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--experiment_name", type=str, help="Experiment name")
    
    args = parser.parse_args()
    
    try:
        # Check dependencies
        check_dependencies()
        
        # Load config
        if args.config and Path(args.config).exists():
            if args.config.endswith('.yaml') or args.config.endswith('.yml'):
                config = TrainingConfig.from_yaml(args.config)
            else:
                config = TrainingConfig.load(args.config)
            logging.info(f"Loaded config from {args.config}")
        else:
            config = TrainingConfig()
            logging.info("Using default configuration")
        
        # Override with command line args
        if args.train_data:
            config.train_data_path = args.train_data
        if args.val_data:
            config.val_data_path = args.val_data
        if args.experiment_name:
            config.experiment_name = args.experiment_name
        if args.resume:
            config.resume_from = args.resume
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, config.log_level.upper()),
            format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Log system info
        logging.info("Initializing training components...")
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logging.info(f"GPU: {gpu_name}, Memory: {gpu_memory:.1f}GB")
        else:
            logging.info("Using CPU for training")
        
        # Initialize tokenizer
        tokenizer_config = TokenizerConfig(
            max_sequence_length=config.seq_length
        )
        tokenizer = ConversationTokenizer(tokenizer_config)
        
        # Update config with tokenizer vocab size
        config.vocab_size = tokenizer.vocab_size
        
        # Setup datasets
        train_dataloader, val_dataloader = setup_datasets(config, tokenizer)
        
        # Initialize trainer
        trainer = Trainer(config, tokenizer)
        
        # Resume from checkpoint if specified
        if config.resume_from:
            trainer.checkpoint_manager.load_checkpoint(
                config.resume_from, trainer.model, trainer.optimizer, trainer.scheduler
            )
        
        # Save config for reproducibility
        config_path = Path(config.checkpoint_dir) / "config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config.save(str(config_path))
        
        logging.info("Training components initialized successfully")
        
        # Start training
        trainer.train(train_dataloader, val_dataloader)
        
    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Training failed: {e}")
        logging.error(traceback.format_exc())
        sys.exit(1)
    finally:
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logging.info("Training session ended")


if __name__ == "__main__":
    main()