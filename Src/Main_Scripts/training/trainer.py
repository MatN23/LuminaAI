# Copyright (c) 2025 Matias Nielsen. All rights reserved.
# Licensed under the Custom License below.

import math
import time
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, Optional, Any
from pathlib import Path
from datetime import datetime
from dataclasses import asdict
import numpy as np

from core.dataset import create_dataloader
from monitoring.logger import TrainingHealthMonitor
from training.checkpoint import CheckpointManager


class EnhancedConversationTrainer:
    """Production trainer with comprehensive monitoring and fault tolerance."""
    
    def __init__(self, model, tokenizer, config, logger):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.logger = logger
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # GPU setup and memory management
        self._setup_gpu()
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = None
        
        # Mixed precision setup
        self.use_amp = config.precision in ["fp16", "bf16"]
        self.dtype = torch.bfloat16 if config.precision == "bf16" else torch.float16
        self.scaler = GradScaler() if config.precision == "fp16" else None
        
        # Model compilation
        self._compile_model()
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_eval_loss = float('inf')
        self.patience_counter = 0
        self.should_stop = False
        self.last_backup_time = time.time()
        
        # Metrics and monitoring
        self.metrics = {
            'train_losses': [],
            'eval_losses': [],
            'learning_rates': [],
            'gradient_norms': [],
            'throughput': [],
            'epoch_times': []
        }
        
        # Health monitoring
        self.health_monitor = TrainingHealthMonitor()
        
        # Checkpoint management
        self.checkpoint_manager = CheckpointManager(config)
        
        logging.info(f"Trainer initialized on {self.device}")
        self._log_memory_usage("Initial")
    
    def _setup_gpu(self):
        """Setup GPU with optimal configuration."""
        if torch.cuda.is_available():
            # Clear cache
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Set memory fraction
            torch.cuda.set_per_process_memory_fraction(0.85)
            
            # Log GPU info
            gpu_props = torch.cuda.get_device_properties(0)
            logging.info(f"GPU: {gpu_props.name}, Memory: {gpu_props.total_memory / 1e9:.1f}GB")
        else:
            logging.warning("CUDA not available, using CPU")
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with parameter grouping."""
        # Separate parameters for weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if any(nd in name for nd in ['bias', 'norm', 'embed']):
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
        
        param_groups = [
            {'params': decay_params, 'weight_decay': self.config.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        
        try:
            # Use fused optimizer if available
            return AdamW(
                param_groups,
                lr=self.config.learning_rate,
                betas=(0.9, 0.95),
                eps=1e-8,
                fused=torch.cuda.is_available()
            )
        except Exception:
            # Fallback to standard AdamW
            return AdamW(
                param_groups,
                lr=self.config.learning_rate,
                betas=(0.9, 0.95),
                eps=1e-8
            )
    
    def _compile_model(self):
        """Compile model with error handling."""
        if self.config.compile and hasattr(torch, 'compile'):
            try:
                logging.info("Compiling model...")
                self.model = torch.compile(self.model, mode='default')
                logging.info("Model compiled successfully")
            except Exception as e:
                logging.warning(f"Model compilation failed: {e}")
    
    def _setup_scheduler(self, total_steps: int):
        """Setup learning rate scheduler."""
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        
        if self.config.lr_scheduler == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer, T_max=total_steps, eta_min=self.config.min_lr
            )
        elif self.config.lr_scheduler == "onecycle":
            self.scheduler = OneCycleLR(
                self.optimizer, max_lr=self.config.learning_rate,
                total_steps=total_steps, pct_start=self.config.warmup_ratio
            )
        else:  # linear
            from torch.optim.lr_scheduler import LinearLR
            self.scheduler = LinearLR(
                self.optimizer, start_factor=0.1, total_iters=warmup_steps
            )
    
    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor, 
                    loss_weights: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute weighted loss with detailed metrics."""
        # Flatten tensors
        flat_logits = logits.view(-1, logits.size(-1))
        flat_labels = labels.view(-1)
        flat_weights = loss_weights.view(-1)
        
        # Compute base loss
        loss = F.cross_entropy(flat_logits, flat_labels, reduction='none')
        
        # Apply weights and mask padding
        mask = (flat_labels != 0).float()
        weighted_loss = loss * flat_weights * mask
        
        # Check for numerical issues
        if torch.isnan(weighted_loss).any() or torch.isinf(weighted_loss).any():
            logging.warning("NaN or Inf detected in loss computation")
            return {
                'loss': torch.tensor(0.0, device=loss.device, requires_grad=True),
                'raw_loss': torch.tensor(0.0, device=loss.device),
                'perplexity': torch.tensor(float('inf'), device=loss.device),
                'valid_tokens': torch.tensor(0.0, device=loss.device)
            }
        
        # Compute final loss
        total_loss = weighted_loss.sum()
        total_weight = mask.sum().clamp(min=1)
        final_loss = total_loss / total_weight
        
        # Compute additional metrics
        raw_loss = (loss * mask).sum() / mask.sum().clamp(min=1)
        perplexity = torch.exp(raw_loss.clamp(max=10))  # Clamp to prevent overflow
        
        return {
            'loss': final_loss,
            'raw_loss': raw_loss,
            'perplexity': perplexity,
            'valid_tokens': mask.sum()
        }
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Enhanced training step with comprehensive monitoring."""
        self.model.train()
        
        # Move batch to device
        batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
        
        # Skip empty batches
        if batch['input_ids'].numel() == 0:
            return {'loss': 0.0, 'perplexity': float('inf'), 'valid_tokens': 0}
        
        # Forward pass with autocast
        with autocast(device_type='cuda', enabled=self.use_amp, dtype=self.dtype):
            logits = self.model(batch['input_ids'], batch['attention_mask'])
            loss_dict = self.compute_loss(logits, batch['labels'], batch['loss_weights'])
            loss = loss_dict['loss'] / self.config.gradient_accumulation_steps
        
        # Check for valid loss
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            logging.warning("Invalid loss detected, skipping batch")
            return {'loss': 0.0, 'perplexity': float('inf'), 'valid_tokens': 0}
        
        # Backward pass
        if self.use_amp and self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Return metrics
        return {
            'loss': loss.item() * self.config.gradient_accumulation_steps,
            'raw_loss': loss_dict['raw_loss'].item(),
            'perplexity': loss_dict['perplexity'].item(),
            'valid_tokens': loss_dict['valid_tokens'].item()
        }
    
    def optimizer_step(self) -> Dict[str, float]:
        """Enhanced optimizer step with gradient monitoring."""
        # Unscale gradients for AMP
        if self.use_amp and self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
        
        # Compute gradient norm before clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.max_grad_norm
        )
        
        # Check for NaN gradients
        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
            logging.warning("NaN/Inf gradients detected, skipping step")
            self.optimizer.zero_grad(set_to_none=True)
            if self.use_amp and self.scaler is not None:
                self.scaler.update()
            return {'grad_norm': 0.0, 'lr': 0.0}
        
        # Optimizer step
        if self.use_amp and self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        # Clear gradients
        self.optimizer.zero_grad(set_to_none=True)
        
        # Update scheduler
        if self.scheduler:
            self.scheduler.step()
        
        # Get current learning rate
        current_lr = self.scheduler.get_last_lr()[0] if self.scheduler else self.config.learning_rate
        
        return {'grad_norm': grad_norm.item(), 'lr': current_lr}
    
    @torch.no_grad()
    def evaluate(self, eval_dataset, max_batches: int = 100) -> Dict[str, float]:
        """Comprehensive evaluation with multiple metrics."""
        self.model.eval()
        
        eval_dataloader = create_dataloader(eval_dataset, self.config, shuffle=False)
        
        total_loss = 0.0
        total_raw_loss = 0.0
        total_tokens = 0
        num_batches = 0
        
        eval_start_time = time.time()
        
        for batch_idx, batch in enumerate(eval_dataloader):
            if batch_idx >= max_batches:
                break
            
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
            
            if batch['input_ids'].numel() == 0:
                continue
            
            with autocast(device_type='cuda', enabled=self.use_amp, dtype=self.dtype):
                logits = self.model(batch['input_ids'], batch['attention_mask'])
                loss_dict = self.compute_loss(logits, batch['labels'], batch['loss_weights'])
            
            if not (torch.isnan(loss_dict['loss']).any() or torch.isinf(loss_dict['loss']).any()):
                total_loss += loss_dict['loss'].item()
                total_raw_loss += loss_dict['raw_loss'].item()
                total_tokens += loss_dict['valid_tokens'].item()
                num_batches += 1
        
        eval_time = time.time() - eval_start_time
        
        if num_batches == 0:
            return {
                'eval_loss': float('inf'),
                'eval_perplexity': float('inf'),
                'eval_time': eval_time,
                'eval_throughput': 0.0
            }
        
        avg_loss = total_loss / num_batches
        avg_raw_loss = total_raw_loss / num_batches
        perplexity = math.exp(min(avg_raw_loss, 10))  # Clamp to prevent overflow
        throughput = total_tokens / eval_time if eval_time > 0 else 0
        
        return {
            'eval_loss': avg_loss,
            'eval_perplexity': perplexity,
            'eval_time': eval_time,
            'eval_throughput': throughput
        }
    
    def train(self, train_dataset, eval_dataset=None):
        """Main training loop with comprehensive monitoring."""
        logging.info("="*80)
        logging.info("STARTING PRODUCTION TRAINING")
        logging.info("="*80)
        
        # Store eval dataset for periodic evaluation
        self.eval_dataset = eval_dataset
        
        # Setup data loaders
        train_dataloader = create_dataloader(train_dataset, self.config, shuffle=True)
        
        # Calculate total steps and setup scheduler
        total_steps = len(train_dataloader) * self.config.num_epochs // self.config.gradient_accumulation_steps
        self._setup_scheduler(total_steps)
        
        # Log training configuration
        self._log_training_config(len(train_dataloader), total_steps)
        
        training_start_time = time.time()
        
        try:
            for epoch in range(self.current_epoch, self.config.num_epochs):
                if self.should_stop:
                    break
                
                logging.info(f"\n{'='*60}")
                logging.info(f"EPOCH {epoch + 1}/{self.config.num_epochs}")
                logging.info(f"{'='*60}")
                
                # Train epoch
                epoch_metrics = self.train_epoch(train_dataloader, epoch)
                
                # Full evaluation at epoch end
                if eval_dataset is not None:
                    eval_metrics = self.evaluate(eval_dataset)
                    epoch_metrics.update(eval_metrics)
                    
                    # Log epoch summary
                    logging.info(f"Epoch {epoch + 1} Summary:")
                    logging.info(f"  Train Loss: {epoch_metrics['avg_loss']:.6f}")
                    logging.info(f"  Eval Loss: {eval_metrics['eval_loss']:.6f}")
                    logging.info(f"  Eval Perplexity: {eval_metrics['eval_perplexity']:.2f}")
                    
                    # Early stopping check
                    if hasattr(self.config, 'early_stopping_patience') and self.config.early_stopping_patience:
                        self._check_early_stopping(eval_metrics['eval_loss'])
                
                # Log epoch metrics
                self.logger.log_metrics(epoch_metrics, epoch, "epoch")
                
                # Save epoch checkpoint
                self.checkpoint_manager.save_checkpoint(
                    self.model, self.optimizer, self.scheduler,
                    self.global_step, epoch + 1, self.metrics,
                    f"epoch_{epoch + 1:03d}"
                )
                
                self.current_epoch = epoch + 1
                
                # Backup checkpoint periodically
                current_time = time.time()
                backup_interval = getattr(self.config, 'backup_every_n_hours', 6) * 3600
                if (current_time - self.last_backup_time) > backup_interval:
                    self._create_backup()
                    self.last_backup_time = current_time
        
        except KeyboardInterrupt:
            logging.info("Training interrupted by user")
        except Exception as e:
            logging.error(f"Training error: {e}")
            raise
        finally:
            total_training_time = time.time() - training_start_time
            logging.info(f"\nTraining finished after {total_training_time / 3600:.2f} hours")
            
            # Save final checkpoint
            self.checkpoint_manager.save_checkpoint(
                self.model, self.optimizer, self.scheduler,
                self.global_step, self.current_epoch, self.metrics,
                "final"
            )
            
            # Save training summary
            self._save_training_summary(total_training_time)
    
    def train_epoch(self, train_dataloader, epoch: int):
        """Train one epoch with comprehensive monitoring."""
        self.model.train()
        
        epoch_metrics = {
            'total_loss': 0.0,
            'total_raw_loss': 0.0,
            'total_tokens': 0,
            'num_batches': 0,
            'grad_norm_sum': 0.0
        }
        
        accumulation_metrics = {
            'loss': 0.0,
            'raw_loss': 0.0,
            'tokens': 0
        }
        
        epoch_start_time = time.time()
        last_log_time = time.time()
        
        for batch_idx, batch in enumerate(train_dataloader):
            # Check for stop signal
            if self.should_stop:
                break
            
            step_start_time = time.time()
            
            # Training step
            step_metrics = self.train_step(batch)
            
            # Accumulate metrics
            accumulation_metrics['loss'] += step_metrics['loss']
            accumulation_metrics['raw_loss'] += step_metrics['raw_loss']
            accumulation_metrics['tokens'] += step_metrics['valid_tokens']
            
            # Optimizer step after accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                opt_metrics = self.optimizer_step()
                self.global_step += 1
                
                # Update epoch metrics
                if accumulation_metrics['loss'] > 0:
                    epoch_metrics['total_loss'] += accumulation_metrics['loss']
                    epoch_metrics['total_raw_loss'] += accumulation_metrics['raw_loss']
                    epoch_metrics['total_tokens'] += accumulation_metrics['tokens']
                    epoch_metrics['num_batches'] += 1
                    epoch_metrics['grad_norm_sum'] += opt_metrics['grad_norm']
                
                # Log metrics
                step_time = time.time() - step_start_time
                tokens_per_sec = accumulation_metrics['tokens'] / step_time if step_time > 0 else 0
                
                self.metrics['train_losses'].append(accumulation_metrics['loss'])
                self.metrics['learning_rates'].append(opt_metrics['lr'])
                self.metrics['gradient_norms'].append(opt_metrics['grad_norm'])
                self.metrics['throughput'].append(tokens_per_sec)
                
                # Health monitoring
                self.health_monitor.update(accumulation_metrics['loss'], opt_metrics['grad_norm'])
                
                # Periodic logging
                current_time = time.time()
                if self.global_step % 50 == 0 or current_time - last_log_time > 30:
                    self._log_training_step(
                        epoch, batch_idx, len(train_dataloader),
                        accumulation_metrics, opt_metrics, tokens_per_sec
                    )
                    last_log_time = current_time
                
                # Log to monitoring backends
                if self.global_step % 10 == 0:
                    self.logger.log_metrics({
                        'train_loss': accumulation_metrics['loss'],
                        'learning_rate': opt_metrics['lr'],
                        'gradient_norm': opt_metrics['grad_norm'],
                        'throughput_tokens_per_sec': tokens_per_sec,
                        'perplexity': math.exp(min(accumulation_metrics['raw_loss'], 10))
                    }, self.global_step, "train")
                
                # System monitoring
                health_check_interval = getattr(self.config, 'health_check_interval', 100)
                if self.global_step % health_check_interval == 0:
                    self.logger.log_system_stats(self.global_step)
                    self._log_memory_usage(f"Step {self.global_step}")
                
                # Periodic evaluation
                eval_every_n_batches = getattr(self.config, 'eval_every_n_batches', 0)
                if (eval_every_n_batches > 0 and 
                    self.global_step % eval_every_n_batches == 0):
                    self._periodic_evaluation()
                
                # Periodic checkpointing
                save_every_n_batches = getattr(self.config, 'save_every_n_batches', 0)
                if (save_every_n_batches > 0 and 
                    self.global_step % save_every_n_batches == 0):
                    self.checkpoint_manager.save_checkpoint(
                        self.model, self.optimizer, self.scheduler,
                        self.global_step, self.current_epoch, self.metrics,
                        f"step_{self.global_step:06d}"
                    )
                
                # Reset accumulation metrics
                accumulation_metrics = {'loss': 0.0, 'raw_loss': 0.0, 'tokens': 0}
        
        # Handle remaining gradients
        if (batch_idx + 1) % self.config.gradient_accumulation_steps != 0:
            opt_metrics = self.optimizer_step()
            self.global_step += 1
        
        # Compute epoch statistics
        epoch_time = time.time() - epoch_start_time
        self.metrics['epoch_times'].append(epoch_time)
        
        if epoch_metrics['num_batches'] > 0:
            avg_loss = epoch_metrics['total_loss'] / epoch_metrics['num_batches']
            avg_raw_loss = epoch_metrics['total_raw_loss'] / epoch_metrics['num_batches']
            avg_grad_norm = epoch_metrics['grad_norm_sum'] / epoch_metrics['num_batches']
            avg_tokens_per_sec = epoch_metrics['total_tokens'] / epoch_time
        else:
            avg_loss = float('inf')
            avg_raw_loss = float('inf')
            avg_grad_norm = 0.0
            avg_tokens_per_sec = 0.0
        
        logging.info(f"Epoch {epoch+1} completed in {epoch_time:.2f}s | "
                    f"Avg Loss: {avg_loss:.6f} | "
                    f"Avg Grad Norm: {avg_grad_norm:.4f} | "
                    f"Throughput: {avg_tokens_per_sec:.0f} tokens/s")
        
        return {
            'avg_loss': avg_loss,
            'avg_raw_loss': avg_raw_loss,
            'avg_grad_norm': avg_grad_norm,
            'epoch_time': epoch_time,
            'throughput': avg_tokens_per_sec
        }
    
    def _log_training_step(self, epoch: int, batch_idx: int, total_batches: int,
                          metrics, opt_metrics, tokens_per_sec: float):
        """Log training step with comprehensive information."""
        # Memory info
        memory_info = ""
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1e9
            memory_cached = torch.cuda.memory_reserved() / 1e9
            memory_info = f" | GPU: {memory_allocated:.1f}GB/{memory_cached:.1f}GB"
        
        # Health status
        health_status = self.health_monitor.get_status()
        health_info = f" | Health: {health_status}"
        
        logging.info(
            f"Epoch {epoch+1} | Step {self.global_step:6d} | "
            f"Batch {batch_idx+1:4d}/{total_batches} | "
            f"Loss: {metrics['loss']:.6f} | "
            f"PPL: {math.exp(min(metrics['raw_loss'], 10)):.2f} | "
            f"LR: {opt_metrics['lr']:.2e} | "
            f"GradNorm: {opt_metrics['grad_norm']:.4f} | "
            f"Tokens/s: {tokens_per_sec:.0f}"
            f"{memory_info}{health_info}"
        )
    
    def _periodic_evaluation(self):
        """Perform periodic evaluation during training."""
        if hasattr(self, 'eval_dataset') and self.eval_dataset is not None:
            eval_metrics = self.evaluate(self.eval_dataset, max_batches=50)
            
            # Log evaluation metrics
            self.logger.log_metrics(eval_metrics, self.global_step, "eval")
            
            logging.info(f"Eval | Step {self.global_step} | "
                        f"Loss: {eval_metrics['eval_loss']:.6f} | "
                        f"PPL: {eval_metrics['eval_perplexity']:.2f}")
            
            # Early stopping check
            if hasattr(self.config, 'early_stopping_patience') and self.config.early_stopping_patience:
                self._check_early_stopping(eval_metrics['eval_loss'])
    
    def _check_early_stopping(self, eval_loss: float):
        """Check early stopping condition."""
        if eval_loss < self.best_eval_loss:
            self.best_eval_loss = eval_loss
            self.patience_counter = 0
            # Save best model
            self.checkpoint_manager.save_checkpoint(
                self.model, self.optimizer, self.scheduler,
                self.global_step, self.current_epoch, self.metrics,
                "best_model"
            )
        else:
            self.patience_counter += 1
            
        if self.patience_counter >= self.config.early_stopping_patience:
            logging.info(f"Early stopping triggered after {self.patience_counter} steps without improvement")
            self.should_stop = True
    
    def _log_training_config(self, batches_per_epoch: int, total_steps: int):
        """Log comprehensive training configuration."""
        config_info = [
            f"Model Parameters: {self.model.get_num_params():,}",
            f"Epochs: {self.config.num_epochs}",
            f"Batches per epoch: {batches_per_epoch:,}",
            f"Total steps: {total_steps:,}",
            f"Effective batch size: {self.config.effective_batch_size}",
            f"Learning rate: {self.config.learning_rate:.2e}",
            f"Weight decay: {self.config.weight_decay}",
            f"Warmup ratio: {self.config.warmup_ratio}",
            f"Max grad norm: {self.config.max_grad_norm}",
            f"Precision: {self.config.precision}",
            f"Device: {self.device}"
        ]
        
        logging.info("Training Configuration:")
        for info in config_info:
            logging.info(f"  {info}")
    
    def _create_backup(self):
        """Create backup of current training state."""
        backup_dir = Path("backups") / self.config.experiment_name
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            self.checkpoint_manager.save_checkpoint(
                self.model, self.optimizer, self.scheduler,
                self.global_step, self.current_epoch, self.metrics,
                f"backup_{timestamp}"
            )
            logging.info(f"Backup created at step {self.global_step}")
        except Exception as e:
            logging.error(f"Failed to create backup: {e}")
    
    def _save_training_summary(self, total_time: float):
        """Save comprehensive training summary."""
        summary = {
            'experiment_name': self.config.experiment_name,
            'total_training_time_hours': total_time / 3600,
            'total_epochs': self.current_epoch,
            'total_steps': self.global_step,
            'final_metrics': {
                'best_eval_loss': self.best_eval_loss,
                'final_train_loss': self.metrics['train_losses'][-1] if self.metrics['train_losses'] else None,
                'avg_throughput': np.mean(self.metrics['throughput']) if self.metrics['throughput'] else 0
            },
            'model_config': asdict(self.config),
            'health_summary': self.health_monitor.get_summary()
        }
        
        summary_path = Path(f"experiments/{self.config.experiment_name}/training_summary.json")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(summary_path, 'w') as f:
            import json
            json.dump(summary, f, indent=2)
        
        logging.info(f"Training summary saved: {summary_path}")
    
    def _log_memory_usage(self, context: str):
        """Log memory usage information."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            max_allocated = torch.cuda.max_memory_allocated() / 1e9
            logging.info(f"{context} - GPU Memory: {allocated:.2f}GB allocated, "
                        f"{reserved:.2f}GB reserved, {max_allocated:.2f}GB max")
        
        # System memory
        try:
            import psutil
            memory = psutil.virtual_memory()
            logging.info(f"{context} - System Memory: {memory.percent:.1f}% used, "
                        f"{memory.available / 1e9:.1f}GB available")
        except ImportError:
            pass
    
    def save_checkpoint(self, epoch_or_step: int, emergency: bool = False, final: bool = False):
        """Save checkpoint (delegated to checkpoint manager)."""
        suffix = "emergency" if emergency else ("final" if final else f"manual_{epoch_or_step}")
        return self.checkpoint_manager.save_checkpoint(
            self.model, self.optimizer, self.scheduler,
            self.global_step, self.current_epoch, self.metrics,
            suffix
        )
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load checkpoint and return current epoch."""
        return self.checkpoint_manager.load_checkpoint(
            checkpoint_path, self.model, self.optimizer, self.scheduler
        )
    
    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: Optional[int] = None) -> str:
        """Generate response with enhanced error handling."""
        self.model.eval()
        
        if max_new_tokens is None:
            max_new_tokens = getattr(self.config, 'max_new_tokens', 512)
        
        try:
            # Create conversation format
            conversation = {
                'messages': [{'role': 'user', 'content': prompt}]
            }
            
            # Encode input
            input_tokens = self.tokenizer.encode_conversation(conversation)
            
            # Add assistant start tokens
            input_tokens.extend([
                self.tokenizer.special_tokens["<|im_start|>"],
                self.tokenizer.special_tokens["<|assistant|>"]
            ])
            
            # Ensure reasonable context length
            if len(input_tokens) >= self.config.seq_length:
                input_tokens = input_tokens[-(self.config.seq_length//2):]
            
            input_ids = torch.tensor([input_tokens], device=self.device, dtype=torch.long)
            
            # Generation loop with safety checks
            generated_tokens = []
            
            for step in range(max_new_tokens):
                # Check sequence length
                if input_ids.size(1) >= self.config.seq_length:
                    input_ids = input_ids[:, -self.config.seq_length//2:]
                
                # Forward pass
                with autocast(device_type='cuda', enabled=self.use_amp, dtype=self.dtype):
                    logits = self.model(input_ids)
                
                # Get next token logits
                temperature = getattr(self.config, 'temperature', 0.7)
                next_token_logits = logits[0, -1, :] / temperature
                
                # Apply top-k filtering
                top_k = getattr(self.config, 'top_k', 50)
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits.scatter_(0, top_k_indices, top_k_logits)
                
                # Apply top-p filtering
                top_p = getattr(self.config, 'top_p', 0.9)
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                # Check for stop tokens
                if next_token.item() == self.tokenizer.special_tokens["<|im_end|>"]:
                    break
                
                generated_tokens.append(next_token.item())
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
                
                # Safety check for infinite loops
                if step > 0 and step % 100 == 0:
                    logging.debug(f"Generation step {step}/{max_new_tokens}")
            
            # Decode response
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            return response.strip()
            
        except Exception as e:
            logging.error(f"Generation failed: {e}")
            return "I apologize, but I encountered an error while generating a response."