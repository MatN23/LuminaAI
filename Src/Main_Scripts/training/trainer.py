"""
Enhanced Training Module - FIXED VERSION WITH INFERENCE PRECISION
Main trainer class with comprehensive monitoring and fault tolerance.
"""

import math
import time
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from contextlib import nullcontext
from typing import Dict, Optional, Any, Union
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
        
        # Mixed precision setup - with version compatibility
        self.use_amp = getattr(config, 'precision', 'fp32') in ["fp16", "bf16"] and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp and getattr(config, 'precision', 'fp32') == "fp16" else None
        
        # Check PyTorch version for autocast compatibility
        self.torch_version = torch.__version__
        logging.info(f"PyTorch version: {self.torch_version}")
        
        # INFERENCE PRECISION SETUP
        self.inference_precision = getattr(config, 'inference_precision', 'auto')
        self._setup_inference_precision()
        
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
        try:
            self.health_monitor = TrainingHealthMonitor()
        except Exception as e:
            logging.warning(f"Failed to initialize health monitor: {e}")
            # Create a simple fallback health monitor
            class SimpleHealthMonitor:
                def update(self, loss, grad_norm): pass
                def get_status(self): return "OK"
                def get_summary(self): return {}
            self.health_monitor = SimpleHealthMonitor()
        
        # Checkpoint management - FIXED
        try:
            self.checkpoint_manager = CheckpointManager(config)
        except Exception as e:
            logging.error(f"Failed to initialize checkpoint manager: {e}")
            # Create a simple fallback checkpoint manager
            class SimpleCheckpointManager:
                def __init__(self, config):
                    self.config = config
                    self.checkpoint_dir = Path("checkpoints")
                    self.checkpoint_dir.mkdir(exist_ok=True)
                
                def save_checkpoint(self, model, optimizer, scheduler, global_step, epoch, metrics, suffix=""):
                    checkpoint_path = self.checkpoint_dir / f"checkpoint_{suffix}_{global_step}.pt"
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                        'global_step': global_step,
                        'epoch': epoch,
                        'metrics': metrics
                    }, checkpoint_path)
                    logging.info(f"Checkpoint saved: {checkpoint_path}")
                    return str(checkpoint_path)
                
                def load_checkpoint(self, path, model, optimizer=None, scheduler=None):
                    checkpoint = torch.load(path, map_location='cpu')
                    model.load_state_dict(checkpoint['model_state_dict'])
                    if optimizer and 'optimizer_state_dict' in checkpoint:
                        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    if scheduler and 'scheduler_state_dict' in checkpoint:
                        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    return checkpoint.get('epoch', 0)
                
                def get_resume_path(self):
                    checkpoints = list(self.checkpoint_dir.glob("*.pt"))
                    return str(max(checkpoints, key=lambda x: x.stat().st_mtime)) if checkpoints else None
            
            self.checkpoint_manager = SimpleCheckpointManager(config)
        
        logging.info(f"Trainer initialized on {self.device}")
        logging.info(f"Model parameters: {self._count_parameters():,}")
        logging.info(f"Training precision: {getattr(config, 'precision', 'fp32')}")
        logging.info(f"Inference precision: {self.inference_precision}")
        self._log_memory_usage("Initial")
    
    def _setup_inference_precision(self):
        """Setup inference precision based on config or auto-detect."""
        if self.inference_precision == "auto":
            if torch.cuda.is_available():
                # Check if GPU supports bfloat16
                try:
                    # Test if bfloat16 is supported
                    test_tensor = torch.tensor([1.0], dtype=torch.bfloat16, device=self.device)
                    self.inference_precision = "bf16"
                    logging.info("Auto-selected bf16 for inference")
                except:
                    # Fallback to fp16
                    try:
                        test_tensor = torch.tensor([1.0], dtype=torch.float16, device=self.device)
                        self.inference_precision = "fp16" 
                        logging.info("Auto-selected fp16 for inference")
                    except:
                        self.inference_precision = "fp32"
                        logging.info("Auto-selected fp32 for inference (GPU precision issues)")
            else:
                self.inference_precision = "fp32"
                logging.info("Auto-selected fp32 for inference (CPU)")
        
        # Validate precision setting
        valid_precisions = ["fp16", "bf16", "fp32"]
        if self.inference_precision not in valid_precisions:
            logging.warning(f"Invalid inference precision {self.inference_precision}, defaulting to fp32")
            self.inference_precision = "fp32"
    
    def set_inference_precision(self, precision: str):
        """
        Dynamically change inference precision.
        
        Args:
            precision: "fp16", "bf16", "fp32", or "auto"
        """
        old_precision = getattr(self, 'inference_precision', 'fp32')
        self.inference_precision = precision
        self._setup_inference_precision()
        logging.info(f"Inference precision changed from {old_precision} to {self.inference_precision}")
    
    def _get_autocast_context(self):
        """Get autocast context with version compatibility for TRAINING."""
        if not self.use_amp:
            return nullcontext()
        
        # For older PyTorch versions, use simpler autocast
        try:
            # Try new API first (PyTorch >= 1.10)
            return autocast(device_type='cuda', enabled=True)
        except TypeError:
            # Fallback to old API
            return autocast(enabled=True)
    
    def _get_inference_autocast_context(self):
        """Get autocast context specifically for INFERENCE with specified precision."""
        if not torch.cuda.is_available() or self.inference_precision == "fp32":
            return nullcontext()
        
        # Determine dtype for inference
        dtype_map = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16
        }
        dtype = dtype_map.get(self.inference_precision, torch.float16)
        
        try:
            # Try new API first (PyTorch >= 1.10)
            return torch.autocast(device_type='cuda', dtype=dtype)
        except TypeError:
            try:
                # Fallback for older PyTorch versions with dtype support
                return autocast(enabled=True)
            except:
                # Ultimate fallback
                return nullcontext()
    
    def _count_parameters(self):
        """Count model parameters."""
        try:
            return sum(p.numel() for p in self.model.parameters())
        except:
            return 0
    
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
            {'params': decay_params, 'weight_decay': getattr(self.config, 'weight_decay', 0.01)},
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
        if getattr(self.config, 'compile', False) and hasattr(torch, 'compile'):
            try:
                logging.info("Compiling model...")
                self.model = torch.compile(self.model, mode='default')
                logging.info("Model compiled successfully")
            except Exception as e:
                logging.warning(f"Model compilation failed: {e}")
    
    def _setup_scheduler(self, total_steps: int):
        """Setup learning rate scheduler."""
        warmup_ratio = getattr(self.config, 'warmup_ratio', 0.1)
        warmup_steps = int(total_steps * warmup_ratio)
        
        lr_scheduler = getattr(self.config, 'lr_scheduler', 'cosine')
        
        if lr_scheduler == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer, T_max=total_steps, eta_min=getattr(self.config, 'min_lr', 1e-6)
            )
        elif lr_scheduler == "onecycle":
            self.scheduler = OneCycleLR(
                self.optimizer, max_lr=self.config.learning_rate,
                total_steps=total_steps, pct_start=warmup_ratio
            )
        else:  # linear
            try:
                from torch.optim.lr_scheduler import LinearLR
                self.scheduler = LinearLR(
                    self.optimizer, start_factor=0.1, total_iters=warmup_steps
                )
            except ImportError:
                # Fallback for older PyTorch versions
                from torch.optim.lr_scheduler import StepLR
                self.scheduler = StepLR(self.optimizer, step_size=warmup_steps, gamma=0.1)
    
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
        
        # Forward pass with autocast (TRAINING precision)
        with self._get_autocast_context():
            logits = self.model(batch['input_ids'], batch['attention_mask'])
            loss_dict = self.compute_loss(logits, batch['labels'], batch['loss_weights'])
            loss = loss_dict['loss'] / getattr(self.config, 'gradient_accumulation_steps', 1)
        
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
            'loss': loss.item() * getattr(self.config, 'gradient_accumulation_steps', 1),
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
        max_grad_norm = getattr(self.config, 'max_grad_norm', 1.0)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_grad_norm
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
    def evaluate(self, eval_dataset, max_batches: int = 100, 
                 use_inference_precision: bool = True) -> Dict[str, float]:
        """Comprehensive evaluation with multiple metrics and optional inference precision."""
        self.model.eval()
        
        eval_dataloader = create_dataloader(eval_dataset, self.config, shuffle=False)
        
        total_loss = 0.0
        total_raw_loss = 0.0
        total_tokens = 0
        num_batches = 0
        
        eval_start_time = time.time()
        
        # Choose autocast context based on parameter
        autocast_context = self._get_inference_autocast_context() if use_inference_precision else self._get_autocast_context()
        
        for batch_idx, batch in enumerate(eval_dataloader):
            if batch_idx >= max_batches:
                break
            
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
            
            if batch['input_ids'].numel() == 0:
                continue
            
            with autocast_context:
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
                'eval_throughput': 0.0,
                'inference_precision': self.inference_precision if use_inference_precision else "training_precision"
            }
        
        avg_loss = total_loss / num_batches
        avg_raw_loss = total_raw_loss / num_batches
        perplexity = math.exp(min(avg_raw_loss, 10))  # Clamp to prevent overflow
        throughput = total_tokens / eval_time if eval_time > 0 else 0
        
        return {
            'eval_loss': avg_loss,
            'eval_perplexity': perplexity,
            'eval_time': eval_time,
            'eval_throughput': throughput,
            'inference_precision': self.inference_precision if use_inference_precision else "training_precision"
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
        gradient_accumulation_steps = getattr(self.config, 'gradient_accumulation_steps', 1)
        total_steps = len(train_dataloader) * self.config.num_epochs // gradient_accumulation_steps
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
                    eval_metrics = self.evaluate(eval_dataset, use_inference_precision=True)
                    epoch_metrics.update(eval_metrics)
                    
                    # Log epoch summary
                    logging.info(f"Epoch {epoch + 1} Summary:")
                    logging.info(f"  Train Loss: {epoch_metrics['avg_loss']:.6f}")
                    logging.info(f"  Eval Loss: {eval_metrics['eval_loss']:.6f}")
                    logging.info(f"  Eval Perplexity: {eval_metrics['eval_perplexity']:.2f}")
                    logging.info(f"  Inference Precision: {eval_metrics['inference_precision']}")
                    
                    # Early stopping check
                    if getattr(self.config, 'early_stopping_patience', None):
                        self._check_early_stopping(eval_metrics['eval_loss'])
                
                # Log epoch metrics
                try:
                    self.logger.log_metrics(epoch_metrics, epoch, "epoch")
                except Exception as e:
                    logging.warning(f"Failed to log epoch metrics: {e}")
                
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
        
        gradient_accumulation_steps = getattr(self.config, 'gradient_accumulation_steps', 1)
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
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
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
                    try:
                        self.logger.log_metrics({
                            'train_loss': accumulation_metrics['loss'],
                            'learning_rate': opt_metrics['lr'],
                            'gradient_norm': opt_metrics['grad_norm'],
                            'throughput_tokens_per_sec': tokens_per_sec,
                            'perplexity': math.exp(min(accumulation_metrics['raw_loss'], 10))
                        }, self.global_step, "train")
                    except Exception as e:
                        logging.debug(f"Failed to log training metrics: {e}")
                
                # System monitoring
                health_check_interval = getattr(self.config, 'health_check_interval', 100)
                if self.global_step % health_check_interval == 0:
                    try:
                        if hasattr(self.logger, 'log_system_stats'):
                            self.logger.log_system_stats(self.global_step)
                    except Exception as e:
                        logging.debug(f"Failed to log system stats: {e}")
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
        if (batch_idx + 1) % gradient_accumulation_steps != 0:
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
            eval_metrics = self.evaluate(self.eval_dataset, max_batches=50, use_inference_precision=True)
            
            # Log evaluation metrics
            try:
                self.logger.log_metrics(eval_metrics, self.global_step, "eval")
            except Exception as e:
                logging.debug(f"Failed to log eval metrics: {e}")
            
            logging.info(f"Eval | Step {self.global_step} | "
                        f"Loss: {eval_metrics['eval_loss']:.6f} | "
                        f"PPL: {eval_metrics['eval_perplexity']:.2f} | "
                        f"Precision: {eval_metrics['inference_precision']}")
            
            # Early stopping check
            if getattr(self.config, 'early_stopping_patience', None):
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
        try:
            model_params = self._count_parameters()
        except:
            model_params = "Unknown"
        
        config_info = [
            f"Model Parameters: {model_params:,}" if isinstance(model_params, int) else f"Model Parameters: {model_params}",
            f"Epochs: {self.config.num_epochs}",
            f"Batches per epoch: {batches_per_epoch:,}",
            f"Total steps: {total_steps:,}",
            f"Effective batch size: {getattr(self.config, 'effective_batch_size', self.config.batch_size)}",
            f"Learning rate: {self.config.learning_rate:.2e}",
            f"Weight decay: {getattr(self.config, 'weight_decay', 0.01)}",
            f"Warmup ratio: {getattr(self.config, 'warmup_ratio', 0.1)}",
            f"Max grad norm: {getattr(self.config, 'max_grad_norm', 1.0)}",
            f"Training precision: {getattr(self.config, 'precision', 'fp32')}",
            f"Inference precision: {self.inference_precision}",
            f"Device: {self.device}"
        ]
        
        logging.info("Training Configuration:")
        for info in config_info:
            logging.info(f"  {info}")
    
    def _create_backup(self):
        """Create backup of current training state."""
        backup_dir = Path("backups") / getattr(self.config, 'experiment_name', 'default')
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
        try:
            # Try to convert config to dict
            try:
                model_config = asdict(self.config)
            except:
                # Fallback to manual conversion
                model_config = {
                    attr: getattr(self.config, attr) 
                    for attr in dir(self.config) 
                    if not attr.startswith('_') and not callable(getattr(self.config, attr))
                }
            
            summary = {
                'experiment_name': getattr(self.config, 'experiment_name', 'unknown'),
                'total_training_time_hours': total_time / 3600,
                'total_epochs': self.current_epoch,
                'total_steps': self.global_step,
                'final_metrics': {
                    'best_eval_loss': self.best_eval_loss,
                    'final_train_loss': self.metrics['train_losses'][-1] if self.metrics['train_losses'] else None,
                    'avg_throughput': np.mean(self.metrics['throughput']) if self.metrics['throughput'] else 0
                },
                'precision_settings': {
                    'training_precision': getattr(self.config, 'precision', 'fp32'),
                    'inference_precision': self.inference_precision
                },
                'model_config': model_config,
                'health_summary': self.health_monitor.get_summary()
            }
            
            summary_path = Path(f"experiments/{getattr(self.config, 'experiment_name', 'default')}/training_summary.json")
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(summary_path, 'w') as f:
                import json
                json.dump(summary, f, indent=2, default=str)  # default=str handles non-serializable objects
            
            logging.info(f"Training summary saved: {summary_path}")
        
        except Exception as e:
            logging.error(f"Failed to save training summary: {e}")
    
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
    def generate(self, prompt: str, max_new_tokens: Optional[int] = None, 
                 inference_precision: Optional[str] = None,
                 temperature: Optional[float] = None,
                 top_k: Optional[int] = None,
                 top_p: Optional[float] = None,
                 **kwargs) -> str:
        """
        Generate response with enhanced error handling and inference precision control.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            inference_precision: Override inference precision for this generation ("fp16", "bf16", "fp32")
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Top-p (nucleus) filtering
            **kwargs: Additional generation parameters
        """
        self.model.eval()
        
        # Override inference precision if specified
        original_precision = self.inference_precision
        if inference_precision is not None:
            temp_precision = self.inference_precision
            self.inference_precision = inference_precision
            self._setup_inference_precision()
        else:
            temp_precision = None
        
        if max_new_tokens is None:
            max_new_tokens = getattr(self.config, 'max_new_tokens', 512)
        
        # Use provided or default generation parameters
        temperature = temperature if temperature is not None else getattr(self.config, 'temperature', 0.7)
        top_k = top_k if top_k is not None else getattr(self.config, 'top_k', 50)
        top_p = top_p if top_p is not None else getattr(self.config, 'top_p', 0.9)
        
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
            
            # Generation loop with safety checks and INFERENCE precision
            generated_tokens = []
            
            logging.debug(f"Starting generation with precision: {self.inference_precision}")
            
            for step in range(max_new_tokens):
                # Check sequence length
                if input_ids.size(1) >= self.config.seq_length:
                    input_ids = input_ids[:, -self.config.seq_length//2:]
                
                # Forward pass with INFERENCE precision
                with self._get_inference_autocast_context():
                    logits = self.model(input_ids)
                
                # Get next token logits
                next_token_logits = logits[0, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits.scatter_(0, top_k_indices, top_k_logits)
                
                # Apply top-p filtering
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
            
            logging.debug(f"Generation completed with {len(generated_tokens)} tokens using {self.inference_precision} precision")
            
            return response.strip()
            
        except Exception as e:
            logging.error(f"Generation failed: {e}")
            return "I apologize, but I encountered an error while generating a response."
        
        finally:
            # Restore original inference precision if it was temporarily changed
            if temp_precision is not None:
                self.inference_precision = original_precision
    
    def benchmark_inference_precision(self, test_prompts: Optional[List[str]] = None, 
                                    max_new_tokens: int = 100) -> Dict[str, Dict[str, Any]]:
        """
        Benchmark different inference precisions with timing and memory usage.
        
        Args:
            test_prompts: List of test prompts (uses default if None)
            max_new_tokens: Maximum tokens to generate per prompt
            
        Returns:
            Dictionary with benchmark results for each precision
        """
        if test_prompts is None:
            test_prompts = [
                "What is machine learning?",
                "Explain quantum computing in simple terms.",
                "Write a short story about a robot.",
            ]
        
        precisions_to_test = ["fp32", "fp16"]
        if torch.cuda.is_available():
            try:
                # Test if bf16 is supported
                test_tensor = torch.tensor([1.0], dtype=torch.bfloat16, device=self.device)
                precisions_to_test.append("bf16")
            except:
                pass
        
        results = {}
        original_precision = self.inference_precision
        
        for precision in precisions_to_test:
            logging.info(f"Benchmarking {precision} precision...")
            
            # Set precision
            self.set_inference_precision(precision)
            
            # Warm up
            try:
                _ = self.generate("Hello", max_new_tokens=5, temperature=0.1)
            except:
                logging.warning(f"Warmup failed for {precision}, skipping...")
                continue
            
            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Benchmark
            total_time = 0
            total_tokens = 0
            memory_peak = 0
            generations = []
            
            for prompt in test_prompts:
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                
                start_time = time.time()
                
                try:
                    response = self.generate(
                        prompt, 
                        max_new_tokens=max_new_tokens,
                        temperature=0.7,
                        top_k=50,
                        top_p=0.9
                    )
                    generations.append(response)
                    
                    # Count tokens (approximate)
                    token_count = len(self.tokenizer.tokenizer.encode(response))
                    total_tokens += token_count
                    
                except Exception as e:
                    logging.error(f"Generation failed for {precision}: {e}")
                    generations.append(f"ERROR: {str(e)}")
                
                end_time = time.time()
                total_time += (end_time - start_time)
                
                if torch.cuda.is_available():
                    memory_peak = max(memory_peak, torch.cuda.max_memory_allocated())
            
            # Calculate metrics
            avg_time_per_prompt = total_time / len(test_prompts) if test_prompts else 0
            tokens_per_second = total_tokens / total_time if total_time > 0 else 0
            
            results[precision] = {
                'total_time_seconds': total_time,
                'avg_time_per_prompt': avg_time_per_prompt,
                'total_tokens_generated': total_tokens,
                'tokens_per_second': tokens_per_second,
                'peak_memory_mb': memory_peak / 1024 / 1024 if memory_peak > 0 else 0,
                'generations': generations,
                'success': len([g for g in generations if not g.startswith("ERROR:")]) == len(test_prompts)
            }
            
            logging.info(f"{precision} results: {tokens_per_second:.1f} tokens/s, "
                        f"{avg_time_per_prompt:.2f}s/prompt, "
                        f"{memory_peak / 1024 / 1024:.1f}MB peak memory")
        
        # Restore original precision
        self.set_inference_precision(original_precision)
        
        return results
    
    def generate_with_multiple_precisions(self, prompt: str, 
                                        precisions: List[str] = None) -> Dict[str, str]:
        """
        Generate the same response with different precisions for comparison.
        
        Args:
            prompt: Input prompt
            precisions: List of precisions to test (default: ["fp32", "fp16"])
            
        Returns:
            Dictionary mapping precision to generated response
        """
        if precisions is None:
            precisions = ["fp32", "fp16"]
            if torch.cuda.is_available():
                try:
                    test_tensor = torch.tensor([1.0], dtype=torch.bfloat16, device=self.device)
                    precisions.append("bf16")
                except:
                    pass
        
        results = {}
        original_precision = self.inference_precision
        
        for precision in precisions:
            try:
                results[precision] = self.generate(
                    prompt,
                    inference_precision=precision,
                    temperature=0.7,  # Fixed temperature for consistency
                    top_k=50,
                    top_p=0.9
                )
                logging.debug(f"Generated with {precision}: {len(results[precision])} chars")
            except Exception as e:
                results[precision] = f"ERROR with {precision}: {str(e)}"
                logging.error(f"Generation failed with {precision}: {e}")
        
        # Restore original precision
        self.inference_precision = original_precision
        
        return results