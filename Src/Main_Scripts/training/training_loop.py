# training_loop.py - Fixed version without circular imports
# Copyright (c) 2025 Matias Nielsen. All rights reserved.

import math
import time
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import asdict

import torch
from core.dataset import create_dataloader


class EnhancedTrainingMixin:
    """Training methods to be mixed into trainer classes."""
    
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
                if hasattr(self, 'should_stop') and self.should_stop:
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
                    if self.config.early_stopping_patience:
                        self._check_early_stopping(eval_metrics['eval_loss'])
                
                # Log epoch metrics
                self.logger.log_metrics(epoch_metrics, epoch, "epoch")
                
                # Save epoch checkpoint
                if hasattr(self, 'checkpoint_manager'):
                    self.checkpoint_manager.save_checkpoint(
                        self.model, self.optimizer, self.scheduler,
                        self.global_step, epoch + 1, self.metrics,
                        f"epoch_{epoch + 1:03d}"
                    )
                
                self.current_epoch = epoch + 1
                
                # Backup checkpoint periodically
                current_time = time.time()
                if (current_time - getattr(self, 'last_backup_time', 0)) > self.config.backup_every_n_hours * 3600:
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
            if hasattr(self, 'checkpoint_manager'):
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
            if hasattr(self, 'should_stop') and self.should_stop:
                break
            
            step_start_time = time.time()
            
            # Training step
            step_metrics = self.train_step(batch)
            
            # Accumulate metrics
            accumulation_metrics['loss'] += step_metrics['loss']
            accumulation_metrics['raw_loss'] += step_metrics.get('raw_loss', step_metrics['loss'])
            accumulation_metrics['tokens'] += step_metrics.get('valid_tokens', 0)
            
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
                if hasattr(self, 'health_monitor'):
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
                if self.global_step % self.config.health_check_interval == 0:
                    self.logger.log_system_stats(self.global_step)
                    self._log_memory_usage(f"Step {self.global_step}")
                
                # Periodic evaluation
                if (self.config.eval_every_n_batches > 0 and 
                    self.global_step % self.config.eval_every_n_batches == 0):
                    self._periodic_evaluation()
                
                # Periodic checkpointing
                if (self.config.save_every_n_batches > 0 and 
                    self.global_step % self.config.save_every_n_batches == 0):
                    if hasattr(self, 'checkpoint_manager'):
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
        health_status = "HEALTHY"
        if hasattr(self, 'health_monitor'):
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
            if self.config.early_stopping_patience:
                self._check_early_stopping(eval_metrics['eval_loss'])

    def _check_early_stopping(self, eval_loss: float):
        """Check early stopping condition."""
        if eval_loss < self.best_eval_loss:
            self.best_eval_loss = eval_loss
            self.patience_counter = 0
            # Save best model
            if hasattr(self, 'checkpoint_manager'):
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
            f"Warmup ratio: {getattr(self.config, 'warmup_ratio', 0.02)}",
            f"Max grad norm: {self.config.max_grad_norm}",
            f"Precision: {getattr(self.config, 'precision', 'fp32')}",
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
        backup_path = backup_dir / f"backup_{timestamp}.pt"
        
        try:
            if hasattr(self, 'checkpoint_manager'):
                self.checkpoint_manager.save_checkpoint(
                    self.model, self.optimizer, self.scheduler,
                    self.global_step, self.current_epoch, self.metrics,
                    str(backup_path)
                )
            logging.info(f"Backup created: {backup_path}")
        except Exception as e:
            logging.error(f"Failed to create backup: {e}")

    def _save_training_summary(self, total_time: float):
        """Save comprehensive training summary."""
        import numpy as np
        
        summary = {
            'experiment_name': self.config.experiment_name,
            'total_training_time_hours': total_time / 3600,
            'total_epochs': getattr(self, 'current_epoch', 0),
            'total_steps': self.global_step,
            'final_metrics': {
                'best_eval_loss': getattr(self, 'best_eval_loss', float('inf')),
                'final_train_loss': self.metrics['train_losses'][-1] if self.metrics['train_losses'] else None,
                'avg_throughput': np.mean(self.metrics['throughput']) if self.metrics['throughput'] else 0
            },
            'model_config': asdict(self.config),
            'health_summary': self.health_monitor.get_summary() if hasattr(self, 'health_monitor') else {}
        }
        
        summary_path = Path(f"experiments/{self.config.experiment_name}/training_summary.json")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(summary_path, 'w') as f:
            import json
            json.dump(summary, f, indent=2)
        
        logging.info(f"Training summary saved: {summary_path}")

    def train_step(self, batch):
        """Training step - should be implemented by the trainer class."""
        raise NotImplementedError("train_step must be implemented by the trainer class")
    
    def optimizer_step(self):
        """Optimizer step - should be implemented by the trainer class."""
        raise NotImplementedError("optimizer_step must be implemented by the trainer class")
    
    def evaluate(self, eval_dataset, max_batches=100):
        """Evaluate - should be implemented by the trainer class."""
        raise NotImplementedError("evaluate must be implemented by the trainer class")
    
    def _setup_scheduler(self, total_steps):
        """Setup scheduler - should be implemented by the trainer class."""
        pass
    
    def _log_memory_usage(self, context):
        """Log memory usage - should be implemented by the trainer class."""
        pass


def create_enhanced_trainer_class(base_trainer_class):
    """Create an enhanced trainer class by mixing in the training methods."""
    
    class EnhancedTrainer(base_trainer_class, EnhancedTrainingMixin):
        """Enhanced trainer with production-ready training loop."""
        
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            
            # Initialize training state if not already present
            if not hasattr(self, 'current_epoch'):
                self.current_epoch = 0
            if not hasattr(self, 'best_eval_loss'):
                self.best_eval_loss = float('inf')
            if not hasattr(self, 'patience_counter'):
                self.patience_counter = 0
            if not hasattr(self, 'should_stop'):
                self.should_stop = False
            if not hasattr(self, 'metrics'):
                self.metrics = {
                    'train_losses': [],
                    'eval_losses': [],
                    'learning_rates': [],
                    'gradient_norms': [],
                    'throughput': [],
                    'epoch_times': []
                }
    
    return EnhancedTrainer