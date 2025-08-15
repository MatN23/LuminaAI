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
            try:
                # Clear cache
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                # Set memory fraction carefully
                torch.cuda.set_per_process_memory_fraction(0.85)
            except Exception as e:
                logging.warning(f"GPU setup failed: {e}")
            
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
            # Try fused optimizer if available
            return AdamW(
                param_groups,
                lr=self.config.learning_rate,
                betas=(0.9, 0.95),
                eps=1e-8,
                fused=torch.cuda.is_available()
            )
        except (TypeError, RuntimeError):
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
        
        # Move batch to device safely
        device_batch = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                device_batch[k] = v.to(self.device, non_blocking=True)
            else:
                device_batch[k] = v
        
        # Skip empty batches
        if device_batch['input_ids'].numel() == 0:
            return {'loss': 0.0, 'perplexity': float('inf'), 'valid_tokens': 0}
        
        # Forward pass with autocast
        with autocast(enabled=self.use_amp):
            logits = self.model(device_batch['input_ids'], device_batch['attention_mask'])
            loss_dict = self.compute_loss(logits, device_batch['labels'], device_batch['loss_weights'])
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
            
            # Move batch to device safely
            device_batch = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    device_batch[k] = v.to(self.device, non_blocking=True)
                else:
                    device_batch[k] = v
            
            if device_batch['input_ids'].numel() == 0:
                continue
            
            with autocast(enabled=self.use_amp):
                logits = self.model(device_batch['input_ids'], device_batch['attention_mask'])
                loss_dict = self.compute_loss(logits, device_batch['labels'], device_batch['loss_weights'])
            
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
    
    def train(self, train_dataset, eval_dataset):
        """Main training loop with comprehensive monitoring and fault tolerance."""
        logging.info("Starting training...")
        
        # Create data loaders
        train_dataloader = create_dataloader(train_dataset, self.config, shuffle=True)
        
        # Calculate total steps and setup scheduler
        total_steps = len(train_dataloader) * self.config.num_epochs // self.config.gradient_accumulation_steps
        self._setup_scheduler(total_steps)
        
        logging.info(f"Training for {self.config.num_epochs} epochs, {total_steps:,} total steps")
        logging.info(f"Effective batch size: {self.config.effective_batch_size}")
        
        # Training metrics
        step_losses = []
        step_times = []
        
        try:
            for epoch in range(self.current_epoch, self.config.num_epochs):
                self.current_epoch = epoch
                epoch_start_time = time.time()
                
                logging.info(f"\n{'='*60}")
                logging.info(f"EPOCH {epoch + 1}/{self.config.num_epochs}")
                logging.info(f"{'='*60}")
                
                # Training phase
                epoch_loss = 0.0
                num_batches = 0
                
                for batch_idx, batch in enumerate(train_dataloader):
                    step_start_time = time.time()
                    
                    # Training step
                    step_metrics = self.train_step(batch)
                    
                    # Accumulate gradients
                    if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                        # Optimizer step
                        opt_metrics = self.optimizer_step()
                        self.global_step += 1
                        
                        # Log metrics
                        current_lr = opt_metrics['lr']
                        grad_norm = opt_metrics['grad_norm']
                        
                        # Update health monitor
                        self.health_monitor.update(step_metrics['loss'], grad_norm)
                        
                        # Accumulate for logging
                        step_losses.append(step_metrics['loss'])
                        step_times.append(time.time() - step_start_time)
                        
                        # Log progress
                        if self.global_step % 10 == 0:  # Log every 10 steps for debug
                            avg_loss = sum(step_losses[-10:]) / min(len(step_losses), 10)
                            throughput = len(step_losses[-10:]) / sum(step_times[-10:]) if step_times[-10:] else 0
                            
                            logging.info(f"Step {self.global_step:6d} | "
                                       f"Loss: {avg_loss:.4f} | "
                                       f"LR: {current_lr:.2e} | "
                                       f"Grad: {grad_norm:.3f} | "
                                       f"Throughput: {throughput:.1f} steps/s")
                        
                        # Log detailed metrics
                        if self.global_step % self.config.eval_every_n_batches == 0:
                            # Evaluation
                            eval_metrics = self.evaluate(eval_dataset)
                            
                            # Log to monitoring
                            train_metrics = {
                                'train_loss': avg_loss,
                                'learning_rate': current_lr,
                                'grad_norm': grad_norm,
                                'perplexity': math.exp(min(avg_loss, 10))
                            }
                            
                            # Combine metrics
                            all_metrics = {**train_metrics, **eval_metrics}
                            self.logger.log_metrics(all_metrics, self.global_step)
                            self.logger.log_system_stats(self.global_step)
                            
                            # Check if this is the best model
                            is_best = eval_metrics['eval_loss'] < self.best_eval_loss
                            if is_best:
                                self.best_eval_loss = eval_metrics['eval_loss']
                                self.patience_counter = 0
                            else:
                                self.patience_counter += 1
                            
                            # Save checkpoint
                            if self.global_step % self.config.save_every_n_batches == 0:
                                self.save_checkpoint(self.global_step, is_best=is_best)
                            
                            logging.info(f"Eval Loss: {eval_metrics['eval_loss']:.6f} | "
                                       f"Eval PPL: {eval_metrics['eval_perplexity']:.2f} | "
                                       f"Best: {self.best_eval_loss:.6f}")
                            
                            # Early stopping check
                            if (self.config.early_stopping_patience and 
                                self.patience_counter >= self.config.early_stopping_patience):
                                logging.info(f"Early stopping triggered after {self.patience_counter} evaluations without improvement")
                                return
                            
                            # Health check
                            health_status = self.health_monitor.get_status()
                            if health_status == "CRITICAL":
                                logging.error("Training health critical - stopping")
                                self.save_checkpoint(self.global_step, emergency=True)
                                raise RuntimeError("Training became unstable")
                            elif health_status == "WARNING":
                                logging.warning("Training health warning detected")
                    
                    epoch_loss += step_metrics['loss']
                    num_batches += 1
                    
                    # Memory cleanup
                    if batch_idx % 100 == 0:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                
                # End of epoch
                epoch_time = time.time() - epoch_start_time
                avg_epoch_loss = epoch_loss / max(num_batches, 1)
                
                logging.info(f"\nEpoch {epoch + 1} completed in {epoch_time:.1f}s")
                logging.info(f"Average loss: {avg_epoch_loss:.6f}")
                
                # Save epoch checkpoint
                self.save_checkpoint(epoch + 1)
                
                # Log epoch metrics
                epoch_metrics = {
                    'epoch': epoch + 1,
                    'epoch_loss': avg_epoch_loss,
                    'epoch_time': epoch_time,
                    'learning_rate': self.scheduler.get_last_lr()[0] if self.scheduler else self.config.learning_rate
                }
                self.logger.log_metrics(epoch_metrics, self.global_step, prefix="epoch")
                
                # Memory logging
                self._log_memory_usage(f"End of Epoch {epoch + 1}")
        
        except KeyboardInterrupt:
            logging.info("Training interrupted by user")
            self.save_checkpoint(self.global_step, emergency=True)
            raise
        except Exception as e:
            logging.error(f"Training failed: {e}")
            self.save_checkpoint(self.global_step, emergency=True)
            raise
        
        # Final checkpoint
        logging.info("Training completed!")
        final_checkpoint = self.save_checkpoint(self.config.num_epochs, final=True)
        logging.info(f"Final checkpoint saved: {final_checkpoint}")
    
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
    
    def save_checkpoint(self, epoch_or_step: int, emergency: bool = False, final: bool = False, is_best: bool = False):
        """Save checkpoint (delegated to checkpoint manager)."""
        suffix = "emergency" if emergency else ("final" if final else None)
        return self.checkpoint_manager.save_checkpoint(
            self.model, self.optimizer, self.scheduler,
            self.global_step, self.current_epoch, self.metrics,
            suffix, is_best
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
            max_new_tokens = self.config.max_new_tokens
        
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
                next_token_logits = logits[0, -1, :] / self.config.temperature
                
                # Apply top-k filtering
                if self.config.top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, self.config.top_k)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits.scatter_(0, top_k_indices, top_k_logits)
                
                # Apply top-p filtering
                if self.config.top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > self.config.top_p
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