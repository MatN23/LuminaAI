# Enhanced Configuration System with Modern Architecture
# Copyright (c) 2025 Matias Nielsen. All rights reserved.

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import math
import random
from collections import defaultdict

# Core ML imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts

import numpy as np
from tqdm import tqdm

# Import our advanced components
from model_manager import (
    ModelConfig, TrainingConfig, PrecisionConfig, DataConfig, HardwareConfig,
    ModelMetadata, ModelManager, ConfigPresets, auto_select_config
)
from subword_transformer import SubwordTokenizer, ModernSubwordTransformer

# Optional imports
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from torch.cuda.amp import GradScaler, autocast
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False

try:
    import deepspeed
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False

class TransformerDataset(Dataset):
    """Enhanced dataset for transformer training with better text processing"""
    
    def __init__(self, texts: List[str], tokenizer: SubwordTokenizer, 
                 max_length: int = 2048, data_config: Optional[DataConfig] = None):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data_config = data_config or DataConfig()
        
        # Filter texts by length if specified
        if self.data_config.min_text_length or self.data_config.max_text_length:
            self.texts = self._filter_texts(texts)
            
        logging.info(f"📊 Dataset created with {len(self.texts):,} texts")
        
    def _filter_texts(self, texts: List[str]) -> List[str]:
        """Filter texts by length criteria"""
        filtered = []
        for text in texts:
            if self.data_config.min_text_length and len(text) < self.data_config.min_text_length:
                continue
            if self.data_config.max_text_length and len(text) > self.data_config.max_text_length:
                text = text[:self.data_config.max_text_length]
            filtered.append(text)
        
        removed = len(texts) - len(filtered)
        if removed > 0:
            logging.info(f"   Filtered out {removed:,} texts based on length criteria")
            
        return filtered
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Handle conversation format if enabled
        if self.data_config.use_conversation_format and isinstance(text, str):
            # Check if it's already formatted or needs formatting
            if not any(token in text for token in [
                self.data_config.user_token, 
                self.data_config.assistant_token,
                self.data_config.system_token
            ]):
                # Assume it's a simple text that should be treated as assistant response
                text = f"{self.data_config.assistant_token}{text}{self.data_config.end_token}"
        
        # Tokenize with proper parameters
        tokens = self.tokenizer.encode(
            text, 
            add_special_tokens=True, 
            max_length=self.max_length
        )
        
        # Ensure we have enough tokens for language modeling
        if len(tokens) < 2:
            # Fallback for very short texts
            tokens = [
                self.tokenizer.vocab.get("<|bos|>", 2),
                self.tokenizer.vocab.get("<|unk|>", 1),
                self.tokenizer.vocab.get("<|eos|>", 3)
            ]
        
        # Pad if necessary
        if len(tokens) < self.max_length:
            pad_token = self.tokenizer.vocab.get("<|pad|>", 0)
            tokens.extend([pad_token] * (self.max_length - len(tokens)))
        
        # Create input and target (shifted for language modeling)
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)
        
        # Create attention mask
        pad_token = self.tokenizer.vocab.get("<|pad|>", 0)
        attention_mask = (input_ids != pad_token).float()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

class AdvancedTrainer:
    """Advanced trainer using the sophisticated components from model_manager.py"""
    
    def __init__(self, model_config: ModelConfig, training_config: TrainingConfig,
                 precision_config: PrecisionConfig, data_config: DataConfig,
                 hardware_config: Optional[HardwareConfig] = None,
                 experiment_name: str = "advanced_training"):
        
        self.model_config = model_config
        self.training_config = training_config
        self.precision_config = precision_config
        self.data_config = data_config
        self.hardware_config = hardware_config or HardwareConfig()
        self.experiment_name = experiment_name
        
        # Device setup
        if self.hardware_config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.hardware_config.device)
        
        # Initialize model manager
        self.model_manager = ModelManager(self.training_config.output_dir)
        
        # Initialize model
        self.model = ModernSubwordTransformer(model_config).to(self.device)
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_loss = float('inf')
        
        # Mixed precision setup
        self.use_amp = (self.precision_config.use_mixed_precision and 
                       self.precision_config.precision_type in ["fp16", "bf16"] and 
                       AMP_AVAILABLE)
        
        if self.use_amp:
            self.scaler = GradScaler(
                init_scale=self.precision_config.initial_scale,
                growth_factor=self.precision_config.growth_factor,
                backoff_factor=self.precision_config.backoff_factor,
                growth_interval=self.precision_config.growth_interval,
                enabled=self.precision_config.use_dynamic_loss_scaling
            )
        
        # Compile model if requested
        if (self.precision_config.use_compile and 
            hasattr(torch, 'compile') and 
            self.device.type == 'cuda'):
            try:
                self.model = torch.compile(
                    self.model, 
                    mode=self.precision_config.compile_mode
                )
                logging.info("✅ Model compiled with torch.compile")
            except Exception as e:
                logging.warning(f"⚠️ Failed to compile model: {e}")
        
        # Wandb setup
        self.use_wandb = WANDB_AVAILABLE and hasattr(training_config, 'wandb_project')
        if self.use_wandb:
            wandb.init(
                project=training_config.wandb_project,
                name=experiment_name,
                config={
                    **model_config.__dict__,
                    **training_config.__dict__,
                    **precision_config.__dict__
                }
            )
        
        self.tokenizer = None  # Will be set later
        
        self._log_initialization()
    
    def _log_initialization(self):
        """Log comprehensive initialization information"""
        logging.info("🚀 AdvancedTrainer Initialized")
        logging.info("=" * 60)
        logging.info(f"📝 Experiment: {self.experiment_name}")
        logging.info(f"💻 Device: {self.device}")
        logging.info(f"🔢 Precision: {self.precision_config.precision_type}")
        logging.info(f"⚡ Mixed Precision: {self.use_amp}")
        logging.info(f"📊 Wandb: {self.use_wandb}")
        
        # Model info
        model_info = self.model.get_model_info()
        logging.info("🧠 Model Architecture:")
        config = model_info['config']
        logging.info(f"   Size: {config['hidden_size']}d × {config['num_layers']}L × {config['num_heads']}A")
        logging.info(f"   Vocabulary: {config['vocab_size']:,} tokens")
        logging.info(f"   Sequence Length: {config['seq_length']:,}")
        logging.info(f"   Parameters: {model_info['parameters']['total']:,}")
        logging.info(f"   Memory: {model_info['memory']['model_mb']:.1f}MB")
        
        # Training config
        logging.info("🏋️ Training Configuration:")
        logging.info(f"   Batch Size: {self.training_config.batch_size}")
        logging.info(f"   Gradient Accumulation: {self.training_config.gradient_accumulation_steps}")
        logging.info(f"   Learning Rate: {self.training_config.learning_rate}")
        logging.info(f"   Optimizer: {self.training_config.optimizer_type}")
        logging.info(f"   Scheduler: {self.training_config.scheduler_type}")
    
    def prepare_optimizer_and_scheduler(self, total_steps: int):
        """Prepare optimizer and scheduler with advanced options"""
        # Optimizer
        if self.training_config.optimizer_type.lower() == "adamw":
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=self.training_config.learning_rate,
                weight_decay=self.training_config.weight_decay,
                betas=(self.training_config.beta1, self.training_config.beta2),
                eps=self.training_config.eps
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.training_config.optimizer_type}")
        
        # Scheduler
        if self.training_config.warmup_steps is None:
            warmup_steps = int(total_steps * self.training_config.warmup_ratio)
        else:
            warmup_steps = self.training_config.warmup_steps
        
        if self.training_config.scheduler_type == "cosine_with_warmup":
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.training_config.learning_rate,
                total_steps=total_steps,
                pct_start=warmup_steps / total_steps,
                anneal_strategy='cos',
                div_factor=1.0 / self.training_config.min_lr_ratio,
                final_div_factor=1.0 / self.training_config.min_lr_ratio
            )
        elif self.training_config.scheduler_type == "cosine_restarts":
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=total_steps // (self.training_config.cosine_restarts + 1),
                T_mult=1,
                eta_min=self.training_config.learning_rate * self.training_config.min_lr_ratio
            )
        else:
            raise ValueError(f"Unsupported scheduler: {self.training_config.scheduler_type}")
        
        logging.info(f"📈 Optimizer & Scheduler prepared:")
        logging.info(f"   Total steps: {total_steps:,}")
        logging.info(f"   Warmup steps: {warmup_steps:,}")
    
    def train(self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None):
        """Enhanced training loop with all advanced features"""
        
        # Data loaders with advanced options
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=True,
            num_workers=self.training_config.num_workers if self.training_config.use_dataloader_workers else 0,
            pin_memory=self.training_config.pin_memory and self.device.type == 'cuda',
            prefetch_factor=self.training_config.prefetch_factor if self.training_config.use_dataloader_workers else 2,
            drop_last=True  # For stable training
        )
        
        eval_loader = None
        if eval_dataset:
            eval_loader = DataLoader(
                eval_dataset,
                batch_size=self.training_config.batch_size,
                shuffle=False,
                num_workers=self.training_config.num_workers if self.training_config.use_dataloader_workers else 0,
                pin_memory=self.training_config.pin_memory and self.device.type == 'cuda'
            )
        
        # Calculate training steps
        steps_per_epoch = len(train_loader) // self.training_config.gradient_accumulation_steps
        if self.training_config.max_steps:
            total_steps = self.training_config.max_steps
            num_epochs = math.ceil(total_steps / steps_per_epoch)
        else:
            num_epochs = self.training_config.max_epochs
            total_steps = steps_per_epoch * num_epochs
        
        # Prepare optimizer and scheduler
        self.prepare_optimizer_and_scheduler(total_steps)
        
        # Calculate evaluation and save intervals
        if self.training_config.eval_steps is None:
            eval_steps = max(1, int(steps_per_epoch * self.training_config.eval_ratio))
        else:
            eval_steps = self.training_config.eval_steps
            
        if self.training_config.save_steps is None:
            save_steps = max(1, int(steps_per_epoch * self.training_config.save_ratio))
        else:
            save_steps = self.training_config.save_steps
        
        # Training info
        logging.info("🏋️ Training Plan:")
        logging.info(f"   Epochs: {num_epochs}")
        logging.info(f"   Steps per epoch: {steps_per_epoch:,}")
        logging.info(f"   Total steps: {total_steps:,}")
        logging.info(f"   Eval every: {eval_steps:,} steps")
        logging.info(f"   Save every: {save_steps:,} steps")
        
        # Start training
        logging.info("🏁 Starting training...")
        start_time = datetime.now()
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            epoch_loss = self._train_epoch(
                train_loader, eval_loader, eval_steps, save_steps
            )
            
            logging.info(f"📊 Epoch {epoch + 1}/{num_epochs} completed. Average loss: {epoch_loss:.6f}")
            
            # Check if we've reached max steps
            if self.training_config.max_steps and self.global_step >= self.training_config.max_steps:
                logging.info(f"🎯 Reached maximum steps ({self.training_config.max_steps:,})")
                break
        
        # Training completed
        total_time = (datetime.now() - start_time).total_seconds()
        
        # Final save
        self._save_checkpoint(force_save=True, is_final=True, total_training_time=total_time)
        
        logging.info("✅ Training completed!")
        logging.info(f"   Total time: {total_time:.1f}s ({total_time/3600:.2f}h)")
        logging.info(f"   Final step: {self.global_step:,}")
        logging.info(f"   Best loss: {self.best_loss:.6f}")
    
    def _train_epoch(self, train_loader: DataLoader, eval_loader: Optional[DataLoader],
                     eval_steps: int, save_steps: int) -> float:
        """Train one epoch with advanced features"""
        
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(
            train_loader, 
            desc=f"Epoch {self.current_epoch + 1}",
            leave=False
        )
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast(dtype=getattr(torch, self.precision_config.precision_type)):
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch.get('attention_mask'),
                        return_dict=True
                    )
                    logits = outputs['logits']
                    
                    # Calculate loss
                    loss_fct = nn.CrossEntropyLoss(ignore_index=self.tokenizer.vocab.get("<|pad|>", 0))
                    loss = loss_fct(
                        logits.view(-1, logits.size(-1)), 
                        batch['labels'].view(-1)
                    )
                    loss = loss / self.training_config.gradient_accumulation_steps
                
                # Backward pass
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch.get('attention_mask'),
                    return_dict=True
                )
                logits = outputs['logits']
                
                # Calculate loss
                loss_fct = nn.CrossEntropyLoss(ignore_index=self.tokenizer.vocab.get("<|pad|>", 0))
                loss = loss_fct(
                    logits.view(-1, logits.size(-1)), 
                    batch['labels'].view(-1)
                )
                loss = loss / self.training_config.gradient_accumulation_steps
                
                # Backward pass
                loss.backward()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Gradient accumulation step
            if (step + 1) % self.training_config.gradient_accumulation_steps == 0:
                if self.use_amp:
                    # Unscale gradients and clip
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.training_config.max_grad_norm
                    )
                    
                    # Optimizer step
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.training_config.max_grad_norm
                    )
                    
                    # Optimizer step
                    self.optimizer.step()
                
                # Scheduler step
                self.scheduler.step()
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                self.global_step += 1
                
                # Logging
                if self.global_step % self.training_config.logging_steps == 0:
                    self._log_training_step(loss.item() * self.training_config.gradient_accumulation_steps)
                
                # Evaluation
                if eval_loader and self.global_step % eval_steps == 0:
                    eval_loss = self._evaluate(eval_loader)
                    self._log_evaluation(eval_loss)
                    
                    # Update best loss
                    if eval_loss < self.best_loss:
                        self.best_loss = eval_loss
                
                # Save checkpoint
                if self.global_step % save_steps == 0:
                    self._save_checkpoint()
                
                # Check max steps
                if self.training_config.max_steps and self.global_step >= self.training_config.max_steps:
                    break
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}',
                'step': self.global_step
            })
        
        return total_loss / max(num_batches, 1)
    
    def _evaluate(self, eval_loader: DataLoader) -> float:
        """Evaluate model with advanced features"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            eval_bar = tqdm(eval_loader, desc="Evaluating", leave=False)
            
            for batch in eval_bar:
                batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
                
                if self.use_amp:
                    with autocast(dtype=getattr(torch, self.precision_config.precision_type)):
                        outputs = self.model(
                            input_ids=batch['input_ids'],
                            attention_mask=batch.get('attention_mask'),
                            return_dict=True
                        )
                        logits = outputs['logits']
                        
                        loss_fct = nn.CrossEntropyLoss(ignore_index=self.tokenizer.vocab.get("<|pad|>", 0))
                        loss = loss_fct(
                            logits.view(-1, logits.size(-1)), 
                            batch['labels'].view(-1)
                        )
                else:
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch.get('attention_mask'),
                        return_dict=True
                    )
                    logits = outputs['logits']
                    
                    loss_fct = nn.CrossEntropyLoss(ignore_index=self.tokenizer.vocab.get("<|pad|>", 0))
                    loss = loss_fct(
                        logits.view(-1, logits.size(-1)), 
                        batch['labels'].view(-1)
                    )
                
                total_loss += loss.item()
                num_batches += 1
                
                eval_bar.set_postfix({'eval_loss': f'{loss.item():.4f}'})
        
        self.model.train()
        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss
    
    def _log_training_step(self, loss: float):
        """Log training step information"""
        lr = self.scheduler.get_last_lr()[0]
        
        logging.info(
            f"Step {self.global_step:6,} | "
            f"Loss: {loss:.6f} | "
            f"LR: {lr:.2e} | "
            f"Epoch: {self.current_epoch + 1}"
        )
        
        if self.use_wandb:
            wandb.log({
                "train/loss": loss,
                "train/learning_rate": lr,
                "train/epoch": self.current_epoch,
                "train/step": self.global_step
            })
    
    def _log_evaluation(self, eval_loss: float):
        """Log evaluation results"""
        perplexity = math.exp(min(eval_loss, 20))  # Clip for numerical stability
        
        logging.info(f"📊 Evaluation | Loss: {eval_loss:.6f} | Perplexity: {perplexity:.2f}")
        
        if self.use_wandb:
            wandb.log({
                "eval/loss": eval_loss,
                "eval/perplexity": perplexity,
                "eval/step": self.global_step
            })
    
    def _save_checkpoint(self, force_save: bool = False, is_final: bool = False,
                        total_training_time: float = 0.0):
        """Save model checkpoint using ModelManager"""
        
        # Create comprehensive metadata
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
                "epochs_completed": self.current_epoch
            },
            hardware_used=str(self.device),
            notes=f"Training completed at step {self.global_step}",
            tags=["advanced_training", self.precision_config.precision_type]
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
            if is_final:
                logging.info(f"🏁 Final model saved with ID: {model_id}")

def load_data(file_path: str, data_config: DataConfig) -> List[str]:
    """Enhanced data loading with better processing"""
    texts = []
    seen = set() if data_config.remove_duplicates else None
    
    logging.info(f"📂 Loading data from: {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Training data file not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        processed_lines = 0
        valid_texts = 0
        skipped_short = 0
        skipped_long = 0
        skipped_duplicates = 0
        
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                
                # Extract text based on common field names
                text = None
                for field in ['text', 'content', 'message', 'body', 'output']:
                    if field in data and data[field]:
                        text = data[field]
                        break
                
                # Handle conversation format
                if not text and 'messages' in data:
                    # Convert messages to conversation format
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
                
                if text and isinstance(text, str):
                    text = text.strip()
                    
                    # Apply filters
                    if len(text) < data_config.min_text_length:
                        skipped_short += 1
                        continue
                        
                    if data_config.max_text_length and len(text) > data_config.max_text_length:
                        text = text[:data_config.max_text_length]
                        skipped_long += 1
                    
                    # Lowercase if requested
                    if data_config.lowercase:
                        text = text.lower()
                    
                    # Remove duplicates
                    if data_config.remove_duplicates:
                        text_hash = hash(text)
                        if text_hash in seen:
                            skipped_duplicates += 1
                            continue
                        seen.add(text_hash)
                    
                    texts.append(text)
                    valid_texts += 1
                    
                    # Apply training size limit
                    if (data_config.max_samples_train and 
                        len(texts) >= data_config.max_samples_train):
                        logging.info(f"   Reached max training samples limit: {data_config.max_samples_train:,}")
                        break
                
                processed_lines += 1
                
                if processed_lines % 10000 == 0:
                    logging.info(f"   Processed {processed_lines:,} lines, {valid_texts:,} valid texts")
                    
            except json.JSONDecodeError:
                continue
            except Exception as e:
                logging.debug(f"Error processing line {line_num}: {e}")
                continue
    
    # Log statistics
    logging.info(f"📊 Data loading completed:")
    logging.info(f"   Lines processed: {processed_lines:,}")
    logging.info(f"   Valid texts: {len(texts):,}")
    if skipped_short > 0:
        logging.info(f"   Skipped (too short): {skipped_short:,}")
    if skipped_long > 0:
        logging.info(f"   Truncated (too long): {skipped_long:,}")
    if skipped_duplicates > 0:
        logging.info(f"   Skipped (duplicates): {skipped_duplicates:,}")
    
    return texts

def setup_logging(log_dir: str = "logs") -> str:
    """Enhanced logging setup"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"advanced_training_{timestamp}.log")
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Root logger
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
        
        # Additional CUDA info
        logging.info(f"   CUDA Version: {torch.version.cuda}")
        logging.info(f"   cuDNN Version: {torch.backends.cudnn.version()}")
        logging.info(f"   cuDNN Enabled: {torch.backends.cudnn.enabled}")
        
        # Memory info for current device
        torch.cuda.empty_cache()
        memory_allocated = torch.cuda.memory_allocated() / 1e9
        memory_reserved = torch.cuda.memory_reserved() / 1e9
        logging.info(f"   Current Memory: {memory_allocated:.1f}GB allocated, {memory_reserved:.1f}GB reserved")
        
    else:
        logging.info("💻 Using CPU")
        logging.info(f"   CPU Count: {torch.get_num_threads()}")

def detect_optimal_config() -> tuple:
    """Detect optimal configuration based on available hardware"""
    logging.info("🔍 Detecting optimal configuration...")
    
    try:
        configs = auto_select_config()
        model_config, training_config, precision_config, data_config = configs
        
        logging.info("✅ Auto-detected configuration:")
        logging.info(f"   Model size: {model_config.hidden_size}d × {model_config.num_layers}L")
        logging.info(f"   Vocabulary: {model_config.vocab_size:,}")
        logging.info(f"   Batch size: {training_config.batch_size}")
        logging.info(f"   Precision: {precision_config.precision_type}")
        
        return configs
        
    except Exception as e:
        logging.warning(f"⚠️ Auto-detection failed: {e}")
        logging.info("🔄 Falling back to conservative configuration...")
        
        # Safe fallback
        model_config = ModelConfig(
            vocab_size=16000,
            hidden_size=1024,
            num_layers=12,
            num_heads=8,
            seq_length=2048
        )
        
        training_config = TrainingConfig(
            batch_size=2,
            gradient_accumulation_steps=8,
            learning_rate=5e-4,
            max_epochs=3
        )
        
        precision_config = PrecisionConfig(precision_type="fp32", use_mixed_precision=False)
        data_config = DataConfig()
        
        return model_config, training_config, precision_config, data_config

def main():
    """Enhanced main training function"""
    parser = argparse.ArgumentParser(description="Advanced ModernSubwordTransformer Training")
    parser.add_argument("--data", required=True, help="Path to training data file (JSONL)")
    parser.add_argument("--config", default="auto", help="Configuration preset or JSON file")
    parser.add_argument("--output", default="models", help="Output directory for models")
    parser.add_argument("--experiment", default="auto", help="Experiment name")
    parser.add_argument("--eval-split", type=float, default=0.1, help="Evaluation split ratio")
    parser.add_argument("--max-samples", type=int, help="Maximum training samples")
    parser.add_argument("--vocab-size", type=int, help="Override vocabulary size")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--learning-rate", type=float, help="Override learning rate")
    parser.add_argument("--epochs", type=int, help="Override number of epochs")
    parser.add_argument("--precision", choices=["auto", "fp32", "fp16", "bf16", "tf32"], 
                       default="auto", help="Training precision")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    parser.add_argument("--wandb-project", help="Wandb project name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Setup logging
    log_file = setup_logging("logs")
    
    # Set random seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Header
    logging.info("🚀 Enhanced ModernSubwordTransformer Training")
    logging.info("=" * 80)
    logging.info(f"📝 Log file: {log_file}")
    logging.info(f"🌱 Random seed: {args.seed}")
    
    # Environment check
    logging.info("🔍 Environment Check:")
    logging.info(f"   Python: {sys.version}")
    logging.info(f"   PyTorch: {torch.__version__}")
    logging.info(f"   CUDA Available: {torch.cuda.is_available()}")
    logging.info(f"   AMP Available: {AMP_AVAILABLE}")
    logging.info(f"   DeepSpeed Available: {DEEPSPEED_AVAILABLE}")
    logging.info(f"   Wandb Available: {WANDB_AVAILABLE}")
    
    get_device_info()
    
    try:
        # Load configuration
        if args.config == "auto":
            model_config, training_config, precision_config, data_config = detect_optimal_config()
        elif args.config == "tiny":
            model_config, training_config, precision_config, data_config = ConfigPresets.tiny_debug()
        elif args.config == "research":
            model_config, training_config, precision_config, data_config = ConfigPresets.research_7b()
        elif os.path.isfile(args.config):
            # Load from JSON file
            with open(args.config, 'r') as f:
                config_data = json.load(f)
            
            model_config = ModelConfig(**config_data.get('model', {}))
            training_config = TrainingConfig(**config_data.get('training', {}))
            precision_config = PrecisionConfig(**config_data.get('precision', {}))
            data_config = DataConfig(**config_data.get('data', {}))
        else:
            raise ValueError(f"Unknown config: {args.config}")
        
        # Apply command line overrides
        if args.vocab_size:
            model_config.vocab_size = args.vocab_size
        if args.batch_size:
            training_config.batch_size = args.batch_size
        if args.learning_rate:
            training_config.learning_rate = args.learning_rate
        if args.epochs:
            training_config.max_epochs = args.epochs
        if args.max_samples:
            data_config.max_samples_train = args.max_samples
        if args.precision != "auto":
            precision_config.precision_type = args.precision
        if args.compile:
            precision_config.use_compile = True
        if args.wandb_project:
            training_config.wandb_project = args.wandb_project
        
        # Set output directory
        training_config.output_dir = args.output
        
        # Generate experiment name
        if args.experiment == "auto":
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"modern_transformer_{timestamp}"
        else:
            experiment_name = args.experiment
        
        # Debug mode adjustments
        if args.debug:
            logging.info("🐛 Debug mode enabled - using minimal configuration")
            model_config = ModelConfig(
                vocab_size=1000,
                hidden_size=256,
                num_layers=4,
                num_heads=4,
                seq_length=512
            )
            training_config.batch_size = 2
            training_config.gradient_accumulation_steps = 2
            training_config.max_epochs = 2
            training_config.logging_steps = 10
            training_config.eval_steps = 50
            training_config.save_steps = 100
            data_config.max_samples_train = 1000
            precision_config.use_mixed_precision = False
            precision_config.use_compile = False
        
        # Log final configuration
        logging.info("⚙️ Final Configuration:")
        logging.info(f"   Experiment: {experiment_name}")
        logging.info(f"   Data file: {args.data}")
        logging.info(f"   Output directory: {training_config.output_dir}")
        logging.info(f"   Model: {model_config.hidden_size}d × {model_config.num_layers}L × {model_config.num_heads}A")
        logging.info(f"   Vocabulary: {model_config.vocab_size:,}")
        logging.info(f"   Sequence length: {model_config.seq_length}")
        logging.info(f"   Batch size: {training_config.batch_size}")
        logging.info(f"   Gradient accumulation: {training_config.gradient_accumulation_steps}")
        logging.info(f"   Learning rate: {training_config.learning_rate}")
        logging.info(f"   Precision: {precision_config.precision_type}")
        logging.info(f"   Mixed precision: {precision_config.use_mixed_precision}")
        
        # Load and prepare data
        logging.info("📦 Loading and preparing data...")
        texts = load_data(args.data, data_config)
        
        if len(texts) == 0:
            raise ValueError("No valid texts found in the data file!")
        
        # Split data
        split_idx = int((1 - args.eval_split) * len(texts))
        train_texts = texts[:split_idx]
        eval_texts = texts[split_idx:] if args.eval_split > 0 else []
        
        logging.info("📊 Data split:")
        logging.info(f"   Training: {len(train_texts):,} texts")
        logging.info(f"   Evaluation: {len(eval_texts):,} texts")
        
        # Train tokenizer
        logging.info("🔤 Training tokenizer...")
        tokenizer = SubwordTokenizer()
        
        # Use a subset for tokenizer training if data is very large
        tokenizer_texts = train_texts
        if len(train_texts) > data_config.tokenizer_train_size:
            tokenizer_texts = random.sample(train_texts, data_config.tokenizer_train_size)
            logging.info(f"   Using {len(tokenizer_texts):,} texts for tokenizer training")
        
        def tokenizer_progress(message):
            logging.info(f"   {message}")
        
        tokenizer.train_from_text(
            '\n'.join(tokenizer_texts), 
            vocab_size=model_config.vocab_size,
            min_freq=data_config.min_frequency,
            progress_callback=tokenizer_progress
        )
        
        # Update model config with actual vocab size
        actual_vocab_size = tokenizer.vocab_size()
        model_config.vocab_size = actual_vocab_size
        
        logging.info(f"✅ Tokenizer trained with {actual_vocab_size:,} tokens")
        
        # Initialize trainer
        logging.info("🏋️ Initializing trainer...")
        trainer = AdvancedTrainer(
            model_config=model_config,
            training_config=training_config,
            precision_config=precision_config,
            data_config=data_config,
            experiment_name=experiment_name
        )
        
        # Set tokenizer reference
        trainer.tokenizer = tokenizer
        
        # Create datasets
        logging.info("🔄 Creating datasets...")
        train_dataset = TransformerDataset(
            train_texts, tokenizer, model_config.seq_length - 1, data_config
        )
        
        eval_dataset = None
        if eval_texts:
            eval_dataset = TransformerDataset(
                eval_texts, tokenizer, model_config.seq_length - 1, data_config
            )
        
        logging.info(f"✅ Datasets created:")
        logging.info(f"   Training samples: {len(train_dataset):,}")
        if eval_dataset:
            logging.info(f"   Evaluation samples: {len(eval_dataset):,}")
        
        # Memory estimation
        model_info = trainer.model.get_model_info()
        estimated_memory = model_info['memory']['estimated_training_mb']
        
        logging.info("💾 Memory estimation:")
        logging.info(f"   Model: {model_info['memory']['model_mb']:.1f}MB")
        logging.info(f"   Training (estimated): {estimated_memory:.1f}MB")
        
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**2)
            memory_utilization = (estimated_memory / total_memory) * 100
            logging.info(f"   GPU utilization: {memory_utilization:.1f}%")
            
            if memory_utilization > 90:
                logging.warning("⚠️ High memory utilization expected - consider reducing batch size")
        
        # Start training
        logging.info("🏁 Starting training...")
        trainer.train(train_dataset, eval_dataset)
        
        # Print final model summary
        trainer.model_manager.print_model_summary()
        
        logging.info("🎉 Training completed successfully!")
        
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

if __name__ == "__main__":
    main()