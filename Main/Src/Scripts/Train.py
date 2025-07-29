# Copyright (c) 2025 Matias Nielsen. All rights reserved.
# Licensed under the Custom License below.

import os
import json
import time
import math
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import gc
from contextlib import contextmanager

# Import shared components
from model_manager import ModelManager, ModelConfig, TrainingConfig, ModelMetadata
from word_transformer import WordTransformer, WordTokenizer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Optimized thread configuration
torch.set_num_threads(os.cpu_count() // 2 or 2)
torch.set_num_interop_threads(2)

@contextmanager
def memory_cleanup():
    """Context manager for automatic memory cleanup."""
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()

def setup_device():
    """Setup the best available device with proper error handling."""
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        logger.info("Using device: MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using device: CUDA ({torch.cuda.get_device_name()})")
        logger.info(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        # Enable optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        device = torch.device("cpu")
        logger.info("Using device: CPU")
    return device

device = setup_device()

class OptimizedWordDataset(Dataset):
    """Optimized word-level dataset with memory-efficient data loading."""
    
    def __init__(self, texts: List[str], tokenizer: WordTokenizer, seq_length: int, 
                 overlap_ratio: float = 0.5, min_seq_length: int = 32):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.min_seq_length = min_seq_length
        self.sequences = []
        
        logger.info("Creating optimized word-level dataset...")
        
        # Pre-tokenize all texts for efficiency
        all_tokens = []
        for i, text in enumerate(texts):
            if i % 1000 == 0:
                logger.info(f"Tokenizing text {i+1}/{len(texts)}")
            tokens = tokenizer.encode(text)
            if len(tokens) >= self.min_seq_length:
                all_tokens.extend(tokens + [tokenizer.vocab.get("</s>", 0)])
        
        # Create overlapping sequences with better memory utilization
        step_size = max(1, int(seq_length * overlap_ratio))
        total_sequences = 0
        
        for i in range(0, len(all_tokens) - seq_length, step_size):
            if i + seq_length + 1 <= len(all_tokens):
                self.sequences.append(all_tokens[i:i + seq_length + 1])
                total_sequences += 1
        
        logger.info(f"Created {len(self.sequences):,} training sequences from {len(all_tokens):,} tokens")
        logger.info(f"Memory usage: ~{len(self.sequences) * seq_length * 4 / 1024**2:.2f} MB")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        return (
            torch.tensor(seq[:-1], dtype=torch.long),
            torch.tensor(seq[1:], dtype=torch.long)
        )

def load_and_process_data(data_path: str, max_samples: Optional[int] = None) -> List[str]:
    """Optimized data loading with streaming and better error handling."""
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    logger.info(f"Loading training data from: {data_path}")
    
    role_tokens = {
        "prompter": "<user>",
        "assistant": "<bot>"
    }
    
    texts = []
    processed_count = 0
    skipped_count = 0
    
    try:
        if data_path.suffix == '.jsonl':
            with open(data_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if max_samples and processed_count >= max_samples:
                        break
                        
                    try:
                        line = line.strip()
                        if not line:
                            continue
                        
                        record = json.loads(line)
                        
                        # Skip deleted or non-English entries
                        if (record.get("deleted", False) or 
                            record.get("lang") != "en"):
                            skipped_count += 1
                            continue
                        
                        # Extract and validate text content
                        text = (record.get("text") or 
                               record.get("content") or 
                               (record.get("message", {}).get("text") 
                                if isinstance(record.get("message"), dict) else ""))
                        
                        text = str(text).strip()
                        if not text or len(text.split()) < 3:
                            skipped_count += 1
                            continue
                        
                        # Add role tokens efficiently
                        role = str(record.get("role", "")).lower()
                        token = role_tokens.get(role, "")
                        
                        final_text = f"{token} {text} </s>" if token else f"{text} </s>"
                        texts.append(final_text)
                        processed_count += 1
                        
                        # Progress logging
                        if processed_count % 10000 == 0:
                            logger.info(f"Processed {processed_count:,} samples...")
                        
                    except (json.JSONDecodeError, KeyError, ValueError) as e:
                        skipped_count += 1
                        if line_num % 10000 == 0:
                            logger.warning(f"Skipped {skipped_count} malformed entries so far")
                        continue
        else:
            # Optimized plain text processing
            with open(data_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    paragraphs = [p.strip() for p in content.split('\n\n') 
                                if p.strip() and len(p.split()) >= 3]
                    texts.extend([f"{p} </s>" for p in paragraphs])
                    processed_count = len(paragraphs)
    
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise
    
    if not texts:
        raise ValueError(f"No valid text data found in {data_path}")
    
    # Log statistics
    total_chars = sum(len(text) for text in texts)
    avg_length = total_chars / len(texts)
    
    logger.info(f"Dataset statistics:")
    logger.info(f"  Samples: {len(texts):,}")
    logger.info(f"  Total characters: {total_chars:,}")
    logger.info(f"  Average length: {avg_length:.1f} chars")
    logger.info(f"  Processed: {processed_count:,}")
    logger.info(f"  Skipped: {skipped_count:,}")
    
    return texts

class AdaptiveLRScheduler:
    """Improved learning rate scheduler with adaptive decay."""
    
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, 
                 min_lr: float = 1e-6, decay_type: str = "cosine"):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_step = 0
        self.decay_type = decay_type
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.plateau_patience = 1000
        
    def step(self, loss: Optional[float] = None) -> float:
        """Update learning rate with optional loss-based adaptation."""
        self.current_step += 1
        
        # Track best loss for plateau detection
        if loss is not None:
            if loss < self.best_loss:
                self.best_loss = loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
        
        # Calculate base learning rate
        if self.current_step < self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * self.current_step / self.warmup_steps
        else:
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            
            if self.decay_type == "cosine":
                lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
            elif self.decay_type == "linear":
                lr = self.base_lr * (1 - progress) + self.min_lr * progress
            else:  # exponential
                lr = self.base_lr * (0.95 ** (progress * 100))
                lr = max(lr, self.min_lr)
        
        # Apply plateau reduction if needed
        if self.patience_counter > self.plateau_patience and loss is not None:
            lr *= 0.5
            self.patience_counter = 0
            logger.info(f"Reducing LR due to plateau: {lr:.2e}")
        
        # Update optimizer
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
    
    def state_dict(self):
        return {
            'current_step': self.current_step,
            'best_loss': self.best_loss,
            'patience_counter': self.patience_counter,
            **{k: v for k, v in self.__dict__.items() if k not in ['optimizer']}
        }
    
    def load_state_dict(self, state_dict):
        for key, value in state_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)

def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Optimized parameter counting."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def calculate_perplexity(loss: float) -> float:
    """Calculate perplexity with overflow protection."""
    return math.exp(min(loss, 20))

@torch.compile(mode="reduce-overhead")  # PyTorch 2.0+ optimization
def compiled_forward(model, inputs):
    """Compiled forward pass for better performance."""
    return model(inputs)

def train_epoch(model, dataloader, criterion, optimizer, scheduler, epoch: int, 
                gradient_accumulation_steps: int = 1, max_grad_norm: float = 1.0,
                log_interval: int = 50) -> Tuple[float, float, float]:
    """Optimized training loop with better memory management and monitoring."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    accumulation_steps = 0
    batch_times = []
    
    try:
        optimizer.zero_grad()
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            batch_start = time.time()
            
            with memory_cleanup():
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                
                # Forward pass with mixed precision
                with torch.autocast(device_type=device.type, enabled=device.type in ['cuda', 'mps']):
                    if hasattr(torch, 'compile') and torch.__version__ >= "2.0":
                        logits = compiled_forward(model, inputs)
                    else:
                        logits = model(inputs)
                    
                    loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
                    loss = loss / gradient_accumulation_steps
                
                # Backward pass
                loss.backward()
                accumulation_steps += 1
                
                # Optimizer step with gradient accumulation
                if accumulation_steps >= gradient_accumulation_steps:
                    if max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                    
                    optimizer.step()
                    current_lr = scheduler.step(loss.item() * gradient_accumulation_steps)
                    optimizer.zero_grad()
                    accumulation_steps = 0
                
                # Efficient statistics calculation
                batch_loss = loss.item() * gradient_accumulation_steps
                total_loss += batch_loss * inputs.size(0)
                
                with torch.no_grad():
                    preds = logits.argmax(dim=-1)
                    correct = (preds == targets).sum().item()
                    total_correct += correct
                    total_tokens += targets.numel()
                
                batch_times.append(time.time() - batch_start)
                
                # Progress logging with performance metrics
                if batch_idx % log_interval == 0 and batch_idx > 0:
                    current_loss = total_loss / total_tokens
                    accuracy = total_correct / total_tokens * 100
                    avg_batch_time = sum(batch_times[-log_interval:]) / len(batch_times[-log_interval:])
                    tokens_per_sec = targets.numel() / avg_batch_time
                    
                    logger.info(f"Epoch {epoch} | Batch {batch_idx}/{len(dataloader)} | "
                               f"Loss: {current_loss:.4f} | Acc: {accuracy:.2f}% | "
                               f"LR: {current_lr:.2e} | Speed: {tokens_per_sec:.0f} tok/s")
                
                # Periodic memory cleanup
                if batch_idx % 100 == 0:
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    elif device.type == 'mps':
                        torch.mps.empty_cache()
    
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.error(f"OOM at epoch {epoch}, batch {batch_idx}")
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            elif device.type == 'mps':
                torch.mps.empty_cache()
            gc.collect()
            raise e
        else:
            raise e
    
    # Calculate final metrics
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0
    avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0.0
    
    return avg_loss, accuracy, avg_batch_time

def generate_sample_text(model, tokenizer, prompt: str = "<user> Hello", 
                        max_length: int = 50, temperature: float = 0.8,
                        top_k: int = 50, top_p: float = 0.9) -> str:
    """Improved text generation with better sampling."""
    model.eval()
    
    try:
        with torch.no_grad():
            input_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
            generated = input_ids.clone()
            
            for _ in range(max_length):
                if generated.size(1) >= model.config.seq_length:
                    break
                
                logits = model(generated)
                next_token_logits = logits[0, -1, :] / temperature
                
                # Top-k and top-p sampling
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
                
                # Stop on end token
                if next_token.item() == tokenizer.vocab.get("</s>", -1):
                    break
            
            response_ids = generated[0][input_ids.size(1):].tolist()
            response = tokenizer.decode(response_ids)
            return response.strip()
    
    except Exception as e:
        logger.error(f"Error generating sample: {e}")
        return "Error during generation"
    finally:
        model.train()

def main():
    """Optimized main training function."""
    
    # Enhanced configuration
    model_config = ModelConfig(
        vocab_size=32000,
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        seq_length=1024,
        dropout=0.1,
        model_type="WordTransformer",
        tokenizer_type="word"
    )
    
    training_config = TrainingConfig(
        learning_rate=2e-4,  # Slightly lower for stability
        weight_decay=0.01,
        batch_size=8 if device.type == 'cuda' else 4,  # Dynamic batch size
        gradient_accumulation_steps=4,  # Reduced for faster updates
        max_epochs=30,  # Reduced for efficiency
        warmup_ratio=0.05,  # Shorter warmup
        save_every=500,  # More frequent saves
        eval_every=250,
        max_grad_norm=1.0,
        label_smoothing=0.1,
        beta1=0.9,
        beta2=0.95
    )
    
    logger.info("🚀 Starting Optimized Word-Level Transformer Training")
    logger.info("=" * 70)
    
    # Initialize model manager
    model_manager = ModelManager("models")
    
    try:
        # Load and process training data with size limit for faster iteration
        max_samples = 100000 if device.type == 'cpu' else None
        texts = load_and_process_data("oasst1_data/oasst1_train.jsonl", max_samples)
        
        # Create and train tokenizer
        logger.info("📚 Training optimized tokenizer...")
        tokenizer = WordTokenizer()
        
        # Sample text for tokenizer training to reduce memory usage
        sample_size = min(50000, len(texts))
        sample_texts = texts[:sample_size] if len(texts) > sample_size else texts
        all_text = " ".join(sample_texts)
        
        tokenizer.train_from_text(all_text, vocab_size=model_config.vocab_size)
        model_config.vocab_size = tokenizer.vocab_size()
        
        logger.info(f"Vocabulary size: {model_config.vocab_size:,}")
        
        # Create optimized dataset
        dataset = OptimizedWordDataset(texts, tokenizer, model_config.seq_length)
        
        # Optimized dataloader
        num_workers = min(4, os.cpu_count() // 2) if device.type == 'cuda' else 0
        dataloader = DataLoader(
            dataset,
            batch_size=training_config.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=device.type == 'cuda',
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None
        )
        
        logger.info(f"Dataset: {len(dataset):,} sequences, {len(dataloader):,} batches/epoch")
        
        # Initialize model with optimizations
        model = WordTransformer(model_config).to(device)
        
        # Enable optimizations
        if hasattr(torch, 'compile') and torch.__version__ >= "2.0":
            logger.info("Enabling PyTorch 2.0 compilation...")
            model = torch.compile(model, mode="reduce-overhead")
        
        total_params, trainable_params = count_parameters(model)
        logger.info(f"Model: {total_params:,} params (~{total_params * 4 / 1024**2:.1f}MB)")
        
        # Optimized training components
        optimizer = optim.AdamW(
            model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
            betas=(training_config.beta1, training_config.beta2),
            eps=1e-8,
            fused=device.type == 'cuda'  # Fused optimizer for CUDA
        )
        
        criterion = nn.CrossEntropyLoss(
            label_smoothing=training_config.label_smoothing,
            ignore_index=tokenizer.vocab.get("<pad>", 0)
        )
        
        # Enhanced scheduler
        total_steps = len(dataloader) * training_config.max_epochs // training_config.gradient_accumulation_steps
        warmup_steps = int(total_steps * training_config.warmup_ratio)
        scheduler = AdaptiveLRScheduler(optimizer, warmup_steps, total_steps, decay_type="cosine")
        
        logger.info(f"Training: {total_steps:,} steps, {warmup_steps:,} warmup")
        
        # Training loop with enhanced monitoring
        logger.info("🎯 Starting optimized training...")
        training_start = time.time()
        best_loss = float('inf')
        
        for epoch in range(1, training_config.max_epochs + 1):
            epoch_start = time.time()
            
            # Train with enhanced monitoring
            avg_loss, accuracy, avg_batch_time = train_epoch(
                model, dataloader, criterion, optimizer, scheduler, epoch,
                training_config.gradient_accumulation_steps,
                training_config.max_grad_norm
            )
            
            # Enhanced metrics
            perplexity = calculate_perplexity(avg_loss)
            epoch_time = time.time() - epoch_start
            tokens_per_sec = len(dataset) * model_config.seq_length / epoch_time
            
            # Comprehensive logging
            logger.info(f"📊 Epoch {epoch}/{training_config.max_epochs}:")
            logger.info(f"   Loss: {avg_loss:.4f} | PPL: {perplexity:.2f} | Acc: {accuracy*100:.2f}%")
            logger.info(f"   Speed: {tokens_per_sec:.0f} tok/s | Time: {epoch_time:.1f}s")
            
            # Sample generation
            if epoch % 3 == 0:
                sample = generate_sample_text(model, tokenizer, "<user> What is AI?", 40)
                logger.info(f"   Sample: {sample}")
            
            # Enhanced model saving
            if avg_loss < best_loss:
                best_loss = avg_loss
                
                metadata = ModelMetadata(
                    model_name="OptimizedWordTransformer",
                    version=f"v2.0_epoch_{epoch}",
                    created_at=datetime.now().isoformat(),
                    last_modified=datetime.now().isoformat(),
                    model_config=model_config,
                    training_config=training_config,
                    dataset_info={
                        "num_samples": len(texts),
                        "vocab_size": model_config.vocab_size,
                        "seq_length": model_config.seq_length,
                        "source": "oasst1_train.jsonl",
                        "preprocessing": "Optimized word-level tokenization"
                    },
                    performance_metrics={
                        "loss": avg_loss,
                        "perplexity": perplexity,
                        "accuracy": accuracy,
                        "tokens_per_second": tokens_per_sec,
                        "batch_time_ms": avg_batch_time * 1000
                    },
                    model_size_mb=0,
                    total_parameters=total_params,
                    trainable_parameters=trainable_params,
                    training_time_hours=(time.time() - training_start) / 3600,
                    epochs_trained=epoch,
                    best_loss=best_loss,
                    best_perplexity=calculate_perplexity(best_loss),
                    hardware_used=f"{device.type.upper()}: {torch.cuda.get_device_name() if device.type == 'cuda' else 'CPU'}",
                    pytorch_version=torch.__version__,
                    cuda_version=torch.version.cuda if device.type == 'cuda' else None,
                    model_hash="",
                    tokenizer_hash="",
                    notes=f"Optimized training with improved performance and memory efficiency. Best at epoch {epoch}.",
                    tags=["optimized", "word-level", "transformer", "v2", f"epoch-{epoch}", "best"]
                )
                
                model_id = model_manager.save_model(model, tokenizer, metadata, optimizer, scheduler)
                logger.info(f"💾 Best model saved: {model_id}")
        
        # Training summary
        total_time = time.time() - training_start
        logger.info("=" * 70)
        logger.info("✅ Optimized training completed!")
        logger.info(f"🎯 Best loss: {best_loss:.4f} (PPL: {calculate_perplexity(best_loss):.2f})")
        logger.info(f"⏱️  Total time: {total_time/3600:.2f}h")
        logger.info(f"🚀 Average speed: {len(dataset) * model_config.seq_length * training_config.max_epochs / total_time:.0f} tok/s")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("❌ Training interrupted")
        return 1
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        with memory_cleanup():
            pass

if __name__ == "__main__":
    exit(main())