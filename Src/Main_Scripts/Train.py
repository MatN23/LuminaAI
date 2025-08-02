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
try:
    from model_manager import ModelManager, ModelConfig, TrainingConfig, ModelMetadata
    from word_transformer import WordTransformer, WordTokenizer
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please ensure model_manager.py and word_transformer.py are in the same directory")
    exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@contextmanager
def memory_cleanup():
    """Context manager for automatic memory cleanup."""
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()
            torch.mps.synchronize()
        gc.collect()

def get_memory_usage():
    """Get current memory usage for monitoring."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        return f"CUDA: {allocated:.2f}GB allocated, {cached:.2f}GB cached"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        allocated = torch.mps.current_allocated_memory() / 1024**3
        return f"MPS: {allocated:.2f}GB allocated"
    else:
        return "CPU mode"

def setup_device():
    """Setup the best available device with conservative memory settings."""
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        logger.info("Using device: MPS (Apple Silicon)")
        # Conservative memory settings for MPS
        torch.mps.set_per_process_memory_fraction(0.7)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using device: CUDA ({torch.cuda.get_device_name()})")
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"CUDA Memory: {total_memory:.1f} GB")
        # Conservative memory settings
        torch.cuda.set_per_process_memory_fraction(0.8)
        # Disable some optimizations that use more memory
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    else:
        device = torch.device("cpu")
        logger.info("Using device: CPU")
        torch.set_num_threads(min(4, os.cpu_count() // 2 or 2))
        torch.set_num_interop_threads(2)
    
    return device

device = setup_device()

class OptimizedWordDataset(Dataset):
    """Memory-efficient word-level dataset with smaller sequences."""
    
    def __init__(self, texts: List[str], tokenizer: WordTokenizer, seq_length: int, 
                 overlap_ratio: float = 0.7, min_seq_length: int = 16):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.min_seq_length = min_seq_length
        self.sequences = []
        
        logger.info("Creating memory-optimized word-level dataset...")
        
        # Pre-tokenize with memory-efficient chunking
        all_tokens = []
        valid_texts = 0
        chunk_size = 1000  # Process texts in smaller chunks
        
        for chunk_start in range(0, len(texts), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(texts))
            chunk_texts = texts[chunk_start:chunk_end]
            
            logger.info(f"Processing texts {chunk_start+1}-{chunk_end}/{len(texts)}")
            
            for text in chunk_texts:
                if not text or not text.strip():
                    continue
                    
                tokens = tokenizer.encode(text.strip())
                if len(tokens) >= self.min_seq_length:
                    all_tokens.extend(tokens + [tokenizer.vocab.get("</s>", 0)])
                    valid_texts += 1
            
            # Periodic memory cleanup
            if chunk_start % (chunk_size * 5) == 0:
                gc.collect()
        
        if not all_tokens:
            raise ValueError("No valid tokenized sequences found!")
        
        logger.info(f"Processed {valid_texts}/{len(texts)} valid texts")
        logger.info(f"Total tokens: {len(all_tokens):,}")
        
        # Create overlapping sequences with memory-efficient approach
        step_size = max(1, int(seq_length * overlap_ratio))
        
        # Limit total sequences to prevent OOM
        max_sequences = min(100000, (len(all_tokens) - seq_length) // step_size + 1)
        
        for i in range(0, len(all_tokens) - seq_length, step_size):
            if len(self.sequences) >= max_sequences:
                break
            if i + seq_length + 1 <= len(all_tokens):
                self.sequences.append(all_tokens[i:i + seq_length + 1])
        
        if not self.sequences:
            raise ValueError(f"No sequences created! Check seq_length ({seq_length}) vs available tokens ({len(all_tokens)})")
        
        logger.info(f"Created {len(self.sequences):,} training sequences (limited to {max_sequences:,})")
        logger.info(f"Memory usage: ~{len(self.sequences) * seq_length * 4 / 1024**2:.2f} MB")
        
        # Clear the large token list to free memory
        del all_tokens
        gc.collect()
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        return (
            torch.tensor(seq[:-1], dtype=torch.long),
            torch.tensor(seq[1:], dtype=torch.long)
        )

def load_and_process_data(data_path: str, max_samples: Optional[int] = None) -> List[str]:
    """Enhanced data loading for OASST2 dataset with better memory management."""
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    logger.info(f"Loading OASST2 training data from: {data_path}")
    
    # OASST2 specific role tokens
    role_tokens = {
        "prompter": "<user>",
        "assistant": "<assistant>"
    }
    
    texts = []
    processed_count = 0
    skipped_count = 0
    stats = {
        "deleted": 0,
        "non_english": 0,
        "empty_text": 0,
        "short_text": 0,
        "review_failed": 0,
        "not_ready": 0,
        "by_role": {"prompter": 0, "assistant": 0, "unknown": 0}
    }
    
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
                    
                    # Skip deleted entries
                    if record.get("deleted", False):
                        stats["deleted"] += 1
                        skipped_count += 1
                        continue
                    
                    # Filter for English language
                    if record.get("lang") != "en":
                        stats["non_english"] += 1
                        skipped_count += 1
                        continue
                    
                    # Skip entries that failed review (OASST2 specific)
                    if not record.get("review_result", True):
                        stats["review_failed"] += 1
                        skipped_count += 1
                        continue
                    
                    # Only use entries from ready conversation trees (OASST2 specific)
                    if record.get("tree_state") != "ready_for_export":
                        stats["not_ready"] += 1
                        skipped_count += 1
                        continue
                    
                    # Extract text content
                    text = record.get("text", "").strip()
                    if not text:
                        stats["empty_text"] += 1
                        skipped_count += 1
                        continue
                    
                    # Filter very short texts AND very long texts to save memory
                    word_count = len(text.split())
                    if word_count < 3:
                        stats["short_text"] += 1
                        skipped_count += 1
                        continue
                    
                    # Skip extremely long texts to save memory
                    if word_count > 500:
                        text = ' '.join(text.split()[:500])  # Truncate
                    
                    # Get role and add appropriate tokens
                    role = record.get("role", "").lower()
                    stats["by_role"][role] = stats["by_role"].get(role, 0) + 1
                    
                    # Add role-specific formatting
                    if role in role_tokens:
                        formatted_text = f"{role_tokens[role]} {text}"
                    else:
                        formatted_text = text
                        stats["by_role"]["unknown"] += 1
                    
                    texts.append(formatted_text)
                    processed_count += 1
                    
                    # Progress logging with memory cleanup
                    if processed_count % 2000 == 0:
                        logger.info(f"Processed {processed_count:,} samples... {get_memory_usage()}")
                        gc.collect()  # Periodic cleanup
                    
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    skipped_count += 1
                    if line_num % 5000 == 0:
                        logger.warning(f"Line {line_num}: Skipped malformed entry")
                    continue
    
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise
    
    if not texts:
        raise ValueError(f"No valid text data found in {data_path}")
    
    # Log comprehensive statistics
    total_chars = sum(len(text) for text in texts)
    avg_length = total_chars / len(texts)
    
    logger.info("=" * 50)
    logger.info("OASST2 Dataset Loading Statistics:")
    logger.info(f"  Successfully processed: {processed_count:,}")
    logger.info(f"  Skipped total: {skipped_count:,}")
    logger.info(f"    - Deleted: {stats['deleted']:,}")
    logger.info(f"    - Non-English: {stats['non_english']:,}")
    logger.info(f"    - Empty text: {stats['empty_text']:,}")
    logger.info(f"    - Too short: {stats['short_text']:,}")
    logger.info(f"    - Review failed: {stats['review_failed']:,}")
    logger.info(f"    - Tree not ready: {stats['not_ready']:,}")
    logger.info(f"  By role:")
    for role, count in stats["by_role"].items():
        if count > 0:
            logger.info(f"    - {role}: {count:,}")
    logger.info(f"  Total characters: {total_chars:,}")
    logger.info(f"  Average length: {avg_length:.1f} chars")
    logger.info(f"  {get_memory_usage()}")
    logger.info("=" * 50)
    
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
            'warmup_steps': self.warmup_steps,
            'total_steps': self.total_steps,
            'min_lr': self.min_lr,
            'base_lr': self.base_lr,
            'decay_type': self.decay_type,
            'plateau_patience': self.plateau_patience
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

def train_epoch(model, dataloader, criterion, optimizer, scheduler, epoch: int, 
                gradient_accumulation_steps: int = 1, max_grad_norm: float = 1.0,
                log_interval: int = 50) -> Tuple[float, float, float]:
    """Memory-optimized training loop with aggressive cleanup."""
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
            
            try:
                with memory_cleanup():
                    inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                    
                    # Forward pass - disable autocast for MPS to save memory
                    if device.type == 'cuda':
                        with torch.autocast(device_type='cuda', enabled=True, dtype=torch.float16):
                            logits = model(inputs)
                            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
                    else:
                        # Use regular precision for MPS/CPU
                        logits = model(inputs)
                        loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
                    
                    loss = loss / gradient_accumulation_steps
                    
                    # Backward pass
                    loss.backward()
                    accumulation_steps += 1
                    
                    # Clear intermediate activations
                    del logits
                    
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
                    
                    # Calculate accuracy less frequently to save memory
                    if batch_idx % 10 == 0:
                        with torch.no_grad():
                            # Re-compute logits only for accuracy calculation
                            if device.type == 'cuda':
                                with torch.autocast(device_type='cuda', enabled=True, dtype=torch.float16):
                                    logits = model(inputs)
                            else:
                                logits = model(inputs)
                            preds = logits.argmax(dim=-1)
                            correct = (preds == targets).sum().item()
                            total_correct += correct
                            del logits, preds
                    
                    total_tokens += targets.numel()
                    batch_times.append(time.time() - batch_start)
                    
                    # Progress logging with memory monitoring
                    if batch_idx % log_interval == 0 and batch_idx > 0:
                        current_loss = total_loss / total_tokens
                        accuracy = total_correct / (total_tokens // 10) * 100 if total_correct > 0 else 0
                        avg_batch_time = sum(batch_times[-log_interval:]) / len(batch_times[-log_interval:])
                        tokens_per_sec = targets.numel() / avg_batch_time
                        
                        logger.info(f"Epoch {epoch} | Batch {batch_idx}/{len(dataloader)} | "
                                   f"Loss: {current_loss:.4f} | Acc: {accuracy:.2f}% | "
                                   f"LR: {current_lr:.2e} | Speed: {tokens_per_sec:.0f} tok/s | "
                                   f"{get_memory_usage()}")
                    
                    # Aggressive memory cleanup every 25 batches
                    if batch_idx % 25 == 0:
                        with memory_cleanup():
                            pass
            
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.error(f"OOM at epoch {epoch}, batch {batch_idx}. Clearing cache and skipping batch...")
                    logger.info(f"Current memory: {get_memory_usage()}")
                    
                    # Aggressive cleanup
                    optimizer.zero_grad()
                    with memory_cleanup():
                        pass
                    
                    # Skip this batch and continue
                    continue
                else:
                    raise e
    
    except Exception as e:
        logger.error(f"Training error at epoch {epoch}: {e}")
        raise e
    
    # Calculate final metrics
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    accuracy = total_correct / (total_tokens // 10) if total_correct > 0 else 0.0
    avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0.0
    
    return avg_loss, accuracy, avg_batch_time

def generate_sample_text(model, tokenizer, prompt: str = "<user> Hello", 
                        max_length: int = 30, temperature: float = 0.8,
                        top_k: int = 50, top_p: float = 0.9) -> str:
    """Memory-efficient text generation."""
    model.eval()
    
    try:
        with torch.no_grad():
            input_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
            generated = input_ids.clone()
            
            for _ in range(max_length):
                if generated.size(1) >= min(model.config.seq_length, 256):  # Limit context length
                    break
                
                # Only use the last part of the sequence if it's getting too long
                if generated.size(1) > 128:
                    context = generated[:, -128:]
                else:
                    context = generated
                
                logits = model(context)
                next_token_logits = logits[0, -1, :] / temperature
                
                # Top-k sampling only to save memory
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
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
        with memory_cleanup():
            pass

def validate_training_setup():
    """Validate that all required files and dependencies are available."""
    required_files = [
        "oasst1_data/oasst1_train.jsonl",
        "model_manager.py",
        "word_transformer.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        logger.error("Missing required files:")
        for file_path in missing_files:
            logger.error(f"  - {file_path}")
        
        if "oasst1_data/oasst1_train.jsonl" in missing_files:
            logger.info("To download the dataset, run: python Dataset_download.py")
        
        return False
    
    return True

def main():
    """Main training function with memory-optimized configuration."""
    
    logger.info("🚀 Starting Memory-Optimized OASST1 Word-Level Transformer Training")
    logger.info("=" * 70)
    logger.info(f"Initial memory: {get_memory_usage()}")
    
    # Validate setup
    if not validate_training_setup():
        logger.error("❌ Training setup validation failed!")
        return 1
    
    # Memory-optimized configuration
    model_config = ModelConfig(
        vocab_size=32000,     # Standard tokenizer vocab
        hidden_size=1024,     # Decent size for agentic tasks
        num_layers=12,        # Good for reasoning
        num_heads=16,         # Keep this
        seq_length=4096,      # Need longer context for tool use
        dropout=0.1,
        model_type="WordTransformer",
        tokenizer_type="word"
    )
    
    # Very conservative batch sizes
    if device.type == 'cuda':
        batch_size = 8
        max_samples = 50000  # Limit dataset size
    elif device.type == 'mps':
        batch_size = 1
        max_samples = 25000
    else:
        batch_size = 1
        max_samples = 10000
    
    training_config = TrainingConfig(
        learning_rate=1e-4,    # Lower LR for stability
        weight_decay=0.01,
        batch_size=batch_size,
        gradient_accumulation_steps=8,  # Higher accumulation to simulate larger batches
        max_epochs=100,         # Fewer epochs
        warmup_ratio=0.05,
        save_every=1000,
        eval_every=500,
        max_grad_norm=1.0,
        label_smoothing=0.1,
        beta1=0.9,
        beta2=0.95
    )
    
    # Initialize model manager
    model_manager = ModelManager("models")
    
    try:
        # Load and process OASST1 training data with memory limits
        logger.info("📚 Loading OASST1 dataset (memory-limited)...")
        texts = load_and_process_data("oasst1_data/oasst1_train.jsonl", max_samples)
        
        if len(texts) == 0:
            raise ValueError("No training data loaded!")
        
        logger.info(f"Memory after data loading: {get_memory_usage()}")
        
        # Create and train tokenizer with smaller vocabulary
        logger.info("🔤 Training compact word-level tokenizer...")
        tokenizer = WordTokenizer()
        
        # Use smaller sample for tokenizer
        sample_size = min(10000, len(texts))
        sample_texts = texts[:sample_size]
        all_text = "\n".join(sample_texts)
        
        tokenizer.train_from_text(all_text, vocab_size=model_config.vocab_size)
        model_config.vocab_size = tokenizer.vocab_size()
        
        logger.info(f"✅ Tokenizer trained - Vocabulary size: {model_config.vocab_size:,}")
        logger.info(f"Memory after tokenizer: {get_memory_usage()}")
        
        # Create memory-optimized dataset
        logger.info("📦 Creating memory-optimized training dataset...")
        dataset = OptimizedWordDataset(texts, tokenizer, model_config.seq_length)
        
        logger.info(f"Memory after dataset creation: {get_memory_usage()}")
        
        # Very conservative dataloader settings
        dataloader = DataLoader(
            dataset,
            batch_size=training_config.batch_size,
            shuffle=True,
            num_workers=0,  # No multiprocessing to save memory
            pin_memory=False,  # Disable pin_memory to save memory
            drop_last=True  # Drop last incomplete batch
        )
        
        logger.info(f"📊 Dataset ready: {len(dataset):,} sequences, {len(dataloader):,} batches/epoch")
        
        # Initialize smaller model
        logger.info("🧠 Initializing memory-efficient model...")
        with memory_cleanup():
            model = WordTransformer(model_config).to(device)
        
        total_params, trainable_params = count_parameters(model)
        model_size_mb = total_params * 4 / 1024**2
        logger.info(f"Model parameters: {total_params:,} (~{model_size_mb:.1f}MB)")
        logger.info(f"Memory after model creation: {get_memory_usage()}")
        
        # Training components with memory-efficient settings
        optimizer = optim.AdamW(
            model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
            betas=(training_config.beta1, training_config.beta2),
            eps=1e-8
        )
        
        criterion = nn.CrossEntropyLoss(
            label_smoothing=training_config.label_smoothing,
            ignore_index=tokenizer.vocab.get("<pad>", 0)
        )
        
        # Enhanced scheduler
        total_steps = len(dataloader) * training_config.max_epochs // training_config.gradient_accumulation_steps
        warmup_steps = int(total_steps * training_config.warmup_ratio)
        scheduler = AdaptiveLRScheduler(optimizer, warmup_steps, total_steps, decay_type="cosine")
        
        logger.info(f"🎯 Training setup: {total_steps:,} steps, {warmup_steps:,} warmup")
        logger.info(f"Memory before training: {get_memory_usage()}")
        
        # Training loop
        logger.info("🚀 Starting memory-optimized training...")
        training_start = time.time()
        best_loss = float('inf')
        
        for epoch in range(1, training_config.max_epochs + 1):
            epoch_start = time.time()
            
            try:
                logger.info(f"Starting epoch {epoch}, memory: {get_memory_usage()}")
                
                # Train epoch with memory monitoring
                avg_loss, accuracy, avg_batch_time = train_epoch(
                    model, dataloader, criterion, optimizer, scheduler, epoch,
                    training_config.gradient_accumulation_steps,
                    training_config.max_grad_norm
                )
                
                # Calculate metrics
                perplexity = calculate_perplexity(avg_loss)
                epoch_time = time.time() - epoch_start
                tokens_per_sec = len(dataset) * model_config.seq_length / epoch_time
                
                # Comprehensive logging
                logger.info("=" * 60)
                logger.info(f"📊 Epoch {epoch}/{training_config.max_epochs} Summary:")
                logger.info(f"   Loss: {avg_loss:.4f} | Perplexity: {perplexity:.2f}")
                logger.info(f"   Accuracy: {accuracy*100:.2f}% | Time: {epoch_time:.1f}s")
                logger.info(f"   Speed: {tokens_per_sec:.0f} tokens/sec")
                logger.info(f"   {get_memory_usage()}")
                
                # Sample generation every few epochs (with memory cleanup)
                if epoch % 2 == 0:
                    with memory_cleanup():
                        sample_prompts = ["<user> What is AI?", "<assistant> I can help"]
                        for prompt in sample_prompts[:1]:  # Just one to save memory
                            sample = generate_sample_text(model, tokenizer, prompt, 20)
                            logger.info(f"   Sample: {prompt} → {sample}")
                
                # Model saving with memory cleanup
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    
                    with memory_cleanup():
                        # Create comprehensive metadata
                        metadata = ModelMetadata(
                            model_name="OASST1_WordTransformer_Compact",
                            version=f"v1.0_epoch_{epoch}",
                            created_at=datetime.now().isoformat(),
                            last_modified=datetime.now().isoformat(),
                            model_config=model_config,
                            training_config=training_config,
                            dataset_info={
                                "name": "OpenAssistant OASST1 (Memory-Optimized)",
                                "num_samples": len(texts),
                                "vocab_size": model_config.vocab_size,
                                "seq_length": model_config.seq_length,
                                "source": "oasst1_train.jsonl",
                                "preprocessing": "Word-level tokenization with role formatting and memory optimization"
                            },
                            performance_metrics={
                                "loss": avg_loss,
                                "perplexity": perplexity,
                                "accuracy": accuracy,
                                "tokens_per_second": tokens_per_sec,
                                "batch_time_ms": avg_batch_time * 1000
                            },
                            model_size_mb=model_size_mb,
                            total_parameters=total_params,
                            trainable_parameters=trainable_params,
                            training_time_hours=(time.time() - training_start) / 3600,
                            epochs_trained=epoch,
                            best_loss=best_loss,
                            best_perplexity=calculate_perplexity(best_loss),
                            hardware_used=f"{device.type.upper()}",
                            pytorch_version=torch.__version__,
                            cuda_version=torch.version.cuda if device.type == 'cuda' else None,
                            model_hash="",
                            tokenizer_hash="",
                            notes=f"Memory-optimized OASST1 word-level transformer. Trained on {len(texts):,} samples with reduced model size and aggressive memory management. Best model at epoch {epoch}.",
                            tags=["oasst1", "word-level", "transformer", "memory-optimized", "best-model", f"epoch-{epoch}"]
                        )
                        
                        model_id = model_manager.save_model(model, tokenizer, metadata, optimizer, scheduler)
                        logger.info(f"💾 Best model saved: {model_id}")
                
                logger.info("=" * 60)
                
                # Aggressive cleanup after each epoch
                with memory_cleanup():
                    pass
                
            except Exception as e:
                logger.error(f"❌ Error in epoch {epoch}: {e}")
                logger.info(f"Memory state: {get_memory_usage()}")
                
                # Try to recover from OOM
                if "out of memory" in str(e).lower():
                    logger.info("Attempting to recover from OOM...")
                    with memory_cleanup():
                        pass
                    # Skip to next epoch
                    continue
                else:
                    raise e
        
        # Training completion summary
        total_time = time.time() - training_start
        logger.info("=" * 70)
        logger.info("✅ Memory-optimized training completed successfully!")
        logger.info(f"🎯 Best loss achieved: {best_loss:.4f}")
        logger.info(f"🎯 Best perplexity: {calculate_perplexity(best_loss):.2f}")
        logger.info(f"⏱️  Total training time: {total_time/3600:.2f} hours")
        logger.info(f"🚀 Average processing speed: {len(dataset) * model_config.seq_length * training_config.max_epochs / total_time:.0f} tokens/sec")
        logger.info(f"💾 Final memory state: {get_memory_usage()}")
        logger.info("=" * 70)
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("❌ Training interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Training failed with error: {e}")
        logger.info(f"Memory state at failure: {get_memory_usage()}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # Final cleanup
        logger.info("Performing final memory cleanup...")
        with memory_cleanup():
            pass

if __name__ == "__main__":
    exit(main())