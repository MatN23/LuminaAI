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
        torch.set_num_threads(os.cpu_count() // 2 or 2)
        torch.set_num_interop_threads(2)
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
        valid_texts = 0
        for i, text in enumerate(texts):
            if i % 1000 == 0 and i > 0:
                logger.info(f"Tokenizing text {i+1}/{len(texts)}")
            
            if not text or not text.strip():
                continue
                
            tokens = tokenizer.encode(text.strip())
            if len(tokens) >= self.min_seq_length:
                all_tokens.extend(tokens + [tokenizer.vocab.get("</s>", 0)])
                valid_texts += 1
        
        if not all_tokens:
            raise ValueError("No valid tokenized sequences found!")
        
        logger.info(f"Processed {valid_texts}/{len(texts)} valid texts")
        logger.info(f"Total tokens: {len(all_tokens):,}")
        
        # Create overlapping sequences with better memory utilization
        step_size = max(1, int(seq_length * overlap_ratio))
        
        for i in range(0, len(all_tokens) - seq_length, step_size):
            if i + seq_length + 1 <= len(all_tokens):
                self.sequences.append(all_tokens[i:i + seq_length + 1])
        
        if not self.sequences:
            raise ValueError(f"No sequences created! Check seq_length ({seq_length}) vs available tokens ({len(all_tokens)})")
        
        logger.info(f"Created {len(self.sequences):,} training sequences")
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
    """Enhanced data loading with better OASST1 dataset handling."""
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    logger.info(f"Loading training data from: {data_path}")
    
    # OASST1 specific role tokens
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
                    
                    # Extract text content
                    text = record.get("text", "").strip()
                    if not text:
                        stats["empty_text"] += 1
                        skipped_count += 1
                        continue
                    
                    # Filter very short texts
                    if len(text.split()) < 3:
                        stats["short_text"] += 1
                        skipped_count += 1
                        continue
                    
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
                    
                    # Progress logging
                    if processed_count % 5000 == 0:
                        logger.info(f"Processed {processed_count:,} samples...")
                    
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
    logger.info("Dataset Loading Statistics:")
    logger.info(f"  Successfully processed: {processed_count:,}")
    logger.info(f"  Skipped total: {skipped_count:,}")
    logger.info(f"    - Deleted: {stats['deleted']:,}")
    logger.info(f"    - Non-English: {stats['non_english']:,}")
    logger.info(f"    - Empty text: {stats['empty_text']:,}")
    logger.info(f"    - Too short: {stats['short_text']:,}")
    logger.info(f"  By role:")
    for role, count in stats["by_role"].items():
        if count > 0:
            logger.info(f"    - {role}: {count:,}")
    logger.info(f"  Total characters: {total_chars:,}")
    logger.info(f"  Average length: {avg_length:.1f} chars")
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
    """Enhanced training loop with better error handling and monitoring."""
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
                    
                    # Forward pass with automatic mixed precision
                    with torch.autocast(device_type=device.type, enabled=device.type in ['cuda', 'mps']):
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
                    # Clear cache and continue
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    elif device.type == 'mps':
                        torch.mps.empty_cache()
                    gc.collect()
                    continue  # Skip this batch
                else:
                    raise e
    
    except Exception as e:
        logger.error(f"Training error at epoch {epoch}: {e}")
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
    """Main training function with comprehensive error handling."""
    
    logger.info("🚀 Starting Enhanced OASST1 Word-Level Transformer Training")
    logger.info("=" * 70)
    
    # Validate setup
    if not validate_training_setup():
        logger.error("❌ Training setup validation failed!")
        return 1
    
    # Configuration optimized for OASST1 dataset
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
    
    # Adjust batch size based on device capabilities
    if device.type == 'cuda':
        batch_size = 8
        max_samples = None  # Use full dataset
    elif device.type == 'mps':
        batch_size = 6
        max_samples = 200000  # Limited for MPS
    else:
        batch_size = 4
        max_samples = 50000  # Much smaller for CPU
    
    training_config = TrainingConfig(
        learning_rate=2e-4,
        weight_decay=0.01,
        batch_size=batch_size,
        gradient_accumulation_steps=4,
        max_epochs=20,
        warmup_ratio=0.05,
        save_every=500,
        eval_every=250,
        max_grad_norm=1.0,
        label_smoothing=0.1,
        beta1=0.9,
        beta2=0.95
    )
    
    # Initialize model manager
    model_manager = ModelManager("models")
    
    try:
        # Load and process OASST1 training data
        logger.info("📚 Loading OASST1 dataset...")
        texts = load_and_process_data("oasst1_data/oasst1_train.jsonl", max_samples)
        
        if len(texts) == 0:
            raise ValueError("No training data loaded!")
        
        # Create and train tokenizer
        logger.info("🔤 Training word-level tokenizer...")
        tokenizer = WordTokenizer()
        
        # Use a representative sample for tokenizer training
        sample_size = min(50000, len(texts))
        sample_texts = texts[:sample_size]
        all_text = "\n".join(sample_texts)
        
        tokenizer.train_from_text(all_text, vocab_size=model_config.vocab_size)
        model_config.vocab_size = tokenizer.vocab_size()
        
        logger.info(f"✅ Tokenizer trained - Vocabulary size: {model_config.vocab_size:,}")
        
        # Create optimized dataset
        logger.info("📦 Creating training dataset...")
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
        
        logger.info(f"📊 Dataset ready: {len(dataset):,} sequences, {len(dataloader):,} batches/epoch")
        
        # Initialize model
        logger.info("🧠 Initializing model...")
        model = WordTransformer(model_config).to(device)
        
        total_params, trainable_params = count_parameters(model)
        model_size_mb = total_params * 4 / 1024**2
        logger.info(f"Model parameters: {total_params:,} (~{model_size_mb:.1f}MB)")
        
        # Training components
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
        
        # Training loop
        logger.info("🚀 Starting training...")
        training_start = time.time()
        best_loss = float('inf')
        
        for epoch in range(1, training_config.max_epochs + 1):
            epoch_start = time.time()
            
            try:
                # Train epoch
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
                
                # Sample generation every few epochs
                if epoch % 3 == 0:
                    sample_prompts = [
                        "<user> What is artificial intelligence?",
                        "<user> How does machine learning work?",
                        "<assistant> I can help you with"
                    ]
                    for prompt in sample_prompts[:1]:  # Just one to save time
                        sample = generate_sample_text(model, tokenizer, prompt, 40)
                        logger.info(f"   Sample: {prompt} → {sample}")
                
                # Model saving
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    
                    # Create comprehensive metadata
                    metadata = ModelMetadata(
                        model_name="OASST1_WordTransformer",
                        version=f"v1.0_epoch_{epoch}",
                        created_at=datetime.now().isoformat(),
                        last_modified=datetime.now().isoformat(),
                        model_config=model_config,
                        training_config=training_config,
                        dataset_info={
                            "name": "OpenAssistant OASST1",
                            "num_samples": len(texts),
                            "vocab_size": model_config.vocab_size,
                            "seq_length": model_config.seq_length,
                            "source": "oasst1_train.jsonl",
                            "preprocessing": "Word-level tokenization with role formatting"
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
                        notes=f"OASST1 word-level transformer trained on {len(texts):,} samples. Best model at epoch {epoch}.",
                        tags=["oasst1", "word-level", "transformer", "best-model", f"epoch-{epoch}"]
                    )
                    
                    model_id = model_manager.save_model(model, tokenizer, metadata, optimizer, scheduler)
                    logger.info(f"💾 Best model saved: {model_id}")
                
                logger.info("=" * 60)
                
            except Exception as e:
                logger.error(f"❌ Error in epoch {epoch}: {e}")
                # Try to continue with next epoch
                continue
        
        # Training completion summary
        total_time = time.time() - training_start
        logger.info("=" * 70)
        logger.info("✅ Training completed successfully!")
        logger.info(f"🎯 Best loss achieved: {best_loss:.4f}")
        logger.info(f"🎯 Best perplexity: {calculate_perplexity(best_loss):.2f}")
        logger.info(f"⏱️  Total training time: {total_time/3600:.2f} hours")
        logger.info(f"🚀 Average processing speed: {len(dataset) * model_config.seq_length * training_config.max_epochs / total_time:.0f} tokens/sec")
        logger.info("=" * 70)
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("❌ Training interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # Final cleanup
        with memory_cleanup():
            pass

if __name__ == "__main__":
    exit(main())