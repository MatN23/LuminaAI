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

# Import shared components
from model_manager import ModelManager, ModelConfig, TrainingConfig, ModelMetadata
from word_transformer import WordTransformer, WordTokenizer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_device():
    """Setup the best available device with proper error handling."""
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        logger.info("Using device: MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using device: CUDA ({torch.cuda.get_device_name()})")
    else:
        device = torch.device("cpu")
        logger.info("Using device: CPU")
    return device

device = setup_device()

class FineTuneDataset(Dataset):
    """Dataset for fine-tuning with conversation formatting."""
    
    def __init__(self, texts: List[str], tokenizer: WordTokenizer, seq_length: int):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.sequences = []
        
        logger.info("Creating fine-tuning dataset...")
        
        for text in texts:
            tokens = tokenizer.encode(text)
            # Create overlapping sequences but with smaller stride for fine-tuning
            stride = seq_length // 4  # More overlap for better fine-tuning
            for i in range(0, len(tokens) - seq_length, stride):
                if i + seq_length + 1 <= len(tokens):
                    self.sequences.append(tokens[i:i + seq_length + 1])
        
        logger.info(f"Created {len(self.sequences):,} fine-tuning sequences")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        return (
            torch.tensor(seq[:-1], dtype=torch.long),
            torch.tensor(seq[1:], dtype=torch.long)
        )

def load_fine_tune_data(data_path: str) -> List[str]:
    """Load and process fine-tuning data with conversation formatting."""
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Fine-tuning data file not found: {data_path}")
    
    logger.info(f"Loading fine-tuning data from: {data_path}")
    
    texts = []
    processed_count = 0
    skipped_count = 0
    
    try:
        if data_path.suffix == '.jsonl':
            conversations = []
            current_conversation = []
            
            with open(data_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        line = line.strip()
                        if not line:
                            continue
                        
                        record = json.loads(line)
                        processed_count += 1
                        
                        # Skip deleted or non-English entries
                        if record.get("deleted", False) or record.get("lang") != "en":
                            skipped_count += 1
                            continue
                        
                        text = (record.get("text") or 
                               record.get("content") or 
                               (record.get("message", {}).get("text") if isinstance(record.get("message"), dict) else ""))
                        
                        text = str(text).strip()
                        if not text or len(text.split()) < 3:
                            skipped_count += 1
                            continue
                        
                        role = str(record.get("role", "")).lower()
                        
                        if role == "prompter":
                            # Start new conversation or add to current
                            if current_conversation:
                                # Save previous conversation
                                if len(current_conversation) >= 2:  # At least one exchange
                                    conversations.append(" ".join(current_conversation) + " </s>")
                                current_conversation = []
                            current_conversation.append(f"<user> {text}")
                        
                        elif role == "assistant":
                            if current_conversation:  # Only if we have a user message
                                current_conversation.append(f"<bot> {text}")
                                # Complete conversation pair
                                conversations.append(" ".join(current_conversation) + " </s>")
                                current_conversation = []
                            else:
                                skipped_count += 1
                    
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping malformed JSON at line {line_num}: {e}")
                        skipped_count += 1
                        continue
                    except Exception as e:
                        logger.warning(f"Error processing line {line_num}: {e}")
                        skipped_count += 1
                        continue
            
            # Add any remaining conversation
            if current_conversation and len(current_conversation) >= 2:
                conversations.append(" ".join(current_conversation) + " </s>")
            
            texts = conversations
            
        else:
            # Plain text file - split into chunks
            with open(data_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    # Split by double newlines (paragraphs)
                    chunks = [chunk.strip() for chunk in content.split('\n\n') if chunk.strip()]
                    texts = [f"<user> {chunk} </s>" for chunk in chunks]
                    processed_count = len(chunks)
    
    except Exception as e:
        logger.error(f"Error loading fine-tuning data: {e}")
        raise
    
    if not texts:
        raise ValueError(f"No valid fine-tuning data found. Processed: {processed_count}, Skipped: {skipped_count}")
    
    logger.info(f"Loaded {len(texts):,} conversation pairs for fine-tuning")
    logger.info(f"Average conversation length: {sum(len(text.split()) for text in texts) / len(texts):.1f} words")
    logger.info(f"Processed: {processed_count}, Skipped: {skipped_count}")
    
    return texts

def extend_vocabulary(base_tokenizer: WordTokenizer, new_texts: List[str], 
                     max_new_tokens: int = 5000) -> Tuple[WordTokenizer, int]:
    """Extend tokenizer vocabulary with new tokens from fine-tuning data."""
    import collections
    import re
    
    # Extract all words from new texts
    all_text = " ".join(new_texts).lower()
    new_words = re.findall(r'\S+', all_text)
    word_counts = collections.Counter(new_words)
    
    # Find words not in base vocabulary
    new_vocab_words = []
    for word, count in word_counts.most_common():
        if word not in base_tokenizer.vocab and len(new_vocab_words) < max_new_tokens:
            if count >= 2:  # Only add words that appear at least twice
                new_vocab_words.append(word)
    
    if not new_vocab_words:
        logger.info("No new vocabulary words needed for fine-tuning")
        return base_tokenizer, 0
    
    # Create extended tokenizer
    extended_tokenizer = WordTokenizer(base_tokenizer.vocab.copy())
    
    # Add new words
    for word in new_vocab_words:
        extended_tokenizer.vocab[word] = extended_tokenizer.next_id
        extended_tokenizer.id_to_token[extended_tokenizer.next_id] = word
        extended_tokenizer.next_id += 1
    
    logger.info(f"Extended vocabulary by {len(new_vocab_words)} tokens")
    logger.info(f"New vocabulary size: {extended_tokenizer.vocab_size()}")
    
    return extended_tokenizer, len(new_vocab_words)

def extend_model_embeddings(model: WordTransformer, old_vocab_size: int, new_vocab_size: int):
    """Extend model embedding layers for new vocabulary."""
    if new_vocab_size <= old_vocab_size:
        return
    
    logger.info(f"Extending model embeddings from {old_vocab_size} to {new_vocab_size}")
    
    # Extend token embedding
    old_embeddings = model.token_embedding.weight.data
    new_embedding = nn.Embedding(new_vocab_size, model.config.hidden_size).to(device)
    
    # Copy old embeddings and initialize new ones
    with torch.no_grad():
        new_embedding.weight[:old_vocab_size] = old_embeddings
        # Initialize new embeddings with small random values
        nn.init.normal_(new_embedding.weight[old_vocab_size:], mean=0.0, std=0.02)
    
    model.token_embedding = new_embedding
    
    # Since we tied weights, the lm_head will automatically use the new embeddings
    model.lm_head.weight = model.token_embedding.weight
    
    # Update config
    model.config.vocab_size = new_vocab_size
    
    logger.info("Model embedding extension completed")

class WarmupCosineScheduler:
    """Learning rate scheduler with warmup and cosine decay."""
    
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_step = 0
    
    def step(self) -> float:
        """Update learning rate and return current LR."""
        self.current_step += 1
        
        if self.current_step < self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * self.current_step / self.warmup_steps
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr

def fine_tune_epoch(model, dataloader, criterion, optimizer, scheduler, epoch: int,
                   gradient_accumulation_steps: int = 1) -> Tuple[float, float]:
    """Fine-tune for one epoch with careful learning rate and gradient management."""
    model.train()
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    accumulation_steps = 0
    
    try:
        optimizer.zero_grad()
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            # Forward pass
            logits = model(inputs)
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            loss = loss / gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Accumulate gradients
            accumulation_steps += 1
            
            if accumulation_steps >= gradient_accumulation_steps:
                # Gradient clipping (more conservative for fine-tuning)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                
                # Optimizer step
                optimizer.step()
                current_lr = scheduler.step()
                optimizer.zero_grad()
                
                accumulation_steps = 0
            
            # Statistics
            total_loss += loss.detach().item() * gradient_accumulation_steps * inputs.numel()
            
            with torch.no_grad():
                preds = logits.argmax(dim=2)
                total_correct += (preds == targets).sum().item()
                total_tokens += targets.numel()
            
            # Progress logging (less frequent for fine-tuning)
            if batch_idx % 50 == 0:
                current_loss = total_loss / total_tokens if total_tokens > 0 else 0
                accuracy = total_correct / total_tokens if total_tokens > 0 else 0
                logger.info(f"Fine-tune Epoch {epoch} | Batch {batch_idx} | Loss: {current_loss:.4f} | "
                           f"Acc: {accuracy*100:.2f}% | LR: {current_lr:.2e}")
            
            # Memory cleanup
            if batch_idx % 25 == 0:
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                elif device.type == 'mps':
                    torch.mps.empty_cache()
                gc.collect()
            
            del inputs, targets, logits, loss, preds
    
    except RuntimeError as e:
        if "out of memory" in str(e):
            logger.error(f"OOM during fine-tuning at epoch {epoch}")
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            elif device.type == 'mps':
                torch.mps.empty_cache()
            gc.collect()
            raise e
        else:
            raise e
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0
    
    return avg_loss, accuracy

def calculate_perplexity(loss: float) -> float:
    """Calculate perplexity from loss."""
    return math.exp(min(loss, 20))

def main():
    """Main fine-tuning function."""
    
    # Fine-tuning configuration (more conservative than training)
    fine_tune_config = TrainingConfig(
        learning_rate=5e-5,  # Much lower learning rate for fine-tuning
        weight_decay=0.01,
        batch_size=2,  # Smaller batch size for fine-tuning
        gradient_accumulation_steps=16,  # Higher accumulation to maintain effective batch size
        max_epochs=10,  # Fewer epochs for fine-tuning
        warmup_ratio=0.05,  # Less warmup needed
        save_every=500,
        eval_every=250,
        max_grad_norm=0.5,  # More conservative gradient clipping
        label_smoothing=0.05,  # Less smoothing for fine-tuning
        beta1=0.9,
        beta2=0.95
    )
    
    logger.info("🎯 Starting Word-Level Transformer Fine-tuning")
    logger.info("=" * 60)
    logger.info("Fine-tuning Configuration:")
    for key, value in asdict(fine_tune_config).items():
        logger.info(f"  {key}: {value}")
    logger.info("=" * 60)
    
    # Initialize model manager
    model_manager = ModelManager("models")
    
    try:
        # List available base models
        available_models = model_manager.list_models()
        if not available_models:
            raise ValueError("No base models found. Please train a base model first using Train.py")
        
        logger.info("📋 Available base models:")
        for i, model in enumerate(available_models):
            logger.info(f"  {i+1}. {model['name']} {model['version']} "
                       f"(Loss: {model['best_loss']:.4f}, Size: {model['size_mb']:.2f}MB)")
        
        # Select base model (auto-select best or let user choose)
        try:
            choice = input(f"\nSelect base model (1-{len(available_models)}) or press Enter for best: ").strip()
            if choice:
                model_idx = int(choice) - 1
                if 0 <= model_idx < len(available_models):
                    base_model_info = available_models[model_idx]
                else:
                    logger.warning("Invalid selection, using best model.")
                    base_model_info = min(available_models, key=lambda x: x['best_loss'])
            else:
                base_model_info = min(available_models, key=lambda x: x['best_loss'])
        except (ValueError, KeyboardInterrupt):
            logger.info("Using best available model.")
            base_model_info = min(available_models, key=lambda x: x['best_loss'])
        
        logger.info(f"🎯 Selected base model: {base_model_info['name']} {base_model_info['version']}")
        
        # Load base model
        logger.info("📥 Loading base model...")
        base_model, base_tokenizer, base_metadata = model_manager.load_model(base_model_info['id'])
        
        logger.info(f"✅ Base model loaded:")
        logger.info(f"  Parameters: {base_metadata.total_parameters:,}")
        logger.info(f"  Vocabulary: {base_metadata.model_config.vocab_size:,}")
        logger.info(f"  Best Loss: {base_metadata.best_loss:.4f}")
        
        # Load fine-tuning data
        fine_tune_data_path = input("\nEnter fine-tuning data path (default: oasst1_data/oasst1_train.jsonl): ").strip()
        if not fine_tune_data_path:
            fine_tune_data_path = "oasst1_data/oasst1_train.jsonl"
        
        texts = load_fine_tune_data(fine_tune_data_path)
        
        # Extend vocabulary if needed
        logger.info("🔤 Checking vocabulary coverage...")
        extended_tokenizer, new_tokens_count = extend_vocabulary(base_tokenizer, texts, max_new_tokens=2000)
        
        # Extend model if vocabulary was extended
        if new_tokens_count > 0:
            extend_model_embeddings(base_model, base_metadata.model_config.vocab_size, extended_tokenizer.vocab_size())
            logger.info(f"📈 Model extended with {new_tokens_count} new tokens")
        
        # Create fine-tuning dataset
        dataset = FineTuneDataset(texts, extended_tokenizer, base_metadata.model_config.seq_length)
        dataloader = DataLoader(
            dataset,
            batch_size=fine_tune_config.batch_size,
            shuffle=True,
            num_workers=1,  # Fewer workers for fine-tuning
            pin_memory=True if device.type == 'cuda' else False
        )
        
        logger.info(f"📚 Fine-tuning dataset: {len(dataset):,} sequences, {len(dataloader):,} batches per epoch")
        
        # Setup fine-tuning components
        # Use lower learning rate and different optimizer settings for fine-tuning
        optimizer = optim.AdamW(
            base_model.parameters(),
            lr=fine_tune_config.learning_rate,
            weight_decay=fine_tune_config.weight_decay,
            betas=(fine_tune_config.beta1, fine_tune_config.beta2),
            eps=1e-8
        )
        
        criterion = nn.CrossEntropyLoss(
            label_smoothing=fine_tune_config.label_smoothing,
            ignore_index=extended_tokenizer.vocab.get("<pad>", 0)
        )
        
        # Learning rate scheduler
        total_steps = len(dataloader) * fine_tune_config.max_epochs // fine_tune_config.gradient_accumulation_steps
        warmup_steps = int(total_steps * fine_tune_config.warmup_ratio)
        scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps, min_lr=1e-6)
        
        logger.info(f"📊 Fine-tuning schedule:")
        logger.info(f"  Total steps: {total_steps:,}")
        logger.info(f"  Warmup steps: {warmup_steps:,}")
        logger.info(f"  Effective batch size: {fine_tune_config.batch_size * fine_tune_config.gradient_accumulation_steps}")
        
        # Fine-tuning loop
        logger.info("🚀 Starting fine-tuning...")
        fine_tune_start_time = time.time()
        best_loss = float('inf')
        global_step = 0
        
        for epoch in range(1, fine_tune_config.max_epochs + 1):
            epoch_start = time.time()
            
            # Fine-tune epoch
            avg_loss, accuracy = fine_tune_epoch(
                base_model, dataloader, criterion, optimizer, scheduler, epoch,
                fine_tune_config.gradient_accumulation_steps
            )
            
            # Calculate metrics
            perplexity = calculate_perplexity(avg_loss)
            epoch_time = time.time() - epoch_start
            elapsed_time = time.time() - fine_tune_start_time
            current_lr = optimizer.param_groups[0]['lr']
            global_step = epoch * len(dataloader) // fine_tune_config.gradient_accumulation_steps
            
            # Log epoch results
            logger.info(f"📊 Fine-tune Epoch {epoch}/{fine_tune_config.max_epochs} Summary:")
            logger.info(f"   Loss: {avg_loss:.4f} | Perplexity: {perplexity:.2f}")
            logger.info(f"   Accuracy: {accuracy*100:.2f}% | LR: {current_lr:.2e}")
            logger.info(f"   Time: {epoch_time:.1f}s | Total: {elapsed_time/3600:.2f}h")
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                
                # Create fine-tuned model metadata
                fine_tuned_metadata = ModelMetadata(
                    model_name=f"{base_metadata.model_name}_FineTuned",
                    version=f"v1.0_epoch_{epoch}",
                    created_at=datetime.now().isoformat(),
                    last_modified=datetime.now().isoformat(),
                    model_config=ModelConfig(
                        vocab_size=extended_tokenizer.vocab_size(),
                        hidden_size=base_metadata.model_config.hidden_size,
                        num_layers=base_metadata.model_config.num_layers,
                        num_heads=base_metadata.model_config.num_heads,
                        seq_length=base_metadata.model_config.seq_length,
                        dropout=base_metadata.model_config.dropout,
                        model_type=base_metadata.model_config.model_type,
                        tokenizer_type=base_metadata.model_config.tokenizer_type
                    ),
                    training_config=fine_tune_config,
                    dataset_info={
                        "base_model": f"{base_metadata.model_name} {base_metadata.version}",
                        "fine_tune_samples": len(texts),
                        "total_tokens": sum(len(extended_tokenizer.encode(text)) for text in texts[:1000]),
                        "avg_tokens": sum(len(extended_tokenizer.encode(text)) for text in texts[:1000]) / min(1000, len(texts)),
                        "source": fine_tune_data_path,
                        "vocabulary_extended": new_tokens_count > 0,
                        "new_tokens_added": new_tokens_count
                    },
                    performance_metrics={
                        "loss": avg_loss,
                        "perplexity": perplexity,
                        "accuracy": accuracy,
                        "base_model_loss": base_metadata.best_loss,
                        "improvement": base_metadata.best_loss - avg_loss,
                        "tokens_per_second": global_step * fine_tune_config.batch_size * base_metadata.model_config.seq_length / elapsed_time
                    },
                    model_size_mb=0,  # Will be calculated by model_manager
                    total_parameters=base_metadata.total_parameters + (new_tokens_count * base_metadata.model_config.hidden_size * 2 if new_tokens_count > 0 else 0),
                    trainable_parameters=base_metadata.trainable_parameters + (new_tokens_count * base_metadata.model_config.hidden_size * 2 if new_tokens_count > 0 else 0),
                    training_time_hours=elapsed_time / 3600,
                    epochs_trained=epoch,
                    best_loss=best_loss,
                    best_perplexity=calculate_perplexity(best_loss),
                    hardware_used=f"{device.type.upper()}: {torch.cuda.get_device_name() if device.type == 'cuda' else 'CPU'}",
                    pytorch_version=torch.__version__,
                    cuda_version=torch.version.cuda if device.type == 'cuda' else None,
                    model_hash="",  # Will be calculated by model_manager
                    tokenizer_hash="",  # Will be calculated by model_manager
                    notes=f"Fine-tuned from {base_metadata.model_name} {base_metadata.version}. "
                          f"Extended vocabulary by {new_tokens_count} tokens. "
                          f"Loss improved from {base_metadata.best_loss:.4f} to {best_loss:.4f}.",
                    tags=["fine-tuned", "word-level", "transformer", "chat", f"epoch-{epoch}", 
                          f"base-{base_metadata.model_name}", "best"]
                )
                
                # Save fine-tuned model
                model_id = model_manager.save_model(base_model, extended_tokenizer, fine_tuned_metadata, optimizer, scheduler)
                logger.info(f"💾 New best fine-tuned model saved: {model_id}")
                logger.info(f"   Loss improvement: {base_metadata.best_loss:.4f} → {best_loss:.4f} "
                           f"({base_metadata.best_loss - best_loss:+.4f})")
        
        # Fine-tuning completed
        total_time = time.time() - fine_tune_start_time
        logger.info("=" * 60)
        logger.info("✅ Fine-tuning completed successfully!")
        logger.info(f"🎯 Best loss: {best_loss:.4f} (Perplexity: {calculate_perplexity(best_loss):.2f})")
        logger.info(f"📈 Improvement: {base_metadata.best_loss:.4f} → {best_loss:.4f} "
                   f"({base_metadata.best_loss - best_loss:+.4f})")
        logger.info(f"⏱️  Fine-tuning time: {total_time/3600:.2f} hours")
        
        if new_tokens_count > 0:
            logger.info(f"🔤 Vocabulary extended by {new_tokens_count} tokens")
        
        # List available models
        logger.info("\n📋 Available models after fine-tuning:")
        for model_info in model_manager.list_models():
            tags_str = f"[{', '.join(model_info['tags'])}]" if model_info['tags'] else ""
            logger.info(f"   {model_info['id']}: {model_info['name']} {model_info['version']} {tags_str}")
            logger.info(f"      Loss: {model_info['best_loss']:.4f}, Size: {model_info['size_mb']:.2f}MB")
        
        logger.info("\n🚀 Ready for chat! Run: python ChatAI.py")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("❌ Fine-tuning interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Fine-tuning failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # Cleanup
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        elif device.type == 'mps':
            torch.mps.empty_cache()
        gc.collect()

if __name__ == "__main__":
    exit(main())