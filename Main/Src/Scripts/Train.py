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

# Optimized thread configuration
torch.set_num_threads(4)
torch.set_num_interop_threads(2)

def setup_device():
    """Setup the best available device with proper error handling."""
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        logger.info("Using device: MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using device: CUDA ({torch.cuda.get_device_name()})")
        logger.info(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = torch.device("cpu")
        logger.info("Using device: CPU")
    return device

device = setup_device()

class WordDataset(Dataset):
    """Word-level dataset for training."""
    
    def __init__(self, texts: List[str], tokenizer: WordTokenizer, seq_length: int):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.sequences = []
        
        logger.info("Creating word-level dataset...")
        
        for text in texts:
            tokens = tokenizer.encode(text)
            # Create overlapping sequences for better data utilization
            for i in range(0, len(tokens) - seq_length, seq_length // 2):
                if i + seq_length + 1 <= len(tokens):
                    self.sequences.append(tokens[i:i + seq_length + 1])
        
        logger.info(f"Created {len(self.sequences):,} training sequences")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        return (
            torch.tensor(seq[:-1], dtype=torch.long),  # Input
            torch.tensor(seq[1:], dtype=torch.long)    # Target (shifted by 1)
        )

def load_and_process_data(data_path: str) -> List[str]:
    """Load and process training data from JSONL or plain text files."""
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
                    try:
                        line = line.strip()
                        if not line:
                            continue
                        
                        record = json.loads(line)
                        processed_count += 1
                        
                        # Skip deleted entries
                        if record.get("deleted", False):
                            skipped_count += 1
                            continue
                        
                        # Skip non-English entries
                        if record.get("lang") != "en":
                            skipped_count += 1
                            continue
                        
                        # Extract text content
                        text = (record.get("text") or 
                               record.get("content") or 
                               (record.get("message", {}).get("text") if isinstance(record.get("message"), dict) else ""))
                        
                        text = str(text).strip()
                        if not text or len(text.split()) < 5:  # Skip very short texts
                            skipped_count += 1
                            continue
                        
                        # Add role tokens
                        role = str(record.get("role", "")).lower()
                        token = role_tokens.get(role, "")
                        
                        if token:
                            texts.append(f"{token} {text} </s>")  # Add end token
                        else:
                            texts.append(f"{text} </s>")
                        
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping malformed JSON at line {line_num}: {e}")
                        skipped_count += 1
                        continue
                    except Exception as e:
                        logger.warning(f"Error processing line {line_num}: {e}")
                        skipped_count += 1
                        continue
        else:
            # Plain text file
            with open(data_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    # Split into paragraphs or sentences
                    paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
                    texts.extend([f"{p} </s>" for p in paragraphs])
                    processed_count = len(paragraphs)
    
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise
    
    if not texts:
        raise ValueError(f"No valid text data found in {data_path}. Processed: {processed_count}, Skipped: {skipped_count}")
    
    logger.info(f"Loaded {len(texts):,} text entries")
    logger.info(f"Total characters: {sum(len(text) for text in texts):,}")
    logger.info(f"Processed: {processed_count}, Skipped: {skipped_count}")
    
    return texts

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
    
    def state_dict(self):
        """Return scheduler state."""
        return {
            'current_step': self.current_step,
            'warmup_steps': self.warmup_steps,
            'total_steps': self.total_steps,
            'min_lr': self.min_lr,
            'base_lr': self.base_lr
        }
    
    def load_state_dict(self, state_dict):
        """Load scheduler state."""
        self.current_step = state_dict['current_step']
        self.warmup_steps = state_dict['warmup_steps']
        self.total_steps = state_dict['total_steps']
        self.min_lr = state_dict['min_lr']
        self.base_lr = state_dict['base_lr']

def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def calculate_perplexity(loss: float) -> float:
    """Calculate perplexity from loss."""
    return math.exp(min(loss, 20))  # Cap to prevent overflow

def train_epoch(model, dataloader, criterion, optimizer, scheduler, epoch: int, 
                gradient_accumulation_steps: int = 1) -> Tuple[float, float]:
    """Train for one epoch with gradient accumulation and proper memory management."""
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
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Optimizer step
                optimizer.step()
                current_lr = scheduler.step()
                optimizer.zero_grad()
                
                accumulation_steps = 0
            
            # Statistics (detach to prevent memory accumulation)
            total_loss += loss.detach().item() * gradient_accumulation_steps * inputs.numel()
            
            with torch.no_grad():
                preds = logits.argmax(dim=2)
                total_correct += (preds == targets).sum().item()
                total_tokens += targets.numel()
            
            # Progress logging
            if batch_idx % 100 == 0:
                current_loss = total_loss / total_tokens if total_tokens > 0 else 0
                accuracy = total_correct / total_tokens if total_tokens > 0 else 0
                logger.info(f"Epoch {epoch} | Batch {batch_idx} | Loss: {current_loss:.4f} | "
                           f"Acc: {accuracy*100:.2f}% | LR: {current_lr:.2e}")
            
            # Memory cleanup
            if batch_idx % 50 == 0:
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                elif device.type == 'mps':
                    torch.mps.empty_cache()
                gc.collect()
            
            # Clear intermediate tensors
            del inputs, targets, logits, loss, preds
    
    except RuntimeError as e:
        if "out of memory" in str(e):
            logger.error(f"CUDA OOM at epoch {epoch}, batch {batch_idx}")
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            elif device.type == 'mps':
                torch.mps.empty_cache()
            gc.collect()
            raise e
        else:
            raise e
    
    # Final cleanup
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        torch.mps.empty_cache()
    gc.collect()
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0
    
    return avg_loss, accuracy

def generate_sample_text(model, tokenizer, prompt: str = "<user> Hello", 
                        max_length: int = 50, temperature: float = 0.8) -> str:
    """Generate sample text for monitoring training progress."""
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
                
                # Simple sampling
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
                
                # Stop if we hit end token
                if next_token.item() == tokenizer.vocab.get("</s>", -1):
                    break
            
            # Decode response
            response_ids = generated[0][input_ids.size(1):].tolist()
            response = tokenizer.decode(response_ids)
            
            return response.strip()
    
    except Exception as e:
        logger.error(f"Error generating sample: {e}")
        return "Error generating sample"
    finally:
        model.train()

def main():
    """Main training function with professional model management."""
    
    # Configuration
    model_config = ModelConfig(
        vocab_size=32000,  # Will be updated after tokenizer training
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        seq_length=1024,
        dropout=0.1,
        model_type="WordTransformer",
        tokenizer_type="word"
    )
    
    training_config = TrainingConfig(
        learning_rate=3e-4,
        weight_decay=0.01,
        batch_size=4,
        gradient_accumulation_steps=8,
        max_epochs=50,
        warmup_ratio=0.1,
        save_every=1000,
        eval_every=500,
        max_grad_norm=1.0,
        label_smoothing=0.1,
        beta1=0.9,
        beta2=0.95
    )
    
    logger.info("🚀 Starting Word-Level Transformer Training")
    logger.info("=" * 60)
    logger.info("Model Configuration:")
    for key, value in asdict(model_config).items():
        logger.info(f"  {key}: {value}")
    
    logger.info("\nTraining Configuration:")
    for key, value in asdict(training_config).items():
        logger.info(f"  {key}: {value}")
    logger.info("=" * 60)
    
    # Initialize model manager
    model_manager = ModelManager("models")
    
    try:
        # Load and process training data
        texts = load_and_process_data("oasst1_data/oasst1_train.jsonl")
        
        # Create and train tokenizer
        logger.info("📚 Training word-level tokenizer...")
        tokenizer = WordTokenizer()
        all_text = " ".join(texts)
        tokenizer.train_from_text(all_text, vocab_size=model_config.vocab_size)
        
        # Update model config with actual vocab size
        model_config.vocab_size = tokenizer.vocab_size()
        logger.info(f"Final vocabulary size: {model_config.vocab_size:,}")
        
        # Create dataset and dataloader
        dataset = WordDataset(texts, tokenizer, model_config.seq_length)
        dataloader = DataLoader(
            dataset,
            batch_size=training_config.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True if device.type == 'cuda' else False,
            persistent_workers=True
        )
        
        logger.info(f"Dataset: {len(dataset):,} sequences, {len(dataloader):,} batches per epoch")
        
        # Initialize model
        model = WordTransformer(model_config).to(device)
        total_params, trainable_params = count_parameters(model)
        
        logger.info(f"🧠 Model initialized:")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")
        logger.info(f"  Model size: ~{total_params * 4 / 1024**2:.2f} MB")
        
        # Training components
        optimizer = optim.AdamW(
            model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
            betas=(training_config.beta1, training_config.beta2)
        )
        
        criterion = nn.CrossEntropyLoss(
            label_smoothing=training_config.label_smoothing,
            ignore_index=tokenizer.vocab.get("<pad>", 0)  # Ignore padding tokens
        )
        
        # Learning rate scheduler
        total_steps = len(dataloader) * training_config.max_epochs // training_config.gradient_accumulation_steps
        warmup_steps = int(total_steps * training_config.warmup_ratio)
        scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps)
        
        logger.info(f"📈 Training schedule:")
        logger.info(f"  Total steps: {total_steps:,}")
        logger.info(f"  Warmup steps: {warmup_steps:,}")
        
        # Training loop
        logger.info("🎯 Starting training...")
        training_start_time = time.time()
        best_loss = float('inf')
        global_step = 0
        
        for epoch in range(1, training_config.max_epochs + 1):
            epoch_start = time.time()
            
            # Train epoch
            avg_loss, accuracy = train_epoch(
                model, dataloader, criterion, optimizer, scheduler, epoch,
                training_config.gradient_accumulation_steps
            )
            
            # Calculate metrics
            perplexity = calculate_perplexity(avg_loss)
            epoch_time = time.time() - epoch_start
            elapsed_time = time.time() - training_start_time
            current_lr = optimizer.param_groups[0]['lr']
            global_step = epoch * len(dataloader) // training_config.gradient_accumulation_steps
            
            # Log epoch results
            logger.info(f"📊 Epoch {epoch}/{training_config.max_epochs} Summary:")
            logger.info(f"   Loss: {avg_loss:.4f} | Perplexity: {perplexity:.2f}")
            logger.info(f"   Accuracy: {accuracy*100:.2f}% | LR: {current_lr:.2e}")
            logger.info(f"   Time: {epoch_time:.1f}s | Total: {elapsed_time/3600:.2f}h")
            
            # Generate sample text
            if epoch % 5 == 0:
                sample = generate_sample_text(model, tokenizer, "<user> How are you?", max_length=30)
                logger.info(f"   Sample: {sample}")
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                
                # Create comprehensive metadata
                metadata = ModelMetadata(
                    model_name="WordTransformer",
                    version=f"v1.0_epoch_{epoch}",
                    created_at=datetime.now().isoformat(),
                    last_modified=datetime.now().isoformat(),
                    model_config=model_config,
                    training_config=training_config,
                    dataset_info={
                        "num_samples": len(texts),
                        "total_tokens": sum(len(tokenizer.encode(text)) for text in texts[:1000]),  # Sample
                        "avg_tokens": sum(len(tokenizer.encode(text)) for text in texts[:1000]) / min(1000, len(texts)),
                        "source": "oasst1_train.jsonl",
                        "preprocessing": "Word-level tokenization with special tokens"
                    },
                    performance_metrics={
                        "loss": avg_loss,
                        "perplexity": perplexity,
                        "accuracy": accuracy,
                        "tokens_per_second": global_step * training_config.batch_size * model_config.seq_length / elapsed_time
                    },
                    model_size_mb=0,  # Will be calculated by model_manager
                    total_parameters=total_params,
                    trainable_parameters=trainable_params,
                    training_time_hours=elapsed_time / 3600,
                    epochs_trained=epoch,
                    best_loss=best_loss,
                    best_perplexity=calculate_perplexity(best_loss),
                    hardware_used=f"{device.type.upper()}: {torch.cuda.get_device_name() if device.type == 'cuda' else 'CPU'}",
                    pytorch_version=torch.__version__,
                    cuda_version=torch.version.cuda if device.type == 'cuda' else None,
                    model_hash="",  # Will be calculated by model_manager
                    tokenizer_hash="",  # Will be calculated by model_manager
                    notes=f"Best checkpoint at epoch {epoch}. Trained on OASST1 dataset with word-level tokenization.",
                    tags=["word-level", "transformer", "chat", f"epoch-{epoch}", "best"]
                )
                
                # Save model with comprehensive metadata
                model_id = model_manager.save_model(model, tokenizer, metadata, optimizer, scheduler)
                logger.info(f"💾 New best model saved: {model_id} (Loss: {best_loss:.4f})")
        
        # Training completed
        total_time = time.time() - training_start_time
        logger.info("=" * 60)
        logger.info("✅ Training completed successfully!")
        logger.info(f"🎯 Best loss: {best_loss:.4f} (Perplexity: {calculate_perplexity(best_loss):.2f})")
        logger.info(f"⏱️  Total training time: {total_time/3600:.2f} hours")
        
        # List available models
        logger.info("\n📋 Available models:")
        for model_info in model_manager.list_models():
            logger.info(f"   {model_info['id']}: {model_info['name']} {model_info['version']} "
                       f"(Loss: {model_info['best_loss']:.4f}, Size: {model_info['size_mb']:.2f}MB)")
        
        logger.info("\n🚀 Ready for chat! Run: python ChatAI.py")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("❌ Training interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
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