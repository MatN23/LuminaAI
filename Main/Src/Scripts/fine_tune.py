# Copyright (c) 2025 Matias Nielsen. All rights reserved.
# Licensed under the Custom License below.

import os
import time
import math
import json
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Tuple, Optional
import gc

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Optimized thread configuration
torch.set_num_threads(4)
torch.set_num_interop_threads(2)

# Device selection
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

class CharDataset(Dataset):
    """Character-level dataset for fine-tuning."""
    
    def __init__(self, text: str, seq_length: int, char_to_ix: Dict[str, int]):
        self.seq_length = seq_length
        # Convert to indices with proper error handling
        self.data = []
        for c in text:
            if c in char_to_ix:
                self.data.append(char_to_ix[c])
            else:
                # Handle unknown characters gracefully
                self.data.append(char_to_ix.get(' ', 0))
        
        self.data = torch.tensor(self.data, dtype=torch.long)
        logger.info(f"Fine-tuning dataset created with {len(self.data):,} characters")

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        return (
            self.data[idx:idx + self.seq_length].clone(),
            self.data[idx + 1:idx + self.seq_length + 1].clone()
        )

def load_text_data(path: str) -> str:
    """
    Load text from JSONL with improved error handling and filtering.
    Supports both OASST format and plain text files.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    
    logger.info(f"Loading fine-tuning data from: {path}")
    
    role_tokens = {
        "prompter": "<|user|>",
        "assistant": "<|bot|>"
    }

    texts = []
    processed_count = 0
    skipped_count = 0

    try:
        if path.suffix == '.jsonl':
            with open(path, "r", encoding="utf-8") as f:
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

                        # Extract text content with multiple fallbacks
                        text = (record.get("text") or 
                               record.get("content") or 
                               (record.get("message", {}).get("text") if isinstance(record.get("message"), dict) else ""))
                        
                        text = str(text).strip()
                        if not text or len(text) < 5:  # Skip very short texts
                            skipped_count += 1
                            continue

                        # Add role tokens if available
                        role = str(record.get("role", "")).lower()
                        token = role_tokens.get(role, "")

                        if token:
                            texts.append(f"{token} {text}")
                        else:
                            texts.append(text)
                            
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
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    texts.append(content)
                    processed_count = 1

    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

    if not texts:
        raise ValueError(f"No valid text data found in {path}. Processed: {processed_count}, Skipped: {skipped_count}")

    result = "\n".join(texts) + "\n"
    logger.info(f"Loaded {len(texts):,} text entries ({len(result):,} characters)")
    logger.info(f"Processed: {processed_count}, Skipped: {skipped_count}")
    
    return result

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer models."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class CharTransformer(nn.Module):
    """Fixed character-level transformer model."""
    
    def __init__(self, vocab_size: int, hidden_size: int, seq_length: int, 
                 num_layers: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_enc = PositionalEncoding(hidden_size, seq_length, dropout)

        # Use TransformerEncoder with causal mask for decoder-only model
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.fc_out = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        self.init_weights()

    def init_weights(self):
        """Initialize model weights properly."""
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc_out.bias.data.zero_()
        self.fc_out.weight.data.uniform_(-initrange, initrange)

    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate causal mask for autoregressive generation."""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        return mask.masked_fill(mask == 1, float('-inf'))

    def forward(self, x):
        seq_len = x.size(1)
        mask = self.generate_square_subsequent_mask(seq_len).to(x.device)
        
        x = self.embedding(x) * math.sqrt(self.hidden_size)
        x = self.pos_enc(x)
        x = self.transformer(x, mask=mask)
        x = self.layer_norm(x)
        x = self.dropout(x)
        
        return self.fc_out(x)

def load_pretrained_model(model_path: str, device) -> Tuple[Dict, Dict[str, int], Dict[int, str], Dict]:
    """Load pre-trained model with comprehensive error handling."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Pre-trained model not found: {model_path}")
    
    logger.info(f"Loading pre-trained model from {model_path}")
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        # Validate required keys
        required_keys = ['char_to_ix', 'ix_to_char', 'model_state_dict']
        missing_keys = [key for key in required_keys if key not in checkpoint]
        if missing_keys:
            raise KeyError(f"Missing required keys in checkpoint: {missing_keys}")
        
        # Extract configuration with fallbacks
        if 'config' in checkpoint:
            # New format with config dict
            config = checkpoint['config'].copy()
        else:
            # Legacy format - reconstruct config
            config = {
                'vocab_size': checkpoint.get('vocab_size', len(checkpoint['char_to_ix'])),
                'hidden_size': checkpoint.get('hidden_size', 512),
                'seq_length': checkpoint.get('seq_length', 512),
                'num_layers': checkpoint.get('num_layers', 6),
                'nhead': checkpoint.get('nhead', 8)
            }
        
        char_to_ix = checkpoint['char_to_ix']
        ix_to_char = checkpoint['ix_to_char']
        
        logger.info(f"Pre-trained model loaded successfully:")
        logger.info(f"  Vocabulary size: {config['vocab_size']}")
        logger.info(f"  Architecture: {config['hidden_size']}d, {config['num_layers']}L, {config['nhead']}H")
        
        return config, char_to_ix, ix_to_char, checkpoint
        
    except Exception as e:
        logger.error(f"Error loading pre-trained model: {e}")
        raise

def adapt_vocabulary(pretrained_char_to_ix: Dict[str, int], pretrained_ix_to_char: Dict[int, str], 
                    new_text: str) -> Tuple[Dict[str, int], Dict[int, str], int]:
    """Adapt vocabulary to include new characters from fine-tuning data."""
    new_chars = set(new_text)
    pretrained_chars = set(pretrained_char_to_ix.keys())
    new_vocab_chars = new_chars - pretrained_chars

    if new_vocab_chars:
        logger.info(f"Found {len(new_vocab_chars)} new characters: {sorted(list(new_vocab_chars))}")
        
        extended_char_to_ix = pretrained_char_to_ix.copy()
        extended_ix_to_char = pretrained_ix_to_char.copy()
        
        next_idx = len(pretrained_char_to_ix)
        for c in sorted(new_vocab_chars):
            extended_char_to_ix[c] = next_idx
            extended_ix_to_char[next_idx] = c
            next_idx += 1
            
        return extended_char_to_ix, extended_ix_to_char, len(new_vocab_chars)
    else:
        logger.info("No new characters found. Using original vocabulary.")
        return pretrained_char_to_ix, pretrained_ix_to_char, 0

def extend_model_vocabulary(model: nn.Module, old_vocab_size: int, new_vocab_size: int):
    """Extend model vocabulary layers while preserving existing weights."""
    if new_vocab_size <= old_vocab_size:
        return
    
    logger.info(f"Extending model vocabulary from {old_vocab_size} to {new_vocab_size}")
    
    # Extend embedding layer
    old_emb_weight = model.embedding.weight.data
    new_embedding = nn.Embedding(new_vocab_size, model.hidden_size).to(device)
    
    # Copy old weights and initialize new ones
    with torch.no_grad():
        new_embedding.weight[:old_vocab_size] = old_emb_weight
        nn.init.uniform_(new_embedding.weight[old_vocab_size:], -0.1, 0.1)
    
    model.embedding = new_embedding
    
    # Extend output layer
    old_fc_weight = model.fc_out.weight.data
    old_fc_bias = model.fc_out.bias.data
    new_fc_out = nn.Linear(model.hidden_size, new_vocab_size).to(device)
    
    # Copy old weights and initialize new ones
    with torch.no_grad():
        new_fc_out.weight[:old_vocab_size] = old_fc_weight
        new_fc_out.bias[:old_vocab_size] = old_fc_bias
        nn.init.uniform_(new_fc_out.weight[old_vocab_size:], -0.1, 0.1)
        nn.init.zeros_(new_fc_out.bias[old_vocab_size:])
    
    model.fc_out = new_fc_out
    
    logger.info("Vocabulary extension completed successfully")

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

def train_epoch(model, dataloader, criterion, optimizer, scheduler, epoch: int, vocab_size: int) -> Tuple[float, float]:
    """Train one epoch with proper error handling and memory management."""
    model.train()
    total_loss = 0
    total_correct = 0
    total_chars = 0
    
    try:
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs.reshape(-1, vocab_size), targets.reshape(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            current_lr = scheduler.step()

            # Statistics
            total_loss += loss.detach().item() * inputs.numel()
            preds = outputs.argmax(dim=2)
            total_correct += (preds == targets).sum().detach().item()
            total_chars += targets.numel()

            # Memory cleanup
            if batch_idx % 50 == 0:
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                elif device.type == 'mps':
                    torch.mps.empty_cache()
                gc.collect()
            
            del inputs, targets, outputs, loss, preds
            
    except RuntimeError as e:
        if "out of memory" in str(e):
            logger.error(f"OOM at epoch {epoch}, batch {batch_idx}")
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            elif device.type == 'mps':
                torch.mps.empty_cache()
            gc.collect()
            raise e
        else:
            raise e

    avg_loss = total_loss / total_chars
    accuracy = total_correct / total_chars
    
    return avg_loss, accuracy

def main():
    """Main fine-tuning function."""
    
    # Configuration
    config = {
        'pretrained_model_path': "Model.pth",
        'data_path': "oasst1_data/oasst1_train.jsonl",
        'output_model_path': "FineTuned_Model.pth",
        'learning_rate': 1e-4,  # Lower LR for fine-tuning
        'epochs': 50,
        'batch_size': 16,
        'dropout': 0.1,
        'warmup_ratio': 0.1,
        'weight_decay': 0.01
    }
    
    logger.info("Starting fine-tuning with configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")

    try:
        # Load fine-tuning dataset
        text = load_text_data(config['data_path'])
        
        # Load pre-trained model
        model_config, pretrained_char_to_ix, pretrained_ix_to_char, checkpoint = load_pretrained_model(
            config['pretrained_model_path'], device
        )
        
        # Adapt vocabulary
        char_to_ix, ix_to_char, new_vocab_count = adapt_vocabulary(
            pretrained_char_to_ix, pretrained_ix_to_char, text
        )

        # Initialize model with original vocabulary size
        model = CharTransformer(
            vocab_size=model_config['vocab_size'],
            hidden_size=model_config['hidden_size'],
            seq_length=model_config['seq_length'],
            num_layers=model_config['num_layers'],
            nhead=model_config['nhead'],
            dropout=config['dropout']
        ).to(device)

        # Load pre-trained weights
        model.load_state_dict(checkpoint['model_state_dict'])

        # Extend vocabulary if needed
        if new_vocab_count > 0:
            extend_model_vocabulary(model, model_config['vocab_size'], len(char_to_ix))

        final_vocab_size = len(char_to_ix)

        logger.info(f"Fine-tuning setup completed:")
        logger.info(f"  Original vocab size: {model_config['vocab_size']}")
        logger.info(f"  Final vocab size: {final_vocab_size}")
        logger.info(f"  Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

        # Create dataset and dataloader
        dataset = CharDataset(text, model_config['seq_length'], char_to_ix)
        dataloader = DataLoader(
            dataset, 
            batch_size=config['batch_size'], 
            shuffle=True, 
            num_workers=2,
            pin_memory=True if device.type == 'cuda' else False
        )

        logger.info(f"Fine-tuning dataset: {len(dataset):,} sequences")

        # Initialize training components
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=config['learning_rate'], 
            weight_decay=config['weight_decay']
        )
        
        total_steps = len(dataloader) * config['epochs']
        warmup_steps = int(total_steps * config['warmup_ratio'])
        scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps)

        logger.info(f"Training steps: {total_steps:,} (warmup: {warmup_steps:,})")

        # Fine-tuning loop
        best_loss = float('inf')
        global_start = time.time()

        for epoch in range(1, config['epochs'] + 1):
            epoch_start = time.time()
            
            avg_loss, accuracy = train_epoch(
                model, dataloader, criterion, optimizer, scheduler, epoch, final_vocab_size
            )

            epoch_time = time.time() - epoch_start
            elapsed = time.time() - global_start
            current_lr = optimizer.param_groups[0]['lr']

            logger.info(
                f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Accuracy: {accuracy*100:.2f}% | "
                f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s | Elapsed: {elapsed:.1f}s"
            )

            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "char_to_ix": char_to_ix,
                    "ix_to_char": ix_to_char,
                    "config": {
                        'vocab_size': final_vocab_size,
                        'hidden_size': model_config['hidden_size'],
                        'seq_length': model_config['seq_length'],
                        'num_layers': model_config['num_layers'],
                        'nhead': model_config['nhead']
                    },
                    "epoch": epoch,
                    "loss": avg_loss,
                    "accuracy": accuracy,
                    "fine_tuned": True,
                    "original_vocab_size": model_config['vocab_size'],
                    "extended_vocab_size": final_vocab_size
                }, config['output_model_path'])
                
                logger.info(f"New best model saved: {config['output_model_path']}")

        logger.info(f"✅ Fine-tuning completed successfully!")
        logger.info(f"Best loss: {best_loss:.4f}")
        logger.info(f"Final model saved to: {config['output_model_path']}")
        
        return 0

    except Exception as e:
        logger.error(f"❌ Fine-tuning failed: {e}")
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