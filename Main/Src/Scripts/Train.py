# Copyright (c) 2025 Matias Nielsen. All rights reserved.
# Licensed under the Custom License below.

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
from typing import Dict, Tuple, Optional, List
import gc

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Optimized thread configuration
torch.set_num_threads(4)  # Increased from 2
torch.set_num_interop_threads(2)  # Increased from 1

# Device selection with better error handling
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

class CharDataset(Dataset):
    """Optimized character-level dataset with better memory usage."""
    
    def __init__(self, text: str, seq_length: int, char_to_ix: Dict[str, int]):
        self.seq_length = seq_length
        logger.info(f"Creating character dataset with sequence length: {seq_length}")
        
        # Convert to indices once and store
        self.data = torch.tensor([char_to_ix[c] for c in text], dtype=torch.long)
        logger.info(f"Dataset created with {len(self.data):,} characters")
        
    def __len__(self):
        return len(self.data) - self.seq_length
    
    def __getitem__(self, idx):
        return (
            self.data[idx:idx + self.seq_length].clone(),
            self.data[idx + 1:idx + self.seq_length + 1].clone()
        )

def load_text_data(path: str) -> str:
    """
    Load and process text data from JSONL or plain text files.
    Includes proper error handling and text filtering.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    
    logger.info(f"Loading data from: {path}")
    
    role_tokens = {
        "prompter": "<|user|>",
        "assistant": "<|bot|>"
    }
    
    texts = []
    
    try:
        if path.suffix == '.jsonl':
            with open(path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        record = json.loads(line.strip())
                        
                        # Skip deleted entries
                        if record.get("deleted", False):
                            continue
                        
                        # Skip non-English entries
                        if record.get("lang") != "en":
                            continue
                        
                        # Extract text content
                        text = record.get("text") or record.get("content") or ""
                        if not text and "message" in record:
                            text = record["message"].get("text", "")
                        
                        text = text.strip()
                        if not text:
                            continue
                        
                        # Add role tokens
                        role = record.get("role", "").lower()
                        token = role_tokens.get(role, "")
                        
                        if token:
                            texts.append(f"{token} {text}")
                        else:
                            texts.append(text)
                            
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping malformed JSON at line {line_num}: {e}")
                        continue
                    except Exception as e:
                        logger.warning(f"Error processing line {line_num}: {e}")
                        continue
        else:
            # Plain text file
            with open(path, "r", encoding="utf-8") as f:
                text_content = f.read()
                texts.append(text_content)
    
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise
    
    if not texts:
        raise ValueError("No valid text data found in file")
    
    result = "\n".join(texts) + "\n"
    logger.info(f"Loaded {len(texts):,} text entries ({len(result):,} characters)")
    
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
    """Fixed character-level transformer using proper encoder architecture."""
    
    def __init__(self, vocab_size: int, hidden_size: int, seq_length: int, 
                 num_layers: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        
        # Embedding layer
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
        
        # Output layers
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.fc_out = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self.init_weights()

    def init_weights(self):
        """Initialize model weights with proper scaling."""
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
        
        # Create causal mask
        mask = self.generate_square_subsequent_mask(seq_len).to(x.device)
        
        # Embedding with scaling
        x = self.embedding(x) * math.sqrt(self.hidden_size)
        x = self.pos_enc(x)
        
        # Transformer with causal mask
        x = self.transformer(x, mask=mask)
        x = self.layer_norm(x)
        x = self.dropout(x)
        
        return self.fc_out(x)

def count_parameters(model) -> int:
    """Count trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_checkpoint(model, optimizer, scheduler, epoch: int, loss: float, path: str):
    """Save training checkpoint with all necessary information."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None,
        'loss': loss,
        'timestamp': time.time()
    }
    torch.save(checkpoint, path)
    logger.info(f"Checkpoint saved: {path}")

def load_checkpoint(model, optimizer, scheduler, path: str) -> Tuple[int, float]:
    """Load training checkpoint and restore training state."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    logger.info(f"Checkpoint loaded: {path}")
    return checkpoint['epoch'], checkpoint['loss']

class WarmupCosineScheduler:
    """Learning rate scheduler with warmup and cosine decay."""
    
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_step = 0
        
    def step(self):
        """Update learning rate based on current step."""
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

def generate_text(model, char_to_ix: Dict[str, int], ix_to_char: Dict[int, str], 
                 prompt: str = "<|user|>", max_length: int = 200, temperature: float = 0.8) -> str:
    """Generate text using the trained model."""
    model.eval()
    
    with torch.no_grad():
        # Encode prompt
        try:
            input_ids = torch.tensor([char_to_ix.get(c, 0) for c in prompt], 
                                   dtype=torch.long).unsqueeze(0).to(device)
            generated = input_ids.clone()
            
            for _ in range(max_length):
                # Use last seq_length tokens to avoid memory issues
                input_seq = generated[:, -512:] if generated.size(1) > 512 else generated
                
                outputs = model(input_seq)
                next_token_logits = outputs[0, -1, :] / temperature
                
                # Apply softmax and sample
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
                
                # Stop if we generate newline
                if next_token.item() == char_to_ix.get('\n', 0):
                    break
            
            # Decode generated text
            generated_text = ''.join([ix_to_char.get(idx.item(), '') for idx in generated[0]])
            return generated_text
            
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return "Error generating text"

def train_epoch(model, dataloader, criterion, optimizer, scheduler, epoch: int) -> Tuple[float, float]:
    """Train for one epoch with proper memory management."""
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
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            current_lr = scheduler.step()

            # Statistics (detach to prevent memory accumulation)
            total_loss += loss.detach().item() * inputs.numel()
            preds = outputs.argmax(dim=2)
            total_correct += (preds == targets).sum().detach().item()
            total_chars += targets.numel()
            
            # Memory cleanup every 50 batches
            if batch_idx % 50 == 0:
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                elif device.type == 'mps':
                    torch.mps.empty_cache()
                gc.collect()
            
            # Clear intermediate tensors
            del inputs, targets, outputs, loss, preds
            
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
    
    avg_loss = total_loss / total_chars
    accuracy = total_correct / total_chars
    
    return avg_loss, accuracy

def main():
    """Main training function with comprehensive configuration."""
    
    # Hyperparameters
    config = {
        'hidden_size': 512,
        'seq_length': 512,
        'batch_size': 16,
        'num_layers': 6,
        'nhead': 8,
        'learning_rate': 3e-4,
        'epochs': 100,
        'dropout': 0.1,
        'warmup_ratio': 0.1,  # 10% of total steps for warmup
        'weight_decay': 0.01,
        'save_every': 25,
        'generate_every': 10
    }
    
    logger.info("Starting training with configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    # Load dataset
    data_path = "oasst1_data/oasst1_train.jsonl"
    try:
        text = load_text_data(data_path)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return 1
    
    # Create vocabulary
    chars = sorted(set(text))
    vocab_size = len(chars)
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    ix_to_char = {i: ch for i, ch in enumerate(chars)}
    
    logger.info(f"Vocabulary size: {vocab_size}")
    logger.info(f"Sample characters: {chars[:20]}...")
    
    # Create dataset and dataloader
    dataset = CharDataset(text, config['seq_length'], char_to_ix)
    dataloader = DataLoader(
        dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False,
        persistent_workers=True
    )
    
    logger.info(f"Dataset size: {len(dataset):,} sequences")
    logger.info(f"Batches per epoch: {len(dataloader):,}")
    
    # Initialize model
    model = CharTransformer(
        vocab_size=vocab_size,
        hidden_size=config['hidden_size'],
        seq_length=config['seq_length'],
        num_layers=config['num_layers'],
        nhead=config['nhead'],
        dropout=config['dropout']
    ).to(device)
    
    logger.info(f"Model parameters: {count_parameters(model):,}")
    
    # Initialize training components
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config['learning_rate'], 
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.95)
    )
    
    # Learning rate scheduler
    total_steps = len(dataloader) * config['epochs']
    warmup_steps = int(total_steps * config['warmup_ratio'])
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps)
    
    logger.info(f"Total training steps: {total_steps:,}")
    logger.info(f"Warmup steps: {warmup_steps:,}")
    
    # Training loop
    logger.info("Starting training...")
    global_start_time = time.time()
    best_loss = float('inf')
    
    try:
        for epoch in range(1, config['epochs'] + 1):
            epoch_start = time.time()
            
            # Train epoch
            avg_loss, accuracy = train_epoch(model, dataloader, criterion, optimizer, scheduler, epoch)
            
            # Timing
            epoch_time = time.time() - epoch_start
            elapsed_time = time.time() - global_start_time
            current_lr = optimizer.param_groups[0]['lr']
            
            # Log progress
            logger.info(
                f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Accuracy: {accuracy*100:.2f}% | "
                f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s | Elapsed: {elapsed_time:.1f}s"
            )
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_path = "Model.pth"
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "char_to_ix": char_to_ix,
                    "ix_to_char": ix_to_char,
                    "config": config,
                    "vocab_size": vocab_size,
                    "epoch": epoch,
                    "loss": avg_loss,
                    "accuracy": accuracy
                }, best_model_path)
                logger.info(f"New best model saved: {best_model_path}")
            
            # Generate sample text
            if epoch % config['generate_every'] == 0:
                logger.info("\n--- Sample Generation ---")
                prompt = text[:20] if len(text) > 20 else text[:5]
                sample = generate_text(model, char_to_ix, ix_to_char, prompt, max_length=200)
                sample_display = sample[:300] + "..." if len(sample) > 300 else sample
                logger.info(sample_display)
                logger.info("--- End Sample ---\n")
            
            # Save checkpoint
            if epoch % config['save_every'] == 0:
                checkpoint_path = f"checkpoint_epoch_{epoch}.pth"
                save_checkpoint(model, optimizer, scheduler, epoch, avg_loss, checkpoint_path)
        
        logger.info("✅ Training completed successfully!")
        logger.info(f"Best loss: {best_loss:.4f}")
        logger.info("Model saved to 'Model.pth'")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("❌ Training interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        return 1
    finally:
        # Final cleanup
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        elif device.type == 'mps':
            torch.mps.empty_cache()
        gc.collect()

if __name__ == "__main__":
    exit(main())