# Copyright (c) 2025 Matias Nielsen. All rights reserved.
# Licensed under the Custom License below.

import os
import sys
import json
import time
import math
import logging
import traceback
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

# NEW: Mixed Precision Training
try:
    from torch.cuda.amp import GradScaler, autocast
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False

# NEW: Gradient Checkpointing
try:
    from torch.utils.checkpoint import checkpoint
    CHECKPOINT_AVAILABLE = True
except ImportError:
    CHECKPOINT_AVAILABLE = False

# Enhanced logging setup
def setup_logging():
    """Setup comprehensive logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('training_debug.log', mode='w')
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info("🔧 Debug logging initialized")
    return logger

logger = setup_logging()

# Configuration classes (enhanced with new options)
@dataclass
class ModelConfig:
    vocab_size: int = 50000
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    seq_length: int = 512
    dropout: float = 0.1
    model_type: str = "transformer"
    tokenizer_type: str = "subword"
    # NEW: Memory optimization options
    gradient_checkpointing: bool = True
    use_flash_attention: bool = False  # For future compatibility

@dataclass
class TrainingConfig:
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    max_epochs: int = 150
    warmup_ratio: float = 0.1
    save_every: int = 1000
    eval_every: int = 500
    max_grad_norm: float = 1.0
    label_smoothing: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.95
    # NEW: Mixed precision and memory optimization
    use_mixed_precision: bool = True
    amp_opt_level: str = "O1"  # O0, O1, O2, O3
    dataloader_num_workers: int = 0
    pin_memory: bool = False

@dataclass
class ModelMetadata:
    model_name: str = "transformer"
    version: str = "v1.0"
    created_at: str = ""
    last_modified: str = ""
    model_config: ModelConfig = None
    training_config: TrainingConfig = None
    dataset_info: dict = None
    performance_metrics: dict = None
    model_size_mb: float = 0.0
    total_parameters: int = 0
    trainable_parameters: int = 0
    training_time_hours: float = 0.0
    epochs_trained: int = 0
    best_loss: float = float('inf')
    best_perplexity: float = float('inf')
    hardware_used: str = ""
    pytorch_version: str = ""
    cuda_version: str = None
    notes: str = ""
    tags: list = None

# ULTRA-AGGRESSIVE memory management (enhanced)
@contextmanager
def ultra_memory_cleanup():
    """Ultra-aggressive memory cleanup with new optimizations."""
    try:
        yield
    finally:
        # Force Python garbage collection multiple times
        for _ in range(3):
            gc.collect()
        
        # CUDA cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.ipc_collect()
            # Force GPU memory release
            if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                torch.cuda.reset_peak_memory_stats()
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()

def setup_device():
    """Setup device with ultra-conservative memory management."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using device: CUDA ({torch.cuda.get_device_name()})")
        logger.info(f"CUDA Capability: {torch.cuda.get_device_capability()}")
        logger.info(f"Mixed Precision Available: {AMP_AVAILABLE}")
        
        # Much more conservative memory fraction
        torch.cuda.set_per_process_memory_fraction(0.70)  # Only use 70% of GPU memory
        torch.cuda.empty_cache()
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using device: MPS (Apple Silicon)")
        torch.mps.empty_cache()
    else:
        device = torch.device("cpu")
        logger.info("Using device: CPU")
        torch.set_num_threads(min(4, os.cpu_count() or 1))
    
    return device

device = setup_device()

class ImprovedTokenizer:
    """Improved tokenizer with better stability."""
    
    def __init__(self):
        self.vocab = {
            "<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3, 
            "<user>": 4, "<assistant>": 5, "\n": 6, " ": 7
        }
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.target_vocab_size = 5000  # Drastically reduced
        self.trained = False
    
    def train_from_text(self, text, vocab_size=None, min_freq=2):
        """Train tokenizer with much smaller vocabulary."""
        if vocab_size:
            self.target_vocab_size = min(vocab_size, 5000)  # Cap at 5000
        
        # Character and word frequency counting
        char_freq = {}
        word_freq = {}
        
        for line in text.split('\n'):
            for char in line:
                if char.isprintable() and char not in self.vocab:
                    char_freq[char] = char_freq.get(char, 0) + 1
            
            words = line.lower().split()
            for word in words:
                if word not in self.vocab:
                    word_freq[word] = word_freq.get(word, 0) + 1
        
        # Add frequent characters first
        current_id = len(self.vocab)
        sorted_chars = sorted(char_freq.items(), key=lambda x: x[1], reverse=True)
        
        for char, freq in sorted_chars:
            if freq >= min_freq and current_id < self.target_vocab_size // 2:
                self.vocab[char] = current_id
                self.id_to_token[current_id] = char
                current_id += 1
        
        # Add frequent words
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        for word, freq in sorted_words:
            if freq >= min_freq and current_id < self.target_vocab_size:
                self.vocab[word] = current_id
                self.id_to_token[current_id] = word
                current_id += 1
        
        self.trained = True
        logger.info(f"Tokenizer trained with {len(self.vocab)} tokens")
    
    def encode(self, text):
        """Encode text with fallback to character-level."""
        if not self.trained:
            raise ValueError("Tokenizer not trained")
        
        tokens = []
        words = text.split()
        
        for word in words:
            if word.lower() in self.vocab:
                tokens.append(self.vocab[word.lower()])
            else:
                for char in word:
                    if char in self.vocab:
                        tokens.append(self.vocab[char])
                    else:
                        tokens.append(self.vocab["<unk>"])
            
            if " " in self.vocab:
                tokens.append(self.vocab[" "])
        
        if tokens and tokens[-1] == self.vocab.get(" ", -1):
            tokens.pop()
        
        return tokens
    
    def decode(self, token_ids):
        """Decode with better text reconstruction."""
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                if token not in ["<pad>", "<bos>", "<eos>"]:
                    tokens.append(token)
        
        text = ""
        for token in tokens:
            if token == " ":
                text += " "
            elif len(token) == 1:
                text += token
            else:
                if text and not text.endswith(" "):
                    text += " "
                text += token
        
        return text.strip()
    
    def vocab_size(self):
        return len(self.vocab)

# NEW: Gradient Checkpointing Wrapper
class CheckpointWrapper(nn.Module):
    """Wrapper to apply gradient checkpointing to any module."""
    
    def __init__(self, module):
        super().__init__()
        self.module = module
    
    def forward(self, *args, **kwargs):
        if self.training and CHECKPOINT_AVAILABLE:
            return checkpoint(self.module, *args, **kwargs)
        else:
            return self.module(*args, **kwargs)

class MiniTransformer(nn.Module):
    """Drastically simplified transformer for ultra-low VRAM with new optimizations."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Much smaller embeddings
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        
        # Simplified positional embeddings - learnable but small
        self.pos_embeddings = nn.Parameter(torch.zeros(config.seq_length, config.hidden_size))
        
        # Minimal transformer layers with optional gradient checkpointing
        self.layers = nn.ModuleList()
        for _ in range(config.num_layers):
            layer = MiniTransformerBlock(config)
            if config.gradient_checkpointing and CHECKPOINT_AVAILABLE:
                layer = CheckpointWrapper(layer)
            self.layers.append(layer)
        
        # Output layers
        self.ln_final = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Tie embeddings to reduce parameters
        self.lm_head.weight = self.embeddings.weight
        
        # Initialize with smaller values
        self._init_weights()
    
    def _init_weights(self):
        """Ultra-conservative weight initialization."""
        nn.init.normal_(self.embeddings.weight, mean=0.0, std=0.01)  # Smaller std
        nn.init.normal_(self.pos_embeddings, mean=0.0, std=0.01)
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.zeros_(module.bias)
                nn.init.ones_(module.weight)
    
    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        token_embeddings = self.embeddings(input_ids)
        pos_embeddings = self.pos_embeddings[:seq_len].unsqueeze(0).expand(batch_size, -1, -1)
        
        hidden_states = token_embeddings + pos_embeddings
        
        # Apply layers one by one and clean up immediately
        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states)
            # Force garbage collection after each layer
            if i % 2 == 1:  # Every other layer
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        hidden_states = self.ln_final(hidden_states)
        logits = self.lm_head(hidden_states)
        
        return logits

class MiniTransformerBlock(nn.Module):
    """Ultra-simplified transformer block."""
    
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        
        # Simplified attention
        self.attn = MiniAttention(config)
        
        # Smaller MLP
        self.mlp = MiniMLP(config)
        
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        # Pre-norm with immediate cleanup
        residual = x
        x_norm = self.ln1(x)
        attn_out = self.attn(x_norm)
        del x_norm  # Immediate cleanup
        x = residual + self.dropout(attn_out)
        del residual, attn_out  # Cleanup
        
        # MLP block
        residual = x
        x_norm = self.ln2(x)
        mlp_out = self.mlp(x_norm)
        del x_norm  # Cleanup
        x = residual + self.dropout(mlp_out)
        del residual, mlp_out  # Cleanup
        
        return x

class MiniAttention(nn.Module):
    """Memory-efficient attention with immediate cleanup."""
    
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.scale = self.head_dim ** -0.5
        
        # Single projection for Q, K, V
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        
        # Register causal mask as buffer
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(config.seq_length, config.seq_length), diagonal=1).bool()
        )
    
    def forward(self, x):
        batch_size, seq_len, hidden_size = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for multi-head
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        del qkv  # Immediate cleanup
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        del q, k  # Cleanup Q and K immediately
        
        # Apply causal mask
        scores.masked_fill_(self.causal_mask[:seq_len, :seq_len], float('-inf'))
        
        # Softmax with stability
        attn_weights = F.softmax(scores, dim=-1)
        del scores  # Cleanup scores
        
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        attn_output = torch.matmul(attn_weights, v)
        del attn_weights, v  # Cleanup
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        output = self.out_proj(attn_output)
        del attn_output  # Final cleanup
        
        return output

class MiniMLP(nn.Module):
    """Smaller MLP block."""
    
    def __init__(self, config):
        super().__init__()
        # Smaller intermediate size
        intermediate_size = max(config.hidden_size * 2, 128)  # Much smaller
        
        self.fc1 = nn.Linear(config.hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class UltraDataset(Dataset):
    """Ultra-efficient dataset with minimal memory usage."""
    
    def __init__(self, texts: List[str], tokenizer, seq_length: int, max_sequences: int = 5000):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.pad_token_id = tokenizer.vocab.get("<pad>", 0)
        
        logger.info(f"Creating ultra-efficient dataset with seq_length={seq_length}...")
        
        # Process in smaller chunks to save memory
        self.sequences = []
        chunk_size = 100
        
        for i in range(0, len(texts), chunk_size):
            if len(self.sequences) >= max_sequences:
                break
                
            chunk = texts[i:i+chunk_size]
            
            for text in chunk:
                if len(self.sequences) >= max_sequences:
                    break
                    
                if not text or len(text.strip()) < 10:
                    continue
                
                try:
                    tokens = tokenizer.encode(text.strip())
                    if len(tokens) < 5:
                        continue
                    
                    # Add special tokens
                    bos_id = tokenizer.vocab.get("<bos>", 2)
                    eos_id = tokenizer.vocab.get("<eos>", 3)
                    full_sequence = [bos_id] + tokens[:seq_length-2] + [eos_id]
                    
                    # Pad to exact length
                    if len(full_sequence) < seq_length + 1:
                        full_sequence.extend([self.pad_token_id] * (seq_length + 1 - len(full_sequence)))
                    else:
                        full_sequence = full_sequence[:seq_length + 1]
                    
                    self.sequences.append(full_sequence)
                    
                except Exception as e:
                    continue
            
            # Cleanup chunk
            del chunk
            if i % 500 == 0:
                gc.collect()
        
        if not self.sequences:
            raise ValueError("No valid sequences created!")
        
        logger.info(f"Created {len(self.sequences):,} sequences")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        input_ids = torch.tensor(sequence[:-1], dtype=torch.long)
        target_ids = torch.tensor(sequence[1:], dtype=torch.long)
        return input_ids, target_ids

class ModelManager:
    """Ultra-lightweight model manager."""
    
    def __init__(self, save_dir):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
    
    def save_model(self, model, tokenizer, metadata, optimizer=None, scheduler=None, force_cpu_save=True):
        """Save model with ultra-aggressive memory management."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_id = f"model_{timestamp}"
        model_path = self.save_dir / model_id
        model_path.mkdir(exist_ok=True)
        
        try:
            # ALWAYS move to CPU and clean up GPU memory
            with ultra_memory_cleanup():
                # Move model to CPU temporarily
                original_device = next(model.parameters()).device
                model.cpu()
                
                # Save state dict with CPU tensors
                state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                torch.save(state_dict, model_path / "model.pth")
                del state_dict
                
                # Move model back
                model.to(original_device)
            
            # Save tokenizer data
            tokenizer_data = {
                'vocab': tokenizer.vocab,
                'id_to_token': tokenizer.id_to_token,
                'vocab_size': tokenizer.vocab_size()
            }
            with open(model_path / "tokenizer.json", 'w') as f:
                json.dump(tokenizer_data, f, indent=2)
            
            # Save metadata
            if hasattr(metadata, '__dict__'):
                metadata_dict = asdict(metadata) if hasattr(metadata, '__dataclass_fields__') else metadata.__dict__
            else:
                metadata_dict = metadata
                
            with open(model_path / "metadata.json", 'w') as f:
                json.dump(metadata_dict, f, indent=2, default=str)
            
            logger.info(f"Model saved to: {model_path}")
            return model_id
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return None

def get_ultra_low_vram_config():
    """Ultra-conservative configuration for minimal VRAM usage with new optimizations."""
    if device.type == 'cuda':
        model_config = ModelConfig(
            vocab_size=20000,   # Extremely small vocabulary
            hidden_size=2048,   # Very small hidden size
            num_layers=24,      # Minimal layers
            num_heads=16,       # Fewer heads
            seq_length=1024,     # Much shorter sequences
            dropout=0.1,
            gradient_checkpointing=True  # NEW: Enable gradient checkpointing
        )
        batch_size = 1      # Single batch
        max_samples = 80000  # Very small dataset
    elif device.type == 'mps':
        model_config = ModelConfig(
            vocab_size=1500,
            hidden_size=96,
            num_layers=2,
            num_heads=4,
            seq_length=48,
            dropout=0.1,
            gradient_checkpointing=True
        )
        batch_size = 1
        max_samples = 1000
    else:  # CPU
        model_config = ModelConfig(
            vocab_size=1000,
            hidden_size=64,
            num_layers=2,
            num_heads=2,
            seq_length=32,
            dropout=0.1,
            gradient_checkpointing=False  # Don't use on CPU
        )
        batch_size = 2
        max_samples = 500
    
    training_config = TrainingConfig(
        learning_rate=1e-3,  # Higher LR for faster convergence
        weight_decay=0.01,
        batch_size=batch_size,
        gradient_accumulation_steps=16,  # Large accumulation to maintain effective batch size
        max_epochs=20,  # Fewer epochs
        warmup_ratio=0.05,  # Less warmup
        max_grad_norm=0.5,   # Smaller gradient norm
        use_mixed_precision=AMP_AVAILABLE and device.type == 'cuda',  # NEW: Enable mixed precision
        dataloader_num_workers=0,  # NEW: No multiprocessing
        pin_memory=False  # NEW: Don't pin memory
    )
    
    return model_config, training_config, max_samples

def load_and_process_data(data_path: str, max_samples: Optional[int] = None) -> List[str]:
    """Load data with ultra-minimal processing."""
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    logger.info(f"Loading data from: {data_path}")
    
    texts = []
    processed_count = 0
    
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if max_samples and processed_count >= max_samples:
                    break
                
                try:
                    line = line.strip()
                    if not line:
                        continue
                    
                    record = json.loads(line)
                    
                    if record.get("deleted", False) or record.get("lang") != "en":
                        continue
                    
                    text = record.get("text", "").strip()
                    if not text or len(text.split()) < 3:
                        continue
                    
                    # Much more aggressive length filtering
                    word_count = len(text.split())
                    if word_count < 3 or word_count > 50:  # Much shorter texts
                        continue
                    
                    role = record.get("role", "").lower()
                    if role == "prompter":
                        formatted_text = f"<user> {text}"
                    elif role == "assistant":
                        formatted_text = f"<assistant> {text}"
                    else:
                        formatted_text = text
                    
                    texts.append(formatted_text)
                    processed_count += 1
                    
                except (json.JSONDecodeError, KeyError):
                    continue
        
        # Immediate cleanup
        gc.collect()
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise
    
    logger.info(f"Loaded {len(texts):,} texts")
    return texts

# NEW: Enhanced training loop with mixed precision
def train_epoch_ultra_efficient(model, dataloader, criterion, optimizer, scheduler, epoch, 
                               gradient_accumulation_steps=1, max_grad_norm=1.0, 
                               use_mixed_precision=False, scaler=None):
    """Ultra-efficient training loop with mixed precision and aggressive memory management."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    num_batches = 0
    accumulation_count = 0
    
    optimizer.zero_grad()
    
    logger.info(f"Starting ultra-efficient epoch {epoch} with {len(dataloader)} batches")
    logger.info(f"Mixed precision: {use_mixed_precision}")
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        try:
            # Ultra-aggressive memory cleanup every 5 batches
            if batch_idx > 0 and batch_idx % 5 == 0:
                with ultra_memory_cleanup():
                    pass
            
            # Skip empty batches
            if inputs.numel() == 0 or targets.numel() == 0:
                continue
                
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            # Validate inputs
            if torch.isnan(inputs).any() or torch.isinf(inputs).any():
                del inputs, targets
                continue
            
            # Forward pass with optional mixed precision
            if use_mixed_precision and scaler is not None:
                with autocast():
                    logits = model(inputs)
                    loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            else:
                logits = model(inputs)
                loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            
            # Check for invalid loss
            if torch.isnan(loss) or torch.isinf(loss):
                del inputs, targets, logits, loss
                continue
            
            # Scale loss and backward
            scaled_loss = loss / gradient_accumulation_steps
            
            if use_mixed_precision and scaler is not None:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            
            # Immediate cleanup of forward pass tensors
            with torch.no_grad():
                predictions = torch.argmax(logits, dim=-1)
                pad_mask = (targets != criterion.ignore_index)
                correct = ((predictions == targets) & pad_mask).sum().item()
                valid_tokens = pad_mask.sum().item()
                
                total_correct += correct
                total_tokens += valid_tokens
                total_loss += loss.item()
                num_batches += 1
            
            # Clean up all tensors immediately
            del logits, predictions, pad_mask, scaled_loss, inputs, targets
            
            accumulation_count += 1
            
            # Gradient step
            if accumulation_count >= gradient_accumulation_steps:
                if use_mixed_precision and scaler is not None:
                    # Mixed precision gradient step
                    if max_grad_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Regular gradient step
                    if max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                
                scheduler.step()
                optimizer.zero_grad()
                accumulation_count = 0
                
                # Extra cleanup after optimizer step
                with ultra_memory_cleanup():
                    pass
            
            # Logging every 5 batches
            if batch_idx % 5 == 0 and num_batches > 0:
                current_loss = total_loss / num_batches
                current_acc = total_correct / max(total_tokens, 1)
                current_lr = optimizer.param_groups[0]['lr']
                
                logger.info(f"Epoch {epoch} | Batch {batch_idx} | "
                           f"Loss: {current_loss:.4f} | Acc: {current_acc:.3f} | "
                           f"LR: {current_lr:.6f}")
            
            del loss  # Final cleanup
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning(f"OOM at batch {batch_idx}, skipping...")
                optimizer.zero_grad()
                
                # Ultra-aggressive cleanup
                for var_name in ['inputs', 'targets', 'logits', 'loss', 'predictions']:
                    if var_name in locals():
                        del locals()[var_name]
                
                with ultra_memory_cleanup():
                    pass
                continue
            else:
                raise e
        except Exception as e:
            logger.warning(f"Error at batch {batch_idx}: {e}")
            continue
    
    # Final gradient step if needed
    if accumulation_count > 0:
        if use_mixed_precision and scaler is not None:
            if max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        optimizer.zero_grad()
    
    if num_batches == 0:
        return float('inf'), 0.0
    
    avg_loss = total_loss / num_batches
    avg_acc = total_correct / max(total_tokens, 1)
    
    return avg_loss, avg_acc

class ImprovedScheduler:
    """Ultra-simple scheduler."""
    
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.1):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lr = optimizer.param_groups[0]['lr']
        self.min_lr = self.base_lr * min_lr_ratio
        self.current_step = 0
    
    def step(self):
        self.current_step += 1
        
        if self.current_step <= self.warmup_steps:
            lr = self.base_lr * self.current_step / self.warmup_steps
        else:
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr

def get_memory_usage():
    """Get current memory usage with enhanced reporting."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        return f"GPU: {allocated:.2f}GB allocated, {cached:.2f}GB cached, {max_allocated:.2f}GB peak"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        allocated = torch.mps.current_allocated_memory() / 1024**3
        return f"MPS: {allocated:.2f}GB allocated"
    else:
        return "CPU mode"

def count_parameters(model):
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def evaluate_model_minimal(model, dataloader, criterion, max_batches=3, use_mixed_precision=False):
    """Ultra-minimal evaluation with optional mixed precision."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    num_batches = 0
    
    try:
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                if batch_idx >= max_batches:
                    break
                
                try:
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    
                    # Forward pass with optional mixed precision
                    if use_mixed_precision and AMP_AVAILABLE:
                        with autocast():
                            logits = model(inputs)
                            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
                    else:
                        logits = model(inputs)
                        loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
                    
                    # Quick accuracy calculation
                    predictions = torch.argmax(logits, dim=-1)
                    pad_mask = (targets != criterion.ignore_index)
                    correct = ((predictions == targets) & pad_mask).sum().item()
                    valid_tokens = pad_mask.sum().item()
                    
                    total_loss += loss.item()
                    total_correct += correct
                    total_tokens += valid_tokens
                    num_batches += 1
                    
                    # Immediate cleanup
                    del logits, loss, predictions, pad_mask, inputs, targets
                    
                except Exception as e:
                    continue
                
                # Memory cleanup after each batch
                with ultra_memory_cleanup():
                    pass
    
    finally:
        model.train()
    
    if num_batches == 0:
        return {'avg_loss': float('inf'), 'accuracy': 0.0, 'perplexity': float('inf')}
    
    avg_loss = total_loss / num_batches
    accuracy = total_correct / max(total_tokens, 1)
    perplexity = math.exp(min(avg_loss, 10))
    
    return {
        'avg_loss': avg_loss,
        'accuracy': accuracy,
        'perplexity': perplexity
    }

def generate_sample_text_minimal(model, tokenizer, prompt="<user> Hello", max_length=20, use_mixed_precision=False):
    """Minimal text generation with optional mixed precision."""
    model.eval()
    
    try:
        with torch.no_grad():
            input_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
            generated = input_ids.clone()
            
            for _ in range(max_length):
                if generated.size(1) >= model.config.seq_length:
                    break
                
                if use_mixed_precision and AMP_AVAILABLE:
                    with autocast():
                        logits = model(generated)
                else:
                    logits = model(generated)
                
                next_token_logits = logits[0, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1)
                
                generated = torch.cat([generated, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
                
                # Stop on EOS
                if next_token.item() == tokenizer.vocab.get("<eos>", -1):
                    break
                
                # Cleanup
                del logits, next_token_logits
            
            response_ids = generated[0][input_ids.size(1):].tolist()
            response = tokenizer.decode(response_ids)
            
            # Cleanup
            del input_ids, generated
            
            return response.strip()
    
    except Exception as e:
        return "Generation failed"
    finally:
        model.train()

def check_environment():
    """Enhanced environment check with optimization features."""
    logger.info("🔍 Checking environment...")
    
    try:
        logger.info(f"Python version: {sys.version}")
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        logger.info(f"Mixed Precision available: {AMP_AVAILABLE}")
        logger.info(f"Gradient Checkpointing available: {CHECKPOINT_AVAILABLE}")
        
        if torch.cuda.is_available():
            logger.info(f"CUDA device count: {torch.cuda.device_count()}")
            props = torch.cuda.get_device_properties(0)
            logger.info(f"GPU: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
            
            # Check for Tensor Core support (for mixed precision)
            major, minor = props.major, props.minor
            if major >= 7 or (major == 6 and minor >= 0):
                logger.info("✅ Tensor Cores supported (good for mixed precision)")
            else:
                logger.info("⚠️ Limited Tensor Core support")
        
        # Check for required files
        required_files = ["oasst1_data/oasst1_train.jsonl"]
        for file_path in required_files:
            path = Path(file_path)
            exists = path.exists()
            logger.info(f"Required file {file_path}: {'✅ EXISTS' if exists else '❌ MISSING'}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Environment check failed: {e}")
        return False

def validate_training_setup():
    """Validate required files exist."""
    required_files = ["oasst1_data/oasst1_train.jsonl"]
    
    for file_path in required_files:
        if not Path(file_path).exists():
            logger.error(f"Missing required file: {file_path}")
            return False
    
    return True

# NEW: Additional memory optimization utilities
def optimize_cuda_settings():
    """Apply additional CUDA optimizations."""
    if torch.cuda.is_available():
        # Enable memory efficiency optimizations
        torch.backends.cudnn.benchmark = False  # Disable for consistent memory usage
        torch.backends.cudnn.deterministic = True  # For reproducibility
        
        # Set memory growth to avoid fragmentation
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        
        logger.info("CUDA optimizations applied")

def create_optimized_dataloader(dataset, batch_size, training_config, shuffle=True):
    """Create memory-optimized dataloader."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=training_config.dataloader_num_workers,
        pin_memory=training_config.pin_memory,
        drop_last=True,
        persistent_workers=False,
        prefetch_factor=2 if training_config.dataloader_num_workers > 0 else 2
    )

def main():
    """Ultra-low VRAM training main function with enhanced optimizations."""
    logger.info("🚀 Starting ULTRA LOW VRAM Training with Enhanced Optimizations")
    logger.info("=" * 60)
    
    # Environment check
    if not check_environment() or not validate_training_setup():
        return 1
    
    # Apply CUDA optimizations
    optimize_cuda_settings()
    
    # Ultra-conservative configuration
    model_config, training_config, max_samples = get_ultra_low_vram_config()
    
    logger.info(f"Ultra-low VRAM Configuration:")
    logger.info(f"  Model: {model_config.hidden_size}x{model_config.num_layers}")
    logger.info(f"  Vocab size: {model_config.vocab_size}")
    logger.info(f"  Sequence length: {model_config.seq_length}")
    logger.info(f"  Batch size: {training_config.batch_size}")
    logger.info(f"  Gradient accumulation: {training_config.gradient_accumulation_steps}")
    logger.info(f"  Mixed precision: {training_config.use_mixed_precision}")
    logger.info(f"  Gradient checkpointing: {model_config.gradient_checkpointing}")
    logger.info(f"  Max samples: {max_samples}")
    
    model_manager = ModelManager("models")
    
    # Initialize mixed precision scaler
    scaler = None
    if training_config.use_mixed_precision and AMP_AVAILABLE:
        scaler = GradScaler()
        logger.info("Mixed precision scaler initialized")
    
    try:
        # Load minimal data
        logger.info("📚 Loading minimal dataset...")
        texts = load_and_process_data("oasst1_data/oasst1_train.jsonl", max_samples)
        
        if len(texts) == 0:
            raise ValueError("No training data loaded!")
        
        # Create tiny tokenizer
        logger.info("🔤 Training ultra-small tokenizer...")
        tokenizer = ImprovedTokenizer()
        
        # Use even smaller sample for tokenizer
        sample_texts = texts[:min(200, len(texts))]
        sample_text = "\n".join(sample_texts)
        
        tokenizer.train_from_text(sample_text, vocab_size=model_config.vocab_size, min_freq=1)
        actual_vocab_size = tokenizer.vocab_size()
        model_config.vocab_size = actual_vocab_size
        
        logger.info(f"Tokenizer: {actual_vocab_size:,} tokens")
        
        # Clean up sample data
        del sample_texts, sample_text
        gc.collect()
        
        # Create ultra-efficient dataset
        logger.info("📦 Creating ultra-efficient dataset...")
        dataset = UltraDataset(texts, tokenizer, model_config.seq_length, max_sequences=len(texts))
        
        # Ultra-small optimized dataloaders
        train_dataloader = create_optimized_dataloader(
            dataset, training_config.batch_size, training_config, shuffle=True
        )
        
        # Minimal eval dataset
        eval_size = min(50, len(dataset) // 10)
        eval_indices = list(range(0, len(dataset), len(dataset) // eval_size))[:eval_size]
        eval_dataset = torch.utils.data.Subset(dataset, eval_indices)
        eval_dataloader = create_optimized_dataloader(
            eval_dataset, training_config.batch_size, training_config, shuffle=False
        )
        
        logger.info(f"Training: {len(dataset):,} sequences")
        logger.info(f"Evaluation: {len(eval_dataset):,} sequences")
        
        # Create ultra-small model
        logger.info("🧠 Creating ultra-small model...")
        with ultra_memory_cleanup():
            model = MiniTransformer(model_config)
            model = model.to(device)
        
        total_params, trainable_params = count_parameters(model)
        model_size_mb = total_params * 4 / 1024**2
        
        logger.info(f"Model parameters: {total_params:,} (~{model_size_mb:.1f}MB)")
        logger.info(f"Gradient checkpointing: {model_config.gradient_checkpointing}")
        
        # Ultra-efficient optimizer
        optimizer = optim.AdamW(
            model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
            eps=1e-8,
            betas=(training_config.beta1, training_config.beta2)
        )
        
        pad_token_id = tokenizer.vocab.get("<pad>", 0)
        criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id, label_smoothing=training_config.label_smoothing)
        
        total_steps = len(train_dataloader) * training_config.max_epochs // training_config.gradient_accumulation_steps
        warmup_steps = int(total_steps * training_config.warmup_ratio)
        
        scheduler = ImprovedScheduler(optimizer, warmup_steps, total_steps)
        
        logger.info(f"Training steps: {total_steps:,}")
        logger.info(f"Warmup steps: {warmup_steps:,}")
        logger.info(f"Memory before training: {get_memory_usage()}")
        
        # Ultra-efficient training loop
        logger.info("🚀 Starting ultra-efficient training with enhanced optimizations...")
        training_start = time.time()
        best_loss = float('inf')
        models_saved = 0
        
        for epoch in range(1, training_config.max_epochs + 1):
            epoch_start = time.time()
            
            logger.info(f"=== Epoch {epoch}/{training_config.max_epochs} ===")
            
            try:
                # Ultra-efficient training with enhanced features
                train_loss, train_acc = train_epoch_ultra_efficient(
                    model, train_dataloader, criterion, optimizer, scheduler, epoch,
                    training_config.gradient_accumulation_steps,
                    training_config.max_grad_norm,
                    training_config.use_mixed_precision,
                    scaler
                )
                
                # Check for invalid loss
                if math.isnan(train_loss) or math.isinf(train_loss):
                    logger.error(f"Invalid loss: {train_loss}, skipping epoch")
                    continue
                
                perplexity = math.exp(min(train_loss, 10))
                epoch_time = time.time() - epoch_start
                
                logger.info(f"Training - Loss: {train_loss:.4f}, Acc: {train_acc:.3f}, PPL: {perplexity:.2f}, Time: {epoch_time:.1f}s")
                
                # Minimal evaluation every 5 epochs
                eval_results = None
                if epoch % 5 == 0 or epoch == 1:
                    with ultra_memory_cleanup():
                        eval_results = evaluate_model_minimal(
                            model, eval_dataloader, criterion, max_batches=3,
                            use_mixed_precision=training_config.use_mixed_precision
                        )
                    logger.info(f"Eval - Loss: {eval_results['avg_loss']:.4f}, Acc: {eval_results['accuracy']:.3f}")
                
                # Sample generation every 10 epochs
                if epoch % 10 == 0:
                    with ultra_memory_cleanup():
                        sample = generate_sample_text_minimal(
                            model, tokenizer, 
                            use_mixed_precision=training_config.use_mixed_precision
                        )
                        logger.info(f"Sample: {sample}")
                
                # Track best loss
                is_best = train_loss < best_loss
                if is_best:
                    best_loss = train_loss
                    logger.info(f"🏆 New best loss: {best_loss:.4f}")
                
                # Save model less frequently (every 10 epochs or if best)
                if is_best or epoch % 10 == 0 or epoch == training_config.max_epochs:
                    performance_metrics = {
                        "train_loss": float(train_loss),
                        "train_accuracy": float(train_acc),
                        "epoch": int(epoch),
                        "is_best": is_best,
                        "mixed_precision_used": training_config.use_mixed_precision,
                        "gradient_checkpointing_used": model_config.gradient_checkpointing,
                    }
                    
                    if eval_results:
                        performance_metrics.update({
                            "eval_loss": float(eval_results['avg_loss']),
                            "eval_accuracy": float(eval_results['accuracy']),
                        })
                    
                    metadata = ModelMetadata(
                        model_name="Ultra_Low_VRAM_Model_Enhanced",
                        version=f"v2.0_epoch_{epoch}",
                        created_at=datetime.now().isoformat(),
                        model_config=model_config,
                        training_config=training_config,
                        performance_metrics=performance_metrics,
                        model_size_mb=float(model_size_mb),
                        total_parameters=int(total_params),
                        epochs_trained=int(epoch),
                        best_loss=float(best_loss),
                        hardware_used=device.type.upper(),
                        pytorch_version=torch.__version__,
                        notes=f"Enhanced ultra-low VRAM training epoch {epoch} with mixed precision and gradient checkpointing",
                        tags=["ultra_low_vram", "enhanced", f"epoch_{epoch}"] + (["best"] if is_best else []) + 
                             (["mixed_precision"] if training_config.use_mixed_precision else []) +
                             (["gradient_checkpointing"] if model_config.gradient_checkpointing else [])
                    )
                    
                    try:
                        with ultra_memory_cleanup():
                            model_id = model_manager.save_model(model, tokenizer, metadata)
                            if model_id:
                                models_saved += 1
                                logger.info(f"💾 Model saved: {model_id}")
                    except Exception as save_error:
                        logger.error(f"Save failed: {save_error}")
                
                logger.info(f"Memory: {get_memory_usage()}")
                
                # Ultra-aggressive cleanup between epochs
                with ultra_memory_cleanup():
                    pass
                
            except Exception as e:
                logger.error(f"Error in epoch {epoch}: {e}")
                
                if "out of memory" in str(e).lower():
                    logger.info("OOM detected, attempting recovery...")
                    
                    optimizer.zero_grad()
                    if scaler is not None:
                        scaler = GradScaler()  # Reset scaler
                    
                    with ultra_memory_cleanup():
                        pass
                    
                    # Further reduce batch size if needed
                    if training_config.batch_size > 1:
                        training_config.batch_size = 1
                        training_config.gradient_accumulation_steps *= 2
                        logger.info("Reduced to batch size 1")
                        
                        # Recreate dataloader
                        train_dataloader = create_optimized_dataloader(
                            dataset, 1, training_config, shuffle=True
                        )
                    
                    continue
                else:
                    raise e
        
        # Training completion
        total_time = time.time() - training_start
        
        logger.info("=" * 60)
        logger.info("✅ Enhanced ultra-low VRAM training completed!")
        logger.info(f"Best loss: {best_loss:.4f}")
        logger.info(f"Models saved: {models_saved}")
        logger.info(f"Training time: {total_time/60:.1f} minutes")
        logger.info(f"Final memory: {get_memory_usage()}")
        logger.info(f"Mixed precision used: {training_config.use_mixed_precision}")
        logger.info(f"Gradient checkpointing used: {model_config.gradient_checkpointing}")
        
        return 0 if models_saved > 0 else 1
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return 1
    finally:
        with ultra_memory_cleanup():
            pass

if __name__ == "__main__":
    exit(main())