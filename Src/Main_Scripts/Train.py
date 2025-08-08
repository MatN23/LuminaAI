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

def check_environment():
    """Check environment and dependencies."""
    logger.info("🔍 Checking environment...")
    
    try:
        # Python version
        logger.info(f"Python version: {sys.version}")
        
        # PyTorch version and setup
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"CUDA device count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                logger.info(f"GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
        
        # MPS availability
        if hasattr(torch.backends, 'mps'):
            logger.info(f"MPS available: {torch.backends.mps.is_available()}")
            logger.info(f"MPS built: {torch.backends.mps.is_built()}")
        
        # Check current directory and files
        cwd = Path.cwd()
        logger.info(f"Current directory: {cwd}")
        
        # Check for required files
        required_files = [
            "oasst1_data/oasst1_train.jsonl"
        ]
        
        for file_path in required_files:
            path = Path(file_path)
            exists = path.exists()
            logger.info(f"Required file {file_path}: {'✅ EXISTS' if exists else '❌ MISSING'}")
            if exists and path.is_file():
                size = path.stat().st_size / 1024**2
                logger.info(f"  Size: {size:.2f} MB")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Environment check failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

# Configuration classes
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

@dataclass
class TrainingConfig:
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    max_epochs: int = 10
    warmup_ratio: float = 0.1
    save_every: int = 1000
    eval_every: int = 500
    max_grad_norm: float = 1.0
    label_smoothing: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.95

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

class ImprovedTokenizer:
    """Improved tokenizer with better stability."""
    
    def __init__(self):
        self.vocab = {
            "<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3, 
            "<user>": 4, "<assistant>": 5, "\n": 6, " ": 7
        }
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.target_vocab_size = 10000
        self.trained = False
    
    def train_from_text(self, text, vocab_size=None, min_freq=2):
        """Train tokenizer with improved frequency analysis."""
        if vocab_size:
            self.target_vocab_size = vocab_size
        
        # Character and word frequency counting
        char_freq = {}
        word_freq = {}
        
        for line in text.split('\n'):
            # Process characters
            for char in line:
                if char.isprintable() and char not in self.vocab:
                    char_freq[char] = char_freq.get(char, 0) + 1
            
            # Process words
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
                # Character-level fallback
                for char in word:
                    if char in self.vocab:
                        tokens.append(self.vocab[char])
                    else:
                        tokens.append(self.vocab["<unk>"])
            
            # Add space token
            if " " in self.vocab:
                tokens.append(self.vocab[" "])
        
        # Remove trailing space
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
        
        # Reconstruct text
        text = ""
        for token in tokens:
            if token == " ":
                text += " "
            elif len(token) == 1:  # Character
                text += token
            else:  # Word
                if text and not text.endswith(" "):
                    text += " "
                text += token
        
        return text.strip()
    
    def vocab_size(self):
        return len(self.vocab)

class StableTransformer(nn.Module):
    """Improved transformer with better numerical stability."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Embeddings with proper scaling
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.pos_embeddings = nn.Embedding(config.seq_length, config.hidden_size)
        
        # Input normalization
        self.input_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        
        # Transformer layers with prenorm
        self.layers = nn.ModuleList()
        for _ in range(config.num_layers):
            layer = TransformerBlock(config)
            self.layers.append(layer)
        
        # Output layers
        self.output_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.dropout = nn.Dropout(config.dropout)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Scale embeddings
        nn.init.normal_(self.embeddings.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embeddings.weight, mean=0.0, std=0.02)
        
        # Tie embeddings and output weights for better performance
        self.lm_head.weight = self.embeddings.weight
    
    def _init_weights(self, module):
        """Improved weight initialization."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)
    
    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Create position IDs
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_embeddings = self.embeddings(input_ids)
        position_embeddings = self.pos_embeddings(position_ids)
        
        # Combine and normalize embeddings
        hidden_states = token_embeddings + position_embeddings
        hidden_states = self.input_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        # Create causal attention mask
        if attention_mask is None:
            attention_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=device), 
                diagonal=1
            ).bool()
        
        # Apply transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        
        # Output normalization and projection
        hidden_states = self.output_norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        return logits

class TransformerBlock(nn.Module):
    """Transformer block with pre-normalization and residual connections."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Pre-normalization layers
        self.ln_1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.ln_2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        
        # Attention
        self.attn = MultiHeadAttention(config)
        
        # MLP
        self.mlp = MLP(config)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x, attention_mask=None):
        # Pre-norm attention
        residual = x
        x = self.ln_1(x)
        attn_output = self.attn(x, attention_mask)
        x = residual + self.dropout(attn_output)
        
        # Pre-norm MLP
        residual = x
        x = self.ln_2(x)
        mlp_output = self.mlp(x)
        x = residual + self.dropout(mlp_output)
        
        return x

class MultiHeadAttention(nn.Module):
    """Stable multi-head attention implementation."""
    
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.scale = self.head_dim ** -0.5
        
        assert config.hidden_size % config.num_heads == 0
        
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x, attention_mask=None):
        batch_size, seq_len, hidden_size = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2), qkv)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply causal mask
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # Softmax with numerical stability
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0, posinf=0.0, neginf=0.0)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        output = self.out_proj(attn_output)
        
        return output

class MLP(nn.Module):
    """MLP block with GELU activation."""
    
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, 4 * config.hidden_size)
        self.fc2 = nn.Linear(4 * config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class ModelManager:
    """Model manager for saving and loading models."""
    
    def __init__(self, save_dir):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
    
    def save_model(self, model, tokenizer, metadata, optimizer=None, scheduler=None, force_cpu_save=False):
        """Save model with proper error handling."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_id = f"model_{timestamp}"
        model_path = self.save_dir / model_id
        model_path.mkdir(exist_ok=True)
        
        try:
            # Save model state dict
            if force_cpu_save:
                model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            else:
                model_state = model.state_dict()
            
            torch.save(model_state, model_path / "model.pth")
            
            # Save tokenizer
            tokenizer_data = {
                'vocab': tokenizer.vocab,
                'id_to_token': tokenizer.id_to_token,
                'vocab_size': tokenizer.vocab_size()
            }
            with open(model_path / "tokenizer.json", 'w') as f:
                json.dump(tokenizer_data, f, indent=2)
            
            # Save metadata
            metadata_dict = asdict(metadata) if hasattr(metadata, '__dict__') else metadata.__dict__
            with open(model_path / "metadata.json", 'w') as f:
                json.dump(metadata_dict, f, indent=2, default=str)
            
            logger.info(f"Model saved to: {model_path}")
            return model_id
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return None
    
    def print_model_summary(self):
        """Print summary of saved models."""
        models = list(self.save_dir.glob("model_*"))
        logger.info(f"Found {len(models)} saved models in {self.save_dir}")
        for model_path in sorted(models):
            logger.info(f"  - {model_path.name}")

@contextmanager
def memory_cleanup():
    """Context manager for memory cleanup."""
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()

def get_memory_usage():
    """Get current memory usage."""
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
    """Setup device with proper memory management."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using device: CUDA ({torch.cuda.get_device_name()})")
        torch.cuda.set_per_process_memory_fraction(0.8)
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

class StableDataset(Dataset):
    """Improved dataset with better validation."""
    
    def __init__(self, texts: List[str], tokenizer, seq_length: int, max_sequences: int = 10000):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        
        logger.info(f"Creating stable dataset with seq_length={seq_length}...")
        
        vocab_size = tokenizer.vocab_size()
        pad_token_id = tokenizer.vocab.get("<pad>", 0)
        bos_token_id = tokenizer.vocab.get("<bos>", 2)
        eos_token_id = tokenizer.vocab.get("<eos>", 3)
        
        self.sequences = []
        
        for text_idx, text in enumerate(texts[:max_sequences // 3]):
            if not text or not text.strip():
                continue
            
            try:
                # Add BOS token
                tokens = [bos_token_id] + tokenizer.encode(text.strip()) + [eos_token_id]
                
                if len(tokens) < 10:  # Too short
                    continue
                
                # Validate tokens
                valid_tokens = []
                for token in tokens:
                    if 0 <= token < vocab_size:
                        valid_tokens.append(token)
                    else:
                        valid_tokens.append(tokenizer.vocab.get("<unk>", 1))
                
                tokens = valid_tokens
                
                # Create sequences with sliding window
                step_size = seq_length // 2
                for start_pos in range(0, len(tokens) - seq_length, step_size):
                    if start_pos + seq_length + 1 <= len(tokens):
                        sequence = tokens[start_pos:start_pos + seq_length + 1]
                        
                        if len(sequence) == seq_length + 1:
                            self.sequences.append(sequence)
                            
                            if len(self.sequences) >= max_sequences:
                                break
                
                if len(self.sequences) >= max_sequences:
                    break
                    
            except Exception as e:
                logger.warning(f"Error processing text {text_idx}: {e}")
                continue
        
        if not self.sequences:
            raise ValueError("No valid sequences created!")
        
        self.pad_token_id = pad_token_id
        self.vocab_size = vocab_size
        
        logger.info(f"Created {len(self.sequences):,} sequences")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        
        input_ids = torch.tensor(sequence[:-1], dtype=torch.long)
        target_ids = torch.tensor(sequence[1:], dtype=torch.long)
        
        return input_ids, target_ids

def load_and_process_data(data_path: str, max_samples: Optional[int] = None) -> List[str]:
    """Load OASST1 data with better processing."""
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
                    
                    if record.get("deleted", False):
                        continue
                    
                    if record.get("lang") != "en":
                        continue
                    
                    text = record.get("text", "").strip()
                    if not text:
                        continue
                    
                    # Better length filtering
                    word_count = len(text.split())
                    if word_count < 5 or word_count > 150:
                        continue
                    
                    # Add role formatting
                    role = record.get("role", "").lower()
                    if role == "prompter":
                        formatted_text = f"<user> {text}"
                    elif role == "assistant":
                        formatted_text = f"<assistant> {text}"
                    else:
                        formatted_text = text
                    
                    texts.append(formatted_text)
                    processed_count += 1
                    
                    if processed_count % 1000 == 0:
                        logger.info(f"Processed {processed_count:,} samples...")
                    
                except (json.JSONDecodeError, KeyError):
                    continue
    
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise
    
    logger.info(f"Loaded {len(texts):,} texts")
    return texts

class ImprovedScheduler:
    """Improved learning rate scheduler with warmup and cosine decay."""
    
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
            # Linear warmup
            lr = self.base_lr * self.current_step / self.warmup_steps
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr

def count_parameters(model):
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def train_epoch(model, dataloader, criterion, optimizer, scheduler, epoch, 
                gradient_accumulation_steps=1, max_grad_norm=1.0):
    """Improved training loop."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    num_batches = 0
    
    optimizer.zero_grad()
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        try:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            # Forward pass
            with torch.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                logits = model(inputs)
                loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            
            # Check for invalid loss
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"Invalid loss at batch {batch_idx}: {loss.item()}")
                optimizer.zero_grad()
                continue
            
            # Scale loss
            loss = loss / gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # Gradient clipping
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            # Calculate accuracy
            with torch.no_grad():
                predictions = torch.argmax(logits, dim=-1)
                correct = (predictions == targets).sum().item()
                total_correct += correct
                total_tokens += targets.numel()
            
            # Statistics
            total_loss += loss.item() * gradient_accumulation_steps
            num_batches += 1
            
            # Logging
            if batch_idx % 50 == 0 and batch_idx > 0:
                current_loss = total_loss / num_batches
                current_acc = total_correct / max(total_tokens, 1)
                logger.info(f"Epoch {epoch} | Batch {batch_idx} | Loss: {current_loss:.4f} | Acc: {current_acc:.3f}")
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning(f"OOM at batch {batch_idx}, skipping...")
                optimizer.zero_grad()
                with memory_cleanup():
                    pass
                continue
            else:
                raise e
    
    avg_loss = total_loss / max(num_batches, 1)
    avg_acc = total_correct / max(total_tokens, 1)
    
    return avg_loss, avg_acc

def get_improved_config():
    """Get improved configuration."""
    if device.type == 'cuda':
        model_config = ModelConfig(
            vocab_size=8000,
            hidden_size=512,
            num_layers=8,
            num_heads=8,
            seq_length=256,
            dropout=0.1
        )
        batch_size = 16
        max_samples = 20000
    elif device.type == 'mps':
        model_config = ModelConfig(
            vocab_size=4000,
            hidden_size=256,
            num_layers=4,
            num_heads=4,
            seq_length=128,
            dropout=0.1
        )
        batch_size = 8
        max_samples = 5000
    else:
        model_config = ModelConfig(
            vocab_size=2000,
            hidden_size=128,
            num_layers=3,
            num_heads=4,
            seq_length=64,
            dropout=0.1
        )
        batch_size = 4
        max_samples = 2000
    
    training_config = TrainingConfig(
        learning_rate=3e-4,
        weight_decay=0.01,
        batch_size=batch_size,
        gradient_accumulation_steps=2,
        max_epochs=10,
        warmup_ratio=0.1,
        max_grad_norm=1.0
    )
    
    return model_config, training_config, max_samples

def validate_training_setup():
    """Validate required files exist."""
    required_files = ["oasst1_data/oasst1_train.jsonl"]
    
    for file_path in required_files:
        if not Path(file_path).exists():
            logger.error(f"Missing required file: {file_path}")
            return False
    
    return True

def generate_sample_text(model, tokenizer, prompt="<user> Hello", max_length=50):
    """Generate sample text for evaluation."""
    model.eval()
    
    try:
        with torch.no_grad():
            input_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
            generated = input_ids.clone()
            
            for _ in range(max_length):
                if generated.size(1) >= model.config.seq_length:
                    break
                
                logits = model(generated)
                
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    break
                
                next_token_logits = logits[0, -1, :]
                next_token_probs = F.softmax(next_token_logits / 0.8, dim=-1)
                
                if torch.isnan(next_token_probs).any():
                    break
                
                next_token = torch.multinomial(next_token_probs, 1)
                generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
                
                # Stop on EOS
                if next_token.item() == tokenizer.vocab.get("<eos>", -1):
                    break
            
            response_ids = generated[0][input_ids.size(1):].tolist()
            response = tokenizer.decode(response_ids)
            return response.strip()
    
    except Exception as e:
        logger.warning(f"Error generating sample: {e}")
        return "Generation failed"
    finally:
        model.train()

def evaluate_model(model, dataloader, criterion, max_batches=5):
    """Evaluate model performance."""
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
                    
                    logits = model(inputs)
                    loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
                    
                    if torch.isnan(loss) or torch.isinf(loss):
                        continue
                    
                    predictions = torch.argmax(logits, dim=-1)
                    correct = (predictions == targets).sum().item()
                    
                    total_loss += loss.item()
                    total_correct += correct
                    total_tokens += targets.numel()
                    num_batches += 1
                    
                except Exception:
                    continue
    
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
    
    finally:
        model.train()
    
    avg_loss = total_loss / max(num_batches, 1)
    accuracy = total_correct / max(total_tokens, 1)
    perplexity = math.exp(min(avg_loss, 20)) if avg_loss < float('inf') else float('inf')
    
    return {
        'avg_loss': avg_loss,
        'accuracy': accuracy,
        'perplexity': perplexity
    }

def main():
    """Main training function with improved stability."""
    logger.info("🚀 Starting Improved OASST1 Training")
    logger.info("=" * 80)
    
    # Environment check
    if not check_environment():
        return 1
    
    if not validate_training_setup():
        return 1
    
    # Configuration
    model_config, training_config, max_samples = get_improved_config()
    
    logger.info(f"Configuration:")
    logger.info(f"  Model: {model_config.hidden_size}x{model_config.num_layers}")
    logger.info(f"  Vocab size: {model_config.vocab_size}")
    logger.info(f"  Sequence length: {model_config.seq_length}")
    logger.info(f"  Batch size: {training_config.batch_size}")
    logger.info(f"  Learning rate: {training_config.learning_rate}")
    logger.info(f"  Max samples: {max_samples}")
    
    model_manager = ModelManager("models")
    
    try:
        # Load data
        logger.info("📚 Loading data...")
        texts = load_and_process_data("oasst1_data/oasst1_train.jsonl", max_samples)
        
        if len(texts) == 0:
            raise ValueError("No training data loaded!")
        
        logger.info(f"Loaded {len(texts):,} texts")
        
        # Create and train tokenizer
        logger.info("🔤 Training tokenizer...")
        tokenizer = ImprovedTokenizer()
        
        # Use sample for tokenizer training
        sample_texts = texts[:min(1000, len(texts))]
        all_text = "\n".join(sample_texts)
        
        tokenizer.train_from_text(all_text, vocab_size=model_config.vocab_size, min_freq=2)
        actual_vocab_size = tokenizer.vocab_size()
        model_config.vocab_size = actual_vocab_size
        
        logger.info(f"Tokenizer trained with {actual_vocab_size:,} tokens")
        
        # Test tokenizer
        test_text = "Hello world! How are you?"
        test_tokens = tokenizer.encode(test_text)
        test_decoded = tokenizer.decode(test_tokens)
        
        logger.info(f"Tokenizer test:")
        logger.info(f"  Original: '{test_text}'")
        logger.info(f"  Tokens: {test_tokens}")
        logger.info(f"  Decoded: '{test_decoded}'")
        
        # Create dataset
        logger.info("📦 Creating dataset...")
        dataset = StableDataset(
            texts, 
            tokenizer, 
            model_config.seq_length,
            max_sequences=min(10000, len(texts) * 2)
        )
        
        logger.info(f"Created {len(dataset):,} training sequences")
        
        # Create dataloaders
        train_dataloader = DataLoader(
            dataset,
            batch_size=training_config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            drop_last=True
        )
        
        # Create evaluation dataset
        eval_size = min(500, len(dataset) // 5)
        eval_indices = torch.randperm(len(dataset))[:eval_size]
        eval_dataset = torch.utils.data.Subset(dataset, eval_indices)
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=training_config.batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False
        )
        
        logger.info(f"Training batches per epoch: {len(train_dataloader)}")
        logger.info(f"Evaluation sequences: {len(eval_dataset)}")
        
        # Initialize model
        logger.info("🧠 Creating model...")
        with memory_cleanup():
            model = StableTransformer(model_config)
            model = model.to(device)
        
        total_params, trainable_params = count_parameters(model)
        model_size_mb = total_params * 4 / 1024**2
        
        logger.info(f"Model parameters: {total_params:,} (~{model_size_mb:.1f}MB)")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
        # Training components
        optimizer = optim.AdamW(
            model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
            betas=(training_config.beta1, training_config.beta2),
            eps=1e-8
        )
        
        pad_token_id = tokenizer.vocab.get("<pad>", 0)
        criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id, label_smoothing=0.1)
        
        total_steps = len(train_dataloader) * training_config.max_epochs // training_config.gradient_accumulation_steps
        warmup_steps = int(total_steps * training_config.warmup_ratio)
        
        scheduler = ImprovedScheduler(optimizer, warmup_steps, total_steps)
        
        logger.info(f"Training steps: {total_steps:,} (warmup: {warmup_steps:,})")
        logger.info(f"Memory before training: {get_memory_usage()}")
        
        # Training loop
        logger.info("🚀 Starting training...")
        training_start = time.time()
        best_loss = float('inf')
        best_accuracy = 0.0
        models_saved = 0
        
        for epoch in range(1, training_config.max_epochs + 1):
            epoch_start = time.time()
            
            logger.info(f"=== Epoch {epoch}/{training_config.max_epochs} ===")
            
            try:
                # Training
                train_loss, train_acc = train_epoch(
                    model, train_dataloader, criterion, optimizer, scheduler, epoch,
                    training_config.gradient_accumulation_steps,
                    training_config.max_grad_norm
                )
                
                # Check for training issues
                if math.isnan(train_loss) or math.isinf(train_loss):
                    logger.error(f"Invalid training loss: {train_loss}")
                    # Reduce learning rate and continue
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= 0.5
                    logger.info(f"Reduced learning rate to {optimizer.param_groups[0]['lr']}")
                    continue
                
                if train_loss > 15.0:
                    logger.warning(f"Very high loss: {train_loss:.4f}")
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= 0.8
                
                perplexity = math.exp(min(train_loss, 20))
                epoch_time = time.time() - epoch_start
                
                logger.info(f"Training Results:")
                logger.info(f"  Loss: {train_loss:.4f}")
                logger.info(f"  Accuracy: {train_acc:.3f} ({train_acc*100:.1f}%)")
                logger.info(f"  Perplexity: {perplexity:.2f}")
                logger.info(f"  Time: {epoch_time:.1f}s")
                
                # Evaluation
                eval_results = None
                if epoch % 2 == 0 or epoch == 1 or epoch == training_config.max_epochs:
                    logger.info("📊 Evaluating...")
                    eval_results = evaluate_model(model, eval_dataloader, criterion)
                    
                    logger.info(f"Evaluation Results:")
                    logger.info(f"  Loss: {eval_results['avg_loss']:.4f}")
                    logger.info(f"  Accuracy: {eval_results['accuracy']:.3f} ({eval_results['accuracy']*100:.1f}%)")
                    logger.info(f"  Perplexity: {eval_results['perplexity']:.2f}")
                
                # Sample generation
                if epoch % 3 == 0:
                    sample = generate_sample_text(model, tokenizer, "<user> Hello")
                    logger.info(f"Sample: <user> Hello → {sample}")
                
                # Track best metrics
                is_best_loss = train_loss < best_loss
                is_best_acc = train_acc > best_accuracy
                
                if is_best_loss:
                    best_loss = train_loss
                    logger.info(f"🏆 New best loss: {best_loss:.4f}")
                
                if is_best_acc:
                    best_accuracy = train_acc
                    logger.info(f"🏆 New best accuracy: {best_accuracy:.3f}")
                
                # Save model
                should_save = (
                    is_best_loss or 
                    is_best_acc or
                    epoch % 3 == 0 or 
                    epoch == training_config.max_epochs
                )
                
                if should_save:
                    performance_metrics = {
                        "train_loss": float(train_loss),
                        "train_accuracy": float(train_acc),
                        "train_perplexity": float(perplexity),
                        "epoch": int(epoch),
                        "learning_rate": float(optimizer.param_groups[0]['lr']),
                        "is_best_loss": is_best_loss,
                        "is_best_accuracy": is_best_acc,
                    }
                    
                    if eval_results:
                        performance_metrics.update({
                            "eval_loss": float(eval_results['avg_loss']),
                            "eval_accuracy": float(eval_results['accuracy']),
                            "eval_perplexity": float(eval_results['perplexity']),
                        })
                    
                    metadata = ModelMetadata(
                        model_name="OASST1_Stable_Transformer",
                        version=f"v1.0_epoch_{epoch}",
                        created_at=datetime.now().isoformat(),
                        model_config=model_config,
                        training_config=training_config,
                        dataset_info={
                            "name": "OpenAssistant OASST1",
                            "num_samples": len(texts),
                            "vocab_size": actual_vocab_size,
                            "seq_length": model_config.seq_length,
                            "train_sequences": len(dataset),
                        },
                        performance_metrics=performance_metrics,
                        model_size_mb=float(model_size_mb),
                        total_parameters=int(total_params),
                        trainable_parameters=int(trainable_params),
                        epochs_trained=int(epoch),
                        best_loss=float(best_loss),
                        best_perplexity=float(math.exp(min(best_loss, 20))),
                        hardware_used=device.type.upper(),
                        pytorch_version=torch.__version__,
                        notes=f"Improved stable training epoch {epoch}",
                        tags=["oasst1", "stable", f"epoch_{epoch}"] + 
                             (["best_loss"] if is_best_loss else []) +
                             (["best_accuracy"] if is_best_acc else [])
                    )
                    
                    try:
                        model_id = model_manager.save_model(model, tokenizer, metadata, force_cpu_save=True)
                        if model_id:
                            models_saved += 1
                            logger.info(f"💾 Model saved: {model_id}")
                    except Exception as save_error:
                        logger.error(f"Failed to save model: {save_error}")
                
                logger.info(f"Memory after epoch: {get_memory_usage()}")
                
                # Memory cleanup
                with memory_cleanup():
                    pass
                
            except Exception as e:
                logger.error(f"Error in epoch {epoch}: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                
                if "out of memory" in str(e).lower():
                    logger.info("Attempting OOM recovery...")
                    with memory_cleanup():
                        pass
                    continue
                else:
                    raise e
        
        # Training completion
        total_time = time.time() - training_start
        
        logger.info("=" * 80)
        logger.info("✅ Training completed successfully!")
        logger.info(f"Best loss: {best_loss:.4f}")
        logger.info(f"Best accuracy: {best_accuracy:.3f} ({best_accuracy*100:.1f}%)")
        logger.info(f"Best perplexity: {math.exp(min(best_loss, 20)):.2f}")
        logger.info(f"Models saved: {models_saved}")
        logger.info(f"Training time: {total_time/3600:.2f} hours")
        logger.info(f"Final memory: {get_memory_usage()}")
        
        # Final save if needed
        if models_saved == 0:
            logger.warning("No models saved! Performing final save...")
            final_metadata = ModelMetadata(
                model_name="OASST1_Final",
                version="v1.0_FINAL",
                created_at=datetime.now().isoformat(),
                model_config=model_config,
                performance_metrics={"final_loss": float(best_loss), "final_accuracy": float(best_accuracy)},
                notes="Final save",
                tags=["oasst1", "final"]
            )
            
            final_id = model_manager.save_model(model, tokenizer, final_metadata, force_cpu_save=True)
            if final_id:
                logger.info(f"Final save successful: {final_id}")
                models_saved += 1
        
        model_manager.print_model_summary()
        
        return 0 if models_saved > 0 else 1
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return 1
    finally:
        with memory_cleanup():
            pass

if __name__ == "__main__":
    exit(main())