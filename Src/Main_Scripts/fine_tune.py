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
from typing import Dict, List, Tuple, Optional, Any, Union
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
            logging.FileHandler('finetuning_debug.log', mode='w')
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info("🔧 Fine-tuning debug logging initialized")
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
class FineTuningConfig:
    learning_rate: float = 1e-5  # Lower LR for fine-tuning
    weight_decay: float = 0.01
    batch_size: int = 4  # Smaller for fine-tuning
    gradient_accumulation_steps: int = 8  # Higher to maintain effective batch size
    max_epochs: int = 5  # Fewer epochs for fine-tuning
    warmup_ratio: float = 0.1
    save_every: int = 500
    eval_every: int = 100
    max_grad_norm: float = 0.5  # Lower for stability
    label_smoothing: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    freeze_embeddings: bool = False  # Option to freeze embeddings
    freeze_layers: int = 0  # Number of bottom layers to freeze
    lora_enabled: bool = False  # LoRA fine-tuning
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.1

@dataclass
class ModelMetadata:
    model_name: str = "transformer"
    version: str = "v1.0"
    created_at: str = ""
    last_modified: str = ""
    model_config: ModelConfig = None
    training_config: FineTuningConfig = None
    pretrained_model: str = None  # Path to pre-trained model
    dataset_info: dict = None
    performance_metrics: dict = None
    model_size_mb: float = 0.0
    total_parameters: int = 0
    trainable_parameters: int = 0
    frozen_parameters: int = 0
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
    """Improved tokenizer with better stability and loading capabilities."""
    
    def __init__(self):
        self.vocab = {
            "<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3, 
            "<user>": 4, "<assistant>": 5, "\n": 6, " ": 7
        }
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.target_vocab_size = 10000
        self.trained = False
    
    def load_from_dict(self, tokenizer_data: dict):
        """Load tokenizer from saved data."""
        self.vocab = tokenizer_data.get('vocab', self.vocab)
        self.id_to_token = tokenizer_data.get('id_to_token', {})
        
        # Convert string keys back to integers for id_to_token
        self.id_to_token = {int(k): v for k, v in self.id_to_token.items()}
        
        self.target_vocab_size = len(self.vocab)
        self.trained = True
        logger.info(f"Tokenizer loaded with {len(self.vocab)} tokens")
    
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

class LoRALayer(nn.Module):
    """Low-Rank Adaptation layer for efficient fine-tuning."""
    
    def __init__(self, in_features: int, out_features: int, rank: int = 16, alpha: float = 32.0, dropout: float = 0.1):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: (batch_size, seq_len, in_features)
        lora_output = F.linear(x, (self.lora_B @ self.lora_A) * self.scaling)
        return self.dropout(lora_output)

class FineTuneableLinear(nn.Module):
    """Linear layer that can be enhanced with LoRA for fine-tuning."""
    
    def __init__(self, original_layer: nn.Linear, lora_config: Optional[dict] = None):
        super().__init__()
        self.original_layer = original_layer
        self.lora_enabled = lora_config is not None
        
        if self.lora_enabled:
            self.lora_layer = LoRALayer(
                in_features=original_layer.in_features,
                out_features=original_layer.out_features,
                rank=lora_config.get('rank', 16),
                alpha=lora_config.get('alpha', 32.0),
                dropout=lora_config.get('dropout', 0.1)
            )
        
        # Freeze original layer for LoRA
        if self.lora_enabled:
            for param in self.original_layer.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        output = self.original_layer(x)
        if self.lora_enabled:
            output = output + self.lora_layer(x)
        return output

class StableTransformer(nn.Module):
    """Improved transformer with fine-tuning capabilities."""
    
    def __init__(self, config, lora_config=None):
        super().__init__()
        self.config = config
        self.lora_config = lora_config
        
        # Embeddings with proper scaling
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.pos_embeddings = nn.Embedding(config.seq_length, config.hidden_size)
        
        # Input normalization
        self.input_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        
        # Transformer layers with prenorm
        self.layers = nn.ModuleList()
        for _ in range(config.num_layers):
            layer = TransformerBlock(config, lora_config=lora_config)
            self.layers.append(layer)
        
        # Output layers
        self.output_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.dropout = nn.Dropout(config.dropout)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Apply LoRA to output layer if enabled
        if lora_config:
            self.lm_head = FineTuneableLinear(self.lm_head, lora_config)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Scale embeddings
        nn.init.normal_(self.embeddings.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embeddings.weight, mean=0.0, std=0.02)
        
        # Tie embeddings and output weights for better performance (if not using LoRA)
        if not lora_config:
            if hasattr(self.lm_head, 'weight'):
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
    
    def freeze_layers(self, num_layers: int):
        """Freeze bottom N layers for fine-tuning."""
        if num_layers > 0:
            logger.info(f"Freezing bottom {num_layers} transformer layers")
            for i in range(min(num_layers, len(self.layers))):
                for param in self.layers[i].parameters():
                    param.requires_grad = False
    
    def freeze_embeddings(self):
        """Freeze embedding layers."""
        logger.info("Freezing embedding layers")
        for param in self.embeddings.parameters():
            param.requires_grad = False
        for param in self.pos_embeddings.parameters():
            param.requires_grad = False
    
    def get_parameter_counts(self):
        """Get counts of total, trainable, and frozen parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        return total, trainable, frozen
    
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
    """Transformer block with pre-normalization and LoRA support."""
    
    def __init__(self, config, lora_config=None):
        super().__init__()
        self.config = config
        
        # Pre-normalization layers
        self.ln_1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.ln_2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        
        # Attention
        self.attn = MultiHeadAttention(config, lora_config=lora_config)
        
        # MLP
        self.mlp = MLP(config, lora_config=lora_config)
        
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
    """Stable multi-head attention implementation with LoRA support."""
    
    def __init__(self, config, lora_config=None):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.scale = self.head_dim ** -0.5
        
        assert config.hidden_size % config.num_heads == 0
        
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        
        # Apply LoRA if enabled
        if lora_config:
            self.qkv = FineTuneableLinear(self.qkv, lora_config)
            self.out_proj = FineTuneableLinear(self.out_proj, lora_config)
        
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
    """MLP block with GELU activation and LoRA support."""
    
    def __init__(self, config, lora_config=None):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, 4 * config.hidden_size)
        self.fc2 = nn.Linear(4 * config.hidden_size, config.hidden_size)
        
        # Apply LoRA if enabled
        if lora_config:
            self.fc1 = FineTuneableLinear(self.fc1, lora_config)
            self.fc2 = FineTuneableLinear(self.fc2, lora_config)
        
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class ModelManager:
    """Model manager for saving, loading, and fine-tuning models."""
    
    def __init__(self, save_dir):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
    
    def load_pretrained_model(self, model_path: Union[str, Path]) -> Tuple[nn.Module, ImprovedTokenizer, dict]:
        """Load a pre-trained model for fine-tuning."""
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model directory not found: {model_path}")
        
        logger.info(f"Loading pre-trained model from: {model_path}")
        
        # Load metadata
        metadata_file = model_path / "metadata.json"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Extract model config
        model_config_dict = metadata.get('model_config', {})
        if isinstance(model_config_dict, dict):
            model_config = ModelConfig(**model_config_dict)
        else:
            model_config = model_config_dict
        
        # Load tokenizer
        tokenizer_file = model_path / "tokenizer.json"
        if not tokenizer_file.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_file}")
        
        with open(tokenizer_file, 'r') as f:
            tokenizer_data = json.load(f)
        
        tokenizer = ImprovedTokenizer()
        tokenizer.load_from_dict(tokenizer_data)
        
        # Create model with same config
        model = StableTransformer(model_config)
        
        # Load model weights
        model_file = model_path / "model.pth"
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        state_dict = torch.load(model_file, map_location='cpu')
        
        # Handle potential key mismatches
        model_state_dict = model.state_dict()
        filtered_state_dict = {}
        
        for key, value in state_dict.items():
            if key in model_state_dict:
                if model_state_dict[key].shape == value.shape:
                    filtered_state_dict[key] = value
                else:
                    logger.warning(f"Shape mismatch for {key}: expected {model_state_dict[key].shape}, got {value.shape}")
            else:
                logger.warning(f"Unknown key in state_dict: {key}")
        
        # Load the filtered state dict
        missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
        
        if missing_keys:
            logger.warning(f"Missing keys in loaded model: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys in loaded model: {unexpected_keys}")
        
        logger.info(f"✅ Pre-trained model loaded successfully")
        logger.info(f"  Model config: {model_config.hidden_size}x{model_config.num_layers}")
        logger.info(f"  Vocab size: {model_config.vocab_size}")
        logger.info(f"  Sequence length: {model_config.seq_length}")
        
        return model, tokenizer, metadata
    
    def prepare_model_for_finetuning(self, model: nn.Module, config: FineTuningConfig) -> nn.Module:
        """Prepare a model for fine-tuning with various strategies."""
        logger.info("🔧 Preparing model for fine-tuning...")
        
        # Apply LoRA if enabled
        if config.lora_enabled:
            logger.info("Applying LoRA (Low-Rank Adaptation)")
            lora_config = {
                'rank': config.lora_rank,
                'alpha': config.lora_alpha,
                'dropout': config.lora_dropout
            }
            
            # Convert existing model to use LoRA
            model = self._apply_lora_to_model(model, lora_config)
        
        # Freeze embeddings if requested
        if config.freeze_embeddings:
            model.freeze_embeddings()
        
        # Freeze bottom layers if requested
        if config.freeze_layers > 0:
            model.freeze_layers(config.freeze_layers)
        
        # Print parameter statistics
        total, trainable, frozen = model.get_parameter_counts()
        logger.info(f"Parameter counts after fine-tuning preparation:")
        logger.info(f"  Total: {total:,}")
        logger.info(f"  Trainable: {trainable:,} ({trainable/total*100:.1f}%)")
        logger.info(f"  Frozen: {frozen:,} ({frozen/total*100:.1f}%)")
        
        return model
    
    def _apply_lora_to_model(self, model: nn.Module, lora_config: dict) -> nn.Module:
        """Apply LoRA to existing model layers."""
        # This would need to be implemented to convert existing layers to LoRA layers
        # For now, we'll create a new model with LoRA support
        new_model = StableTransformer(model.config, lora_config=lora_config)
        
        # Copy non-LoRA parameters
        model_dict = model.state_dict()
        new_model_dict = new_model.state_dict()
        
        # Copy compatible parameters
        for key, value in model_dict.items():
            if key in new_model_dict and new_model_dict[key].shape == value.shape:
                new_model_dict[key] = value
        
        new_model.load_state_dict(new_model_dict, strict=False)
        return new_model
    
    def save_model(self, model, tokenizer, metadata, optimizer=None, scheduler=None, force_cpu_save=True):
        """Save model with proper error handling - ALWAYS force CPU save to prevent OOM."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_id = f"finetuned_{timestamp}"
        model_path = self.save_dir / model_id
        model_path.mkdir(exist_ok=True)
        
        try:
            # ALWAYS save to CPU to prevent memory issues
            model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save(model_state, model_path / "model.pth")
            
            # Move model back to device
            model.to(device)
            
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
            
            # Save fine-tuning specific info
            finetune_info = {
                'is_finetuned': True,
                'pretrained_model': metadata.pretrained_model,
                'finetuning_config': asdict(metadata.training_config) if hasattr(metadata.training_config, '__dict__') else metadata.training_config.__dict__,
                'parameter_counts': {
                    'total': metadata.total_parameters,
                    'trainable': metadata.trainable_parameters,
                    'frozen': metadata.frozen_parameters
                }
            }
            
            with open(model_path / "finetune_info.json", 'w') as f:
                json.dump(finetune_info, f, indent=2, default=str)
            
            logger.info(f"Fine-tuned model saved to: {model_path}")
            return model_id
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return None
    
    def print_model_summary(self):
        """Print summary of saved models."""
        models = list(self.save_dir.glob("*"))
        finetuned_models = [m for m in models if m.name.startswith("finetuned_")]
        
        logger.info(f"Found {len(models)} total models, {len(finetuned_models)} fine-tuned models in {self.save_dir}")
        for model_path in sorted(models):
            model_type = "FINE-TUNED" if model_path.name.startswith("finetuned_") else "TRAINED"
            logger.info(f"  - {model_path.name} ({model_type})")

# Memory management and device setup
@contextmanager
def memory_cleanup():
    """Context manager for aggressive memory cleanup."""
    try:
        yield
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.ipc_collect()
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()

def get_memory_usage():
    """Get current memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return f"CUDA: {allocated:.2f}GB allocated, {cached:.2f}GB cached, {total:.1f}GB total"
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
        torch.cuda.set_per_process_memory_fraction(0.85)
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
    """Fixed dataset with proper sequence creation."""
    
    def __init__(self, texts: List[str], tokenizer, seq_length: int, max_sequences: int = 10000):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        
        logger.info(f"Creating dataset with seq_length={seq_length}...")
        
        vocab_size = tokenizer.vocab_size()
        pad_token_id = tokenizer.vocab.get("<pad>", 0)
        bos_token_id = tokenizer.vocab.get("<bos>", 2)
        eos_token_id = tokenizer.vocab.get("<eos>", 3)
        
        logger.info(f"Tokenizer vocab size: {vocab_size}")
        logger.info(f"Special tokens - PAD: {pad_token_id}, BOS: {bos_token_id}, EOS: {eos_token_id}")
        
        self.sequences = []
        processed_texts = 0
        
        for text_idx, text in enumerate(texts):
            if len(self.sequences) >= max_sequences:
                break
                
            if not text or not text.strip():
                continue
            
            try:
                text_clean = text.strip()
                tokens = tokenizer.encode(text_clean)
                
                if not tokens or len(tokens) < 5:
                    continue
                
                full_sequence = [bos_token_id] + tokens + [eos_token_id]
                
                valid_tokens = []
                for token in full_sequence:
                    if 0 <= token < vocab_size:
                        valid_tokens.append(token)
                    else:
                        valid_tokens.append(tokenizer.vocab.get("<unk>", 1))
                
                if len(valid_tokens) > seq_length + 1:
                    for start in range(0, len(valid_tokens) - seq_length, seq_length // 2):
                        if start + seq_length + 1 <= len(valid_tokens):
                            sequence = valid_tokens[start:start + seq_length + 1]
                            if len(sequence) == seq_length + 1:
                                self.sequences.append(sequence)
                                if len(self.sequences) >= max_sequences:
                                    break
                elif len(valid_tokens) >= seq_length + 1:
                    sequence = valid_tokens[:seq_length + 1]
                    if len(sequence) == seq_length + 1:
                        self.sequences.append(sequence)
                else:
                    sequence = valid_tokens + [pad_token_id] * (seq_length + 1 - len(valid_tokens))
                    if len(sequence) == seq_length + 1:
                        self.sequences.append(sequence)
                
                processed_texts += 1
                
                if processed_texts % 1000 == 0:
                    logger.info(f"Processed {processed_texts} texts, created {len(self.sequences)} sequences")
                    
            except Exception as e:
                logger.warning(f"Error processing text {text_idx}: {e}")
                continue
        
        if not self.sequences:
            raise ValueError("No valid sequences created! Check your data and tokenizer.")
        
        self.pad_token_id = pad_token_id
        self.vocab_size = vocab_size
        
        logger.info(f"Final dataset: {len(self.sequences):,} sequences from {processed_texts} texts")
        
        # Validate sequences
        invalid_sequences = 0
        for i, seq in enumerate(self.sequences):
            if len(seq) != seq_length + 1:
                invalid_sequences += 1
            elif any(token < 0 or token >= vocab_size for token in seq):
                invalid_sequences += 1
        
        if invalid_sequences > 0:
            raise ValueError(f"Found {invalid_sequences} invalid sequences!")
        
        logger.info("✅ All sequences validated successfully")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        if idx >= len(self.sequences):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.sequences)}")
            
        sequence = self.sequences[idx]
        input_ids = torch.tensor(sequence[:-1], dtype=torch.long)
        target_ids = torch.tensor(sequence[1:], dtype=torch.long)
        
        return input_ids, target_ids

def load_and_process_data(data_path: str, max_samples: Optional[int] = None) -> List[str]:
    """Load OASST1 data with better processing."""
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    logger.info(f"Loading fine-tuning data from: {data_path}")
    
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
                    
                    word_count = len(text.split())
                    if word_count < 5 or word_count > 150:
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
                    
                    if processed_count % 1000 == 0:
                        logger.info(f"Processed {processed_count:,} samples...")
                    
                except (json.JSONDecodeError, KeyError):
                    continue
    
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise
    
    logger.info(f"Loaded {len(texts):,} texts for fine-tuning")
    return texts

class ImprovedScheduler:
    """Improved learning rate scheduler with warmup and cosine decay for fine-tuning."""
    
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

def count_parameters(model):
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return total, trainable, frozen

def finetune_epoch(model, dataloader, criterion, optimizer, scheduler, epoch, 
                   gradient_accumulation_steps=1, max_grad_norm=1.0):
    """Fine-tuning epoch with lower learning rates and careful gradient handling."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    num_batches = 0
    accumulation_count = 0
    
    optimizer.zero_grad()
    
    logger.info(f"Starting fine-tuning epoch {epoch} with {len(dataloader)} batches")
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        try:
            if batch_idx > 0 and batch_idx % 20 == 0:
                with memory_cleanup():
                    pass
                logger.info(f"Memory cleanup at batch {batch_idx}: {get_memory_usage()}")
            
            if inputs.numel() == 0 or targets.numel() == 0:
                logger.warning(f"Empty batch at index {batch_idx}, skipping")
                continue
                
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            if torch.isnan(inputs).any() or torch.isinf(inputs).any():
                logger.warning(f"Invalid inputs at batch {batch_idx}, skipping")
                continue
            
            with torch.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                logits = model(inputs)
                
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    logger.warning(f"Invalid logits at batch {batch_idx}, skipping")
                    optimizer.zero_grad()
                    continue
                
                flat_logits = logits.view(-1, logits.size(-1))
                flat_targets = targets.view(-1)
                loss = criterion(flat_logits, flat_targets)
            
            if torch.isnan(loss) or torch.isinf(loss) or loss.item() < 0:
                logger.warning(f"Invalid loss at batch {batch_idx}: {loss.item()}, skipping")
                optimizer.zero_grad()
                continue
            
            scaled_loss = loss / gradient_accumulation_steps
            scaled_loss.backward()
            
            del logits, flat_logits, flat_targets, scaled_loss
            
            accumulation_count += 1
            
            if accumulation_count >= gradient_accumulation_steps:
                if max_grad_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    
                    if grad_norm > max_grad_norm * 5:  # More conservative for fine-tuning
                        logger.warning(f"Large gradients detected: {grad_norm:.2f}, reducing LR")
                        for param_group in optimizer.param_groups:
                            param_group['lr'] *= 0.8
                
                optimizer.step()
                current_lr = scheduler.step()
                optimizer.zero_grad()
                accumulation_count = 0
            
            # Calculate accuracy
            with torch.no_grad():
                with torch.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                    pred_logits = model(inputs)
                predictions = torch.argmax(pred_logits, dim=-1)
                del pred_logits
                
                pad_mask = (targets != criterion.ignore_index)
                correct = ((predictions == targets) & pad_mask).sum().item()
                valid_tokens = pad_mask.sum().item()
                
                del predictions, pad_mask
                
                total_correct += correct
                total_tokens += valid_tokens
            
            total_loss += loss.item()
            num_batches += 1
            del loss
            
            if batch_idx % 10 == 0:
                current_loss = total_loss / max(num_batches, 1)
                current_acc = total_correct / max(total_tokens, 1)
                current_lr = optimizer.param_groups[0]['lr']
                
                logger.info(f"FT Epoch {epoch} | Batch {batch_idx}/{len(dataloader)} | "
                           f"Loss: {current_loss:.4f} | Acc: {current_acc:.3f} ({current_acc*100:.1f}%) | "
                           f"LR: {current_lr:.8f} | Tokens: {total_tokens}")
                logger.info(f"Memory: {get_memory_usage()}")
            
        except RuntimeError as e:
            error_msg = str(e).lower()
            if "out of memory" in error_msg:
                logger.warning(f"OOM at batch {batch_idx}, clearing cache...")
                optimizer.zero_grad()
                
                for var_name in ['inputs', 'targets', 'logits', 'loss']:
                    if var_name in locals():
                        del locals()[var_name]
                
                with memory_cleanup():
                    pass
                continue
            else:
                logger.error(f"Runtime error at batch {batch_idx}: {e}")
                raise e
        except Exception as e:
            logger.error(f"Unexpected error at batch {batch_idx}: {e}")
            for var_name in ['inputs', 'targets', 'logits', 'loss']:
                if var_name in locals():
                    del locals()[var_name]
            continue
    
    if accumulation_count > 0:
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()
    
    if num_batches == 0:
        logger.error("No batches processed! Check your dataset and dataloader.")
        return float('inf'), 0.0
    
    avg_loss = total_loss / num_batches
    avg_acc = total_correct / max(total_tokens, 1)
    
    logger.info(f"Fine-tuning epoch {epoch} completed: {num_batches} batches, {total_tokens} tokens processed")
    
    return avg_loss, avg_acc

def evaluate_model(model, dataloader, criterion, max_batches=5):
    """Evaluation function for fine-tuned models."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    num_batches = 0
    
    logger.info(f"Starting evaluation with max {max_batches} batches from {len(dataloader)} available")
    
    try:
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                if batch_idx >= max_batches:
                    break
                
                try:
                    if inputs.numel() == 0 or targets.numel() == 0:
                        continue
                        
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    
                    if torch.isnan(inputs).any() or torch.isinf(inputs).any():
                        continue
                    
                    with torch.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                        logits = model(inputs)
                    
                    if torch.isnan(logits).any() or torch.isinf(logits).any():
                        del logits
                        continue
                    
                    loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
                    
                    if torch.isnan(loss) or torch.isinf(loss):
                        del logits, loss
                        continue
                    
                    predictions = torch.argmax(logits, dim=-1)
                    pad_mask = (targets != criterion.ignore_index)
                    correct = ((predictions == targets) & pad_mask).sum().item()
                    valid_tokens = pad_mask.sum().item()
                    
                    total_loss += loss.item()
                    total_correct += correct
                    total_tokens += valid_tokens
                    num_batches += 1
                    
                    logger.info(f"Eval batch {batch_idx}: loss={loss.item():.4f}, acc={correct/max(valid_tokens,1):.3f}")
                    
                    del logits, loss, predictions, pad_mask, inputs, targets
                    
                except Exception as e:
                    logger.warning(f"Error in eval batch {batch_idx}: {e}")
                    for var_name in ['inputs', 'targets', 'logits', 'loss', 'predictions']:
                        if var_name in locals():
                            del locals()[var_name]
                    continue
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
    
    finally:
        model.train()
        with memory_cleanup():
            pass
    
    if num_batches == 0:
        logger.warning("No evaluation batches processed!")
        return {'avg_loss': float('inf'), 'accuracy': 0.0, 'perplexity': float('inf')}
    
    avg_loss = total_loss / num_batches
    accuracy = total_correct / max(total_tokens, 1)
    perplexity = math.exp(min(avg_loss, 20)) if avg_loss < float('inf') else float('inf')
    
    logger.info(f"Evaluation completed: {num_batches} batches, {total_tokens} tokens")
    
    return {
        'avg_loss': avg_loss,
        'accuracy': accuracy,
        'perplexity': perplexity
    }

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

def get_finetuning_config(device_type: str, use_lora: bool = False):
    """Get fine-tuning specific configuration."""
    if device_type == 'cuda':
        config = FineTuningConfig(
            learning_rate=2e-5,  # Lower learning rate for fine-tuning
            weight_decay=0.01,
            batch_size=4,  # Smaller batch size
            gradient_accumulation_steps=8,  # Higher accumulation
            max_epochs=3,  # Fewer epochs
            warmup_ratio=0.1,
            max_grad_norm=0.5,  # Lower gradient clipping
            label_smoothing=0.1,
            freeze_embeddings=False,
            freeze_layers=0,  # Can be adjusted based on needs
            lora_enabled=use_lora,
            lora_rank=16,
            lora_alpha=32.0,
            lora_dropout=0.1
        )
        max_samples = 5000
    elif device_type == 'mps':
        config = FineTuningConfig(
            learning_rate=1e-5,
            batch_size=2,
            gradient_accumulation_steps=12,
            max_epochs=2,
            freeze_layers=2,  # Freeze more layers on smaller devices
            lora_enabled=use_lora,
            lora_rank=8
        )
        max_samples = 2000
    else:
        config = FineTuningConfig(
            learning_rate=5e-6,
            batch_size=1,
            gradient_accumulation_steps=16,
            max_epochs=2,
            freeze_layers=4,
            lora_enabled=use_lora,
            lora_rank=4
        )
        max_samples = 1000
    
    return config, max_samples

def main():
    """Main fine-tuning function."""
    logger.info("🚀 Starting OASST1 Fine-Tuning")
    logger.info("=" * 80)
    
    if not check_environment():
        return 1
    
    # Parse arguments (in a real implementation, use argparse)
    pretrained_model_path = "models/model_20250108_123456"  # Example path
    use_lora = False  # Set to True to enable LoRA fine-tuning
    finetune_data_path = "oasst1_data/oasst1_train.jsonl"
    
    # Check if paths exist
    if not Path(pretrained_model_path).exists():
        logger.error(f"Pre-trained model path not found: {pretrained_model_path}")
        logger.info("Available models:")
        model_dir = Path("models")
        if model_dir.exists():
            for model_path in sorted(model_dir.glob("model_*")):
                logger.info(f"  - {model_path}")
        return 1
    
    if not Path(finetune_data_path).exists():
        logger.error(f"Fine-tuning data not found: {finetune_data_path}")
        return 1
    
    # Configuration
    finetuning_config, max_samples = get_finetuning_config(device.type, use_lora)
    
    logger.info(f"Fine-tuning Configuration:")
    logger.info(f"  Pre-trained model: {pretrained_model_path}")
    logger.info(f"  LoRA enabled: {finetuning_config.lora_enabled}")
    logger.info(f"  Learning rate: {finetuning_config.learning_rate}")
    logger.info(f"  Batch size: {finetuning_config.batch_size}")
    logger.info(f"  Max epochs: {finetuning_config.max_epochs}")
    logger.info(f"  Freeze layers: {finetuning_config.freeze_layers}")
    logger.info(f"  Max samples: {max_samples}")
    
    model_manager = ModelManager("models")
    
    try:
        # Load pre-trained model
        logger.info("📂 Loading pre-trained model...")
        model, tokenizer, pretrained_metadata = model_manager.load_pretrained_model(pretrained_model_path)
        
        # Move to device
        model = model.to(device)
        
        # Prepare model for fine-tuning
        model = model_manager.prepare_model_for_finetuning(model, finetuning_config)
        
        # Load fine-tuning data
        logger.info("📚 Loading fine-tuning data...")
        texts = load_and_process_data(finetune_data_path, max_samples)
        
        if len(texts) == 0:
            raise ValueError("No fine-tuning data loaded!")
        
        logger.info(f"Loaded {len(texts):,} texts for fine-tuning")
        
        # Create dataset
        logger.info("📦 Creating fine-tuning dataset...")
        dataset = StableDataset(
            texts, 
            tokenizer, 
            model.config.seq_length,
            max_sequences=min(5000, len(texts) * 2)
        )
        
        logger.info(f"Created {len(dataset):,} fine-tuning sequences")
        
        # Create dataloaders
        train_dataloader = DataLoader(
            dataset,
            batch_size=finetuning_config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            drop_last=True,
            persistent_workers=False
        )
        
        # Create evaluation dataset
        eval_size = min(100, len(dataset) // 10)
        eval_indices = torch.randperm(len(dataset))[:eval_size]
        eval_dataset = torch.utils.data.Subset(dataset, eval_indices)
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=finetuning_config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=False
        )
        
        total_params, trainable_params, frozen_params = count_parameters(model)
        
        logger.info(f"Model parameter counts:")
        logger.info(f"  Total: {total_params:,}")
        logger.info(f"  Trainable: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
        logger.info(f"  Frozen: {frozen_params:,} ({frozen_params/total_params*100:.1f}%)")
        
        # Setup optimizer and scheduler for fine-tuning
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),  # Only trainable parameters
            lr=finetuning_config.learning_rate,
            weight_decay=finetuning_config.weight_decay,
            betas=(finetuning_config.beta1, finetuning_config.beta2),
            eps=1e-8
        )
        
        pad_token_id = tokenizer.vocab.get("<pad>", 0)
        criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id, label_smoothing=finetuning_config.label_smoothing)
        
        total_steps = len(train_dataloader) * finetuning_config.max_epochs // finetuning_config.gradient_accumulation_steps
        warmup_steps = int(total_steps * finetuning_config.warmup_ratio)
        
        scheduler = ImprovedScheduler(optimizer, warmup_steps, total_steps)
        
        logger.info(f"Fine-tuning steps: {total_steps:,} (warmup: {warmup_steps:,})")
        logger.info(f"Memory before fine-tuning: {get_memory_usage()}")
        
        # Fine-tuning loop
        logger.info("🚀 Starting fine-tuning...")
        training_start = time.time()
        best_loss = float('inf')
        best_accuracy = 0.0
        models_saved = 0
        
        for epoch in range(1, finetuning_config.max_epochs + 1):
            epoch_start = time.time()
            
            logger.info(f"=== Fine-tuning Epoch {epoch}/{finetuning_config.max_epochs} ===")
            
            try:
                train_loss, train_acc = finetune_epoch(
                    model, train_dataloader, criterion, optimizer, scheduler, epoch,
                    finetuning_config.gradient_accumulation_steps,
                    finetuning_config.max_grad_norm
                )
                
                if math.isnan(train_loss) or math.isinf(train_loss):
                    logger.error(f"Invalid training loss: {train_loss}")
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= 0.5
                    logger.info(f"Reduced learning rate to {optimizer.param_groups[0]['lr']}")
                    continue
                
                perplexity = math.exp(min(train_loss, 20))
                epoch_time = time.time() - epoch_start
                
                logger.info(f"Fine-tuning Results:")
                logger.info(f"  Loss: {train_loss:.4f}")
                logger.info(f"  Accuracy: {train_acc:.3f} ({train_acc*100:.1f}%)")
                logger.info(f"  Perplexity: {perplexity:.2f}")
                logger.info(f"  Time: {epoch_time:.1f}s")
                
                # Evaluation
                eval_results = None
                if epoch % 1 == 0:  # Evaluate every epoch for fine-tuning
                    logger.info("📊 Evaluating...")
                    with memory_cleanup():
                        eval_results = evaluate_model(model, eval_dataloader, criterion)
                    
                    logger.info(f"Evaluation Results:")
                    logger.info(f"  Loss: {eval_results['avg_loss']:.4f}")
                    logger.info(f"  Accuracy: {eval_results['accuracy']:.3f} ({eval_results['accuracy']*100:.1f}%)")
                    logger.info(f"  Perplexity: {eval_results['perplexity']:.2f}")
                
                # Sample generation
                if epoch % 1 == 0:
                    with memory_cleanup():
                        sample = generate_sample_text(model, tokenizer, "<user> Hello, how are you?")
                        logger.info(f"Sample: <user> Hello, how are you? → {sample}")
                
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
                    epoch == finetuning_config.max_epochs
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
                        model_name="OASST1_FineTuned_Transformer",
                        version=f"v1.0_ft_epoch_{epoch}",
                        created_at=datetime.now().isoformat(),
                        model_config=model.config,
                        training_config=finetuning_config,
                        pretrained_model=str(pretrained_model_path),
                        dataset_info={
                            "name": "OpenAssistant OASST1 Fine-tuning",
                            "num_samples": len(texts),
                            "vocab_size": tokenizer.vocab_size(),
                            "seq_length": model.config.seq_length,
                            "train_sequences": len(dataset),
                            "base_model": pretrained_metadata.get("model_name", "Unknown")
                        },
                        performance_metrics=performance_metrics,
                        model_size_mb=float(total_params * 4 / 1024**2),
                        total_parameters=int(total_params),
                        trainable_parameters=int(trainable_params),
                        frozen_parameters=int(frozen_params),
                        epochs_trained=int(epoch),
                        best_loss=float(best_loss),
                        best_perplexity=float(math.exp(min(best_loss, 20))),
                        hardware_used=device.type.upper(),
                        pytorch_version=torch.__version__,
                        notes=f"Fine-tuned from {pretrained_model_path} on epoch {epoch}" + 
                              (f" with LoRA (rank={finetuning_config.lora_rank})" if finetuning_config.lora_enabled else ""),
                        tags=["oasst1", "finetuned", f"epoch_{epoch}"] + 
                             (["lora"] if finetuning_config.lora_enabled else []) +
                             (["best_loss"] if is_best_loss else []) +
                             (["best_accuracy"] if is_best_acc else [])
                    )
                    
                    try:
                        with memory_cleanup():
                            model_id = model_manager.save_model(model, tokenizer, metadata)
                            if model_id:
                                models_saved += 1
                                logger.info(f"💾 Fine-tuned model saved: {model_id}")
                    except Exception as save_error:
                        logger.error(f"Failed to save model: {save_error}")
                
                logger.info(f"Memory after epoch: {get_memory_usage()}")
                
                # Memory cleanup between epochs
                with memory_cleanup():
                    pass
                
            except Exception as e:
                logger.error(f"Error in fine-tuning epoch {epoch}: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                
                if "out of memory" in str(e).lower():
                    logger.info("Attempting OOM recovery...")
                    
                    optimizer.zero_grad()
                    
                    for var_name in ['inputs', 'targets', 'logits', 'loss', 'predictions']:
                        if var_name in locals():
                            del locals()[var_name]
                    
                    with memory_cleanup():
                        pass
                    
                    # Reduce batch size for fine-tuning
                    current_batch_size = finetuning_config.batch_size
                    if current_batch_size > 1:
                        finetuning_config.batch_size = max(1, current_batch_size // 2)
                        finetuning_config.gradient_accumulation_steps *= 2
                        logger.info(f"Reduced batch size to {finetuning_config.batch_size}, "
                                   f"increased grad accumulation to {finetuning_config.gradient_accumulation_steps}")
                        
                        train_dataloader = DataLoader(
                            dataset,
                            batch_size=finetuning_config.batch_size,
                            shuffle=True,
                            num_workers=0,
                            pin_memory=False,
                            drop_last=True
                        )
                    
                    continue
                else:
                    raise e
        
        # Fine-tuning completion
        total_time = time.time() - training_start
        
        logger.info("=" * 80)
        logger.info("✅ Fine-tuning completed successfully!")
        logger.info(f"Pre-trained model: {pretrained_model_path}")
        logger.info(f"Best loss: {best_loss:.4f}")
        logger.info(f"Best accuracy: {best_accuracy:.3f} ({best_accuracy*100:.1f}%)")
        logger.info(f"Best perplexity: {math.exp(min(best_loss, 20)):.2f}")
        logger.info(f"Models saved: {models_saved}")
        logger.info(f"Fine-tuning time: {total_time/3600:.2f} hours")
        logger.info(f"Parameter efficiency: {trainable_params/total_params*100:.1f}% trainable")
        if finetuning_config.lora_enabled:
            logger.info(f"LoRA configuration: rank={finetuning_config.lora_rank}, alpha={finetuning_config.lora_alpha}")
        logger.info(f"Final memory: {get_memory_usage()}")
        
        # Final comparison with base model
        logger.info("\n🔍 Fine-tuning Summary:")
        logger.info(f"  Base model performance: {pretrained_metadata.get('performance_metrics', {}).get('train_loss', 'N/A')}")
        logger.info(f"  Fine-tuned performance: {best_loss:.4f}")
        improvement = ""
        base_loss = pretrained_metadata.get('performance_metrics', {}).get('train_loss')
        if base_loss and isinstance(base_loss, (int, float)):
            if best_loss < base_loss:
                improvement = f" (improved by {base_loss - best_loss:.4f})"
            else:
                improvement = f" (degraded by {best_loss - base_loss:.4f})"
        logger.info(f"  Loss change: {improvement}")
        
        # Final save if needed
        if models_saved == 0:
            logger.warning("No models saved! Performing final save...")
            final_metadata = ModelMetadata(
                model_name="OASST1_FineTuned_FINAL",
                version="v1.0_FT_FINAL",
                created_at=datetime.now().isoformat(),
                model_config=model.config,
                training_config=finetuning_config,
                pretrained_model=str(pretrained_model_path),
                performance_metrics={"final_loss": float(best_loss), "final_accuracy": float(best_accuracy)},
                total_parameters=int(total_params),
                trainable_parameters=int(trainable_params),
                frozen_parameters=int(frozen_params),
                notes="Final fine-tuned save",
                tags=["oasst1", "finetuned", "final"]
            )
            
            with memory_cleanup():
                final_id = model_manager.save_model(model, tokenizer, final_metadata)
                if final_id:
                    logger.info(f"Final save successful: {final_id}")
                    models_saved += 1
        
        # Generate final samples
        logger.info("\n🎯 Final Sample Generations:")
        test_prompts = [
            "<user> Hello, how are you?",
            "<user> What is machine learning?",
            "<user> Can you help me with Python?",
            "<user> Explain quantum computing simply."
        ]
        
        for prompt in test_prompts:
            try:
                with memory_cleanup():
                    response = generate_sample_text(model, tokenizer, prompt, max_length=40)
                    logger.info(f"  {prompt} → {response}")
            except Exception as e:
                logger.warning(f"  {prompt} → Generation failed: {e}")
        
        model_manager.print_model_summary()
        
        return 0 if models_saved > 0 else 1
        
    except KeyboardInterrupt:
        logger.info("Fine-tuning interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Fine-tuning failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return 1
    finally:
        with memory_cleanup():
            pass

def create_finetuning_script_with_args():
    """Create a command-line interface for the fine-tuning script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune OASST1 models")
    parser.add_argument("--pretrained_model", type=str, required=True,
                       help="Path to pre-trained model directory")
    parser.add_argument("--data_path", type=str, default="oasst1_data/oasst1_train.jsonl",
                       help="Path to fine-tuning data")
    parser.add_argument("--use_lora", action="store_true",
                       help="Enable LoRA fine-tuning")
    parser.add_argument("--lora_rank", type=int, default=16,
                       help="LoRA rank")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                       help="Learning rate for fine-tuning")
    parser.add_argument("--batch_size", type=int, default=None,
                       help="Batch size (auto-detected if not specified)")
    parser.add_argument("--max_epochs", type=int, default=3,
                       help="Maximum number of epochs")
    parser.add_argument("--freeze_layers", type=int, default=0,
                       help="Number of bottom layers to freeze")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to use")
    parser.add_argument("--output_dir", type=str, default="models",
                       help="Output directory for saved models")
    
    return parser

if __name__ == "__main__":
    # For command-line usage, uncomment the following:
    # parser = create_finetuning_script_with_args()
    # args = parser.parse_args()
    # 
    # # Override global variables with command line arguments
    # pretrained_model_path = args.pretrained_model
    # use_lora = args.use_lora
    # finetune_data_path = args.data_path
    # # ... etc
    
    exit(main())