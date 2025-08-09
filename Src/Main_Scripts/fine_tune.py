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

# Try to import optional memory optimization libraries
try:
    import bitsandbytes as bnb
    HAS_BITSANDBYTES = True
except ImportError:
    HAS_BITSANDBYTES = False

try:
    from peft import get_peft_model, LoraConfig, TaskType, PeftModel
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False

try:
    from accelerate import Accelerator, init_empty_weights, load_checkpoint_and_dispatch
    from accelerate.utils import set_seed
    HAS_ACCELERATE = True
except ImportError:
    HAS_ACCELERATE = False

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
        
        # Check optional libraries for memory optimization
        logger.info(f"BitsAndBytes available: {HAS_BITSANDBYTES}")
        logger.info(f"PEFT available: {HAS_PEFT}")
        logger.info(f"Accelerate available: {HAS_ACCELERATE}")
        
        # MPS availability
        if hasattr(torch.backends, 'mps'):
            logger.info(f"MPS available: {torch.backends.mps.is_available()}")
            logger.info(f"MPS built: {torch.backends.mps.is_built()}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Environment check failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

# Enhanced Configuration classes with memory optimization options
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
    gradient_checkpointing: bool = False  # Enable gradient checkpointing

@dataclass
class FineTuningConfig:
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    max_epochs: int = 5
    warmup_ratio: float = 0.1
    save_every: int = 500
    eval_every: int = 100
    max_grad_norm: float = 0.5
    label_smoothing: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    freeze_embeddings: bool = False
    freeze_layers: int = 0
    
    # LoRA settings
    lora_enabled: bool = False
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = None
    
    # Memory optimization settings
    use_mixed_precision: bool = True  # Enable FP16/BF16
    use_gradient_checkpointing: bool = True  # Enable gradient checkpointing
    use_cpu_offload: bool = False  # Offload parts to CPU
    use_quantization: bool = False  # Use quantized training
    quantization_bits: int = 8  # 4 or 8 bit quantization
    dataloader_num_workers: int = 0  # Reduce for memory savings
    pin_memory: bool = False  # Disable to save memory
    
    # 8-bit optimizer settings
    use_8bit_optimizer: bool = False  # Use 8-bit AdamW
    
    # Memory cleanup settings
    aggressive_memory_cleanup: bool = True
    empty_cache_every_n_steps: int = 10

@dataclass
class ModelMetadata:
    model_name: str = "transformer"
    version: str = "v1.0"
    created_at: str = ""
    last_modified: str = ""
    model_config: ModelConfig = None
    training_config: FineTuningConfig = None
    pretrained_model: str = None
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
    memory_optimizations: dict = None  # Track which optimizations were used
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

class MemoryOptimizedLinear(nn.Module):
    """Memory-optimized linear layer with optional quantization."""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True, 
                 use_quantization: bool = False, quantization_bits: int = 8):
        super().__init__()
        self.use_quantization = use_quantization and HAS_BITSANDBYTES
        
        if self.use_quantization:
            if quantization_bits == 8:
                self.linear = bnb.nn.Linear8bitLt(in_features, out_features, bias=bias, has_fp16_weights=False)
            elif quantization_bits == 4:
                self.linear = bnb.nn.Linear4bit(in_features, out_features, bias=bias)
            else:
                logger.warning(f"Unsupported quantization bits: {quantization_bits}, falling back to standard linear")
                self.linear = nn.Linear(in_features, out_features, bias=bias)
        else:
            self.linear = nn.Linear(in_features, out_features, bias=bias)
    
    def forward(self, x):
        return self.linear(x)

class StableTransformer(nn.Module):
    """Memory-optimized transformer with gradient checkpointing and mixed precision support."""
    
    def __init__(self, config, training_config=None):
        super().__init__()
        self.config = config
        self.training_config = training_config or FineTuningConfig()
        
        # Enable gradient checkpointing if requested
        self.gradient_checkpointing = (
            config.gradient_checkpointing or 
            self.training_config.use_gradient_checkpointing
        )
        
        # Embeddings with proper scaling
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.pos_embeddings = nn.Embedding(config.seq_length, config.hidden_size)
        
        # Input normalization
        self.input_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        
        # Transformer layers
        self.layers = nn.ModuleList()
        for _ in range(config.num_layers):
            layer = TransformerBlock(config, self.training_config)
            self.layers.append(layer)
        
        # Output layers with optional quantization
        self.output_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.dropout = nn.Dropout(config.dropout)
        
        # Use memory-optimized linear layer for output
        self.lm_head = MemoryOptimizedLinear(
            config.hidden_size, 
            config.vocab_size, 
            bias=False,
            use_quantization=self.training_config.use_quantization,
            quantization_bits=self.training_config.quantization_bits
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Scale embeddings
        nn.init.normal_(self.embeddings.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embeddings.weight, mean=0.0, std=0.02)
        
        # Tie embeddings and output weights if not using quantization
        if not self.training_config.use_quantization:
            if hasattr(self.lm_head.linear, 'weight'):
                self.lm_head.linear.weight = self.embeddings.weight
    
    def _init_weights(self, module):
        """Improved weight initialization."""
        if isinstance(module, (nn.Linear, MemoryOptimizedLinear)):
            if hasattr(module, 'linear'):
                nn.init.normal_(module.linear.weight, mean=0.0, std=0.02)
                if module.linear.bias is not None:
                    nn.init.zeros_(module.linear.bias)
            elif hasattr(module, 'weight'):
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
    
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory savings."""
        self.gradient_checkpointing = True
        logger.info("Gradient checkpointing enabled")
    
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
        
        # Apply transformer layers with optional gradient checkpointing
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    layer, hidden_states, attention_mask, use_reentrant=False
                )
            else:
                hidden_states = layer(hidden_states, attention_mask)
        
        # Output normalization and projection
        hidden_states = self.output_norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        return logits

class TransformerBlock(nn.Module):
    """Transformer block with memory optimizations."""
    
    def __init__(self, config, training_config=None):
        super().__init__()
        self.config = config
        self.training_config = training_config or FineTuningConfig()
        
        # Pre-normalization layers
        self.ln_1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.ln_2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        
        # Attention
        self.attn = MultiHeadAttention(config, training_config)
        
        # MLP
        self.mlp = MLP(config, training_config)
        
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
    """Memory-optimized multi-head attention."""
    
    def __init__(self, config, training_config=None):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.scale = self.head_dim ** -0.5
        self.training_config = training_config or FineTuningConfig()
        
        assert config.hidden_size % config.num_heads == 0
        
        # Use memory-optimized linear layers
        self.qkv = MemoryOptimizedLinear(
            config.hidden_size, 
            3 * config.hidden_size, 
            bias=False,
            use_quantization=self.training_config.use_quantization,
            quantization_bits=self.training_config.quantization_bits
        )
        self.out_proj = MemoryOptimizedLinear(
            config.hidden_size, 
            config.hidden_size, 
            bias=False,
            use_quantization=self.training_config.use_quantization,
            quantization_bits=self.training_config.quantization_bits
        )
        
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x, attention_mask=None):
        batch_size, seq_len, hidden_size = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2), qkv)
        
        # Use memory-efficient attention if available
        if hasattr(F, 'scaled_dot_product_attention'):
            # PyTorch 2.0+ memory-efficient attention
            if attention_mask is not None:
                # Convert boolean mask to float mask for scaled_dot_product_attention
                float_mask = torch.zeros_like(attention_mask, dtype=torch.float)
                float_mask.masked_fill_(attention_mask, float('-inf'))
                attention_mask = float_mask.unsqueeze(0).unsqueeze(0)
            
            attn_output = F.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=attention_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=attention_mask is None  # Use causal attention if no mask provided
            )
        else:
            # Fallback to manual attention computation
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            
            if attention_mask is not None:
                scores = scores.masked_fill(attention_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = torch.nan_to_num(attn_weights, nan=0.0, posinf=0.0, neginf=0.0)
            attn_weights = self.dropout(attn_weights)
            attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        output = self.out_proj(attn_output)
        
        return output

class MLP(nn.Module):
    """Memory-optimized MLP block."""
    
    def __init__(self, config, training_config=None):
        super().__init__()
        self.training_config = training_config or FineTuningConfig()
        
        self.fc1 = MemoryOptimizedLinear(
            config.hidden_size, 
            4 * config.hidden_size,
            use_quantization=self.training_config.use_quantization,
            quantization_bits=self.training_config.quantization_bits
        )
        self.fc2 = MemoryOptimizedLinear(
            4 * config.hidden_size, 
            config.hidden_size,
            use_quantization=self.training_config.use_quantization,
            quantization_bits=self.training_config.quantization_bits
        )
        
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Enhanced memory management and device setup
@contextmanager
def memory_cleanup(aggressive=True):
    """Context manager for memory cleanup with aggressive option."""
    try:
        yield
    finally:
        if aggressive:
            # More thorough cleanup
            for obj in gc.get_objects():
                if torch.is_tensor(obj):
                    try:
                        obj.detach_()
                        del obj
                    except:
                        pass
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.ipc_collect()
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()

def get_memory_usage():
    """Get detailed current memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return f"CUDA: {allocated:.2f}GB alloc, {cached:.2f}GB cached, {max_allocated:.2f}GB max, {total:.1f}GB total"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        allocated = torch.mps.current_allocated_memory() / 1024**3
        return f"MPS: {allocated:.2f}GB allocated"
    else:
        return "CPU mode"

def setup_device_optimized(config: FineTuningConfig):
    """Setup device with advanced memory optimizations."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using device: CUDA ({torch.cuda.get_device_name()})")
        
        # More aggressive memory management for fine-tuning
        torch.cuda.set_per_process_memory_fraction(0.90)  # Use more VRAM
        torch.cuda.empty_cache()
        
        # Enable memory efficiency optimizations
        if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
            torch.backends.cuda.enable_flash_sdp(True)
        if hasattr(torch.backends.cuda, 'enable_math_sdp'):
            torch.backends.cuda.enable_math_sdp(True)
        if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using device: MPS (Apple Silicon)")
        torch.mps.empty_cache()
    else:
        device = torch.device("cpu")
        logger.info("Using device: CPU")
        torch.set_num_threads(min(4, os.cpu_count() or 1))
    
    logger.info(f"Initial memory: {get_memory_usage()}")
    return device

class ModelManager:
    """Enhanced model manager with memory optimization support."""
    
    def __init__(self, save_dir):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
    
    def load_pretrained_model(self, model_path: Union[str, Path], 
                             training_config: Optional[FineTuningConfig] = None) -> Tuple[nn.Module, ImprovedTokenizer, dict]:
        """Load a pre-trained model with memory optimizations."""
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
        
        # Update model config with training-specific settings
        if training_config:
            model_config.gradient_checkpointing = training_config.use_gradient_checkpointing
        
        # Load tokenizer
        tokenizer_file = model_path / "tokenizer.json"
        if not tokenizer_file.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_file}")
        
        with open(tokenizer_file, 'r') as f:
            tokenizer_data = json.load(f)
        
        tokenizer = ImprovedTokenizer()
        tokenizer.load_from_dict(tokenizer_data)
        
        # Create model with memory optimizations
        if training_config and HAS_ACCELERATE and training_config.use_cpu_offload:
            # Use accelerate for CPU offloading
            with init_empty_weights():
                model = StableTransformer(model_config, training_config)
            
            model = load_checkpoint_and_dispatch(
                model,
                str(model_path / "model.pth"),
                device_map="auto",
                offload_folder=str(model_path / "offload"),
                offload_state_dict=True
            )
        else:
            # Standard loading
            model = StableTransformer(model_config, training_config)
            
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
        """Prepare a model for fine-tuning with memory optimizations."""
        logger.info("🔧 Preparing model for fine-tuning...")
        
        memory_optimizations = []
        
        # Enable gradient checkpointing if requested
        if config.use_gradient_checkpointing:
            if hasattr(model, 'enable_gradient_checkpointing'):
                model.enable_gradient_checkpointing()
                memory_optimizations.append("gradient_checkpointing")
                logger.info("✓ Gradient checkpointing enabled")
        
        # Apply LoRA if enabled and PEFT is available
        if config.lora_enabled and HAS_PEFT:
            logger.info("Applying LoRA (Low-Rank Adaptation) with PEFT")
            
            # Define target modules for LoRA
            target_modules = config.lora_target_modules or ["qkv", "out_proj", "fc1", "fc2"]
            
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=target_modules,
                bias="none"
            )
            
            model = get_peft_model(model, peft_config)
            memory_optimizations.append(f"lora_rank_{config.lora_rank}")
            logger.info(f"✓ LoRA applied with rank={config.lora_rank}, alpha={config.lora_alpha}")
        
        elif config.lora_enabled and not HAS_PEFT:
            logger.warning("LoRA requested but PEFT not available. Install with: pip install peft")
        
        # Freeze embeddings if requested
        if config.freeze_embeddings:
            if hasattr(model, 'freeze_embeddings'):
                model.freeze_embeddings()
                memory_optimizations.append("freeze_embeddings")
            elif hasattr(model, 'base_model') and hasattr(model.base_model, 'freeze_embeddings'):
                model.base_model.freeze_embeddings()
                memory_optimizations.append("freeze_embeddings")
        
        # Freeze bottom layers if requested
        if config.freeze_layers > 0:
            if hasattr(model, 'freeze_layers'):
                model.freeze_layers(config.freeze_layers)
                memory_optimizations.append(f"freeze_{config.freeze_layers}_layers")
            elif hasattr(model, 'base_model') and hasattr(model.base_model, 'freeze_layers'):
                model.base_model.freeze_layers(config.freeze_layers)
                memory_optimizations.append(f"freeze_{config.freeze_layers}_layers")
        
        # Print parameter statistics
        if hasattr(model, 'print_trainable_parameters'):
            # PEFT model
            model.print_trainable_parameters()
        else:
            # Standard model
            total, trainable, frozen = self._get_parameter_counts(model)
            logger.info(f"Parameter counts after fine-tuning preparation:")
            logger.info(f"  Total: {total:,}")
            logger.info(f"  Trainable: {trainable:,} ({trainable/total*100:.1f}%)")
            logger.info(f"  Frozen: {frozen:,} ({frozen/total*100:.1f}%)")
        
        # Store memory optimizations info
        model._memory_optimizations = memory_optimizations
        
        return model
    
    def _get_parameter_counts(self, model):
        """Get parameter counts for any model."""
        if hasattr(model, 'get_parameter_counts'):
            return model.get_parameter_counts()
        else:
            total = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            frozen = total - trainable
            return total, trainable, frozen
    
    def save_model(self, model, tokenizer, metadata, optimizer=None, scheduler=None, force_cpu_save=True):
        """Save model with memory optimization tracking."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_id = f"finetuned_{timestamp}"
        model_path = self.save_dir / model_id
        model_path.mkdir(exist_ok=True)
        
        try:
            # Save PEFT adapter if it's a PEFT model
            if HAS_PEFT and hasattr(model, 'save_pretrained'):
                logger.info("Saving PEFT adapter weights...")
                model.save_pretrained(model_path / "peft_adapter")
                
                # Also save the base model state dict
                if hasattr(model, 'base_model'):
                    base_model_state = {k: v.cpu() for k, v in model.base_model.state_dict().items()}
                    torch.save(base_model_state, model_path / "base_model.pth")
            else:
                # ALWAYS save to CPU to prevent memory issues
                model_state = {k: v.cpu() for k, v in model.state_dict().items()}
                torch.save(model_state, model_path / "model.pth")
            
            # Move model back to device
            if hasattr(model, 'to'):
                model.to(device)
            
            # Save tokenizer
            tokenizer_data = {
                'vocab': tokenizer.vocab,
                'id_to_token': tokenizer.id_to_token,
                'vocab_size': tokenizer.vocab_size()
            }
            with open(model_path / "tokenizer.json", 'w') as f:
                json.dump(tokenizer_data, f, indent=2)
            
            # Enhanced metadata with memory optimization info
            if hasattr(metadata, '__dict__'):
                metadata_dict = metadata.__dict__.copy()
            else:
                metadata_dict = asdict(metadata) if hasattr(metadata, '__dataclass_fields__') else metadata
            
            # Add memory optimization info
            if hasattr(model, '_memory_optimizations'):
                metadata_dict['memory_optimizations'] = {
                    'optimizations_applied': model._memory_optimizations,
                    'mixed_precision': getattr(metadata.training_config, 'use_mixed_precision', False),
                    'gradient_checkpointing': getattr(metadata.training_config, 'use_gradient_checkpointing', False),
                    'quantization': getattr(metadata.training_config, 'use_quantization', False),
                    'cpu_offload': getattr(metadata.training_config, 'use_cpu_offload', False)
                }
            
            with open(model_path / "metadata.json", 'w') as f:
                json.dump(metadata_dict, f, indent=2, default=str)
            
            # Save fine-tuning specific info
            finetune_info = {
                'is_finetuned': True,
                'pretrained_model': metadata.pretrained_model,
                'is_peft_model': HAS_PEFT and hasattr(model, 'save_pretrained'),
                'finetuning_config': asdict(metadata.training_config) if hasattr(metadata.training_config, '__dataclass_fields__') else metadata.training_config.__dict__,
                'parameter_counts': {
                    'total': metadata.total_parameters,
                    'trainable': metadata.trainable_parameters,
                    'frozen': metadata.frozen_parameters
                },
                'memory_optimizations': getattr(model, '_memory_optimizations', [])
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
            
            # Check for memory optimization info
            finetune_info_file = model_path / "finetune_info.json"
            optimizations = []
            if finetune_info_file.exists():
                try:
                    with open(finetune_info_file, 'r') as f:
                        info = json.load(f)
                        optimizations = info.get('memory_optimizations', [])
                except:
                    pass
            
            opt_str = f" [{', '.join(optimizations)}]" if optimizations else ""
            logger.info(f"  - {model_path.name} ({model_type}){opt_str}")

device = None

class StableDataset(Dataset):
    """Memory-efficient dataset with reduced memory footprint."""
    
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
        
        # Convert to more memory-efficient format
        self.sequences = torch.tensor(self.sequences, dtype=torch.long)
        logger.info(f"Dataset converted to tensor: {self.sequences.shape}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        if idx >= len(self.sequences):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.sequences)}")
            
        sequence = self.sequences[idx]
        input_ids = sequence[:-1]
        target_ids = sequence[1:]
        
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

def get_optimized_optimizer(model, config: FineTuningConfig):
    """Get memory-optimized optimizer."""
    
    # Filter parameters that require gradients
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    
    if config.use_8bit_optimizer and HAS_BITSANDBYTES:
        logger.info("Using 8-bit AdamW optimizer for memory savings")
        optimizer = bnb.optim.AdamW8bit(
            params_to_optimize,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(config.beta1, config.beta2),
            eps=1e-8
        )
    else:
        logger.info("Using standard AdamW optimizer")
        optimizer = optim.AdamW(
            params_to_optimize,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(config.beta1, config.beta2),
            eps=1e-8
        )
    
    return optimizer

def finetune_epoch_optimized(model, dataloader, criterion, optimizer, scheduler, epoch, 
                            config: FineTuningConfig, accelerator=None):
    """Memory-optimized fine-tuning epoch with mixed precision and gradient checkpointing."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    num_batches = 0
    accumulation_count = 0
    
    # Setup mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if config.use_mixed_precision and torch.cuda.is_available() else None
    
    optimizer.zero_grad()
    
    logger.info(f"Starting memory-optimized fine-tuning epoch {epoch} with {len(dataloader)} batches")
    logger.info(f"Memory optimizations: mixed_precision={config.use_mixed_precision}, "
               f"gradient_checkpointing={config.use_gradient_checkpointing}")
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        try:
            # Aggressive memory cleanup
            if config.aggressive_memory_cleanup and batch_idx % config.empty_cache_every_n_steps == 0:
                with memory_cleanup(aggressive=True):
                    pass
                if batch_idx > 0:
                    logger.info(f"Memory cleanup at batch {batch_idx}: {get_memory_usage()}")
            
            if inputs.numel() == 0 or targets.numel() == 0:
                logger.warning(f"Empty batch at index {batch_idx}, skipping")
                continue
                
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            if torch.isnan(inputs).any() or torch.isinf(inputs).any():
                logger.warning(f"Invalid inputs at batch {batch_idx}, skipping")
                continue
            
            # Mixed precision forward pass
            if config.use_mixed_precision and scaler is not None:
                with torch.cuda.amp.autocast():
                    logits = model(inputs)
                    
                    if torch.isnan(logits).any() or torch.isinf(logits).any():
                        logger.warning(f"Invalid logits at batch {batch_idx}, skipping")
                        optimizer.zero_grad()
                        continue
                    
                    flat_logits = logits.view(-1, logits.size(-1))
                    flat_targets = targets.view(-1)
                    loss = criterion(flat_logits, flat_targets)
            else:
                # Standard precision
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
            
            # Backward pass with mixed precision
            scaled_loss = loss / config.gradient_accumulation_steps
            
            if config.use_mixed_precision and scaler is not None:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            
            # Clean up intermediate tensors
            del logits, flat_logits, flat_targets, scaled_loss
            
            accumulation_count += 1
            
            # Optimizer step with gradient accumulation
            if accumulation_count >= config.gradient_accumulation_steps:
                if config.max_grad_norm > 0:
                    if config.use_mixed_precision and scaler is not None:
                        scaler.unscale_(optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    else:
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    
                    if grad_norm > config.max_grad_norm * 5:
                        logger.warning(f"Large gradients detected: {grad_norm:.2f}, reducing LR")
                        for param_group in optimizer.param_groups:
                            param_group['lr'] *= 0.8
                
                if config.use_mixed_precision and scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                current_lr = scheduler.step()
                optimizer.zero_grad()
                accumulation_count = 0
            
            # Calculate accuracy (without storing gradients)
            with torch.no_grad():
                if config.use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        pred_logits = model(inputs)
                else:
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
            
            # Progress logging
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
                logger.warning(f"OOM at batch {batch_idx}, clearing cache and reducing batch size...")
                optimizer.zero_grad()
                
                # Clear all possible tensors
                for var_name in ['inputs', 'targets', 'logits', 'loss', 'scaled_loss', 'pred_logits', 'predictions']:
                    if var_name in locals():
                        del locals()[var_name]
                
                # Aggressive memory cleanup
                with memory_cleanup(aggressive=True):
                    pass
                
                logger.info(f"Memory after OOM cleanup: {get_memory_usage()}")
                continue
            else:
                logger.error(f"Runtime error at batch {batch_idx}: {e}")
                raise e
        except Exception as e:
            logger.error(f"Unexpected error at batch {batch_idx}: {e}")
            # Clean up and continue
            for var_name in ['inputs', 'targets', 'logits', 'loss']:
                if var_name in locals():
                    del locals()[var_name]
            continue
    
    # Final gradient step if needed
    if accumulation_count > 0:
        if config.max_grad_norm > 0:
            if config.use_mixed_precision and scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
        optimizer.zero_grad()
    
    if num_batches == 0:
        logger.error("No batches processed! Check your dataset and dataloader.")
        return float('inf'), 0.0
    
    avg_loss = total_loss / num_batches
    avg_acc = total_correct / max(total_tokens, 1)
    
    logger.info(f"Fine-tuning epoch {epoch} completed: {num_batches} batches, {total_tokens} tokens processed")
    
    return avg_loss, avg_acc

def evaluate_model_optimized(model, dataloader, criterion, config: FineTuningConfig, max_batches=5):
    """Memory-optimized evaluation function."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    num_batches = 0
    
    logger.info(f"Starting optimized evaluation with max {max_batches} batches from {len(dataloader)} available")
    
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
                    
                    # Use mixed precision for evaluation too
                    if config.use_mixed_precision:
                        with torch.cuda.amp.autocast():
                            logits = model(inputs)
                    else:
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
                
                # Clean up memory during evaluation
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

def generate_sample_text_optimized(model, tokenizer, prompt="<user> Hello", max_length=50, config=None):
    """Memory-optimized sample text generation."""
    model.eval()
    
    try:
        with torch.no_grad():
            input_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
            generated = input_ids.clone()
            
            for _ in range(max_length):
                if generated.size(1) >= model.config.seq_length:
                    break
                
                # Use mixed precision for generation if enabled
                if config and config.use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        logits = model(generated)
                else:
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
        # Clean up generation tensors
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def get_finetuning_config_optimized(device_type: str, use_lora: bool = False, memory_aggressive: bool = True):
    """Get memory-optimized fine-tuning configuration."""
    base_config = {
        'use_mixed_precision': True,
        'use_gradient_checkpointing': True,
        'aggressive_memory_cleanup': memory_aggressive,
        'empty_cache_every_n_steps': 5 if memory_aggressive else 10,
        'dataloader_num_workers': 0,
        'pin_memory': False,
        'use_8bit_optimizer': HAS_BITSANDBYTES,
        'lora_enabled': use_lora and (HAS_PEFT or not use_lora)
    }
    
    if device_type == 'cuda':
        config = FineTuningConfig(
            learning_rate=2e-5,
            weight_decay=0.01,
            batch_size=2 if memory_aggressive else 4,
            gradient_accumulation_steps=16 if memory_aggressive else 8,
            max_epochs=3,
            warmup_ratio=0.1,
            max_grad_norm=0.5,
            label_smoothing=0.1,
            freeze_embeddings=False,
            freeze_layers=0,
            lora_rank=8 if memory_aggressive else 16,
            lora_alpha=16 if memory_aggressive else 32,
            lora_dropout=0.1,
            use_quantization=memory_aggressive and HAS_BITSANDBYTES,
            quantization_bits=8,
            **base_config
        )
        max_samples = 3000 if memory_aggressive else 5000
        
    elif device_type == 'mps':
        config = FineTuningConfig(
            learning_rate=1e-5,
            batch_size=1,
            gradient_accumulation_steps=24,
            max_epochs=2,
            freeze_layers=2,
            lora_rank=4,
            lora_alpha=8,
            use_mixed_precision=False,  # MPS doesn't support mixed precision yet
            use_8bit_optimizer=False,   # BitsAndBytes doesn't support MPS
            use_quantization=False,
            **{k: v for k, v in base_config.items() if k not in ['use_mixed_precision', 'use_8bit_optimizer']}
        )
        max_samples = 1500
        
    else:  # CPU
        config = FineTuningConfig(
            learning_rate=5e-6,
            batch_size=1,
            gradient_accumulation_steps=32,
            max_epochs=2,
            freeze_layers=6,
            lora_rank=4,
            lora_alpha=8,
            use_mixed_precision=False,
            use_8bit_optimizer=False,
            use_quantization=False,
            use_gradient_checkpointing=False,  # May be slower on CPU
            **{k: v for k, v in base_config.items() if k not in ['use_mixed_precision', 'use_8bit_optimizer', 'use_gradient_checkpointing']}
        )
        max_samples = 800
    
    return config, max_samples

def count_parameters(model):
    """Count model parameters with PEFT support."""
    if HAS_PEFT and hasattr(model, 'print_trainable_parameters'):
        # PEFT model - get the counts differently
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        frozen_params = total_params - trainable_params
        return total_params, trainable_params, frozen_params
    else:
        # Standard model
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen = total - trainable
        return total, trainable, frozen

def main():
    """Main memory-optimized fine-tuning function."""
    logger.info("🚀 Starting Memory-Optimized OASST1 Fine-Tuning")
    logger.info("=" * 80)
    
    if not check_environment():
        return 1
    
    # Parse arguments (in a real implementation, use argparse)
    pretrained_model_path = "models/model_20250108_123456"  # Example path
    use_lora = True  # Enable LoRA by default for memory efficiency
    memory_aggressive = True  # Enable aggressive memory optimizations
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
    
    # Setup memory-optimized device
    global device
    device = setup_device_optimized(FineTuningConfig())
    
    # Get optimized configuration
    finetuning_config, max_samples = get_finetuning_config_optimized(
        device.type, use_lora, memory_aggressive
    )
    
    logger.info(f"Memory-Optimized Fine-tuning Configuration:")
    logger.info(f"  Pre-trained model: {pretrained_model_path}")
    logger.info(f"  LoRA enabled: {finetuning_config.lora_enabled}")
    logger.info(f"  Mixed precision: {finetuning_config.use_mixed_precision}")
    logger.info(f"  Gradient checkpointing: {finetuning_config.use_gradient_checkpointing}")
    logger.info(f"  8-bit optimizer: {finetuning_config.use_8bit_optimizer}")
    logger.info(f"  Quantization: {finetuning_config.use_quantization}")
    logger.info(f"  Aggressive cleanup: {finetuning_config.aggressive_memory_cleanup}")
    logger.info(f"  Learning rate: {finetuning_config.learning_rate}")
    logger.info(f"  Batch size: {finetuning_config.batch_size}")
    logger.info(f"  Max epochs: {finetuning_config.max_epochs}")
    logger.info(f"  Max samples: {max_samples}")
    
    model_manager = ModelManager("models")
    
    # Initialize accelerator if available
    accelerator = None
    if HAS_ACCELERATE and finetuning_config.use_cpu_offload:
        accelerator = Accelerator(
            mixed_precision="fp16" if finetuning_config.use_mixed_precision else "no",
            gradient_accumulation_steps=finetuning_config.gradient_accumulation_steps
        )
        logger.info("Using Accelerate for distributed training and CPU offloading")
    
    try:
        # Load pre-trained model with memory optimizations
        logger.info("📂 Loading pre-trained model with memory optimizations...")
        with memory_cleanup():
            model, tokenizer, pretrained_metadata = model_manager.load_pretrained_model(
                pretrained_model_path, finetuning_config
            )
        
        # Move to device
        model = model.to(device)
        
        # Prepare model for fine-tuning with optimizations
        model = model_manager.prepare_model_for_finetuning(model, finetuning_config)
        
        logger.info(f"Memory after model preparation: {get_memory_usage()}")
        
        # Load fine-tuning data
        logger.info("📚 Loading fine-tuning data...")
        texts = load_and_process_data(finetune_data_path, max_samples)
        
        if len(texts) == 0:
            raise ValueError("No fine-tuning data loaded!")
        
        logger.info(f"Loaded {len(texts):,} texts for fine-tuning")
        
        # Create memory-efficient dataset
        logger.info("📦 Creating memory-efficient dataset...")
        with memory_cleanup():
            dataset = StableDataset(
                texts, 
                tokenizer, 
                model.config.seq_length,
                max_sequences=min(3000, len(texts) * 2)
            )
        
        logger.info(f"Created {len(dataset):,} fine-tuning sequences")
        logger.info(f"Memory after dataset creation: {get_memory_usage()}")
        
        # Create optimized dataloaders
        train_dataloader = DataLoader(
            dataset,
            batch_size=finetuning_config.batch_size,
            shuffle=True,
            num_workers=finetuning_config.dataloader_num_workers,
            pin_memory=finetuning_config.pin_memory,
            drop_last=True,
            persistent_workers=False
        )
        
        # Create evaluation dataset
        eval_size = min(50, len(dataset) // 20)  # Smaller eval set for memory
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
        
        # Setup memory-optimized optimizer and scheduler
        optimizer = get_optimized_optimizer(model, finetuning_config)
        
        pad_token_id = tokenizer.vocab.get("<pad>", 0)
        criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id, label_smoothing=finetuning_config.label_smoothing)
        
        total_steps = len(train_dataloader) * finetuning_config.max_epochs // finetuning_config.gradient_accumulation_steps
        warmup_steps = int(total_steps * finetuning_config.warmup_ratio)
        
        scheduler = ImprovedScheduler(optimizer, warmup_steps, total_steps)
        
        # Prepare with accelerator if available
        if accelerator:
            model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
                model, optimizer, train_dataloader, eval_dataloader
            )
        
        logger.info(f"Fine-tuning steps: {total_steps:,} (warmup: {warmup_steps:,})")
        logger.info(f"Memory before fine-tuning: {get_memory_usage()}")
        
        # Memory-optimized fine-tuning loop
        logger.info("🚀 Starting memory-optimized fine-tuning...")
        training_start = time.time()
        best_loss = float('inf')
        best_accuracy = 0.0
        models_saved = 0
        
        for epoch in range(1, finetuning_config.max_epochs + 1):
            epoch_start = time.time()
            
            logger.info(f"=== Memory-Optimized Fine-tuning Epoch {epoch}/{finetuning_config.max_epochs} ===")
            
            try:
                # Memory-optimized training epoch
                train_loss, train_acc = finetune_epoch_optimized(
                    model, train_dataloader, criterion, optimizer, scheduler, epoch, 
                    finetuning_config, accelerator
                )
                
                if math.isnan(train_loss) or math.isinf(train_loss):
                    logger.error(f"Invalid training loss: {train_loss}")
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= 0.5
                    logger.info(f"Reduced learning rate to {optimizer.param_groups[0]['lr']}")
                    continue
                
                perplexity = math.exp(min(train_loss, 20))
                epoch_time = time.time() - epoch_start
                
                logger.info(f"Memory-Optimized Fine-tuning Results:")
                logger.info(f"  Loss: {train_loss:.4f}")
                logger.info(f"  Accuracy: {train_acc:.3f} ({train_acc*100:.1f}%)")
                logger.info(f"  Perplexity: {perplexity:.2f}")
                logger.info(f"  Time: {epoch_time:.1f}s")
                logger.info(f"  Memory: {get_memory_usage()}")
                
                # Memory-optimized evaluation
                eval_results = None
                if epoch % 1 == 0:
                    logger.info("📊 Memory-optimized evaluation...")
                    with memory_cleanup():
                        eval_results = evaluate_model_optimized(
                            model, eval_dataloader, criterion, finetuning_config, max_batches=3
                        )
                    
                    logger.info(f"Evaluation Results:")
                    logger.info(f"  Loss: {eval_results['avg_loss']:.4f}")
                    logger.info(f"  Accuracy: {eval_results['accuracy']:.3f} ({eval_results['accuracy']*100:.1f}%)")
                    logger.info(f"  Perplexity: {eval_results['perplexity']:.2f}")
                
                # Memory-optimized sample generation
                if epoch % 1 == 0:
                    with memory_cleanup():
                        sample = generate_sample_text_optimized(
                            model, tokenizer, "<user> Hello, how are you?", max_length=30, config=finetuning_config
                        )
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
                
                # Save model with memory optimizations info
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
                        "memory_optimizations_used": {
                            "mixed_precision": finetuning_config.use_mixed_precision,
                            "gradient_checkpointing": finetuning_config.use_gradient_checkpointing,
                            "lora": finetuning_config.lora_enabled,
                            "8bit_optimizer": finetuning_config.use_8bit_optimizer,
                            "quantization": finetuning_config.use_quantization,
                            "aggressive_cleanup": finetuning_config.aggressive_memory_cleanup
                        }
                    }
                    
                    if eval_results:
                        performance_metrics.update({
                            "eval_loss": float(eval_results['avg_loss']),
                            "eval_accuracy": float(eval_results['accuracy']),
                            "eval_perplexity": float(eval_results['perplexity']),
                        })
                    
                    metadata = ModelMetadata(
                        model_name="OASST1_MemoryOptimized_FineTuned",
                        version=f"v1.0_ft_epoch_{epoch}_optimized",
                        created_at=datetime.now().isoformat(),
                        model_config=model.config if hasattr(model, 'config') else model.base_model.config,
                        training_config=finetuning_config,
                        pretrained_model=str(pretrained_model_path),
                        dataset_info={
                            "name": "OpenAssistant OASST1 Memory-Optimized Fine-tuning",
                            "num_samples": len(texts),
                            "vocab_size": tokenizer.vocab_size(),
                            "seq_length": model.config.seq_length if hasattr(model, 'config') else model.base_model.config.seq_length,
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
                        memory_optimizations={
                            "mixed_precision": finetuning_config.use_mixed_precision,
                            "gradient_checkpointing": finetuning_config.use_gradient_checkpointing,
                            "lora_enabled": finetuning_config.lora_enabled,
                            "lora_rank": finetuning_config.lora_rank if finetuning_config.lora_enabled else None,
                            "8bit_optimizer": finetuning_config.use_8bit_optimizer,
                            "quantization": finetuning_config.use_quantization,
                            "cpu_offload": finetuning_config.use_cpu_offload,
                            "aggressive_cleanup": finetuning_config.aggressive_memory_cleanup
                        },
                        notes=f"Memory-optimized fine-tuning from {pretrained_model_path} on epoch {epoch}" + 
                              (f" with LoRA (rank={finetuning_config.lora_rank})" if finetuning_config.lora_enabled else "") +
                              f" using {device.type.upper()} with aggressive memory optimizations",
                        tags=["oasst1", "finetuned", "memory_optimized", f"epoch_{epoch}"] + 
                             (["lora"] if finetuning_config.lora_enabled else []) +
                             (["mixed_precision"] if finetuning_config.use_mixed_precision else []) +
                             (["gradient_checkpointing"] if finetuning_config.use_gradient_checkpointing else []) +
                             (["8bit_optimizer"] if finetuning_config.use_8bit_optimizer else []) +
                             (["quantized"] if finetuning_config.use_quantization else []) +
                             (["best_loss"] if is_best_loss else []) +
                             (["best_accuracy"] if is_best_acc else [])
                    )
                    
                    try:
                        with memory_cleanup(aggressive=True):
                            model_id = model_manager.save_model(model, tokenizer, metadata)
                            if model_id:
                                models_saved += 1
                                logger.info(f"💾 Memory-optimized model saved: {model_id}")
                    except Exception as save_error:
                        logger.error(f"Failed to save model: {save_error}")
                
                logger.info(f"Memory after epoch: {get_memory_usage()}")
                
                # Aggressive memory cleanup between epochs
                with memory_cleanup(aggressive=True):
                    pass
                
            except Exception as e:
                logger.error(f"Error in fine-tuning epoch {epoch}: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                
                if "out of memory" in str(e).lower():
                    logger.info("Attempting aggressive OOM recovery...")
                    
                    optimizer.zero_grad()
                    
                    # Clear all possible variables
                    for var_name in ['inputs', 'targets', 'logits', 'loss', 'predictions']:
                        if var_name in locals():
                            del locals()[var_name]
                    
                    with memory_cleanup(aggressive=True):
                        pass
                    
                    # Further reduce batch size and increase accumulation
                    current_batch_size = finetuning_config.batch_size
                    if current_batch_size > 1:
                        finetuning_config.batch_size = 1
                        finetuning_config.gradient_accumulation_steps *= current_batch_size
                        logger.info(f"Reduced batch size to 1, increased grad accumulation to {finetuning_config.gradient_accumulation_steps}")
                        
                        train_dataloader = DataLoader(
                            dataset,
                            batch_size=1,
                            shuffle=True,
                            num_workers=0,
                            pin_memory=False,
                            drop_last=True
                        )
                        
                        if accelerator:
                            model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
                                model, optimizer, train_dataloader, eval_dataloader
                            )
                    
                    logger.info(f"Memory after OOM recovery: {get_memory_usage()}")
                    continue
                else:
                    raise e
        
        # Fine-tuning completion
        total_time = time.time() - training_start
        
        logger.info("=" * 80)
        logger.info("✅ Memory-optimized fine-tuning completed successfully!")
        logger.info(f"Pre-trained model: {pretrained_model_path}")
        logger.info(f"Best loss: {best_loss:.4f}")
        logger.info(f"Best accuracy: {best_accuracy:.3f} ({best_accuracy*100:.1f}%)")
        logger.info(f"Best perplexity: {math.exp(min(best_loss, 20)):.2f}")
        logger.info(f"Models saved: {models_saved}")
        logger.info(f"Fine-tuning time: {total_time/3600:.2f} hours")
        logger.info(f"Parameter efficiency: {trainable_params/total_params*100:.1f}% trainable")
        
        # Print memory optimization summary
        logger.info("\n🧠 Memory Optimizations Applied:")
        if finetuning_config.use_mixed_precision:
            logger.info("  ✓ Mixed precision (FP16/BF16) training")
        if finetuning_config.use_gradient_checkpointing:
            logger.info("  ✓ Gradient checkpointing")
        if finetuning_config.lora_enabled:
            logger.info(f"  ✓ LoRA (rank={finetuning_config.lora_rank}, alpha={finetuning_config.lora_alpha})")
        if finetuning_config.use_8bit_optimizer:
            logger.info("  ✓ 8-bit optimizer")
        if finetuning_config.use_quantization:
            logger.info(f"  ✓ {finetuning_config.quantization_bits}-bit quantization")
        if finetuning_config.aggressive_memory_cleanup:
            logger.info("  ✓ Aggressive memory cleanup")
        if finetuning_config.freeze_layers > 0:
            logger.info(f"  ✓ Frozen {finetuning_config.freeze_layers} layers")
        
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
                model_name="OASST1_MemoryOptimized_FINAL",
                version="v1.0_FT_FINAL_OPTIMIZED",
                created_at=datetime.now().isoformat(),
                model_config=model.config if hasattr(model, 'config') else model.base_model.config,
                training_config=finetuning_config,
                pretrained_model=str(pretrained_model_path),
                performance_metrics={"final_loss": float(best_loss), "final_accuracy": float(best_accuracy)},
                total_parameters=int(total_params),
                trainable_parameters=int(trainable_params),
                frozen_parameters=int(frozen_params),
                memory_optimizations=finetuning_config.__dict__,
                notes="Final memory-optimized fine-tuned save",
                tags=["oasst1", "finetuned", "memory_optimized", "final"]
            )
            
            with memory_cleanup(aggressive=True):
                final_id = model_manager.save_model(model, tokenizer, final_metadata)
                if final_id:
                    logger.info(f"Final save successful: {final_id}")
                    models_saved += 1
        
        # Generate final samples with memory optimization
        logger.info("\n🎯 Final Memory-Optimized Sample Generations:")
        test_prompts = [
            "<user> Hello, how are you?",
            "<user> What is machine learning?",
            "<user> Can you help me with Python?",
            "<user> Explain quantum computing simply."
        ]
        
        for prompt in test_prompts:
            try:
                with memory_cleanup():
                    response = generate_sample_text_optimized(
                        model, tokenizer, prompt, max_length=30, config=finetuning_config
                    )
                    logger.info(f"  {prompt} → {response}")
            except Exception as e:
                logger.warning(f"  {prompt} → Generation failed: {e}")
        
        model_manager.print_model_summary()
        
        return 0 if models_saved > 0 else 1
        
    except KeyboardInterrupt:
        logger.info("Memory-optimized fine-tuning interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Memory-optimized fine-tuning failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return 1
    finally:
        # Final aggressive cleanup
        with memory_cleanup(aggressive=True):
            pass

def create_finetuning_script_with_args():
    """Create a command-line interface for the memory-optimized fine-tuning script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Memory-optimized fine-tune OASST1 models")
    parser.add_argument("--pretrained_model", type=str, required=True,
                       help="Path to pre-trained model directory")
    parser.add_argument("--data_path", type=str, default="oasst1_data/oasst1_train.jsonl",
                       help="Path to fine-tuning data")
    parser.add_argument("--use_lora", action="store_true", default=True,
                       help="Enable LoRA fine-tuning (default: True)")
    parser.add_argument("--lora_rank", type=int, default=8,
                       help="LoRA rank (default: 8 for memory efficiency)")
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
    
    # Memory optimization arguments
    parser.add_argument("--no_mixed_precision", action="store_true",
                       help="Disable mixed precision training")
    parser.add_argument("--no_gradient_checkpointing", action="store_true",
                       help="Disable gradient checkpointing")
    parser.add_argument("--no_8bit_optimizer", action="store_true",
                       help="Disable 8-bit optimizer")
    parser.add_argument("--enable_quantization", action="store_true",
                       help="Enable model quantization")
    parser.add_argument("--quantization_bits", type=int, default=8, choices=[4, 8],
                       help="Quantization bits (4 or 8)")
    parser.add_argument("--no_aggressive_cleanup", action="store_true",
                       help="Disable aggressive memory cleanup")
    parser.add_argument("--cpu_offload", action="store_true",
                       help="Enable CPU offloading (requires accelerate)")
    parser.add_argument("--memory_aggressive", action="store_true", default=True,
                       help="Use aggressive memory optimization settings")
    
    return parser

# Example usage functions for different memory optimization scenarios

def example_memory_optimized_training():
    """Example of how to use the memory-optimized training."""
    
    print("🧠 Memory-Optimized Fine-Tuning Examples")
    print("=" * 50)
    
    print("\n1. Ultra Memory Efficient (for 8GB+ VRAM):")
    print("   - LoRA with rank 4-8")
    print("   - Mixed precision (FP16)")
    print("   - Gradient checkpointing")
    print("   - 8-bit optimizer")
    print("   - Batch size 1-2")
    print("   - Aggressive memory cleanup")
    
    print("\n2. Balanced Memory/Speed (for 16GB+ VRAM):")
    print("   - LoRA with rank 16")
    print("   - Mixed precision")
    print("   - Gradient checkpointing")
    print("   - Batch size 4")
    print("   - Standard memory cleanup")
    
    print("\n3. CPU/MPS Optimized:")
    print("   - LoRA with rank 4")
    print("   - No mixed precision (not supported)")
    print("   - Freeze more layers")
    print("   - Batch size 1")
    print("   - CPU offloading")
    
    print("\n📋 Memory Optimization Checklist:")
    print("✓ Install optional dependencies:")
    print("  pip install bitsandbytes peft accelerate")
    print("✓ Monitor memory usage during training")
    print("✓ Adjust batch size based on available VRAM")
    print("✓ Use LoRA for parameter-efficient fine-tuning")
    print("✓ Enable gradient checkpointing for 30-50% memory savings")
    print("✓ Use mixed precision for ~50% memory reduction")
    print("✓ Consider 8-bit quantization for inference")

def estimate_memory_requirements(model_params_millions, batch_size=1, seq_length=512, use_optimizations=True):
    """Estimate VRAM requirements for fine-tuning."""
    
    # Base model memory (parameters + gradients + optimizer states)
    model_memory_gb = model_params_millions * 4 * 3 / 1024  # 4 bytes per param, 3x for gradients+optimizer
    
    # Activation memory (depends on batch size and sequence length)
    activation_memory_gb = batch_size * seq_length * model_params_millions * 4 / (1024**3)
    
    total_memory_gb = model_memory_gb + activation_memory_gb
    
    if use_optimizations:
        # Apply memory optimization reductions
        if True:  # Mixed precision
            total_memory_gb *= 0.6
        if True:  # Gradient checkpointing
            total_memory_gb *= 0.7
        if True:  # LoRA (only train 1-5% of parameters)
            total_memory_gb *= 0.3
        if True:  # 8-bit optimizer
            total_memory_gb *= 0.8
    
    return total_memory_gb

def print_memory_recommendations():
    """Print memory usage recommendations for different model sizes."""
    
    print("\n🎯 Memory Requirements Estimation (with optimizations)")
    print("=" * 60)
    
    model_sizes = [
        (125, "Small (125M params)"),
        (350, "Medium (350M params)"), 
        (760, "Large (760M params)"),
        (1300, "XL (1.3B params)"),
        (2700, "XXL (2.7B params)")
    ]
    
    for params_m, name in model_sizes:
        memory_needed = estimate_memory_requirements(params_m, batch_size=1, use_optimizations=True)
        print(f"{name:20} | ~{memory_needed:.1f}GB VRAM (batch_size=1)")
    
    print("\n💡 Tips for reducing memory usage:")
    print("• Use smaller batch sizes (1-2) with higher gradient accumulation")
    print("• Enable all memory optimizations (LoRA + mixed precision + checkpointing)")
    print("• Freeze more transformer layers")
    print("• Use shorter sequence lengths if possible")
    print("• Consider CPU offloading for very large models")

# Installation check and recommendation function
def check_and_recommend_installations():
    """Check optional dependencies and recommend installations."""
    
    print("\n📦 Dependency Check and Recommendations")
    print("=" * 50)
    
    # Check PyTorch version
    print(f"PyTorch: {torch.__version__} ✓")
    
    # Check optional memory optimization libraries
    if HAS_BITSANDBYTES:
        print("BitsAndBytes: Available ✓")
        print("  → Enables 8-bit optimizers and quantization")
    else:
        print("BitsAndBytes: Not installed ❌")
        print("  → Install with: pip install bitsandbytes")
        print("  → Enables 8-bit training and quantization")
    
    if HAS_PEFT:
        print("PEFT: Available ✓") 
        print("  → Enables LoRA and other parameter-efficient methods")
    else:
        print("PEFT: Not installed ❌")
        print("  → Install with: pip install peft")
        print("  → Enables LoRA for memory-efficient fine-tuning")
    
    if HAS_ACCELERATE:
        print("Accelerate: Available ✓")
        print("  → Enables distributed training and CPU offloading")
    else:
        print("Accelerate: Not installed ❌") 
        print("  → Install with: pip install accelerate")
        print("  → Enables mixed precision and model distribution")
    
    print(f"\nCUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f}GB)")
    
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("MPS (Apple Silicon): Available ✓")
    
    print("\n🚀 Recommended installation for optimal memory usage:")
    print("pip install torch torchvision torchaudio")
    print("pip install bitsandbytes peft accelerate")
    
    print("\n⚡ For maximum memory efficiency, ensure all dependencies are installed!")

if __name__ == "__main__":
    # Print helpful information
    print("🧠 Memory-Optimized Fine-Tuning for OASST1")
    print("=" * 50)
    
    # Check dependencies
    check_and_recommend_installations()
    
    # Show memory recommendations  
    print_memory_recommendations()
    
    # Show examples
    example_memory_optimized_training()
    
    print("\n" + "=" * 50)
    print("🚀 Starting fine-tuning...")

    exit(main())