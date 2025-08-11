# Copyright (c) 2025 Enhanced Training System. All rights reserved.
# Licensed under the Enhanced Custom License.

import os
import sys
import json
import time
import math
import logging
import traceback
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, asdict, field
from collections import defaultdict, deque
from enum import Enum
import threading
import queue

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.nn.functional as F
import torch.distributed as dist
import gc
from contextlib import contextmanager, nullcontext
import psutil
import numpy as np

# Enhanced imports with better error handling
try:
    import deepspeed
    from deepspeed.ops.adam import FusedAdam
    from deepspeed.runtime.zero.stage_1_and_2 import estimate_zero2_model_states_mem_needs_all_live
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False

try:
    from torch.cuda.amp import GradScaler, autocast
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False

try:
    from torch.utils.checkpoint import checkpoint
    CHECKPOINT_AVAILABLE = True
except ImportError:
    CHECKPOINT_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================================
# ENHANCED CONFIGURATION SYSTEM
# ============================================================================

class PrecisionType(Enum):
    """Enumeration for supported precision types."""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"

class OptimizationLevel(Enum):
    """Optimization levels for memory and performance."""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    EXTREME = "extreme"

class DeviceType(Enum):
    """Supported device types."""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"

@dataclass
class HardwareConfig:
    """Hardware-specific configuration."""
    device_type: DeviceType
    device_name: str
    total_memory_gb: float
    compute_capability: Optional[Tuple[int, int]] = None
    supports_bf16: bool = False
    supports_flash_attention: bool = False
    tensor_cores_available: bool = False
    memory_bandwidth_gbps: Optional[float] = None
    
    def __post_init__(self):
        """Auto-detect hardware capabilities."""
        if self.device_type == DeviceType.CUDA and torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            self.compute_capability = (props.major, props.minor)
            self.tensor_cores_available = props.major >= 7
            self.supports_bf16 = props.major >= 8 or (props.major == 7 and props.minor >= 5)
            # Rough memory bandwidth estimates based on GPU generation
            if props.major >= 8:  # Ampere+
                self.memory_bandwidth_gbps = 900.0  # Rough estimate
            elif props.major == 7:  # Turing/Volta
                self.memory_bandwidth_gbps = 600.0
            
    def get_optimal_precision(self) -> PrecisionType:
        """Determine optimal precision based on hardware."""
        if self.device_type == DeviceType.CUDA:
            if self.supports_bf16:
                return PrecisionType.BF16
            elif self.tensor_cores_available:
                return PrecisionType.FP16
            else:
                return PrecisionType.FP32
        else:
            return PrecisionType.FP32

@dataclass
class ModelConfig:
    """Enhanced model configuration with auto-scaling."""
    vocab_size: int = 50000
    hidden_size: int = 2048
    num_layers: int = 24
    num_heads: int = 16
    seq_length: int = 1024
    dropout: float = 0.1
    model_type: str = "transformer"
    
    # Enhanced features
    use_rotary_embeddings: bool = True
    use_grouped_query_attention: bool = False
    num_kv_heads: Optional[int] = None
    use_swiglu: bool = True
    gradient_checkpointing: bool = True
    use_flash_attention: bool = False
    layer_norm_eps: float = 1e-6
    initializer_range: float = 0.02
    
    # Memory optimization
    tie_word_embeddings: bool = True
    use_cache: bool = False
    
    def __post_init__(self):
        """Auto-configure based on settings."""
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_heads // 4 if self.use_grouped_query_attention else self.num_heads
        
        # Ensure head dimensions are valid
        assert self.hidden_size % self.num_heads == 0
        self.head_dim = self.hidden_size // self.num_heads
    
    @classmethod
    def get_optimized_config(cls, hardware_config: HardwareConfig, 
                           optimization_level: OptimizationLevel = OptimizationLevel.BALANCED):
        """Create optimized config based on hardware and optimization level."""
        memory_gb = hardware_config.total_memory_gb
        
        if optimization_level == OptimizationLevel.EXTREME or memory_gb < 4:
            return cls(
                vocab_size=8000,
                hidden_size=384,
                num_layers=6,
                num_heads=6,
                seq_length=256,
                dropout=0.1,
                use_grouped_query_attention=True,
                gradient_checkpointing=True,
            )
        elif optimization_level == OptimizationLevel.AGGRESSIVE or memory_gb < 8:
            return cls(
                vocab_size=16000,
                hidden_size=768,
                num_layers=12,
                num_heads=12,
                seq_length=512,
                dropout=0.1,
                use_grouped_query_attention=True,
                gradient_checkpointing=True,
            )
        elif optimization_level == OptimizationLevel.BALANCED or memory_gb < 16:
            return cls(
                vocab_size=32000,
                hidden_size=1536,
                num_layers=18,
                num_heads=12,
                seq_length=1024,
                dropout=0.1,
                use_grouped_query_attention=False,
                gradient_checkpointing=True,
            )
        else:  # CONSERVATIVE or high memory
            return cls(
                vocab_size=50000,
                hidden_size=2048,
                num_layers=24,
                num_heads=16,
                seq_length=2048,
                dropout=0.1,
                gradient_checkpointing=False,
            )

@dataclass
class TrainingConfig:
    """Enhanced training configuration."""
    # Learning parameters
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    max_epochs: int = 150
    max_steps: Optional[int] = None
    
    # Scheduling
    warmup_ratio: float = 0.1
    scheduler_type: str = "cosine"  # "linear", "cosine", "polynomial"
    min_lr_ratio: float = 0.1
    
    # Optimization
    optimizer_type: str = "adamw"  # "adam", "adamw", "adafactor"
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    max_grad_norm: float = 1.0
    label_smoothing: float = 0.0
    
    # Precision and memory
    precision_type: PrecisionType = PrecisionType.FP16
    use_mixed_precision: bool = True
    use_loss_scaling: bool = True
    gradient_checkpointing: bool = True
    
    # Data loading
    dataloader_num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    
    # DeepSpeed
    use_deepspeed: bool = True
    zero_stage: int = 2
    offload_optimizer: bool = False
    offload_params: bool = False
    
    # Monitoring and saving
    eval_every: int = 500
    save_every: int = 1000
    log_every: int = 10
    max_checkpoints: int = 5
    
    # Advanced features
    use_gradient_checkpointing: bool = True
    compile_model: bool = False  # torch.compile
    use_fused_ops: bool = True
    
    def __post_init__(self):
        """Auto-adjust settings based on precision type."""
        if self.precision_type == PrecisionType.BF16:
            self.use_loss_scaling = False
        elif self.precision_type == PrecisionType.FP32:
            self.use_mixed_precision = False
            self.use_loss_scaling = False

@dataclass 
class DataConfig:
    """Data processing configuration."""
    dataset_path: str = "oasst1_data/oasst1_train.jsonl"
    max_samples: Optional[int] = None
    validation_split: float = 0.1
    shuffle: bool = True
    
    # Text processing
    min_length: int = 10
    max_length: int = 2048
    truncate_strategy: str = "right"  # "left", "right", "middle"
    
    # Tokenization
    tokenizer_type: str = "custom"  # "custom", "hf"
    tokenizer_model: Optional[str] = None
    vocab_size: int = 32000
    
    # Special tokens
    pad_token: str = "<pad>"
    unk_token: str = "<unk>"
    bos_token: str = "<bos>"
    eos_token: str = "<eos>"
    user_token: str = "<user>"
    assistant_token: str = "<assistant>"

@dataclass
class ModelMetadata:
    """Comprehensive model metadata."""
    model_name: str = "enhanced_transformer"
    version: str = "v2.0"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_modified: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Configuration
    model_config: Optional[ModelConfig] = None
    training_config: Optional[TrainingConfig] = None
    data_config: Optional[DataConfig] = None
    hardware_config: Optional[HardwareConfig] = None
    
    # Training metrics
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    training_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Model info
    model_size_mb: float = 0.0
    total_parameters: int = 0
    trainable_parameters: int = 0
    
    # Training info
    epochs_trained: int = 0
    steps_trained: int = 0
    training_time_hours: float = 0.0
    best_loss: float = float('inf')
    best_perplexity: float = float('inf')
    convergence_epoch: Optional[int] = None
    
    # Environment
    pytorch_version: str = field(default_factory=lambda: torch.__version__)
    cuda_version: Optional[str] = field(default_factory=lambda: torch.version.cuda)
    python_version: str = field(default_factory=lambda: sys.version.split()[0])
    
    # Additional info
    notes: str = ""
    tags: List[str] = field(default_factory=list)
    git_hash: Optional[str] = None
    reproducibility_info: Dict[str, Any] = field(default_factory=dict)

# ============================================================================
# ENHANCED LOGGING AND MONITORING
# ============================================================================

class ColoredFormatter(logging.Formatter):
    """Colored console formatter."""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green  
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)

class MetricsTracker:
    """Advanced metrics tracking with moving averages and outlier detection."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics = defaultdict(lambda: deque(maxlen=window_size))
        self.global_metrics = defaultdict(list)
        self.best_metrics = {}
        
    def update(self, **kwargs):
        """Update metrics with outlier detection."""
        for key, value in kwargs.items():
            if not isinstance(value, (int, float)) or math.isnan(value) or math.isinf(value):
                continue
                
            self.metrics[key].append(value)
            self.global_metrics[key].append(value)
            
            # Track best values
            if key.endswith('_loss') or key.endswith('_error'):
                if key not in self.best_metrics or value < self.best_metrics[key]:
                    self.best_metrics[key] = value
            else:
                if key not in self.best_metrics or value > self.best_metrics[key]:
                    self.best_metrics[key] = value
    
    def get_average(self, key: str, window: Optional[int] = None) -> float:
        """Get moving average."""
        if key not in self.metrics:
            return 0.0
        values = list(self.metrics[key])
        if window:
            values = values[-window:]
        return sum(values) / len(values) if values else 0.0
    
    def get_trend(self, key: str, window: int = 20) -> str:
        """Get trend direction."""
        if key not in self.metrics or len(self.metrics[key]) < window:
            return "stable"
        
        values = list(self.metrics[key])[-window:]
        if len(values) < 2:
            return "stable"
            
        slope = np.polyfit(range(len(values)), values, 1)[0]
        if abs(slope) < 1e-6:
            return "stable"
        return "improving" if slope < 0 else "degrading"
    
    def detect_anomaly(self, key: str, value: float, threshold: float = 3.0) -> bool:
        """Detect if value is anomalous using z-score."""
        if key not in self.metrics or len(self.metrics[key]) < 10:
            return False
            
        values = list(self.metrics[key])
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        if std_val == 0:
            return False
            
        z_score = abs((value - mean_val) / std_val)
        return z_score > threshold

def setup_enhanced_logging(log_file: str = "enhanced_training.log") -> logging.Logger:
    """Setup comprehensive logging system."""
    logger = logging.getLogger("enhanced_training")
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = ColoredFormatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(log_file, mode='w')
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(funcName)s:%(lineno)d | %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger

logger = setup_enhanced_logging()

# ============================================================================
# HARDWARE DETECTION AND OPTIMIZATION
# ============================================================================

class HardwareDetector:
    """Comprehensive hardware detection and optimization."""
    
    @staticmethod
    def detect_hardware() -> HardwareConfig:
        """Detect and configure hardware settings."""
        if torch.cuda.is_available():
            device_type = DeviceType.CUDA
            props = torch.cuda.get_device_properties(0)
            device_name = props.name
            total_memory_gb = props.total_memory / 1024**3
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device_type = DeviceType.MPS
            device_name = "Apple Silicon"
            # Rough estimate - MPS doesn't expose memory info
            total_memory_gb = psutil.virtual_memory().total / 1024**3 * 0.6  
        else:
            device_type = DeviceType.CPU
            device_name = "CPU"
            total_memory_gb = psutil.virtual_memory().total / 1024**3
        
        return HardwareConfig(
            device_type=device_type,
            device_name=device_name,
            total_memory_gb=total_memory_gb
        )
    
    @staticmethod
    def optimize_cuda_settings(hardware_config: HardwareConfig):
        """Apply CUDA optimizations."""
        if hardware_config.device_type != DeviceType.CUDA:
            return
            
        # Memory management
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Memory allocation strategy
        memory_fraction = 0.85 if hardware_config.total_memory_gb >= 24 else 0.75
        torch.cuda.set_per_process_memory_fraction(memory_fraction)
        
        # Environment optimizations
        os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:128')
        os.environ.setdefault('CUDA_LAUNCH_BLOCKING', '0')
        
        logger.info(f"CUDA optimizations applied (memory fraction: {memory_fraction:.1%})")

# ============================================================================
# ENHANCED MEMORY MANAGEMENT
# ============================================================================

class MemoryManager:
    """Advanced memory management with monitoring and automatic optimization."""
    
    def __init__(self, device_type: DeviceType):
        self.device_type = device_type
        self.peak_memory = 0.0
        self.cleanup_threshold = 0.8  # Trigger cleanup at 80% memory usage
        
    def get_memory_stats(self) -> Dict[str, float]:
        """Get comprehensive memory statistics."""
        stats = {"cpu_memory_gb": psutil.virtual_memory().used / 1024**3}
        
        if self.device_type == DeviceType.CUDA and torch.cuda.is_available():
            stats.update({
                "gpu_allocated_gb": torch.cuda.memory_allocated() / 1024**3,
                "gpu_cached_gb": torch.cuda.memory_reserved() / 1024**3,
                "gpu_max_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
                "gpu_total_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3
            })
            self.peak_memory = max(self.peak_memory, stats["gpu_allocated_gb"])
        elif self.device_type == DeviceType.MPS:
            if hasattr(torch.mps, 'current_allocated_memory'):
                stats["mps_allocated_gb"] = torch.mps.current_allocated_memory() / 1024**3
        
        return stats
    
    def get_memory_usage_str(self) -> str:
        """Get formatted memory usage string."""
        stats = self.get_memory_stats()
        if self.device_type == DeviceType.CUDA:
            return f"GPU: {stats['gpu_allocated_gb']:.1f}GB/{stats['gpu_total_gb']:.1f}GB ({stats['gpu_allocated_gb']/stats['gpu_total_gb']*100:.1f}%)"
        elif self.device_type == DeviceType.MPS:
            return f"MPS: {stats.get('mps_allocated_gb', 0):.1f}GB"
        else:
            return f"CPU: {stats['cpu_memory_gb']:.1f}GB"
    
    def should_cleanup(self) -> bool:
        """Determine if memory cleanup is needed."""
        if self.device_type == DeviceType.CUDA:
            stats = self.get_memory_stats()
            return stats["gpu_allocated_gb"] / stats["gpu_total_gb"] > self.cleanup_threshold
        return False
    
    @contextmanager
    def cleanup_context(self, aggressive: bool = False):
        """Context manager for automatic memory cleanup."""
        try:
            yield
        finally:
            self.cleanup_memory(aggressive=aggressive)
    
    def cleanup_memory(self, aggressive: bool = False):
        """Perform memory cleanup."""
        # Python garbage collection
        if aggressive:
            for _ in range(3):
                gc.collect()
        else:
            gc.collect()
        
        # Device-specific cleanup
        if self.device_type == DeviceType.CUDA and torch.cuda.is_available():
            torch.cuda.empty_cache()
            if aggressive:
                torch.cuda.synchronize()
                torch.cuda.ipc_collect()
        elif self.device_type == DeviceType.MPS:
            if hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()

# ============================================================================
# ENHANCED TOKENIZER
# ============================================================================

class EnhancedTokenizer:
    """Production-ready tokenizer with subword support."""
    
    def __init__(self, data_config: DataConfig):
        self.data_config = data_config
        self.vocab = {}
        self.id_to_token = {}
        self.special_tokens = {
            data_config.pad_token: 0,
            data_config.unk_token: 1, 
            data_config.bos_token: 2,
            data_config.eos_token: 3,
            data_config.user_token: 4,
            data_config.assistant_token: 5
        }
        self.vocab.update(self.special_tokens)
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.trained = False
        
    def train_subword_tokenizer(self, texts: List[str], vocab_size: int = 32000) -> None:
        """Train a simple subword tokenizer using BPE-like approach."""
        logger.info(f"Training subword tokenizer with vocab_size={vocab_size}")
        
        # Collect character and word frequencies
        char_freq = defaultdict(int)
        word_freq = defaultdict(int)
        
        # Sample texts for efficiency
        sample_size = min(10000, len(texts))
        sample_texts = texts[:sample_size] if len(texts) > sample_size else texts
        
        for text in sample_texts:
            # Character frequencies
            for char in text:
                if char.isprintable():
                    char_freq[char] += 1
            
            # Word frequencies  
            words = text.lower().split()
            for word in words:
                if len(word) > 1:  # Skip single chars
                    word_freq[word] += 1
        
        # Add high-frequency characters first
        current_id = len(self.special_tokens)
        sorted_chars = sorted(char_freq.items(), key=lambda x: x[1], reverse=True)
        
        for char, freq in sorted_chars[:vocab_size // 4]:  # Reserve 1/4 for chars
            if current_id >= vocab_size:
                break
            if char not in self.vocab:
                self.vocab[char] = current_id
                self.id_to_token[current_id] = char
                current_id += 1
        
        # Add common subwords using simple BPE
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        # Add frequent short words
        for word, freq in sorted_words:
            if current_id >= vocab_size:
                break
            if len(word) <= 8 and freq >= 5 and word not in self.vocab:
                self.vocab[word] = current_id
                self.id_to_token[current_id] = word
                current_id += 1
        
        # Simple bigram merging for subwords
        bigram_freq = defaultdict(int)
        for word, freq in sorted_words[:1000]:  # Top words only
            chars = list(word)
            for i in range(len(chars) - 1):
                bigram = chars[i] + chars[i + 1]
                if len(bigram) >= 2:
                    bigram_freq[bigram] += freq
        
        sorted_bigrams = sorted(bigram_freq.items(), key=lambda x: x[1], reverse=True)
        for bigram, freq in sorted_bigrams:
            if current_id >= vocab_size:
                break
            if freq >= 10 and bigram not in self.vocab:
                self.vocab[bigram] = current_id
                self.id_to_token[current_id] = bigram
                current_id += 1
        
        self.trained = True
        logger.info(f"Tokenizer trained with {len(self.vocab)} tokens")
    
    def encode(self, text: str) -> List[int]:
        """Enhanced encoding with subword fallback."""
        if not self.trained:
            raise ValueError("Tokenizer not trained")
        
        tokens = []
        words = text.split()
        
        for i, word in enumerate(words):
            word_lower = word.lower()
            
            # Try exact word match first
            if word_lower in self.vocab:
                tokens.append(self.vocab[word_lower])
            else:
                # Subword tokenization - try longest matches first
                remaining = word_lower
                while remaining:
                    matched = False
                    
                    # Try progressively shorter substrings
                    for length in range(min(8, len(remaining)), 0, -1):
                        substr = remaining[:length]
                        if substr in self.vocab:
                            tokens.append(self.vocab[substr])
                            remaining = remaining[length:]
                            matched = True
                            break
                    
                    if not matched:
                        # Fall back to character level
                        char = remaining[0]
                        if char in self.vocab:
                            tokens.append(self.vocab[char])
                        else:
                            tokens.append(self.vocab[self.data_config.unk_token])
                        remaining = remaining[1:]
            
            # Add space token between words (except last word)
            if i < len(words) - 1 and " " in self.vocab:
                tokens.append(self.vocab[" "])
        
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """Enhanced decoding with proper spacing."""
        if not self.trained:
            raise ValueError("Tokenizer not trained")
        
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                # Skip special tokens in output
                if token not in [self.data_config.pad_token, self.data_config.bos_token, 
                               self.data_config.eos_token]:
                    tokens.append(token)
        
        # Reconstruct text with proper spacing
        result = ""
        for i, token in enumerate(tokens):
            if token == " ":
                result += " "
            elif len(token) == 1 and not token.isalnum():
                result += token
            else:
                if result and not result.endswith(" ") and token not in ".,!?;:":
                    result += " "
                result += token
        
        return result.strip()
    
    def vocab_size(self) -> int:
        return len(self.vocab)

# ============================================================================
# ENHANCED MODEL ARCHITECTURE
# ============================================================================

def rotate_half(x):
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    """Apply rotary position embedding."""
    cos = cos[position_ids].unsqueeze(1)  # [seq_len, 1, head_dim]
    sin = sin[position_ids].unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class RMSNorm(nn.Module):
    """RMSNorm implementation."""
    
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)

class SwiGLUMLP(nn.Module):
    """SwiGLU MLP implementation."""
    
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding."""
    
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Build here to make `torch.jit.trace` work
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )

class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention (GQA) implementation."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        self.dropout = nn.Dropout(config.dropout)
        
        if config.use_rotary_embeddings:
            self.rotary_emb = RotaryEmbedding(self.head_dim, max_position_embeddings=config.seq_length)
        
    def _repeat_kv(self, hidden_states, n_rep):
        """Repeat k/v heads if n_kv_heads < n_heads."""
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
    
    def forward(self, hidden_states, attention_mask=None, position_ids=None):
        bsz, q_len, _ = hidden_states.size()
        
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        if self.config.use_rotary_embeddings:
            cos, sin = self.rotary_emb(value_states, seq_len=q_len)
            position_ids = torch.arange(q_len, device=hidden_states.device).unsqueeze(0) if position_ids is None else position_ids
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        
        # Repeat k/v heads if n_kv_heads < n_heads
        key_states = self._repeat_kv(key_states, self.num_key_value_groups)
        value_states = self._repeat_kv(value_states, self.num_key_value_groups)
        
        # Attention computation
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        
        return self.o_proj(attn_output)

class TransformerBlock(nn.Module):
    """Enhanced transformer block with modern improvements."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Attention
        if config.use_grouped_query_attention:
            self.self_attn = GroupedQueryAttention(config)
        else:
            self.self_attn = GroupedQueryAttention(config)  # Still use GQA but with same num heads
        
        # MLP
        intermediate_size = int(config.hidden_size * 2.6666) if config.use_swiglu else config.hidden_size * 4
        if config.use_swiglu:
            self.mlp = SwiGLUMLP(config.hidden_size, intermediate_size)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(config.hidden_size, intermediate_size),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(intermediate_size, config.hidden_size)
            )
        
        # Layer norms
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, hidden_states, attention_mask=None, position_ids=None):
        # Self attention with pre-norm
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask, position_ids)
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        
        # MLP with pre-norm
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states

class EnhancedTransformer(nn.Module):
    """State-of-the-art transformer implementation with modern optimizations."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        
        # Embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        
        # Transformer layers with optional gradient checkpointing
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])
        
        # Output norm
        self.norm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Language modeling head
        if config.tie_word_embeddings:
            self.lm_head = None  # Will use embed_tokens.weight
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Enhanced weight initialization."""
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, RMSNorm):
            module.weight.data.fill_(1.0)
    
    def get_input_embeddings(self):
        return self.embed_tokens
    
    def set_input_embeddings(self, value):
        self.embed_tokens = value
    
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, device):
        """Prepare causal attention mask."""
        batch_size, seq_length = input_shape
        
        # Create causal mask
        causal_mask = torch.triu(
            torch.ones((seq_length, seq_length), dtype=torch.bool, device=device),
            diagonal=1
        )
        
        if attention_mask is not None:
            # Expand padding mask to 4D
            expanded_mask = attention_mask[:, None, None, :].expand(
                batch_size, 1, seq_length, seq_length
            ).to(dtype=torch.bool)
            
            # Combine with causal mask
            combined_mask = expanded_mask | causal_mask[None, None, :, :]
        else:
            combined_mask = causal_mask[None, None, :, :].expand(
                batch_size, 1, seq_length, seq_length
            )
        
        # Convert to attention mask (large negative values)
        attention_mask = torch.zeros_like(combined_mask, dtype=torch.float32, device=device)
        attention_mask.masked_fill_(combined_mask, torch.finfo(torch.float32).min)
        
        return attention_mask
    
    def forward(self, input_ids, attention_mask=None, position_ids=None, use_cache=False):
        batch_size, seq_length = input_ids.shape
        device = input_ids.device
        
        # Create position IDs if not provided
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Prepare attention mask
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long, device=device)
        
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), device
        )
        
        # Token embeddings
        hidden_states = self.embed_tokens(input_ids)
        
        # Apply layers
        for i, layer in enumerate(self.layers):
            if self.config.gradient_checkpointing and self.training:
                hidden_states = checkpoint(layer, hidden_states, attention_mask, position_ids)
            else:
                hidden_states = layer(hidden_states, attention_mask, position_ids)
        
        # Final norm
        hidden_states = self.norm(hidden_states)
        
        # Language modeling head
        if self.lm_head is not None:
            logits = self.lm_head(hidden_states)
        else:
            # Tied embeddings
            logits = F.linear(hidden_states, self.embed_tokens.weight)
        
        return logits

# ============================================================================
# ENHANCED DATA PROCESSING
# ============================================================================

class SmartDataset(Dataset):
    """Memory-efficient dataset with intelligent caching and preprocessing."""
    
    def __init__(self, texts: List[str], tokenizer: EnhancedTokenizer, 
                 config: ModelConfig, max_sequences: Optional[int] = None):
        self.tokenizer = tokenizer
        self.seq_length = config.seq_length
        self.pad_token_id = tokenizer.vocab.get(tokenizer.data_config.pad_token, 0)
        self.bos_token_id = tokenizer.vocab.get(tokenizer.data_config.bos_token, 2)
        self.eos_token_id = tokenizer.vocab.get(tokenizer.data_config.eos_token, 3)
        
        logger.info(f"Creating smart dataset with seq_length={self.seq_length}")
        
        self.sequences = []
        processed_count = 0
        
        # Process in batches to manage memory
        batch_size = 1000
        for i in range(0, len(texts), batch_size):
            if max_sequences and processed_count >= max_sequences:
                break
                
            batch_texts = texts[i:i+batch_size]
            
            for text in batch_texts:
                if max_sequences and processed_count >= max_sequences:
                    break
                
                if not text or len(text.strip()) < 10:
                    continue
                
                try:
                    # Tokenize with length checking
                    tokens = tokenizer.encode(text.strip())
                    if len(tokens) < 5 or len(tokens) > self.seq_length - 2:
                        continue
                    
                    # Create sequence with special tokens
                    sequence = [self.bos_token_id] + tokens + [self.eos_token_id]
                    
                    # Pad or truncate
                    if len(sequence) < self.seq_length + 1:
                        sequence.extend([self.pad_token_id] * (self.seq_length + 1 - len(sequence)))
                    else:
                        sequence = sequence[:self.seq_length + 1]
                    
                    self.sequences.append(sequence)
                    processed_count += 1
                    
                except Exception:
                    continue
            
            # Memory cleanup
            if i % (batch_size * 10) == 0:
                gc.collect()
        
        if not self.sequences:
            raise ValueError("No valid sequences created!")
        
        logger.info(f"Smart dataset created with {len(self.sequences):,} sequences")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        input_ids = torch.tensor(sequence[:-1], dtype=torch.long)
        labels = torch.tensor(sequence[1:], dtype=torch.long)
        return input_ids, labels

# ============================================================================
# TRAINING ENGINE
# ============================================================================

class TrainingEngine:
    """Advanced training engine with comprehensive features."""
    
    def __init__(self, model: EnhancedTransformer, tokenizer: EnhancedTokenizer,
                 model_config: ModelConfig, training_config: TrainingConfig,
                 hardware_config: HardwareConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.training_config = training_config
        self.hardware_config = hardware_config
        
        # Setup device
        self.device = torch.device(hardware_config.device_type.value)
        self.model = self.model.to(self.device)
        
        # Memory manager
        self.memory_manager = MemoryManager(hardware_config.device_type)
        
        # Metrics tracking
        self.metrics_tracker = MetricsTracker()
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        self.training_start_time = None
        
        # Setup training components
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_criterion()
        self._setup_scaler()
        self._setup_deepspeed()
        
        # Model compilation
        if training_config.compile_model and hasattr(torch, 'compile'):
            logger.info("Compiling model with torch.compile")
            self.model = torch.compile(self.model)
    
    def _setup_optimizer(self):
        """Setup optimizer with advanced options."""
        params = self.model.parameters()
        
        if self.training_config.optimizer_type == "adamw":
            self.optimizer = optim.AdamW(
                params,
                lr=self.training_config.learning_rate,
                betas=(self.training_config.beta1, self.training_config.beta2),
                eps=self.training_config.eps,
                weight_decay=self.training_config.weight_decay,
                fused=self.training_config.use_fused_ops and self.hardware_config.device_type == DeviceType.CUDA
            )
        elif self.training_config.optimizer_type == "adam":
            self.optimizer = optim.Adam(
                params,
                lr=self.training_config.learning_rate,
                betas=(self.training_config.beta1, self.training_config.beta2),
                eps=self.training_config.eps,
                weight_decay=self.training_config.weight_decay,
                fused=self.training_config.use_fused_ops and self.hardware_config.device_type == DeviceType.CUDA
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.training_config.optimizer_type}")
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        # Calculate total steps
        # This will be updated when dataloader is available
        self.total_steps = self.training_config.max_steps or 1000
        self.warmup_steps = int(self.total_steps * self.training_config.warmup_ratio)
        
        if self.training_config.scheduler_type == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.total_steps - self.warmup_steps,
                eta_min=self.training_config.learning_rate * self.training_config.min_lr_ratio
            )
        elif self.training_config.scheduler_type == "linear":
            from torch.optim.lr_scheduler import LinearLR
            self.scheduler = LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=self.training_config.min_lr_ratio,
                total_iters=self.total_steps - self.warmup_steps
            )
        else:
            # Simple scheduler
            self.scheduler = None
        
        # Warmup scheduler
        if self.warmup_steps > 0:
            from torch.optim.lr_scheduler import LinearLR
            self.warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=self.warmup_steps
            )
        else:
            self.warmup_scheduler = None
    
    def _setup_criterion(self):
        """Setup loss function."""
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.tokenizer.vocab.get(self.tokenizer.data_config.pad_token, 0),
            label_smoothing=self.training_config.label_smoothing
        )
    
    def _setup_scaler(self):
        """Setup gradient scaler for mixed precision."""
        if (self.training_config.use_mixed_precision and 
            self.training_config.precision_type != PrecisionType.FP32 and
            AMP_AVAILABLE):
            
            if self.training_config.precision_type == PrecisionType.BF16:
                self.scaler = GradScaler(enabled=False)  # BF16 doesn't need scaling
            else:  # FP16
                self.scaler = GradScaler(
                    init_scale=2**16,
                    growth_factor=2.0,
                    backoff_factor=0.5,
                    growth_interval=2000
                )
        else:
            self.scaler = None
    
    def _setup_deepspeed(self):
        """Setup DeepSpeed if available and requested."""
        self.deepspeed_engine = None
        
        if not (self.training_config.use_deepspeed and DEEPSPEED_AVAILABLE):
            return
        
        # Create DeepSpeed config
        ds_config = self._create_deepspeed_config()
        
        try:
            self.deepspeed_engine, optimizer, _, scheduler = deepspeed.initialize(
                model=self.model,
                config=ds_config,
                model_parameters=self.model.parameters()
            )
            
            # Override optimizers
            self.optimizer = optimizer
            if scheduler:
                self.scheduler = scheduler
            
            logger.info("DeepSpeed initialized successfully")
            
        except Exception as e:
            logger.warning(f"DeepSpeed initialization failed: {e}")
            self.training_config.use_deepspeed = False
    
    def _create_deepspeed_config(self) -> dict:
        """Create DeepSpeed configuration."""
        config = {
            "train_batch_size": self.training_config.batch_size * self.training_config.gradient_accumulation_steps,
            "train_micro_batch_size_per_gpu": self.training_config.batch_size,
            "gradient_accumulation_steps": self.training_config.gradient_accumulation_steps,
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": self.training_config.learning_rate,
                    "betas": [self.training_config.beta1, self.training_config.beta2],
                    "eps": self.training_config.eps,
                    "weight_decay": self.training_config.weight_decay
                }
            },
            "scheduler": {
                "type": "WarmupDecayLR",
                "params": {
                    "warmup_min_lr": 0,
                    "warmup_max_lr": self.training_config.learning_rate,
                    "warmup_num_steps": self.warmup_steps,
                    "total_num_steps": self.total_steps
                }
            },
            "gradient_clipping": self.training_config.max_grad_norm,
            "steps_per_print": self.training_config.log_every,
            "wall_clock_breakdown": False
        }
        
        # Precision configuration
        if self.training_config.precision_type == PrecisionType.BF16:
            config["bf16"] = {"enabled": True}
            config["fp16"] = {"enabled": False}
        elif self.training_config.precision_type == PrecisionType.FP16:
            config["fp16"] = {
                "enabled": True,
                "loss_scale": 0,  # Dynamic
                "loss_scale_window": 1000,
                "initial_scale_power": 16,
                "hysteresis": 2,
                "min_loss_scale": 1
            }
            config["bf16"] = {"enabled": False}
        else:
            config["fp16"] = {"enabled": False}
            config["bf16"] = {"enabled": False}
        
        # ZeRO configuration
        if self.training_config.zero_stage >= 1:
            config["zero_optimization"] = {
                "stage": self.training_config.zero_stage,
                "offload_optimizer": {
                    "device": "cpu" if self.training_config.offload_optimizer else "none"
                },
                "offload_param": {
                    "device": "cpu" if self.training_config.offload_params else "none"
                },
                "overlap_comm": True,
                "contiguous_gradients": True,
                "sub_group_size": 1e9,
                "reduce_bucket_size": "auto"
            }
        
        return config
    
    def update_scheduler_steps(self, total_steps: int):
        """Update scheduler with actual total steps."""
        self.total_steps = total_steps
        self.warmup_steps = int(total_steps * self.training_config.warmup_ratio)
        
        # Recreate schedulers with correct steps
        self._setup_scheduler()
        
        if self.deepspeed_engine:
            # Update DeepSpeed config
            ds_config = self._create_deepspeed_config()
            # Note: DeepSpeed doesn't support runtime config updates easily
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train one epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        num_batches = 0
        
        epoch_start = time.time()
        
        for batch_idx, (input_ids, labels) in enumerate(dataloader):
            try:
                # Memory cleanup check
                if self.memory_manager.should_cleanup():
                    with self.memory_manager.cleanup_context():
                        pass
                
                # Move to device
                input_ids = input_ids.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                # Skip empty batches
                if input_ids.numel() == 0 or labels.numel() == 0:
                    continue
                
                # Forward pass with autocast
                autocast_context = nullcontext()
                if self.training_config.use_mixed_precision and self.scaler is not None:
                    if self.training_config.precision_type == PrecisionType.BF16:
                        autocast_context = autocast(dtype=torch.bfloat16)
                    else:
                        autocast_context = autocast(dtype=torch.float16)
                
                with autocast_context:
                    if self.deepspeed_engine:
                        logits = self.deepspeed_engine(input_ids)
                    else:
                        logits = self.model(input_ids)
                    
                    loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                    loss = loss / self.training_config.gradient_accumulation_steps
                
                # Check for invalid loss
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"Invalid loss at batch {batch_idx}: {loss.item()}")
                    continue
                
                # Backward pass
                if self.deepspeed_engine:
                    self.deepspeed_engine.backward(loss)
                else:
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()
                
                # Calculate metrics
                with torch.no_grad():
                    predictions = torch.argmax(logits, dim=-1)
                    pad_mask = (labels != self.criterion.ignore_index)
                    correct = ((predictions == labels) & pad_mask).sum().item()
                    valid_tokens = pad_mask.sum().item()
                    
                    total_correct += correct
                    total_tokens += valid_tokens
                    total_loss += loss.item() * self.training_config.gradient_accumulation_steps
                    num_batches += 1
                
                # Optimizer step
                if (batch_idx + 1) % self.training_config.gradient_accumulation_steps == 0:
                    if self.deepspeed_engine:
                        self.deepspeed_engine.step()
                    else:
                        # Gradient clipping and optimization
                        if self.scaler is not None:
                            if self.training_config.max_grad_norm > 0:
                                self.scaler.unscale_(self.optimizer)
                                torch.nn.utils.clip_grad_norm_(
                                    self.model.parameters(), 
                                    self.training_config.max_grad_norm
                                )
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            if self.training_config.max_grad_norm > 0:
                                torch.nn.utils.clip_grad_norm_(
                                    self.model.parameters(), 
                                    self.training_config.max_grad_norm
                                )
                            self.optimizer.step()
                        
                        # Learning rate scheduling
                        if self.global_step < self.warmup_steps and self.warmup_scheduler:
                            self.warmup_scheduler.step()
                        elif self.scheduler:
                            self.scheduler.step()
                        
                        self.optimizer.zero_grad()
                    
                    self.global_step += 1
                
                # Logging
                if batch_idx % self.training_config.log_every == 0:
                    current_loss = total_loss / max(num_batches, 1)
                    current_acc = total_correct / max(total_tokens, 1)
                    current_lr = self.optimizer.param_groups[0]['lr']
                    
                    logger.info(
                        f"Epoch {self.epoch} | Step {self.global_step} | "
                        f"Batch {batch_idx}/{len(dataloader)} | "
                        f"Loss: {current_loss:.4f} | Acc: {current_acc:.3f} | "
                        f"LR: {current_lr:.2e} | {self.memory_manager.get_memory_usage_str()}"
                    )
                    
                    # Update metrics
                    self.metrics_tracker.update(
                        train_loss=current_loss,
                        train_accuracy=current_acc,
                        learning_rate=current_lr,
                        epoch=self.epoch,
                        step=self.global_step
                    )
                
                # Clean up tensors
                del input_ids, labels, logits, loss, predictions, pad_mask
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning(f"OOM at batch {batch_idx}, cleaning up and continuing...")
                    
                    # Aggressive cleanup
                    for var_name in ['input_ids', 'labels', 'logits', 'loss', 'predictions']:
                        if var_name in locals():
                            del locals()[var_name]
                    
                    if not self.deepspeed_engine:
                        self.optimizer.zero_grad()
                    
                    with self.memory_manager.cleanup_context(aggressive=True):
                        pass
                    continue
                else:
                    raise e
            except Exception as e:
                logger.warning(f"Error at batch {batch_idx}: {e}")
                continue
        
        # Calculate epoch metrics
        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / max(num_batches, 1)
        avg_accuracy = total_correct / max(total_tokens, 1)
        
        return {
            'loss': avg_loss,
            'accuracy': avg_accuracy,
            'epoch_time': epoch_time,
            'num_batches': num_batches,
            'total_tokens': total_tokens
        }
    
    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader, max_batches: Optional[int] = None) -> Dict[str, float]:
        """Evaluate the model."""
        self.model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        num_batches = 0
        
        eval_start = time.time()
        
        for batch_idx, (input_ids, labels) in enumerate(dataloader):
            if max_batches and batch_idx >= max_batches:
                break
            
            try:
                input_ids = input_ids.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                # Forward pass
                autocast_context = nullcontext()
                if self.training_config.use_mixed_precision and self.scaler is not None:
                    if self.training_config.precision_type == PrecisionType.BF16:
                        autocast_context = autocast(dtype=torch.bfloat16)
                    else:
                        autocast_context = autocast(dtype=torch.float16)
                
                with autocast_context:
                    if self.deepspeed_engine:
                        logits = self.deepspeed_engine(input_ids)
                    else:
                        logits = self.model(input_ids)
                    
                    loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                
                # Calculate metrics
                predictions = torch.argmax(logits, dim=-1)
                pad_mask = (labels != self.criterion.ignore_index)
                correct = ((predictions == labels) & pad_mask).sum().item()
                valid_tokens = pad_mask.sum().item()
                
                total_loss += loss.item()
                total_correct += correct
                total_tokens += valid_tokens
                num_batches += 1
                
                # Clean up
                del input_ids, labels, logits, loss, predictions, pad_mask
                
            except Exception as e:
                logger.warning(f"Error in evaluation batch {batch_idx}: {e}")
                continue
        
        eval_time = time.time() - eval_start
        
        if num_batches == 0:
            return {
                'loss': float('inf'),
                'accuracy': 0.0,
                'perplexity': float('inf'),
                'eval_time': eval_time
            }
        
        avg_loss = total_loss / num_batches
        avg_accuracy = total_correct / max(total_tokens, 1)
        perplexity = math.exp(min(avg_loss, 10))  # Cap to prevent overflow
        
        self.model.train()  # Return to training mode
        
        return {
            'loss': avg_loss,
            'accuracy': avg_accuracy,
            'perplexity': perplexity,
            'eval_time': eval_time,
            'num_batches': num_batches
        }
    
    @torch.no_grad()
    def generate_sample(self, prompt: str = "<user> Hello", max_length: int = 50, 
                       temperature: float = 1.0, top_k: int = 50, top_p: float = 0.9) -> str:
        """Generate sample text with advanced sampling."""
        self.model.eval()
        
        try:
            # Encode prompt
            input_ids = torch.tensor(
                self.tokenizer.encode(prompt), dtype=torch.long, device=self.device
            ).unsqueeze(0)
            
            generated = input_ids.clone()
            
            for _ in range(max_length):
                if generated.size(1) >= self.model_config.seq_length:
                    break
                
                # Forward pass
                autocast_context = nullcontext()
                if self.training_config.use_mixed_precision and self.scaler is not None:
                    if self.training_config.precision_type == PrecisionType.BF16:
                        autocast_context = autocast(dtype=torch.bfloat16)
                    else:
                        autocast_context = autocast(dtype=torch.float16)
                
                with autocast_context:
                    if self.deepspeed_engine:
                        logits = self.deepspeed_engine(generated)
                    else:
                        logits = self.model(generated)
                
                # Get next token logits
                next_token_logits = logits[0, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
                
                # Stop on EOS token
                if next_token.item() == self.tokenizer.vocab.get(self.tokenizer.data_config.eos_token, -1):
                    break
            
            # Decode generated sequence (skip original prompt)
            response_ids = generated[0][input_ids.size(1):].tolist()
            response = self.tokenizer.decode(response_ids)
            
            return response.strip()
        
        except Exception as e:
            logger.warning(f"Generation failed: {e}")
            return "Generation failed"
        finally:
            self.model.train()

# ============================================================================
# MODEL MANAGEMENT
# ============================================================================

class EnhancedModelManager:
    """Advanced model management with versioning and metadata."""
    
    def __init__(self, save_dir: str = "models"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Track saved models
        self.model_registry = self.save_dir / "model_registry.json"
        self.saved_models = self._load_registry()
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load model registry."""
        if self.model_registry.exists():
            try:
                with open(self.model_registry, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load model registry: {e}")
        return {}
    
    def _save_registry(self):
        """Save model registry."""
        try:
            with open(self.model_registry, 'w') as f:
                json.dump(self.saved_models, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to save model registry: {e}")
    
    def save_checkpoint(self, training_engine: TrainingEngine, metadata: ModelMetadata,
                       is_best: bool = False, cleanup_old: bool = True) -> Optional[str]:
        """Save model checkpoint with full state."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_id = f"checkpoint_{timestamp}"
        checkpoint_path = self.save_dir / checkpoint_id
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Update metadata
            metadata.last_modified = datetime.now().isoformat()
            metadata.epochs_trained = training_engine.epoch
            metadata.steps_trained = training_engine.global_step
            
            with training_engine.memory_manager.cleanup_context():
                # Save model state
                if training_engine.deepspeed_engine:
                    # DeepSpeed checkpoint
                    training_engine.deepspeed_engine.save_checkpoint(str(checkpoint_path))
                    logger.info("DeepSpeed checkpoint saved")
                else:
                    # Regular PyTorch checkpoint
                    model = training_engine.model
                    if hasattr(model, 'module'):  # Unwrap DDP
                        model = model.module
                    
                    # Create comprehensive checkpoint
                    checkpoint = {
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': training_engine.optimizer.state_dict(),
                        'epoch': training_engine.epoch,
                        'global_step': training_engine.global_step,
                        'best_loss': training_engine.best_loss,
                        'model_config': asdict(training_engine.model_config),
                        'training_config': asdict(training_engine.training_config),
                        'rng_state': torch.get_rng_state(),
                    }
                    
                    if training_engine.scaler is not None:
                        checkpoint['scaler_state_dict'] = training_engine.scaler.state_dict()
                    
                    if torch.cuda.is_available():
                        checkpoint['cuda_rng_state'] = torch.cuda.get_rng_state()
                    
                    torch.save(checkpoint, checkpoint_path / "checkpoint.pth")
                
                # Save tokenizer
                tokenizer_data = {
                    'vocab': training_engine.tokenizer.vocab,
                    'id_to_token': training_engine.tokenizer.id_to_token,
                    'data_config': asdict(training_engine.tokenizer.data_config)
                }
                with open(checkpoint_path / "tokenizer.json", 'w') as f:
                    json.dump(tokenizer_data, f, indent=2)
                
                # Save metadata
                metadata_dict = asdict(metadata)
                with open(checkpoint_path / "metadata.json", 'w') as f:
                    json.dump(metadata_dict, f, indent=2, default=str)
                
                # Save training metrics
                if hasattr(training_engine, 'metrics_tracker'):
                    metrics_data = {
                        'global_metrics': dict(training_engine.metrics_tracker.global_metrics),
                        'best_metrics': training_engine.metrics_tracker.best_metrics,
                        'window_size': training_engine.metrics_tracker.window_size
                    }
                    with open(checkpoint_path / "metrics.json", 'w') as f:
                        json.dump(metrics_data, f, indent=2, default=str)
            
            # Update registry
            self.saved_models[checkpoint_id] = {
                'path': str(checkpoint_path),
                'timestamp': timestamp,
                'epoch': training_engine.epoch,
                'global_step': training_engine.global_step,
                'best_loss': training_engine.best_loss,
                'is_best': is_best,
                'model_size_mb': metadata.model_size_mb,
                'total_parameters': metadata.total_parameters
            }
            self._save_registry()
            
            # Cleanup old checkpoints
            if cleanup_old:
                self._cleanup_old_checkpoints()
            
            logger.info(f"Checkpoint saved: {checkpoint_id}")
            return checkpoint_id
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            # Cleanup failed checkpoint
            if checkpoint_path.exists():
                import shutil
                shutil.rmtree(checkpoint_path, ignore_errors=True)
            return None
    
    def _cleanup_old_checkpoints(self, keep_last: int = 5, keep_best: int = 3):
        """Clean up old checkpoints to save disk space."""
        if len(self.saved_models) <= keep_last:
            return
        
        # Sort by timestamp
        sorted_models = sorted(
            self.saved_models.items(),
            key=lambda x: x[1]['timestamp'],
            reverse=True
        )
        
        # Keep best models
        best_models = sorted(
            [(k, v) for k, v in self.saved_models.items() if v.get('is_best', False)],
            key=lambda x: x[1]['best_loss']
        )[:keep_best]
        
        best_model_ids = {k for k, v in best_models}
        
        # Determine models to keep
        models_to_keep = set()
        
        # Keep recent models
        for i, (model_id, model_info) in enumerate(sorted_models):
            if i < keep_last:
                models_to_keep.add(model_id)
        
        # Keep best models
        models_to_keep.update(best_model_ids)
        
        # Delete old models
        deleted_count = 0
        for model_id, model_info in list(self.saved_models.items()):
            if model_id not in models_to_keep:
                model_path = Path(model_info['path'])
                if model_path.exists():
                    import shutil
                    shutil.rmtree(model_path, ignore_errors=True)
                    deleted_count += 1
                del self.saved_models[model_id]
        
        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} old checkpoints")
            self._save_registry()

# ============================================================================
# DATA LOADING AND PROCESSING
# ============================================================================

def load_and_process_data(data_config: DataConfig) -> Tuple[List[str], List[str]]:
    """Enhanced data loading with comprehensive preprocessing."""
    data_path = Path(data_config.dataset_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    logger.info(f"Loading data from: {data_path}")
    
    texts = []
    processed_count = 0
    skipped_count = 0
    
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            for line_no, line in enumerate(f, 1):
                if data_config.max_samples and processed_count >= data_config.max_samples:
                    break
                
                try:
                    line = line.strip()
                    if not line:
                        continue
                    
                    record = json.loads(line)
                    
                    # Skip deleted or non-English content
                    if record.get("deleted", False) or record.get("lang") != "en":
                        skipped_count += 1
                        continue
                    
                    text = record.get("text", "").strip()
                    if not text:
                        skipped_count += 1
                        continue
                    
                    # Length filtering
                    if len(text) < data_config.min_length or len(text) > data_config.max_length:
                        skipped_count += 1
                        continue
                    
                    # Format based on role
                    role = record.get("role", "").lower()
                    if role == "prompter":
                        formatted_text = f"{data_config.user_token} {text}"
                    elif role == "assistant":
                        formatted_text = f"{data_config.assistant_token} {text}"
                    else:
                        formatted_text = text
                    
                    texts.append(formatted_text)
                    processed_count += 1
                    
                    # Progress logging
                    if line_no % 10000 == 0:
                        logger.info(f"Processed {line_no:,} lines, kept {processed_count:,} texts")
                    
                except (json.JSONDecodeError, KeyError) as e:
                    skipped_count += 1
                    if line_no % 10000 == 0:
                        logger.warning(f"Parse error at line {line_no}: {e}")
                    continue
        
        logger.info(f"Data loading complete: {processed_count:,} texts loaded, {skipped_count:,} skipped")
        
        # Split into train/validation
        if data_config.validation_split > 0:
            split_idx = int(len(texts) * (1 - data_config.validation_split))
            if data_config.shuffle:
                import random
                random.shuffle(texts)
            train_texts = texts[:split_idx]
            val_texts = texts[split_idx:]
            logger.info(f"Split: {len(train_texts):,} train, {len(val_texts):,} validation")
            return train_texts, val_texts
        else:
            return texts, []
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def create_optimized_dataloader(dataset: SmartDataset, batch_size: int,
                              training_config: TrainingConfig,
                              shuffle: bool = True) -> DataLoader:
    """Create optimized data loader with advanced features."""
    
    # Determine number of workers based on system
    if training_config.dataloader_num_workers == 0:
        num_workers = 0  # Single-threaded
    elif training_config.dataloader_num_workers < 0:
        num_workers = min(4, os.cpu_count() or 1)  # Auto-detect
    else:
        num_workers = training_config.dataloader_num_workers
    
    # Pin memory only if using GPU and have enough system RAM
    pin_memory = (training_config.pin_memory and 
                 torch.cuda.is_available() and 
                 psutil.virtual_memory().available > 8 * 1024**3)  # 8GB available
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=training_config.persistent_workers and num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else 2
    )

# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def main():
    """Enhanced main training function."""
    logger.info("🚀 Starting Enhanced Ultra-Efficient Training System")
    logger.info("=" * 80)
    
    try:
        # Hardware detection and optimization
        logger.info("🔍 Detecting hardware configuration...")
        hardware_config = HardwareDetector.detect_hardware()
        HardwareDetector.optimize_cuda_settings(hardware_config)
        
        logger.info(f"Hardware: {hardware_config.device_name} ({hardware_config.total_memory_gb:.1f}GB)")
        logger.info(f"Device type: {hardware_config.device_type.value}")
        logger.info(f"Optimal precision: {hardware_config.get_optimal_precision().value}")
        
        # Determine optimization level based on available memory
        if hardware_config.total_memory_gb < 6:
            optimization_level = OptimizationLevel.EXTREME
        elif hardware_config.total_memory_gb < 12:
            optimization_level = OptimizationLevel.AGGRESSIVE
        elif hardware_config.total_memory_gb < 24:
            optimization_level = OptimizationLevel.BALANCED
        else:
            optimization_level = OptimizationLevel.CONSERVATIVE
        
        logger.info(f"Optimization level: {optimization_level.value}")
        
        # Create configurations
        model_config = ModelConfig.get_optimized_config(hardware_config, optimization_level)
        
        training_config = TrainingConfig(
            precision_type=hardware_config.get_optimal_precision(),
            use_mixed_precision=hardware_config.device_type == DeviceType.CUDA,
            batch_size=2 if optimization_level == OptimizationLevel.EXTREME else 4,
            gradient_accumulation_steps=32 if optimization_level == OptimizationLevel.EXTREME else 16,
            use_deepspeed=DEEPSPEED_AVAILABLE and hardware_config.device_type == DeviceType.CUDA,
            dataloader_num_workers=0 if optimization_level == OptimizationLevel.EXTREME else 2,
            max_epochs=50 if optimization_level in [OptimizationLevel.EXTREME, OptimizationLevel.AGGRESSIVE] else 25
        )
        
        data_config = DataConfig(
            max_samples=5000 if optimization_level == OptimizationLevel.EXTREME else None,
            vocab_size=model_config.vocab_size
        )
        
        # Log configuration
        logger.info("📋 Configuration:")
        logger.info(f"  Model: {model_config.hidden_size}D x {model_config.num_layers}L x {model_config.num_heads}H")
        logger.info(f"  Sequence length: {model_config.seq_length}")
        logger.info(f"  Vocab size: {model_config.vocab_size:,}")
        logger.info(f"  Precision: {training_config.precision_type.value}")
        logger.info(f"  Batch size: {training_config.batch_size} x {training_config.gradient_accumulation_steps}")
        logger.info(f"  Learning rate: {training_config.learning_rate}")
        logger.info(f"  DeepSpeed: {training_config.use_deepspeed}")
        
        # Load and process data
        logger.info("📚 Loading and processing data...")
        train_texts, val_texts = load_and_process_data(data_config)
        
        # Create and train tokenizer
        logger.info("🔤 Training tokenizer...")
        tokenizer = EnhancedTokenizer(data_config)
        
        # Use sample for tokenizer training
        sample_size = min(5000, len(train_texts))
        sample_texts = train_texts[:sample_size]
        tokenizer.train_subword_tokenizer(sample_texts, data_config.vocab_size)
        
        # Update model config with actual vocab size
        model_config.vocab_size = tokenizer.vocab_size()
        logger.info(f"Tokenizer trained: {tokenizer.vocab_size():,} tokens")
        
        # Create datasets
        logger.info("📦 Creating datasets...")
        train_dataset = SmartDataset(train_texts, tokenizer, model_config)
        train_dataloader = create_optimized_dataloader(
            train_dataset, training_config.batch_size, training_config, shuffle=True
        )
        
        val_dataloader = None
        if val_texts:
            val_dataset = SmartDataset(val_texts, tokenizer, model_config, max_sequences=500)
            val_dataloader = create_optimized_dataloader(
                val_dataset, training_config.batch_size, training_config, shuffle=False
            )
        
        logger.info(f"Training dataset: {len(train_dataset):,} sequences")
        if val_dataloader:
            logger.info(f"Validation dataset: {len(val_dataset):,} sequences")
        
        # Create model
        logger.info("🧠 Creating model...")
        model = EnhancedTransformer(model_config)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model_size_mb = total_params * 4 / (1024 ** 2)  # Assuming fp32
        
        logger.info(f"Model created: {total_params:,} parameters ({model_size_mb:.1f}MB)")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
        # Create training engine
        logger.info("⚙️ Setting up training engine...")
        training_engine = TrainingEngine(
            model, tokenizer, model_config, training_config, hardware_config
        )
        
        # Update scheduler with correct total steps
        total_steps = len(train_dataloader) * training_config.max_epochs // training_config.gradient_accumulation_steps
        training_engine.update_scheduler_steps(total_steps)
        
        logger.info(f"Total training steps: {total_steps:,}")
        logger.info(f"Warmup steps: {training_engine.warmup_steps:,}")
        
        # Create model manager
        model_manager = EnhancedModelManager("enhanced_models")
        
        # Create metadata
        metadata = ModelMetadata(
            model_name="Enhanced_Transformer",
            version="v2.0",
            model_config=model_config,
            training_config=training_config,
            data_config=data_config,
            hardware_config=hardware_config,
            model_size_mb=model_size_mb,
            total_parameters=total_params,
            trainable_parameters=trainable_params,
            tags=[
                optimization_level.value,
                training_config.precision_type.value,
                hardware_config.device_type.value,
                f"hidden_{model_config.hidden_size}",
                f"layers_{model_config.num_layers}"
            ]
        )
        
        # Training loop
        logger.info("🚀 Starting training...")
        training_start_time = time.time()
        training_engine.training_start_time = training_start_time
        
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 5
        
        for epoch in range(1, training_config.max_epochs + 1):
            training_engine.epoch = epoch
            
            logger.info(f"{'='*20} Epoch {epoch}/{training_config.max_epochs} {'='*20}")
            
            # Training phase
            train_metrics = training_engine.train_epoch(train_dataloader)
            
            # Log training metrics
            logger.info(
                f"Training - Loss: {train_metrics['loss']:.4f}, "
                f"Acc: {train_metrics['accuracy']:.3f}, "
                f"Time: {train_metrics['epoch_time']:.1f}s, "
                f"Tokens: {train_metrics['total_tokens']:,}"
            )
            
            # Validation phase
            val_metrics = {}
            if val_dataloader and epoch % 2 == 0:  # Validate every 2 epochs
                val_metrics = training_engine.evaluate(val_dataloader, max_batches=10)
                logger.info(
                    f"Validation - Loss: {val_metrics['loss']:.4f}, "
                    f"Acc: {val_metrics['accuracy']:.3f}, "
                    f"PPL: {val_metrics['perplexity']:.2f}"
                )
                
                # Early stopping check
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= max_patience:
                    logger.info(f"Early stopping triggered (patience: {max_patience})")
                    break
            
            # Sample generation
            if epoch % 5 == 0:
                sample_text = training_engine.generate_sample(
                    prompt="<user> Hello, how are you?",
                    max_length=30,
                    temperature=0.8
                )
                logger.info(f"Sample: {sample_text}")
            
            # Update tracking
            is_best = train_metrics['loss'] < training_engine.best_loss
            if is_best:
                training_engine.best_loss = train_metrics['loss']
                logger.info(f"🏆 New best training loss: {training_engine.best_loss:.4f}")
            
            # Update metadata
            metadata.performance_metrics.update({
                f"epoch_{epoch}_train_loss": train_metrics['loss'],
                f"epoch_{epoch}_train_accuracy": train_metrics['accuracy'],
            })
            
            if val_metrics:
                metadata.performance_metrics.update({
                    f"epoch_{epoch}_val_loss": val_metrics['loss'],
                    f"epoch_{epoch}_val_accuracy": val_metrics['accuracy'],
                })
            
            # Save checkpoint
            if epoch % 5 == 0 or is_best or epoch == training_config.max_epochs:
                checkpoint_id = model_manager.save_checkpoint(
                    training_engine, metadata, is_best=is_best
                )
                if checkpoint_id:
                    logger.info(f"💾 Checkpoint saved: {checkpoint_id}")
            
            # Memory status
            logger.info(f"Memory: {training_engine.memory_manager.get_memory_usage_str()}")
            
            # Performance summary
            elapsed_time = time.time() - training_start_time
            estimated_total = elapsed_time * training_config.max_epochs / epoch
            remaining_time = estimated_total - elapsed_time
            
            logger.info(
                f"Progress: {epoch}/{training_config.max_epochs} "
                f"({100*epoch/training_config.max_epochs:.1f}%) | "
                f"Elapsed: {timedelta(seconds=int(elapsed_time))} | "
                f"Remaining: {timedelta(seconds=int(remaining_time))}"
            )
        
        # Training completion
        total_training_time = time.time() - training_start_time
        
        # Final metadata update
        metadata.training_time_hours = total_training_time / 3600
        metadata.epochs_trained = training_engine.epoch
        metadata.steps_trained = training_engine.global_step
        metadata.best_loss = training_engine.best_loss
        metadata.performance_metrics['final_train_loss'] = train_metrics['loss']
        if val_metrics:
            metadata.performance_metrics['final_val_loss'] = val_metrics['loss']
        
        # Save final checkpoint
        final_checkpoint_id = model_manager.save_checkpoint(
            training_engine, metadata, is_best=True
        )
        
        # Final evaluation and sample generation
        logger.info("🔍 Final evaluation...")
        if val_dataloader:
            final_val_metrics = training_engine.evaluate(val_dataloader)
            logger.info(
                f"Final Validation - Loss: {final_val_metrics['loss']:.4f}, "
                f"Acc: {final_val_metrics['accuracy']:.3f}, "
                f"PPL: {final_val_metrics['perplexity']:.2f}"
            )
        
        # Generate multiple samples
        logger.info("🎭 Generating final samples...")
        sample_prompts = [
            "<user> Hello, how are you?",
            "<user> What is machine learning?",
            "<user> Tell me a joke",
            "<assistant> I can help you with"
        ]
        
        for prompt in sample_prompts:
            sample = training_engine.generate_sample(
                prompt=prompt,
                max_length=40,
                temperature=0.7,
                top_k=50,
                top_p=0.9
            )
            logger.info(f"Prompt: {prompt}")
            logger.info(f"Sample: {sample}")
            logger.info("-" * 50)
        
        # Training summary
        logger.info("=" * 80)
        logger.info("✅ Enhanced Training Complete!")
        logger.info(f"🏆 Best training loss: {training_engine.best_loss:.4f}")
        logger.info(f"⏱️  Total training time: {timedelta(seconds=int(total_training_time))}")
        logger.info(f"📊 Total steps: {training_engine.global_step:,}")
        logger.info(f"🔧 Hardware: {hardware_config.device_name}")
        logger.info(f"⚙️  Precision: {training_config.precision_type.value}")
        logger.info(f"🧠 Model size: {total_params:,} parameters ({model_size_mb:.1f}MB)")
        logger.info(f"💾 Final checkpoint: {final_checkpoint_id}")
        logger.info(f"📈 Optimization level: {optimization_level.value}")
        
        # Performance statistics
        if hasattr(training_engine, 'metrics_tracker'):
            avg_loss = training_engine.metrics_tracker.get_average('train_loss')
            loss_trend = training_engine.metrics_tracker.get_trend('train_loss')
            logger.info(f"📉 Average loss: {avg_loss:.4f} (trend: {loss_trend})")
        
        # Memory statistics
        final_memory_stats = training_engine.memory_manager.get_memory_stats()
        if 'gpu_max_allocated_gb' in final_memory_stats:
            logger.info(f"🎯 Peak GPU memory: {final_memory_stats['gpu_max_allocated_gb']:.1f}GB")
        
        logger.info("=" * 80)
        return 0
        
    except KeyboardInterrupt:
        logger.info("⚠️  Training interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return 1
    finally:
        # Final cleanup
        if 'training_engine' in locals():
            with training_engine.memory_manager.cleanup_context(aggressive=True):
                pass
        logger.info("🧹 Final cleanup completed")

# ============================================================================
# UTILITY FUNCTIONS AND CLI
# ============================================================================

def setup_environment():
    """Setup the training environment."""
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    
    # Set environment variables for optimal performance
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    
    # Disable TensorFloat-32 for reproducibility (optional)
    # torch.backends.cuda.matmul.allow_tf32 = False
    # torch.backends.cudnn.allow_tf32 = False

def validate_requirements():
    """Validate that all requirements are met."""
    requirements = [
        ("oasst1_data/oasst1_train.jsonl", "Training data file"),
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
        ("psutil", "Process utilities")
    ]
    
    all_valid = True
    
    for req, description in requirements:
        if req.endswith('.jsonl'):
            # File requirement
            if not Path(req).exists():
                logger.error(f"❌ Missing {description}: {req}")
                all_valid = False
            else:
                logger.info(f"✅ Found {description}")
        else:
            # Module requirement  
            try:
                __import__(req)
                logger.info(f"✅ {description} available")
            except ImportError:
                logger.error(f"❌ Missing {description}")
                all_valid = False
    
    # Optional requirements
    optional_reqs = [
        ("deepspeed", "DeepSpeed", DEEPSPEED_AVAILABLE),
        ("wandb", "Weights & Biases", WANDB_AVAILABLE),
        ("transformers", "Transformers", TRANSFORMERS_AVAILABLE)
    ]
    
    for req, description, available in optional_reqs:
        if available:
            logger.info(f"✅ {description} available (optional)")
        else:
            logger.warning(f"⚠️  {description} not available (optional)")
    
    return all_valid

def print_system_info():
    """Print comprehensive system information."""
    logger.info("🖥️  System Information:")
    logger.info(f"  Python: {sys.version.split()[0]}")
    logger.info(f"  PyTorch: {torch.__version__}")
    logger.info(f"  Platform: {sys.platform}")
    
    # CPU info
    logger.info(f"  CPU cores: {os.cpu_count()}")
    memory_gb = psutil.virtual_memory().total / (1024**3)
    logger.info(f"  System RAM: {memory_gb:.1f}GB")
    
    # GPU info
    if torch.cuda.is_available():
        logger.info(f"  CUDA devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"    GPU {i}: {props.name} ({props.total_memory/1024**3:.1f}GB)")
            logger.info(f"    Compute capability: {props.major}.{props.minor}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        logger.info("  Apple Metal Performance Shaders available")
    else:
        logger.info("  CPU-only mode")
    
    # Optimization features
    logger.info("🚀 Optimization Features:")
    logger.info(f"  Mixed Precision (AMP): {AMP_AVAILABLE}")
    logger.info(f"  Gradient Checkpointing: {CHECKPOINT_AVAILABLE}")
    logger.info(f"  DeepSpeed: {DEEPSPEED_AVAILABLE}")
    logger.info(f"  Torch Compile: {hasattr(torch, 'compile')}")
    
    if hasattr(torch.backends.cudnn, 'benchmark'):
        logger.info(f"  CuDNN Benchmark: {torch.backends.cudnn.benchmark}")

def create_sample_config():
    """Create sample configuration files."""
    config_dir = Path("configs")
    config_dir.mkdir(exist_ok=True)
    
    # Sample model config
    sample_model_config = {
        "vocab_size": 32000,
        "hidden_size": 1024,
        "num_layers": 12,
        "num_heads": 8,
        "seq_length": 512,
        "dropout": 0.1,
        "use_rotary_embeddings": True,
        "use_grouped_query_attention": False,
        "use_swiglu": True,
        "gradient_checkpointing": True
    }
    
    # Sample training config
    sample_training_config = {
        "learning_rate": 1e-4,
        "batch_size": 4,
        "gradient_accumulation_steps": 16,
        "max_epochs": 10,
        "warmup_ratio": 0.1,
        "precision_type": "bf16",
        "use_deepspeed": True,
        "zero_stage": 2,
        "max_grad_norm": 1.0
    }
    
    # Sample data config
    sample_data_config = {
        "dataset_path": "oasst1_data/oasst1_train.jsonl",
        "max_samples": 10000,
        "validation_split": 0.1,
        "min_length": 10,
        "max_length": 1024,
        "vocab_size": 32000
    }
    
    # Write configs
    configs = [
        ("model_config.json", sample_model_config),
        ("training_config.json", sample_training_config),
        ("data_config.json", sample_data_config)
    ]
    
    for filename, config in configs:
        config_path = config_dir / filename
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"📄 Created sample config: {config_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Ultra-Efficient Training System")
    parser.add_argument("--create-configs", action="store_true", help="Create sample configuration files")
    parser.add_argument("--validate", action="store_true", help="Validate requirements and environment")
    parser.add_argument("--system-info", action="store_true", help="Print system information")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger("enhanced_training").setLevel(getattr(logging, args.log_level))
    
    if args.create_configs:
        create_sample_config()
        sys.exit(0)
    
    if args.system_info:
        print_system_info()
        if not args.validate:
            sys.exit(0)
    
    if args.validate:
        if validate_requirements():
            logger.info("✅ All requirements validated")
            sys.exit(0)
        else:
            logger.error("❌ Requirements validation failed")
            sys.exit(1)
    
    # Setup environment and run training
    setup_environment()
    exit_code = main()
    sys.exit(exit_code)