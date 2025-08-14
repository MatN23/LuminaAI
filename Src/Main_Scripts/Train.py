# Production-Ready Conversational Transformer Training System
# Enhanced version with monitoring, fault tolerance, and deployment features

import json
import logging
import os
import time
import math
import signal
import psutil
import threading
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Iterator, Callable, Any
import warnings
from datetime import datetime
import traceback
import shutil
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
import numpy as np
import tiktoken

# High-performance imports
try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

try:
    import tensorboard
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

from torch.cuda.amp import autocast, GradScaler

# Configure for performance and memory efficiency
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# Memory optimization settings
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Set memory fraction to leave some GPU memory free
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(0.85)

# =============================================================================
# PRODUCTION LOGGING AND MONITORING
# =============================================================================

class ProductionLogger:
    """Enhanced logging with structured format and monitoring."""
    
    def __init__(self, log_level: str = "INFO", experiment_name: str = None):
        self.experiment_name = experiment_name or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.log_dir = Path(f"logs/{self.experiment_name}")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup structured logging
        log_format = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # File handler with rotation
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            self.log_dir / "training.log",
            maxBytes=50*1024*1024,  # 50MB
            backupCount=5
        )
        file_handler.setFormatter(log_format)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_format)
        
        # Configure root logger
        self.logger = logging.getLogger()
        self.logger.setLevel(getattr(logging, log_level.upper()))
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Prevent duplicate logs
        self.logger.propagate = False
        
        # Performance metrics
        self.metrics_file = self.log_dir / "metrics.jsonl"
        self.system_stats = []
        self.training_stats = []
        
        # Initialize monitoring
        self._setup_monitoring()
    
    def _setup_monitoring(self):
        """Setup system monitoring."""
        # Initialize wandb if available
        if HAS_WANDB:
            try:
                wandb.init(
                    project="conversational-transformer",
                    name=self.experiment_name,
                    dir=str(self.log_dir)
                )
                self.use_wandb = True
            except Exception as e:
                logging.warning(f"Failed to initialize wandb: {e}")
                self.use_wandb = False
        else:
            self.use_wandb = False
        
        # Initialize tensorboard if available
        if HAS_TENSORBOARD:
            try:
                self.tb_writer = SummaryWriter(str(self.log_dir / "tensorboard"))
                self.use_tensorboard = True
            except Exception as e:
                logging.warning(f"Failed to initialize tensorboard: {e}")
                self.use_tensorboard = False
        else:
            self.use_tensorboard = False
    
    def log_metrics(self, metrics: Dict[str, Any], step: int, prefix: str = ""):
        """Log metrics to all available backends."""
        timestamp = datetime.now().isoformat()
        
        # Prepare metrics with prefix
        prefixed_metrics = {}
        for key, value in metrics.items():
            full_key = f"{prefix}/{key}" if prefix else key
            prefixed_metrics[full_key] = value
        
        # Log to file
        metric_entry = {
            "timestamp": timestamp,
            "step": step,
            "metrics": prefixed_metrics
        }
        
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(metric_entry) + '\n')
        
        # Log to wandb
        if self.use_wandb:
            wandb.log(prefixed_metrics, step=step)
        
        # Log to tensorboard
        if self.use_tensorboard:
            for key, value in prefixed_metrics.items():
                if isinstance(value, (int, float)):
                    self.tb_writer.add_scalar(key, value, step)
    
    def log_system_stats(self, step: int):
        """Log system performance statistics."""
        stats = {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_available_gb": psutil.virtual_memory().available / 1e9,
        }
        
        if torch.cuda.is_available():
            stats.update({
                "gpu_memory_allocated_gb": torch.cuda.memory_allocated() / 1e9,
                "gpu_memory_reserved_gb": torch.cuda.memory_reserved() / 1e9,
                "gpu_utilization": torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0
            })
        
        self.log_metrics(stats, step, "system")
    
    def close(self):
        """Close all logging backends."""
        if self.use_wandb:
            wandb.finish()
        
        if self.use_tensorboard:
            self.tb_writer.close()

# =============================================================================
# ENHANCED CONFIGURATION WITH VALIDATION
# =============================================================================

@dataclass
class Config:
    """Enhanced configuration with validation and serialization."""
    # Model architecture
    vocab_size: int = 50304
    hidden_size: int = 512
    num_layers: int = 8
    num_heads: int = 8
    num_kv_heads: int = 4
    seq_length: int = 1024
    intermediate_size: int = 1536
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    dropout: float = 0.0
    
    # Training parameters
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    eval_every_n_batches: int = 500
    save_every_n_batches: int = 1000
    max_grad_norm: float = 1.0
    precision: str = "fp16"
    compile: bool = False
    
    # Data parameters
    train_data_path: str = "data/train.jsonl"
    eval_data_path: str = "data/eval.jsonl"
    num_workers: int = 2
    assistant_loss_weight: float = 1.5
    max_conversations_per_file: int = 10000
    
    # Generation parameters
    max_new_tokens: int = 512
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 50
    
    # Stability and optimization
    init_std: float = 0.02
    layer_norm_eps: float = 1e-5
    use_stable_embedding: bool = True
    gradient_checkpointing: bool = False
    
    # Production settings
    experiment_name: str = None
    seed: int = 42
    log_level: str = "INFO"
    save_total_limit: int = 5
    early_stopping_patience: int = None
    min_lr: float = 1e-6
    lr_scheduler: str = "cosine"  # cosine, linear, onecycle
    
    # Monitoring and fault tolerance
    health_check_interval: int = 100
    auto_resume: bool = True
    backup_every_n_hours: int = 6
    max_retries: int = 3
    
    def __post_init__(self):
        self.validate()
        
        # Set experiment name if not provided
        if self.experiment_name is None:
            self.experiment_name = f"transformer_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Ensure vocab size is efficient
        if self.vocab_size % 64 != 0:
            self.vocab_size = ((self.vocab_size + 63) // 64) * 64
        
        self.effective_batch_size = self.batch_size * self.gradient_accumulation_steps
    
    def validate(self):
        """Validate configuration parameters."""
        assert self.hidden_size % self.num_heads == 0, "hidden_size must be divisible by num_heads"
        assert self.num_heads % self.num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
        assert self.precision in ["fp16", "bf16", "fp32"], f"Invalid precision: {self.precision}"
        assert self.lr_scheduler in ["cosine", "linear", "onecycle"], f"Invalid scheduler: {self.lr_scheduler}"
        assert self.learning_rate > 0, "Learning rate must be positive"
        assert self.weight_decay >= 0, "Weight decay must be non-negative"
        assert self.num_epochs > 0, "Number of epochs must be positive"
        assert self.warmup_ratio >= 0 and self.warmup_ratio <= 1, "Warmup ratio must be between 0 and 1"
    
    def save(self, path: str):
        """Save configuration to file."""
        config_dict = asdict(self)
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    @classmethod
    def load(cls, path: str) -> 'Config':
        """Load configuration from file."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

# =============================================================================
# FAULT-TOLERANT TRAINING ORCHESTRATOR
# =============================================================================

class TrainingOrchestrator:
    """Orchestrates training with fault tolerance and monitoring."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = ProductionLogger(config.log_level, config.experiment_name)
        
        # Set random seeds for reproducibility
        self._set_seeds(config.seed)
        
        # Create experiment directory
        self.experiment_dir = Path(f"experiments/{config.experiment_name}")
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        config.save(str(self.experiment_dir / "config.yaml"))
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
        # State tracking
        self.is_training = False
        self.should_stop = False
        self.last_backup_time = time.time()
        
        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()
        
        # Health monitoring
        self.health_stats = {
            'last_loss': float('inf'),
            'loss_history': [],
            'consecutive_nan_batches': 0,
            'gradient_norm_history': []
        }
    
    def _set_seeds(self, seed: int):
        """Set random seeds for reproducibility."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        import random
        random.seed(seed)
        
        # Ensure deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logging.info(f"Received signal {signum}, initiating graceful shutdown...")
            self.should_stop = True
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def initialize_training(self):
        """Initialize all training components."""
        logging.info("Initializing training components...")
        
        # Initialize tokenizer
        self.tokenizer = ConversationTokenizer()
        self.config.vocab_size = self.tokenizer.vocab_size
        
        # Initialize model
        self.model = TransformerModel(self.config)
        
        # Initialize trainer
        self.trainer = EnhancedConversationTrainer(
            self.model, self.tokenizer, self.config, self.logger
        )
        
        logging.info("Training components initialized successfully")
    
    def run_training(self):
        """Run the complete training pipeline with fault tolerance."""
        max_retries = self.config.max_retries
        retry_count = 0
        
        while retry_count <= max_retries:
            try:
                self._run_training_iteration()
                break  # Success
                
            except KeyboardInterrupt:
                logging.info("Training interrupted by user")
                self._save_emergency_checkpoint()
                break
                
            except Exception as e:
                retry_count += 1
                logging.error(f"Training failed (attempt {retry_count}/{max_retries + 1}): {e}")
                logging.error(traceback.format_exc())
                
                if retry_count <= max_retries:
                    logging.info(f"Retrying training in 30 seconds...")
                    time.sleep(30)
                    
                    # Try to recover from last checkpoint
                    if self.config.auto_resume:
                        self._attempt_recovery()
                else:
                    logging.error("Maximum retries exceeded, training failed")
                    self._save_emergency_checkpoint()
                    raise
    
    def _run_training_iteration(self):
        """Single training iteration."""
        self.is_training = True
        
        try:
            # Initialize if not already done
            if self.trainer is None:
                self.initialize_training()
            
            # Setup datasets
            train_dataset, eval_dataset = self._setup_datasets()
            
            # Run training
            self.trainer.train(train_dataset, eval_dataset)
            
            logging.info("Training completed successfully!")
            
        finally:
            self.is_training = False
            self.logger.close()
    
    def _setup_datasets(self):
        """Setup training and evaluation datasets."""
        logging.info("Setting up datasets...")
        
        # Validate data files
        if not Path(self.config.train_data_path).exists():
            raise FileNotFoundError(f"Training data not found: {self.config.train_data_path}")
        
        train_dataset = ConversationDataset(
            self.config.train_data_path, self.tokenizer, self.config, "train"
        )
        
        eval_dataset = None
        if Path(self.config.eval_data_path).exists():
            eval_dataset = ConversationDataset(
                self.config.eval_data_path, self.tokenizer, self.config, "eval"
            )
        
        return train_dataset, eval_dataset
    
    def _attempt_recovery(self):
        """Attempt to recover from the latest checkpoint."""
        checkpoint_dir = Path("checkpoints")
        if not checkpoint_dir.exists():
            return
        
        # Find latest checkpoint
        checkpoints = list(checkpoint_dir.glob("*.pt"))
        if not checkpoints:
            return
        
        latest_checkpoint = max(checkpoints, key=lambda x: x.stat().st_mtime)
        
        try:
            logging.info(f"Attempting recovery from {latest_checkpoint}")
            if self.trainer:
                self.trainer.load_checkpoint(str(latest_checkpoint))
            logging.info("Recovery successful")
        except Exception as e:
            logging.error(f"Recovery failed: {e}")
    
    def _save_emergency_checkpoint(self):
        """Save emergency checkpoint on failure."""
        if self.trainer and self.is_training:
            try:
                checkpoint_path = f"checkpoints/emergency_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
                self.trainer.save_checkpoint(self.trainer.current_epoch, emergency=True)
                logging.info(f"Emergency checkpoint saved: {checkpoint_path}")
            except Exception as e:
                logging.error(f"Failed to save emergency checkpoint: {e}")

# =============================================================================
# ENHANCED TOKENIZER (Same as before but with better error handling)
# =============================================================================

class ConversationTokenizer:
    """Production tokenizer with enhanced error handling and validation."""
    
    def __init__(self, model_name: str = "gpt2"):
        try:
            self.tokenizer = tiktoken.get_encoding(model_name)
        except Exception as e:
            logging.error(f"Failed to load tokenizer {model_name}: {e}")
            raise
            
        self.base_vocab_size = self.tokenizer.n_vocab
        
        # Special tokens for conversation structure
        self.special_tokens = {
            "<|im_start|>": self.base_vocab_size,
            "<|im_end|>": self.base_vocab_size + 1,
            "<|user|>": self.base_vocab_size + 2,
            "<|assistant|>": self.base_vocab_size + 3,
            "<|system|>": self.base_vocab_size + 4,
        }
        
        self.vocab_size = self.base_vocab_size + len(self.special_tokens)
        self._reverse_special_tokens = {v: k for k, v in self.special_tokens.items()}
        
        # Pad vocab size to be efficient
        if self.vocab_size % 64 != 0:
            self.vocab_size = ((self.vocab_size + 63) // 64) * 64
            
        logging.info(f"Tokenizer initialized with vocab size: {self.vocab_size}")
    
    def encode_conversation(self, conversation: Dict[str, any]) -> List[int]:
        """Encode conversation with enhanced error handling."""
        try:
            tokens = []
            messages = conversation.get('messages', [])
            
            if not messages:
                return tokens
            
            for message in messages:
                role = message.get('role', '').lower()
                content = message.get('content', '').strip()
                
                if not content:
                    continue
                
                # Validate role
                if role not in ['user', 'prompter', 'assistant', 'system']:
                    role = 'user'  # Default fallback
                
                # Start message
                tokens.append(self.special_tokens["<|im_start|>"])
                
                # Add role
                if role == 'user' or role == 'prompter':
                    tokens.append(self.special_tokens["<|user|>"])
                elif role == 'assistant':
                    tokens.append(self.special_tokens["<|assistant|>"])
                else:
                    tokens.append(self.special_tokens["<|system|>"])
                
                # Add content with error handling
                try:
                    content_tokens = self.tokenizer.encode(content)
                    tokens.extend(content_tokens)
                except Exception as e:
                    logging.warning(f"Failed to encode content: {e}")
                    continue
                
                # End message
                tokens.append(self.special_tokens["<|im_end|>"])
            
            return tokens
            
        except Exception as e:
            logging.error(f"Conversation encoding failed: {e}")
            return []
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode with enhanced error handling."""
        try:
            # Filter out special tokens if requested
            if skip_special_tokens:
                filtered_tokens = []
                for token_id in token_ids:
                    if token_id not in self._reverse_special_tokens and token_id < self.base_vocab_size:
                        filtered_tokens.append(token_id)
                token_ids = filtered_tokens
            
            # Clamp invalid tokens
            token_ids = [max(0, min(token_id, self.base_vocab_size - 1)) for token_id in token_ids]
            
            return self.tokenizer.decode(token_ids)
        except Exception as e:
            logging.warning(f"Decode error: {e}")
            return "<decode_error>"
    
    def is_special_token(self, token_id: int) -> bool:
        """Check if token is a special token."""
        return token_id in self._reverse_special_tokens
    
    def get_role_token(self, role: str) -> int:
        """Get token ID for a role."""
        role_map = {
            'user': self.special_tokens["<|user|>"],
            'prompter': self.special_tokens["<|user|>"], 
            'assistant': self.special_tokens["<|assistant|>"],
            'system': self.special_tokens["<|system|>"]
        }
        return role_map.get(role.lower(), self.special_tokens["<|user|>"])

# =============================================================================
# MODEL ARCHITECTURE (Enhanced with better stability)
# =============================================================================

class RMSNorm(nn.Module):
    """Enhanced RMSNorm with better numerical stability."""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        
        # Initialize with proper scaling
        with torch.no_grad():
            self.weight.fill_(1.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Enhanced numerical stability
        dtype = x.dtype
        x_fp32 = x.float()
        
        # Compute RMS with better numerical properties
        variance = x_fp32.pow(2).mean(-1, keepdim=True)
        x_normed = x_fp32 * torch.rsqrt(variance + self.eps)
        
        # Apply weight and convert back to original dtype
        return (x_normed * self.weight.float()).to(dtype)

class RotaryEmbedding(nn.Module):
    """Enhanced RoPE with better caching and stability."""
    
    def __init__(self, dim: int, max_seq_len: int = 8192, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        
        # Create frequency tensor with enhanced precision
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))
        self.register_buffer("inv_freq", inv_freq.float(), persistent=False)
        
        # Pre-compute embeddings
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len: int):
        """Build cache with better precision."""
        device = self.inv_freq.device
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # Use higher precision for cos/sin computation
        emb_fp64 = emb.double()
        self.register_buffer("cos_cached", emb_fp64.cos().float(), persistent=False)
        self.register_buffer("sin_cached", emb_fp64.sin().float(), persistent=False)
        self._cached_seq_len = seq_len
    
    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self._cached_seq_len:
            self._build_cache(max(seq_len, min(self._cached_seq_len * 2, self.max_seq_len)))
        
        cos = self.cos_cached[:seq_len].to(device)
        sin = self.sin_cached[:seq_len].to(device)
        
        return cos, sin

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, 
                        cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding with proper shape handling."""
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    
    # Ensure proper broadcasting
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, dim]
    sin = sin.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, dim]
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class GroupedQueryAttention(nn.Module):
    """Enhanced GQA with better stability and optional flash attention."""
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.scale = self.head_dim ** -0.5
        
        # Linear projections
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        
        # RoPE
        self.rope = RotaryEmbedding(self.head_dim, config.seq_length, config.rope_theta)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else None
        
        self._init_weights()
    
    def _init_weights(self):
        """Enhanced weight initialization."""
        std = self.config.init_std
        
        # Initialize projections with scaled normal distribution
        for proj in [self.q_proj, self.k_proj, self.v_proj]:
            nn.init.normal_(proj.weight, mean=0.0, std=std)
        
        # Output projection with scaled initialization for stability
        nn.init.normal_(self.o_proj.weight, mean=0.0, std=std / math.sqrt(2 * self.config.num_layers))
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, D = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        cos, sin = self.rope(L, x.device)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Expand K, V for GQA if needed
        if self.num_queries_per_kv > 1:
            k = k.repeat_interleave(self.num_queries_per_kv, dim=1)
            v = v.repeat_interleave(self.num_queries_per_kv, dim=1)
        
        # Use Flash Attention if available and beneficial
        if HAS_FLASH_ATTN and L > 512 and x.dtype in [torch.float16, torch.bfloat16]:
            try:
                # Reshape for flash attention
                q = q.transpose(1, 2).contiguous()  # [B, L, H, D]
                k = k.transpose(1, 2).contiguous()  # [B, L, H, D]
                v = v.transpose(1, 2).contiguous()  # [B, L, H, D]
                
                out = flash_attn_func(q, k, v, causal=True, softmax_scale=self.scale)
                out = out.reshape(B, L, self.hidden_size)
                
            except Exception as e:
                logging.warning(f"Flash attention failed, falling back to standard: {e}")
                # Fall back to standard attention
                out = self._standard_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attention_mask)
        else:
            # Standard attention
            out = self._standard_attention(q, k, v, attention_mask)
        
        return self.o_proj(out)
    
    def _standard_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                          attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Standard attention implementation with enhanced stability."""
        B, H, L, D = q.shape
        
        # Compute attention scores with improved numerical stability
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply causal mask
        causal_mask = torch.triu(torch.ones(L, L, device=q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal_mask, -1e4)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            scores = scores + (1.0 - attention_mask) * -1e4
        
        # Stable softmax computation
        scores_max = scores.detach().max(dim=-1, keepdim=True)[0]
        scores_stable = scores - scores_max
        attn = F.softmax(scores_stable, dim=-1, dtype=torch.float32).to(q.dtype)
        
        # Apply dropout
        if self.dropout is not None and self.training:
            attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, L, self.hidden_size)
        
        return out

class SwiGLU(nn.Module):
    """Enhanced SwiGLU with better initialization."""
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self._init_weights()
    
    def _init_weights(self):
        """Enhanced weight initialization for stability."""
        std = self.config.init_std
        
        # GLU initialization
        nn.init.normal_(self.gate_proj.weight, mean=0.0, std=std)
        nn.init.normal_(self.up_proj.weight, mean=0.0, std=std)
        
        # Output projection with scaling
        nn.init.normal_(self.down_proj.weight, mean=0.0, std=std / math.sqrt(2 * self.config.num_layers))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)

class TransformerBlock(nn.Module):
    """Enhanced transformer block with optional gradient checkpointing."""
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.input_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = GroupedQueryAttention(config)
        self.post_attn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = SwiGLU(config)
        
        # Gradient checkpointing flag
        self.gradient_checkpointing = config.gradient_checkpointing
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.gradient_checkpointing and self.training:
            return self._forward_with_checkpointing(x, attention_mask)
        else:
            return self._forward_impl(x, attention_mask)
    
    def _forward_impl(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward implementation with residual connections."""
        # Self-attention with pre-norm
        attn_out = self.self_attn(self.input_norm(x), attention_mask)
        x = x + attn_out
        
        # MLP with pre-norm
        mlp_out = self.mlp(self.post_attn_norm(x))
        x = x + mlp_out
        
        return x
    
    def _forward_with_checkpointing(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward with gradient checkpointing."""
        return torch.utils.checkpoint.checkpoint(
            self._forward_impl, x, attention_mask, use_reentrant=False
        )

class TransformerModel(nn.Module):
    """Enhanced transformer model with better initialization and monitoring."""
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        # Embedding layer
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Optional embedding scaling for stability
        if config.use_stable_embedding:
            self.embed_scale = math.sqrt(config.hidden_size)
        else:
            self.embed_scale = 1.0
        
        # Transformer layers
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        
        # Final layer norm
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        
        # Language modeling head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Weight tying
        self.lm_head.weight = self.embed_tokens.weight
        
        # Initialize weights
        self._init_weights()
        
        # Parameter counting and logging
        n_params = sum(p.numel() for p in self.parameters())
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logging.info(f"Model initialized with {n_params:,} total parameters ({n_trainable:,} trainable)")
    
    def _init_weights(self):
        """Enhanced weight initialization for better stability."""
        # Embedding initialization
        nn.init.normal_(self.embed_tokens.weight, mean=0.0, std=self.config.init_std)
        
        # Apply residual scaling for stability
        with torch.no_grad():
            for layer in self.layers:
                # Scale attention output projection
                layer.self_attn.o_proj.weight.data *= 0.67
                # Scale MLP output projection
                layer.mlp.down_proj.weight.data *= 0.67
    
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None,
                return_hidden_states: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """Enhanced forward pass with optional hidden state return."""
        # Input validation
        if torch.any(input_ids >= self.config.vocab_size) or torch.any(input_ids < 0):
            logging.warning("Input contains invalid token IDs, clamping...")
            input_ids = torch.clamp(input_ids, 0, self.config.vocab_size - 1)
        
        # Embedding
        x = self.embed_tokens(input_ids) * self.embed_scale
        
        # Store hidden states if requested
        hidden_states = [] if return_hidden_states else None
        
        # Transformer layers
        for layer in self.layers:
            x = layer(x, attention_mask)
            if return_hidden_states:
                hidden_states.append(x)
        
        # Final layer norm
        x = self.norm(x)
        
        # Language modeling head
        logits = self.lm_head(x)
        
        if return_hidden_states:
            return logits, hidden_states
        return logits
    
    def get_num_params(self, non_embedding: bool = True) -> int:
        """Get parameter count."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.embed_tokens.weight.numel()
        return n_params

# =============================================================================
# ENHANCED DATASET WITH VALIDATION AND MONITORING
# =============================================================================

class ConversationDataset(Dataset):
    """Enhanced dataset with better error handling and monitoring."""
    
    def __init__(self, data_path: str, tokenizer: ConversationTokenizer, 
                 config: Config, split: str = "train"):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.config = config
        self.split = split
        
        # Statistics tracking
        self.stats = {
            'total_loaded': 0,
            'valid_conversations': 0,
            'invalid_conversations': 0,
            'tokenization_errors': 0,
            'avg_token_length': 0,
            'max_token_length': 0,
            'min_token_length': float('inf')
        }
        
        # Load conversations with validation
        self.conversations = self._load_and_validate_conversations()
        self._compute_statistics()
        
        logging.info(f"Dataset {split}: {len(self.conversations):,} conversations from {data_path}")
        logging.info(f"Average tokens: {self.stats['avg_token_length']:.1f}, "
                    f"Max: {self.stats['max_token_length']}, Min: {self.stats['min_token_length']}")
    
    def _load_and_validate_conversations(self) -> List[Dict]:
        """Load and validate conversations with comprehensive error handling."""
        conversations = []
        
        if not self.data_path.exists():
            logging.error(f"Data file not found: {self.data_path}")
            return conversations
        
        logging.info(f"Loading {self.split} data from {self.data_path}")
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line_no, line in enumerate(f, 1):
                try:
                    conversation = json.loads(line.strip())
                    self.stats['total_loaded'] += 1
                    
                    if self._validate_conversation(conversation):
                        conversations.append(conversation)
                        self.stats['valid_conversations'] += 1
                    else:
                        self.stats['invalid_conversations'] += 1
                        
                except json.JSONDecodeError as e:
                    self.stats['invalid_conversations'] += 1
                    if line_no <= 10:  # Only log first few errors
                        logging.warning(f"JSON decode error at line {line_no}: {e}")
                except Exception as e:
                    self.stats['invalid_conversations'] += 1
                    logging.warning(f"Error loading conversation {line_no}: {e}")
                
                # Progress logging for large datasets
                if line_no % 10000 == 0:
                    logging.info(f"Processed {line_no:,} lines, {len(conversations):,} valid conversations")
        
        return conversations
    
    def _validate_conversation(self, conversation: Dict) -> bool:
        """Comprehensive conversation validation."""
        if 'messages' not in conversation:
            return False
        
        messages = conversation['messages']
        if not messages or len(messages) < 2:
            return False
        
        # Check message structure and content
        has_user = False
        has_assistant = False
        
        for msg in messages:
            if not isinstance(msg, dict):
                return False
            
            role = msg.get('role', '').lower()
            content = msg.get('content', '').strip()
            
            if not content:
                return False
            
            # Track roles
            if role in ['user', 'prompter']:
                has_user = True
            elif role == 'assistant':
                has_assistant = True
        
        # Require both user and assistant messages
        return has_user and has_assistant
    
    def _compute_statistics(self):
        """Compute dataset statistics."""
        if not self.conversations:
            return
        
        token_lengths = []
        
        # Sample conversations for statistics (to avoid processing all)
        sample_size = min(1000, len(self.conversations))
        sample_indices = np.random.choice(len(self.conversations), sample_size, replace=False)
        
        for idx in sample_indices:
            try:
                tokens = self.tokenizer.encode_conversation(self.conversations[idx])
                if tokens:
                    token_lengths.append(len(tokens))
            except Exception:
                self.stats['tokenization_errors'] += 1
        
        if token_lengths:
            self.stats['avg_token_length'] = np.mean(token_lengths)
            self.stats['max_token_length'] = max(token_lengths)
            self.stats['min_token_length'] = min(token_lengths)
    
    def _process_conversation(self, conversation: Dict) -> Optional[Dict[str, torch.Tensor]]:
        """Process conversation with enhanced error handling."""
        try:
            tokens = self.tokenizer.encode_conversation(conversation)
            
            # Validate token sequence
            if not tokens or len(tokens) < 10:
                return None
            
            # Handle sequence length
            if len(tokens) > self.config.seq_length:
                # Truncate from the beginning to keep the most recent context
                tokens = tokens[-self.config.seq_length:]
            else:
                # Pad to sequence length
                pad_length = self.config.seq_length - len(tokens)
                tokens.extend([0] * pad_length)
            
            tokens = torch.tensor(tokens, dtype=torch.long)
            
            # Create attention mask
            attention_mask = (tokens != 0).float()
            
            # Create labels for next token prediction
            labels = tokens.clone()
            
            # Create loss weights with role-based weighting
            loss_weights = self._create_loss_weights(tokens)
            
            return {
                'input_ids': tokens[:-1],
                'labels': labels[1:],
                'attention_mask': attention_mask[:-1],
                'loss_weights': loss_weights[1:]
            }
            
        except Exception as e:
            logging.debug(f"Error processing conversation: {e}")
            return None
    
    def _create_loss_weights(self, tokens: torch.Tensor) -> torch.Tensor:
        """Create loss weights with assistant response emphasis."""
        loss_weights = torch.ones_like(tokens, dtype=torch.float)
        assistant_token = self.tokenizer.get_role_token('assistant')
        im_end_token = self.tokenizer.special_tokens["<|im_end|>"]
        
        in_assistant_response = False
        for i, token_id in enumerate(tokens):
            if token_id == assistant_token:
                in_assistant_response = True
            elif token_id == im_end_token:
                in_assistant_response = False
            
            # Weight assistant responses higher, but not padding
            if in_assistant_response and token_id != 0:
                loss_weights[i] = self.config.assistant_loss_weight
        
        return loss_weights
    
    def __len__(self) -> int:
        return len(self.conversations)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get processed conversation with fallback."""
        conversation = self.conversations[idx]
        processed = self._process_conversation(conversation)
        
        # Return dummy sample if processing fails
        if processed is None:
            seq_len = self.config.seq_length - 1
            return {
                'input_ids': torch.zeros(seq_len, dtype=torch.long),
                'labels': torch.zeros(seq_len, dtype=torch.long),
                'attention_mask': torch.zeros(seq_len, dtype=torch.float),
                'loss_weights': torch.zeros(seq_len, dtype=torch.float)
            }
        
        return processed
    
    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        return self.stats.copy()

def create_dataloader(dataset: ConversationDataset, config: Config, shuffle: bool = True) -> DataLoader:
    """Create optimized dataloader with error handling."""
    try:
        return DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=shuffle,
            num_workers=config.num_workers,
            pin_memory=torch.cuda.is_available(),
            prefetch_factor=2 if config.num_workers > 0 else None,
            drop_last=True,
            persistent_workers=config.num_workers > 0
        )
    except Exception as e:
        logging.warning(f"Failed to create optimized dataloader: {e}")
        # Fallback to basic dataloader
        return DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=shuffle,
            num_workers=0,
            drop_last=True
        )

# =============================================================================
# ENHANCED TRAINER WITH COMPREHENSIVE MONITORING
# =============================================================================

class EnhancedConversationTrainer:
    """Production trainer with comprehensive monitoring and fault tolerance."""
    
    def __init__(self, model: TransformerModel, tokenizer: ConversationTokenizer, 
                 config: Config, logger: ProductionLogger):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.logger = logger
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # GPU setup and memory management
        self._setup_gpu()
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = None
        
        # Mixed precision setup
        self.use_amp = config.precision in ["fp16", "bf16"]
        self.dtype = torch.bfloat16 if config.precision == "bf16" else torch.float16
        self.scaler = GradScaler() if config.precision == "fp16" else None
        
        # Model compilation
        self._compile_model()
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_eval_loss = float('inf')
        self.patience_counter = 0
        
        # Metrics and monitoring
        self.metrics = {
            'train_losses': [],
            'eval_losses': [],
            'learning_rates': [],
            'gradient_norms': [],
            'throughput': [],
            'epoch_times': []
        }
        
        # Health monitoring
        self.health_monitor = TrainingHealthMonitor()
        
        # Checkpoint management
        self.checkpoint_manager = CheckpointManager(config)
        
        logging.info(f"Trainer initialized on {self.device}")
        self._log_memory_usage("Initial")
    
    def _setup_gpu(self):
        """Setup GPU with optimal configuration."""
        if torch.cuda.is_available():
            # Clear cache
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Set memory fraction
            torch.cuda.set_per_process_memory_fraction(0.85)
            
            # Log GPU info
            gpu_props = torch.cuda.get_device_properties(0)
            logging.info(f"GPU: {gpu_props.name}, Memory: {gpu_props.total_memory / 1e9:.1f}GB")
        else:
            logging.warning("CUDA not available, using CPU")
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with parameter grouping."""
        # Separate parameters for weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if any(nd in name for nd in ['bias', 'norm', 'embed']):
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
        
        param_groups = [
            {'params': decay_params, 'weight_decay': self.config.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        
        try:
            # Use fused optimizer if available
            return AdamW(
                param_groups,
                lr=self.config.learning_rate,
                betas=(0.9, 0.95),
                eps=1e-8,
                fused=torch.cuda.is_available()
            )
        except Exception:
            # Fallback to standard AdamW
            return AdamW(
                param_groups,
                lr=self.config.learning_rate,
                betas=(0.9, 0.95),
                eps=1e-8
            )
    
    def _compile_model(self):
        """Compile model with error handling."""
        if self.config.compile and hasattr(torch, 'compile'):
            try:
                logging.info("Compiling model...")
                self.model = torch.compile(self.model, mode='default')
                logging.info("Model compiled successfully")
            except Exception as e:
                logging.warning(f"Model compilation failed: {e}")
    
    def _setup_scheduler(self, total_steps: int):
        """Setup learning rate scheduler."""
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        
        if self.config.lr_scheduler == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer, T_max=total_steps, eta_min=self.config.min_lr
            )
        elif self.config.lr_scheduler == "onecycle":
            self.scheduler = OneCycleLR(
                self.optimizer, max_lr=self.config.learning_rate,
                total_steps=total_steps, pct_start=self.config.warmup_ratio
            )
        else:  # linear
            from torch.optim.lr_scheduler import LinearLR
            self.scheduler = LinearLR(
                self.optimizer, start_factor=0.1, total_iters=warmup_steps
            )
    
    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor, 
                    loss_weights: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute weighted loss with detailed metrics."""
        # Flatten tensors
        flat_logits = logits.view(-1, logits.size(-1))
        flat_labels = labels.view(-1)
        flat_weights = loss_weights.view(-1)
        
        # Compute base loss
        loss = F.cross_entropy(flat_logits, flat_labels, reduction='none')
        
        # Apply weights and mask padding
        mask = (flat_labels != 0).float()
        weighted_loss = loss * flat_weights * mask
        
        # Check for numerical issues
        if torch.isnan(weighted_loss).any() or torch.isinf(weighted_loss).any():
            logging.warning("NaN or Inf detected in loss computation")
            return {
                'loss': torch.tensor(0.0, device=loss.device, requires_grad=True),
                'raw_loss': torch.tensor(0.0, device=loss.device),
                'perplexity': torch.tensor(float('inf'), device=loss.device),
                'valid_tokens': torch.tensor(0.0, device=loss.device)
            }
        
        # Compute final loss
        total_loss = weighted_loss.sum()
        total_weight = mask.sum().clamp(min=1)
        final_loss = total_loss / total_weight
        
        # Compute additional metrics
        raw_loss = (loss * mask).sum() / mask.sum().clamp(min=1)
        perplexity = torch.exp(raw_loss.clamp(max=10))  # Clamp to prevent overflow
        
        return {
            'loss': final_loss,
            'raw_loss': raw_loss,
            'perplexity': perplexity,
            'valid_tokens': mask.sum()
        }
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Enhanced training step with comprehensive monitoring."""
        self.model.train()
        
        # Move batch to device
        batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
        
        # Skip empty batches
        if batch['input_ids'].numel() == 0:
            return {'loss': 0.0, 'perplexity': float('inf'), 'valid_tokens': 0}
        
        # Forward pass with autocast
        with autocast(device_type='cuda', enabled=self.use_amp, dtype=self.dtype):
            logits = self.model(batch['input_ids'], batch['attention_mask'])
            loss_dict = self.compute_loss(logits, batch['labels'], batch['loss_weights'])
            loss = loss_dict['loss'] / self.config.gradient_accumulation_steps
        
        # Check for valid loss
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            logging.warning("Invalid loss detected, skipping batch")
            return {'loss': 0.0, 'perplexity': float('inf'), 'valid_tokens': 0}
        
        # Backward pass
        if self.use_amp and self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Return metrics
        return {
            'loss': loss.item() * self.config.gradient_accumulation_steps,
            'raw_loss': loss_dict['raw_loss'].item(),
            'perplexity': loss_dict['perplexity'].item(),
            'valid_tokens': loss_dict['valid_tokens'].item()
        }
    
    def optimizer_step(self) -> Dict[str, float]:
        """Enhanced optimizer step with gradient monitoring."""
        # Unscale gradients for AMP
        if self.use_amp and self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
        
        # Compute gradient norm before clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.max_grad_norm
        )
        
        # Check for NaN gradients
        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
            logging.warning("NaN/Inf gradients detected, skipping step")
            self.optimizer.zero_grad(set_to_none=True)
            if self.use_amp and self.scaler is not None:
                self.scaler.update()
            return {'grad_norm': 0.0, 'lr': 0.0}
        
        # Optimizer step
        if self.use_amp and self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        # Clear gradients
        self.optimizer.zero_grad(set_to_none=True)
        
        # Update scheduler
        if self.scheduler:
            self.scheduler.step()
        
        # Get current learning rate
        current_lr = self.scheduler.get_last_lr()[0] if self.scheduler else self.config.learning_rate
        
        return {'grad_norm': grad_norm.item(), 'lr': current_lr}
    
    @torch.no_grad()
    def evaluate(self, eval_dataset: ConversationDataset, max_batches: int = 100) -> Dict[str, float]:
        """Comprehensive evaluation with multiple metrics."""
        self.model.eval()
        
        eval_dataloader = create_dataloader(eval_dataset, self.config, shuffle=False)
        
        total_loss = 0.0
        total_raw_loss = 0.0
        total_tokens = 0
        num_batches = 0
        
        eval_start_time = time.time()
        
        for batch_idx, batch in enumerate(eval_dataloader):
            if batch_idx >= max_batches:
                break
            
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
            
            if batch['input_ids'].numel() == 0:
                continue
            
            with autocast(device_type='cuda', enabled=self.use_amp, dtype=self.dtype):
                logits = self.model(batch['input_ids'], batch['attention_mask'])
                loss_dict = self.compute_loss(logits, batch['labels'], batch['loss_weights'])
            
            if not (torch.isnan(loss_dict['loss']).any() or torch.isinf(loss_dict['loss']).any()):
                total_loss += loss_dict['loss'].item()
                total_raw_loss += loss_dict['raw_loss'].item()
                total_tokens += loss_dict['valid_tokens'].item()
                num_batches += 1
        
        eval_time = time.time() - eval_start_time
        
        if num_batches == 0:
            return {
                'eval_loss': float('inf'),
                'eval_perplexity': float('inf'),
                'eval_time': eval_time,
                'eval_throughput': 0.0
            }
        
        avg_loss = total_loss / num_batches
        avg_raw_loss = total_raw_loss / num_batches
        perplexity = math.exp(min(avg_raw_loss, 10))  # Clamp to prevent overflow
        throughput = total_tokens / eval_time if eval_time > 0 else 0
        
        return {
            'eval_loss': avg_loss,
            'eval_perplexity': perplexity,
            'eval_time': eval_time,
            'eval_throughput': throughput
        }
    
    def train_epoch(self, train_dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train one epoch with comprehensive monitoring."""
        self.model.train()
        
        epoch_metrics = {
            'total_loss': 0.0,
            'total_raw_loss': 0.0,
            'total_tokens': 0,
            'num_batches': 0,
            'grad_norm_sum': 0.0
        }
        
        accumulation_metrics = {
            'loss': 0.0,
            'raw_loss': 0.0,
            'tokens': 0
        }
        
        epoch_start_time = time.time()
        last_log_time = time.time()
        
        for batch_idx, batch in enumerate(train_dataloader):
            # Check for stop signal
            if hasattr(self, 'should_stop') and self.should_stop:
                break
            
            step_start_time = time.time()
            
            # Training step
            step_metrics = self.train_step(batch)
            
            # Accumulate metrics
            accumulation_metrics['loss'] += step_metrics['loss']
            accumulation_metrics['raw_loss'] += step_metrics['raw_loss']
            accumulation_metrics['tokens'] += step_metrics['valid_tokens']
            
            # Optimizer step after accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                opt_metrics = self.optimizer_step()
                self.global_step += 1
                
                # Update epoch metrics
                if accumulation_metrics['loss'] > 0:
                    epoch_metrics['total_loss'] += accumulation_metrics['loss']
                    epoch_metrics['total_raw_loss'] += accumulation_metrics['raw_loss']
                    epoch_metrics['total_tokens'] += accumulation_metrics['tokens']
                    epoch_metrics['num_batches'] += 1
                    epoch_metrics['grad_norm_sum'] += opt_metrics['grad_norm']
                
                # Log metrics
                step_time = time.time() - step_start_time
                tokens_per_sec = accumulation_metrics['tokens'] / step_time if step_time > 0 else 0
                
                self.metrics['train_losses'].append(accumulation_metrics['loss'])
                self.metrics['learning_rates'].append(opt_metrics['lr'])
                self.metrics['gradient_norms'].append(opt_metrics['grad_norm'])
                self.metrics['throughput'].append(tokens_per_sec)
                
                # Health monitoring
                self.health_monitor.update(accumulation_metrics['loss'], opt_metrics['grad_norm'])
                
                # Periodic logging
                current_time = time.time()
                if self.global_step % 50 == 0 or current_time - last_log_time > 30:
                    self._log_training_step(
                        epoch, batch_idx, len(train_dataloader),
                        accumulation_metrics, opt_metrics, tokens_per_sec
                    )
                    last_log_time = current_time
                
                # Log to monitoring backends
                if self.global_step % 10 == 0:
                    self.logger.log_metrics({
                        'train_loss': accumulation_metrics['loss'],
                        'learning_rate': opt_metrics['lr'],
                        'gradient_norm': opt_metrics['grad_norm'],
                        'throughput_tokens_per_sec': tokens_per_sec,
                        'perplexity': math.exp(min(accumulation_metrics['raw_loss'], 10))
                    }, self.global_step, "train")
                
                # System monitoring
                if self.global_step % self.config.health_check_interval == 0:
                    self.logger.log_system_stats(self.global_step)
                    self._log_memory_usage(f"Step {self.global_step}")
                
                # Periodic evaluation
                if (self.config.eval_every_n_batches > 0 and 
                    self.global_step % self.config.eval_every_n_batches == 0):
                    self._periodic_evaluation()
                
                # Periodic checkpointing
                if (self.config.save_every_n_batches > 0 and 
                    self.global_step % self.config.save_every_n_batches == 0):
                    self.checkpoint_manager.save_checkpoint(
                        self.model, self.optimizer, self.scheduler,
                        self.global_step, self.current_epoch, self.metrics,
                        f"step_{self.global_step:06d}"
                    )
                
                # Reset accumulation metrics
                accumulation_metrics = {'loss': 0.0, 'raw_loss': 0.0, 'tokens': 0}
        
        # Handle remaining gradients
        if (batch_idx + 1) % self.config.gradient_accumulation_steps != 0:
            opt_metrics = self.optimizer_step()
            self.global_step += 1
        
        # Compute epoch statistics
        epoch_time = time.time() - epoch_start_time
        self.metrics['epoch_times'].append(epoch_time)
        
        if epoch_metrics['num_batches'] > 0:
            avg_loss = epoch_metrics['total_loss'] / epoch_metrics['num_batches']
            avg_raw_loss = epoch_metrics['total_raw_loss'] / epoch_metrics['num_batches']
            avg_grad_norm = epoch_metrics['grad_norm_sum'] / epoch_metrics['num_batches']
            avg_tokens_per_sec = epoch_metrics['total_tokens'] / epoch_time
        else:
            avg_loss = float('inf')
            avg_raw_loss = float('inf')
            avg_grad_norm = 0.0
            avg_tokens_per_sec = 0.0
        
        logging.info(f"Epoch {epoch+1} completed in {epoch_time:.2f}s | "
                    f"Avg Loss: {avg_loss:.6f} | "
                    f"Avg Grad Norm: {avg_grad_norm:.4f} | "
                    f"Throughput: {avg_tokens_per_sec:.0f} tokens/s")
        
        return {
            'avg_loss': avg_loss,
            'avg_raw_loss': avg_raw_loss,
            'avg_grad_norm': avg_grad_norm,
            'epoch_time': epoch_time,
            'throughput': avg_tokens_per_sec
        }
    
    def _log_training_step(self, epoch: int, batch_idx: int, total_batches: int,
                          metrics: Dict[str, float], opt_metrics: Dict[str, float],
                          tokens_per_sec: float):
        """Log training step with comprehensive information."""
        # Memory info
        memory_info = ""
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1e9
            memory_cached = torch.cuda.memory_reserved() / 1e9
            memory_info = f" | GPU: {memory_allocated:.1f}GB/{memory_cached:.1f}GB"
        
        # Health status
        health_status = self.health_monitor.get_status()
        health_info = f" | Health: {health_status}"
        
        logging.info(
            f"Epoch {epoch+1} | Step {self.global_step:6d} | "
            f"Batch {batch_idx+1:4d}/{total_batches} | "
            f"Loss: {metrics['loss']:.6f} | "
            f"PPL: {math.exp(min(metrics['raw_loss'], 10)):.2f} | "
            f"LR: {opt_metrics['lr']:.2e} | "
            f"GradNorm: {opt_metrics['grad_norm']:.4f} | "
            f"Tokens/s: {tokens_per_sec:.0f}"
            f"{memory_info}{health_info}"
        )
    
    def _periodic_evaluation(self):
        """Perform periodic evaluation during training."""
        if hasattr(self, 'eval_dataset') and self.eval_dataset is not None:
            eval_metrics = self.evaluate(self.eval_dataset, max_batches=50)
            
            # Log evaluation metrics
            self.logger.log_metrics(eval_metrics, self.global_step, "eval")
            
            logging.info(f"Eval | Step {self.global_step} | "
                        f"Loss: {eval_metrics['eval_loss']:.6f} | "
                        f"PPL: {eval_metrics['eval_perplexity']:.2f}")
            
            # Early stopping check
            if self.config.early_stopping_patience:
                self._check_early_stopping(eval_metrics['eval_loss'])
    
    def _check_early_stopping(self, eval_loss: float):
        """Check early stopping condition."""
        if eval_loss < self.best_eval_loss:
            self.best_eval_loss = eval_loss
            self.patience_counter = 0
            # Save best model
            self.checkpoint_manager.save_checkpoint(
                self.model, self.optimizer, self.scheduler,
                self.global_step, self.current_epoch, self.metrics,
                "best_model"
            )
        else:
            self.patience_counter += 1
            
        if self.patience_counter >= self.config.early_stopping_patience:
            logging.info(f"Early stopping triggered after {self.patience_counter} steps without improvement")
            self.should_stop = True
    
    def _log_memory_usage(self, context: str):
        """Log memory usage information."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            max_allocated = torch.cuda.max_memory_allocated() / 1e9
            logging.info(f"{context} - GPU Memory: {allocated:.2f}GB allocated, "
                        f"{reserved:.2f}GB reserved, {max_allocated:.2f}GB max")
        
        # System memory
        memory = psutil.virtual_memory()
        logging.info(f"{context} - System Memory: {memory.percent:.1f}% used, "
                    f"{memory.available / 1e9:.1f}GB available")
    
    def train(self, train_dataset: ConversationDataset, 
              eval_dataset: Optional[ConversationDataset] = None):
        """Main training loop with comprehensive monitoring."""
        logging.info("="*80)
        logging.info("STARTING PRODUCTION TRAINING")
        logging.info("="*80)
        
        # Store eval dataset for periodic evaluation
        self.eval_dataset = eval_dataset
        
        # Setup data loaders
        train_dataloader = create_dataloader(train_dataset, self.config, shuffle=True)
        
        # Calculate total steps and setup scheduler
        total_steps = len(train_dataloader) * self.config.num_epochs // self.config.gradient_accumulation_steps
        self._setup_scheduler(total_steps)
        
        # Log training configuration
        self._log_training_config(len(train_dataloader), total_steps)
        
        training_start_time = time.time()
        
        try:
            for epoch in range(self.current_epoch, self.config.num_epochs):
                if hasattr(self, 'should_stop') and self.should_stop:
                    break
                
                logging.info(f"\n{'='*60}")
                logging.info(f"EPOCH {epoch + 1}/{self.config.num_epochs}")
                logging.info(f"{'='*60}")
                
                # Train epoch
                epoch_metrics = self.train_epoch(train_dataloader, epoch)
                
                # Full evaluation at epoch end
                if eval_dataset is not None:
                    eval_metrics = self.evaluate(eval_dataset)
                    epoch_metrics.update(eval_metrics)
                    
                    # Log epoch summary
                    logging.info(f"Epoch {epoch + 1} Summary:")
                    logging.info(f"  Train Loss: {epoch_metrics['avg_loss']:.6f}")
                    logging.info(f"  Eval Loss: {eval_metrics['eval_loss']:.6f}")
                    logging.info(f"  Eval Perplexity: {eval_metrics['eval_perplexity']:.2f}")
                    
                    # Early stopping check
                    if self.config.early_stopping_patience:
                        self._check_early_stopping(eval_metrics['eval_loss'])
                
                # Log epoch metrics
                self.logger.log_metrics(epoch_metrics, epoch, "epoch")
                
                # Save epoch checkpoint
                self.checkpoint_manager.save_checkpoint(
                    self.model, self.optimizer, self.scheduler,
                    self.global_step, epoch + 1, self.metrics,
                    f"epoch_{epoch + 1:03d}"
                )
                
                self.current_epoch = epoch + 1
                
                # Backup checkpoint periodically
                current_time = time.time()
                if (current_time - getattr(self, 'last_backup_time', 0)) > self.config.backup_every_n_hours * 3600:
                    self._create_backup()
                    self.last_backup_time = current_time
        
        except KeyboardInterrupt:
            logging.info("Training interrupted by user")
        except Exception as e:
            logging.error(f"Training error: {e}")
            raise
        finally:
            total_training_time = time.time() - training_start_time
            logging.info(f"\nTraining finished after {total_training_time / 3600:.2f} hours")
            
            # Save final checkpoint
            self.checkpoint_manager.save_checkpoint(
                self.model, self.optimizer, self.scheduler,
                self.global_step, self.current_epoch, self.metrics,
                "final"
            )
            
            # Save training summary
            self._save_training_summary(total_training_time)
    
    def _log_training_config(self, batches_per_epoch: int, total_steps: int):
        """Log comprehensive training configuration."""
        config_info = [
            f"Model Parameters: {self.model.get_num_params():,}",
            f"Epochs: {self.config.num_epochs}",
            f"Batches per epoch: {batches_per_epoch:,}",
            f"Total steps: {total_steps:,}",
            f"Effective batch size: {self.config.effective_batch_size}",
            f"Learning rate: {self.config.learning_rate:.2e}",
            f"Weight decay: {self.config.weight_decay}",
            f"Warmup ratio: {self.config.warmup_ratio}",
            f"Max grad norm: {self.config.max_grad_norm}",
            f"Precision: {self.config.precision}",
            f"Device: {self.device}"
        ]
        
        logging.info("Training Configuration:")
        for info in config_info:
            logging.info(f"  {info}")
    
    def _create_backup(self):
        """Create backup of current training state."""
        backup_dir = Path("backups") / self.config.experiment_name
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"backup_{timestamp}.pt"
        
        try:
            self.checkpoint_manager.save_checkpoint(
                self.model, self.optimizer, self.scheduler,
                self.global_step, self.current_epoch, self.metrics,
                str(backup_path)
            )
            logging.info(f"Backup created: {backup_path}")
        except Exception as e:
            logging.error(f"Failed to create backup: {e}")
    
    def _save_training_summary(self, total_time: float):
        """Save comprehensive training summary."""
        summary = {
            'experiment_name': self.config.experiment_name,
            'total_training_time_hours': total_time / 3600,
            'total_epochs': self.current_epoch,
            'total_steps': self.global_step,
            'final_metrics': {
                'best_eval_loss': self.best_eval_loss,
                'final_train_loss': self.metrics['train_losses'][-1] if self.metrics['train_losses'] else None,
                'avg_throughput': np.mean(self.metrics['throughput']) if self.metrics['throughput'] else 0
            },
            'model_config': asdict(self.config),
            'health_summary': self.health_monitor.get_summary()
        }
        
        summary_path = Path(f"experiments/{self.config.experiment_name}/training_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logging.info(f"Training summary saved: {summary_path}")
    
    def save_checkpoint(self, epoch_or_step: int, emergency: bool = False, final: bool = False):
        """Save checkpoint (delegated to checkpoint manager)."""
        suffix = "emergency" if emergency else ("final" if final else f"manual_{epoch_or_step}")
        return self.checkpoint_manager.save_checkpoint(
            self.model, self.optimizer, self.scheduler,
            self.global_step, self.current_epoch, self.metrics,
            suffix
        )
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load checkpoint and return current epoch."""
        return self.checkpoint_manager.load_checkpoint(
            checkpoint_path, self.model, self.optimizer, self.scheduler
        )
    
    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: Optional[int] = None) -> str:
        """Generate response with enhanced error handling."""
        self.model.eval()
        
        if max_new_tokens is None:
            max_new_tokens = self.config.max_new_tokens
        
        try:
            # Create conversation format
            conversation = {
                'messages': [{'role': 'user', 'content': prompt}]
            }
            
            # Encode input
            input_tokens = self.tokenizer.encode_conversation(conversation)
            
            # Add assistant start tokens
            input_tokens.extend([
                self.tokenizer.special_tokens["<|im_start|>"],
                self.tokenizer.special_tokens["<|assistant|>"]
            ])
            
            # Ensure reasonable context length
            if len(input_tokens) >= self.config.seq_length:
                input_tokens = input_tokens[-(self.config.seq_length//2):]
            
            input_ids = torch.tensor([input_tokens], device=self.device, dtype=torch.long)
            
            # Generation loop with safety checks
            generated_tokens = []
            
            for step in range(max_new_tokens):
                # Check sequence length
                if input_ids.size(1) >= self.config.seq_length:
                    input_ids = input_ids[:, -self.config.seq_length//2:]
                
                # Forward pass
                with autocast(device_type='cuda', enabled=self.use_amp, dtype=self.dtype):
                    logits = self.model(input_ids)
                
                # Get next token logits
                next_token_logits = logits[0, -1, :] / self.config.temperature
                
                # Apply top-k filtering
                if self.config.top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, self.config.top_k)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits.scatter_(0, top_k_indices, top_k_logits)
                
                # Apply top-p filtering
                if self.config.top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > self.config.top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                # Check for stop tokens
                if next_token.item() == self.tokenizer.special_tokens["<|im_end|>"]:
                    break
                
                generated_tokens.append(next_token.item())
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
                
                # Safety check for infinite loops
                if step > 0 and step % 100 == 0:
                    logging.debug(f"Generation step {step}/{max_new_tokens}")
            
            # Decode response
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            return response.strip()
            
        except Exception as e:
            logging.error(f"Generation failed: {e}")
            return "I apologize, but I encountered an error while generating a response."

# =============================================================================
# SUPPORTING CLASSES FOR PRODUCTION FEATURES
# =============================================================================

class TrainingHealthMonitor:
    """Monitor training health and detect anomalies."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.loss_history = []
        self.grad_norm_history = []
        self.consecutive_nan_count = 0
        self.consecutive_high_loss_count = 0
        
    def update(self, loss: float, grad_norm: float):
        """Update monitoring with new metrics."""
        # Check for NaN/Inf
        if math.isnan(loss) or math.isinf(loss):
            self.consecutive_nan_count += 1
        else:
            self.consecutive_nan_count = 0
            
        # Update history
        self.loss_history.append(loss)
        self.grad_norm_history.append(grad_norm)
        
        # Maintain window size
        if len(self.loss_history) > self.window_size:
            self.loss_history.pop(0)
            self.grad_norm_history.pop(0)
        
        # Check for divergence
        if len(self.loss_history) > 10:
            recent_avg = np.mean(self.loss_history[-10:])
            if len(self.loss_history) > 20:
                earlier_avg = np.mean(self.loss_history[-20:-10])
                if recent_avg > earlier_avg * 1.5:
                    self.consecutive_high_loss_count += 1
                else:
                    self.consecutive_high_loss_count = 0
    
    def get_status(self) -> str:
        """Get current health status."""
        if self.consecutive_nan_count > 3:
            return "CRITICAL"
        elif self.consecutive_high_loss_count > 10:
            return "WARNING"
        elif len(self.grad_norm_history) > 0 and self.grad_norm_history[-1] > 100:
            return "WARNING"
        else:
            return "HEALTHY"
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive health summary."""
        return {
            'total_nan_episodes': self.consecutive_nan_count,
            'avg_loss': np.mean(self.loss_history) if self.loss_history else 0,
            'avg_grad_norm': np.mean(self.grad_norm_history) if self.grad_norm_history else 0,
            'loss_std': np.std(self.loss_history) if len(self.loss_history) > 1 else 0,
            'status': self.get_status()
        }

class CheckpointManager:
    """Manage model checkpoints with versioning and cleanup."""
    
    def __init__(self, config: Config):
        self.config = config
        self.checkpoint_dir = Path("checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)
        
    def save_checkpoint(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                       scheduler: Optional[Any], global_step: int, epoch: int,
                       metrics: Dict, suffix: str) -> str:
        """Save checkpoint with comprehensive state."""
        checkpoint_path = self.checkpoint_dir / f"model_{suffix}.pt"
        
        # Unwrap compiled model if needed
        model_state = model.state_dict()
        if hasattr(model, '_orig_mod'):
            model_state = model._orig_mod.state_dict()
        
        checkpoint = {
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'config': asdict(self.config),
            'global_step': global_step,
            'epoch': epoch,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
            'pytorch_version': torch.__version__,
        }
        
        try:
            torch.save(checkpoint, checkpoint_path)
            logging.info(f"Checkpoint saved: {checkpoint_path}")
            
            # Cleanup old checkpoints
            self._cleanup_old_checkpoints()
            
        except Exception as e:
            logging.error(f"Failed to save checkpoint: {e}")
            raise
        
        return str(checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str, model: nn.Module,
                       optimizer: torch.optim.Optimizer,
                       scheduler: Optional[Any]) -> int:
        """Load checkpoint with validation."""
        logging.info(f"Loading checkpoint: {checkpoint_path}")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Load model state
            model_to_load = model
            if hasattr(model, '_orig_mod'):
                model_to_load = model._orig_mod
            
            model_to_load.load_state_dict(checkpoint['model_state_dict'])
            
            # Load optimizer state
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load scheduler state if available
            if checkpoint.get('scheduler_state_dict') and scheduler:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            global_step = checkpoint.get('global_step', 0)
            epoch = checkpoint.get('epoch', 0)
            
            logging.info(f"Resumed from epoch {epoch}, step {global_step}")
            return epoch
            
        except Exception as e:
            logging.error(f"Failed to load checkpoint: {e}")
            raise
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints to save space."""
        if self.config.save_total_limit <= 0:
            return
        
        # Get all checkpoint files except special ones
        checkpoint_files = []
        for path in self.checkpoint_dir.glob("model_*.pt"):
            if not any(special in path.name for special in ['best', 'final', 'emergency']):
                checkpoint_files.append(path)
        
        # Sort by modification time
        checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Remove old checkpoints
        for old_checkpoint in checkpoint_files[self.config.save_total_limit:]:
            try:
                old_checkpoint.unlink()
                logging.debug(f"Removed old checkpoint: {old_checkpoint}")
            except Exception as e:
                logging.warning(f"Failed to remove old checkpoint {old_checkpoint}: {e}")

# =============================================================================
# ENHANCED UTILITY FUNCTIONS
# =============================================================================

def validate_environment():
    """Validate the training environment."""
    issues = []
    
    # Check PyTorch version
    torch_version = torch.__version__
    if torch_version < "2.0.0":
        issues.append(f"PyTorch version {torch_version} is old, recommend >= 2.0.0")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        issues.append("CUDA not available, training will be slow on CPU")
    else:
        # Check GPU memory
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        if gpu_memory < 8:
            issues.append(f"GPU memory {gpu_memory:.1f}GB may be insufficient")
    
    # Check disk space
    disk_usage = shutil.disk_usage(".")
    free_gb = disk_usage.free / 1e9
    if free_gb < 10:
        issues.append(f"Low disk space: {free_gb:.1f}GB free")
    
    # Check system memory
    memory = psutil.virtual_memory()
    if memory.available / 1e9 < 4:
        issues.append(f"Low system memory: {memory.available / 1e9:.1f}GB available")
    
    return issues

def process_oasst_data(input_path: str, output_path: str, max_conversations: int = None) -> int:
    """Enhanced OASST data processing with validation."""
    conversations = []
    stats = {'processed': 0, 'valid': 0, 'errors': 0}
    
    logging.info(f"Processing OASST data: {input_path} -> {output_path}")
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            if max_conversations and stats['valid'] >= max_conversations:
                break
            
            try:
                data = json.loads(line.strip())
                stats['processed'] += 1
                
                # Extract and validate messages
                if 'messages' in data and len(data['messages']) >= 2:
                    messages = []
                    for msg in data['messages']:
                        role = msg.get('role', '').lower()
                        content = msg.get('content', '').strip()
                        
                        if not content:
                            continue
                        
                        # Normalize role names
                        if role == 'prompter':
                            role = 'user'
                        elif role not in ['user', 'assistant', 'system']:
                            role = 'user'
                        
                        messages.append({'role': role, 'content': content})
                    
                    if len(messages) >= 2:
                        conversation = {
                            'conversation_id': data.get('conversation_id', f'conv_{line_no}'),
                            'messages': messages,
                            'metadata': {
                                'source': 'oasst',
                                'processed_at': datetime.now().isoformat()
                            }
                        }
                        conversations.append(conversation)
                        stats['valid'] += 1
                
            except Exception as e:
                stats['errors'] += 1
                if line_no <= 10:
                    logging.warning(f"Error processing line {line_no}: {e}")
            
            # Progress update
            if line_no % 10000 == 0:
                logging.info(f"Processed {line_no:,} lines, {stats['valid']:,} valid conversations")
    
    # Write processed data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for conv in conversations:
            f.write(json.dumps(conv) + '\n')
    
    logging.info(f"Processing complete: {stats['valid']:,} valid conversations from {stats['processed']:,} total")
    logging.info(f"Output written to: {output_path}")
    
    return stats['valid']

def validate_data_comprehensive(data_path: str, tokenizer: ConversationTokenizer, 
                               max_check: int = 5000) -> Dict[str, Any]:
    """Comprehensive data validation with detailed statistics."""
    stats = {
        'file_info': {},
        'conversation_stats': {},
        'token_stats': {},
        'quality_metrics': {},
        'errors': []
    }
    
    # File information
    try:
        file_path = Path(data_path)
        if not file_path.exists():
            stats['errors'].append(f"File not found: {data_path}")
            return stats
        
        stats['file_info'] = {
            'path': str(file_path),
            'size_mb': file_path.stat().st_size / 1e6,
            'modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
        }
    except Exception as e:
        stats['errors'].append(f"File access error: {e}")
        return stats
    
    # Initialize counters
    conversation_lengths = []
    token_lengths = []
    role_counts = {'user': 0, 'assistant': 0, 'system': 0, 'other': 0}
    quality_issues = []
    
    total_lines = 0
    valid_conversations = 0
    
    logging.info(f"Validating data: {data_path}")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            if line_no > max_check:
                break
            
            total_lines += 1
            
            try:
                conversation = json.loads(line.strip())
                
                # Validate structure
                if 'messages' not in conversation:
                    quality_issues.append(f"Line {line_no}: Missing 'messages' field")
                    continue
                
                messages = conversation['messages']
                if not isinstance(messages, list) or len(messages) == 0:
                    quality_issues.append(f"Line {line_no}: Empty or invalid messages")
                    continue
                
                conversation_lengths.append(len(messages))
                
                # Analyze messages
                has_user = False
                has_assistant = False
                total_content_length = 0
                
                for msg_idx, msg in enumerate(messages):
                    if not isinstance(msg, dict):
                        quality_issues.append(f"Line {line_no}, Message {msg_idx}: Invalid message format")
                        continue
                    
                    role = msg.get('role', '').lower()
                    content = msg.get('content', '')
                    
                    if not content or not content.strip():
                        quality_issues.append(f"Line {line_no}, Message {msg_idx}: Empty content")
                        continue
                    
                    total_content_length += len(content)
                    
                    # Count roles
                    if role in ['user', 'prompter']:
                        role_counts['user'] += 1
                        has_user = True
                    elif role == 'assistant':
                        role_counts['assistant'] += 1
                        has_assistant = True
                    elif role == 'system':
                        role_counts['system'] += 1
                    else:
                        role_counts['other'] += 1
                
                # Check conversation quality
                if not (has_user and has_assistant):
                    quality_issues.append(f"Line {line_no}: Missing user or assistant messages")
                    continue
                
                # Tokenize and analyze
                try:
                    tokens = tokenizer.encode_conversation(conversation)
                    if tokens:
                        token_lengths.append(len(tokens))
                        valid_conversations += 1
                    else:
                        quality_issues.append(f"Line {line_no}: Tokenization failed")
                except Exception as e:
                    quality_issues.append(f"Line {line_no}: Tokenization error: {e}")
                
            except json.JSONDecodeError as e:
                quality_issues.append(f"Line {line_no}: JSON decode error: {e}")
            except Exception as e:
                quality_issues.append(f"Line {line_no}: Processing error: {e}")
            
            # Progress update
            if line_no % 1000 == 0:
                logging.info(f"Validated {line_no:,} lines...")
    
    # Compute statistics
    stats['conversation_stats'] = {
        'total_lines': total_lines,
        'valid_conversations': valid_conversations,
        'invalid_conversations': total_lines - valid_conversations,
        'avg_messages_per_conversation': np.mean(conversation_lengths) if conversation_lengths else 0,
        'max_messages': max(conversation_lengths) if conversation_lengths else 0,
        'min_messages': min(conversation_lengths) if conversation_lengths else 0,
        'role_distribution': role_counts
    }
    
    stats['token_stats'] = {
        'avg_tokens': np.mean(token_lengths) if token_lengths else 0,
        'median_tokens': np.median(token_lengths) if token_lengths else 0,
        'max_tokens': max(token_lengths) if token_lengths else 0,
        'min_tokens': min(token_lengths) if token_lengths else 0,
        'std_tokens': np.std(token_lengths) if token_lengths else 0
    }
    
    stats['quality_metrics'] = {
        'success_rate': valid_conversations / total_lines if total_lines > 0 else 0,
        'error_rate': len(quality_issues) / total_lines if total_lines > 0 else 0,
        'total_quality_issues': len(quality_issues),
        'quality_issues_sample': quality_issues[:20]  # First 20 issues
    }
    
    return stats

def estimate_training_time(config: Config, dataset_size: int) -> Dict[str, float]:
    """Estimate training time and resource requirements."""
    # Model parameter estimation
    params = estimate_parameters(config)
    
    # Rough estimates based on empirical data
    tokens_per_sample = config.seq_length
    total_tokens = dataset_size * tokens_per_sample * config.num_epochs
    
    # GPU estimates (rough approximations)
    if torch.cuda.is_available():
        gpu_props = torch.cuda.get_device_properties(0)
        gpu_memory_gb = gpu_props.total_memory / 1e9
        
        # Memory requirements (rough estimate)
        model_memory = params * 4 / 1e9  # 4 bytes per parameter (fp32)
        optimizer_memory = params * 8 / 1e9  # Adam needs ~8 bytes per parameter
        activation_memory = config.batch_size * config.seq_length * config.hidden_size * 4 / 1e9
        total_memory_needed = model_memory + optimizer_memory + activation_memory
        
        # Throughput estimates (very rough)
        if "A100" in gpu_props.name:
            tokens_per_sec = 50000
        elif "V100" in gpu_props.name:
            tokens_per_sec = 25000
        elif "T4" in gpu_props.name:
            tokens_per_sec = 10000
        else:
            tokens_per_sec = 5000  # Conservative estimate
        
        # Adjust for precision
        if config.precision in ["fp16", "bf16"]:
            tokens_per_sec *= 1.5
            total_memory_needed *= 0.6
        
        estimated_time_hours = total_tokens / tokens_per_sec / 3600
        memory_utilization = min(total_memory_needed / gpu_memory_gb, 1.0)
        
    else:
        # CPU estimates (much slower)
        tokens_per_sec = 100
        estimated_time_hours = total_tokens / tokens_per_sec / 3600
        memory_utilization = 0.5  # Assume reasonable CPU memory
    
    return {
        'estimated_hours': estimated_time_hours,
        'estimated_days': estimated_time_hours / 24,
        'total_tokens': total_tokens,
        'tokens_per_second': tokens_per_sec,
        'memory_utilization': memory_utilization,
        'memory_warning': memory_utilization > 0.9
    }

def create_data_summary_report(data_paths: List[str], tokenizer: ConversationTokenizer, 
                              output_path: str = "data_summary_report.html"):
    """Create comprehensive HTML report of dataset analysis."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Dataset Analysis Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
            .metric { display: inline-block; margin: 10px; padding: 10px; background: #f5f5f5; border-radius: 3px; }
            .error { color: red; }
            .warning { color: orange; }
            .success { color: green; }
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
        </style>
    </head>
    <body>
        <h1>Dataset Analysis Report</h1>
        <p>Generated on: {timestamp}</p>
    """.format(timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    for data_path in data_paths:
        logging.info(f"Analyzing {data_path}...")
        stats = validate_data_comprehensive(data_path, tokenizer)
        
        html_content += f"""
        <div class="section">
            <h2>Dataset: {os.path.basename(data_path)}</h2>
            
            <h3>File Information</h3>
            <div class="metric">Size: {stats['file_info'].get('size_mb', 0):.1f} MB</div>
            <div class="metric">Modified: {stats['file_info'].get('modified', 'Unknown')}</div>
            
            <h3>Conversation Statistics</h3>
            <div class="metric">Total Lines: {stats['conversation_stats'].get('total_lines', 0):,}</div>
            <div class="metric">Valid Conversations: {stats['conversation_stats'].get('valid_conversations', 0):,}</div>
            <div class="metric">Success Rate: {stats['quality_metrics'].get('success_rate', 0):.2%}</div>
            
            <h3>Token Statistics</h3>
            <div class="metric">Avg Tokens: {stats['token_stats'].get('avg_tokens', 0):.1f}</div>
            <div class="metric">Max Tokens: {stats['token_stats'].get('max_tokens', 0):,}</div>
            <div class="metric">Min Tokens: {stats['token_stats'].get('min_tokens', 0):,}</div>
            
            <h3>Role Distribution</h3>
            <table>
                <tr><th>Role</th><th>Count</th></tr>
        """
        
        role_dist = stats['conversation_stats'].get('role_distribution', {})
        for role, count in role_dist.items():
            html_content += f"<tr><td>{role}</td><td>{count:,}</td></tr>"
        
        html_content += """
            </table>
            
            <h3>Quality Issues (Sample)</h3>
            <ul>
        """
        
        issues = stats['quality_metrics'].get('quality_issues_sample', [])
        for issue in issues[:10]:  # Show first 10 issues
            html_content += f"<li class='error'>{issue}</li>"
        
        html_content += """
            </ul>
        </div>
        """
    
    html_content += """
    </body>
    </html>
    """
    
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    logging.info(f"Data summary report saved: {output_path}")

# =============================================================================
# ENHANCED CONFIGURATION PRESETS
# =============================================================================

class ConfigPresets:
    """Enhanced configuration presets for different scenarios."""
    
    @staticmethod
    def debug() -> Config:
        """Minimal config for debugging and testing."""
        return Config(
            # Tiny model for fast iteration
            vocab_size=1024,
            hidden_size=256,
            num_layers=4,
            num_heads=4,
            num_kv_heads=2,
            seq_length=512,
            intermediate_size=512,
            
            # Fast training settings
            batch_size=2,
            gradient_accumulation_steps=2,
            num_epochs=1,
            learning_rate=1e-3,
            weight_decay=0.01,
            eval_every_n_batches=50,
            save_every_n_batches=100,
            precision="fp32",
            compile=False,
            num_workers=0,
            
            # Monitoring and stability
            experiment_name="debug_run",
            log_level="DEBUG",
            health_check_interval=10,
            save_total_limit=3,
            early_stopping_patience=None,
            max_retries=1
        )
    
    @staticmethod
    def small() -> Config:
        """Small model for limited resources."""
        return Config(
            # Small but capable model
            hidden_size=512,
            num_layers=8,
            num_heads=8,
            num_kv_heads=4,
            seq_length=1024,
            intermediate_size=1536,
            
            # Balanced training settings
            batch_size=4,
            gradient_accumulation_steps=4,
            num_epochs=3,
            learning_rate=5e-4,
            weight_decay=0.01,
            eval_every_n_batches=500,
            save_every_n_batches=1000,
            precision="fp16",
            compile=True,
            num_workers=2,
            
            # Production settings
            experiment_name="small_model",
            log_level="INFO",
            health_check_interval=100,
            save_total_limit=5,
            early_stopping_patience=5,
            backup_every_n_hours=12
        )
    
    @staticmethod
    def medium() -> Config:
        """Medium model for serious training."""
        return Config(
            # Medium-scale model
            hidden_size=1024,
            num_layers=16,
            num_heads=16,
            num_kv_heads=8,
            seq_length=2048,
            intermediate_size=2816,
            
            # Serious training configuration
            batch_size=4,
            gradient_accumulation_steps=8,
            num_epochs=5,
            learning_rate=3e-4,
            weight_decay=0.01,
            eval_every_n_batches=1000,
            save_every_n_batches=2000,
            precision="bf16",
            compile=True,
            num_workers=4,
            
            # Production monitoring
            experiment_name="medium_model",
            log_level="INFO",
            health_check_interval=100,
            save_total_limit=10,
            early_stopping_patience=10,
            backup_every_n_hours=6,
            gradient_checkpointing=True
        )
    
    @staticmethod
    def large() -> Config:
        """Large model for high-end training."""
        return Config(
            # Large-scale model
            hidden_size=2048,
            num_layers=24,
            num_heads=32,
            num_kv_heads=8,
            seq_length=4096,
            intermediate_size=5504,
            
            # Large-scale training
            batch_size=2,
            gradient_accumulation_steps=16,
            num_epochs=3,
            learning_rate=2e-4,
            weight_decay=0.01,
            eval_every_n_batches=2000,
            save_every_n_batches=5000,
            precision="bf16",
            compile=True,
            num_workers=8,
            
            # Enterprise monitoring
            experiment_name="large_model",
            log_level="INFO",
            health_check_interval=200,
            save_total_limit=15,
            early_stopping_patience=15,
            backup_every_n_hours=4,
            gradient_checkpointing=True,
            lr_scheduler="cosine",
            warmup_ratio=0.05
        )

# =============================================================================
# MAIN FUNCTION WITH COMPREHENSIVE CLI
# =============================================================================

def setup_logging_basic():
    """Setup basic logging before full system initialization."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def main():
    """Enhanced main function with comprehensive CLI and error handling."""
    import argparse
    
    # Setup basic logging first
    setup_logging_basic()
    
    parser = argparse.ArgumentParser(
        description='Production-Ready Conversational Transformer Training System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick debug run
  python train.py --config debug --test-generation
  
  # Production training
  python train.py --config medium --train-data data/train.jsonl --eval-data data/eval.jsonl
  
  # Resume training
  python train.py --config medium --resume checkpoints/model_epoch_005.pt
  
  # Data processing and validation
  python train.py --process-oasst raw_data.jsonl processed_data.jsonl
  python train.py --validate-data processed_data.jsonl --create-report
  
  # Custom configuration
  python train.py --config small --epochs 10 --lr 1e-4 --batch-size 8
        """
    )
    
    # Configuration options
    parser.add_argument('--config', choices=['debug', 'small', 'medium', 'large'], 
                       default='debug', help='Configuration preset')
    parser.add_argument('--config-file', type=str, help='Load config from YAML file')
    
    # Data options
    parser.add_argument('--train-data', type=str, default='data/train.jsonl',
                       help='Training data path')
    parser.add_argument('--eval-data', type=str, default='data/eval.jsonl',
                       help='Evaluation data path')
    
    # Training overrides
    parser.add_argument('--epochs', type=int, help='Override number of epochs')
    parser.add_argument('--lr', type=float, help='Override learning rate')
    parser.add_argument('--batch-size', type=int, help='Override batch size')
    parser.add_argument('--grad-accum', type=int, help='Override gradient accumulation steps')
    parser.add_argument('--precision', choices=['fp16', 'bf16', 'fp32'], help='Override precision')
    
    # Experiment options
    parser.add_argument('--experiment-name', type=str, help='Experiment name')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Testing and validation
    parser.add_argument('--test-generation', action='store_true', help='Test generation after training')
    parser.add_argument('--validate-data', type=str, help='Validate data file format')
    parser.add_argument('--create-report', action='store_true', help='Create comprehensive data report')
    
    # Data processing
    parser.add_argument('--process-oasst', nargs=2, metavar=('INPUT', 'OUTPUT'),
                       help='Process OASST data: input_file output_file')
    parser.add_argument('--max-conversations', type=int, help='Limit conversations processed')
    
    # System options
    parser.add_argument('--check-environment', action='store_true', help='Check training environment')
    parser.add_argument('--estimate-time', action='store_true', help='Estimate training time')
    parser.add_argument('--dry-run', action='store_true', help='Setup everything but don\'t train')
    
    args = parser.parse_args()
    
    # Environment validation
    if args.check_environment:
        logging.info("Checking training environment...")
        issues = validate_environment()
        if issues:
            logging.warning("Environment issues found:")
            for issue in issues:
                logging.warning(f"  - {issue}")
        else:
            logging.info("Environment looks good!")
        
        if not args.dry_run:
            return 0
    
    # Data processing
    if args.process_oasst:
        input_file, output_file = args.process_oasst
        try:
            count = process_oasst_data(input_file, output_file, args.max_conversations)
            logging.info(f"Successfully processed {count} conversations")
            return 0
        except Exception as e:
            logging.error(f"Data processing failed: {e}")
            return 1
    
    # Data validation
    if args.validate_data:
        try:
            tokenizer = ConversationTokenizer()
            stats = validate_data_comprehensive(args.validate_data, tokenizer)
            
            logging.info("Data Validation Results:")
            logging.info(f"  Valid conversations: {stats['conversation_stats']['valid_conversations']:,}")
            logging.info(f"  Success rate: {stats['quality_metrics']['success_rate']:.2%}")
            logging.info(f"  Average tokens: {stats['token_stats']['avg_tokens']:.1f}")
            
            if args.create_report:
                create_data_summary_report([args.validate_data], tokenizer)
            
            return 0
        except Exception as e:
            logging.error(f"Data validation failed: {e}")
            return 1
    
    # Load configuration
    try:
        if args.config_file:
            config = Config.load(args.config_file)
        else:
            config_map = {
                'debug': ConfigPresets.debug,
                'small': ConfigPresets.small,
                'medium': ConfigPresets.medium,
                'large': ConfigPresets.large,
            }
            config = config_map[args.config]()
        
        # Apply CLI overrides
        if args.epochs is not None:
            config.num_epochs = args.epochs
        if args.lr is not None:
            config.learning_rate = args.lr
        if args.batch_size is not None:
            config.batch_size = args.batch_size
        if args.grad_accum is not None:
            config.gradient_accumulation_steps = args.grad_accum
        if args.precision is not None:
            config.precision = args.precision
        if args.experiment_name is not None:
            config.experiment_name = args.experiment_name
        
        config.train_data_path = args.train_data
        config.eval_data_path = args.eval_data
        config.seed = args.seed
        
        # Re-validate after overrides
        config.validate()
        
    except Exception as e:
        logging.error(f"Configuration error: {e}")
        return 1
    
    # Training time estimation
    if args.estimate_time:
        try:
            # Estimate dataset size
            if Path(config.train_data_path).exists():
                with open(config.train_data_path, 'r') as f:
                    dataset_size = sum(1 for _ in f)
            else:
                dataset_size = 10000  # Default estimate
            
            estimates = estimate_training_time(config, dataset_size)
            
            logging.info("Training Time Estimates:")
            logging.info(f"  Dataset size: {dataset_size:,} conversations")
            logging.info(f"  Estimated time: {estimates['estimated_hours']:.1f} hours ({estimates['estimated_days']:.1f} days)")
            logging.info(f"  Total tokens: {estimates['total_tokens']:,}")
            logging.info(f"  Throughput: {estimates['tokens_per_second']:,} tokens/sec")
            logging.info(f"  Memory utilization: {estimates['memory_utilization']:.1%}")
            
            if estimates['memory_warning']:
                logging.warning("  ⚠️  High memory utilization expected - consider reducing batch size")
            
            if not args.dry_run:
                return 0
        except Exception as e:
            logging.error(f"Time estimation failed: {e}")
            return 1
    
    # Dry run
    if args.dry_run:
        logging.info("Dry run completed successfully!")
        return 0
    
    # Main training
    logging.info("="*80)
    logging.info("PRODUCTION CONVERSATIONAL TRANSFORMER TRAINING")
    logging.info("="*80)
    
    try:
        # Initialize training orchestrator
        orchestrator = TrainingOrchestrator(config)
        
        # Log configuration
        logging.info(f"Configuration: {args.config}")
        logging.info(f"Model parameters: ~{estimate_parameters(config):,}")
        logging.info(f"Experiment: {config.experiment_name}")
        
        # Run training
        orchestrator.run_training()
        
        # Test generation if requested
        if args.test_generation and orchestrator.trainer:
            logging.info("\n" + "="*60)
            logging.info("TESTING GENERATION")
            logging.info("="*60)
            
            test_prompts = [
                "Hello, how are you today?",
                "What is machine learning?",
                "Write a simple Python function to calculate factorial.",
                "Explain the concept of recursion in programming.",
                "What are the benefits of using transformers in NLP?"
            ]
            
            for i, prompt in enumerate(test_prompts, 1):
                logging.info(f"\nTest {i}/5:")
                logging.info(f"User: {prompt}")
                try:
                    response = orchestrator.trainer.generate(prompt)
                    logging.info(f"Assistant: {response}")
                except Exception as e:
                    logging.error(f"Generation failed: {e}")
                logging.info("-" * 50)
        
        logging.info("\n🎉 Training completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
        return 1
    except Exception as e:
        logging.error(f"Training failed: {e}")
        logging.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    exit(main())