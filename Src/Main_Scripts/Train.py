# Production-Ready Conversational Transformer Training System
# Hardcoded Version - Just run: python train.py

import json
import logging
import os
import time
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Iterator, Any
from collections import defaultdict
import tempfile
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.checkpoint import checkpoint
import numpy as np

# Distributed training
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

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

from torch.cuda.amp import autocast, GradScaler

# Configure for performance
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# =============================================================================
# HARDCODED CONFIGURATION
# =============================================================================

@dataclass
class TrainingConfig:
    """Hardcoded training configuration."""
    # Model architecture - Small but functional
    vocab_size: int = 8192
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    num_kv_heads: int = 4
    seq_length: int = 1024
    intermediate_size: int = 2048
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    dropout: float = 0.0
    use_gradient_checkpointing: bool = True
    
    # Training parameters - Conservative settings
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_steps: int = 10000
    warmup_ratio: float = 0.1
    eval_steps: int = 500
    save_steps: int = 1000
    max_grad_norm: float = 1.0
    precision: str = "bf16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "fp16"
    compile_model: bool = True
    
    # Data parameters - Hardcoded paths
    train_data_path: str = "data/train_conversations.jsonl"
    eval_data_path: str = "data/eval_conversations.jsonl"
    num_workers: int = 2
    prefetch_factor: int = 2
    pin_memory: bool = True
    dataloader_persistent_workers: bool = True
    
    # Loss configuration
    assistant_loss_weight: float = 2.0
    ignore_user_tokens_in_loss: bool = True
    label_smoothing: float = 0.0
    
    # Generation parameters
    max_new_tokens: int = 256
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    
    # Distributed training - Auto-detect
    distributed: bool = False
    local_rank: int = -1
    world_size: int = 1
    
    # Monitoring - Disabled by default
    use_wandb: bool = False
    wandb_project: str = "conversational-transformer"
    wandb_run_name: Optional[str] = None
    log_interval: int = 10
    
    # Checkpointing - Sensible defaults
    checkpoint_dir: str = "checkpoints"
    keep_last_n_checkpoints: int = 3
    save_optimizer_state: bool = True
    
    # Performance
    torch_compile_mode: str = "reduce-overhead"
    use_fused_optimizer: bool = True
    
    def __post_init__(self):
        """Validate and adjust configuration."""
        assert self.hidden_size % self.num_heads == 0, "hidden_size must be divisible by num_heads"
        assert self.num_heads % self.num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
        
        # Ensure vocab size is efficient
        if self.vocab_size % 64 != 0:
            self.vocab_size = ((self.vocab_size + 63) // 64) * 64
        
        self.warmup_steps = int(self.max_steps * self.warmup_ratio)
        self.effective_batch_size = self.batch_size * self.gradient_accumulation_steps
        
        # Auto-detect distributed
        if 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1:
            self.distributed = True
            self.world_size = int(os.environ['WORLD_SIZE'])
            self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        # Adjust for distributed training
        if self.distributed:
            self.effective_batch_size *= self.world_size
        
        # Create checkpoint directory
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Auto-adjust precision for compatibility
        if not torch.cuda.is_available():
            self.precision = "fp32"
        elif self.precision == "bf16" and not torch.cuda.is_bf16_supported():
            self.precision = "fp16"

# =============================================================================
# TOKENIZATION SYSTEM
# =============================================================================

class ConversationTokenizer:
    """Production tokenizer with proper special token handling."""
    
    def __init__(self, vocab_size: int = 8192, pad_to_multiple: int = 64):
        """Initialize with proper vocabulary management."""
        # Base vocabulary
        self.base_vocab_size = vocab_size
        
        # Special tokens with reserved IDs
        self.special_tokens = {
            '<pad>': 0,
            '<unk>': 1, 
            '<bos>': 2,
            '<eos>': 3,
            '<user>': 4,
            '<assistant>': 5,
            '<system>': 6,
            '<turn_start>': 7,
            '<turn_end>': 8,
        }
        
        self.id_to_token = {v: k for k, v in self.special_tokens.items()}
        
        # Reserve space for special tokens
        self.vocab_start = len(self.special_tokens)
        self.vocab_size = max(self.base_vocab_size, self.vocab_start + 1000)
        
        # Pad to efficient multiple
        if self.vocab_size % pad_to_multiple != 0:
            self.vocab_size = ((self.vocab_size + pad_to_multiple - 1) // pad_to_multiple) * pad_to_multiple
        
        # Create simple word-level vocabulary
        self._build_simple_vocab()
        
        logging.info(f"Tokenizer initialized: vocab_size={self.vocab_size}, special_tokens={len(self.special_tokens)}")
    
    def _build_simple_vocab(self):
        """Build a simple vocabulary."""
        # Common words and programming terms
        common_words = [
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i', 'it', 'for', 'not', 'on', 'with',
            'he', 'as', 'you', 'do', 'at', 'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her',
            'she', 'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what', 'so', 'up',
            'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me', 'when', 'make', 'can', 'like', 'time',
            'no', 'just', 'him', 'know', 'take', 'people', 'into', 'year', 'your', 'good', 'some', 'could',
            'them', 'see', 'other', 'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over', 'think',
            'also', 'back', 'after', 'use', 'two', 'how', 'our', 'work', 'first', 'well', 'way', 'even',
            'new', 'want', 'because', 'any', 'these', 'give', 'day', 'most', 'us', 'is', 'was', 'are',
            'been', 'has', 'had', 'were', 'said', 'each', 'which', 'their', 'what', 'where', 'when',
            'python', 'programming', 'code', 'function', 'class', 'import', 'return', 'if', 'else', 'for',
            'while', 'try', 'except', 'with', 'as', 'def', 'lambda', 'list', 'dict', 'string', 'int', 'float',
            'machine', 'learning', 'data', 'model', 'train', 'test', 'algorithm', 'neural', 'network', 'deep'
        ]
        
        # Add common words to vocabulary
        for i, word in enumerate(common_words):
            if self.vocab_start + i < self.vocab_size:
                self.id_to_token[self.vocab_start + i] = word
        
        # Fill remaining with dummy tokens
        for i in range(len(common_words), self.vocab_size - self.vocab_start):
            self.id_to_token[self.vocab_start + i] = f'<token_{i}>'
        
        # Create reverse mapping
        self.token_to_id = {v: k for k, v in self.id_to_token.items()}
    
    def _simple_tokenize(self, text: str) -> List[str]:
        """Simple word-level tokenization."""
        text = text.lower().strip()
        words = []
        current_word = ""
        
        for char in text:
            if char.isalnum():
                current_word += char
            else:
                if current_word:
                    words.append(current_word)
                    current_word = ""
                if not char.isspace():
                    words.append(char)
        
        if current_word:
            words.append(current_word)
        
        return words
    
    def encode_text(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        words = self._simple_tokenize(text)
        token_ids = []
        
        for word in words:
            if word in self.token_to_id:
                token_ids.append(self.token_to_id[word])
            else:
                token_ids.append(self.special_tokens['<unk>'])
        
        return token_ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text."""
        words = []
        for token_id in token_ids:
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                if skip_special_tokens and token.startswith('<') and token.endswith('>'):
                    continue
                words.append(token)
        
        return ' '.join(words)
    
    def encode_conversation(self, messages: List[Dict[str, str]]) -> List[int]:
        """Encode conversation with proper special token handling."""
        tokens = [self.special_tokens['<bos>']]
        
        for message in messages:
            role = message.get('role', '').lower()
            content = message.get('content', '').strip()
            
            if not content:
                continue
            
            # Add turn start
            tokens.append(self.special_tokens['<turn_start>'])
            
            # Add role token
            if role in ['user', 'human']:
                tokens.append(self.special_tokens['<user>'])
            elif role == 'assistant':
                tokens.append(self.special_tokens['<assistant>'])
            else:
                tokens.append(self.special_tokens['<system>'])
            
            # Add content
            content_tokens = self.encode_text(content)
            tokens.extend(content_tokens)
            
            # Add turn end
            tokens.append(self.special_tokens['<turn_end>'])
        
        tokens.append(self.special_tokens['<eos>'])
        return tokens
    
    def is_special_token(self, token_id: int) -> bool:
        """Check if token is special."""
        return token_id in self.special_tokens.values()
    
    def get_special_token_id(self, token: str) -> int:
        """Get ID for special token."""
        return self.special_tokens.get(token, self.special_tokens['<unk>'])

# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

class RMSNorm(nn.Module):
    """RMS Normalization with optional bias."""
    
    def __init__(self, dim: int, eps: float = 1e-6, bias: bool = False):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        if self.bias is not None:
            output = output + self.bias
        return output

class RotaryEmbedding(nn.Module):
    """Optimized Rotary Position Embedding."""
    
    def __init__(self, dim: int, max_seq_len: int = 8192, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
    
    def _update_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """Update cached cos/sin values."""
        if seq_len > self._seq_len_cached or self._cos_cached is None or self._cos_cached.device != device:
            self._seq_len_cached = max(seq_len, self._seq_len_cached * 2)
            t = torch.arange(self._seq_len_cached, device=device, dtype=dtype)
            freqs = torch.outer(t, self.inv_freq.to(device=device, dtype=dtype))
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = emb.cos()
            self._sin_cached = emb.sin()
    
    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype = torch.float32) -> Tuple[torch.Tensor, torch.Tensor]:
        self._update_cache(seq_len, device, dtype)
        return self._cos_cached[:seq_len], self._sin_cached[:seq_len]

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, 
                        cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding efficiently."""
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class GroupedQueryAttention(nn.Module):
    """Optimized Grouped Query Attention."""
    
    def __init__(self, config: TrainingConfig):
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
        
        self.rope = RotaryEmbedding(self.head_dim, config.seq_length, config.rope_theta)
        self.dropout = nn.Dropout(config.dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        """Proper weight initialization."""
        for proj in [self.q_proj, self.k_proj, self.v_proj]:
            nn.init.xavier_uniform_(proj.weight, gain=1 / math.sqrt(2))
        
        nn.init.xavier_uniform_(self.o_proj.weight, gain=1 / math.sqrt(2 * self.config.num_layers))
    
    def forward(self, 
                x: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, D = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)  
        v = self.v_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        cos, sin = self.rope(L, x.device, x.dtype)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Expand K, V for GQA if needed
        if self.num_queries_per_kv > 1:
            k = k[:, :, None, :, :].expand(B, self.num_kv_heads, self.num_queries_per_kv, L, self.head_dim)
            v = v[:, :, None, :, :].expand(B, self.num_kv_heads, self.num_queries_per_kv, L, self.head_dim)
            k = k.reshape(B, self.num_heads, L, self.head_dim)
            v = v.reshape(B, self.num_heads, L, self.head_dim)
        
        # Attention computation
        if HAS_FLASH_ATTN and x.is_cuda and attention_mask is None and x.dtype in [torch.float16, torch.bfloat16]:
            # Use Flash Attention
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            out = flash_attn_func(q, k, v, causal=True, dropout_p=self.dropout.p if self.training else 0.0)
            out = out.reshape(B, L, self.hidden_size)
        else:
            # Standard attention
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            
            # Apply causal mask
            if attention_mask is None:
                mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1)
                scores.masked_fill_(mask, float('-inf'))
            else:
                scores = scores + attention_mask
            
            attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
            attn_weights = self.dropout(attn_weights)
            
            out = torch.matmul(attn_weights, v)
            out = out.transpose(1, 2).reshape(B, L, self.hidden_size)
        
        return self.o_proj(out)

class SwiGLU(nn.Module):
    """SwiGLU activation function."""
    
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self._init_weights()
    
    def _init_weights(self):
        """Proper weight initialization."""
        nn.init.xavier_uniform_(self.gate_proj.weight)
        nn.init.xavier_uniform_(self.up_proj.weight)
        nn.init.xavier_uniform_(self.down_proj.weight, gain=1 / math.sqrt(2 * self.config.num_layers))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        hidden = self.dropout(gate * up)
        return self.down_proj(hidden)

class TransformerBlock(nn.Module):
    """Transformer block with gradient checkpointing support."""
    
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        self.input_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = GroupedQueryAttention(config)
        self.post_attn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = SwiGLU(config)
    
    def forward(self, 
                x: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm architecture with residual connections
        residual = x
        x = self.input_norm(x)
        x = self.self_attn(x, attention_mask, position_ids)
        x = residual + x
        
        residual = x
        x = self.post_attn_norm(x)
        x = self.mlp(x)
        x = residual + x
        
        return x

class TransformerModel(nn.Module):
    """Enhanced transformer model."""
    
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Weight tying
        self.lm_head.weight = self.embed_tokens.weight
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        self._init_weights()
        
        n_params = sum(p.numel() for p in self.parameters())
        logging.info(f"Model initialized with {n_params:,} parameters")
    
    def _init_weights(self):
        """Proper model initialization."""
        nn.init.normal_(self.embed_tokens.weight, std=0.02)
        
        for layer in self.layers:
            layer.self_attn.o_proj.weight.data *= (2 * self.config.num_layers) ** -0.5
            layer.mlp.down_proj.weight.data *= (2 * self.config.num_layers) ** -0.5
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                use_cache: bool = False) -> torch.Tensor:
        
        x = self.embed_tokens(input_ids)
        x = self.dropout(x)
        
        for layer in self.layers:
            if self.training and self.config.use_gradient_checkpointing:
                x = checkpoint(layer, x, attention_mask, position_ids, use_reentrant=False)
            else:
                x = layer(x, attention_mask, position_ids)
        
        x = self.norm(x)
        return self.lm_head(x)

# =============================================================================
# DATASET AND DATA LOADING
# =============================================================================

class ConversationDataset(IterableDataset):
    """Memory-efficient conversation dataset."""
    
    def __init__(self, 
                 data_path: str, 
                 tokenizer: ConversationTokenizer, 
                 config: TrainingConfig,
                 split: str = "train"):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.config = config
        self.split = split
        
        # Check if data exists, create if not
        if not self.data_path.exists():
            if split == "train":
                logging.info(f"Creating sample training data at {data_path}")
                create_sample_conversation_data(data_path, 5000)
            else:
                logging.info(f"Creating sample evaluation data at {data_path}")
                create_sample_conversation_data(data_path, 500)
        
        self.total_conversations = self._count_conversations()
        logging.info(f"Dataset {split}: {self.total_conversations:,} conversations")
    
    def _count_conversations(self) -> int:
        """Count total conversations safely."""
        count = 0
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                for _ in f:
                    count += 1
        except Exception as e:
            logging.error(f"Error counting conversations: {e}")
            return 0
        return count
    
    def _create_attention_mask(self, tokens: List[int]) -> torch.Tensor:
        """Create proper attention mask."""
        mask = torch.ones(len(tokens), dtype=torch.long)
        pad_id = self.tokenizer.get_special_token_id('<pad>')
        
        for i, token_id in enumerate(tokens):
            if token_id == pad_id:
                mask[i] = 0
        
        return mask
    
    def _create_loss_mask(self, tokens: List[int]) -> torch.Tensor:
        """Create mask for loss computation."""
        loss_mask = torch.zeros(len(tokens), dtype=torch.float)
        
        assistant_id = self.tokenizer.get_special_token_id('<assistant>')
        turn_end_id = self.tokenizer.get_special_token_id('<turn_end>')
        
        in_assistant_turn = False
        
        for i, token_id in enumerate(tokens):
            if token_id == assistant_id:
                in_assistant_turn = True
            elif token_id == turn_end_id:
                in_assistant_turn = False
            elif in_assistant_turn and not self.tokenizer.is_special_token(token_id):
                loss_mask[i] = self.config.assistant_loss_weight
        
        return loss_mask
    
    def _process_conversation(self, conversation: Dict[str, Any]) -> Optional[Dict[str, torch.Tensor]]:
        """Process a single conversation."""
        try:
            messages = conversation.get('messages', [])
            if not messages:
                return None
            
            # Validate message structure
            for msg in messages:
                if not isinstance(msg, dict) or 'role' not in msg or 'content' not in msg:
                    return None
                if not msg['content'].strip():
                    return None
            
            # Encode conversation
            tokens = self.tokenizer.encode_conversation(messages)
            
            # Skip if too short or too long
            if len(tokens) < 10 or len(tokens) > self.config.seq_length:
                return None
            
            # Pad or truncate to sequence length
            if len(tokens) < self.config.seq_length:
                pad_id = self.tokenizer.get_special_token_id('<pad>')
                tokens.extend([pad_id] * (self.config.seq_length - len(tokens)))
            else:
                tokens = tokens[:self.config.seq_length]
            
            # Convert to tensors
            input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
            labels = torch.tensor(tokens[1:], dtype=torch.long)
            
            # Create masks
            attention_mask = self._create_attention_mask(tokens[:-1])
            loss_mask = self._create_loss_mask(tokens[1:])
            
            return {
                'input_ids': input_ids,
                'labels': labels,
                'attention_mask': attention_mask,
                'loss_mask': loss_mask
            }
            
        except Exception as e:
            return None
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate over processed conversations."""
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                for line_no, line in enumerate(f, 1):
                    try:
                        conversation = json.loads(line.strip())
                        processed = self._process_conversation(conversation)
                        if processed is not None:
                            yield processed
                    except json.JSONDecodeError:
                        continue
                    except Exception:
                        continue
        except Exception as e:
            logging.error(f"Error reading data file: {e}")

def create_dataloader(dataset: ConversationDataset, 
                     config: TrainingConfig, 
                     shuffle: bool = True,
                     sampler: Optional[DistributedSampler] = None) -> DataLoader:
    """Create optimized dataloader."""
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory and torch.cuda.is_available(),
        prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
        persistent_workers=config.dataloader_persistent_workers and config.num_workers > 0,
        drop_last=True
    )

# =============================================================================
# TRAINER
# =============================================================================

class ConversationTrainer:
    """Production trainer with all optimizations."""
    
    def __init__(self, 
                 model: TransformerModel, 
                 tokenizer: ConversationTokenizer, 
                 config: TrainingConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
        # Setup device and distributed training
        self._setup_distributed()
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Setup distributed model
        if self.config.distributed and torch.cuda.is_available():
            self.model = DDP(
                self.model, 
                device_ids=[self.config.local_rank],
                output_device=self.config.local_rank,
                find_unused_parameters=False
            )
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup mixed precision
        self.use_amp = config.precision in ["fp16", "bf16"]
        self.dtype = torch.bfloat16 if config.precision == "bf16" else torch.float16
        self.scaler = GradScaler() if config.precision == "fp16" else None
        
        # Compile model
        if config.compile_model and hasattr(torch, 'compile'):
            try:
                model_to_compile = self.model.module if hasattr(self.model, 'module') else self.model
                compiled_model = torch.compile(model_to_compile, mode=config.torch_compile_mode)
                if hasattr(self.model, 'module'):
                    self.model.module = compiled_model
                else:
                    self.model = compiled_model
                logging.info("Model compiled successfully")
            except Exception as e:
                logging.warning(f"Model compilation failed: {e}")
        
        # Training state
        self.global_step = 0
        self.scheduler = None
        self.best_eval_loss = float('inf')
        
        # Metrics tracking
        self.metrics = defaultdict(list)
        
        # Setup monitoring
        self._setup_monitoring()
    
    def _setup_distributed(self):
        """Setup distributed training."""
        if self.config.distributed:
            if not dist.is_initialized():
                dist.init_process_group(backend='nccl')
            
            self.config.local_rank = int(os.environ.get('LOCAL_RANK', 0))
            self.config.world_size = dist.get_world_size()
            torch.cuda.set_device(self.config.local_rank)
            self.device = torch.device(f'cuda:{self.config.local_rank}')
            self.is_main_process = self.config.local_rank == 0
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.is_main_process = True
    
    def _setup_monitoring(self):
        """Setup monitoring with wandb."""
        if HAS_WANDB and self.config.use_wandb and self.is_main_process:
            wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name,
                config=self.config.__dict__
            )
            logging.info("W&B monitoring initialized")
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with proper parameter grouping."""
        decay_params = []
        no_decay_params = []
        
        model_to_optimize = self.model.module if hasattr(self.model, 'module') else self.model
        
        for name, param in model_to_optimize.named_parameters():
            if not param.requires_grad:
                continue
            
            if any(keyword in name for keyword in ['bias', 'norm', 'embed']):
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        param_groups = [
            {'params': decay_params, 'weight_decay': self.config.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        
        use_fused = self.config.use_fused_optimizer and torch.cuda.is_available()
        
        return AdamW(
            param_groups,
            lr=self.config.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            fused=use_fused
        )
    
    def compute_loss(self, 
                    logits: torch.Tensor, 
                    labels: torch.Tensor, 
                    loss_mask: torch.Tensor) -> torch.Tensor:
        """Compute loss with proper masking."""
        shift_logits = logits.view(-1, logits.size(-1))
        shift_labels = labels.view(-1)
        shift_mask = loss_mask.view(-1)
        
        if self.config.label_smoothing > 0:
            log_probs = F.log_softmax(shift_logits, dim=-1)
            nll_loss = F.nll_loss(log_probs, shift_labels, reduction='none')
            smooth_loss = -log_probs.mean(dim=-1)
            loss = (1 - self.config.label_smoothing) * nll_loss + self.config.label_smoothing * smooth_loss
        else:
            loss = F.cross_entropy(shift_logits, shift_labels, reduction='none')
        
        masked_loss = loss * shift_mask
        
        mask_sum = shift_mask.sum()
        if mask_sum > 0:
            return masked_loss.sum() / mask_sum
        else:
            return masked_loss.sum()
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single training step."""
        self.model.train()
        
        batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
        
        with autocast(enabled=self.use_amp, dtype=self.dtype):
            logits = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask']
            )
            loss = self.compute_loss(logits, batch['labels'], batch['loss_mask'])
            loss = loss / self.config.gradient_accumulation_steps
        
        if self.use_amp and self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        return loss.item() * self.config.gradient_accumulation_steps
    
    def optimizer_step(self):
        """Perform optimizer step with gradient clipping."""
        if self.use_amp and self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
        
        model_to_clip = self.model.module if hasattr(self.model, 'module') else self.model
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model_to_clip.parameters(), 
            self.config.max_grad_norm
        )
        
        if self.use_amp and self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        self.optimizer.zero_grad(set_to_none=True)
        
        if self.scheduler:
            self.scheduler.step()
        
        return grad_norm.item()
    
    @torch.no_grad()
    def evaluate(self, eval_dataloader: DataLoader, max_batches: int = 100) -> Dict[str, float]:
        """Comprehensive evaluation."""
        self.model.eval()
        
        total_loss = 0.0
        total_tokens = 0
        num_batches = 0
        
        for batch in eval_dataloader:
            if num_batches >= max_batches:
                break
            
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
            
            with autocast(enabled=self.use_amp, dtype=self.dtype):
                logits = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
                loss = self.compute_loss(logits, batch['labels'], batch['loss_mask'])
            
            valid_tokens = batch['loss_mask'].sum().item()
            
            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens
            num_batches += 1
        
        if total_tokens > 0:
            avg_loss = total_loss / total_tokens
            perplexity = math.exp(min(avg_loss, 20))
        else:
            avg_loss = float('inf')
            perplexity = float('inf')
        
        return {
            'eval_loss': avg_loss,
            'eval_perplexity': perplexity,
            'eval_batches': num_batches
        }
    
    def train(self, train_dataloader: DataLoader, eval_dataloader: Optional[DataLoader] = None):
        """Main training loop."""
        if self.is_main_process:
            logging.info("Starting training...")
            logging.info(f"Max steps: {self.config.max_steps:,}")
            logging.info(f"Effective batch size: {self.config.effective_batch_size:,}")
            logging.info(f"Device: {self.device}")
        
        # Setup scheduler
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.config.learning_rate,
            total_steps=self.config.max_steps,
            pct_start=self.config.warmup_ratio,
            anneal_strategy='cos'
        )
        
        # Training loop
        train_iterator = iter(train_dataloader)
        accumulation_loss = 0.0
        start_time = time.time()
        
        for step in range(self.config.max_steps):
            step_start = time.time()
            
            # Accumulation loop
            for micro_step in range(self.config.gradient_accumulation_steps):
                try:
                    batch = next(train_iterator)
                except StopIteration:
                    train_iterator = iter(train_dataloader)
                    batch = next(train_iterator)
                
                loss = self.train_step(batch)
                accumulation_loss += loss
            
            # Optimizer step
            grad_norm = self.optimizer_step()
            self.global_step += 1
            
            # Metrics tracking
            step_time = time.time() - step_start
            lr = self.scheduler.get_last_lr()[0]
            
            self.metrics['train_loss'].append(accumulation_loss)
            self.metrics['learning_rate'].append(lr)
            self.metrics['grad_norm'].append(grad_norm)
            self.metrics['step_time'].append(step_time)
            
            # Logging
            if self.is_main_process and (step + 1) % self.config.log_interval == 0:
                tokens_per_sec = (self.config.effective_batch_size * self.config.seq_length) / step_time
                
                log_msg = (
                    f"Step {step + 1:6d} | Loss: {accumulation_loss:.6f} | "
                    f"LR: {lr:.2e} | Grad Norm: {grad_norm:.3f} | "
                    f"Tokens/s: {tokens_per_sec:.0f} | Time: {step_time:.2f}s"
                )
                logging.info(log_msg)
                
                if HAS_WANDB and self.config.use_wandb:
                    wandb.log({
                        'train/loss': accumulation_loss,
                        'train/learning_rate': lr,
                        'train/grad_norm': grad_norm,
                        'train/tokens_per_second': tokens_per_sec,
                        'train/step_time': step_time,
                        'step': self.global_step
                    })
            
            # Evaluation
            if eval_dataloader and (step + 1) % self.config.eval_steps == 0:
                eval_metrics = self.evaluate(eval_dataloader)
                
                if self.is_main_process:
                    logging.info(
                        f"Eval | Loss: {eval_metrics['eval_loss']:.6f} | "
                        f"Perplexity: {eval_metrics['eval_perplexity']:.2f}"
                    )
                    
                    if HAS_WANDB and self.config.use_wandb:
                        wandb.log({
                            'eval/loss': eval_metrics['eval_loss'],
                            'eval/perplexity': eval_metrics['eval_perplexity'],
                            'step': self.global_step
                        })
                    
                    if eval_metrics['eval_loss'] < self.best_eval_loss:
                        self.best_eval_loss = eval_metrics['eval_loss']
                        self.save_checkpoint(step + 1, is_best=True)
                
                self.model.train()
            
            # Checkpointing
            if self.is_main_process and (step + 1) % self.config.save_steps == 0:
                self.save_checkpoint(step + 1)
            
            accumulation_loss = 0.0
        
        # Final checkpoint
        if self.is_main_process:
            self.save_checkpoint(self.config.max_steps, final=True)
            
            total_time = time.time() - start_time
            logging.info(f"Training completed in {total_time / 3600:.2f} hours")
    
    def save_checkpoint(self, step: int, final: bool = False, is_best: bool = False):
        """Save model checkpoint."""
        if not self.is_main_process:
            return
        
        if final:
            checkpoint_name = "final"
        elif is_best:
            checkpoint_name = "best"
        else:
            checkpoint_name = f"step_{step:06d}"
        
        checkpoint_path = Path(self.config.checkpoint_dir) / f"model_{checkpoint_name}.pt"
        
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        
        checkpoint = {
            'model_state_dict': model_to_save.state_dict(),
            'config': self.config,
            'global_step': self.global_step,
            'best_eval_loss': self.best_eval_loss,
            'tokenizer_info': {
                'vocab_size': self.tokenizer.vocab_size,
                'special_tokens': self.tokenizer.special_tokens
            }
        }
        
        if self.config.save_optimizer_state and not is_best:
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
            if self.scheduler:
                checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            if self.scaler:
                checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        logging.info(f"Checkpoint saved: {checkpoint_path}")
        
        if not (final or is_best):
            self._cleanup_old_checkpoints()
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints to save space."""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        step_checkpoints = sorted([
            f for f in checkpoint_dir.glob("model_step_*.pt")
        ], key=lambda x: int(x.stem.split('_')[-1]))
        
        if len(step_checkpoints) > self.config.keep_last_n_checkpoints:
            for checkpoint in step_checkpoints[:-self.config.keep_last_n_checkpoints]:
                checkpoint.unlink()

# =============================================================================
# TEXT GENERATION
# =============================================================================

class TextGenerator:
    """Enhanced text generator."""
    
    def __init__(self, model: TransformerModel, tokenizer: ConversationTokenizer, config: TrainingConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = next(model.parameters()).device
    
    def _apply_repetition_penalty(self, 
                                 logits: torch.Tensor, 
                                 input_ids: torch.Tensor, 
                                 penalty: float) -> torch.Tensor:
        """Apply repetition penalty to logits."""
        if penalty == 1.0:
            return logits
        
        unique_ids = input_ids.unique()
        logits_penalty = torch.ones_like(logits)
        logits_penalty[unique_ids] = penalty
        penalty_logits = logits / logits_penalty
        
        return penalty_logits
    
    def _top_k_top_p_filtering(self, 
                              logits: torch.Tensor, 
                              top_k: int = 0, 
                              top_p: float = 1.0) -> torch.Tensor:
        """Apply top-k and top-p filtering."""
        batch_size, vocab_size = logits.shape
        
        if top_k > 0:
            top_k = min(top_k, vocab_size)
            top_k_values, top_k_indices = torch.topk(logits, top_k, dim=-1)
            indices_to_remove = logits < top_k_values[..., -1, None]
            logits = logits.masked_fill(indices_to_remove, float('-inf'))
        
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 0] = False
            
            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )
            logits = logits.masked_fill(indices_to_remove, float('-inf'))
        
        return logits
    
    @torch.no_grad()
    def generate(self, 
                prompt: str, 
                max_new_tokens: Optional[int] = None,
                temperature: Optional[float] = None,
                top_p: Optional[float] = None,
                top_k: Optional[int] = None,
                repetition_penalty: Optional[float] = None,
                do_sample: bool = True) -> str:
        """Generate response."""
        self.model.eval()
        
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        temperature = temperature or self.config.temperature
        top_p = top_p or self.config.top_p
        top_k = top_k or self.config.top_k
        repetition_penalty = repetition_penalty or self.config.repetition_penalty
        
        # Create conversation format
        messages = [{'role': 'user', 'content': prompt}]
        
        # Encode input
        input_tokens = self.tokenizer.encode_conversation(messages)
        
        # Add assistant start token
        assistant_id = self.tokenizer.get_special_token_id('<assistant>')
        turn_start_id = self.tokenizer.get_special_token_id('<turn_start>')
        
        input_tokens.extend([turn_start_id, assistant_id])
        
        input_ids = torch.tensor([input_tokens], device=self.device, dtype=torch.long)
        
        generated_tokens = []
        
        stop_tokens = {
            self.tokenizer.get_special_token_id('<turn_end>'),
            self.tokenizer.get_special_token_id('<eos>'),
            self.tokenizer.get_special_token_id('<user>')
        }
        
        for _ in range(max_new_tokens):
            with autocast(enabled=self.config.precision in ["fp16", "bf16"]):
                logits = self.model(input_ids)
            
            next_token_logits = logits[0, -1, :] / temperature
            
            if repetition_penalty != 1.0:
                next_token_logits = self._apply_repetition_penalty(
                    next_token_logits, input_ids[0], repetition_penalty
                )
            
            if do_sample:
                filtered_logits = self._top_k_top_p_filtering(
                    next_token_logits.unsqueeze(0), top_k, top_p
                ).squeeze(0)
                
                probs = F.softmax(filtered_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            if next_token.item() in stop_tokens:
                break
            
            if self.tokenizer.is_special_token(next_token.item()):
                continue
            
            generated_tokens.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            
            if len(generated_tokens) > max_new_tokens:
                break
        
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return response.strip()

# =============================================================================
# DATA UTILITIES
# =============================================================================

def create_sample_conversation_data(output_path: str, num_conversations: int = 1000):
    """Create realistic sample conversational data."""
    import random
    
    templates = [
        {
            "topics": ["programming", "coding", "software development"],
            "patterns": [
                ("How do I learn {topic}?", "Learning {topic} requires practice and patience. Start with the basics and work your way up. Focus on understanding fundamental concepts like variables, functions, and control structures. Practice coding regularly and build small projects to apply what you learn."),
                ("What's the best way to {action} in {topic}?", "For {action} in {topic}, I recommend following best practices and established patterns. Start by understanding the problem thoroughly, then break it down into smaller, manageable pieces. Use proper naming conventions and write clean, readable code."),
                ("Can you explain {concept}?", "{concept} is an important concept in {topic}. It helps you organize and structure your code effectively. Think of it as a way to group related functionality together and make your programs more modular and maintainable.")
            ]
        },
        {
            "topics": ["science", "physics", "biology", "chemistry"],
            "patterns": [
                ("What is {concept}?", "{concept} is a fundamental principle that governs how things work in the natural world. It describes the relationship between different forces and helps us understand complex phenomena through mathematical models and experimental observation."),
                ("How does {process} work?", "{process} works through a series of interconnected steps and mechanisms. Each stage builds upon the previous one, creating a complex but elegant system that achieves a specific biological or physical function."),
                ("Why is {topic} important?", "{topic} is crucial because it helps us understand the world around us and solve real-world problems. It provides the foundation for technological advances and medical breakthroughs that improve our daily lives.")
            ]
        },
        {
            "topics": ["cooking", "recipes", "food"],
            "patterns": [
                ("How do I cook {dish}?", "To cook {dish}, you'll need fresh ingredients and proper preparation. Start by gathering all your ingredients and reading through the recipe completely. Prepare your workspace and follow each step carefully, paying attention to timing and temperature."),
                ("What's a good recipe for {dish}?", "Here's a great recipe for {dish}: Start with quality ingredients and don't rush the process. The key is to balance flavors and textures while maintaining proper cooking techniques. Season as you go and taste frequently."),
                ("How long does it take to make {dish}?", "Making {dish} typically takes about 30-45 minutes of active preparation, plus additional time for cooking or baking. The exact time depends on your skill level and the specific recipe you're following.")
            ]
        }
    ]
    
    conversations = []
    
    for i in range(num_conversations):
        template = random.choice(templates)
        topic = random.choice(template["topics"])
        user_pattern, assistant_pattern = random.choice(template["patterns"])
        
        # Fill in the patterns
        concept = random.choice(["functions", "variables", "loops", "classes", "objects", "algorithms"])
        action = random.choice(["optimize", "debug", "implement", "design", "refactor"])
        dish = random.choice(["pasta", "soup", "salad", "bread", "cake", "stir-fry"])
        process = random.choice(["photosynthesis", "metabolism", "evolution", "chemical reactions"])
        
        user_msg = user_pattern.format(
            topic=topic, concept=concept, action=action, dish=dish, process=process
        )
        assistant_msg = assistant_pattern.format(
            topic=topic, concept=concept, action=action, dish=dish, process=process
        )
        
        # Create conversation
        conversation = {
            "conversation_id": f"sample_{i:06d}",
            "messages": [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg}
            ]
        }
        
        # Sometimes add follow-up
        if random.random() < 0.3:
            follow_ups = [
                "Can you give me more details about that?",
                "That's helpful, but I have another question.",
                "What about more advanced techniques?",
                "Are there any common mistakes I should avoid?"
            ]
            responses = [
                "Certainly! Let me elaborate on that point. It's important to understand the underlying principles and how they apply in different situations.",
                "Great question! Here's some additional information that might help you understand the concept better.",
                "For more advanced techniques, you'll want to focus on optimization and best practices that experienced practitioners use.",
                "Yes, here are some common pitfalls to watch out for. These mistakes can be frustrating but are great learning opportunities."
            ]
            
            follow_up = random.choice(follow_ups)
            response = random.choice(responses)
            
            conversation["messages"].extend([
                {"role": "user", "content": follow_up},
                {"role": "assistant", "content": response}
            ])
        
        conversations.append(conversation)
    
    # Write to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for conv in conversations:
            f.write(json.dumps(conv) + '\n')
    
    logging.info(f"Created {num_conversations} sample conversations at {output_path}")

# =============================================================================
# MAIN TRAINING SCRIPT
# =============================================================================

def setup_logging():
    """Setup comprehensive logging."""
    os.makedirs("logs", exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)8s | %(message)s',
        handlers=[
            logging.FileHandler("logs/training.log"),
            logging.StreamHandler()
        ],
        force=True
    )
    
    logging.info("=" * 80)
    logging.info("HARDCODED CONVERSATIONAL TRANSFORMER TRAINING")
    logging.info("=" * 80)
    
    if torch.cuda.is_available():
        logging.info(f"CUDA Device: {torch.cuda.get_device_name()}")
        logging.info(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    else:
        logging.info("Running on CPU")
    
    logging.info(f"PyTorch version: {torch.__version__}")
    logging.info(f"Flash Attention available: {HAS_FLASH_ATTN}")
    logging.info(f"W&B available: {HAS_WANDB}")

def main():
    """Main training function - hardcoded configuration."""
    
    # Setup logging
    setup_logging()
    
    # Create hardcoded configuration
    config = TrainingConfig()
    
    logging.info("Using hardcoded configuration:")
    logging.info(f"  Model size: {config.hidden_size}d, {config.num_layers} layers")
    logging.info(f"  Vocabulary size: {config.vocab_size:,}")
    logging.info(f"  Sequence length: {config.seq_length:,}")
    logging.info(f"  Training steps: {config.max_steps:,}")
    logging.info(f"  Batch size: {config.batch_size} (effective: {config.effective_batch_size})")
    logging.info(f"  Learning rate: {config.learning_rate:.2e}")
    logging.info(f"  Precision: {config.precision}")
    
    # Estimate model size
    embed_params = config.vocab_size * config.hidden_size
    attn_params = config.hidden_size * config.hidden_size * 4  # Q, K, V, O projections (simplified)
    mlp_params = config.hidden_size * config.intermediate_size * 3  # gate, up, down
    layer_params = (attn_params + mlp_params + config.hidden_size * 2) * config.num_layers
    total_params = embed_params + layer_params + config.hidden_size
    
    logging.info(f"  Estimated parameters: ~{total_params:,}")
    
    # Initialize tokenizer
    tokenizer = ConversationTokenizer(config.vocab_size)
    config.vocab_size = tokenizer.vocab_size
    
    logging.info(f"Tokenizer initialized (vocab_size={tokenizer.vocab_size})")
    
    # Create data directories
    os.makedirs("data", exist_ok=True)
    
    # Create datasets (will auto-generate sample data if not found)
    logging.info("Setting up datasets...")
    train_dataset = ConversationDataset(config.train_data_path, tokenizer, config, "train")
    eval_dataset = ConversationDataset(config.eval_data_path, tokenizer, config, "eval")
    
    # Create distributed samplers if needed
    train_sampler = None
    eval_sampler = None
    
    if config.distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        eval_sampler = DistributedSampler(eval_dataset, shuffle=False)
    
    # Create dataloaders
    train_dataloader = create_dataloader(train_dataset, config, shuffle=not config.distributed, sampler=train_sampler)
    eval_dataloader = create_dataloader(eval_dataset, config, shuffle=False, sampler=eval_sampler)
    
    # Initialize model
    logging.info("Initializing model...")
    model = TransformerModel(config)
    
    # Initialize trainer
    trainer = ConversationTrainer(model, tokenizer, config)
    
    # Log training info
    logging.info("Training configuration:")
    logging.info(f"  Max steps: {config.max_steps:,}")
    logging.info(f"  Effective batch size: {config.effective_batch_size:,}")
    logging.info(f"  Learning rate: {config.learning_rate:.2e}")
    logging.info(f"  Precision: {config.precision}")
    logging.info(f"  Gradient accumulation steps: {config.gradient_accumulation_steps}")
    logging.info(f"  Gradient checkpointing: {config.use_gradient_checkpointing}")
    logging.info(f"  Device: {trainer.device}")
    
    # Start training
    logging.info("Starting training...")
    
    try:
        trainer.train(train_dataloader, eval_dataloader)
    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
        trainer.save_checkpoint(trainer.global_step, final=True)
    except Exception as e:
        logging.error(f"Training failed: {e}")
        # Save emergency checkpoint
        try:
            trainer.save_checkpoint(trainer.global_step, final=True)
        except:
            pass
        return 1
    
    # Test generation
    logging.info("Testing generation...")
    generator = TextGenerator(trainer.model, tokenizer, config)
    
    test_prompts = [
        "How do I learn Python programming?",
        "Explain machine learning in simple terms.",
        "What are the benefits of exercise?",
        "Can you help me understand recursion in programming?"
    ]
    
    for prompt in test_prompts:
        logging.info(f"\nPrompt: {prompt}")
        try:
            response = generator.generate(prompt, max_new_tokens=128)
            logging.info(f"Response: {response}")
        except Exception as e:
            logging.error(f"Generation failed: {e}")
    
    # Cleanup distributed
    if config.distributed:
        dist.destroy_process_group()
    
    logging.info("Training completed successfully!")
    logging.info("Checkpoints saved in: ./checkpoints/")
    logging.info("Logs saved in: ./logs/")
    
    return 0

if __name__ == "__main__":
    exit(main())