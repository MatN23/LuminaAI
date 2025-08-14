# Production-Ready Conversational Transformer Training System
# Optimized for performance, scalability, and maintainability

import json
import logging
import time
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Iterator
import warnings
import gc, torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np
import tiktoken
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
gc.collect()
torch.cuda.empty_cache()

# High-performance imports
try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False

from torch.cuda.amp import autocast, GradScaler

# Configure for performance
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class Config:
    """Unified configuration for model, training, and data."""
    # Model architecture
    vocab_size: int = 50304  # GPT-4 vocab size, padded to multiple of 64
    hidden_size: int = 2048
    num_layers: int = 24
    num_heads: int = 16
    num_kv_heads: int = 8
    seq_length: int = 2048
    intermediate_size: int = 5504
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    dropout: float = 0.0
    
    # Training parameters
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_steps: int = 10000  # Reduced from 50000
    warmup_ratio: float = 0.1
    eval_steps: int = 500   # Reduced from 1000
    save_steps: int = 2000  # Reduced from 5000
    max_grad_norm: float = 1.0
    precision: str = "fp16"
    compile: bool = True
    
    # Data parameters  
    train_data_path: str = "oasst1_data/oasst1_train.jsonl"
    eval_data_path: str = "oasst1_data/oasst1_validation.jsonl"
    num_workers: int = 2  # Reduced from 8 to match system recommendation
    assistant_loss_weight: float = 2.0
    max_conversations_per_file: int = 10000  # For memory management
    
    # Generation parameters
    max_new_tokens: int = 512
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 50
    
    def __post_init__(self):
        assert self.hidden_size % self.num_heads == 0, "hidden_size must be divisible by num_heads"
        assert self.num_heads % self.num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
        
        # Ensure vocab size is efficient
        if self.vocab_size % 64 != 0:
            self.vocab_size = ((self.vocab_size + 63) // 64) * 64
            
        self.warmup_steps = int(self.max_steps * self.warmup_ratio)
        self.effective_batch_size = self.batch_size * self.gradient_accumulation_steps

# =============================================================================
# TOKENIZATION WITH TIKTOKEN
# =============================================================================

class ConversationTokenizer:
    """Production tokenizer using tiktoken with conversation-specific formatting."""
    
    def __init__(self, model_name: str = "gpt2"):
        self.tokenizer = tiktoken.get_encoding(model_name)
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
        
        # Pad vocab size to be efficient (multiple of 64)
        if self.vocab_size % 64 != 0:
            self.vocab_size = ((self.vocab_size + 63) // 64) * 64
            
        logging.info(f"Tokenizer initialized with vocab size: {self.vocab_size}")
    
    def encode_conversation(self, conversation: Dict[str, any]) -> List[int]:
        """Encode a conversation with proper formatting."""
        tokens = []
        messages = conversation.get('messages', [])
        
        for message in messages:
            role = message.get('role', '').lower()
            content = message.get('content', '').strip()
            
            if not content:
                continue
                
            # Start message
            tokens.append(self.special_tokens["<|im_start|>"])
            
            # Add role
            if role == 'user' or role == 'prompter':
                tokens.append(self.special_tokens["<|user|>"])
            elif role == 'assistant':
                tokens.append(self.special_tokens["<|assistant|>"])
            else:
                tokens.append(self.special_tokens["<|system|>"])
            
            # Add content
            content_tokens = self.tokenizer.encode(content)
            tokens.extend(content_tokens)
            
            # End message
            tokens.append(self.special_tokens["<|im_end|>"])
        
        return tokens
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode tokens back to text."""
        # Filter out special tokens if requested
        if skip_special_tokens:
            filtered_tokens = []
            for token_id in token_ids:
                if token_id not in self._reverse_special_tokens and token_id < self.base_vocab_size:
                    filtered_tokens.append(token_id)
            token_ids = filtered_tokens
        
        try:
            return self.tokenizer.decode(token_ids)
        except Exception:
            # Fallback for out-of-vocab tokens
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
# EFFICIENT MODEL ARCHITECTURE
# =============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 8192, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
    
    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
              if seq_len > self._seq_len_cached or self._cos_cached is None or self._cos_cached.device != device:
                    self._seq_len_cached = seq_len
                    t = torch.arange(seq_len, device=device, dtype=torch.float32)
                    freqs = torch.outer(t, self.inv_freq.to(device))
                    emb = torch.cat((freqs, freqs), dim=-1)

                    self._cos_cached = emb.cos().clone()
                    self._sin_cached = emb.sin().clone()

              return self._cos_cached[:seq_len], self._sin_cached[:seq_len]


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, 
                        cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class GroupedQueryAttention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        
        self.rope = RotaryEmbedding(self.head_dim, config.seq_length, config.rope_theta)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for proj in [self.q_proj, self.k_proj, self.v_proj]:
            nn.init.xavier_uniform_(proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.o_proj.weight, gain=1 / math.sqrt(2 * self.config.num_layers))
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, D = x.shape
        
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        cos, sin = self.rope(L, x.device)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Expand K, V for GQA
        if self.num_queries_per_kv > 1:
            k = k[:, :, None, :, :].expand(B, self.num_kv_heads, self.num_queries_per_kv, L, self.head_dim)
            v = v[:, :, None, :, :].expand(B, self.num_kv_heads, self.num_queries_per_kv, L, self.head_dim)
            k = k.reshape(B, self.num_heads, L, self.head_dim)
            v = v.reshape(B, self.num_heads, L, self.head_dim)
        
        # Use FlashAttention if available
        if HAS_FLASH_ATTN and x.is_cuda and attention_mask is None:
            q = q.transpose(1, 2)  # (B, L, H, D)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            out = flash_attn_func(q, k, v, causal=True)
            out = out.reshape(B, L, self.hidden_size)
        else:
            # Standard attention
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            
            # Apply causal mask
            if attention_mask is None:
                causal_mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1)
                scores.masked_fill_(causal_mask, float('-inf'))
            else:
                # Fix the broadcasting issue: reshape attention_mask from [B, L] to [B, 1, 1, L]
                if attention_mask.dim() == 2:  # [batch_size, seq_len]
                    attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
                    # Create causal mask and combine with attention mask
                    causal_mask = torch.triu(torch.ones(L, L, device=x.device, dtype=attention_mask.dtype), diagonal=1)
                    # Convert boolean mask to additive mask (0 for valid, -inf for invalid)
                    causal_mask = causal_mask * float('-inf')
                    # Combine masks
                    combined_mask = attention_mask + causal_mask.unsqueeze(0).unsqueeze(0)
                    scores = scores + combined_mask
                else:
                    scores = scores + attention_mask
            
            attn = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
            out = torch.matmul(attn, v)
            out = out.transpose(1, 2).reshape(B, L, self.hidden_size)
        
        return self.o_proj(out)

class SwiGLU(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.gate_proj.weight)
        nn.init.xavier_uniform_(self.up_proj.weight)
        nn.init.xavier_uniform_(self.down_proj.weight, gain=1 / math.sqrt(2 * self.config.num_layers))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)

class TransformerBlock(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.input_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = GroupedQueryAttention(config)
        self.post_attn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = SwiGLU(config)
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm architecture
        x = x + self.self_attn(self.input_norm(x), attention_mask)
        x = x + self.mlp(self.post_attn_norm(x))
        return x

class TransformerModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Weight tying
        self.lm_head.weight = self.embed_tokens.weight
        
        self._init_weights()
        
        # Count parameters
        n_params = sum(p.numel() for p in self.parameters())
        logging.info(f"Model initialized with {n_params:,} parameters")
    
    def _init_weights(self):
        nn.init.normal_(self.embed_tokens.weight, std=0.02)
        
        # Apply scaling to deeper layers
        for layer in self.layers:
            layer.self_attn.o_proj.weight.data *= (2 * self.config.num_layers) ** -0.5
            layer.mlp.down_proj.weight.data *= (2 * self.config.num_layers) ** -0.5
    
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        
        for layer in self.layers:
            x = layer(x, attention_mask)
        
        x = self.norm(x)
        return self.lm_head(x)

# =============================================================================
# MEMORY-EFFICIENT DATASET
# =============================================================================

class ConversationDataset(IterableDataset):
    """Memory-efficient streaming dataset for conversational data."""
    
    def __init__(self, data_path: str, tokenizer: ConversationTokenizer, 
                 config: Config, split: str = "train"):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.config = config
        self.split = split
        
        # Count total conversations for progress tracking
        self.total_conversations = self._count_conversations()
        logging.info(f"Dataset {split}: {self.total_conversations:,} conversations from {data_path}")
    
    def _count_conversations(self) -> int:
        """Count total conversations in the file."""
        if not self.data_path.exists():
            return 0
        
        count = 0
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for _ in f:
                count += 1
        return count
    
    def _process_conversation(self, conversation: Dict) -> Optional[Dict[str, torch.Tensor]]:
        """Process a single conversation into model inputs."""
        try:
            tokens = self.tokenizer.encode_conversation(conversation)
            
            # Skip if too short or too long
            if len(tokens) < 10 or len(tokens) > self.config.seq_length:
                return None
            
            # Pad to sequence length
            if len(tokens) < self.config.seq_length:
                tokens.extend([0] * (self.config.seq_length - len(tokens)))
            else:
                tokens = tokens[:self.config.seq_length]
            
            tokens = torch.tensor(tokens, dtype=torch.long)
            
            # Create attention mask
            attention_mask = (tokens != 0).float()
            
            # Create labels with assistant token weighting
            labels = tokens.clone()
            
            # Create loss weights
            loss_weights = torch.ones_like(tokens, dtype=torch.float)
            assistant_token = self.tokenizer.get_role_token('assistant')
            
            # Weight assistant responses higher
            assistant_positions = (tokens == assistant_token).float()
            if assistant_positions.sum() > 0:
                # Create forward mask from assistant tokens
                assistant_mask = torch.zeros_like(tokens, dtype=torch.float)
                in_assistant_response = False
                
                for i, token_id in enumerate(tokens):
                    if token_id == assistant_token:
                        in_assistant_response = True
                    elif token_id == self.tokenizer.special_tokens["<|im_end|>"]:
                        in_assistant_response = False
                    
                    if in_assistant_response:
                        assistant_mask[i] = 1.0
                
                loss_weights = loss_weights + (assistant_mask * (self.config.assistant_loss_weight - 1.0))
            
            return {
                'input_ids': tokens[:-1],
                'labels': labels[1:],
                'attention_mask': attention_mask[:-1],
                'loss_weights': loss_weights[1:]
            }
            
        except Exception as e:
            logging.warning(f"Error processing conversation: {e}")
            return None
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate over conversations."""
        if not self.data_path.exists():
            logging.error(f"Data file not found: {self.data_path}")
            return
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    conversation = json.loads(line.strip())
                    processed = self._process_conversation(conversation)
                    if processed is not None:
                        yield processed
                except json.JSONDecodeError:
                    continue

def create_dataloader(dataset: ConversationDataset, config: Config, shuffle: bool = True) -> DataLoader:
    """Create optimized dataloader."""
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=True,
        prefetch_factor=2 if config.num_workers > 0 else None
    )

# =============================================================================
# PRODUCTION TRAINER
# =============================================================================

class ConversationTrainer:
    """Production-ready trainer with all optimizations."""
    
    def __init__(self, model: TransformerModel, tokenizer: ConversationTokenizer, config: Config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Setup optimizer with layer-wise learning rate decay
        self.optimizer = self._create_optimizer()
        
        # Setup mixed precision
        self.use_amp = config.precision in ["fp16", "bf16"]
        self.dtype = torch.bfloat16 if config.precision == "bf16" else torch.float16
        self.scaler = GradScaler() if config.precision == "fp16" else None
        
        # Compile model
        if config.compile and hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model, mode='reduce-overhead')
                logging.info("Model compiled successfully")
            except Exception as e:
                logging.warning(f"Model compilation failed: {e}")
        
        self.global_step = 0
        self.scheduler = None
        
        # Metrics tracking
        self.metrics = {
            'train_loss': [],
            'eval_loss': [],
            'learning_rates': [],
            'step_times': []
        }
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with proper weight decay."""
        # Separate parameters for weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'bias' in name or 'norm' in name or 'embed' in name:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
        
        param_groups = [
            {'params': decay_params, 'weight_decay': self.config.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        
        return AdamW(
            param_groups,
            lr=self.config.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            fused=torch.cuda.is_available()
        )
    
    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor, 
                    loss_weights: torch.Tensor) -> torch.Tensor:
        """Compute weighted cross-entropy loss."""
        # Shift for causal modeling
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_weights = loss_weights[..., 1:].contiguous()
        
        # Compute loss
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction='none'
        )
        loss = loss.view_as(shift_labels)
        
        # Apply weights and mask padding
        mask = (shift_labels != 0).float()
        weighted_loss = loss * shift_weights * mask
        
        return weighted_loss.sum() / mask.sum().clamp(min=1)
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single training step."""
        self.model.train()
        
        # Move batch to device
        batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
        
        with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=self.dtype):
            logits = self.model(batch['input_ids'], batch['attention_mask'])
            loss = self.compute_loss(logits, batch['labels'], batch['loss_weights'])
            loss = loss / self.config.gradient_accumulation_steps
        
        # Backward pass
        if self.use_amp and self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        return loss.item() * self.config.gradient_accumulation_steps
    
    def optimizer_step(self):
        """Perform optimizer step with gradient clipping."""
        if self.use_amp and self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
        
        self.optimizer.zero_grad(set_to_none=True)
        if self.scheduler:
            self.scheduler.step()
    
    @torch.no_grad()
    def evaluate(self, eval_dataloader: DataLoader, max_batches: int = 100) -> float:
        """Evaluate model on validation data."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for batch in eval_dataloader:
            if num_batches >= max_batches:
                break
            
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
            
            with autocast(enabled=self.use_amp, dtype=self.dtype):
                logits = self.model(batch['input_ids'], batch['attention_mask'])
                loss = self.compute_loss(logits, batch['labels'], batch['loss_weights'])
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def train(self, train_dataloader: DataLoader, eval_dataloader: Optional[DataLoader] = None):
        """Main training loop."""
        logging.info("Starting training...")
        logging.info(f"Max steps: {self.config.max_steps}")
        logging.info(f"Effective batch size: {self.config.effective_batch_size}")
        logging.info(f"Device: {self.device}")
        
        # Setup scheduler
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.config.learning_rate,
            total_steps=self.config.max_steps,
            pct_start=self.config.warmup_ratio,
            anneal_strategy='cos'
        )
        
        train_iterator = iter(train_dataloader)
        accumulation_loss = 0.0
        start_time = time.time()
        
        for step in range(self.config.max_steps):
            step_start = time.time()
            
            # Training steps
            for micro_step in range(self.config.gradient_accumulation_steps):
                try:
                    batch = next(train_iterator)
                except StopIteration:
                    train_iterator = iter(train_dataloader)
                    batch = next(train_iterator)
                
                loss = self.train_step(batch)
                accumulation_loss += loss
            
            # Optimizer step
            self.optimizer_step()
            self.global_step += 1
            
            # Logging
            step_time = time.time() - step_start
            self.metrics['step_times'].append(step_time)
            self.metrics['train_loss'].append(accumulation_loss)
            self.metrics['learning_rates'].append(self.scheduler.get_last_lr()[0])
            
            if (step + 1) % 100 == 0:
                lr = self.scheduler.get_last_lr()[0]
                tokens_per_sec = (self.config.effective_batch_size * self.config.seq_length) / step_time
                
                logging.info(
                    f"Step {step + 1:6d} | Loss: {accumulation_loss:.6f} | "
                    f"LR: {lr:.2e} | Tokens/s: {tokens_per_sec:.0f} | "
                    f"Step time: {step_time:.2f}s"
                )
            
            # Evaluation
            if eval_dataloader and (step + 1) % self.config.eval_steps == 0:
                eval_loss = self.evaluate(eval_dataloader)
                self.metrics['eval_loss'].append(eval_loss)
                logging.info(f"Eval loss: {eval_loss:.6f}")
                self.model.train()
            
            # Checkpointing
            if (step + 1) % self.config.save_steps == 0:
                self.save_checkpoint(step + 1)
            
            accumulation_loss = 0.0
        
        # Final checkpoint
        self.save_checkpoint(self.config.max_steps, final=True)
        
        total_time = time.time() - start_time
        logging.info(f"Training completed in {total_time / 3600:.2f} hours")
    
    def save_checkpoint(self, step: int, final: bool = False):
        """Save model checkpoint."""
        os.makedirs("checkpoints", exist_ok=True)
        
        suffix = "final" if final else f"step_{step:06d}"
        checkpoint_path = f"checkpoints/model_{suffix}.pt"
        
        # Unwrap compiled model if needed
        model_state = self.model.state_dict()
        if hasattr(self.model, '_orig_mod'):
            model_state = self.model._orig_mod.state_dict()
        
        checkpoint = {
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'config': self.config,
            'global_step': self.global_step,
            'metrics': self.metrics,
            'tokenizer_info': {
                'vocab_size': self.tokenizer.vocab_size,
                'special_tokens': self.tokenizer.special_tokens
            }
        }
        
        torch.save(checkpoint, checkpoint_path)
        logging.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Save metrics separately
        metrics_path = f"checkpoints/metrics_{suffix}.json"
        with open(metrics_path, 'w') as f:
            json.dump({k: v[-1000:] for k, v in self.metrics.items()}, f, indent=2)
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load checkpoint and return global step."""
        logging.info(f"Loading checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load model state
        model_to_load = self.model
        if hasattr(self.model, '_orig_mod'):
            model_to_load = self.model._orig_mod
        
        model_to_load.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer and scheduler
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint.get('scheduler_state_dict') and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training state
        self.global_step = checkpoint.get('global_step', 0)
        self.metrics = checkpoint.get('metrics', self.metrics)
        
        logging.info(f"Resumed from step {self.global_step}")
        return self.global_step
    
    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: Optional[int] = None) -> str:
        """Generate response to a prompt."""
        self.model.eval()
        
        if max_new_tokens is None:
            max_new_tokens = self.config.max_new_tokens
        
        # Create conversation format
        conversation = {
            'messages': [
                {'role': 'user', 'content': prompt}
            ]
        }
        
        # Encode input
        input_tokens = self.tokenizer.encode_conversation(conversation)
        
        # Add assistant start token
        input_tokens.extend([
            self.tokenizer.special_tokens["<|im_start|>"],
            self.tokenizer.special_tokens["<|assistant|>"]
        ])
        
        # Convert to tensor
        input_ids = torch.tensor([input_tokens], device=self.device, dtype=torch.long)
        
        # Generation loop with advanced sampling
        generated_tokens = []
        
        for _ in range(max_new_tokens):
            with autocast(enabled=self.use_amp, dtype=self.dtype):
                logits = self.model(input_ids)
            
            # Get next token logits
            next_token_logits = logits[0, -1, :] / self.config.temperature
            
            # Apply top-k filtering
            if self.config.top_k > 0:
                top_k_logits, top_k_indices = torch.topk(next_token_logits, self.config.top_k)
                next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                next_token_logits.scatter_(0, top_k_indices, top_k_logits)
            
            # Apply top-p (nucleus) filtering
            if self.config.top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
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
            
            # Prevent runaway generation
            if len(generated_tokens) > max_new_tokens:
                break
        
        # Decode response
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return response.strip()

# =============================================================================
# CONFIGURATION PRESETS
# =============================================================================

class ConfigPresets:
    """Predefined configurations for different use cases."""
    
    @staticmethod
    def debug() -> Config:
        """Minimal config for debugging."""
        return Config(
            # Tiny model
            vocab_size=1024,
            hidden_size=256,
            num_layers=4,
            num_heads=8,
            num_kv_heads=4,
            seq_length=512,
            intermediate_size=512,
            
            # Fast training
            batch_size=2,
            gradient_accumulation_steps=2,
            max_steps=500,  # Reduced from 1000
            warmup_ratio=0.1,
            eval_steps=100,  # Reduced from 200
            save_steps=200,  # Reduced from 500
            precision="fp32",
            compile=False,
            num_workers=0
        )
    
    @staticmethod
    def small() -> Config:
        """Small production model."""
        return Config(
            # Small model (similar to GPT-2 small)
            hidden_size=768,
            num_layers=12,
            num_heads=12,
            num_kv_heads=4,
            seq_length=2048,
            intermediate_size=2048,
            
            # Training config
            batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=5e-4,
            max_steps=10000,  # Reduced from 50000
            warmup_ratio=0.1,
            eval_steps=500,   # Reduced from 1000
            save_steps=2000,  # Reduced from 5000
            num_workers=2,    # Reduced from default
        )
    
    @staticmethod
    def medium() -> Config:
        """Medium model for serious training."""
        return Config(
            # Medium model
            hidden_size=1536,
            num_layers=24,
            num_heads=16,
            num_kv_heads=8,
            seq_length=2048,
            intermediate_size=4096,
            
            # Training config
            batch_size=2,
            gradient_accumulation_steps=8,
            learning_rate=3e-4,
            max_steps=20000,  # Reduced from 100000
            warmup_ratio=0.1,
            eval_steps=1000,  # Reduced from 2000
            save_steps=4000,  # Reduced from 10000
            num_workers=2,    # Reduced from default
        )
    
    @staticmethod
    def large() -> Config:
        """Large model for production use."""
        return Config(
            # Large model
            hidden_size=2048,
            num_layers=32,
            num_heads=16,
            num_kv_heads=8,
            seq_length=4096,
            intermediate_size=5504,
            
            # Training config
            batch_size=2,
            gradient_accumulation_steps=16,
            learning_rate=2e-4,
            max_steps=40000,  # Reduced from 200000
            warmup_ratio=0.05,
            eval_steps=2000,  # Reduced from 2500
            save_steps=8000,  # Reduced from 10000
            num_workers=2,    # Reduced from default
        )

# =============================================================================
# DATA UTILITIES
# =============================================================================

def create_sample_data(output_path: str, num_conversations: int = 1000):
    """Create sample conversational data for testing."""
    import random
    
    sample_conversations = []
    
    topics = [
        "programming", "science", "history", "cooking", "travel", "health",
        "technology", "music", "art", "literature", "sports", "movies"
    ]
    
    for i in range(num_conversations):
        topic = random.choice(topics)
        
        # Generate realistic conversation
        conversation = {
            "conversation_id": f"sample_{i:06d}",
            "messages": [
                {
                    "role": "user",
                    "content": f"Can you tell me about {topic}? I'm really interested in learning more."
                },
                {
                    "role": "assistant", 
                    "content": f"I'd be happy to tell you about {topic}! It's a fascinating subject with many interesting aspects. "
                              f"There are several key concepts you should understand when exploring {topic}. "
                              f"Would you like me to focus on any particular aspect of {topic}?"
                },
                {
                    "role": "user",
                    "content": f"Yes, could you give me some specific examples related to {topic}?"
                },
                {
                    "role": "assistant",
                    "content": f"Certainly! Here are some great examples of {topic} that might interest you: "
                              f"First, let me explain the fundamentals. Then I'll show you how these principles "
                              f"apply in real-world scenarios. This should give you a solid foundation in {topic}."
                }
            ]
        }
        
        sample_conversations.append(conversation)
    
    # Write to JSONL
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for conv in sample_conversations:
            f.write(json.dumps(conv) + '\n')
    
    logging.info(f"Created {num_conversations} sample conversations at {output_path}")

def validate_data(data_path: str, tokenizer: ConversationTokenizer, max_check: int = 1000) -> Dict:
    """Validate conversation data quality."""
    stats = {
        'total_conversations': 0,
        'valid_conversations': 0,
        'token_lengths': [],
        'message_counts': [],
        'role_distribution': {},
        'errors': []
    }
    
    if not os.path.exists(data_path):
        stats['errors'].append(f"File not found: {data_path}")
        return stats
    
    with open(data_path, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            if stats['total_conversations'] >= max_check:
                break
                
            try:
                conv = json.loads(line.strip())
                stats['total_conversations'] += 1
                
                # Validate structure
                if 'messages' not in conv:
                    stats['errors'].append(f"Line {line_no}: Missing 'messages' key")
                    continue
                
                messages = conv['messages']
                if not messages:
                    stats['errors'].append(f"Line {line_no}: Empty messages")
                    continue
                
                # Count messages and roles
                stats['message_counts'].append(len(messages))
                for msg in messages:
                    role = msg.get('role', 'unknown')
                    stats['role_distribution'][role] = stats['role_distribution'].get(role, 0) + 1
                
                # Test tokenization
                try:
                    tokens = tokenizer.encode_conversation(conv)
                    stats['token_lengths'].append(len(tokens))
                    stats['valid_conversations'] += 1
                except Exception as e:
                    stats['errors'].append(f"Line {line_no}: Tokenization error: {e}")
                    
            except json.JSONDecodeError as e:
                stats['errors'].append(f"Line {line_no}: JSON decode error: {e}")
    
    # Calculate statistics
    if stats['token_lengths']:
        stats['avg_token_length'] = np.mean(stats['token_lengths'])
        stats['max_token_length'] = max(stats['token_lengths'])
        stats['min_token_length'] = min(stats['token_lengths'])
    
    if stats['message_counts']:
        stats['avg_message_count'] = np.mean(stats['message_counts'])
    
    return stats

# =============================================================================
# MAIN TRAINING SCRIPT
# =============================================================================

def setup_logging(log_file: str = "training.log"):
    """Setup comprehensive logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)8s | %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Log system info
    logging.info("=" * 80)
    logging.info("PRODUCTION CONVERSATIONAL TRANSFORMER TRAINING")
    logging.info("=" * 80)
    
    if torch.cuda.is_available():
        logging.info(f"CUDA Device: {torch.cuda.get_device_name()}")
        logging.info(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        logging.info(f"CUDA Compute Capability: {torch.cuda.get_device_capability()}")
    else:
        logging.info("Running on CPU")
    
    logging.info(f"PyTorch version: {torch.__version__}")
    logging.info(f"Flash Attention available: {HAS_FLASH_ATTN}")

def estimate_parameters(config: Config) -> int:
    """Estimate total model parameters."""
    # Embedding parameters
    embed_params = config.vocab_size * config.hidden_size
    
    # Transformer block parameters
    # Attention: Q, K, V projections + output projection
    attn_params = (
        config.hidden_size * config.hidden_size +  # Q
        config.hidden_size * (config.hidden_size * config.num_kv_heads // config.num_heads) * 2 +  # K, V
        config.hidden_size * config.hidden_size  # output
    )
    
    # MLP parameters
    mlp_params = (
        config.hidden_size * config.intermediate_size * 2 +  # gate + up
        config.intermediate_size * config.hidden_size  # down
    )
    
    # Layer norm parameters
    norm_params = config.hidden_size * 2  # input + post-attn norm per layer
    
    # Total per layer
    layer_params = attn_params + mlp_params + norm_params
    
    # Total model
    total_params = (
        embed_params +  # embedding (shared with lm_head)
        layer_params * config.num_layers +  # all transformer layers
        config.hidden_size  # final norm
    )
    
    return total_params

def estimate_training_time(steps: int, config: Config, device_name: str) -> str:
    """Estimate training time based on hardware."""
    # Rough estimates (tokens per second)
    performance_map = {
        'A100': 15000,
        'H100': 25000,
        '4090': 8000,
        '3090': 6000,
        'V100': 10000,
        'T4': 3000,
        'CPU': 100
    }
    
    # Find matching device
    tokens_per_sec = 5000  # Default estimate
    for device, perf in performance_map.items():
        if device in device_name:
            tokens_per_sec = perf
            break
    
    # Calculate time
    total_tokens = steps * config.effective_batch_size * config.seq_length
    total_seconds = total_tokens / tokens_per_sec
    
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    
    return f"{hours}h {minutes}m"

def main():
    """Main training function - no command line arguments needed."""
    
    # Setup logging
    setup_logging()
    
    # Configuration setup - modify these as needed
    USE_CONFIG = "small"  # Options: "debug", "small", "medium", "large"
    CREATE_SAMPLE_DATA = True  # Set to True to create sample data
    VALIDATE_DATA = True  # Set to True to validate data before training
    RESUME_CHECKPOINT = None  # Path to checkpoint if resuming
    
    # Data paths
    TRAIN_DATA_PATH = "oasst1_data/train.jsonl"
    EVAL_DATA_PATH = "oasst1_data/validation.jsonl"
    
    # Create sample data if requested and doesn't exist
    if CREATE_SAMPLE_DATA:
        if not os.path.exists(TRAIN_DATA_PATH):
            logging.info("Creating sample training data...")
            create_sample_data(TRAIN_DATA_PATH, 5000)
        
        if not os.path.exists(EVAL_DATA_PATH):
            logging.info("Creating sample evaluation data...")
            create_sample_data(EVAL_DATA_PATH, 500)
        
        if not os.path.exists(TRAIN_DATA_PATH):
            logging.error("Failed to create sample data!")
            return 1
    
    # Get configuration
    config_map = {
        "debug": ConfigPresets.debug,
        "small": ConfigPresets.small,
        "medium": ConfigPresets.medium,
        "large": ConfigPresets.large
    }
    
    config = config_map[USE_CONFIG]()
    config.train_data_path = TRAIN_DATA_PATH
    config.eval_data_path = EVAL_DATA_PATH
    
    logging.info(f"Using {USE_CONFIG} configuration")
    logging.info(f"Model parameters: ~{estimate_parameters(config):,}")
    
    # Initialize tokenizer
    tokenizer = ConversationTokenizer("gpt2")
    config.vocab_size = tokenizer.vocab_size
    
    logging.info(f"Tokenizer initialized (vocab_size={tokenizer.vocab_size})")
    
    # Validate data if requested
    if VALIDATE_DATA:
        logging.info("Validating training data...")
        stats = validate_data(config.train_data_path, tokenizer)
        
        logging.info(f"Validation results:")
        logging.info(f"  Total conversations: {stats['total_conversations']:,}")
        logging.info(f"  Valid conversations: {stats['valid_conversations']:,}")
        
        if stats.get('avg_token_length'):
            logging.info(f"  Avg token length: {stats['avg_token_length']:.1f}")
            logging.info(f"  Token length range: {stats['min_token_length']}-{stats['max_token_length']}")
        
        if stats['errors']:
            logging.warning(f"Found {len(stats['errors'])} errors:")
            for error in stats['errors'][:10]:  # Show first 10 errors
                logging.warning(f"  {error}")
        
        if stats['valid_conversations'] == 0:
            logging.error("No valid conversations found! Check your data format.")
            return 1
    
    # Create datasets
    logging.info("Creating datasets...")
    train_dataset = ConversationDataset(config.train_data_path, tokenizer, config, "train")
    
    eval_dataset = None
    if os.path.exists(config.eval_data_path):
        eval_dataset = ConversationDataset(config.eval_data_path, tokenizer, config, "eval")
    
    # Create dataloaders
    train_dataloader = create_dataloader(train_dataset, config, shuffle=True)
    eval_dataloader = create_dataloader(eval_dataset, config, shuffle=False) if eval_dataset else None
    
    # Initialize model
    logging.info("Initializing model...")
    model = TransformerModel(config)
    
    # Initialize trainer
    trainer = ConversationTrainer(model, tokenizer, config)
    
    # Resume from checkpoint if requested
    start_step = 0
    if RESUME_CHECKPOINT and os.path.exists(RESUME_CHECKPOINT):
        start_step = trainer.load_checkpoint(RESUME_CHECKPOINT)
    
    # Log training info
    logging.info("Training configuration:")
    logging.info(f"  Max steps: {config.max_steps:,}")
    logging.info(f"  Effective batch size: {config.effective_batch_size:,}")
    logging.info(f"  Learning rate: {config.learning_rate:.2e}")
    logging.info(f"  Precision: {config.precision}")
    logging.info(f"  Gradient accumulation steps: {config.gradient_accumulation_steps}")
    
    # Estimate training time
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name()
    else:
        device_name = "CPU"
    
    estimated_time = estimate_training_time(config.max_steps - start_step, config, device_name)
    logging.info(f"Estimated training time: {estimated_time}")
    
    # Start training
    logging.info("Starting training...")
    try:
        trainer.train(train_dataloader, eval_dataloader)
    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
        trainer.save_checkpoint(trainer.global_step, final=True)
    except Exception as e:
        logging.error(f"Training failed: {e}")
        return 1
    
    # Test generation
    logging.info("Testing generation...")
    test_prompts = [
        "How do I learn Python programming?",
        "Explain machine learning in simple terms.",
        "What are the benefits of exercise?",
    ]
    
    for prompt in test_prompts:
        logging.info(f"\nPrompt: {prompt}")
        response = trainer.generate(prompt, max_new_tokens=128)
        logging.info(f"Response: {response}")
    
    logging.info("Training completed successfully!")
    return 0

if __name__ == "__main__":
    exit(main())