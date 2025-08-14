"""
Model Architecture Module
Contains the transformer model implementation with enhanced stability features.
"""

import math
import logging
from typing import Optional, Tuple, Union, List, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional flash attention
try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False


def estimate_parameters(config) -> int:
    """Estimate number of model parameters."""
    # Embedding
    embed_params = config.vocab_size * config.hidden_size
    
    # Attention layers
    attn_params_per_layer = (
        config.hidden_size * config.hidden_size * 3 +  # q, k, v projections
        config.hidden_size * config.hidden_size  # output projection
    )
    
    # MLP layers  
    mlp_params_per_layer = (
        config.hidden_size * config.intermediate_size * 2 +  # gate and up
        config.intermediate_size * config.hidden_size  # down
    )
    
    # Layer norms
    norm_params_per_layer = config.hidden_size * 2  # input and post-attn norms
    
    # Total per layer
    params_per_layer = attn_params_per_layer + mlp_params_per_layer + norm_params_per_layer
    
    # Final norm and LM head (tied with embedding)
    final_norm = config.hidden_size
    
    total_params = embed_params + (params_per_layer * config.num_layers) + final_norm
    
    return total_params


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
    
    def __init__(self, config):
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
    
    def __init__(self, config):
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
    
    def __init__(self, config):
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
    
    def __init__(self, config):
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