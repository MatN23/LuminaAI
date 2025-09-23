# Copyright (c) 2025 MatN23. All rights reserved.
# Licensed under the Custom License below.

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
    
    # MLP or MoE layers
    if hasattr(config, 'use_moe') and config.use_moe:
        # MoE parameters
        gate_params = config.hidden_size  # gating network
        expert_params = (
            config.hidden_size * config.intermediate_size * 2 +  # gate and up
            config.intermediate_size * config.hidden_size  # down
        ) * config.num_experts
        mlp_params_per_layer = gate_params + expert_params
    else:
        # Standard MLP
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
        if HAS_FLASH_ATTN and L > 1024 and x.dtype in [torch.float16, torch.bfloat16]:
            try:
                # Ensure correct shapes for flash attention [B, L, H, D]
                q_flash = q.transpose(1, 2).contiguous()  # [B, L, H, D]
                k_flash = k.transpose(1, 2).contiguous()  # [B, L, H, D]
                v_flash = v.transpose(1, 2).contiguous()  # [B, L, H, D]
                
                out = flash_attn_func(q_flash, k_flash, v_flash, causal=True, softmax_scale=self.scale)
                out = out.reshape(B, L, self.hidden_size)
                
            except Exception as e:
                logging.warning(f"Flash attention failed, falling back to standard: {e}")
                # Fall back to standard attention with correct shapes
                out = self._standard_attention(q, k, v, attention_mask)
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


class SwiGLUExpert(nn.Module):
    """Single SwiGLU expert for MoE."""
    
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class OptimizedMoELayer(nn.Module):
    """FIXED: Production-ready MoE with stable routing and load balancing."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.top_k = config.moe_top_k
        self.hidden_size = config.hidden_size
        self.capacity_factor = getattr(config, 'capacity_factor', 1.5)  # FIXED: Increased default
        
        # Gating network
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        
        # Expert networks - use ModuleList for proper device handling
        self.experts = nn.ModuleList([SwiGLUExpert(config) for _ in range(config.num_experts)])
        
        # Load balancing - FIXED: Much lower default weight
        self.load_balancing_weight = getattr(config, 'load_balancing_weight', 0.001)
        
        # FIXED: Add routing temperature for stability
        self.routing_temperature = getattr(config, 'routing_temperature', 1.0)
        
        # FIXED: Add noise for load balancing
        self.noise_std = getattr(config, 'routing_noise_std', 0.1)
        
        self._init_weights()
    
    def _init_weights(self):
        """FIXED: Better initialization for routing stability."""
        std = self.config.init_std
        
        # FIXED: Proper gate initialization (was too small at 0.001)
        nn.init.normal_(self.gate.weight, mean=0.0, std=0.02)
        
        # Initialize expert weights
        for expert in self.experts:
            nn.init.normal_(expert.gate_proj.weight, mean=0.0, std=std)
            nn.init.normal_(expert.up_proj.weight, mean=0.0, std=std)
            nn.init.normal_(expert.down_proj.weight, mean=0.0, std=std / math.sqrt(2 * self.config.num_layers))
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, hidden_size = x.shape
        total_tokens = batch_size * seq_len
        x_flat = x.view(-1, hidden_size)  # [total_tokens, hidden_size]
        
        # FIXED: Add training noise for better load balancing
        gate_logits = self.gate(x_flat)  # [total_tokens, num_experts]
        
        if self.training and self.noise_std > 0:
            # Add noise during training to encourage exploration
            noise = torch.randn_like(gate_logits) * self.noise_std
            gate_logits = gate_logits + noise
        
        # FIXED: Apply temperature scaling for more stable routing
        gate_logits = gate_logits / self.routing_temperature
        gate_probs = F.softmax(gate_logits, dim=-1)
        
        # Top-k gating with capacity enforcement
        top_k_probs, top_k_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        
        # Normalize top-k probabilities
        top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-8)
        
        # FIXED: More conservative capacity calculation
        base_capacity = total_tokens * self.top_k // self.num_experts
        expert_capacity = max(8, int(base_capacity * self.capacity_factor))  # Minimum 8 tokens
        
        # Initialize output and tracking tensors
        output = torch.zeros_like(x_flat)
        expert_counts = torch.zeros(self.num_experts, dtype=torch.long, device=x.device)
        dropped_tokens = 0  # Track dropped tokens for debugging
        
        # FIXED: More stable scatter/gather routing
        for k in range(self.top_k):
            # Get k-th choice for each token
            expert_ids = top_k_indices[:, k]  # [total_tokens]
            weights = top_k_probs[:, k]  # [total_tokens]
            
            # Process each expert in parallel
            for expert_id in range(self.num_experts):
                # Find tokens assigned to this expert
                mask = (expert_ids == expert_id)
                
                if not mask.any():
                    continue
                
                # FIXED: More stable capacity handling
                token_indices = torch.where(mask)[0]
                if len(token_indices) > expert_capacity:
                    # FIXED: Keep first tokens instead of random sampling
                    # This provides more stable training
                    token_indices = token_indices[:expert_capacity]
                    dropped_tokens += len(torch.where(mask)[0]) - expert_capacity
                    
                    # Update mask for consistency
                    new_mask = torch.zeros_like(mask)
                    new_mask[token_indices] = True
                    mask = new_mask
                
                if mask.any():
                    # Gather tokens for this expert
                    expert_input = x_flat[mask]  # [n_tokens_for_expert, hidden_size]
                    expert_weights = weights[mask]  # [n_tokens_for_expert]
                    
                    # Forward through expert
                    expert_output = self.experts[expert_id](expert_input)  # [n_tokens_for_expert, hidden_size]
                    
                    # FIXED: Apply weights more carefully to avoid NaN
                    expert_weights = torch.clamp(expert_weights, min=1e-8, max=10.0)
                    weighted_output = expert_output * expert_weights.unsqueeze(-1)
                    output[mask] += weighted_output
                    
                    # Update expert usage count
                    expert_counts[expert_id] += mask.sum()
        
        # FIXED: More stable load balancing loss computation
        load_balancing_loss = self._compute_stable_load_balancing_loss(
            gate_probs, expert_counts, total_tokens, dropped_tokens
        )
        
        # Reshape output
        output = output.view(batch_size, seq_len, hidden_size)
        
        return output, load_balancing_loss
    
    def _compute_stable_load_balancing_loss(self, 
                                          gate_probs: torch.Tensor,
                                          expert_counts: torch.Tensor,
                                          total_tokens: int,
                                          dropped_tokens: int) -> torch.Tensor:
        """FIXED: More stable load balancing loss computation."""
        
        # Expert usage fractions (how often each expert is used)
        expert_usage = expert_counts.float() / max(total_tokens * self.top_k, 1)
        
        # Gate importance (average probability assigned to each expert)
        gate_importance = gate_probs.mean(dim=0)  # [num_experts]
        
        # FIXED: More stable loss formulation
        # Standard Switch Transformer auxiliary loss
        aux_loss = torch.sum(expert_usage * gate_importance) * self.num_experts
        
        # FIXED: Add penalty for dropped tokens to encourage better load balancing
        if dropped_tokens > 0:
            drop_penalty = (dropped_tokens / max(total_tokens, 1)) * 0.1
            aux_loss = aux_loss + drop_penalty
        
        # FIXED: Clamp the loss to prevent explosion
        aux_loss = torch.clamp(aux_loss, max=1.0)
        
        return aux_loss * self.load_balancing_weight


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
    """Enhanced transformer block with optimized MoE support."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = GroupedQueryAttention(config)
        self.post_attn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        
        # MoE or standard MLP
        if hasattr(config, 'use_moe') and config.use_moe:
            self.mlp = OptimizedMoELayer(config)
            self.use_moe = True
        else:
            self.mlp = SwiGLU(config)
            self.use_moe = False
        
        # Gradient checkpointing flag
        self.gradient_checkpointing = config.gradient_checkpointing
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self.gradient_checkpointing and self.training:
            return self._forward_with_checkpointing(x, attention_mask)
        else:
            return self._forward_impl(x, attention_mask)
    
    def _forward_impl(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward implementation with residual connections."""
        # Self-attention with pre-norm
        attn_out = self.self_attn(self.input_norm(x), attention_mask)
        x = x + attn_out
        
        # MLP/MoE with pre-norm
        if self.use_moe:
            mlp_out, load_balancing_loss = self.mlp(self.post_attn_norm(x))
            x = x + mlp_out
            return x, load_balancing_loss
        else:
            mlp_out = self.mlp(self.post_attn_norm(x))
            x = x + mlp_out
            return x
    
    def _forward_with_checkpointing(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward with gradient checkpointing."""
        return torch.utils.checkpoint.checkpoint(
            self._forward_impl, x, attention_mask, use_reentrant=False
        )


class DeepSeekTransformer(nn.Module):
    """Production-ready DeepSeek-style transformer with FIXED MoE."""
    
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
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        
        # MoE tracking
        self.use_moe = hasattr(config, 'use_moe') and config.use_moe
        
        # Initialize weights
        self._init_weights()
        
        # Parameter counting and logging
        n_params = sum(p.numel() for p in self.parameters())
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logging.info(f"DeepSeek model initialized with {n_params:,} total parameters ({n_trainable:,} trainable)")
        
        if self.use_moe:
            # Calculate active parameters for MoE
            n_active = self._calculate_active_params()
            efficiency = (n_active / n_params) * 100
            logging.info(f"MoE model with {n_active:,} active parameters per forward pass ({efficiency:.1f}% efficiency)")
    
    def _calculate_active_params(self) -> int:
        """Calculate active parameters for MoE model."""
        if not self.use_moe:
            return sum(p.numel() for p in self.parameters())
        
        # Non-MoE parameters
        active_params = (
            self.embed_tokens.weight.numel() +
            self.norm.weight.numel()
        )
        
        # Add LM head if not tied
        if not self.config.tie_word_embeddings:
            active_params += self.lm_head.weight.numel()
        
        # Per-layer active parameters
        for layer in self.layers:
            # Attention parameters (always active)
            active_params += sum(p.numel() for p in layer.self_attn.parameters())
            active_params += layer.input_norm.weight.numel()
            active_params += layer.post_attn_norm.weight.numel()
            
            # MoE parameters (only top-k experts active)
            if hasattr(layer.mlp, 'gate'):
                active_params += layer.mlp.gate.weight.numel()  # Gate always active
                # Top-k experts out of total experts
                expert_params = sum(p.numel() for p in layer.mlp.experts[0].parameters())
                active_params += expert_params * self.config.moe_top_k
        
        return active_params
    
    def _init_weights(self):
        """FIXED: Enhanced weight initialization following DeepSeek principles."""
        # Embedding initialization
        nn.init.normal_(self.embed_tokens.weight, mean=0.0, std=self.config.init_std)
        
        # LM head initialization (if not tied)
        if not self.config.tie_word_embeddings:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=self.config.init_std)
        
        # FIXED: Less aggressive residual scaling for MoE stability
        with torch.no_grad():
            for layer_idx, layer in enumerate(self.layers):
                # Progressive scaling: deeper layers get smaller initialization
                depth_scale = 1.0 / math.sqrt((layer_idx + 1) * 2)
                
                # Scale attention output projection
                layer.self_attn.o_proj.weight.data *= 0.8 * depth_scale  # Less aggressive
                
                # FIXED: Different scaling for MoE vs standard MLP
                if hasattr(layer.mlp, 'down_proj'):  # Standard MLP
                    layer.mlp.down_proj.weight.data *= 0.8 * depth_scale
                elif hasattr(layer.mlp, 'experts'):  # MoE - FIXED: Less aggressive scaling
                    for expert in layer.mlp.experts:
                        expert.down_proj.weight.data *= 0.9  # Much less aggressive for MoE
    
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None,
                return_hidden_states: bool = False,
                return_aux_loss: bool = True) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Enhanced forward pass with comprehensive output options."""
        # Input validation
        if torch.any(input_ids >= self.config.vocab_size) or torch.any(input_ids < 0):
            logging.warning("Input contains invalid token IDs, clamping...")
            input_ids = torch.clamp(input_ids, 0, self.config.vocab_size - 1)
        
        # Embedding
        x = self.embed_tokens(input_ids) * self.embed_scale
        
        # Store hidden states if requested
        hidden_states = [] if return_hidden_states else None
        
        # Track MoE load balancing losses
        total_aux_loss = 0.0
        aux_losses = []
        
        # Transformer layers
        for layer in self.layers:
            if self.use_moe and hasattr(layer.mlp, 'gate'):
                result = layer(x, attention_mask)
                if isinstance(result, tuple):
                    x, aux_loss = result
                    # FIXED: Clamp aux loss to prevent explosion
                    aux_loss = torch.clamp(aux_loss, max=1.0)
                    total_aux_loss += aux_loss
                    if return_aux_loss:
                        aux_losses.append(aux_loss)
                else:
                    x = result
            else:
                x = layer(x, attention_mask)
            
            if return_hidden_states:
                hidden_states.append(x)
        
        # Final layer norm
        x = self.norm(x)
        
        # Language modeling head
        logits = self.lm_head(x)
        
        # Return appropriate format based on requested outputs
        outputs = [logits]
        
        if return_hidden_states:
            outputs.append(hidden_states)
        
        if self.use_moe and return_aux_loss:
            outputs.append(total_aux_loss)
            outputs.append(aux_losses)
        
        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)
    
    def get_num_params(self, non_embedding: bool = True) -> int:
        """Get parameter count."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.embed_tokens.weight.numel()
            # Subtract LM head if tied to embeddings
            if self.config.tie_word_embeddings:
                n_params -= self.lm_head.weight.numel()
        return n_params
    
    def get_memory_footprint(self) -> dict:
        """Get detailed memory footprint analysis."""
        total_params = sum(p.numel() for p in self.parameters())
        total_size = sum(p.numel() * p.element_size() for p in self.parameters())
        
        breakdown = {
            'total_parameters': total_params,
            'total_size_mb': total_size / (1024 * 1024),
            'embedding_params': self.embed_tokens.weight.numel(),
            'attention_params': sum(p.numel() for layer in self.layers for p in layer.self_attn.parameters()),
            'mlp_params': sum(p.numel() for layer in self.layers for p in layer.mlp.parameters()),
            'norm_params': sum(p.numel() for layer in self.layers for p in [layer.input_norm, layer.post_attn_norm]) + self.norm.weight.numel(),
        }
        
        if self.use_moe:
            # MoE-specific breakdown
            total_expert_params = sum(
                sum(p.numel() for expert in layer.mlp.experts for p in expert.parameters())
                for layer in self.layers if hasattr(layer.mlp, 'experts')
            )
            active_expert_params = total_expert_params * (self.config.moe_top_k / self.config.num_experts)
            
            breakdown.update({
                'expert_params_total': total_expert_params,
                'expert_params_active': int(active_expert_params),
                'gate_params': sum(layer.mlp.gate.weight.numel() for layer in self.layers if hasattr(layer.mlp, 'gate')),
                'parameter_efficiency': (breakdown['total_parameters'] - total_expert_params + active_expert_params) / breakdown['total_parameters']
            })
        
        return breakdown
    
    def estimate_flops(self, seq_len: int, batch_size: int = 1) -> dict:
        """Estimate FLOPs for forward pass."""
        hidden_size = self.config.hidden_size
        vocab_size = self.config.vocab_size
        num_layers = self.config.num_layers
        intermediate_size = self.config.intermediate_size
        
        # Embedding lookup: negligible FLOPs
        embedding_flops = 0
        
        # Per-layer FLOPs
        attention_flops_per_layer = (
            # QKV projections
            3 * seq_len * hidden_size * hidden_size +
            # Attention computation
            2 * self.config.num_heads * seq_len * seq_len * (hidden_size // self.config.num_heads) +
            # Output projection
            seq_len * hidden_size * hidden_size
        )
        
        if self.use_moe:
            # MoE FLOPs (only active experts)
            mlp_flops_per_layer = (
                # Gate computation
                seq_len * hidden_size * self.config.num_experts +
                # Active experts only
                seq_len * hidden_size * intermediate_size * 3 * self.config.moe_top_k  # gate, up, down projections
            )
        else:
            # Standard MLP FLOPs
            mlp_flops_per_layer = seq_len * hidden_size * intermediate_size * 3  # gate, up, down
        
        # Layer norm FLOPs (negligible)
        norm_flops_per_layer = seq_len * hidden_size * 4  # 2 layer norms + final norm
        
        total_layer_flops = num_layers * (attention_flops_per_layer + mlp_flops_per_layer + norm_flops_per_layer)
        
        # LM head FLOPs
        lm_head_flops = seq_len * hidden_size * vocab_size
        
        total_flops = batch_size * (embedding_flops + total_layer_flops + lm_head_flops)
        
        return {
            'total_flops': total_flops,
            'flops_per_token': total_flops / (batch_size * seq_len),
            'attention_flops': batch_size * num_layers * attention_flops_per_layer,
            'mlp_flops': batch_size * num_layers * mlp_flops_per_layer,
            'lm_head_flops': batch_size * lm_head_flops,
        }


# Configuration class for easy model setup
class DeepSeekConfig:
    """Configuration class for DeepSeek-style transformer."""
    
    def __init__(
        self,
        vocab_size: int = 50257,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        num_kv_heads: Optional[int] = None,
        intermediate_size: Optional[int] = None,
        seq_length: int = 2048,
        dropout: float = 0.0,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 10000.0,
        init_std: float = 0.02,
        use_stable_embedding: bool = True,
        tie_word_embeddings: bool = True,
        gradient_checkpointing: bool = False,
        # MoE parameters - FIXED defaults
        use_moe: bool = False,
        num_experts: int = 8,
        moe_top_k: int = 1,  # FIXED: Default to top-1
        capacity_factor: float = 1.5,  # FIXED: Increased default
        load_balancing_weight: float = 0.001,  # FIXED: Much lower default
        routing_temperature: float = 1.0,  # FIXED: Add temperature control
        routing_noise_std: float = 0.1,  # FIXED: Add noise for load balancing
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.intermediate_size = intermediate_size if intermediate_size is not None else 4 * hidden_size
        self.seq_length = seq_length
        self.dropout = dropout
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.init_std = init_std
        self.use_stable_embedding = use_stable_embedding
        self.tie_word_embeddings = tie_word_embeddings
        self.gradient_checkpointing = gradient_checkpointing
        
        # MoE configuration - FIXED
        self.use_moe = use_moe
        self.num_experts = num_experts
        self.moe_top_k = moe_top_k
        self.capacity_factor = capacity_factor
        self.load_balancing_weight = load_balancing_weight
        self.routing_temperature = routing_temperature
        self.routing_noise_std = routing_noise_std
        
        # Validation
        assert self.num_heads % self.num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
        if self.use_moe:
            assert self.moe_top_k <= self.num_experts, "top_k must be <= num_experts"
            assert self.moe_top_k >= 1, "top_k must be >= 1"
        
        # Store any additional kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    @classmethod
    def small(cls, **kwargs):
        """Small model configuration (GPT-2 Small scale) with FIXED MoE."""
        defaults = {
            'hidden_size': 768,
            'num_layers': 12,
            'num_heads': 12,
            'num_kv_heads': 4,  # GQA
            'vocab_size': 50257,
            'use_moe': True,
            'num_experts': 8,
            'moe_top_k': 1,  # FIXED: Top-1 routing
            'capacity_factor': 1.5,  # FIXED: Higher capacity
            'load_balancing_weight': 0.001,  # FIXED: Lower weight
        }
        defaults.update(kwargs)
        return cls(**defaults)
    
    @classmethod
    def medium(cls, **kwargs):
        """Medium model configuration (GPT-2 Medium scale) with FIXED MoE."""
        defaults = {
            'hidden_size': 1024,
            'num_layers': 24,
            'num_heads': 16,
            'num_kv_heads': 4,  # GQA
            'vocab_size': 50257,
            'use_moe': True,
            'num_experts': 8,
            'moe_top_k': 1,  # FIXED: Top-1 routing
            'capacity_factor': 1.5,  # FIXED: Higher capacity
            'load_balancing_weight': 0.001,  # FIXED: Lower weight
        }
        defaults.update(kwargs)
        return cls(**defaults)
    
    @classmethod
    def large(cls, **kwargs):
        """Large model configuration (GPT-2 Large scale) with FIXED MoE."""
        defaults = {
            'hidden_size': 1280,
            'num_layers': 36,
            'num_heads': 20,
            'num_kv_heads': 4,  # GQA
            'vocab_size': 50257,
            'use_moe': True,
            'num_experts': 8,
            'moe_top_k': 1,  # FIXED: Top-1 routing
            'capacity_factor': 1.5,  # FIXED: Higher capacity
            'load_balancing_weight': 0.001,  # FIXED: Lower weight
        }
        defaults.update(kwargs)
        return cls(**defaults)
    
    @classmethod
    def moe_small(cls, **kwargs):
        """Small MoE model configuration with FIXED settings."""
        defaults = {
            'hidden_size': 768,
            'num_layers': 12,
            'num_heads': 12,
            'num_kv_heads': 4,
            'vocab_size': 50257,
            'use_moe': True,
            'num_experts': 8,
            'moe_top_k': 1,  # FIXED: Top-1 routing for stability
            'capacity_factor': 1.5,  # FIXED: Higher capacity
            'load_balancing_weight': 0.001,  # FIXED: Much lower weight
            'routing_temperature': 1.0,  # FIXED: Add temperature
            'routing_noise_std': 0.1,  # FIXED: Add noise
        }
        defaults.update(kwargs)
        return cls(**defaults)
    
    @classmethod
    def moe_large(cls, **kwargs):
        """Large MoE model configuration with FIXED settings."""
        defaults = {
            'hidden_size': 1280,
            'num_layers': 36,
            'num_heads': 20,
            'num_kv_heads': 4,
            'vocab_size': 50257,
            'use_moe': True,
            'num_experts': 8,
            'moe_top_k': 1,  # FIXED: Top-1 routing for stability
            'capacity_factor': 1.25,  # FIXED: Slightly lower for large models
            'load_balancing_weight': 0.0005,  # FIXED: Even lower for large models
            'routing_temperature': 0.8,  # FIXED: Lower temperature for more focused routing
            'routing_noise_std': 0.05,  # FIXED: Less noise for large models
        }
        defaults.update(kwargs)
        return cls(**defaults)


# Factory function for easy model creation
def create_deepseek_model(config_name: str = 'small', **kwargs) -> DeepSeekTransformer:
    """Factory function to create DeepSeek models with predefined configurations."""
    config_mapping = {
        'small': DeepSeekConfig.small,
        'medium': DeepSeekConfig.medium,
        'large': DeepSeekConfig.large,
        'moe_small': DeepSeekConfig.moe_small,
        'moe_large': DeepSeekConfig.moe_large,
    }
    
    if config_name not in config_mapping:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(config_mapping.keys())}")
    
    config = config_mapping[config_name](**kwargs)
    return DeepSeekTransformer(config)


# FIXED: Add debugging utilities
def debug_moe_routing(model, input_ids, attention_mask=None):
    """Debug MoE routing to identify issues."""
    model.eval()
    routing_info = []
    
    def capture_routing(name):
        def hook(module, input, output):
            if hasattr(module, 'gate'):
                x_flat = input[0].view(-1, input[0].size(-1))
                gate_logits = module.gate(x_flat)
                gate_probs = F.softmax(gate_logits, dim=-1)
                
                # Calculate routing statistics
                entropy = -torch.sum(gate_probs * torch.log(gate_probs + 1e-8), dim=-1).mean()
                expert_usage = gate_probs.sum(dim=0) / gate_probs.sum()
                max_prob = gate_probs.max(dim=-1)[0].mean()
                
                routing_info.append({
                    'layer': name,
                    'entropy': entropy.item(),
                    'expert_usage': expert_usage.cpu().numpy(),
                    'max_routing_prob': max_prob.item(),
                    'usage_std': expert_usage.std().item()
                })
        return hook
    
    # Register hooks
    hooks = []
    for name, module in model.named_modules():
        if hasattr(module, 'gate'):
            hooks.append(module.register_forward_hook(capture_routing(name)))
    
    # Forward pass
    try:
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
    finally:
        # Remove hooks
        for hook in hooks:
            hook.remove()
    
    # Print routing analysis
    print("=== MoE Routing Analysis ===")
    for info in routing_info:
        print(f"Layer {info['layer']}:")
        print(f"  Entropy: {info['entropy']:.3f} (higher = more balanced)")
        print(f"  Max routing prob: {info['max_routing_prob']:.3f}")
        print(f"  Usage std: {info['usage_std']:.3f} (lower = more balanced)")
        print(f"  Expert usage: {[f'{u:.3f}' for u in info['expert_usage']]}")
        
        if info['entropy'] < 1.0:
            print(f"  ⚠️ Very low entropy - routing is deterministic")
        if info['usage_std'] > 0.3:
            print(f"  ⚠️ High usage variation - experts are imbalanced")
        print()
    
    return routing_info


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Test FIXED MoE configurations
    print("=== FIXED DeepSeek-Style Transformer with Stable MoE ===\n")
    
    # Test FIXED MoE model
    print("1. FIXED MoE Model:")
    model_moe = create_deepseek_model('moe_small')
    print(f"Parameters: {model_moe.get_num_params():,}")
    memory_info = model_moe.get_memory_footprint()
    print(f"Memory footprint: {memory_info}")
    if 'parameter_efficiency' in memory_info:
        print(f"Parameter efficiency: {memory_info['parameter_efficiency']:.2%}")
    
    # Test forward pass with debugging
    batch_size, seq_len = 2, 128
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
    print("\n2. Forward Pass Test:")
    with torch.no_grad():
        outputs = model_moe(input_ids, return_aux_loss=True)
        if isinstance(outputs, tuple):
            logits_moe, aux_loss = outputs[:2]
            print(f"Output shape: {logits_moe.shape}")
            print(f"Auxiliary loss: {aux_loss:.6f}")
            
            # Check if aux loss is reasonable
            if aux_loss > 1.0:
                print("⚠️ High auxiliary loss - might cause instability")
            elif aux_loss < 1e-6:
                print("⚠️ Very low auxiliary loss - load balancing might not be working")
            else:
                print("✅ Auxiliary loss in reasonable range")
        else:
            print(f"Output shape: {outputs.shape}")
    
    # Test routing debugging
    print("\n3. Routing Analysis:")
    routing_stats = debug_moe_routing(model_moe, input_ids)
    
    print("\n=== FIXED MoE Test Complete ===")
    print("\nKey Fixes Applied:")
    print("  ✅ Gate initialization: std=0.02 (was 0.001)")
    print("  ✅ Load balancing weight: 0.001 (was 0.075)")
    print("  ✅ Capacity factor: 1.5 (was 1.25)")
    print("  ✅ Top-1 routing (was top-2)")
    print("  ✅ Stable token capacity handling")
    print("  ✅ Auxiliary loss clamping")
    print("  ✅ Less aggressive weight scaling")
    print("  ✅ Added routing temperature and noise")