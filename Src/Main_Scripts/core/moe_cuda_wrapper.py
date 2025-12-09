"""
Python wrapper for CUDA MoE operations.

This provides a clean interface to the CUDA kernels with:
- Automatic fallback to PyTorch implementations
- Input validation and error handling
- Benchmarking utilities
- Easy integration with existing PyTorch code
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional
import warnings

# Try to import CUDA extension
try:
    import core.moe_cuda_ops
    CUDA_OPS_AVAILABLE = True
except ImportError:
    CUDA_OPS_AVAILABLE = False
    warnings.warn(
        "CUDA MoE operations not available. Install with: python setup.py install\n"
        "Falling back to PyTorch implementations (slower)."
    )


class MoECUDAOps:
    """
    High-level interface for CUDA-accelerated MoE operations.
    
    This class provides methods that automatically use CUDA kernels when
    available, with fallback to PyTorch implementations.
    """
    
    @staticmethod
    def topk_gating(
        gate_logits: torch.Tensor, 
        k: int, 
        temperature: float = 1.0,
        use_cuda: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform top-k gating with temperature scaling.
        
        Args:
            gate_logits: [num_tokens, num_experts] raw logits
            k: number of experts to select
            temperature: temperature for softmax (lower = more peaked)
            use_cuda: whether to use CUDA kernel (if available)
            
        Returns:
            top_k_indices: [num_tokens, k] selected expert indices
            top_k_weights: [num_tokens, k] normalized routing weights
        """
        if use_cuda and CUDA_OPS_AVAILABLE and gate_logits.is_cuda:
            return moe_cuda_ops.topk_gating(gate_logits, k, temperature)
        else:
            # PyTorch fallback
            scaled_logits = gate_logits / temperature
            top_k_values, top_k_indices = torch.topk(scaled_logits, k, dim=-1)
            top_k_weights = F.softmax(top_k_values, dim=-1)
            return top_k_indices, top_k_weights
    
    @staticmethod
    def compute_expert_capacity(
        top_k_indices: torch.Tensor,
        num_experts: int,
        use_cuda: bool = True
    ) -> torch.Tensor:
        """
        Compute how many tokens are assigned to each expert.
        
        Args:
            top_k_indices: [num_tokens, k] expert assignments
            num_experts: total number of experts
            use_cuda: whether to use CUDA kernel
            
        Returns:
            expert_counts: [num_experts] tokens per expert
        """
        if use_cuda and CUDA_OPS_AVAILABLE and top_k_indices.is_cuda:
            return moe_cuda_ops.compute_expert_capacity(top_k_indices, num_experts)
        else:
            # PyTorch fallback
            expert_counts = torch.zeros(num_experts, dtype=torch.int32, device=top_k_indices.device)
            for k_idx in range(top_k_indices.size(1)):
                counts = torch.bincount(
                    top_k_indices[:, k_idx],
                    minlength=num_experts
                )
                expert_counts += counts.int()
            return expert_counts
    
    @staticmethod
    def dispatch_tokens(
        tokens: torch.Tensor,
        top_k_indices: torch.Tensor,
        num_experts: int,
        capacity: int,
        use_cuda: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Dispatch tokens to expert-specific buffers.
        
        Args:
            tokens: [num_tokens, hidden_dim] input tokens
            top_k_indices: [num_tokens, k] expert assignments
            num_experts: total number of experts
            capacity: maximum tokens per expert
            use_cuda: whether to use CUDA kernel
            
        Returns:
            expert_inputs: [num_experts, capacity, hidden_dim] batched expert inputs
            token_map: [num_experts, capacity] mapping back to original tokens
        """
        if use_cuda and CUDA_OPS_AVAILABLE and tokens.is_cuda:
            return moe_cuda_ops.dispatch_tokens(
                tokens, top_k_indices, num_experts, capacity
            )
        else:
            # PyTorch fallback
            num_tokens, hidden_dim = tokens.shape
            k = top_k_indices.size(1)
            
            expert_inputs = torch.zeros(
                num_experts, capacity, hidden_dim,
                dtype=tokens.dtype, device=tokens.device
            )
            token_map = torch.full(
                (num_experts, capacity), -1,
                dtype=torch.int32, device=tokens.device
            )
            expert_positions = torch.zeros(num_experts, dtype=torch.int32, device=tokens.device)
            
            for token_idx in range(num_tokens):
                for k_idx in range(k):
                    expert_id = top_k_indices[token_idx, k_idx].item()
                    pos = expert_positions[expert_id].item()
                    
                    if pos < capacity:
                        expert_inputs[expert_id, pos] = tokens[token_idx]
                        token_map[expert_id, pos] = token_idx * k + k_idx
                        expert_positions[expert_id] += 1
            
            return expert_inputs, token_map
    
    @staticmethod
    def combine_expert_outputs(
        expert_outputs: torch.Tensor,
        token_map: torch.Tensor,
        top_k_weights: torch.Tensor,
        num_tokens: int,
        k: int,
        use_cuda: bool = True
    ) -> torch.Tensor:
        """
        Combine expert outputs back to original token positions.
        
        Args:
            expert_outputs: [num_experts, capacity, hidden_dim] expert results
            token_map: [num_experts, capacity] mapping to original tokens
            top_k_weights: [num_tokens, k] routing weights
            num_tokens: number of original tokens
            k: experts per token
            use_cuda: whether to use CUDA kernel
            
        Returns:
            combined: [num_tokens, hidden_dim] combined outputs
        """
        if use_cuda and CUDA_OPS_AVAILABLE and expert_outputs.is_cuda:
            return moe_cuda_ops.combine_expert_outputs(
                expert_outputs, token_map, top_k_weights, num_tokens, k
            )
        else:
            # PyTorch fallback
            num_experts, capacity, hidden_dim = expert_outputs.shape
            combined = torch.zeros(
                num_tokens, hidden_dim,
                dtype=expert_outputs.dtype, device=expert_outputs.device
            )
            
            for expert_id in range(num_experts):
                for pos in range(capacity):
                    token_weight_idx = token_map[expert_id, pos].item()
                    if token_weight_idx < 0:
                        continue
                    
                    token_idx = token_weight_idx // k
                    weight_idx = token_weight_idx % k
                    
                    if token_idx >= num_tokens:
                        continue
                    
                    weight = top_k_weights[token_idx, weight_idx]
                    expert_out = expert_outputs[expert_id, pos]
                    combined[token_idx] += weight * expert_out
            
            return combined
    
    @staticmethod
    def compute_load_balancing_loss(
        gate_probs: torch.Tensor,
        top_k_indices: torch.Tensor,
        use_cuda: bool = True
    ) -> torch.Tensor:
        """
        Compute load balancing auxiliary loss.
        
        Args:
            gate_probs: [num_tokens, num_experts] softmax probabilities
            top_k_indices: [num_tokens, k] expert assignments
            use_cuda: whether to use CUDA kernel
            
        Returns:
            aux_loss: scalar load balancing loss
        """
        if use_cuda and CUDA_OPS_AVAILABLE and gate_probs.is_cuda:
            return moe_cuda_ops.compute_load_balancing_loss(gate_probs, top_k_indices)
        else:
            # PyTorch fallback
            num_tokens, num_experts = gate_probs.shape
            k = top_k_indices.size(1)
            
            # Compute expert usage
            expert_usage = torch.zeros(num_experts, device=gate_probs.device)
            for k_idx in range(k):
                counts = torch.bincount(
                    top_k_indices[:, k_idx],
                    minlength=num_experts
                )
                expert_usage += counts.float() / (num_tokens * k)
            
            # Compute gate importance
            gate_importance = gate_probs.mean(dim=0)
            
            # Compute loss
            aux_loss = torch.sum(expert_usage * gate_importance) * num_experts
            return aux_loss.unsqueeze(0)


def benchmark_moe_ops(
    num_tokens: int = 1024,
    hidden_dim: int = 768,
    num_experts: int = 8,
    k: int = 2,
    num_runs: int = 100,
    warmup: int = 10
):
    """
    Benchmark CUDA vs PyTorch implementations.
    
    Args:
        num_tokens: number of tokens to route
        hidden_dim: hidden dimension size
        num_experts: number of experts
        k: experts to select per token
        num_runs: number of benchmark iterations
        warmup: number of warmup iterations
    """
    import time
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create test data
    gate_logits = torch.randn(num_tokens, num_experts, device=device)
    tokens = torch.randn(num_tokens, hidden_dim, device=device)
    capacity = (num_tokens * k // num_experts) * 2
    
    print(f"\n{'='*70}")
    print(f"MoE Operations Benchmark")
    print(f"{'='*70}")
    print(f"Tokens: {num_tokens}, Hidden: {hidden_dim}, Experts: {num_experts}, K: {k}")
    print(f"Device: {device}, CUDA Ops Available: {CUDA_OPS_AVAILABLE}")
    print(f"{'='*70}\n")
    
    # Benchmark top-k gating
    if CUDA_OPS_AVAILABLE and device.type == 'cuda':
        # Warmup
        for _ in range(warmup):
            indices, weights = MoECUDAOps.topk_gating(gate_logits, k, use_cuda=True)
        torch.cuda.synchronize()
        
        # Benchmark CUDA
        start = time.perf_counter()
        for _ in range(num_runs):
            indices, weights = MoECUDAOps.topk_gating(gate_logits, k, use_cuda=True)
        torch.cuda.synchronize()
        cuda_time = (time.perf_counter() - start) / num_runs * 1000
        
        # Benchmark PyTorch
        for _ in range(warmup):
            indices, weights = MoECUDAOps.topk_gating(gate_logits, k, use_cuda=False)
        torch.cuda.synchronize()
        
        start = time.perf_counter()
        for _ in range(num_runs):
            indices, weights = MoECUDAOps.topk_gating(gate_logits, k, use_cuda=False)
        torch.cuda.synchronize()
        pytorch_time = (time.perf_counter() - start) / num_runs * 1000
        
        speedup = pytorch_time / cuda_time
        
        print(f"Top-K Gating:")
        print(f"  CUDA:    {cuda_time:.3f} ms")
        print(f"  PyTorch: {pytorch_time:.3f} ms")
        print(f"  Speedup: {speedup:.2f}x")
    else:
        # PyTorch only
        for _ in range(warmup):
            indices, weights = MoECUDAOps.topk_gating(gate_logits, k, use_cuda=False)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        for _ in range(num_runs):
            indices, weights = MoECUDAOps.topk_gating(gate_logits, k, use_cuda=False)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        pytorch_time = (time.perf_counter() - start) / num_runs * 1000
        
        print(f"Top-K Gating (PyTorch): {pytorch_time:.3f} ms")
    
    # Benchmark dispatch and combine
    top_k_indices, top_k_weights = MoECUDAOps.topk_gating(gate_logits, k, use_cuda=False)
    
    if CUDA_OPS_AVAILABLE and device.type == 'cuda':
        # CUDA dispatch
        for _ in range(warmup):
            expert_inputs, token_map = MoECUDAOps.dispatch_tokens(
                tokens, top_k_indices, num_experts, capacity, use_cuda=True
            )
        torch.cuda.synchronize()
        
        start = time.perf_counter()
        for _ in range(num_runs):
            expert_inputs, token_map = MoECUDAOps.dispatch_tokens(
                tokens, top_k_indices, num_experts, capacity, use_cuda=True
            )
        torch.cuda.synchronize()
        cuda_dispatch = (time.perf_counter() - start) / num_runs * 1000
        
        # PyTorch dispatch
        for _ in range(warmup):
            expert_inputs, token_map = MoECUDAOps.dispatch_tokens(
                tokens, top_k_indices, num_experts, capacity, use_cuda=False
            )
        torch.cuda.synchronize()
        
        start = time.perf_counter()
        for _ in range(num_runs):
            expert_inputs, token_map = MoECUDAOps.dispatch_tokens(
                tokens, top_k_indices, num_experts, capacity, use_cuda=False
            )
        torch.cuda.synchronize()
        pytorch_dispatch = (time.perf_counter() - start) / num_runs * 1000
        
        print(f"\nToken Dispatch:")
        print(f"  CUDA:    {cuda_dispatch:.3f} ms")
        print(f"  PyTorch: {pytorch_dispatch:.3f} ms")
        print(f"  Speedup: {pytorch_dispatch / cuda_dispatch:.2f}x")
    
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    # Run benchmark
    benchmark_moe_ops()