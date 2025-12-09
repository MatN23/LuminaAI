"""
CUDA MoE Debugging Script
=========================

This will tell us EXACTLY what's happening:
1. Is CUDA actually being called?
2. How much time is CUDA vs PyTorch?
3. Where is the bottleneck?
"""

import torch
import torch.nn as nn
import time
from typing import Dict, Any

# Try to import CUDA ops
try:
    from core import moe_cuda_ops
    CUDA_AVAILABLE = True
    print("✅ CUDA ops imported successfully")
except ImportError as e:
    CUDA_AVAILABLE = False
    print(f"❌ CUDA ops import failed: {e}")

# Check if CUDA is actually available
print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
print(f"CUDA ops module available: {CUDA_AVAILABLE}")

print("\n" + "="*70)
print("DETAILED MoE LAYER PROFILING")
print("="*70 + "\n")


class ProfilingMoELayer(nn.Module):
    """
    Instrumented MoE layer that tracks EXACTLY what's happening.
    """
    
    def __init__(self, config):
        super().__init__()
        from model import MoEFFNLayer
        self.moe = MoEFFNLayer(config)
        
        # Timing buckets
        self.timings = {
            'gate_compute': 0.0,
            'cuda_routing': 0.0,
            'pytorch_routing': 0.0,
            'dispatch': 0.0,
            'expert_compute': 0.0,
            'combine': 0.0,
            'aux_loss': 0.0,
            'total': 0.0,
        }
        
        self.call_counts = {
            'cuda_routing': 0,
            'pytorch_routing': 0,
            'total_forwards': 0,
        }
    
    def forward(self, x):
        """Instrumented forward pass."""
        start_total = time.perf_counter()
        
        batch_size, seq_len, hidden_size = x.shape
        x_flat = x.view(-1, hidden_size)
        total_tokens = x_flat.shape[0]
        
        # Gate computation
        start = time.perf_counter()
        gate_logits = self.moe.gate(x_flat)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.timings['gate_compute'] += (time.perf_counter() - start) * 1000
        
        # Routing
        start = time.perf_counter()
        if self.moe.use_cuda_ops and CUDA_AVAILABLE and x.is_cuda:
            try:
                # Check if it will actually use CUDA
                from core.moe_cuda_wrapper import MoECUDAOps
                will_use_cuda = MoECUDAOps.should_use_cuda(
                    total_tokens, 
                    self.moe.num_experts, 
                    hidden_size,
                    True, 
                    True
                )
                
                if will_use_cuda:
                    top_k_indices, top_k_probs = moe_cuda_ops.topk_gating(
                        gate_logits, self.moe.top_k, self.moe.routing_temperature
                    )
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    self.timings['cuda_routing'] += (time.perf_counter() - start) * 1000
                    self.call_counts['cuda_routing'] += 1
                else:
                    # Falls back to PyTorch
                    top_k_values, top_k_indices = torch.topk(
                        gate_logits / self.moe.routing_temperature, 
                        self.moe.top_k, 
                        dim=-1
                    )
                    top_k_probs = torch.nn.functional.softmax(top_k_values, dim=-1)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    self.timings['pytorch_routing'] += (time.perf_counter() - start) * 1000
                    self.call_counts['pytorch_routing'] += 1
            except Exception as e:
                print(f"❌ CUDA routing failed: {e}")
                # Fallback
                top_k_values, top_k_indices = torch.topk(
                    gate_logits / self.moe.routing_temperature, 
                    self.moe.top_k, 
                    dim=-1
                )
                top_k_probs = torch.nn.functional.softmax(top_k_values, dim=-1)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                self.timings['pytorch_routing'] += (time.perf_counter() - start) * 1000
                self.call_counts['pytorch_routing'] += 1
        else:
            # PyTorch routing
            top_k_values, top_k_indices = torch.topk(
                gate_logits / self.moe.routing_temperature, 
                self.moe.top_k, 
                dim=-1
            )
            top_k_probs = torch.nn.functional.softmax(top_k_values, dim=-1)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.timings['pytorch_routing'] += (time.perf_counter() - start) * 1000
            self.call_counts['pytorch_routing'] += 1
        
        # Dispatch
        start = time.perf_counter()
        capacity = int((total_tokens * self.moe.top_k / self.moe.num_experts) * self.moe.capacity_factor)
        expert_inputs = torch.zeros(
            self.moe.num_experts, capacity, hidden_size,
            dtype=x_flat.dtype, device=x_flat.device
        )
        token_map = torch.full(
            (self.moe.num_experts, capacity), -1,
            dtype=torch.int32, device=x_flat.device
        )
        expert_positions = torch.zeros(self.moe.num_experts, dtype=torch.int32, device=x_flat.device)
        
        for token_idx in range(total_tokens):
            for k_idx in range(self.moe.top_k):
                expert_id = top_k_indices[token_idx, k_idx].item()
                pos = expert_positions[expert_id].item()
                
                if pos < capacity:
                    expert_inputs[expert_id, pos] = x_flat[token_idx]
                    token_map[expert_id, pos] = token_idx * self.moe.top_k + k_idx
                    expert_positions[expert_id] += 1
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.timings['dispatch'] += (time.perf_counter() - start) * 1000
        
        # Expert computation (THE BOTTLENECK)
        start = time.perf_counter()
        expert_outputs = torch.zeros_like(expert_inputs)
        for expert_id in range(self.moe.num_experts):
            mask = (token_map[expert_id] >= 0)
            if mask.any():
                valid_inputs = expert_inputs[expert_id][mask]
                expert_outputs[expert_id][mask] = self.moe.experts[expert_id](valid_inputs)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.timings['expert_compute'] += (time.perf_counter() - start) * 1000
        
        # Combine
        start = time.perf_counter()
        output = torch.zeros_like(x_flat)
        for expert_id in range(self.moe.num_experts):
            for pos in range(capacity):
                token_weight_idx = token_map[expert_id, pos].item()
                if token_weight_idx < 0:
                    continue
                
                token_idx = token_weight_idx // self.moe.top_k
                weight_idx = token_weight_idx % self.moe.top_k
                
                if token_idx >= total_tokens:
                    continue
                
                weight = top_k_probs[token_idx, weight_idx]
                expert_out = expert_outputs[expert_id, pos]
                output[token_idx] += weight * expert_out
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.timings['combine'] += (time.perf_counter() - start) * 1000
        
        # Aux loss
        start = time.perf_counter()
        gate_probs = torch.nn.functional.softmax(gate_logits, dim=-1)
        expert_usage = torch.zeros(self.moe.num_experts, device=gate_probs.device)
        for k in range(self.moe.top_k):
            expert_counts = torch.bincount(
                top_k_indices[:, k], 
                minlength=self.moe.num_experts
            )
            expert_usage += expert_counts.float()
        expert_usage = expert_usage / (total_tokens * self.moe.top_k + 1e-9)
        gate_importance = gate_probs.mean(dim=0)
        aux_loss = torch.sum(expert_usage * gate_importance) * self.moe.num_experts
        aux_loss = torch.clamp(aux_loss * self.moe.load_balancing_weight, max=1.0)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.timings['aux_loss'] += (time.perf_counter() - start) * 1000
        
        # Total
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.timings['total'] += (time.perf_counter() - start_total) * 1000
        self.call_counts['total_forwards'] += 1
        
        return output.view(batch_size, seq_len, hidden_size), aux_loss
    
    def print_profile(self):
        """Print detailed profiling results."""
        n = self.call_counts['total_forwards']
        if n == 0:
            print("No forwards recorded!")
            return
        
        print("\n" + "="*70)
        print("MoE LAYER PROFILING RESULTS")
        print("="*70)
        print(f"Total forward passes: {n}")
        print(f"CUDA routing calls: {self.call_counts['cuda_routing']}")
        print(f"PyTorch routing calls: {self.call_counts['pytorch_routing']}")
        
        if self.call_counts['cuda_routing'] == 0:
            print("\n❌ WARNING: CUDA ROUTING NEVER USED!")
            print("   Your CUDA kernels are NOT being called!")
        
        print(f"\n{'Operation':<20} {'Total (ms)':<12} {'Avg (ms)':<12} {'% of Total':<12}")
        print("-"*70)
        
        total_time = self.timings['total']
        
        for key in ['gate_compute', 'cuda_routing', 'pytorch_routing', 
                    'dispatch', 'expert_compute', 'combine', 'aux_loss']:
            time_ms = self.timings[key]
            avg_ms = time_ms / n if n > 0 else 0
            pct = (time_ms / total_time * 100) if total_time > 0 else 0
            print(f"{key:<20} {time_ms:>10.2f}  {avg_ms:>10.4f}  {pct:>10.1f}%")
        
        print("-"*70)
        print(f"{'TOTAL':<20} {total_time:>10.2f}  {total_time/n:>10.4f}  {100.0:>10.1f}%")
        print("="*70)
        
        # Identify bottleneck
        max_time = max(self.timings.values())
        bottleneck = [k for k, v in self.timings.items() if v == max_time][0]
        print(f"\n🔍 BOTTLENECK: {bottleneck} ({max_time/n:.4f} ms per call)")
        
        if bottleneck == 'expert_compute':
            print("   ✅ This is expected - expert computation should dominate")
        elif bottleneck == 'dispatch' or bottleneck == 'combine':
            print("   ⚠️  Dispatch/combine is slow - CUDA should help here!")
        elif bottleneck == 'pytorch_routing':
            print("   ❌ Using PyTorch routing instead of CUDA!")


def debug_moe_performance(config):
    """Run comprehensive debugging."""
    print("\n" + "="*70)
    print("CREATING TEST MODEL")
    print("="*70)
    
    # Create profiling layer
    profiling_layer = ProfilingMoELayer(config).cuda()
    
    print(f"\nConfiguration:")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Num experts: {config.num_experts}")
    print(f"  Top-K: {config.moe_top_k}")
    print(f"  CUDA enabled: {profiling_layer.moe.use_cuda_ops}")
    
    # Create test input
    batch_size = 25
    seq_len = 256
    x = torch.randn(batch_size, seq_len, config.hidden_size, device='cuda')
    
    print(f"\nTest input: {batch_size}x{seq_len}x{config.hidden_size} = {batch_size*seq_len} tokens")
    
    # Warmup
    print("\nWarming up (10 iterations)...")
    for _ in range(10):
        _ = profiling_layer(x)
    
    # Reset stats
    profiling_layer.timings = {k: 0.0 for k in profiling_layer.timings}
    profiling_layer.call_counts = {k: 0 for k in profiling_layer.call_counts}
    
    # Profile
    print("Profiling (100 iterations)...")
    for _ in range(100):
        output, aux_loss = profiling_layer(x)
    
    # Print results
    profiling_layer.print_profile()
    
    # Additional checks
    print("\n" + "="*70)
    print("SANITY CHECKS")
    print("="*70)
    
    # Check if should_use_cuda would return True
    from core.moe_cuda_wrapper import MoECUDAOps
    total_tokens = batch_size * seq_len
    should_use = MoECUDAOps.should_use_cuda(
        total_tokens,
        config.num_experts,
        config.hidden_size,
        True,
        True
    )
    print(f"MoECUDAOps.should_use_cuda() returns: {should_use}")
    print(f"  Tokens: {total_tokens} (threshold: {MoECUDAOps.CUDA_THRESHOLD_TOKENS})")
    print(f"  Experts: {config.num_experts} (threshold: {MoECUDAOps.CUDA_THRESHOLD_EXPERTS})")
    print(f"  Hidden: {config.hidden_size} (threshold: {MoECUDAOps.CUDA_THRESHOLD_HIDDEN})")


if __name__ == "__main__":
    # Create config matching your training
    from model import DeepSeekConfig
    
    config = DeepSeekConfig(
        hidden_size=128,
        num_experts=8,
        moe_top_k=2,
        num_layers=2,
        num_heads=2,
        use_moe=True,
        use_cuda_moe=True,
    )
    
    debug_moe_performance(config)