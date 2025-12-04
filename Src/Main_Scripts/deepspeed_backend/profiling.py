# Copyright (c) 2025 MatN23. All rights reserved.
# Licensed under the Custom License below.

import torch
import torch.distributed as dist
from typing import Dict, List, Optional, Any
from collections import defaultdict
import time
import json
from pathlib import Path


class MemoryTracker:
    """
    Track GPU memory usage across training stages.
    """
    
    def __init__(self, device: str = 'cuda', track_history: bool = True):
        self.device = device
        self.track_history = track_history
        
        # Memory snapshots
        self.snapshots: List[Dict[str, float]] = []
        self.current_snapshot: Dict[str, Any] = {}
        
        # Peak memory tracking
        self.peak_allocated = 0
        self.peak_reserved = 0
        
        # Stage-specific tracking
        self.stage_memory: Dict[str, List[float]] = defaultdict(list)
        
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        
        # Reset statistics
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device=device)
    
    def snapshot(self, stage: str = 'default') -> Dict[str, float]:
        """
        Take memory snapshot at current point.
        
        Args:
            stage: Training stage identifier (e.g., 'forward', 'backward', 'optimizer')
        
        Returns:
            Memory statistics in GB
        """
        if not torch.cuda.is_available():
            return {'error': 'CUDA not available'}
        
        allocated = torch.cuda.memory_allocated(self.device) / 1e9
        reserved = torch.cuda.memory_reserved(self.device) / 1e9
        max_allocated = torch.cuda.max_memory_allocated(self.device) / 1e9
        max_reserved = torch.cuda.max_memory_reserved(self.device) / 1e9
        
        snapshot = {
            'timestamp': time.time(),
            'stage': stage,
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'max_allocated_gb': max_allocated,
            'max_reserved_gb': max_reserved,
            'rank': self.rank,
        }
        
        # Track peak memory
        self.peak_allocated = max(self.peak_allocated, allocated)
        self.peak_reserved = max(self.peak_reserved, reserved)
        
        # Store stage-specific data
        self.stage_memory[stage].append(allocated)
        
        if self.track_history:
            self.snapshots.append(snapshot)
        
        self.current_snapshot = snapshot
        return snapshot
    
    def get_current_memory(self) -> Dict[str, float]:
        """Get current memory usage"""
        return self.snapshot('current')
    
    def get_peak_memory(self) -> Dict[str, float]:
        """Get peak memory usage"""
        return {
            'peak_allocated_gb': self.peak_allocated,
            'peak_reserved_gb': self.peak_reserved,
            'rank': self.rank,
        }
    
    def get_stage_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for each tracked stage"""
        stats = {}
        
        for stage, memories in self.stage_memory.items():
            if memories:
                stats[stage] = {
                    'mean_gb': sum(memories) / len(memories),
                    'max_gb': max(memories),
                    'min_gb': min(memories),
                    'count': len(memories),
                }
        
        return stats
    
    def reset(self):
        """Reset all tracked statistics"""
        self.snapshots.clear()
        self.stage_memory.clear()
        self.current_snapshot = {}
        self.peak_allocated = 0
        self.peak_reserved = 0
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device=self.device)
    
    def print_summary(self):
        """Print memory usage summary"""
        print(f"\n{'='*60}")
        print(f"Memory Tracker Summary (Rank {self.rank})")
        print(f"{'='*60}")
        
        current = self.get_current_memory()
        print(f"Current Allocated: {current['allocated_gb']:.2f} GB")
        print(f"Current Reserved:  {current['reserved_gb']:.2f} GB")
        
        peak = self.get_peak_memory()
        print(f"\nPeak Allocated:    {peak['peak_allocated_gb']:.2f} GB")
        print(f"Peak Reserved:     {peak['peak_reserved_gb']:.2f} GB")
        
        print(f"\nStage Statistics:")
        for stage, stats in self.get_stage_statistics().items():
            print(f"  {stage:15s}: "
                  f"mean={stats['mean_gb']:.2f} GB, "
                  f"max={stats['max_gb']:.2f} GB, "
                  f"samples={stats['count']}")
        
        print(f"{'='*60}\n")
    
    def save_history(self, output_path: str):
        """Save memory history to JSON"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'snapshots': self.snapshots,
            'stage_statistics': self.get_stage_statistics(),
            'peak_memory': self.get_peak_memory(),
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"[Rank {self.rank}] Saved memory history: {output_path}")


class FLOPCounter:
    """
    Count FLOPs (Floating Point Operations) during training.
    Useful for measuring computational efficiency.
    """
    
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.total_flops = 0
        self.layer_flops: Dict[str, int] = {}
        self.hooks = []
        
        # Register hooks for FLOP counting
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to count FLOPs"""
        def hook(module, input, output):
            module_name = module.__class__.__name__
            
            # Estimate FLOPs based on module type
            flops = 0
            
            if isinstance(module, torch.nn.Linear):
                # Linear: 2 * in_features * out_features per sample
                batch_size = input[0].size(0) if isinstance(input, tuple) else input.size(0)
                flops = 2 * module.in_features * module.out_features * batch_size
            
            elif isinstance(module, torch.nn.Conv2d):
                # Conv2d: 2 * C_in * C_out * K_h * K_w * H_out * W_out
                batch_size = input[0].size(0) if isinstance(input, tuple) else input.size(0)
                out_h = output.size(2)
                out_w = output.size(3)
                kernel_ops = module.kernel_size[0] * module.kernel_size[1]
                flops = (2 * module.in_channels * module.out_channels * 
                        kernel_ops * out_h * out_w * batch_size)
            
            elif isinstance(module, torch.nn.MultiheadAttention):
                # Attention: 4 * seq_len^2 * hidden_dim (approximate)
                if isinstance(input, tuple) and len(input) > 0:
                    seq_len = input[0].size(1)
                    hidden_dim = input[0].size(2)
                    flops = 4 * seq_len * seq_len * hidden_dim
            
            self.total_flops += flops
            
            # Track per-layer FLOPs
            layer_key = f"{module_name}_{id(module)}"
            self.layer_flops[layer_key] = self.layer_flops.get(layer_key, 0) + flops
        
        # Register hooks on all modules
        for module in self.model.modules():
            self.hooks.append(module.register_forward_hook(hook))
    
    def reset(self):
        """Reset FLOP counters"""
        self.total_flops = 0
        self.layer_flops.clear()
    
    def get_total_flops(self) -> int:
        """Get total FLOPs counted"""
        return self.total_flops
    
    def get_layer_flops(self) -> Dict[str, int]:
        """Get per-layer FLOP counts"""
        return self.layer_flops
    
    def get_tflops(self) -> float:
        """Get total FLOPs in TFLOPs"""
        return self.total_flops / 1e12
    
    def cleanup(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def print_summary(self):
        """Print FLOP counting summary"""
        print(f"\n{'='*60}")
        print(f"FLOP Counter Summary")
        print(f"{'='*60}")
        print(f"Total FLOPs: {self.get_tflops():.2f} TFLOPs")
        
        print(f"\nTop 10 Layers by FLOPs:")
        sorted_layers = sorted(
            self.layer_flops.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        for layer_name, flops in sorted_layers:
            print(f"  {layer_name:40s}: {flops/1e9:.2f} GFLOPs")
        
        print(f"{'='*60}\n")


class ZeROProfiler:
    """
    Comprehensive profiler for ZeRO-optimized training.
    Tracks memory, FLOPs, and timing across ZeRO stages.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        zero_stage: int = 0,
        expert_registry: Optional[Dict] = None,
    ):
        self.model = model
        self.zero_stage = zero_stage
        self.expert_registry = expert_registry or {}
        
        # Component profilers
        self.memory_tracker = MemoryTracker()
        self.flop_counter = FLOPCounter(model)
        
        # Timing tracking
        self.timers: Dict[str, List[float]] = defaultdict(list)
        self.current_timers: Dict[str, float] = {}
        
        # Expert-specific profiling
        self.expert_timers: Dict[str, List[float]] = defaultdict(list)
        self.expert_memory: Dict[str, List[float]] = defaultdict(list)
        
        # Iteration tracking
        self.iteration = 0
        
        self.rank = dist.get_rank() if dist.is_initialized() else 0
    
    def start_timer(self, name: str):
        """Start timing a section"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.current_timers[name] = time.time()
    
    def end_timer(self, name: str) -> float:
        """End timing a section and return elapsed time"""
        if name not in self.current_timers:
            return 0.0
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        elapsed = time.time() - self.current_timers[name]
        self.timers[name].append(elapsed)
        del self.current_timers[name]
        
        return elapsed
    
    def profile_forward(self):
        """Profile forward pass"""
        self.start_timer('forward')
        self.memory_tracker.snapshot('forward_start')
    
    def profile_forward_end(self):
        """End forward pass profiling"""
        elapsed = self.end_timer('forward')
        self.memory_tracker.snapshot('forward_end')
        return elapsed
    
    def profile_backward(self):
        """Profile backward pass"""
        self.start_timer('backward')
        self.memory_tracker.snapshot('backward_start')
    
    def profile_backward_end(self):
        """End backward pass profiling"""
        elapsed = self.end_timer('backward')
        self.memory_tracker.snapshot('backward_end')
        return elapsed
    
    def profile_optimizer_step(self):
        """Profile optimizer step"""
        self.start_timer('optimizer')
        self.memory_tracker.snapshot('optimizer_start')
    
    def profile_optimizer_end(self):
        """End optimizer step profiling"""
        elapsed = self.end_timer('optimizer')
        self.memory_tracker.snapshot('optimizer_end')
        return elapsed
    
    def profile_expert(self, expert_name: str):
        """Profile specific expert execution"""
        self.start_timer(f'expert_{expert_name}')
        
        # Track memory if expert exists
        if expert_name in self.expert_registry:
            mem_before = torch.cuda.memory_allocated() / 1e9
            self.expert_memory[expert_name].append(mem_before)
    
    def profile_expert_end(self, expert_name: str):
        """End expert profiling"""
        elapsed = self.end_timer(f'expert_{expert_name}')
        self.expert_timers[expert_name].append(elapsed)
        return elapsed
    
    def step(self):
        """Increment iteration counter"""
        self.iteration += 1
    
    def get_timing_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get timing statistics for all tracked sections"""
        stats = {}
        
        for name, times in self.timers.items():
            if times:
                stats[name] = {
                    'mean_ms': (sum(times) / len(times)) * 1000,
                    'min_ms': min(times) * 1000,
                    'max_ms': max(times) * 1000,
                    'total_s': sum(times),
                    'count': len(times),
                }
        
        return stats
    
    def get_expert_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get per-expert statistics"""
        stats = {}
        
        for expert_name, times in self.expert_timers.items():
            if times:
                stats[expert_name] = {
                    'mean_time_ms': (sum(times) / len(times)) * 1000,
                    'total_time_s': sum(times),
                    'count': len(times),
                }
                
                if expert_name in self.expert_memory:
                    memories = self.expert_memory[expert_name]
                    stats[expert_name]['mean_memory_gb'] = sum(memories) / len(memories)
        
        return stats
    
    def get_throughput(self, batch_size: int) -> Dict[str, float]:
        """
        Calculate training throughput.
        
        Args:
            batch_size: Batch size per iteration
        
        Returns:
            Throughput metrics
        """
        stats = self.get_timing_statistics()
        
        throughput = {}
        
        if 'forward' in stats:
            forward_time = stats['forward']['mean_ms'] / 1000
            throughput['samples_per_sec_forward'] = batch_size / forward_time if forward_time > 0 else 0
        
        if 'backward' in stats:
            backward_time = stats['backward']['mean_ms'] / 1000
            throughput['samples_per_sec_backward'] = batch_size / backward_time if backward_time > 0 else 0
        
        # Combined throughput
        total_time = sum(
            stats[key]['mean_ms'] / 1000 
            for key in ['forward', 'backward', 'optimizer']
            if key in stats
        )
        
        if total_time > 0:
            throughput['samples_per_sec_total'] = batch_size / total_time
        
        return throughput
    
    def print_summary(self, batch_size: Optional[int] = None):
        """Print comprehensive profiling summary"""
        print(f"\n{'='*80}")
        print(f"ZeRO Profiler Summary (Rank {self.rank}, Stage {self.zero_stage})")
        print(f"{'='*80}")
        
        # Timing statistics
        print(f"\nTiming Statistics:")
        for name, stats in self.get_timing_statistics().items():
            print(f"  {name:20s}: "
                  f"mean={stats['mean_ms']:.2f}ms, "
                  f"min={stats['min_ms']:.2f}ms, "
                  f"max={stats['max_ms']:.2f}ms")
        
        # Throughput
        if batch_size:
            print(f"\nThroughput (batch_size={batch_size}):")
            for key, value in self.get_throughput(batch_size).items():
                print(f"  {key:30s}: {value:.2f}")
        
        # Expert statistics
        if self.expert_timers:
            print(f"\nExpert Statistics:")
            for expert_name, stats in self.get_expert_statistics().items():
                print(f"  {expert_name:20s}: "
                      f"mean={stats['mean_time_ms']:.2f}ms, "
                      f"calls={stats['count']}")
        
        # Memory statistics
        print(f"\nMemory Statistics:")
        self.memory_tracker.print_summary()
        
        # FLOP statistics
        print(f"\nFLOP Statistics:")
        self.flop_counter.print_summary()
        
        print(f"{'='*80}\n")
    
    def save_profile(self, output_dir: str):
        """Save profiling data to disk"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        profile_data = {
            'rank': self.rank,
            'zero_stage': self.zero_stage,
            'iteration': self.iteration,
            'timing_statistics': self.get_timing_statistics(),
            'expert_statistics': self.get_expert_statistics(),
            'memory_statistics': self.memory_tracker.get_stage_statistics(),
            'peak_memory': self.memory_tracker.get_peak_memory(),
            'total_flops': self.flop_counter.get_total_flops(),
        }
        
        profile_path = output_dir / f"profile_rank_{self.rank}.json"
        with open(profile_path, 'w') as f:
            json.dump(profile_data, f, indent=2)
        
        # Save memory history
        self.memory_tracker.save_history(
            output_dir / f"memory_history_rank_{self.rank}.json"
        )
        
        print(f"[Rank {self.rank}] Saved profile data: {output_dir}")
    
    def cleanup(self):
        """Clean up profiler resources"""
        self.flop_counter.cleanup()
        self.memory_tracker.reset()
        self.timers.clear()
        self.current_timers.clear()


def profile_zero_stage_overhead(
    model: torch.nn.Module,
    optimizer: Any,
    zero_stages: List[int] = [0, 1, 2, 3],
    num_iterations: int = 10,
) -> Dict[int, Dict[str, float]]:
    """
    Benchmark overhead of different ZeRO stages.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        zero_stages: List of ZeRO stages to benchmark
        num_iterations: Number of iterations per stage
    
    Returns:
        Benchmark results per stage
    """
    results = {}
    
    print(f"\n{'='*60}")
    print(f"Benchmarking ZeRO Stage Overhead")
    print(f"{'='*60}\n")
    
    for stage in zero_stages:
        print(f"Benchmarking ZeRO Stage {stage}...")
        
        profiler = ZeROProfiler(model, zero_stage=stage)
        
        for i in range(num_iterations):
            # Simulate training iteration
            profiler.profile_forward()
            # ... forward pass ...
            profiler.profile_forward_end()
            
            profiler.profile_backward()
            # ... backward pass ...
            profiler.profile_backward_end()
            
            profiler.profile_optimizer_step()
            # ... optimizer step ...
            profiler.profile_optimizer_end()
            
            profiler.step()
        
        # Collect results
        timing_stats = profiler.get_timing_statistics()
        memory_stats = profiler.memory_tracker.get_peak_memory()
        
        results[stage] = {
            'forward_ms': timing_stats.get('forward', {}).get('mean_ms', 0),
            'backward_ms': timing_stats.get('backward', {}).get('mean_ms', 0),
            'optimizer_ms': timing_stats.get('optimizer', {}).get('mean_ms', 0),
            'peak_memory_gb': memory_stats.get('peak_allocated_gb', 0),
        }
        
        profiler.cleanup()
    
    # Print comparison
    print(f"\nZeRO Stage Comparison:")
    print(f"{'Stage':<10}{'Forward (ms)':<15}{'Backward (ms)':<15}{'Optimizer (ms)':<15}{'Peak Mem (GB)':<15}")
    print(f"{'-'*70}")
    
    for stage, stats in results.items():
        print(f"{stage:<10}"
              f"{stats['forward_ms']:<15.2f}"
              f"{stats['backward_ms']:<15.2f}"
              f"{stats['optimizer_ms']:<15.2f}"
              f"{stats['peak_memory_gb']:<15.2f}")
    
    print(f"{'='*60}\n")
    
    return results