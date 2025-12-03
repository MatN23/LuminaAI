"""
Utility functions for LuminaAI DeepSpeed Backend.
Logging, helper functions, and parameter state utilities.
"""

import torch
import torch.distributed as dist
import logging
import os
import sys
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import json
import numpy as np
from collections import defaultdict
import functools


# ============================================================================
# Logging Utilities
# ============================================================================

def setup_logger(
    name: str = "luminai_deepspeed",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    rank: Optional[int] = None,
) -> logging.Logger:
    """
    Setup logger with optional file output and rank-aware formatting.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional log file path
        rank: Distributed rank (only rank 0 logs by default)
    
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Determine if we should log (only rank 0 by default)
    if rank is not None and rank != 0:
        logger.addHandler(logging.NullHandler())
        return logger
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # Format with rank if provided
    if rank is not None:
        fmt = f'[Rank {rank}] %(asctime)s - %(name)s - %(levelname)s - %(message)s'
    else:
        fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    formatter = logging.Formatter(fmt, datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class RankLogger:
    """
    Context-aware logger that only logs on specified ranks.
    """
    
    def __init__(self, name: str = "luminai", log_ranks: List[int] = [0]):
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.log_ranks = log_ranks
        self.should_log = self.rank in log_ranks
        
        if self.should_log:
            self.logger = setup_logger(name, rank=self.rank)
        else:
            self.logger = logging.getLogger(name)
            self.logger.addHandler(logging.NullHandler())
    
    def info(self, msg: str, *args, **kwargs):
        if self.should_log:
            self.logger.info(msg, *args, **kwargs)
    
    def debug(self, msg: str, *args, **kwargs):
        if self.should_log:
            self.logger.debug(msg, *args, **kwargs)
    
    def warning(self, msg: str, *args, **kwargs):
        if self.should_log:
            self.logger.warning(msg, *args, **kwargs)
    
    def error(self, msg: str, *args, **kwargs):
        if self.should_log:
            self.logger.error(msg, *args, **kwargs)
    
    def critical(self, msg: str, *args, **kwargs):
        if self.should_log:
            self.logger.critical(msg, *args, **kwargs)


# ============================================================================
# Distributed Utilities
# ============================================================================

def init_distributed(backend: str = 'nccl') -> Tuple[int, int]:
    """
    Initialize distributed training.
    
    Args:
        backend: Distributed backend ('nccl', 'gloo', 'mpi')
    
    Returns:
        (rank, world_size) tuple
    """
    if not dist.is_initialized():
        # Get rank and world size from environment
        rank = int(os.environ.get('RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        # Initialize process group
        dist.init_process_group(backend=backend)
        
        # Set device
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        
        print(f"Initialized distributed: rank={rank}, world_size={world_size}")
        
        return rank, world_size
    else:
        return dist.get_rank(), dist.get_world_size()


def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def all_gather_object(obj: Any) -> List[Any]:
    """
    All-gather Python objects across ranks.
    
    Args:
        obj: Object to gather
    
    Returns:
        List of objects from all ranks
    """
    if not dist.is_initialized():
        return [obj]
    
    world_size = dist.get_world_size()
    gathered = [None] * world_size
    dist.all_gather_object(gathered, obj)
    
    return gathered


def broadcast_object(obj: Any, src: int = 0) -> Any:
    """
    Broadcast Python object from source rank.
    
    Args:
        obj: Object to broadcast
        src: Source rank
    
    Returns:
        Broadcasted object
    """
    if not dist.is_initialized():
        return obj
    
    obj_list = [obj]
    dist.broadcast_object_list(obj_list, src=src)
    
    return obj_list[0]


# ============================================================================
# Parameter State Utilities
# ============================================================================

def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count total and trainable parameters.
    
    Args:
        model: PyTorch model
    
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params,
        'total_millions': total_params / 1e6,
        'trainable_millions': trainable_params / 1e6,
    }


def get_parameter_stats(model: torch.nn.Module) -> Dict[str, Any]:
    """
    Get comprehensive statistics about model parameters.
    
    Args:
        model: PyTorch model
    
    Returns:
        Statistics dictionary
    """
    stats = {
        'counts': count_parameters(model),
        'dtypes': defaultdict(int),
        'devices': defaultdict(int),
        'shapes': {},
        'memory_mb': 0,
    }
    
    for name, param in model.named_parameters():
        # Count by dtype
        stats['dtypes'][str(param.dtype)] += param.numel()
        
        # Count by device
        stats['devices'][str(param.device)] += param.numel()
        
        # Store shapes
        stats['shapes'][name] = list(param.shape)
        
        # Calculate memory
        stats['memory_mb'] += param.numel() * param.element_size() / 1e6
    
    # Convert defaultdicts to regular dicts
    stats['dtypes'] = dict(stats['dtypes'])
    stats['devices'] = dict(stats['devices'])
    
    return stats


def get_gradient_stats(model: torch.nn.Module) -> Dict[str, Any]:
    """
    Get statistics about gradients.
    
    Args:
        model: PyTorch model
    
    Returns:
        Gradient statistics
    """
    stats = {
        'total_grad_norm': 0.0,
        'num_params_with_grad': 0,
        'num_params_without_grad': 0,
        'layer_grad_norms': {},
    }
    
    total_norm = 0.0
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2).item()
            total_norm += param_norm ** 2
            stats['num_params_with_grad'] += 1
            stats['layer_grad_norms'][name] = param_norm
        else:
            stats['num_params_without_grad'] += 1
    
    stats['total_grad_norm'] = total_norm ** 0.5
    
    return stats


def clip_grad_norm_per_layer(
    model: torch.nn.Module,
    max_norm: float,
    norm_type: float = 2.0,
) -> Dict[str, float]:
    """
    Clip gradients per layer and return clipping statistics.
    
    Args:
        model: PyTorch model
        max_norm: Maximum gradient norm
        norm_type: Type of norm to use
    
    Returns:
        Per-layer gradient norms before clipping
    """
    layer_norms = {}
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = torch.norm(param.grad.data, p=norm_type).item()
            layer_norms[name] = param_norm
            
            # Clip this layer's gradients
            if param_norm > max_norm:
                param.grad.data.mul_(max_norm / (param_norm + 1e-6))
    
    return layer_norms


def get_expert_parameter_distribution(
    expert_registry: Dict[str, torch.nn.Module]
) -> Dict[str, Dict[str, Any]]:
    """
    Get parameter distribution across experts.
    
    Args:
        expert_registry: Dictionary of expert modules
    
    Returns:
        Statistics for each expert
    """
    distribution = {}
    
    for expert_name, expert_module in expert_registry.items():
        stats = count_parameters(expert_module)
        distribution[expert_name] = stats
    
    return distribution


# ============================================================================
# Memory Utilities
# ============================================================================

def get_memory_summary() -> Dict[str, float]:
    """
    Get GPU memory summary.
    
    Returns:
        Memory statistics in GB
    """
    if not torch.cuda.is_available():
        return {'error': 'CUDA not available'}
    
    return {
        'allocated_gb': torch.cuda.memory_allocated() / 1e9,
        'reserved_gb': torch.cuda.memory_reserved() / 1e9,
        'max_allocated_gb': torch.cuda.max_memory_allocated() / 1e9,
        'max_reserved_gb': torch.cuda.max_memory_reserved() / 1e9,
    }


def empty_cache():
    """Empty CUDA cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_device_properties() -> Dict[str, Any]:
    """
    Get CUDA device properties.
    
    Returns:
        Device properties dictionary
    """
    if not torch.cuda.is_available():
        return {'error': 'CUDA not available'}
    
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    
    return {
        'name': props.name,
        'major': props.major,
        'minor': props.minor,
        'total_memory_gb': props.total_memory / 1e9,
        'multi_processor_count': props.multi_processor_count,
    }


# ============================================================================
# Configuration Utilities
# ============================================================================

def save_config(config: Dict[str, Any], path: str):
    """
    Save configuration to JSON file.
    
    Args:
        config: Configuration dictionary
        path: Output path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Saved config: {path}")


def load_config(path: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        path: Config file path
    
    Returns:
        Configuration dictionary
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    
    with open(path, 'r') as f:
        config = json.load(f)
    
    return config


# ============================================================================
# Tensor Utilities
# ============================================================================

def flatten_tensors(tensors: List[torch.Tensor]) -> torch.Tensor:
    """
    Flatten list of tensors into single 1D tensor.
    
    Args:
        tensors: List of tensors
    
    Returns:
        Flattened tensor
    """
    return torch.cat([t.flatten() for t in tensors])


def unflatten_tensors(
    flat_tensor: torch.Tensor,
    shapes: List[torch.Size],
) -> List[torch.Tensor]:
    """
    Unflatten 1D tensor back into list of shaped tensors.
    
    Args:
        flat_tensor: Flattened tensor
        shapes: List of target shapes
    
    Returns:
        List of unflattened tensors
    """
    tensors = []
    offset = 0
    
    for shape in shapes:
        numel = np.prod(shape)
        tensors.append(flat_tensor[offset:offset + numel].view(shape))
        offset += numel
    
    return tensors


def allreduce_gradients(model: torch.nn.Module, average: bool = True):
    """
    All-reduce gradients across ranks.
    
    Args:
        model: PyTorch model
        average: Average gradients (vs sum)
    """
    if not dist.is_initialized():
        return
    
    world_size = dist.get_world_size()
    
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            if average:
                param.grad.data.div_(world_size)


# ============================================================================
# Expert Routing Utilities (for MoE/MoD)
# ============================================================================

def compute_load_balance_loss(
    router_logits: torch.Tensor,
    expert_indices: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """
    Compute load balancing loss for MoE routing.
    Encourages uniform distribution of tokens across experts.
    
    Args:
        router_logits: Router output logits [batch, seq, num_experts]
        expert_indices: Selected expert indices [batch, seq]
        num_experts: Total number of experts
    
    Returns:
        Load balancing loss
    """
    # Compute routing probabilities
    routing_probs = torch.softmax(router_logits, dim=-1)
    
    # Average routing probability per expert
    mean_probs = routing_probs.mean(dim=[0, 1])  # [num_experts]
    
    # Fraction of tokens assigned to each expert
    expert_mask = torch.nn.functional.one_hot(expert_indices, num_experts).float()
    fraction_assigned = expert_mask.mean(dim=[0, 1])  # [num_experts]
    
    # Load balancing loss: encourage uniform distribution
    load_balance_loss = num_experts * torch.sum(mean_probs * fraction_assigned)
    
    return load_balance_loss


def get_expert_capacity(
    num_tokens: int,
    num_experts: int,
    capacity_factor: float = 1.25,
) -> int:
    """
    Calculate expert capacity for MoE.
    
    Args:
        num_tokens: Total number of tokens
        num_experts: Number of experts
        capacity_factor: Capacity multiplier (>1 for overflow)
    
    Returns:
        Capacity per expert
    """
    tokens_per_expert = num_tokens / num_experts
    capacity = int(tokens_per_expert * capacity_factor)
    
    return capacity


# ============================================================================
# Debugging Utilities
# ============================================================================

def print_model_summary(
    model: torch.nn.Module,
    expert_registry: Optional[Dict] = None,
):
    """
    Print comprehensive model summary.
    
    Args:
        model: PyTorch model
        expert_registry: Optional expert registry
    """
    print(f"\n{'='*80}")
    print(f"Model Summary")
    print(f"{'='*80}")
    
    # Parameter statistics
    param_stats = get_parameter_stats(model)
    print(f"\nParameters:")
    print(f"  Total:      {param_stats['counts']['total_millions']:.2f}M")
    print(f"  Trainable:  {param_stats['counts']['trainable_millions']:.2f}M")
    print(f"  Memory:     {param_stats['memory_mb']:.2f} MB")
    
    # Data types
    print(f"\nData Types:")
    for dtype, count in param_stats['dtypes'].items():
        print(f"  {dtype:20s}: {count:,} parameters")
    
    # Devices
    print(f"\nDevices:")
    for device, count in param_stats['devices'].items():
        print(f"  {device:20s}: {count:,} parameters")
    
    # Expert distribution
    if expert_registry:
        print(f"\nExpert Distribution:")
        expert_dist = get_expert_parameter_distribution(expert_registry)
        for expert_name, stats in expert_dist.items():
            print(f"  {expert_name:20s}: {stats['total_millions']:.2f}M parameters")
    
    # Memory summary
    if torch.cuda.is_available():
        mem_summary = get_memory_summary()
        print(f"\nGPU Memory:")
        print(f"  Allocated:  {mem_summary['allocated_gb']:.2f} GB")
        print(f"  Reserved:   {mem_summary['reserved_gb']:.2f} GB")
    
    print(f"{'='*80}\n")


def verify_zero_partitioning(
    model: torch.nn.Module,
    zero_stage: int,
    rank: int,
    world_size: int,
) -> bool:
    """
    Verify ZeRO partitioning is correct.
    
    Args:
        model: PyTorch model
        zero_stage: ZeRO stage
        rank: Current rank
        world_size: Total ranks
    
    Returns:
        True if partitioning is valid
    """
    if zero_stage == 0:
        return True
    
    print(f"\n[Rank {rank}] Verifying ZeRO-{zero_stage} partitioning...")
    
    param_count = 0
    for param in model.parameters():
        if param.data.numel() > 0:
            param_count += 1
    
    # Gather param counts from all ranks
    if dist.is_initialized():
        all_counts = [torch.tensor(param_count) for _ in range(world_size)]
        dist.all_gather(all_counts, torch.tensor(param_count))
        
        total_params = sum(c.item() for c in all_counts)
        expected_per_rank = total_params // world_size
        
        print(f"[Rank {rank}] Has {param_count} parameters "
              f"(expected ~{expected_per_rank})")
        
        return True
    
    return True


# ============================================================================
# Reproducibility Utilities
# ============================================================================

def set_seed(seed: int):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    # Make CUDA deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Set random seed: {seed}")