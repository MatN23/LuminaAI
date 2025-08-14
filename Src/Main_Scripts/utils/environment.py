# Copyright (c) 2025 Matias Nielsen. All rights reserved.
# Licensed under the Custom License below.

import shutil
import torch
from typing import List, Dict, Any


def validate_environment() -> List[str]:
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
    try:
        import psutil
        memory = psutil.virtual_memory()
        if memory.available / 1e9 < 4:
            issues.append(f"Low system memory: {memory.available / 1e9:.1f}GB available")
    except ImportError:
        issues.append("psutil not available, cannot check system memory")
    
    return issues


def estimate_training_time(config, dataset_size: int) -> Dict[str, float]:
    """Estimate training time and resource requirements."""
    from core.model import estimate_parameters
    
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


def get_system_info() -> Dict[str, Any]:
    """Get comprehensive system information."""
    info = {
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'python_version': None,
        'system_memory_gb': None,
        'disk_space_gb': None
    }
    
    # Python version
    import sys
    info['python_version'] = sys.version
    
    # CUDA info
    if torch.cuda.is_available():
        info['cuda_version'] = torch.version.cuda
        info['gpu_count'] = torch.cuda.device_count()
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    # System memory
    try:
        import psutil
        memory = psutil.virtual_memory()
        info['system_memory_gb'] = memory.total / 1e9
        info['available_memory_gb'] = memory.available / 1e9
        info['cpu_count'] = psutil.cpu_count()
    except ImportError:
        pass
    
    # Disk space
    try:
        disk_usage = shutil.disk_usage(".")
        info['disk_space_gb'] = disk_usage.free / 1e9
    except Exception:
        pass
    
    return info