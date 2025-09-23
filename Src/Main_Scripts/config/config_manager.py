# Copyright (c) 2025 MatN23. All rights reserved.
# Licensed under the Custom License below.

from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
import yaml
import torch
import os
import json


@dataclass
class Config:
    """Base configuration class for MoE models with comprehensive parameter support."""
    
    # Model architecture
    hidden_size: int = 4096
    num_layers: int = 32
    num_heads: int = 32
    seq_length: int = 2048
    vocab_size: int = 50000
    
    # MoE specific
    use_moe: bool = True
    num_experts: int = 8
    moe_top_k: int = 1
    capacity_factor: float = 1.25
    load_balancing_weight: float = 0.01
    expert_parallel_size: Optional[int] = None
    
    # Training parameters
    micro_batch_size: Optional[int] = None
    effective_batch_size: int = 512
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    
    # Memory and precision
    precision: str = "auto"
    inference_precision: str = "auto"
    use_flash_attention: Optional[bool] = None
    gradient_checkpointing: bool = True
    
    # DeepSpeed configuration
    use_deepspeed: bool = True
    zero_stage: int = 0  # 0 = auto-select
    cpu_offload: bool = False
    cpu_offload_optimizer: bool = False
    cpu_offload_parameters: bool = False
    nvme_offload_optimizer: bool = False
    nvme_offload_parameters: bool = False
    aggressive_cpu_offload: bool = False
    gradient_compression: bool = False
    
    # Advanced settings
    compile: bool = True
    streaming_threshold_gb: float = 64.0
    
    def __post_init__(self):
        """Calculate derived parameters."""
        if self.gradient_accumulation_steps == 1:
            self.gradient_accumulation_steps = max(1, self.effective_batch_size // (self.micro_batch_size or 1))
    
    def validate(self):
        """Validate configuration parameters."""
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.use_moe and self.num_experts <= 1:
            raise ValueError("num_experts must be > 1 when using MoE")
        if self.moe_top_k > self.num_experts:
            raise ValueError("moe_top_k cannot exceed num_experts")
    
    def _get_gpu_memory_gb(self) -> Optional[float]:
        """Get GPU memory in GB."""
        if not torch.cuda.is_available():
            return None
        try:
            return torch.cuda.get_device_properties(0).total_memory / (1024**3)
        except:
            return None
    
    def _auto_select_precision(self) -> str:
        """Auto-select precision based on hardware."""
        if not torch.cuda.is_available():
            return "fp32"
        
        # Check GPU compute capability
        try:
            major, minor = torch.cuda.get_device_capability()
            if major >= 8:  # A100, H100, etc.
                return "bf16"
            elif major >= 7:  # V100, etc.
                return "fp16"
            else:
                return "fp32"
        except:
            return "fp16"
    
    def _estimate_parameters(self) -> int:
        """Estimate total model parameters."""
        if not self.use_moe:
            # Standard transformer
            embed_params = self.vocab_size * self.hidden_size
            layer_params = 12 * self.hidden_size**2  # Rough estimate
            total_params = embed_params + (self.num_layers * layer_params)
        else:
            # MoE transformer
            embed_params = self.vocab_size * self.hidden_size
            # Each expert has roughly 8 * hidden_size^2 parameters (2 linear layers)
            expert_params = self.num_experts * 8 * self.hidden_size**2
            other_layer_params = 4 * self.hidden_size**2  # Non-MoE parts
            total_layer_params = expert_params + other_layer_params
            total_params = embed_params + (self.num_layers * total_layer_params)
        
        return int(total_params)
    
    def get_active_parameters(self) -> int:
        """Get active parameters (for MoE, this is much smaller than total)."""
        if not self.use_moe:
            return self._estimate_parameters()
        
        total_params = self._estimate_parameters()
        # For MoE, active params = total params / num_experts * top_k (roughly)
        expert_reduction_factor = self.num_experts / self.moe_top_k
        active_params = int(total_params / expert_reduction_factor * (self.moe_top_k / self.num_experts))
        
        # More accurate calculation
        embed_params = self.vocab_size * self.hidden_size
        active_expert_params = self.moe_top_k * 8 * self.hidden_size**2
        other_layer_params = 4 * self.hidden_size**2
        active_layer_params = active_expert_params + other_layer_params
        active_total = embed_params + (self.num_layers * active_layer_params)
        
        return int(active_total)
    
    def get_memory_estimate_gb(self) -> Dict[str, float]:
        """Estimate memory usage in GB."""
        total_params = self._estimate_parameters()
        active_params = self.get_active_parameters()
        
        # Bytes per parameter based on precision
        if self.precision in ["fp16", "bf16"]:
            bytes_per_param = 2
        else:
            bytes_per_param = 4
        
        # Model memory (total parameters for storage, active for computation)
        model_memory = total_params * bytes_per_param / (1024**3)
        activation_memory = active_params * bytes_per_param * 2 / (1024**3)  # Rough estimate
        
        # Optimizer memory (typically 2x model for Adam)
        optimizer_memory = model_memory * 2 if not self.cpu_offload_optimizer else model_memory * 0.2
        
        # Gradient memory
        gradient_memory = model_memory if not self.cpu_offload else model_memory * 0.1
        
        total_memory = model_memory + activation_memory + optimizer_memory + gradient_memory
        
        return {
            'model': model_memory,
            'activations': activation_memory,
            'optimizer': optimizer_memory,
            'gradients': gradient_memory,
            'total': total_memory,
            'total_parameters': total_params,
            'active_parameters': active_params
        }


class ConfigPresets:
    """Predefined configurations for different model sizes."""
    
    @staticmethod
    def debug() -> Config:
        """Minimal configuration for debugging and testing."""
        return Config(
            hidden_size=512,
            num_layers=4,
            num_heads=8,
            seq_length=512,
            vocab_size=10000,
            micro_batch_size=2,
            effective_batch_size=16,
            precision="fp32",
            zero_stage=1,
            use_deepspeed=True,
            compile=False
        )
    
    @staticmethod
    def b1() -> Config:
        """1B active parameter model (8x1B total)."""
        return Config(
            hidden_size=1536,
            num_layers=24,
            num_heads=12,
            seq_length=2048,
            vocab_size=50000,
            micro_batch_size=4,
            effective_batch_size=128,
            precision="auto",
            zero_stage=2,
            gradient_checkpointing=True
        )
    
    @staticmethod
    def b7() -> Config:
        """7B active parameter model (8x7B total) - Mixtral-style."""
        return Config(
            hidden_size=4096,
            num_layers=32,
            num_heads=32,
            seq_length=4096,
            vocab_size=50000,
            micro_batch_size=2,
            effective_batch_size=256,
            precision="auto",
            zero_stage=0,  # Auto-select
            cpu_offload=False,
            use_flash_attention=True
        )
    
    @staticmethod
    def b14() -> Config:
        """14B active parameter model (8x14B total)."""
        return Config(
            hidden_size=5120,
            num_layers=40,
            num_heads=40,
            seq_length=4096,
            vocab_size=50000,
            micro_batch_size=1,
            effective_batch_size=512,
            precision="bf16",
            zero_stage=0,
            cpu_offload=True,
            use_flash_attention=True,
            streaming_threshold_gb=48.0
        )
    
    @staticmethod
    def b50() -> Config:
        """50B active parameter model (8x50B total)."""
        return Config(
            hidden_size=8192,
            num_layers=64,
            num_heads=64,
            seq_length=4096,
            vocab_size=50000,
            micro_batch_size=1,
            effective_batch_size=1024,
            precision="bf16",
            zero_stage=3,
            cpu_offload=True,
            cpu_offload_optimizer=True,
            nvme_offload_optimizer=True,
            use_flash_attention=True,
            streaming_threshold_gb=32.0
        )
    
    @staticmethod
    def b100() -> Config:
        """100B active parameter model (8x100B total)."""
        return Config(
            hidden_size=12288,
            num_layers=80,
            num_heads=96,
            seq_length=4096,
            vocab_size=100000,
            micro_batch_size=1,
            effective_batch_size=2048,
            precision="bf16",
            zero_stage=3,
            cpu_offload=True,
            cpu_offload_optimizer=True,
            cpu_offload_parameters=True,
            nvme_offload_optimizer=True,
            nvme_offload_parameters=True,
            use_flash_attention=True,
            streaming_threshold_gb=24.0
        )
    
    @staticmethod
    def b200() -> Config:
        """200B active parameter model (8x200B total)."""
        return Config(
            hidden_size=16384,
            num_layers=96,
            num_heads=128,
            seq_length=4096,
            vocab_size=100000,
            micro_batch_size=1,
            effective_batch_size=4096,
            precision="bf16",
            zero_stage=3,
            cpu_offload=True,
            cpu_offload_optimizer=True,
            cpu_offload_parameters=True,
            nvme_offload_optimizer=True,
            nvme_offload_parameters=True,
            aggressive_cpu_offload=True,
            use_flash_attention=True,
            streaming_threshold_gb=16.0
        )
    
    @staticmethod
    def b300() -> Config:
        """300B active parameter model (8x300B = 2.4T total)."""
        return Config(
            hidden_size=20480,
            num_layers=120,
            num_heads=160,
            seq_length=4096,
            vocab_size=100000,
            micro_batch_size=1,
            effective_batch_size=8192,
            precision="bf16",
            zero_stage=3,
            cpu_offload=True,
            cpu_offload_optimizer=True,
            cpu_offload_parameters=True,
            nvme_offload_optimizer=True,
            nvme_offload_parameters=True,
            aggressive_cpu_offload=True,
            use_flash_attention=True,
            streaming_threshold_gb=12.0
        )
    
    # Dense model variants (same active parameter count as MoE counterparts)
    
    @staticmethod
    def dense_1b() -> Config:
        """1B parameter dense model (equivalent active params to b1 MoE)."""
        return Config(
            hidden_size=2048,
            num_layers=24,
            num_heads=16,
            seq_length=2048,
            vocab_size=50000,
            micro_batch_size=2,
            effective_batch_size=128,
            precision="auto",
            zero_stage=1,
            cpu_offload=False,
            gradient_compression=False,
            use_moe=False,  # Dense model
            gradient_checkpointing=True,
            use_flash_attention=True
        )
    
    @staticmethod
    def dense_7b() -> Config:
        """7B parameter dense model (equivalent active params to b7 MoE)."""
        return Config(
            hidden_size=4096,
            num_layers=32,
            num_heads=32,
            seq_length=4096,
            vocab_size=50000,
            micro_batch_size=2,
            effective_batch_size=256,
            precision="auto",
            zero_stage=0,  # Auto-select
            cpu_offload=False,
            gradient_compression=False,
            use_moe=False,  # Dense model
            gradient_checkpointing=True,
            use_flash_attention=True
        )
    
    @staticmethod
    def dense_14b() -> Config:
        """14B parameter dense model (equivalent active params to b14 MoE)."""
        return Config(
            hidden_size=5120,
            num_layers=40,
            num_heads=40,
            seq_length=4096,
            vocab_size=50000,
            micro_batch_size=1,
            effective_batch_size=512,
            precision="auto",
            zero_stage=0,  # Auto-select
            cpu_offload=False,
            gradient_compression=True,
            use_moe=False,  # Dense model
            gradient_checkpointing=True,
            use_flash_attention=True,
            streaming_threshold_gb=25.0
        )
    
    @staticmethod
    def dense_50b() -> Config:
        """50B parameter dense model (equivalent active params to b50 MoE)."""
        return Config(
            hidden_size=8192,
            num_layers=64,
            num_heads=64,
            seq_length=4096,
            vocab_size=50000,
            micro_batch_size=1,
            effective_batch_size=1024,
            precision="bf16",
            zero_stage=3,
            cpu_offload=True,
            cpu_offload_optimizer=True,
            cpu_offload_parameters=True,
            aggressive_cpu_offload=True,
            gradient_compression=True,
            use_moe=False,  # Dense model
            gradient_checkpointing=True,
            use_flash_attention=True,
            streaming_threshold_gb=50.0
        )
    
    @staticmethod
    def dense_100b() -> Config:
        """100B parameter dense model (equivalent active params to b100 MoE)."""
        return Config(
            hidden_size=12288,
            num_layers=80,
            num_heads=96,
            seq_length=4096,
            vocab_size=100000,
            micro_batch_size=1,
            effective_batch_size=2048,
            precision="bf16",
            zero_stage=3,
            cpu_offload=True,
            cpu_offload_optimizer=True,
            cpu_offload_parameters=True,
            aggressive_cpu_offload=True,
            gradient_compression=True,
            use_moe=False,  # Dense model
            gradient_checkpointing=True,
            use_flash_attention=True,
            streaming_threshold_gb=100.0
        )
    
    @staticmethod
    def dense_200b() -> Config:
        """200B parameter dense model (equivalent active params to b200 MoE)."""
        return Config(
            hidden_size=16384,
            num_layers=100,
            num_heads=128,
            seq_length=4096,
            vocab_size=100000,
            micro_batch_size=1,
            effective_batch_size=4096,
            precision="bf16",
            zero_stage=3,
            cpu_offload=True,
            cpu_offload_optimizer=True,
            cpu_offload_parameters=True,
            aggressive_cpu_offload=True,
            nvme_offload_optimizer=True,
            nvme_offload_parameters=True,
            gradient_compression=True,
            use_moe=False,  # Dense model
            gradient_checkpointing=True,
            use_flash_attention=True,
            streaming_threshold_gb=200.0
        )
    
    @staticmethod
    def dense_300b() -> Config:
        """300B parameter dense model (equivalent active params to b300 MoE)."""
        return Config(
            hidden_size=20480,
            num_layers=120,
            num_heads=160,
            seq_length=4096,
            vocab_size=100000,
            micro_batch_size=1,
            effective_batch_size=8192,
            precision="bf16",
            zero_stage=3,
            cpu_offload=True,
            cpu_offload_optimizer=True,
            cpu_offload_parameters=True,
            aggressive_cpu_offload=True,
            nvme_offload_optimizer=True,
            nvme_offload_parameters=True,
            gradient_compression=True,
            use_moe=False,  # Dense model
            gradient_checkpointing=True,
            use_flash_attention=True,
            streaming_threshold_gb=300.0
        )
    
    @staticmethod
    def get_preset_info() -> Dict[str, Dict[str, Any]]:
        """Get comprehensive information about all available presets."""
        return {
            "debug": {
                "description": "Minimal configuration for debugging and testing",
                "use_case": "Development and debugging",
                "active_params": "~500K",
                "total_params": "~4M (8x500K)",
                "memory_usage": "Very Low (~10MB)",
                "training_speed": "Very Fast",
                "precision": "fp32 (debugging clarity)",
                "moe_pattern": "8x experts, top-1 routing",
                "deepspeed": "ZeRO-1",
                "hardware": "Any GPU with 2GB+"
            },
            "b1": {
                "description": "1B active parameter model (8x1B total)",
                "use_case": "Resource-constrained environments, experimentation",
                "active_params": "~1B",
                "total_params": "~8B (8x1B)",
                "memory_usage": "Medium (~4-8GB)",
                "training_speed": "Fast",
                "precision": "Auto-detected (fp16/bf16)",
                "moe_pattern": "8x experts, top-1 routing",
                "deepspeed": "ZeRO-2 with auto-optimization",
                "hardware": "RTX 3090/4090, A6000"
            },
            "b7": {
                "description": "7B active parameter model (8x7B total) - Mixtral-style",
                "use_case": "General purpose training, production applications",
                "active_params": "~7B",
                "total_params": "~56B (8x7B)",
                "memory_usage": "High (~28GB)",
                "training_speed": "Medium",
                "precision": "Auto-detected mixed precision",
                "moe_pattern": "8x experts, top-1 routing (Mixtral pattern)",
                "deepspeed": "Auto ZeRO stage with CPU offload",
                "hardware": "A100-40GB/80GB recommended"
            },
            "b14": {
                "description": "14B active parameter model (8x14B total)",
                "use_case": "High-performance applications, research",
                "active_params": "~14B",
                "total_params": "~112B (8x14B)",
                "memory_usage": "Very High (~56GB)",
                "training_speed": "Slow",
                "precision": "Mixed BF16 with TF32",
                "moe_pattern": "8x experts, top-1 routing",
                "deepspeed": "ZeRO-3 with CPU offload",
                "hardware": "Multiple A100-80GB or H100"
            },
            "b50": {
                "description": "50B active parameter model (8x50B total)",
                "use_case": "Large-scale research, advanced applications",
                "active_params": "~50B",
                "total_params": "~400B (8x50B)",
                "memory_usage": "Extreme (~200GB)",
                "training_speed": "Very Slow",
                "precision": "Mixed BF16 with TF32",
                "moe_pattern": "8x experts, top-1 routing",
                "deepspeed": "ZeRO-3 with aggressive CPU/NVMe offload",
                "hardware": "Multi-node A100/H100 clusters"
            },
            "b100": {
                "description": "100B active parameter model (8x100B total)",
                "use_case": "Cutting-edge research, enterprise applications",
                "active_params": "~100B",
                "total_params": "~800B (8x100B)",
                "memory_usage": "Extreme (~400GB)",
                "training_speed": "Very Slow",
                "precision": "Mixed BF16 with TF32",
                "moe_pattern": "8x experts, top-1 routing",
                "deepspeed": "ZeRO-3 with full offloading",
                "hardware": "Large multi-node H100 clusters"
            },
            "b200": {
                "description": "200B active parameter model (8x200B total)",
                "use_case": "Enterprise production, frontier research",
                "active_params": "~200B",
                "total_params": "~1.6T (8x200B)",
                "memory_usage": "Extreme (~800GB)",
                "training_speed": "Very Slow",
                "precision": "Mixed BF16 with TF32",
                "moe_pattern": "8x experts, top-1 routing",
                "deepspeed": "ZeRO-3 with NVMe offloading",
                "hardware": "Massive multi-node H100 clusters with NVMe"
            },
            "b300": {
                "description": "300B active parameter model (8x300B total)",
                "use_case": "Advanced research, state-of-the-art models",
                "active_params": "~300B", 
                "total_params": "~2.4T (8x300B)",
                "memory_usage": "Extreme (~1.2TB)",
                "training_speed": "Very Slow",
                "precision": "Mixed BF16 with TF32",
                "moe_pattern": "8x experts, top-1 routing",
                "deepspeed": "ZeRO-3 with full NVMe offloading",
                "hardware": "Supercomputer-scale H100 clusters"
            },
            "dense_1b": {
                "description": "1B parameter dense model (equivalent to b1 MoE active)",
                "use_case": "Resource-efficient baseline, MoE comparison",
                "active_params": "~1B",
                "total_params": "~1B (dense)",
                "memory_usage": "Low (~2-4GB)",
                "training_speed": "Fast",
                "precision": "Auto-detected (fp16/bf16)",
                "moe_pattern": "Dense (no experts)",
                "deepspeed": "ZeRO-1",
                "hardware": "RTX 3090/4090, A6000"
            },
            "dense_7b": {
                "description": "7B parameter dense model (equivalent to b7 MoE active)",
                "use_case": "Dense baseline for comparison with MoE",
                "active_params": "~7B",
                "total_params": "~7B (dense)",
                "memory_usage": "Medium (~14GB)",
                "training_speed": "Medium",
                "precision": "Auto-detected mixed precision",
                "moe_pattern": "Dense (no experts)",
                "deepspeed": "Auto ZeRO stage",
                "hardware": "A100-40GB/80GB"
            },
            "dense_14b": {
                "description": "14B parameter dense model (equivalent to b14 MoE active)",
                "use_case": "High-performance dense baseline",
                "active_params": "~14B",
                "total_params": "~14B (dense)",
                "memory_usage": "High (~28GB)",
                "training_speed": "Slow",
                "precision": "Mixed BF16 with TF32",
                "moe_pattern": "Dense (no experts)",
                "deepspeed": "ZeRO-2/3 with CPU offload",
                "hardware": "A100-80GB or H100"
            },
            "dense_50b": {
                "description": "50B parameter dense model (equivalent to b50 MoE active)",
                "use_case": "Large dense model for research comparison",
                "active_params": "~50B",
                "total_params": "~50B (dense)",
                "memory_usage": "Very High (~100GB)",
                "training_speed": "Very Slow",
                "precision": "Mixed BF16 with TF32",
                "moe_pattern": "Dense (no experts)",
                "deepspeed": "ZeRO-3 with aggressive CPU offload",
                "hardware": "Multi-GPU A100/H100 clusters"
            },
            "dense_100b": {
                "description": "100B parameter dense model (equivalent to b100 MoE active)",
                "use_case": "Very large dense baseline",
                "active_params": "~100B",
                "total_params": "~100B (dense)",
                "memory_usage": "Extreme (~200GB)",
                "training_speed": "Very Slow",
                "precision": "Mixed BF16 with TF32",
                "moe_pattern": "Dense (no experts)",
                "deepspeed": "ZeRO-3 with full offloading",
                "hardware": "Large multi-node H100 clusters"
            },
            "dense_200b": {
                "description": "200B parameter dense model (equivalent to b200 MoE active)",
                "use_case": "Enterprise-scale dense model",
                "active_params": "~200B",
                "total_params": "~200B (dense)",
                "memory_usage": "Extreme (~400GB)",
                "training_speed": "Very Slow",
                "precision": "Mixed BF16 with TF32",
                "moe_pattern": "Dense (no experts)",
                "deepspeed": "ZeRO-3 with NVMe offloading",
                "hardware": "Massive multi-node H100 clusters with NVMe"
            },
            "dense_300b": {
                "description": "300B parameter dense model (equivalent to b300 MoE active)",
                "use_case": "Largest dense model for frontier research",
                "active_params": "~300B",
                "total_params": "~300B (dense)",
                "memory_usage": "Extreme (~600GB)",
                "training_speed": "Very Slow",
                "precision": "Mixed BF16 with TF32",
                "moe_pattern": "Dense (no experts)",
                "deepspeed": "ZeRO-3 with full NVMe offloading",
                "hardware": "Supercomputer-scale H100 clusters"
            }
        }
    
    @staticmethod
    def compare_presets() -> Dict[str, Any]:
        """Compare all presets across key dimensions with 8x MoE considerations."""
        presets = {
            # MoE models
            'debug': ConfigPresets.debug(),
            'b1': ConfigPresets.b1(),
            'b7': ConfigPresets.b7(),
            'b14': ConfigPresets.b14(),
            'b50': ConfigPresets.b50(),
            'b100': ConfigPresets.b100(),
            'b200': ConfigPresets.b200(),
            'b300': ConfigPresets.b300(),
            # Dense models
            'dense_1b': ConfigPresets.dense_1b(),
            'dense_7b': ConfigPresets.dense_7b(),
            'dense_14b': ConfigPresets.dense_14b(),
            'dense_50b': ConfigPresets.dense_50b(),
            'dense_100b': ConfigPresets.dense_100b(),
            'dense_200b': ConfigPresets.dense_200b(),
            'dense_300b': ConfigPresets.dense_300b()
        }
        
        comparison = {}
        
        for name, config in presets.items():
            # Get memory estimates
            memory_est = config.get_memory_estimate_gb()
            
            # Determine model type
            model_type = "8x MoE" if config.use_moe else "Dense"
            
            comparison[name] = {
                'model_type': model_type,
                'active_parameters': memory_est['active_parameters'],
                'total_parameters': memory_est['total_parameters'], 
                'parameter_efficiency': f"{memory_est['active_parameters'] / memory_est['total_parameters']:.1%}",
                'hidden_size': config.hidden_size,
                'num_layers': config.num_layers,
                'sequence_length': config.seq_length,
                'micro_batch_size': config.micro_batch_size,
                'effective_batch_size': config.effective_batch_size,
                'learning_rate': config.learning_rate,
                'precision': config.precision,
                'moe_pattern': f"{config.num_experts}x experts, top-{config.moe_top_k}" if config.use_moe else "Dense (no experts)",
                'capacity_factor': config.capacity_factor if config.use_moe else "N/A",
                'load_balancing_weight': config.load_balancing_weight if config.use_moe else "N/A",
                'deepspeed_stage': f"ZeRO-{config.zero_stage}" if config.zero_stage > 0 else "Auto",
                'cpu_offload': config.cpu_offload,
                'nvme_offload': bool(config.nvme_offload_optimizer or config.nvme_offload_parameters),
                'gradient_compression': config.gradient_compression,
                'flash_attention': config.use_flash_attention,
                'estimated_memory_gb': memory_est['total'],
                'streaming_threshold_gb': config.streaming_threshold_gb
            }
        
        return comparison


class ConfigManager:
    """Enhanced configuration management with hardware optimization and validation for both MoE and dense models."""
    
    @staticmethod
    def create_config(
        preset: str = "b7",
        overrides: Optional[Dict[str, Any]] = None,
        optimize_for_hardware: bool = True,
        target_memory_usage: float = 0.9
    ) -> Config:
        """Create a configuration with optional overrides and hardware optimization."""
        
        # Get base configuration
        if hasattr(ConfigPresets, preset):
            config = getattr(ConfigPresets, preset)()
        else:
            available_presets = list(ConfigPresets.get_preset_info().keys())
            raise ValueError(f"Unknown preset: {preset}. Available: {available_presets}")
        
        # Apply overrides
        if overrides:
            for key, value in overrides.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                else:
                    print(f"Warning: Unknown config parameter '{key}' ignored")
        
        # Hardware optimization
        if optimize_for_hardware:
            config = ConfigManager.optimize_for_hardware(config, target_memory_usage)
        
        return config
    
    @staticmethod
    def optimize_for_hardware(config: Config, target_memory_usage: float = 0.9) -> Config:
        """Optimize configuration based on available hardware with both MoE and dense model support."""
        
        if not torch.cuda.is_available():
            print("Warning: CUDA not available, using CPU-optimized settings")
            config.precision = "fp32"
            config.use_deepspeed = False
            config.compile = False
            # Disable MoE for CPU-only setups
            if config.use_moe:
                config.use_moe = False
                print("Disabled MoE for CPU-only setup")
            return config
        
        # Get hardware info
        gpu_count = torch.cuda.device_count()
        gpu_memory_gb = config._get_gpu_memory_gb() or 8  # Default fallback
        total_memory_gb = gpu_memory_gb * gpu_count
        
        # Get memory estimate
        memory_est = config.get_memory_estimate_gb()
        required_memory = memory_est['total']
        
        # Identify model type
        model_type = "8x MoE" if config.use_moe else "Dense"
        print(f"Hardware detected: {gpu_count} GPUs, {gpu_memory_gb:.1f}GB each")
        print(f"Model: {model_type} with {memory_est['active_parameters']/1e9:.1f}B active params")
        print(f"Required memory: {required_memory:.1f}GB, Available: {total_memory_gb:.1f}GB")
        
        # Adjust micro batch size based on GPU memory and model size
        if config.micro_batch_size is None:
            # More sophisticated batch size calculation
            if memory_est['active_parameters'] > 100e9:  # >100B active params
                config.micro_batch_size = max(1, int(gpu_memory_gb / 20))
            elif memory_est['active_parameters'] > 50e9:  # >50B active params
                config.micro_batch_size = max(1, int(gpu_memory_gb / 15))
            elif memory_est['active_parameters'] > 10e9:  # >10B active params
                config.micro_batch_size = max(1, int(gpu_memory_gb / 10))
            else:
                config.micro_batch_size = max(1, int(gpu_memory_gb / 8))
        
        # MoE-specific optimizations
        if config.use_moe:
            # Optimize expert parallelism for 8x MoE pattern
            if hasattr(config, 'expert_parallel_size') and config.expert_parallel_size is None:
                # For 8 experts, try to distribute evenly across GPUs
                if gpu_count >= 8:
                    config.expert_parallel_size = 8  # One expert per GPU
                elif gpu_count >= 4:
                    config.expert_parallel_size = gpu_count  # Multiple experts per GPU
                else:
                    config.expert_parallel_size = 1  # All experts on each GPU
            
            # Adjust capacity factor based on available memory
            if hasattr(config, 'capacity_factor'):
                if required_memory > total_memory_gb * 0.8:
                    config.capacity_factor = min(config.capacity_factor, 1.0)
                    print("Reduced MoE capacity factor due to memory constraints")
        
        # Enable CPU offloading based on memory pressure
        memory_pressure = required_memory / total_memory_gb
        if memory_pressure > target_memory_usage:
            print(f"Enabling CPU offloading (memory pressure: {memory_pressure:.2f}x)")
            config.cpu_offload = True
            config.cpu_offload_optimizer = True
            
            if memory_pressure > 1.5:
                config.cpu_offload_parameters = True
                if hasattr(config, 'aggressive_cpu_offload'):
                    config.aggressive_cpu_offload = True
                print("Enabled aggressive CPU offloading")
            
            # Enable NVMe offloading for very large models
            if memory_pressure > 2.0:
                if hasattr(config, 'nvme_offload_optimizer'):
                    config.nvme_offload_optimizer = True
                if hasattr(config, 'nvme_offload_parameters'):
                    config.nvme_offload_parameters = True
                print("Enabled NVMe offloading for extreme memory constraints")
        
        # Auto-adjust ZeRO stage based on model size and memory
        if config.zero_stage == 0:  # Auto-select
            if memory_pressure > 2.0 or memory_est['total_parameters'] > 1e12:  # >1T params
                config.zero_stage = 3
                print("Selected ZeRO-3 for large model")
            elif memory_pressure > 1.2 or memory_est['total_parameters'] > 100e9:  # >100B params
                config.zero_stage = 2
                print("Selected ZeRO-2 for medium model")
            else:
                config.zero_stage = 1
                print("Selected ZeRO-1 for small model")
        
        # Enable gradient compression for multi-GPU setups
        if gpu_count > 4 and hasattr(config, 'gradient_compression'):
            config.gradient_compression = True
            print("Enabled gradient compression for multi-GPU setup")
        
        # Adjust precision based on GPU capabilities and model size
        if config.precision == "auto":
            config.precision = config._auto_select_precision()
            # Override for very large models
            if memory_est['total_parameters'] > 500e9:  # >500B params
                config.precision = "bf16"
        
        # Enable Flash Attention for large models
        if hasattr(config, 'use_flash_attention') and config.use_flash_attention is None:
            config.use_flash_attention = memory_est['active_parameters'] > 7e9  # >7B active params
        
        # Streaming optimizations for very large models
        if hasattr(config, 'streaming_threshold_gb'):
            if memory_est['total_parameters'] > 400e9:  # >400B params
                config.streaming_threshold_gb = min(config.streaming_threshold_gb, 32)
        
        # Dense model specific optimizations
        if not config.use_moe:
            # Dense models can use slightly larger batch sizes since they're more memory efficient
            if config.micro_batch_size == 1 and gpu_memory_gb > 40:  # Large GPUs
                config.micro_batch_size = 2
            
            # Dense models benefit more from gradient checkpointing
            if memory_est['total_parameters'] > 10e9:
                config.gradient_checkpointing = True
        
        print(f"Optimized config: {model_type}, ZeRO-{config.zero_stage}, {config.precision}, "
              f"CPU offload: {config.cpu_offload}, Micro batch: {config.micro_batch_size}")
        
        return config
    
    @staticmethod
    def validate_config(config: Config, strict: bool = False) -> List[str]:
        """Validate configuration with enhanced MoE and dense model checks."""
        warnings = []
        
        try:
            config.validate()
        except ValueError as e:
            if strict:
                raise
            warnings.append(f"Validation error: {e}")
        
        # Hardware compatibility checks
        if torch.cuda.is_available():
            gpu_memory_gb = config._get_gpu_memory_gb()
            gpu_count = torch.cuda.device_count()
            memory_est = config.get_memory_estimate_gb()
            
            per_gpu_memory = memory_est['total'] / gpu_count
            if gpu_memory_gb and per_gpu_memory > gpu_memory_gb * 0.9:
                warnings.append(f"High per-GPU memory usage: {per_gpu_memory:.1f}GB > {gpu_memory_gb * 0.9:.1f}GB")
                warnings.append("Consider enabling CPU offloading or reducing batch size")
            
            # Check for minimum GPU requirements for large models
            if memory_est['total_parameters'] > 400e9 and gpu_count < 4:
                warnings.append("Large models (>400B params) typically require 4+ GPUs for efficient training")
        
        # MoE-specific validations
        if config.use_moe:
            if hasattr(config, 'num_experts') and config.num_experts != 8:
                warnings.append("Non-standard expert count: 8 experts recommended for optimal 8x MoE pattern")
            
            if hasattr(config, 'moe_top_k') and config.moe_top_k != 1:
                warnings.append("Top-1 routing recommended for 8x MoE pattern efficiency and stability")
            
            # Check expert parallelism settings
            if (hasattr(config, 'expert_parallel_size') and 
                hasattr(config, 'num_experts') and 
                config.expert_parallel_size and
                config.expert_parallel_size > config.num_experts):
                warnings.append("Expert parallel size should not exceed number of experts")
            
            # Load balancing checks
            if hasattr(config, 'load_balancing_weight') and config.load_balancing_weight == 0:
                warnings.append("Consider enabling load balancing for MoE training stability")
        
        # Dense model specific validations
        else:
            if memory_est['total_parameters'] > 100e9 and not config.gradient_checkpointing:
                warnings.append("Consider enabling gradient checkpointing for large dense models")
                
            # Dense models can be more sensitive to batch size
            effective_batch = getattr(config, 'effective_batch_size', 0)
            if effective_batch < 64:
                warnings.append("Small batch sizes may hurt dense model training stability")
        
        # DeepSpeed compatibility
        if (hasattr(config, 'use_deepspeed') and config.use_deepspeed and 
            config.zero_stage > 2 and not config.cpu_offload):
            warnings.append("ZeRO-3 typically requires CPU offloading for optimal performance")
        
        # Precision warnings
        if (hasattr(config, 'precision') and config.precision == "fp32" and 
            hasattr(config, 'get_memory_estimate_gb')):
            memory_est = config.get_memory_estimate_gb()
            if memory_est['total_parameters'] > 50e9:
                warnings.append("Consider using mixed precision (bf16/fp16) for large models to reduce memory usage")
        
        # Batch size warnings
        effective_batch = getattr(config, 'effective_batch_size', 0)
        if effective_batch > 2048:
            warnings.append("Very large effective batch size may impact training dynamics")
        elif effective_batch < 32:
            warnings.append("Small effective batch size may impact training stability")
        
        return warnings
    
    @staticmethod
    def save_config_with_metadata(config: Config, path: str):
        """Save configuration with comprehensive metadata including both MoE and dense model information."""
        config_dict = asdict(config)
        
        # Get memory and parameter estimates
        memory_est = config.get_memory_estimate_gb()
        total_params = config._estimate_parameters()
        active_params = config.get_active_parameters()
        
        # Determine model architecture
        model_arch = "8x MoE (Mixture of Experts)" if config.use_moe else "Dense Transformer"
        
        # Add comprehensive metadata
        config_dict["_metadata"] = {
            "created": datetime.now().isoformat(),
            "lumina_version": "2.0",
            "framework": "PyTorch + DeepSpeed",
            "architecture": model_arch,
            "model_comparison": {
                "type": "MoE" if config.use_moe else "Dense",
                "efficiency_ratio": f"{active_params / total_params:.1%}" if config.use_moe else "100% (dense)",
                "memory_advantage": f"{(1 - active_params / total_params) * 100:.1f}% less active" if config.use_moe else "N/A"
            },
            "moe_pattern": f"{getattr(config, 'num_experts', 8)}x experts, top-{getattr(config, 'moe_top_k', 1)} routing" if config.use_moe else "Dense (no experts)",
            "estimated_parameters": {
                "total": int(total_params),
                "active": int(active_params),
                "efficiency": f"{active_params / total_params:.1%}",
                "expert_size": f"{active_params / getattr(config, 'num_experts', 1) / 1e9:.1f}B per expert" if config.use_moe else "N/A",
                "dense_equivalent": f"{active_params / 1e9:.1f}B dense model" if config.use_moe else f"{total_params / 1e9:.1f}B dense model"
            },
            "memory_estimate_gb": memory_est,
            "hardware_requirements": {
                "min_gpu_memory_gb": memory_est['total'] / torch.cuda.device_count() if torch.cuda.is_available() else "N/A",
                "recommended_gpus": max(1, int(memory_est['total'] / 40)),  # Assuming A100-40GB baseline
                "optimal_gpus": max(4, min(8, int(memory_est['total'] / 80))),  # H100-80GB optimal
                "supports_cpu_offload": getattr(config, 'cpu_offload', False),
                "supports_nvme_offload": bool(
                    getattr(config, 'nvme_offload_optimizer', False) or 
                    getattr(config, 'nvme_offload_parameters', False)
                ),
                "requires_multi_node": memory_est['total'] > 320,  # >4x H100-80GB
            },
            "training_characteristics": {
                "effective_batch_size": getattr(config, 'effective_batch_size', 'auto'),
                "micro_batch_size": getattr(config, 'micro_batch_size', 'auto'),
                "gradient_accumulation": getattr(config, 'gradient_accumulation_steps', 1),
                "sequence_length": getattr(config, 'seq_length', 'default'),
                "precision": getattr(config, 'precision', 'auto'),
                "deepspeed_stage": f"ZeRO-{config.zero_stage}" if config.zero_stage > 0 else "Auto",
                "flash_attention": getattr(config, 'use_flash_attention', False),
                "gradient_compression": getattr(config, 'gradient_compression', False),
                "gradient_checkpointing": getattr(config, 'gradient_checkpointing', False)
            }
        }
        
        # Add MoE or Dense specific metadata
        if config.use_moe:
            config_dict["_metadata"]["moe_specifics"] = {
                "expert_parallelism": getattr(config, 'expert_parallel_size', 'auto'),
                "capacity_factor": getattr(config, 'capacity_factor', 'default'),
                "load_balancing_weight": getattr(config, 'load_balancing_weight', 'default'),
                "top_k_routing": getattr(config, 'moe_top_k', 1),
                "streaming_threshold_gb": getattr(config, 'streaming_threshold_gb', 'default')
            }
        else:
            config_dict["_metadata"]["dense_specifics"] = {
                "total_flops_advantage": "More FLOPs but simpler routing vs MoE equivalent",
                "memory_usage": "Higher memory usage but no expert routing overhead",
                "architecture_simplicity": "Traditional transformer without expert routing complexity"
            }
        
        # Ensure directory exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False, indent=2)
        
        # Enhanced output information
        model_type = "MoE" if config.use_moe else "Dense"
        print(f"Configuration saved to {path}")
        print(f"Model: {model_type} - {active_params/1e9:.0f}B active params ({total_params/1e9:.0f}B total)")
        print(f"Active parameters: {active_params:,.0f}")
        print(f"Total parameters: {total_params:,.0f}")
        if config.use_moe:
            print(f"Parameter efficiency: {active_params / total_params:.1%}")
            print(f"Memory advantage vs dense: {(1 - active_params / total_params) * 100:.1f}%")
        print(f"Estimated memory: {memory_est['total']:.1f}GB")
        
        # Hardware recommendations
        if torch.cuda.is_available():
            recommended_gpus = max(1, int(memory_est['total'] / 40))
            optimal_gpus = max(1, int(memory_est['total'] / 80))
            print(f"Recommended GPUs: {recommended_gpus}x A100-40GB or {optimal_gpus}x H100-80GB")
    
    @staticmethod
    def get_preset_recommendations(target_use_case: str = "general") -> Dict[str, str]:
        """Get preset recommendations based on use case, including both MoE and dense options."""
        recommendations = {
            "debug": "debug - Minimal setup for testing and development",
            "experimentation": "b1 (MoE) or dense_1b - Good for research and prototyping",
            "general": "b7 (MoE) or dense_7b - Balanced performance for most applications",
            "production": "b14 (MoE) or dense_14b - High-quality results for production use", 
            "research": "b50+ (MoE) or dense_50b+ - State-of-the-art capabilities for advanced research",
            "enterprise": "b200+ (MoE) or dense_200b+ - Enterprise-scale applications with maximum quality",
            "moe_comparison": "Use matching pairs (e.g., b7 vs dense_7b) to compare MoE vs dense architectures",
            "efficiency_focused": "MoE variants (b1, b7, b14, etc.) - Better parameter efficiency",
            "simplicity_focused": "Dense variants (dense_1b, dense_7b, etc.) - Simpler architecture, no expert routing"
        }
        
        if target_use_case in recommendations:
            return {target_use_case: recommendations[target_use_case]}
        else:
            return recommendations
    
    @staticmethod
    def compare_moe_vs_dense(active_param_size: str) -> Dict[str, Any]:
        """Compare MoE vs Dense models with equivalent active parameter counts."""
        size_map = {
            "1b": ("b1", "dense_1b"),
            "7b": ("b7", "dense_7b"), 
            "14b": ("b14", "dense_14b"),
            "50b": ("b50", "dense_50b"),
            "100b": ("b100", "dense_100b"),
            "200b": ("b200", "dense_200b"),
            "300b": ("b300", "dense_300b")
        }
        
        if active_param_size not in size_map:
            raise ValueError(f"Unknown size: {active_param_size}. Available: {list(size_map.keys())}")
        
        moe_preset, dense_preset = size_map[active_param_size]
        moe_config = getattr(ConfigPresets, moe_preset)()
        dense_config = getattr(ConfigPresets, dense_preset)()
        
        moe_memory = moe_config.get_memory_estimate_gb()
        dense_memory = dense_config.get_memory_estimate_gb()
        
        return {
            "comparison_summary": {
                "active_params": f"{active_param_size.upper()} active parameters",
                "moe_total_params": f"{moe_memory['total_parameters']/1e9:.0f}B total",
                "dense_total_params": f"{dense_memory['total_parameters']/1e9:.0f}B total",
                "parameter_efficiency": f"MoE: {(moe_memory['active_parameters']/moe_memory['total_parameters'])*100:.1f}%, Dense: 100%",
                "memory_usage": f"MoE: {moe_memory['total']:.1f}GB, Dense: {dense_memory['total']:.1f}GB"
            },
            "moe_advantages": [
                "Parameter efficient - only activates subset of experts",
                "Potentially better scaling with model size",
                "Can achieve higher capacity with same compute",
                "Interesting for research into conditional computation"
            ],
            "dense_advantages": [
                "Simpler architecture and training",
                "No expert routing overhead",
                "More predictable memory usage",
                "Easier to debug and optimize"
            ],
            "moe_config": moe_preset,
            "dense_config": dense_preset,
            "detailed_comparison": ConfigPresets.compare_presets()
        }


# Example usage and utility functions
def main():
    """Example usage of the enhanced configuration system."""
    
    print("=" * 80)
    print("LUMINA 2.0 - Enhanced Configuration System")
    print("Supporting both MoE and Dense model architectures")
    print("=" * 80)
    
    # Show available presets
    print("\n Available Model Presets:")
    preset_info = ConfigPresets.get_preset_info()
    
    print("\n MoE Models (8x Expert Pattern):")
    moe_presets = ["debug", "b1", "b7", "b14", "b50", "b100", "b200", "b300"]
    for preset in moe_presets:
        if preset in preset_info:
            info = preset_info[preset]
            print(f"  {preset}: {info['active_params']} active, {info['total_params']} total - {info['use_case']}")
    
    print("\n Dense Models (Traditional Architecture):")
    dense_presets = ["dense_1b", "dense_7b", "dense_14b", "dense_50b", "dense_100b", "dense_200b", "dense_300b"]
    for preset in dense_presets:
        if preset in preset_info:
            info = preset_info[preset]
            print(f"  {preset}: {info['active_params']} params - {info['use_case']}")
    
    # Example: Create and optimize a configuration
    print(f"\n Example: Creating optimized b7 MoE configuration")
    try:
        config = ConfigManager.create_config("b7", optimize_for_hardware=True)
        print(f" Configuration created successfully")
        
        # Validate configuration
        warnings = ConfigManager.validate_config(config)
        if warnings:
            print(f" Validation warnings:")
            for warning in warnings[:3]:  # Show first 3 warnings
                print(f"  {warning}")
        else:
            print(" Configuration passed all validations")
            
    except Exception as e:
        print(f" Error creating configuration: {e}")
    
    # Example: Compare MoE vs Dense
    print(f"\n Example: Comparing 7B MoE vs Dense models")
    try:
        comparison = ConfigManager.compare_moe_vs_dense("7b")
        summary = comparison["comparison_summary"]
        print(f" Comparison Results:")
        print(f"  Active Parameters: {summary['active_params']}")
        print(f"  MoE Total: {summary['moe_total_params']}")
        print(f"  Dense Total: {summary['dense_total_params']}")
        print(f"  Efficiency: {summary['parameter_efficiency']}")
        print(f"  Memory Usage: {summary['memory_usage']}")
        
    except Exception as e:
        print(f" Error comparing models: {e}")
    
    # Show usage recommendations
    print(f"\n Usage Recommendations:")
    recommendations = ConfigManager.get_preset_recommendations()
    for use_case, recommendation in list(recommendations.items())[:5]:
        print(f"  {use_case.title()}: {recommendation}")
    
    print(f"\n Quick Start Commands:")
    print(f"  # Create MoE model")
    print(f"  config = ConfigManager.create_config('b7')")
    print(f"  ")
    print(f"  # Create equivalent dense model")  
    print(f"  config = ConfigManager.create_config('dense_7b')")
    print(f"  ")
    print(f"  # Compare architectures")
    print(f"  comparison = ConfigManager.compare_moe_vs_dense('7b')")
    print(f"  ")
    print(f"  # Save with metadata")
    print(f"  ConfigManager.save_config_with_metadata(config, 'config.yaml')")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()