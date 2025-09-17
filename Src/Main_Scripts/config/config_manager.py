# Copyright (c) 2025 Matias Nielsen. All rights reserved.
# Licensed under the Custom License below.

import yaml
import os
import psutil
import torch
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path


@dataclass
class Config:
    """Enhanced configuration with validation and serialization."""
    # Model architecture
    vocab_size: int = 50304
    hidden_size: int = 512
    num_layers: int = 8
    num_heads: int = 8
    num_kv_heads: int = 4
    seq_length: int = 1024
    intermediate_size: Optional[int] = None  # Auto-calculated if None
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    dropout: float = 0.0
    
    # Training parameters
    batch_size: int = 2
    micro_batch_size: Optional[int] = None  # Auto-calculated if None
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 3
    warmup_ratio: float = 0.15
    eval_every_n_batches: int = 500
    save_every_n_batches: int = 1000
    max_grad_norm: float = 1.0
    precision: str = "auto"  # Changed default to "auto"
    inference_precision: str = "auto"  # Changed default to "auto"
    compile: bool = False
    
    # Data parameters
    train_data_path: str = "data/train.jsonl"
    eval_data_path: str = "data/eval.jsonl"
    num_workers: int = 2
    assistant_loss_weight: float = 1.5
    max_conversations_per_file: int = 10000
    streaming_threshold_gb: float = 10.0  # Auto-enable streaming for large datasets
    prefetch_factor: int = 2
    pin_memory: bool = True
    
    # Generation parameters
    max_new_tokens: int = 512
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 50
    
    # Stability and optimization
    init_std: float = 0.02
    layer_norm_eps: float = 1e-5
    use_stable_embedding: bool = True
    gradient_checkpointing: bool = True
    tie_word_embeddings: bool = True
    use_flash_attention: bool = True  # New
    
    # MoE parameters - Updated to match 8x pattern
    use_moe: bool = True  # Default to True now
    num_experts: int = 8  # Fixed to 8 for consistency
    moe_top_k: int = 1   # Changed to 1 (top-1 routing like Mixtral)
    capacity_factor: float = 1.25
    load_balancing_weight: float = 0.01
    expert_parallel_size: Optional[int] = None  # Auto-calculated
    
    # DeepSpeed parameters - Enhanced
    use_deepspeed: bool = True
    zero_stage: int = 0  # 0 means auto-select based on model size
    cpu_offload: bool = False  # Auto-enabled if needed
    cpu_offload_optimizer: bool = False
    cpu_offload_parameters: bool = False  # New
    aggressive_cpu_offload: bool = False
    nvme_path: Optional[str] = None
    nvme_offload_optimizer: bool = False  # New
    nvme_offload_parameters: bool = False  # New
    gradient_compression: bool = False  # New
    communication_backend: str = "nccl"  # New
    overlap_comm: bool = True  # New
    contiguous_gradients: bool = True  # New
    allgather_bucket_size: int = 500000000  # New
    reduce_bucket_size: int = 500000000  # New
    
    # Production settings
    experiment_name: Optional[str] = None
    seed: int = 42
    log_level: str = "INFO"
    save_total_limit: int = 5
    early_stopping_patience: Optional[int] = None
    min_lr: float = 1e-6
    lr_scheduler: str = "cosine"
    
    # Monitoring and fault tolerance
    health_check_interval: int = 100
    auto_resume: bool = True
    backup_every_n_hours: int = 6
    max_retries: int = 3
    enable_wandb: bool = False  # New
    wandb_project: Optional[str] = None  # New
    wandb_entity: Optional[str] = None  # New
    
    # Advanced precision settings
    auto_tune_precision: bool = True  # Changed default to True
    precision_target: str = "balanced"
    dynamic_precision: bool = False
    tf32_enabled: Optional[bool] = None
    fp16_loss_scale: float = 65536.0  # New
    bf16_enabled: bool = True  # New
    
    # Memory optimization
    max_memory_usage: float = 0.9  # New: Use up to 90% of GPU memory
    memory_cleanup_interval: int = 1000  # New
    enable_cpu_adam: bool = False  # New
    partition_activations: bool = False  # New
    
    # Multi-node settings
    master_addr: Optional[str] = None  # New
    master_port: int = 29500  # New
    world_size: Optional[int] = None  # New
    rank: Optional[int] = None  # New
    local_rank: Optional[int] = None  # New
    
    # Data processing
    data_cache_dir: str = "data/cache"  # New
    tokenizer_cache_dir: str = "tokenizers/cache"  # New
    max_seq_length_percentile: float = 0.95  # New: Truncate outliers
    
    # Checkpointing enhancements
    save_optimizer_states: bool = True  # New
    checkpoint_compression: bool = True  # New
    async_save: bool = True  # New
    universal_checkpoint: bool = True  # New
    
    # Performance profiling
    profile_memory: bool = False  # New
    profile_communication: bool = False  # New
    log_throughput: bool = True  # New
    
    def __post_init__(self):
        self.validate()
        self._auto_configure()
        
    def _auto_configure(self):
        """Auto-configure settings based on hardware and model size."""
        if self.experiment_name is None:
            self.experiment_name = f"transformer_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Ensure vocab_size is divisible by 64 for efficiency
        if self.vocab_size % 64 != 0:
            self.vocab_size = ((self.vocab_size + 63) // 64) * 64
        
        # Auto-calculate intermediate_size if not specified (SwiGLU pattern)
        if self.intermediate_size is None:
            self.intermediate_size = int(self.hidden_size * 8 / 3)
            # Round to nearest multiple of 64 for efficiency
            self.intermediate_size = ((self.intermediate_size + 63) // 64) * 64
        
        # Auto-calculate micro_batch_size for DeepSpeed
        if self.micro_batch_size is None:
            self.micro_batch_size = max(1, self.batch_size // max(1, torch.cuda.device_count()))
        
        # Auto-configure expert parallelism
        if self.use_moe and self.expert_parallel_size is None:
            num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
            # Try to evenly distribute 8 experts across GPUs
            self.expert_parallel_size = min(self.num_experts, num_gpus)
        
        # Auto-configure DeepSpeed ZeRO stage based on model size
        if self.zero_stage == 0:
            estimated_params = self._estimate_parameters()
            gpu_memory_gb = self._get_gpu_memory_gb()
            
            if estimated_params > 50e9 or (gpu_memory_gb and estimated_params * 2 > gpu_memory_gb * 1e9):
                self.zero_stage = 3
                self.cpu_offload = True
            elif estimated_params > 10e9:
                self.zero_stage = 3
            elif estimated_params > 1e9:
                self.zero_stage = 2
            else:
                self.zero_stage = 1
        
        # Auto-enable aggressive optimizations for large models
        if self._estimate_parameters() > 50e9:
            self.gradient_compression = True
            self.cpu_offload_optimizer = True
            self.cpu_offload_parameters = True
            
        # Auto-configure precision based on hardware
        if self.precision == "auto":
            self.precision = self._auto_select_precision()
            
        if self.inference_precision == "auto":
            self.inference_precision = self._auto_select_precision(for_inference=True)
            
        # Auto-enable TF32 on Ampere+ GPUs
        if self.tf32_enabled is None:
            self.tf32_enabled = self._supports_tf32()
            
        # Calculate effective batch size
        world_size = self.world_size or (torch.cuda.device_count() if torch.cuda.is_available() else 1)
        self.effective_batch_size = self.micro_batch_size * self.gradient_accumulation_steps * world_size
        
        # Auto-configure data loading
        if self.num_workers == 2:  # Default value
            self.num_workers = min(os.cpu_count() or 4, 16)
            
        # Setup cache directories
        Path(self.data_cache_dir).mkdir(parents=True, exist_ok=True)
        Path(self.tokenizer_cache_dir).mkdir(parents=True, exist_ok=True)
    
    def _estimate_parameters(self) -> int:
        """Estimate total model parameters."""
        # Embedding parameters
        embed_params = self.vocab_size * self.hidden_size * 2  # input + output embeddings
        
        # Attention parameters per layer
        attention_params_per_layer = (
            4 * self.hidden_size * self.hidden_size +  # Q, K, V, O projections
            self.hidden_size  # norm
        )
        
        # Feed-forward parameters per layer (including MoE)
        if self.use_moe:
            ff_params_per_layer = (
                self.num_experts * 3 * self.hidden_size * self.intermediate_size +  # gate, up, down for each expert
                self.hidden_size * self.num_experts +  # routing layer
                self.hidden_size  # norm
            )
        else:
            ff_params_per_layer = (
                3 * self.hidden_size * self.intermediate_size +  # gate, up, down
                self.hidden_size  # norm
            )
        
        total_params = (
            embed_params + 
            self.num_layers * (attention_params_per_layer + ff_params_per_layer)
        )
        
        return total_params
    
    def _get_gpu_memory_gb(self) -> Optional[float]:
        """Get available GPU memory in GB."""
        if not torch.cuda.is_available():
            return None
        try:
            props = torch.cuda.get_device_properties(0)
            return props.total_memory / (1024**3)
        except:
            return None
    
    def _auto_select_precision(self, for_inference: bool = False) -> str:
        """Auto-select precision based on hardware capabilities."""
        if not torch.cuda.is_available():
            return "fp32"
            
        # Check GPU architecture
        if torch.cuda.get_device_capability()[0] >= 8:  # A100, H100, etc.
            return "mixed_bf16" if not for_inference else "bf16"
        elif torch.cuda.get_device_capability()[0] >= 7:  # V100, RTX series
            return "mixed_fp16" if not for_inference else "fp16"
        else:
            return "fp32"
    
    def _supports_tf32(self) -> bool:
        """Check if current GPU supports TF32."""
        if not torch.cuda.is_available():
            return False
        return torch.cuda.get_device_capability()[0] >= 8
    
    def get_active_parameters(self) -> int:
        """Get number of active parameters (important for MoE models)."""
        if not self.use_moe:
            return self._estimate_parameters()
        
        # For MoE, only a fraction of experts are active per token
        total_params = self._estimate_parameters()
        
        # Calculate non-expert parameters
        embed_params = self.vocab_size * self.hidden_size * 2
        attention_params = self.num_layers * (4 * self.hidden_size * self.hidden_size + self.hidden_size)
        routing_params = self.num_layers * self.hidden_size * self.num_experts
        norm_params = self.num_layers * self.hidden_size
        
        non_expert_params = embed_params + attention_params + routing_params + norm_params
        
        # Calculate expert parameters (only top-k are active)
        expert_params_per_layer = self.num_experts * 3 * self.hidden_size * self.intermediate_size
        total_expert_params = self.num_layers * expert_params_per_layer
        active_expert_params = self.num_layers * self.moe_top_k * 3 * self.hidden_size * self.intermediate_size
        
        return int(non_expert_params + active_expert_params)
    
    def validate(self):
        """Enhanced validation with better error messages."""
        # Basic architecture validation
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(f"hidden_size ({self.hidden_size}) must be divisible by num_heads ({self.num_heads})")
        
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(f"num_heads ({self.num_heads}) must be divisible by num_kv_heads ({self.num_kv_heads})")
        
        # Precision validation
        valid_precisions = ["fp32", "fp16", "bf16", "mixed_fp16", "mixed_bf16", "tf32", "auto"]
        if self.precision not in valid_precisions:
            raise ValueError(f"Invalid precision: {self.precision}. Valid options: {valid_precisions}")
        
        if self.inference_precision not in valid_precisions + ["dynamic"]:
            raise ValueError(f"Invalid inference_precision: {self.inference_precision}")
        
        # Training parameters validation
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        
        if not (0 <= self.warmup_ratio <= 1):
            raise ValueError("Warmup ratio must be between 0 and 1")
        
        # MoE validation - Updated for 8x pattern
        if self.use_moe:
            if self.num_experts != 8:
                raise ValueError("num_experts must be 8 for consistency with 8x pattern")
            
            if self.moe_top_k not in [1, 2]:
                raise ValueError("moe_top_k should be 1 or 2 (1 recommended for 8x pattern)")
            
            if self.moe_top_k > self.num_experts:
                raise ValueError("moe_top_k cannot exceed num_experts")
            
            if self.capacity_factor < 1.0:
                raise ValueError("capacity_factor must be at least 1.0")
                
            if self.expert_parallel_size and self.expert_parallel_size > self.num_experts:
                raise ValueError("expert_parallel_size cannot exceed num_experts")
        
        # DeepSpeed validation
        if self.use_deepspeed:
            if self.zero_stage not in [0, 1, 2, 3]:
                raise ValueError("zero_stage must be 0 (auto), 1, 2, or 3")
            
            if self.nvme_path and not Path(self.nvme_path).exists():
                raise ValueError(f"NVMe path does not exist: {self.nvme_path}")
        
        # Memory validation
        if not (0.1 <= self.max_memory_usage <= 1.0):
            raise ValueError("max_memory_usage must be between 0.1 and 1.0")
        
        # Data validation
        if self.streaming_threshold_gb <= 0:
            raise ValueError("streaming_threshold_gb must be positive")
    
    def get_memory_estimate_gb(self) -> Dict[str, float]:
        """Estimate memory usage in GB for different components."""
        params = self._estimate_parameters()
        active_params = self.get_active_parameters()
        
        # Parameter memory (in bytes)
        if self.precision in ["fp16", "bf16", "mixed_fp16", "mixed_bf16"]:
            param_bytes = params * 2
        else:
            param_bytes = params * 4
        
        # Gradient memory (same as parameters)
        grad_bytes = param_bytes
        
        # Optimizer states (Adam: 2x parameters for momentum and variance)
        optimizer_bytes = params * 4 * 2  # Always fp32 for stability
        
        # Activations (rough estimate)
        activation_bytes = (
            self.batch_size * self.seq_length * self.hidden_size * self.num_layers * 4
        )  # Rough approximation
        
        # Convert to GB
        param_gb = param_bytes / (1024**3)
        grad_gb = grad_bytes / (1024**3)
        optimizer_gb = optimizer_bytes / (1024**3)
        activation_gb = activation_bytes / (1024**3)
        
        total_gb = param_gb + grad_gb + optimizer_gb + activation_gb
        
        # Apply DeepSpeed optimizations
        if self.use_deepspeed:
            if self.zero_stage >= 2:
                grad_gb = grad_gb / max(1, torch.cuda.device_count())
            if self.zero_stage >= 3:
                param_gb = param_gb / max(1, torch.cuda.device_count())
            if self.cpu_offload_optimizer:
                optimizer_gb = 0  # Moved to CPU
        
        return {
            "parameters": param_gb,
            "gradients": grad_gb, 
            "optimizer": optimizer_gb,
            "activations": activation_gb,
            "total": param_gb + grad_gb + optimizer_gb + activation_gb,
            "active_parameters": active_params,
            "total_parameters": params
        }
    
    def save(self, path: str):
        """Save configuration to file with metadata."""
        config_dict = asdict(self)
        
        # Add metadata
        config_dict["_metadata"] = {
            "created": datetime.now().isoformat(),
            "lumina_version": "2.0",
            "estimated_parameters": self._estimate_parameters(),
            "active_parameters": self.get_active_parameters(),
            "memory_estimate": self.get_memory_estimate_gb()
        }
        
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    
    @classmethod
    def load(cls, path: str) -> 'Config':
        """Load configuration from file."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Remove metadata if present
        config_dict.pop("_metadata", None)
        
        return cls(**config_dict)
    
    def to_deepspeed_config(self) -> Dict[str, Any]:
        """Generate DeepSpeed configuration."""
        ds_config = {
            "train_batch_size": self.effective_batch_size,
            "train_micro_batch_size_per_gpu": self.micro_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "gradient_clipping": self.max_grad_norm,
            
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": self.learning_rate,
                    "weight_decay": self.weight_decay,
                    "betas": [0.9, 0.95],
                    "eps": 1e-8
                }
            },
            
            "scheduler": {
                "type": "WarmupDecayLR",
                "params": {
                    "warmup_min_lr": self.min_lr,
                    "warmup_max_lr": self.learning_rate,
                    "warmup_num_steps": int(self.warmup_ratio * 10000),  # Rough estimate
                    "total_num_steps": 10000  # Will be updated during training
                }
            },
            
            "communication_data_type": "fp32",
            "gradient_compression": {
                "enabled": self.gradient_compression
            },
            
            "wall_clock_breakdown": False,
            "memory_breakdown": self.profile_memory
        }
        
        # Precision settings
        if self.precision in ["fp16", "mixed_fp16"]:
            ds_config["fp16"] = {
                "enabled": True,
                "loss_scale": self.fp16_loss_scale,
                "loss_scale_window": 1000,
                "hysteresis": 2,
                "consecutive_hysteresis": False,
                "auto_cast": False
            }
        elif self.precision in ["bf16", "mixed_bf16"]:
            ds_config["bf16"] = {
                "enabled": True
            }
        
        # ZeRO configuration
        if self.zero_stage > 0:
            zero_config = {
                "stage": self.zero_stage,
                "overlap_comm": self.overlap_comm,
                "contiguous_gradients": self.contiguous_gradients,
                "sub_group_size": 1000000000,
                "reduce_bucket_size": self.reduce_bucket_size,
                "allgather_partitions": True,
                "reduce_scatter": True,
                "allgather_bucket_size": self.allgather_bucket_size,
                "stage3_prefetch_bucket_size": 50000000,
                "stage3_param_persistence_threshold": 100000,
                "stage3_max_live_parameters": 1000000000,
                "stage3_max_reuse_distance": 1000000000
            }
            
            if self.cpu_offload:
                zero_config["offload_optimizer"] = {
                    "device": "cpu",
                    "pin_memory": True
                }
                
            if self.cpu_offload_parameters:
                zero_config["offload_param"] = {
                    "device": "cpu",
                    "pin_memory": True
                }
            
            if self.nvme_path:
                if self.nvme_offload_optimizer:
                    zero_config["offload_optimizer"] = {
                        "device": "nvme",
                        "nvme_path": self.nvme_path,
                        "pin_memory": True
                    }
                if self.nvme_offload_parameters:
                    zero_config["offload_param"] = {
                        "device": "nvme", 
                        "nvme_path": self.nvme_path,
                        "pin_memory": True
                    }
            
            ds_config["zero_optimization"] = zero_config
        
        # Activation checkpointing
        if self.gradient_checkpointing:
            ds_config["activation_checkpointing"] = {
                "partition_activations": self.partition_activations,
                "contiguous_memory_optimization": True,
                "cpu_checkpointing": False,
                "number_checkpoints": 4,
                "synchronize_checkpoint_boundary": False,
                "profile": False
            }
        
        return ds_config


class ConfigPresets:
    """Updated configuration presets following 8x MoE pattern."""
    
    @staticmethod
    def debug() -> Config:
        """Minimal config for debugging and testing."""
        return Config(
            # Tiny model for fast iteration
            vocab_size=1024,
            hidden_size=128,
            num_layers=2,
            num_heads=2,
            num_kv_heads=1,
            seq_length=256,
            intermediate_size=256,
            
            # Fast training settings
            batch_size=2,
            micro_batch_size=1,
            gradient_accumulation_steps=2,
            num_epochs=1,
            learning_rate=5e-5,
            weight_decay=0.01,
            eval_every_n_batches=50,
            save_every_n_batches=100,
            precision="fp32",
            inference_precision="fp32",
            compile=False,
            num_workers=0,
            
            # MoE settings - 8x pattern but smaller
            use_moe=True,
            num_experts=8,
            moe_top_k=1,  # Top-1 routing
            capacity_factor=1.1,
            load_balancing_weight=0.005,
            expert_parallel_size=2,
            
            # DeepSpeed settings
            use_deepspeed=True,
            zero_stage=1,
            cpu_offload=False,
            
            # Monitoring and stability
            experiment_name="debug_run",
            log_level="DEBUG",
            health_check_interval=10,
            save_total_limit=3,
            early_stopping_patience=None,
            max_retries=1,
            lr_scheduler="cosine",
            gradient_checkpointing=False,
            use_flash_attention=False,
            
            # Precision settings
            auto_tune_precision=False,
            precision_target="balanced",
            dynamic_precision=False,
            
            # Memory settings
            max_memory_usage=0.7,
            streaming_threshold_gb=1.0
        )
    
    @staticmethod
    def b1() -> Config:
        """1B active parameter model (8x1B = 8B total)."""
        return Config(
            # Model architecture for ~1B active parameters
            hidden_size=1536,
            num_layers=16,
            num_heads=12,
            num_kv_heads=4,
            seq_length=2048,
            intermediate_size=None,  # Auto-calculated
            
            # Training settings
            batch_size=8,
            micro_batch_size=1,
            gradient_accumulation_steps=4,
            num_epochs=3,
            learning_rate=3e-4,
            weight_decay=0.01,
            eval_every_n_batches=500,
            save_every_n_batches=1000,
            precision="auto",
            inference_precision="auto",
            compile=True,
            num_workers=4,
            
            # MoE settings - 8x pattern
            use_moe=True,
            num_experts=8,
            moe_top_k=1,
            capacity_factor=1.25,
            load_balancing_weight=0.01,
            
            # DeepSpeed settings
            use_deepspeed=True,
            zero_stage=2,
            cpu_offload=False,
            gradient_compression=False,
            
            # Production settings
            experiment_name="b1_8x1b",
            log_level="INFO",
            health_check_interval=100,
            save_total_limit=5,
            early_stopping_patience=5,
            backup_every_n_hours=12,
            lr_scheduler="cosine",
            gradient_checkpointing=True,
            use_flash_attention=True,
            
            # Precision settings
            auto_tune_precision=True,
            precision_target="balanced",
            dynamic_precision=False,
            
            # Memory settings
            max_memory_usage=0.85,
            streaming_threshold_gb=5.0
        )
    
    @staticmethod
    def b7() -> Config:
        """7B active parameter model (8x7B = 56B total) - Mixtral-style."""
        return Config(
            # Model architecture for ~7B active parameters
            hidden_size=4096,
            num_layers=32,
            num_heads=32,
            num_kv_heads=8,
            seq_length=4096,
            intermediate_size=None,  # Auto-calculated to ~11008
            
            # Training settings
            batch_size=16,
            micro_batch_size=1,
            gradient_accumulation_steps=8,
            num_epochs=3,
            learning_rate=1e-4,
            weight_decay=0.01,
            eval_every_n_batches=1000,
            save_every_n_batches=2000,
            precision="auto",
            inference_precision="auto",
            compile=True,
            num_workers=8,
            
            # MoE settings - 8x pattern like Mixtral
            use_moe=True,
            num_experts=8,
            moe_top_k=1,  # Top-1 routing for efficiency
            capacity_factor=1.25,
            load_balancing_weight=0.01,
            
            # DeepSpeed settings
            use_deepspeed=True,
            zero_stage=0,  # Auto-select
            cpu_offload=False,  # Auto-enabled if needed
            gradient_compression=False,
            overlap_comm=True,
            contiguous_gradients=True,
            
            # Production settings
            experiment_name="b7_8x7b_mixtral",
            log_level="INFO",
            health_check_interval=100,
            save_total_limit=10,
            early_stopping_patience=10,
            backup_every_n_hours=6,
            lr_scheduler="cosine",
            gradient_checkpointing=True,
            use_flash_attention=True,
            
            # Precision settings
            auto_tune_precision=True,
            precision_target="balanced",
            dynamic_precision=False,
            
            # Memory settings
            max_memory_usage=0.9,
            streaming_threshold_gb=10.0,
            
            # Monitoring
            enable_wandb=False,
            profile_memory=False,
            log_throughput=True
        )
    
    @staticmethod
    def b14() -> Config:
        """14B active parameter model (8x14B = 112B total)."""
        return Config(
            # Model architecture for ~14B active parameters
            hidden_size=5120,
            num_layers=40,
            num_heads=40,
            num_kv_heads=10,
            seq_length=4096,
            intermediate_size=None,  # Auto-calculated
            
            # Training settings
            batch_size=32,
            micro_batch_size=1,
            gradient_accumulation_steps=16,
            num_epochs=2,
            learning_rate=8e-5,
            weight_decay=0.01,
            eval_every_n_batches=2000,
            save_every_n_batches=5000,
            precision="auto",
            inference_precision="auto",
            compile=True,
            num_workers=8,
            
            # MoE settings - 8x pattern
            use_moe=True,
            num_experts=8,
            moe_top_k=1,
            capacity_factor=1.25,
            load_balancing_weight=0.015,
            
            # DeepSpeed settings
            use_deepspeed=True,
            zero_stage=0,  # Auto-select (likely ZeRO-3)
            cpu_offload=False,  # Auto-enabled if needed
            gradient_compression=True,
            overlap_comm=True,
            contiguous_gradients=True,
            
            # Production settings
            experiment_name="b14_8x14b",
            log_level="INFO",
            health_check_interval=200,
            save_total_limit=15,
            early_stopping_patience=15,
            backup_every_n_hours=4,
            lr_scheduler="cosine",
            gradient_checkpointing=True,
            use_flash_attention=True,
            warmup_ratio=0.05,
            
            # Precision settings
            auto_tune_precision=False,
            precision_target="quality",
            dynamic_precision=False,
            tf32_enabled=None,  # Auto-detect
            
            # Memory settings
            max_memory_usage=0.9,
            streaming_threshold_gb=25.0,
            enable_cpu_adam=False,
            
            # Monitoring
            enable_wandb=False,
            profile_memory=True,
            log_throughput=True
        )
    
    @staticmethod
    def b50() -> Config:
        """50B active parameter model (8x50B = 400B total)."""
        return Config(
            # Model architecture for ~50B active parameters
            hidden_size=8192,
            num_layers=64,
            num_heads=64,
            num_kv_heads=16,
            seq_length=128000,
            intermediate_size=None,  # Auto-calculated
            
            # Training settings
            batch_size=128,
            micro_batch_size=1,
            gradient_accumulation_steps=32,
            num_epochs=2,
            learning_rate=6e-5,
            weight_decay=0.01,
            eval_every_n_batches=3000,
            save_every_n_batches=8000,
            precision="auto",
            inference_precision="auto",
            compile=True,
            num_workers=12,
            
            # MoE settings - 8x pattern
            use_moe=True,
            num_experts=8,
            moe_top_k=1,
            capacity_factor=1.25,
            load_balancing_weight=0.02,
            
            # DeepSpeed settings - Aggressive optimization needed
            use_deepspeed=True,
            zero_stage=3,
            cpu_offload=True,
            cpu_offload_optimizer=True,
            cpu_offload_parameters=True,
            aggressive_cpu_offload=True,
            gradient_compression=True,
            overlap_comm=True,
            contiguous_gradients=True,
            
            # Production settings
            experiment_name="b50_8x50b",
            log_level="INFO",
            health_check_interval=300,
            save_total_limit=18,
            early_stopping_patience=18,
            backup_every_n_hours=3,
            lr_scheduler="cosine",
            gradient_checkpointing=True,
            use_flash_attention=True,
            warmup_ratio=0.08,
            
            # Precision settings
            auto_tune_precision=False,
            precision_target="quality",
            dynamic_precision=False,
            tf32_enabled=True,
            
            # Memory settings
            max_memory_usage=0.95,
            streaming_threshold_gb=50.0,
            enable_cpu_adam=True,
            partition_activations=True,
            
            # Advanced checkpointing
            universal_checkpoint=True,
            checkpoint_compression=True,
            async_save=True,
            
            # Monitoring
            enable_wandb=False,
            profile_memory=True,
            profile_communication=True,
            log_throughput=True
        )
    
    @staticmethod
    def b100() -> Config:
        """100B active parameter model (8x100B = 800B total)."""
        return Config(
            # Model architecture for ~100B active parameters  
            hidden_size=12288,
            num_layers=80,
            num_heads=96,
            num_kv_heads=24,
            seq_length=200000,
            intermediate_size=None,  # Auto-calculated
            
            # Training settings
            batch_size=256,
            micro_batch_size=1,
            gradient_accumulation_steps=64,
            num_epochs=2,
            learning_rate=5e-5,
            weight_decay=0.01,
            eval_every_n_batches=5000,
            save_every_n_batches=10000,
            precision="auto",
            inference_precision="auto",
            compile=True,
            num_workers=16,
            
            # MoE settings - 8x pattern
            use_moe=True,
            num_experts=8,
            moe_top_k=1,
            capacity_factor=1.25,
            load_balancing_weight=0.025,
            
            # DeepSpeed settings - Maximum optimization
            use_deepspeed=True,
            zero_stage=3,
            cpu_offload=True,
            cpu_offload_optimizer=True,
            cpu_offload_parameters=True,
            aggressive_cpu_offload=True,
            gradient_compression=True,
            overlap_comm=True,
            contiguous_gradients=True,
            
            # Production settings
            experiment_name="b100_8x100b",
            log_level="INFO",
            health_check_interval=500,
            save_total_limit=20,
            early_stopping_patience=20,
            backup_every_n_hours=2,
            lr_scheduler="cosine",
            gradient_checkpointing=True,
            use_flash_attention=True,
            warmup_ratio=0.1,
            
            # Precision settings
            auto_tune_precision=False,
            precision_target="quality",
            dynamic_precision=False,
            tf32_enabled=True,
            
            # Memory settings
            max_memory_usage=0.95,
            streaming_threshold_gb=100.0,
            enable_cpu_adam=True,
            partition_activations=True,
            
            # Advanced settings
            universal_checkpoint=True,
            checkpoint_compression=True,
            async_save=True,
            
            # Monitoring
            enable_wandb=False,
            profile_memory=True,
            profile_communication=True,
            log_throughput=True
        )
    
    @staticmethod
    def b200() -> Config:
        """200B active parameter model (8x200B = 1.6T total)."""
        return Config(
            # Model architecture for ~200B active parameters
            hidden_size=16384,
            num_layers=100,
            num_heads=128,
            num_kv_heads=32,
            seq_length=1000000,
            intermediate_size=None,  # Auto-calculated
            
            # Training settings
            batch_size=256,
            micro_batch_size=1,
            gradient_accumulation_steps=64,
            num_epochs=2,
            learning_rate=4e-5,
            weight_decay=0.01,
            eval_every_n_batches=5000,
            save_every_n_batches=10000,
            precision="mixed_bf16",
            inference_precision="bf16",
            compile=True,
            num_workers=16,
            
            # MoE settings - 8x pattern
            use_moe=True,
            num_experts=8,
            moe_top_k=1,
            capacity_factor=1.25,
            load_balancing_weight=0.03,
            
            # DeepSpeed settings - Maximum optimization + NVMe
            use_deepspeed=True,
            zero_stage=3,
            cpu_offload=True,
            cpu_offload_optimizer=True,
            cpu_offload_parameters=True,
            aggressive_cpu_offload=True,
            nvme_offload_optimizer=True,
            nvme_offload_parameters=True,
            gradient_compression=True,
            overlap_comm=True,
            contiguous_gradients=True,
            
            # Production settings
            experiment_name="b200_8x200b",
            log_level="INFO",
            health_check_interval=500,
            save_total_limit=20,
            early_stopping_patience=20,
            backup_every_n_hours=2,
            lr_scheduler="cosine",
            gradient_checkpointing=True,
            use_flash_attention=True,
            warmup_ratio=0.1,
            
            # Precision settings
            auto_tune_precision=False,
            precision_target="production",
            dynamic_precision=False,
            tf32_enabled=True,
            
            # Memory settings
            max_memory_usage=0.95,
            streaming_threshold_gb=200.0,
            enable_cpu_adam=True,
            partition_activations=True,
            
            # Advanced settings
            universal_checkpoint=True,
            checkpoint_compression=True,
            async_save=True,
            
            # Monitoring
            enable_wandb=False,
            profile_memory=True,
            profile_communication=True,
            log_throughput=True
        )
    
    @staticmethod
    def b300() -> Config:
        """300B active parameter model (8x300B = 2.4T total)."""
        return Config(
            # Model architecture for ~300B active parameters
            hidden_size=20480,
            num_layers=120,
            num_heads=160,
            num_kv_heads=40,
            seq_length=204800,
            intermediate_size=None,  # Auto-calculated
            
            # Training settings
            batch_size=256,
            micro_batch_size=1,
            gradient_accumulation_steps=64,
            num_epochs=2,
            learning_rate=3e-5,
            weight_decay=0.01,
            eval_every_n_batches=5000,
            save_every_n_batches=10000,
            precision="mixed_bf16",
            inference_precision="bf16",
            compile=True,
            num_workers=16,
            
            # MoE settings - 8x pattern
            use_moe=True,
            num_experts=8,
            moe_top_k=1,
            capacity_factor=1.25,
            load_balancing_weight=0.035,
            
            # DeepSpeed settings - Ultimate optimization
            use_deepspeed=True,
            zero_stage=3,
            cpu_offload=True,
            cpu_offload_optimizer=True,
            cpu_offload_parameters=True,
            aggressive_cpu_offload=True,
            nvme_offload_optimizer=True,
            nvme_offload_parameters=True,
            gradient_compression=True,
            overlap_comm=True,
            contiguous_gradients=True,
            
            # Production settings
            experiment_name="b300_8x300b",
            log_level="INFO",
            health_check_interval=500,
            save_total_limit=20,
            early_stopping_patience=20,
            backup_every_n_hours=2,
            lr_scheduler="cosine",
            gradient_checkpointing=True,
            use_flash_attention=True,
            warmup_ratio=0.1,
            
            # Precision settings
            auto_tune_precision=False,
            precision_target="quality",
            dynamic_precision=False,
            tf32_enabled=True,
            
            # Memory settings
            max_memory_usage=0.95,
            streaming_threshold_gb=300.0,
            enable_cpu_adam=True,
            partition_activations=True,
            
            # Advanced settings
            universal_checkpoint=True,
            checkpoint_compression=True,
            async_save=True,
            
            # Monitoring
            enable_wandb=False,
            profile_memory=True,
            profile_communication=True,
            log_throughput=True
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
            }
        }
    
    @staticmethod
    def compare_presets() -> Dict[str, Any]:
        """Compare all presets across key dimensions with 8x MoE considerations."""
        presets = {
            'debug': ConfigPresets.debug(),
            'b1': ConfigPresets.b1(),
            'b7': ConfigPresets.b7(),
            'b14': ConfigPresets.b14(),
            'b50': ConfigPresets.b50(),
            'b100': ConfigPresets.b100(),
            'b200': ConfigPresets.b200(),
            'b300': ConfigPresets.b300()
        }
        
        comparison = {}
        
        for name, config in presets.items():
            # Get memory estimates
            memory_est = config.get_memory_estimate_gb()
            
            comparison[name] = {
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
                'inference_precision': config.inference_precision,
                'moe_pattern': f"{config.num_experts}x experts, top-{config.moe_top_k}",
                'capacity_factor': config.capacity_factor,
                'load_balancing_weight': config.load_balancing_weight,
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
    """Enhanced configuration management with hardware optimization and validation."""
    
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
            raise ValueError(f"Unknown preset: {preset}. Available: {list(ConfigPresets.get_preset_info().keys())}")
        
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
        """Optimize configuration based on available hardware."""
        
        if not torch.cuda.is_available():
            print("Warning: CUDA not available, using CPU-optimized settings")
            config.precision = "fp32"
            config.use_deepspeed = False
            config.compile = False
            return config
        
        # Get hardware info
        gpu_count = torch.cuda.device_count()
        gpu_memory_gb = config._get_gpu_memory_gb() or 8  # Default fallback
        total_memory_gb = gpu_memory_gb * gpu_count
        
        # Get memory estimate
        memory_est = config.get_memory_estimate_gb()
        required_memory = memory_est['total']
        
        print(f"Hardware detected: {gpu_count} GPUs, {gpu_memory_gb:.1f}GB each")
        print(f"Required memory: {required_memory:.1f}GB, Available: {total_memory_gb:.1f}GB")
        
        # Adjust micro batch size based on GPU memory
        if config.micro_batch_size is None:
            config.micro_batch_size = max(1, int(gpu_memory_gb / 10))  # Rough heuristic
        
        # Enable CPU offloading if needed
        if required_memory > total_memory_gb * target_memory_usage:
            print("Enabling CPU offloading due to memory constraints")
            config.cpu_offload = True
            config.cpu_offload_optimizer = True
            if required_memory > total_memory_gb * 1.5:
                config.cpu_offload_parameters = True
                config.aggressive_cpu_offload = True
        
        # Adjust ZeRO stage based on memory requirements
        if config.zero_stage == 0:  # Auto-select
            if required_memory > total_memory_gb * 2:
                config.zero_stage = 3
            elif required_memory > total_memory_gb:
                config.zero_stage = 2
            else:
                config.zero_stage = 1
        
        # Enable gradient compression for multi-GPU setups
        if gpu_count > 4:
            config.gradient_compression = True
        
        # Optimize expert parallelism for MoE
        if config.use_moe and config.expert_parallel_size is None:
            config.expert_parallel_size = min(config.num_experts, gpu_count)
        
        # Adjust precision based on GPU capabilities
        if config.precision == "auto":
            config.precision = config._auto_select_precision()
        
        print(f"Optimized config: ZeRO-{config.zero_stage}, {config.precision}, CPU offload: {config.cpu_offload}")
        
        return config
    
    @staticmethod
    def validate_config(config: Config, strict: bool = False) -> List[str]:
        """Validate configuration and return list of warnings/errors."""
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
            memory_est = config.get_memory_estimate_gb()
            
            if gpu_memory_gb and memory_est['total'] > gpu_memory_gb * 0.9:
                warnings.append(f"High memory usage: {memory_est['total']:.1f}GB > {gpu_memory_gb * 0.9:.1f}GB")
                warnings.append("Consider enabling CPU offloading or reducing batch size")
        
        # MoE-specific validations
        if config.use_moe:
            if config.num_experts != 8:
                warnings.append("Non-standard expert count: 8 experts recommended for consistency")
            
            if config.moe_top_k != 1:
                warnings.append("Top-1 routing recommended for 8x MoE pattern efficiency")
        
        # DeepSpeed compatibility
        if config.use_deepspeed and config.zero_stage > 2 and not config.cpu_offload:
            warnings.append("ZeRO-3 typically requires CPU offloading for optimal performance")
        
        return warnings
    
    @staticmethod 
    def save_config_with_metadata(config: Config, path: str):
        """Save configuration with comprehensive metadata."""
        config_dict = asdict(config)
        
        # Add comprehensive metadata
        config_dict["_metadata"] = {
            "created": datetime.now().isoformat(),
            "lumina_version": "2.0",
            "framework": "PyTorch + DeepSpeed",
            "moe_pattern": f"{config.num_experts}x experts, top-{config.moe_top_k} routing",
            "estimated_parameters": {
                "total": config._estimate_parameters(),
                "active": config.get_active_parameters(),
                "efficiency": f"{config.get_active_parameters() / config._estimate_parameters():.1%}"
            },
            "memory_estimate_gb": config.get_memory_estimate_gb(),
            "hardware_requirements": {
                "min_gpu_memory_gb": config.get_memory_estimate_gb()['total'] / torch.cuda.device_count() if torch.cuda.is_available() else "N/A",
                "recommended_gpus": max(1, int(config.get_memory_estimate_gb()['total'] / 40)),  # Assuming A100-40GB
                "supports_cpu_offload": config.cpu_offload,
                "supports_nvme_offload": bool(config.nvme_offload_optimizer or config.nvme_offload_parameters)
            },
            "training_characteristics": {
                "effective_batch_size": config.effective_batch_size,
                "gradient_accumulation": config.gradient_accumulation_steps,
                "precision": config.precision,
                "deepspeed_stage": f"ZeRO-{config.zero_stage}" if config.zero_stage > 0 else "Auto"
            }
        }
        
        # Ensure directory exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False, indent=2)
        
        print(f"Configuration saved to {path}")
        print(f"Active parameters: {config.get_active_parameters():,}")
        print(f"Total parameters: {config._estimate_parameters():,}")
        print(f"Parameter efficiency: {config.get_active_parameters() / config._estimate_parameters():.1%}")