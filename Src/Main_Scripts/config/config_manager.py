# Copyright (c) 2025 Matias Nielsen. All rights reserved.
# Licensed under the Custom License below.

import yaml
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, Any


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
    intermediate_size: int = 1536
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    dropout: float = 0.0
    
    # Training parameters - FIXED: Reduced learning rate and adjusted warmup
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-4  # Reduced from 5e-4 to prevent high loss/PPL
    weight_decay: float = 0.01
    num_epochs: int = 3
    warmup_ratio: float = 0.15  # Increased from 0.1 for better stability
    eval_every_n_batches: int = 500
    save_every_n_batches: int = 1000
    max_grad_norm: float = 1.0
    precision: str = "fp16"
    inference_precision: str = "fp16"
    compile: bool = False
    
    # Data parameters
    train_data_path: str = "data/train.jsonl"
    eval_data_path: str = "data/eval.jsonl"
    num_workers: int = 2
    assistant_loss_weight: float = 1.5
    max_conversations_per_file: int = 10000
    
    # Generation parameters
    max_new_tokens: int = 512
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 50
    
    # Stability and optimization
    init_std: float = 0.02
    layer_norm_eps: float = 1e-5
    use_stable_embedding: bool = True
    gradient_checkpointing: bool = True  # Changed to True by default
    
    # MoE parameters - ADDED FOR YOUR MODEL
    use_moe: bool = False
    num_experts: int = 8
    moe_top_k: int = 2
    capacity_factor: float = 1.25
    load_balancing_weight: float = 0.01
    
    # Production settings
    experiment_name: str = None
    seed: int = 42
    log_level: str = "INFO"
    save_total_limit: int = 5
    early_stopping_patience: int = None
    min_lr: float = 1e-6
    lr_scheduler: str = "cosine"  # Changed back to cosine for better convergence
    
    # Monitoring and fault tolerance
    health_check_interval: int = 100
    auto_resume: bool = True
    backup_every_n_hours: int = 6
    max_retries: int = 3
    
    # Advanced precision settings
    auto_tune_precision: bool = False
    precision_target: str = "balanced"
    dynamic_precision: bool = False
    tf32_enabled: bool = None
    
    def __post_init__(self):
        self.validate()
        
        if self.experiment_name is None:
            self.experiment_name = f"transformer_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if self.vocab_size % 64 != 0:
            self.vocab_size = ((self.vocab_size + 63) // 64) * 64
        
        self.effective_batch_size = self.batch_size * self.gradient_accumulation_steps
    
    def validate(self):
        """Validate configuration parameters."""
        assert self.hidden_size % self.num_heads == 0, "hidden_size must be divisible by num_heads"
        assert self.num_heads % self.num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
        
        valid_training_precisions = ["fp32", "fp16", "bf16", "mixed_fp16", "auto","mixed_bf16", "tf32"]
        assert self.precision in valid_training_precisions, f"Invalid training precision: {self.precision}. Valid options: {valid_training_precisions}"
        
        valid_inference_precisions = ["fp32", "fp16", "bf16", "mixed_fp16", "mixed_bf16", "auto","tf32", "dynamic"]
        assert self.inference_precision in valid_inference_precisions, f"Invalid inference precision: {self.inference_precision}. Valid options: {valid_inference_precisions}"
        
        valid_schedulers = ["cosine", "linear", "onecycle", None]
        assert self.lr_scheduler in valid_schedulers, f"Invalid scheduler: {self.lr_scheduler}. Valid options: {valid_schedulers}"
        assert self.learning_rate > 0, "Learning rate must be positive"
        assert self.weight_decay >= 0, "Weight decay must be non-negative"
        assert self.num_epochs > 0, "Number of epochs must be positive"
        assert self.warmup_ratio >= 0 and self.warmup_ratio <= 1, "Warmup ratio must be between 0 and 1"
        
        # MoE validation
        if self.use_moe:
            assert self.num_experts > 0, "num_experts must be positive"
            assert self.moe_top_k > 0, "moe_top_k must be positive"
            assert self.moe_top_k <= self.num_experts, "moe_top_k cannot exceed num_experts"
            assert self.capacity_factor >= 1.0, "capacity_factor must be at least 1.0"
            assert self.load_balancing_weight >= 0, "load_balancing_weight must be non-negative"
        
        valid_targets = ["speed", "memory", "quality", "balanced", "production"]
        assert self.precision_target in valid_targets, f"Invalid precision target: {self.precision_target}. Valid options: {valid_targets}"
    
    def save(self, path: str):
        """Save configuration to file."""
        config_dict = asdict(self)
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    @classmethod
    def load(cls, path: str) -> 'Config':
        """Load configuration from file."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)


class ConfigPresets:
    """Enhanced configuration presets for different scenarios with comprehensive precision support."""
    
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
            
            # Fast training settings - FIXED: Reduced learning rate
            batch_size=2,
            gradient_accumulation_steps=2,
            num_epochs=1,
            learning_rate=5e-5,  # Reduced from 1e-3
            weight_decay=0.01,
            eval_every_n_batches=50,
            save_every_n_batches=100,
            precision="fp32",
            inference_precision="fp32",
            compile=False,
            num_workers=0,
            
            # MoE settings - small for debugging
            use_moe=True,
            num_experts=4,  # Small number for debugging
            moe_top_k=2,
            capacity_factor=1.1,
            load_balancing_weight=0.005,
            
            # Monitoring and stability
            experiment_name="debug_run",
            log_level="DEBUG",
            health_check_interval=10,
            save_total_limit=3,
            early_stopping_patience=None,
            max_retries=1,
            lr_scheduler="cosine",
            gradient_checkpointing=True,  # Enabled
            
            # Precision settings
            auto_tune_precision=False,
            precision_target="balanced",
            dynamic_precision=False
        )
    
    @staticmethod
    def b1() -> Config:
        """1B parameter model for limited resources."""
        return Config(
            # 1B parameter model
            hidden_size=1024,
            num_layers=12,
            num_heads=16,
            num_kv_heads=4,
            seq_length=2048,
            intermediate_size=2736,
            
            # Balanced training settings
            batch_size=4,
            gradient_accumulation_steps=4,
            num_epochs=3,
            learning_rate=3e-4,
            weight_decay=0.01,
            eval_every_n_batches=500,
            save_every_n_batches=1000,
            precision="fp16",
            inference_precision="fp16",
            compile=True,
            num_workers=2,
            
            # MoE settings for 1B model
            use_moe=True,
            num_experts=8,
            moe_top_k=2,
            capacity_factor=1.25,
            load_balancing_weight=0.01,
            
            # Production settings
            experiment_name="b1",
            log_level="INFO",
            health_check_interval=100,
            save_total_limit=5,
            early_stopping_patience=5,
            backup_every_n_hours=12,
            lr_scheduler="cosine",
            gradient_checkpointing=True,  # Enabled
            
            # Precision settings
            auto_tune_precision=True,
            precision_target="balanced",
            dynamic_precision=False
        )
    
    @staticmethod
    def b7() -> Config:
        """7B parameter model for serious training."""
        return Config(
            # 7B parameter model
            hidden_size=2048,
            num_layers=22,
            num_heads=16,
            num_kv_heads=8,
            seq_length=4096,
            intermediate_size=5504,
            
            # Serious training configuration
            batch_size=2,
            gradient_accumulation_steps=16,
            num_epochs=3,
            learning_rate=1e-4,
            weight_decay=0.01,
            eval_every_n_batches=1000,
            save_every_n_batches=2000,
            precision="mixed_bf16",
            inference_precision="bf16",
            compile=True,
            num_workers=4,
            
            # MoE settings for 7B model
            use_moe=True,
            num_experts=16,
            moe_top_k=2,
            capacity_factor=1.5,
            load_balancing_weight=0.01,
            
            # Production monitoring
            experiment_name="b7",
            log_level="INFO",
            health_check_interval=100,
            save_total_limit=10,
            early_stopping_patience=10,
            backup_every_n_hours=6,
            gradient_checkpointing=True,  # Already enabled
            lr_scheduler="cosine",
            
            # Precision settings
            auto_tune_precision=True,
            precision_target="balanced",
            dynamic_precision=False
        )
    
    @staticmethod
    def b14() -> Config:
        """14B parameter model for high-end training."""
        return Config(
            # 14B parameter model
            hidden_size=2560,
            num_layers=28,
            num_heads=20,
            num_kv_heads=10,
            seq_length=4096,
            intermediate_size=6912,
            
            # Large-scale training
            batch_size=1,
            gradient_accumulation_steps=32,
            num_epochs=2,
            learning_rate=8e-5,
            weight_decay=0.01,
            eval_every_n_batches=2000,
            save_every_n_batches=5000,
            precision="mixed_bf16",
            inference_precision="bf16",
            compile=True,
            num_workers=8,
            
            # MoE settings for 14B model
            use_moe=True,
            num_experts=32,
            moe_top_k=2,
            capacity_factor=1.8,
            load_balancing_weight=0.015,
            
            # Enterprise monitoring
            experiment_name="b14",
            log_level="INFO",
            health_check_interval=200,
            save_total_limit=15,
            early_stopping_patience=15,
            backup_every_n_hours=4,
            gradient_checkpointing=True,  # Already enabled
            lr_scheduler="cosine",
            warmup_ratio=0.05,
            
            # Precision settings
            auto_tune_precision=False,
            precision_target="quality",
            dynamic_precision=False,
            tf32_enabled=True
        )
    
    @staticmethod
    def b50() -> Config:
        """50B parameter model for massive training."""
        return Config(
            # 50B parameter model
            hidden_size=5120,
            num_layers=40,
            num_heads=40,
            num_kv_heads=10,
            seq_length=4096,
            intermediate_size=13824,
            
            # Massive-scale training
            batch_size=1,
            gradient_accumulation_steps=48,
            num_epochs=2,
            learning_rate=6e-5,
            weight_decay=0.01,
            eval_every_n_batches=3000,
            save_every_n_batches=8000,
            precision="mixed_bf16",
            inference_precision="bf16",
            compile=True,
            num_workers=12,
            
            # MoE settings for 50B model
            use_moe=True,
            num_experts=64,
            moe_top_k=2,
            capacity_factor=2.0,
            load_balancing_weight=0.02,
            
            # Enterprise monitoring
            experiment_name="b50",
            log_level="INFO",
            health_check_interval=300,
            save_total_limit=18,
            early_stopping_patience=18,
            backup_every_n_hours=3,
            gradient_checkpointing=True,  # Already enabled
            lr_scheduler="cosine",
            warmup_ratio=0.08,
            
            # Precision settings
            auto_tune_precision=False,
            precision_target="quality",
            dynamic_precision=False,
            tf32_enabled=True
        )
    
    @staticmethod
    def b100() -> Config:
        """100B parameter model for extreme training."""
        return Config(
            # 100B parameter model
            hidden_size=7168,
            num_layers=48,
            num_heads=56,
            num_kv_heads=14,
            seq_length=8192,
            intermediate_size=18432,
            
            # Extreme-scale training
            batch_size=1,
            gradient_accumulation_steps=64,
            num_epochs=2,
            learning_rate=5e-5,
            weight_decay=0.01,
            eval_every_n_batches=5000,
            save_every_n_batches=10000,
            precision="mixed_bf16",
            inference_precision="bf16",
            compile=True,
            num_workers=16,
            
            # MoE settings for 100B model
            use_moe=True,
            num_experts=96,
            moe_top_k=2,
            capacity_factor=2.2,
            load_balancing_weight=0.025,
            
            # Enterprise monitoring
            experiment_name="b100",
            log_level="INFO",
            health_check_interval=500,
            save_total_limit=20,
            early_stopping_patience=20,
            backup_every_n_hours=2,
            gradient_checkpointing=True,  # Already enabled
            lr_scheduler="cosine",
            warmup_ratio=0.1,
            
            # Precision settings
            auto_tune_precision=False,
            precision_target="quality",
            dynamic_precision=False,
            tf32_enabled=True
        )
    
    @staticmethod
    def b200() -> Config:
        """200B parameter model for production deployment."""
        return Config(
            # 200B parameter model
            hidden_size=8192,
            num_layers=56,
            num_heads=64,
            num_kv_heads=16,
            seq_length=8192,
            intermediate_size=22016,
            
            # Production-scale training
            batch_size=1,
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
            
            # MoE settings for 200B model
            use_moe=True,
            num_experts=128,
            moe_top_k=2,
            capacity_factor=2.5,
            load_balancing_weight=0.03,
            
            # Enterprise monitoring
            experiment_name="b200",
            log_level="INFO",
            health_check_interval=500,
            save_total_limit=20,
            early_stopping_patience=20,
            backup_every_n_hours=2,
            gradient_checkpointing=True,  # Already enabled
            lr_scheduler="cosine",
            warmup_ratio=0.1,
            
            # Precision settings
            auto_tune_precision=False,
            precision_target="production",
            dynamic_precision=False,
            tf32_enabled=True
        )
    
    @staticmethod
    def b300() -> Config:
        """300B parameter model for research and experimentation."""
        return Config(
            # 300B parameter model
            hidden_size=9216,
            num_layers=64,
            num_heads=72,
            num_kv_heads=18,
            seq_length=8192,
            intermediate_size=24576,
            
            # Research-scale training
            batch_size=1,
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
            
            # MoE settings for 300B model
            use_moe=True,
            num_experts=144,
            moe_top_k=2,
            capacity_factor=2.8,
            load_balancing_weight=0.035,
            
            # Research monitoring
            experiment_name="b300",
            log_level="INFO",
            health_check_interval=500,
            save_total_limit=20,
            early_stopping_patience=20,
            backup_every_n_hours=2,
            gradient_checkpointing=True,  # Already enabled
            lr_scheduler="cosine",
            warmup_ratio=0.1,
            
            # Precision settings
            auto_tune_precision=False,
            precision_target="quality",
            dynamic_precision=False,
            tf32_enabled=True
        )
    
    @staticmethod
    def b3_inference() -> Config:
        """3B parameter model optimized for inference performance."""
        return Config(
            # 3B parameter model optimized for inference
            hidden_size=2560,
            num_layers=24,
            num_heads=20,
            num_kv_heads=10,
            seq_length=2048,
            intermediate_size=6912,
            
            # Training settings
            batch_size=1,
            gradient_accumulation_steps=32,
            num_epochs=1,
            learning_rate=8e-6,
            weight_decay=0.01,
            precision="fp16",
            inference_precision="fp16",
            compile=True,
            num_workers=4,
            
            # MoE settings for inference
            use_moe=True,
            num_experts=16,
            moe_top_k=2,
            capacity_factor=1.3,
            load_balancing_weight=0.008,
            
            # Generation parameters optimized for quality and speed
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            
            # Fast inference settings
            experiment_name="b3_inference",
            log_level="INFO",
            gradient_checkpointing=True,  # Enabled (was False)
            use_stable_embedding=True,
            lr_scheduler="cosine",
            
            # Precision settings optimized for speed
            auto_tune_precision=True,
            precision_target="speed",
            dynamic_precision=False,
            tf32_enabled=True
        )
    
    @staticmethod
    def b6_quality() -> Config:
        """6B parameter model focused on generation quality."""
        return Config(
            # 6B parameter model focused on quality
            hidden_size=3200,
            num_layers=32,
            num_heads=25,
            num_kv_heads=10,
            seq_length=4096,
            intermediate_size=8640,
            
            # Training for quality
            batch_size=1,
            gradient_accumulation_steps=32,
            num_epochs=4,
            learning_rate=1e-4,
            weight_decay=0.01,
            precision="mixed_bf16",
            inference_precision="bf16",
            compile=True,
            num_workers=6,
            
            # MoE settings for quality
            use_moe=True,
            num_experts=32,
            moe_top_k=2,
            capacity_factor=1.8,
            load_balancing_weight=0.015,
            
            # Generation parameters for quality
            max_new_tokens=1024,
            temperature=0.8,
            top_p=0.95,
            top_k=100,
            
            # Quality settings
            experiment_name="b6_quality",
            log_level="INFO",
            gradient_checkpointing=True,  # Already enabled
            use_stable_embedding=True,
            init_std=0.015,
            dropout=0.1,
            early_stopping_patience=20,
            lr_scheduler="cosine",
            
            # Precision settings optimized for quality
            auto_tune_precision=False,
            precision_target="quality",
            dynamic_precision=False,
            tf32_enabled=False
        )
    
    @staticmethod
    def m120_speed() -> Config:
        """120M parameter model optimized for maximum speed."""
        return Config(
            # 120M parameter model for speed
            hidden_size=768,
            num_layers=16,
            num_heads=12,
            num_kv_heads=4,
            seq_length=1024,
            intermediate_size=2048,
            
            # Speed-focused training
            batch_size=8,
            gradient_accumulation_steps=2,
            num_epochs=2,
            learning_rate=5e-4,
            weight_decay=0.01,
            eval_every_n_batches=200,
            save_every_n_batches=500,
            precision="fp16",
            inference_precision="fp16",
            compile=True,
            num_workers=8,
            
            # MoE settings for speed
            use_moe=True,
            num_experts=8,
            moe_top_k=2,
            capacity_factor=1.2,
            load_balancing_weight=0.008,
            
            # Speed generation parameters
            max_new_tokens=128,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            
            # Speed optimization settings
            experiment_name="m120_speed",
            log_level="INFO",
            gradient_checkpointing=True,  # Enabled (was False)
            use_stable_embedding=False,
            health_check_interval=50,
            save_total_limit=3,
            lr_scheduler="cosine",
            
            # Precision settings for maximum speed
            auto_tune_precision=True,
            precision_target="speed",
            dynamic_precision=False,
            tf32_enabled=True
        )
    
    @staticmethod
    def m70_memory() -> Config:
        """70M parameter model optimized for minimal memory usage."""
        return Config(
            # 70M parameter model for memory efficiency
            hidden_size=768,
            num_layers=12,
            num_heads=12,
            num_kv_heads=4,
            seq_length=1024,
            intermediate_size=2048,
            
            # Memory-conscious training
            batch_size=1,
            gradient_accumulation_steps=32,
            num_epochs=3,
            learning_rate=2e-4,
            weight_decay=0.01,
            eval_every_n_batches=1000,
            save_every_n_batches=2000,
            precision="fp16",
            inference_precision="fp16",
            compile=True,
            num_workers=2,
            
            # MoE settings for memory optimization
            use_moe=True,
            num_experts=8,
            moe_top_k=2,
            capacity_factor=1.2,
            load_balancing_weight=0.008,
            
            # Memory-conscious generation
            max_new_tokens=256,
            temperature=0.8,
            top_p=0.9,
            top_k=50,
            
            # Memory optimization settings
            experiment_name="m70_memory",
            log_level="INFO",
            gradient_checkpointing=True,  # Already enabled
            use_stable_embedding=True,
            save_total_limit=2,
            lr_scheduler="cosine",
            
            # Precision settings for memory efficiency
            auto_tune_precision=True,
            precision_target="memory",
            dynamic_precision=False,
            tf32_enabled=False
        )
    
    @staticmethod
    def get_preset_info() -> Dict[str, Dict[str, Any]]:
        """Get information about all available presets."""
        return {
            "debug": {
                "description": "Minimal configuration for debugging and testing",
                "use_case": "Development and debugging",
                "model_size": "Tiny (~0.5M parameters)",
                "memory_usage": "Very Low",
                "training_speed": "Very Fast",
                "precision": "fp32 (debugging clarity)",
                "moe_experts": 4,
                "gradient_checkpointing": True
            },
            "b1": {
                "description": "1B parameter model for limited resources",
                "use_case": "Resource-constrained environments",
                "model_size": "Small (~1B parameters)",
                "memory_usage": "Low",
                "training_speed": "Fast",
                "precision": "fp16 with auto-tuning",
                "moe_experts": 8,
                "gradient_checkpointing": True
            },
            "b7": {
                "description": "7B parameter model for serious training",
                "use_case": "General purpose training",
                "model_size": "Medium (~7B parameters)",
                "memory_usage": "Medium-High",
                "training_speed": "Medium",
                "precision": "mixed_bf16",
                "moe_experts": 16,
                "gradient_checkpointing": True
            },
            "b14": {
                "description": "14B parameter model for high-end training",
                "use_case": "High-performance applications",
                "model_size": "Large (~14B parameters)",
                "memory_usage": "High",
                "training_speed": "Slow",
                "precision": "mixed_bf16 with TF32",
                "moe_experts": 32,
                "gradient_checkpointing": True
            },
            "b50": {
                "description": "50B parameter model for massive training",
                "use_case": "Research and scaling",
                "model_size": "Large (~50B parameters)",
                "memory_usage": "Very High",
                "training_speed": "Slow",
                "precision": "mixed_bf16 with TF32",
                "moe_experts": 64,
                "gradient_checkpointing": True
            },
            "b100": {
                "description": "100B parameter model for extreme training",
                "use_case": "Cutting-edge research",
                "model_size": "Massive (~100B parameters)",
                "memory_usage": "Extreme",
                "training_speed": "Very Slow",
                "precision": "mixed_bf16 with TF32",
                "moe_experts": 96,
                "gradient_checkpointing": True
            },
            "b200": {
                "description": "200B parameter model for production deployment",
                "use_case": "Enterprise production",
                "model_size": "Massive (~200B parameters)",
                "memory_usage": "Extreme",
                "training_speed": "Very Slow",
                "precision": "mixed_bf16 with TF32",
                "moe_experts": 128,
                "gradient_checkpointing": True
            },
            "b300": {
                "description": "300B parameter model for research",
                "use_case": "Advanced research",
                "model_size": "Massive (~300B parameters)",
                "memory_usage": "Extreme",
                "training_speed": "Very Slow",
                "precision": "mixed_bf16 with TF32",
                "moe_experts": 144,
                "gradient_checkpointing": True
            },
            "b3_inference": {
                "description": "Optimized for inference performance",
                "use_case": "Production inference",
                "model_size": "Medium (~3B parameters)",
                "memory_usage": "Medium",
                "training_speed": "Fast",
                "precision": "fp16 for speed",
                "moe_experts": 16,
                "gradient_checkpointing": True
            },
            "b6_quality": {
                "description": "Optimized for generation quality",
                "use_case": "High-quality text generation",
                "model_size": "Large (~6B parameters)",
                "memory_usage": "High",
                "training_speed": "Slow",
                "precision": "mixed_bf16 for stability",
                "moe_experts": 32,
                "gradient_checkpointing": True
            },
            "m120_speed": {
                "description": "Optimized for maximum speed",
                "use_case": "Real-time applications",
                "model_size": "Small (~120M parameters)",
                "memory_usage": "Low",
                "training_speed": "Very Fast",
                "precision": "fp16 with TF32",
                "moe_experts": 8,
                "gradient_checkpointing": True
            },
            "m70_memory": {
                "description": "Optimized for minimal memory usage",
                "use_case": "Memory-constrained environments",
                "model_size": "Small (~70M parameters)",
                "memory_usage": "Very Low",
                "training_speed": "Medium",
                "precision": "fp16 with gradient checkpointing",
                "moe_experts": 8,
                "gradient_checkpointing": True
            }
        }
    
    @staticmethod
    def compare_presets() -> Dict[str, Any]:
        """Compare all presets across key dimensions."""
        presets = {
            'debug': ConfigPresets.debug(),
            'b1': ConfigPresets.b1(),
            'b7': ConfigPresets.b7(),
            'b14': ConfigPresets.b14(),
            'b50': ConfigPresets.b50(),
            'b100': ConfigPresets.b100(),
            'b200': ConfigPresets.b200(),
            'b300': ConfigPresets.b300(),
            'b3_inference': ConfigPresets.b3_inference(),
            'b6_quality': ConfigPresets.b6_quality(),
            'm120_speed': ConfigPresets.m120_speed(),
            'm70_memory': ConfigPresets.m70_memory()
        }
        
        comparison = {}
        
        for name, config in presets.items():
            # Estimate model parameters
            embed_params = config.vocab_size * config.hidden_size
            attention_params = config.num_layers * (
                4 * config.hidden_size * config.hidden_size + 
                2 * config.hidden_size * config.intermediate_size
            )
            total_params = embed_params + attention_params
            
            comparison[name] = {
                'estimated_parameters': total_params,
                'hidden_size': config.hidden_size,
                'num_layers': config.num_layers,
                'sequence_length': config.seq_length,
                'batch_size': config.batch_size,
                'effective_batch_size': config.effective_batch_size,
                'learning_rate': config.learning_rate,
                'precision': config.precision,
                'inference_precision': config.inference_precision,
                'auto_tune_precision': config.auto_tune_precision,
                'precision_target': config.precision_target,
                'dynamic_precision': config.dynamic_precision,
                'gradient_checkpointing': config.gradient_checkpointing,
                'compile': config.compile,
                'use_moe': config.use_moe,
                'num_experts': config.num_experts,
                'moe_top_k': config.moe_top_k
            }
        
        return comparison