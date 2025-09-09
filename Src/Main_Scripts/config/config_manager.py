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
    gradient_checkpointing: bool = False
    
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
            
            # Precision settings
            auto_tune_precision=False,
            precision_target="balanced",
            dynamic_precision=False
        )
    
    @staticmethod
    def small() -> Config:
        """Larger small model for limited resources (~2B parameters)."""
        return Config(
            # Larger small model
            hidden_size=1280,
            num_layers=16,
            num_heads=20,
            num_kv_heads=5,
            seq_length=2048,
            intermediate_size=3456,
            
            # Balanced training settings - FIXED: Reduced learning rate
            batch_size=4,
            gradient_accumulation_steps=4,
            num_epochs=3,
            learning_rate=3e-4,  # Reduced from 5e-4
            weight_decay=0.01,
            eval_every_n_batches=500,
            save_every_n_batches=1000,
            precision="fp16",
            inference_precision="fp16",
            compile=True,
            num_workers=2,
            
            # Enhanced MoE settings for 2B model
            use_moe=True,
            num_experts=16,  # Increased from 8
            moe_top_k=2,
            capacity_factor=1.5,  # Increased from 1.25
            load_balancing_weight=0.015,  # Increased from 0.01
            
            # Production settings
            experiment_name="small_2b_model",
            log_level="INFO",
            health_check_interval=100,
            save_total_limit=5,
            early_stopping_patience=5,
            backup_every_n_hours=12,
            lr_scheduler="cosine",
            
            # Precision settings
            auto_tune_precision=True,
            precision_target="balanced",
            dynamic_precision=False
        )
    
    @staticmethod
    def medium() -> Config:
        """Larger medium model for serious training (~14B parameters)."""
        return Config(
            # 14B parameter model (doubled from 7B)
            hidden_size=2560,
            num_layers=28,
            num_heads=20,
            num_kv_heads=10,
            seq_length=4096,
            intermediate_size=6912,
            
            # Serious training configuration - FIXED: Reduced learning rate
            batch_size=2,
            gradient_accumulation_steps=16,
            num_epochs=3,
            learning_rate=1e-4,  # Reduced from 2e-4
            weight_decay=0.01,
            eval_every_n_batches=1000,
            save_every_n_batches=2000,
            precision="mixed_bf16",
            inference_precision="bf16",
            compile=True,
            num_workers=4,
            
            # Enhanced MoE settings for 14B model
            use_moe=True,
            num_experts=32,  # Doubled from 16
            moe_top_k=2,
            capacity_factor=1.8,  # Increased from 1.5
            load_balancing_weight=0.015,  # Increased from 0.01
            
            # Production monitoring
            experiment_name="medium_14b_model",
            log_level="INFO",
            health_check_interval=100,
            save_total_limit=10,
            early_stopping_patience=10,
            backup_every_n_hours=6,
            gradient_checkpointing=True,
            lr_scheduler="cosine",
            
            # Precision settings
            auto_tune_precision=True,
            precision_target="balanced",
            dynamic_precision=False
        )
    
    @staticmethod
    def large() -> Config:
        """Large model for high-end training (~30B parameters)."""
        return Config(
            # 30B parameter model (unchanged)
            hidden_size=4096,
            num_layers=32,
            num_heads=32,
            num_kv_heads=8,
            seq_length=4096,
            intermediate_size=11008,
            
            # Large-scale training - FIXED: Reduced learning rate
            batch_size=1,
            gradient_accumulation_steps=32,
            num_epochs=2,
            learning_rate=8e-5,  # Reduced from 1e-4
            weight_decay=0.01,
            eval_every_n_batches=2000,
            save_every_n_batches=5000,
            precision="mixed_bf16",
            inference_precision="bf16",
            compile=True,
            num_workers=8,
            
            # MoE settings for 30B model (unchanged)
            use_moe=True,
            num_experts=32,  # Even more experts
            moe_top_k=2,
            capacity_factor=1.8,
            load_balancing_weight=0.015,
            
            # Enterprise monitoring
            experiment_name="large_30b_model",
            log_level="INFO",
            health_check_interval=200,
            save_total_limit=15,
            early_stopping_patience=15,
            backup_every_n_hours=4,
            gradient_checkpointing=True,
            lr_scheduler="cosine",
            warmup_ratio=0.05,
            
            # Precision settings
            auto_tune_precision=False,
            precision_target="quality",
            dynamic_precision=False,
            tf32_enabled=True
        )
    
    @staticmethod
    def xlarge() -> Config:
        """Even larger model for massive training (~400B parameters)."""
        return Config(
            # 400B parameter model (doubled from 200B)
            hidden_size=10240,
            num_layers=64,
            num_heads=80,
            num_kv_heads=20,
            seq_length=8192,
            intermediate_size=27648,
            
            # Massive-scale training
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
            
            # Enhanced MoE settings for 400B model
            use_moe=True,
            num_experts=128,  # Doubled from 64
            moe_top_k=2,
            capacity_factor=2.2,  # Increased from 2.0
            load_balancing_weight=0.025,  # Increased from 0.02
            
            # Enterprise monitoring
            experiment_name="xlarge_400b_model",
            log_level="INFO",
            health_check_interval=500,
            save_total_limit=20,
            early_stopping_patience=20,
            backup_every_n_hours=2,
            gradient_checkpointing=True,
            lr_scheduler="cosine",
            warmup_ratio=0.1,
            
            # Precision settings
            auto_tune_precision=False,
            precision_target="quality",
            dynamic_precision=False,
            tf32_enabled=True
        )
    
    @staticmethod
    def inference_optimized() -> Config:
        """Larger configuration optimized specifically for inference performance (~700M parameters)."""
        return Config(
            # Larger model optimized for inference
            hidden_size=1536,
            num_layers=24,
            num_heads=24,
            num_kv_heads=6,
            seq_length=2048,
            intermediate_size=4096,
            
            # Training settings (if fine-tuning) - FIXED: Reduced learning rate
            batch_size=1,
            gradient_accumulation_steps=32,
            num_epochs=1,
            learning_rate=8e-6,  # Reduced from 1e-5
            weight_decay=0.01,
            precision="fp16",
            inference_precision="fp16",
            compile=True,
            num_workers=4,
            
            # Enhanced MoE settings for inference
            use_moe=True,
            num_experts=16,  # Increased from 8
            moe_top_k=2,
            capacity_factor=1.3,  # Increased from 1.2
            load_balancing_weight=0.008,  # Increased from 0.005
            
            # Generation parameters optimized for quality and speed
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            
            # Fast inference settings
            experiment_name="inference_optimized",
            log_level="INFO",
            gradient_checkpointing=False,
            use_stable_embedding=True,
            lr_scheduler="cosine",
            
            # Precision settings optimized for speed
            auto_tune_precision=True,
            precision_target="speed",
            dynamic_precision=False,
            tf32_enabled=True
        )
    
    @staticmethod
    def quality_focused() -> Config:
        """Larger configuration focused on generation quality (~6B parameters)."""
        return Config(
            # Quality-focused model (doubled from 3B)
            hidden_size=4096,
            num_layers=32,
            num_heads=32,
            num_kv_heads=8,
            seq_length=4096,
            intermediate_size=11008,
            
            # Training for quality - FIXED: Reduced learning rate
            batch_size=1,
            gradient_accumulation_steps=32,
            num_epochs=4,
            learning_rate=1e-4,  # Reduced from 1.5e-4
            weight_decay=0.01,
            precision="mixed_bf16",
            inference_precision="bf16",
            compile=True,
            num_workers=6,
            
            # Enhanced MoE settings for quality
            use_moe=True,
            num_experts=32,  # Doubled from 16
            moe_top_k=2,
            capacity_factor=1.8,  # Increased from 1.5
            load_balancing_weight=0.015,  # Increased from 0.01
            
            # Generation parameters for quality
            max_new_tokens=1024,
            temperature=0.8,
            top_p=0.95,
            top_k=100,
            
            # Quality settings
            experiment_name="quality_focused_6b",
            log_level="INFO",
            gradient_checkpointing=True,
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
    def speed_optimized() -> Config:
        """Larger configuration optimized specifically for maximum speed (~120M parameters)."""
        return Config(
            # Larger lean model for speed
            hidden_size=768,
            num_layers=16,
            num_heads=12,
            num_kv_heads=4,
            seq_length=1024,
            intermediate_size=2048,
            
            # Speed-focused training - FIXED: Reduced learning rate
            batch_size=8,
            gradient_accumulation_steps=2,
            num_epochs=2,
            learning_rate=5e-4,  # Reduced from 8e-4
            weight_decay=0.01,
            eval_every_n_batches=200,
            save_every_n_batches=500,
            precision="fp16",
            inference_precision="fp16",
            compile=True,
            num_workers=8,
            
            # Enhanced MoE settings for speed
            use_moe=True,
            num_experts=8,  # Increased from 4
            moe_top_k=2,
            capacity_factor=1.2,  # Increased from 1.1
            load_balancing_weight=0.008,  # Increased from 0.005
            
            # Speed generation parameters
            max_new_tokens=128,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            
            # Speed optimization settings
            experiment_name="speed_optimized_120m",
            log_level="INFO",
            gradient_checkpointing=False,
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
    def memory_optimized() -> Config:
        """Larger configuration optimized for minimal memory usage (~70M parameters)."""
        return Config(
            # Larger memory-efficient model
            hidden_size=768,
            num_layers=12,
            num_heads=12,
            num_kv_heads=4,
            seq_length=1024,
            intermediate_size=2048,
            
            # Memory-conscious training - FIXED: Reduced learning rate
            batch_size=1,
            gradient_accumulation_steps=32,
            num_epochs=3,
            learning_rate=2e-4,  # Reduced from 3e-4
            weight_decay=0.01,
            eval_every_n_batches=1000,
            save_every_n_batches=2000,
            precision="fp16",
            inference_precision="fp16",
            compile=True,
            num_workers=2,
            
            # Enhanced MoE settings for memory optimization
            use_moe=True,
            num_experts=8,  # Increased from 4
            moe_top_k=2,
            capacity_factor=1.2,  # Increased from 1.1
            load_balancing_weight=0.008,  # Increased from 0.005
            
            # Memory-conscious generation
            max_new_tokens=256,
            temperature=0.8,
            top_p=0.9,
            top_k=50,
            
            # Memory optimization settings
            experiment_name="memory_optimized_70m",
            log_level="INFO",
            gradient_checkpointing=True,
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
    def production() -> Config:
        """Larger production-ready configuration with reliability focus (~1B parameters)."""
        return Config(
            # Larger production-scale model
            hidden_size=2048,
            num_layers=24,
            num_heads=32,
            num_kv_heads=8,
            seq_length=2048,
            intermediate_size=5504,
            
            # Reliable production training - FIXED: Reduced learning rate
            batch_size=2,
            gradient_accumulation_steps=16,
            num_epochs=3,
            learning_rate=1.5e-4,  # Reduced from 2e-4
            weight_decay=0.01,
            eval_every_n_batches=500,
            save_every_n_batches=1000,
            precision="mixed_bf16",
            inference_precision="bf16",
            compile=True,
            num_workers=4,
            
            # Enhanced MoE settings for production
            use_moe=True,
            num_experts=24,  # Increased from 12
            moe_top_k=2,
            capacity_factor=1.5,  # Increased from 1.3
            load_balancing_weight=0.015,  # Increased from 0.01
            
            # Production generation parameters
            max_new_tokens=512,
            temperature=0.8,
            top_p=0.9,
            top_k=50,
            
            # Production reliability settings
            experiment_name="production_1b_model",
            log_level="INFO",
            gradient_checkpointing=True,
            use_stable_embedding=True,
            health_check_interval=100,
            save_total_limit=10,
            early_stopping_patience=8,
            backup_every_n_hours=4,
            max_retries=5,
            auto_resume=True,
            lr_scheduler="cosine",
            
            # Production precision settings
            auto_tune_precision=False,
            precision_target="production",
            dynamic_precision=False,
            tf32_enabled=None
        )
    
    @staticmethod
    def experimental() -> Config:
        """Larger experimental configuration with advanced features (~3B parameters)."""
        return Config(
            # Larger experimental model architecture
            hidden_size=3200,
            num_layers=32,
            num_heads=25,
            num_kv_heads=10,
            seq_length=4096,
            intermediate_size=8640,
            
            # Experimental training settings - FIXED: Reduced learning rate
            batch_size=1,
            gradient_accumulation_steps=24,
            num_epochs=4,
            learning_rate=1.2e-4,  # Reduced from 1.8e-4
            weight_decay=0.02,
            eval_every_n_batches=300,
            save_every_n_batches=800,
            precision="mixed_bf16",
            inference_precision="dynamic",
            compile=True,
            num_workers=6,
            
            # Enhanced MoE settings for experimentation
            use_moe=True,
            num_experts=32,  # Increased from 24
            moe_top_k=2,
            capacity_factor=1.8,  # Increased from 1.6
            load_balancing_weight=0.02,  # Increased from 0.015
            
            # Experimental generation parameters
            max_new_tokens=768,
            temperature=0.85,
            top_p=0.92,
            top_k=60,
            
            # Experimental settings
            experiment_name="experimental_3b_model",
            log_level="DEBUG",
            gradient_checkpointing=True,
            use_stable_embedding=True,
            init_std=0.018,
            dropout=0.05,
            health_check_interval=75,
            save_total_limit=8,
            early_stopping_patience=12,
            lr_scheduler="cosine",
            warmup_ratio=0.15,
            
            # Advanced precision settings
            auto_tune_precision=True,
            precision_target="balanced",
            dynamic_precision=True,
            tf32_enabled=True
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
                "moe_experts": 4
            },
            "small": {
                "description": "Small model for limited resources",
                "use_case": "Resource-constrained environments",
                "model_size": "Small (~2B parameters)",
                "memory_usage": "Low",
                "training_speed": "Fast",
                "precision": "fp16 with auto-tuning",
                "moe_experts": 16
            },
            "medium": {
                "description": "Medium model for serious training",
                "use_case": "General purpose training",
                "model_size": "Medium (~14B parameters)",
                "memory_usage": "Medium-High",
                "training_speed": "Medium",
                "precision": "mixed_bf16",
                "moe_experts": 32
            },
            "large": {
                "description": "Large model for high-end training",
                "use_case": "High-performance applications",
                "model_size": "Large (~30B parameters)",
                "memory_usage": "Very High",
                "training_speed": "Slow",
                "precision": "mixed_bf16 with TF32",
                "moe_experts": 32
            },
            "xlarge": {
                "description": "Extra large model for massive training",
                "use_case": "Research and massive scaling",
                "model_size": "Massive (~400B parameters)",
                "memory_usage": "Extreme",
                "training_speed": "Very Slow",
                "precision": "mixed_bf16 with TF32",
                "moe_experts": 128
            },
            "inference_optimized": {
                "description": "Optimized for inference performance",
                "use_case": "Production inference",
                "model_size": "Medium (~700M parameters)",
                "memory_usage": "Medium",
                "training_speed": "Fast",
                "precision": "fp16 for speed",
                "moe_experts": 16
            },
            "quality_focused": {
                "description": "Optimized for generation quality",
                "use_case": "High-quality text generation",
                "model_size": "Large (~6B parameters)",
                "memory_usage": "High",
                "training_speed": "Slow",
                "precision": "mixed_bf16 for stability",
                "moe_experts": 32
            },
            "speed_optimized": {
                "description": "Optimized for maximum speed",
                "use_case": "Real-time applications",
                "model_size": "Small (~120M parameters)",
                "memory_usage": "Low",
                "training_speed": "Very Fast",
                "precision": "fp16 with TF32",
                "moe_experts": 8
            },
            "memory_optimized": {
                "description": "Optimized for minimal memory usage",
                "use_case": "Memory-constrained environments",
                "model_size": "Small (~70M parameters)",
                "memory_usage": "Very Low",
                "training_speed": "Medium",
                "precision": "fp16 with gradient checkpointing",
                "moe_experts": 8
            },
            "production": {
                "description": "Production-ready with reliability focus",
                "use_case": "Production deployments",
                "model_size": "Medium (~1B parameters)",
                "memory_usage": "Medium",
                "training_speed": "Medium",
                "precision": "mixed_bf16 for reliability",
                "moe_experts": 24
            },
            "experimental": {
                "description": "Experimental configuration with advanced features",
                "use_case": "Research and experimentation",
                "model_size": "Large (~3B parameters)",
                "memory_usage": "High",
                "training_speed": "Medium",
                "precision": "dynamic precision selection",
                "moe_experts": 32
            }
        }
    
    @staticmethod
    def compare_presets() -> Dict[str, Any]:
        """Compare all presets across key dimensions."""
        presets = {
            'debug': ConfigPresets.debug(),
            'small': ConfigPresets.small(),
            'medium': ConfigPresets.medium(),
            'large': ConfigPresets.large(),
            'xlarge': ConfigPresets.xlarge(),
            'inference_optimized': ConfigPresets.inference_optimized(),
            'quality_focused': ConfigPresets.quality_focused(),
            'speed_optimized': ConfigPresets.speed_optimized(),
            'memory_optimized': ConfigPresets.memory_optimized(),
            'production': ConfigPresets.production(),
            'experimental': ConfigPresets.experimental()
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