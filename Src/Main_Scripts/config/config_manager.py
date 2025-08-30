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
    
    # Training parameters
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    eval_every_n_batches: int = 500
    save_every_n_batches: int = 1000
    max_grad_norm: float = 1.0
    precision: str = "fp16"  # Training precision - EXPANDED OPTIONS
    inference_precision: str = "auto"  # Inference precision - EXPANDED OPTIONS
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
    
    # Production settings
    experiment_name: str = None
    seed: int = 42
    log_level: str = "INFO"
    save_total_limit: int = 5
    early_stopping_patience: int = None
    min_lr: float = 1e-6
    lr_scheduler: str = None  # Changed from "cosine" to None
    
    # Monitoring and fault tolerance
    health_check_interval: int = 100
    auto_resume: bool = True
    backup_every_n_hours: int = 6
    max_retries: int = 3
    
    # NEW: Advanced precision settings
    auto_tune_precision: bool = False  # Auto-tune inference precision
    precision_target: str = "balanced"  # Target for auto-tuning: speed, memory, quality, balanced
    dynamic_precision: bool = False  # Use dynamic precision selection
    tf32_enabled: bool = None  # Enable TF32 (None = auto-detect)
    
    def __post_init__(self):
        self.validate()
        
        # Set experiment name if not provided
        if self.experiment_name is None:
            self.experiment_name = f"transformer_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Ensure vocab size is efficient
        if self.vocab_size % 64 != 0:
            self.vocab_size = ((self.vocab_size + 63) // 64) * 64
        
        self.effective_batch_size = self.batch_size * self.gradient_accumulation_steps
    
    def validate(self):
        """Validate configuration parameters."""
        assert self.hidden_size % self.num_heads == 0, "hidden_size must be divisible by num_heads"
        assert self.num_heads % self.num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
        
        # EXPANDED precision validation
        valid_training_precisions = ["fp32", "fp16", "bf16", "mixed_fp16", "mixed_bf16", "tf32"]
        assert self.precision in valid_training_precisions, f"Invalid training precision: {self.precision}. Valid options: {valid_training_precisions}"
        
        valid_inference_precisions = ["fp32", "fp16", "bf16", "mixed_fp16", "mixed_bf16", "tf32", "auto", "dynamic"]
        assert self.inference_precision in valid_inference_precisions, f"Invalid inference precision: {self.inference_precision}. Valid options: {valid_inference_precisions}"
        
        # Modified validation to allow None for lr_scheduler
        valid_schedulers = ["cosine", "linear", "onecycle", None]
        assert self.lr_scheduler in valid_schedulers, f"Invalid scheduler: {self.lr_scheduler}. Valid options: {valid_schedulers}"
        assert self.learning_rate > 0, "Learning rate must be positive"
        assert self.weight_decay >= 0, "Weight decay must be non-negative"
        assert self.num_epochs > 0, "Number of epochs must be positive"
        assert self.warmup_ratio >= 0 and self.warmup_ratio <= 1, "Warmup ratio must be between 0 and 1"
        
        # Precision target validation
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
            
            # Fast training settings
            batch_size=2,
            gradient_accumulation_steps=2,
            num_epochs=1,
            learning_rate=1e-3,
            weight_decay=0.01,
            eval_every_n_batches=50,
            save_every_n_batches=100,
            precision="fp32",  # Use fp32 for debugging clarity
            inference_precision="fp32",  # Use fp32 for debugging
            compile=False,
            num_workers=0,
            
            # Monitoring and stability
            experiment_name="debug_run",
            log_level="DEBUG",
            health_check_interval=10,
            save_total_limit=3,
            early_stopping_patience=None,
            max_retries=1,
            lr_scheduler=None,  # Set to None
            
            # Precision settings
            auto_tune_precision=False,
            precision_target="balanced",
            dynamic_precision=False
        )
    
    @staticmethod
    def small() -> Config:
        """Small model for limited resources (~125M parameters)."""
        return Config(
            # Small but capable model
            hidden_size=768,
            num_layers=6,
            num_heads=12,
            num_kv_heads=4,
            seq_length=2048,
            intermediate_size=2048,
            
            # Balanced training settings
            batch_size=4,
            gradient_accumulation_steps=4,
            num_epochs=3,
            learning_rate=5e-4,
            weight_decay=0.01,
            eval_every_n_batches=500,
            save_every_n_batches=1000,
            precision="fp16",
            inference_precision="auto",  # Auto-select best inference precision
            compile=True,
            num_workers=2,
            
            # Production settings
            experiment_name="small_model",
            log_level="INFO",
            health_check_interval=100,
            save_total_limit=5,
            early_stopping_patience=5,
            backup_every_n_hours=12,
            lr_scheduler=None,  # Set to None
            
            # Precision settings
            auto_tune_precision=True,  # Auto-tune for small models
            precision_target="balanced",
            dynamic_precision=False
        )
    
    @staticmethod
    def medium() -> Config:
        """Medium model for serious training (~1B parameters)."""
        return Config(
            # 1B parameter model
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
            learning_rate=2e-4,
            weight_decay=0.01,
            eval_every_n_batches=1000,
            save_every_n_batches=2000,
            precision="mixed_bf16",  # Mixed precision for better stability
            inference_precision="bf16",  # Use bf16 for better numerical stability in inference
            compile=True,
            num_workers=4,
            
            # Production monitoring
            experiment_name="medium_1b_model",
            log_level="INFO",
            health_check_interval=100,
            save_total_limit=10,
            early_stopping_patience=10,
            backup_every_n_hours=6,
            gradient_checkpointing=True,
            lr_scheduler=None,  # Set to None
            
            # Precision settings
            auto_tune_precision=True,
            precision_target="balanced",
            dynamic_precision=False
        )
    
    @staticmethod
    def large() -> Config:
        """Large model for high-end training (~7B parameters)."""
        return Config(
            # 7B parameter model
            hidden_size=4096,
            num_layers=32,
            num_heads=32,
            num_kv_heads=8,
            seq_length=4096,
            intermediate_size=11008,
            
            # Large-scale training
            batch_size=1,
            gradient_accumulation_steps=32,
            num_epochs=2,
            learning_rate=1e-4,
            weight_decay=0.01,
            eval_every_n_batches=2000,
            save_every_n_batches=5000,
            precision="mixed_bf16",  # Use mixed bf16 for large models
            inference_precision="bf16",  # Use bf16 for large models
            compile=True,
            num_workers=8,
            
            # Enterprise monitoring
            experiment_name="large_7b_model",
            log_level="INFO",
            health_check_interval=200,
            save_total_limit=15,
            early_stopping_patience=15,
            backup_every_n_hours=4,
            gradient_checkpointing=True,
            lr_scheduler=None,  # Already set to None
            warmup_ratio=0.05,
            
            # Precision settings
            auto_tune_precision=False,  # Manual control for large models
            precision_target="quality",
            dynamic_precision=False,
            tf32_enabled=True  # Enable TF32 for Ampere GPUs
        )
    
    @staticmethod
    def inference_optimized() -> Config:
        """Configuration optimized specifically for inference performance (~350M parameters)."""
        return Config(
            # Medium model optimized for inference
            hidden_size=1280,
            num_layers=16,
            num_heads=20,
            num_kv_heads=5,
            seq_length=2048,
            intermediate_size=3456,
            
            # Training settings (if fine-tuning)
            batch_size=1,
            gradient_accumulation_steps=32,
            num_epochs=1,
            learning_rate=1e-5,
            weight_decay=0.01,
            precision="fp16",
            inference_precision="fp16",  # Optimized for speed
            compile=True,
            num_workers=4,
            
            # Generation parameters optimized for quality and speed
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            
            # Fast inference settings
            experiment_name="inference_optimized",
            log_level="INFO",
            gradient_checkpointing=False,  # Disable for faster inference
            use_stable_embedding=True,
            lr_scheduler=None,  # Set to None
            
            # Precision settings optimized for speed
            auto_tune_precision=True,
            precision_target="speed",
            dynamic_precision=False,
            tf32_enabled=True
        )
    
    @staticmethod
    def quality_focused() -> Config:
        """Configuration focused on generation quality over speed (~3B parameters)."""
        return Config(
            # Quality-focused model
            hidden_size=3200,
            num_layers=26,
            num_heads=25,
            num_kv_heads=5,
            seq_length=4096,
            intermediate_size=8640,
            
            # Training for quality
            batch_size=1,
            gradient_accumulation_steps=32,
            num_epochs=4,
            learning_rate=1.5e-4,
            weight_decay=0.01,
            precision="mixed_bf16",  # Best numerical precision for training
            inference_precision="bf16",  # Best numerical precision for inference
            compile=True,
            num_workers=6,
            
            # Generation parameters for quality
            max_new_tokens=1024,
            temperature=0.8,
            top_p=0.95,
            top_k=100,
            
            # Quality settings
            experiment_name="quality_focused_3b",
            log_level="INFO",
            gradient_checkpointing=True,
            use_stable_embedding=True,
            init_std=0.015,  # More conservative initialization
            dropout=0.1,  # Add some regularization
            early_stopping_patience=20,
            lr_scheduler=None,  # Set to None
            
            # Precision settings optimized for quality
            auto_tune_precision=False,  # Manual control for quality
            precision_target="quality",
            dynamic_precision=False,
            tf32_enabled=False  # Disable TF32 for maximum precision
        )
    
    @staticmethod
    def speed_optimized() -> Config:
        """Configuration optimized specifically for maximum speed (~60M parameters)."""
        return Config(
            # Lean model for speed
            hidden_size=512,
            num_layers=12,
            num_heads=8,
            num_kv_heads=4,
            seq_length=1024,
            intermediate_size=1376,
            
            # Speed-focused training
            batch_size=8,
            gradient_accumulation_steps=2,
            num_epochs=2,
            learning_rate=8e-4,
            weight_decay=0.01,
            eval_every_n_batches=200,
            save_every_n_batches=500,
            precision="fp16",  # Fastest precision
            inference_precision="fp16",  # Fastest inference
            compile=True,
            num_workers=8,
            
            # Speed generation parameters
            max_new_tokens=128,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            
            # Speed optimization settings
            experiment_name="speed_optimized_60m",
            log_level="INFO",
            gradient_checkpointing=False,
            use_stable_embedding=False,  # Slight speed improvement
            health_check_interval=50,
            save_total_limit=3,
            lr_scheduler=None,  # Set to None
            
            # Precision settings for maximum speed
            auto_tune_precision=True,
            precision_target="speed",
            dynamic_precision=False,
            tf32_enabled=True
        )
    
    @staticmethod
    def memory_optimized() -> Config:
        """Configuration optimized for minimal memory usage (~35M parameters)."""
        return Config(
            # Memory-efficient model
            hidden_size=512,
            num_layers=8,
            num_heads=8,
            num_kv_heads=4,
            seq_length=1024,
            intermediate_size=1376,
            
            # Memory-conscious training
            batch_size=1,
            gradient_accumulation_steps=32,
            num_epochs=3,
            learning_rate=3e-4,
            weight_decay=0.01,
            eval_every_n_batches=1000,
            save_every_n_batches=2000,
            precision="fp16",  # Memory efficient
            inference_precision="fp16",  # Memory efficient
            compile=True,
            num_workers=2,
            
            # Memory-conscious generation
            max_new_tokens=256,
            temperature=0.8,
            top_p=0.9,
            top_k=50,
            
            # Memory optimization settings
            experiment_name="memory_optimized_35m",
            log_level="INFO",
            gradient_checkpointing=True,  # Trade compute for memory
            use_stable_embedding=True,
            save_total_limit=2,  # Keep fewer checkpoints
            lr_scheduler=None,  # Set to None
            
            # Precision settings for memory efficiency
            auto_tune_precision=True,
            precision_target="memory",
            dynamic_precision=False,
            tf32_enabled=False  # Save memory
        )
    
    @staticmethod
    def production() -> Config:
        """Production-ready configuration with reliability focus (~500M parameters)."""
        return Config(
            # Production-scale model
            hidden_size=1536,
            num_layers=18,
            num_heads=24,
            num_kv_heads=8,
            seq_length=2048,
            intermediate_size=4096,
            
            # Reliable production training
            batch_size=2,
            gradient_accumulation_steps=16,
            num_epochs=3,
            learning_rate=2e-4,
            weight_decay=0.01,
            eval_every_n_batches=500,
            save_every_n_batches=1000,
            precision="mixed_bf16",  # Most reliable mixed precision
            inference_precision="bf16",  # Reliable inference precision
            compile=True,
            num_workers=4,
            
            # Production generation parameters
            max_new_tokens=512,
            temperature=0.8,
            top_p=0.9,
            top_k=50,
            
            # Production reliability settings
            experiment_name="production_500m_model",
            log_level="INFO",
            gradient_checkpointing=True,
            use_stable_embedding=True,
            health_check_interval=100,
            save_total_limit=10,
            early_stopping_patience=8,
            backup_every_n_hours=4,
            max_retries=5,
            auto_resume=True,
            lr_scheduler=None,  # Set to None
            
            # Production precision settings
            auto_tune_precision=False,  # Manual control for production
            precision_target="production",
            dynamic_precision=False,
            tf32_enabled=None  # Auto-detect based on hardware
        )
    
    @staticmethod
    def experimental() -> Config:
        """Experimental configuration with advanced features (~1.5B parameters)."""
        return Config(
            # Experimental model architecture
            hidden_size=2560,
            num_layers=24,
            num_heads=20,
            num_kv_heads=10,
            seq_length=3072,
            intermediate_size=6912,
            
            # Experimental training settings
            batch_size=1,
            gradient_accumulation_steps=24,
            num_epochs=4,
            learning_rate=1.8e-4,
            weight_decay=0.02,
            eval_every_n_batches=300,
            save_every_n_batches=800,
            precision="mixed_bf16",
            inference_precision="dynamic",  # Use dynamic precision
            compile=True,
            num_workers=6,
            
            # Experimental generation parameters
            max_new_tokens=768,
            temperature=0.85,
            top_p=0.92,
            top_k=60,
            
            # Experimental settings
            experiment_name="experimental_1_5b_model",
            log_level="DEBUG",
            gradient_checkpointing=True,
            use_stable_embedding=True,
            init_std=0.018,
            dropout=0.05,
            health_check_interval=75,
            save_total_limit=8,
            early_stopping_patience=12,
            lr_scheduler=None,  # Changed from "onecycle" to None
            warmup_ratio=0.15,
            
            # Advanced precision settings
            auto_tune_precision=True,
            precision_target="balanced",
            dynamic_precision=True,  # Enable dynamic precision selection
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
                "precision": "fp32 (debugging clarity)"
            },
            "small": {
                "description": "Small model for limited resources",
                "use_case": "Resource-constrained environments",
                "model_size": "Small (~125M parameters)",
                "memory_usage": "Low",
                "training_speed": "Fast",
                "precision": "fp16 with auto-tuning"
            },
            "medium": {
                "description": "Medium model for serious training",
                "use_case": "General purpose training",
                "model_size": "Medium (~1B parameters)",
                "memory_usage": "Medium-High",
                "training_speed": "Medium",
                "precision": "mixed_bf16"
            },
            "large": {
                "description": "Large model for high-end training",
                "use_case": "High-performance applications",
                "model_size": "Large (~7B parameters)",
                "memory_usage": "Very High",
                "training_speed": "Slow",
                "precision": "mixed_bf16 with TF32"
            },
            "inference_optimized": {
                "description": "Optimized for inference performance",
                "use_case": "Production inference",
                "model_size": "Medium (~350M parameters)",
                "memory_usage": "Medium",
                "training_speed": "Fast",
                "precision": "fp16 for speed"
            },
            "quality_focused": {
                "description": "Optimized for generation quality",
                "use_case": "High-quality text generation",
                "model_size": "Large (~3B parameters)",
                "memory_usage": "High",
                "training_speed": "Slow",
                "precision": "mixed_bf16 for stability"
            },
            "speed_optimized": {
                "description": "Optimized for maximum speed",
                "use_case": "Real-time applications",
                "model_size": "Small (~60M parameters)",
                "memory_usage": "Low",
                "training_speed": "Very Fast",
                "precision": "fp16 with TF32"
            },
            "memory_optimized": {
                "description": "Optimized for minimal memory usage",
                "use_case": "Memory-constrained environments",
                "model_size": "Small (~35M parameters)",
                "memory_usage": "Very Low",
                "training_speed": "Medium",
                "precision": "fp16 with gradient checkpointing"
            },
            "production": {
                "description": "Production-ready with reliability focus",
                "use_case": "Production deployments",
                "model_size": "Medium (~500M parameters)",
                "memory_usage": "Medium",
                "training_speed": "Medium",
                "precision": "mixed_bf16 for reliability"
            },
            "experimental": {
                "description": "Experimental configuration with advanced features",
                "use_case": "Research and experimentation",
                "model_size": "Large (~1.5B parameters)",
                "memory_usage": "High",
                "training_speed": "Medium",
                "precision": "dynamic precision selection"
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
                'compile': config.compile
            }
        
        return comparison