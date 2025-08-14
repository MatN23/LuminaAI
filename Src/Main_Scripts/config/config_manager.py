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
    precision: str = "fp16"
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
    lr_scheduler: str = "cosine"  # cosine, linear, onecycle
    
    # Monitoring and fault tolerance
    health_check_interval: int = 100
    auto_resume: bool = True
    backup_every_n_hours: int = 6
    max_retries: int = 3
    
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
        assert self.precision in ["fp16", "bf16", "fp32"], f"Invalid precision: {self.precision}"
        assert self.lr_scheduler in ["cosine", "linear", "onecycle"], f"Invalid scheduler: {self.lr_scheduler}"
        assert self.learning_rate > 0, "Learning rate must be positive"
        assert self.weight_decay >= 0, "Weight decay must be non-negative"
        assert self.num_epochs > 0, "Number of epochs must be positive"
        assert self.warmup_ratio >= 0 and self.warmup_ratio <= 1, "Warmup ratio must be between 0 and 1"
    
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
    """Enhanced configuration presets for different scenarios."""
    
    @staticmethod
    def debug() -> Config:
        """Minimal config for debugging and testing."""
        return Config(
            # Tiny model for fast iteration
            vocab_size=1024,
            hidden_size=256,
            num_layers=4,
            num_heads=4,
            num_kv_heads=2,
            seq_length=512,
            intermediate_size=512,
            
            # Fast training settings
            batch_size=2,
            gradient_accumulation_steps=2,
            num_epochs=1,
            learning_rate=1e-3,
            weight_decay=0.01,
            eval_every_n_batches=50,
            save_every_n_batches=100,
            precision="fp32",
            compile=False,
            num_workers=0,
            
            # Monitoring and stability
            experiment_name="debug_run",
            log_level="DEBUG",
            health_check_interval=10,
            save_total_limit=3,
            early_stopping_patience=None,
            max_retries=1
        )
    
    @staticmethod
    def small() -> Config:
        """Small model for limited resources."""
        return Config(
            # Small but capable model
            hidden_size=512,
            num_layers=8,
            num_heads=8,
            num_kv_heads=4,
            seq_length=1024,
            intermediate_size=1536,
            
            # Balanced training settings
            batch_size=4,
            gradient_accumulation_steps=4,
            num_epochs=3,
            learning_rate=5e-4,
            weight_decay=0.01,
            eval_every_n_batches=500,
            save_every_n_batches=1000,
            precision="fp16",
            compile=True,
            num_workers=2,
            
            # Production settings
            experiment_name="small_model",
            log_level="INFO",
            health_check_interval=100,
            save_total_limit=5,
            early_stopping_patience=5,
            backup_every_n_hours=12
        )
    
    @staticmethod
    def medium() -> Config:
        """Medium model for serious training."""
        return Config(
            # Medium-scale model
            hidden_size=1024,
            num_layers=16,
            num_heads=16,
            num_kv_heads=8,
            seq_length=2048,
            intermediate_size=2816,
            
            # Serious training configuration
            batch_size=4,
            gradient_accumulation_steps=8,
            num_epochs=5,
            learning_rate=3e-4,
            weight_decay=0.01,
            eval_every_n_batches=1000,
            save_every_n_batches=2000,
            precision="bf16",
            compile=True,
            num_workers=4,
            
            # Production monitoring
            experiment_name="medium_model",
            log_level="INFO",
            health_check_interval=100,
            save_total_limit=10,
            early_stopping_patience=10,
            backup_every_n_hours=6,
            gradient_checkpointing=True
        )
    
    @staticmethod
    def large() -> Config:
        """Large model for high-end training."""
        return Config(
            # Large-scale model
            hidden_size=2048,
            num_layers=24,
            num_heads=32,
            num_kv_heads=8,
            seq_length=4096,
            intermediate_size=5504,
            
            # Large-scale training
            batch_size=2,
            gradient_accumulation_steps=16,
            num_epochs=3,
            learning_rate=2e-4,
            weight_decay=0.01,
            eval_every_n_batches=2000,
            save_every_n_batches=5000,
            precision="bf16",
            compile=True,
            num_workers=8,
            
            # Enterprise monitoring
            experiment_name="large_model",
            log_level="INFO",
            health_check_interval=200,
            save_total_limit=15,
            early_stopping_patience=15,
            backup_every_n_hours=4,
            gradient_checkpointing=True,
            lr_scheduler="cosine",
            warmup_ratio=0.05
        )