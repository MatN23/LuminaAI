# Enhanced Configuration System with Modern Architecture
# Copyright (c) 2025 Matias Nielsen. All rights reserved.

import os
import math
import json
import logging
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
import hashlib
import pickle

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Enhanced model configuration with modern architecture options."""
    
    # Basic architecture
    vocab_size: int = 32000
    hidden_size: int = 2048
    num_layers: int = 24
    num_heads: int = 16
    seq_length: int = 4096
    dropout: float = 0.0  # Modern models often use 0 dropout
    
    # Advanced architecture options
    model_type: str = "ModernSubwordTransformer"
    tokenizer_type: str = "subword"
    intermediate_size: Optional[int] = None  # Will be calculated if None
    
    # Modern improvements
    use_rotary_pos_emb: bool = True  # RoPE instead of sinusoidal
    rotary_dim: Optional[int] = None  # Will be head_dim if None
    use_rms_norm: bool = True  # RMSNorm instead of LayerNorm
    use_glu_variants: bool = True  # Use SwiGLU/GeGLU instead of basic GELU
    glu_variant: str = "swiglu"  # "swiglu", "geglu", "reglu"
    
    # Attention improvements
    use_grouped_query_attention: bool = True
    num_key_value_heads: Optional[int] = None  # Will be calculated if None
    attention_bias: bool = False
    use_flash_attention: bool = True  # If available
    
    # Activation and normalization
    activation: str = "swiglu"
    layer_norm_eps: float = 1e-6
    rms_norm_eps: float = 1e-6
    
    # Embedding and weight tying
    tie_word_embeddings: bool = True
    embed_dropout: float = 0.0
    
    # Initialization
    initializer_range: float = 0.02
    use_scaled_init: bool = True  # Scale initialization by depth
    
    # Gradient checkpointing
    gradient_checkpointing: bool = False  # Set dynamically
    
    def __post_init__(self):
        # Set derived values
        if self.intermediate_size is None:
            if self.use_glu_variants:
                # GLU variants need larger intermediate size
                self.intermediate_size = int(8 * self.hidden_size / 3)  # Common for SwiGLU
            else:
                self.intermediate_size = 4 * self.hidden_size
        
        if self.use_grouped_query_attention and self.num_key_value_heads is None:
            # Default to 1/4 of query heads for GQA
            self.num_key_value_heads = max(1, self.num_heads // 4)
        elif not self.use_grouped_query_attention:
            self.num_key_value_heads = self.num_heads
        
        if self.use_rotary_pos_emb and self.rotary_dim is None:
            self.rotary_dim = self.hidden_size // self.num_heads
        
        # Validate configuration
        self._validate()
    
    def _validate(self):
        """Validate configuration consistency."""
        assert self.hidden_size % self.num_heads == 0, "hidden_size must be divisible by num_heads"
        if self.use_grouped_query_attention:
            assert self.num_heads % self.num_key_value_heads == 0, "num_heads must be divisible by num_key_value_heads"
        assert self.vocab_size > 0, "vocab_size must be positive"
        assert self.num_layers > 0, "num_layers must be positive"
        assert 0 <= self.dropout <= 1, "dropout must be between 0 and 1"

@dataclass
class TrainingConfig:
    """Enhanced training configuration with modern optimization techniques."""
    
    # Basic optimization
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    max_epochs: int = 3
    max_steps: Optional[int] = None  # If set, overrides max_epochs
    
    # Learning rate scheduling
    scheduler_type: str = "cosine_with_warmup"
    warmup_steps: Optional[int] = None
    warmup_ratio: float = 0.03
    min_lr_ratio: float = 0.1
    cosine_restarts: int = 0
    
    # Advanced optimization
    optimizer_type: str = "adamw"
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    
    # Gradient management
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True
    
    # Regularization
    label_smoothing: float = 0.0
    dropout_schedule: bool = False
    
    # Evaluation and saving
    eval_steps: Optional[int] = None
    eval_ratio: float = 0.1
    save_steps: Optional[int] = None
    save_ratio: float = 0.2
    save_total_limit: int = 3
    
    # Data efficiency
    use_dataloader_workers: bool = True
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2

@dataclass
class PrecisionConfig:
    """Enhanced precision configuration with automatic detection."""
    
    precision_type: str = "auto"  # "auto", "bf16", "fp16", "tf32", "fp32"
    use_mixed_precision: bool = True
    use_compile: bool = True  # torch.compile if available
    compile_mode: str = "default"  # "default", "reduce-overhead", "max-autotune"
    
    # Loss scaling (for fp16)
    use_dynamic_loss_scaling: bool = True
    initial_scale: float = 2**15
    growth_factor: float = 2.0
    backoff_factor: float = 0.5
    growth_interval: int = 2000
    
    # Memory optimization
    use_cpu_offload: bool = False
    use_parameter_offload: bool = False
    
    def __post_init__(self):
        if self.precision_type == "auto":
            self.precision_type = self._detect_best_precision()
    
    def _detect_best_precision(self) -> str:
        """Detect the best available precision."""
        if not torch.cuda.is_available():
            return "fp32"
        
        major, minor = torch.cuda.get_device_capability()
        
        # Ampere (RTX 30xx, A100, etc.) and newer support BF16
        if major >= 8:
            return "bf16"
        elif major >= 7:  # Volta/Turing support FP16
            return "fp16"
        elif major >= 6:  # Older GPUs - use TF32 if available
            return "tf32"
        else:
            return "fp32"

@dataclass
class DataConfig:
    """Data processing and loading configuration."""
    
    # Data paths and sources
    train_data_path: str = "data/train.jsonl"
    eval_data_path: Optional[str] = None
    test_data_path: Optional[str] = None
    
    # Data processing
    max_samples_train: Optional[int] = None
    max_samples_eval: Optional[int] = None
    train_split_ratio: float = 0.9
    
    # Text processing
    min_text_length: int = 10
    max_text_length: Optional[int] = None
    remove_duplicates: bool = True
    lowercase: bool = False
    
    # Tokenizer training
    tokenizer_train_size: int = 100000
    min_frequency: int = 2
    
    # Conversation formatting
    use_conversation_format: bool = True
    user_token: str = "<|user|>"
    assistant_token: str = "<|assistant|>"
    system_token: str = "<|system|>"
    end_token: str = "<|end|>"
    
    # Data augmentation
    use_data_augmentation: bool = False
    augmentation_probability: float = 0.1

@dataclass
class HardwareConfig:
    """Hardware-specific optimizations."""
    
    device: str = "auto"
    mixed_precision: bool = True
    
    # Memory management
    memory_efficient_attention: bool = True
    gradient_checkpointing: bool = True
    cpu_offload: bool = False
    
    # Multi-GPU settings
    use_ddp: bool = False
    use_fsdp: bool = False
    
    # Compilation
    use_torch_compile: bool = True
    compile_backend: str = "inductor"

@dataclass
class ModelMetadata:
    """Comprehensive model metadata for tracking and management."""
    
    model_name: str
    version: str = "1.0"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_modified: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Configuration
    model_config: Dict[str, Any] = field(default_factory=dict)
    training_config: Dict[str, Any] = field(default_factory=dict)
    precision_config: Dict[str, Any] = field(default_factory=dict)
    data_config: Dict[str, Any] = field(default_factory=dict)
    
    # Performance metrics
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Model info
    model_size_mb: float = 0.0
    total_parameters: int = 0
    trainable_parameters: int = 0
    
    # Training info
    epochs_trained: int = 0
    total_training_time: float = 0.0
    best_loss: float = float('inf')
    best_perplexity: float = float('inf')
    
    # Environment info
    hardware_used: str = "Unknown"
    pytorch_version: str = field(default_factory=lambda: torch.__version__)
    cuda_version: Optional[str] = field(default_factory=lambda: torch.version.cuda if torch.cuda.is_available() else None)
    
    # Additional info
    notes: str = ""
    tags: List[str] = field(default_factory=list)
    
    # File integrity
    model_hash: Optional[str] = None
    tokenizer_hash: Optional[str] = None

class ModelManager:
    """Enhanced model manager with comprehensive functionality."""
    
    def __init__(self, base_path: str = "models"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.base_path / "checkpoints").mkdir(exist_ok=True)
        (self.base_path / "tokenizers").mkdir(exist_ok=True)
        (self.base_path / "metadata").mkdir(exist_ok=True)
        (self.base_path / "exports").mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(f"{__name__}.ModelManager")
        self.logger.info(f"ModelManager initialized at {self.base_path}")
    
    def generate_model_id(self, metadata: ModelMetadata) -> str:
        """Generate unique model ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name_clean = "".join(c for c in metadata.model_name if c.isalnum() or c in "_-").lower()
        version_clean = "".join(c for c in metadata.version if c.isalnum() or c in "_-.").lower()
        return f"{name_clean}_{version_clean}_{timestamp}"
    
    def calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def save_model(self, model: nn.Module, tokenizer, metadata: ModelMetadata,
                   optimizer=None, scheduler=None, deepspeed_engine=None) -> Optional[str]:
        """Enhanced model saving with comprehensive metadata."""
        try:
            model_id = self.generate_model_id(metadata)
            model_dir = self.base_path / "checkpoints" / model_id
            model_dir.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"Saving model to {model_dir}")
            
            # Update metadata with current info
            metadata.last_modified = datetime.now().isoformat()
            if hasattr(model, 'count_parameters'):
                metadata.total_parameters = model.count_parameters()
                metadata.trainable_parameters = model.count_trainable_parameters()
                metadata.model_size_mb = model.estimate_memory_mb()
            else:
                metadata.total_parameters = sum(p.numel() for p in model.parameters())
                metadata.trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
                metadata.model_size_mb = metadata.total_parameters * 4 / (1024 * 1024)
            
            # Save model
            model_path = model_dir / "model.pt"
            if deepspeed_engine is not None:
                # Save DeepSpeed checkpoint
                deepspeed_engine.save_checkpoint(str(model_dir))
                self.logger.info("DeepSpeed checkpoint saved")
            else:
                # Regular PyTorch checkpoint
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'model_config': metadata.model_config,
                    'metadata': asdict(metadata)
                }
                
                if optimizer is not None:
                    checkpoint['optimizer_state_dict'] = optimizer.state_dict()
                if scheduler is not None:
                    checkpoint['scheduler_state_dict'] = scheduler.state_dict()
                
                torch.save(checkpoint, model_path)
                metadata.model_hash = self.calculate_file_hash(model_path)
            
            # Save tokenizer
            tokenizer_dir = model_dir / "tokenizer"
            tokenizer_dir.mkdir(exist_ok=True)
            
            if hasattr(tokenizer, 'save_vocab'):
                tokenizer.save_vocab(
                    str(tokenizer_dir / "vocab.json"),
                    str(tokenizer_dir / "merges.txt")
                )
            else:
                # Fallback: save tokenizer as pickle
                with open(tokenizer_dir / "tokenizer.pkl", 'wb') as f:
                    pickle.dump(tokenizer, f)
            
            # Calculate tokenizer hash
            tokenizer_files = list(tokenizer_dir.glob("*"))
            if tokenizer_files:
                tokenizer_content = b""
                for tf in sorted(tokenizer_files):
                    with open(tf, 'rb') as f:
                        tokenizer_content += f.read()
                metadata.tokenizer_hash = hashlib.sha256(tokenizer_content).hexdigest()
            
            # Save metadata
            metadata_path = model_dir / "metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(asdict(metadata), f, indent=2, default=str)
            
            # Save human-readable model info
            info_path = model_dir / "model_info.txt"
            self._save_model_info(info_path, metadata, model)
            
            # Update index
            self._update_model_index(model_id, metadata)
            
            self.logger.info(f"Model saved successfully: {model_id}")
            return model_id
            
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
            return None
    
    def load_model(self, model_id: str, device: torch.device = None, load_optimizer: bool = True):
        """Load model with all components."""
        model_dir = self.base_path / "checkpoints" / model_id
        if not model_dir.exists():
            raise FileNotFoundError(f"Model {model_id} not found")
        
        self.logger.info(f"Loading model from {model_dir}")
        
        # Load metadata
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata_dict = json.load(f)
        metadata = ModelMetadata(**metadata_dict)
        
        # Load tokenizer
        tokenizer = self._load_tokenizer(model_dir / "tokenizer")
        
        # Load model
        model_path = model_dir / "model.pt"
        deepspeed_path = model_dir / "latest"  # DeepSpeed checkpoint
        
        if deepspeed_path.exists():
            # DeepSpeed checkpoint - requires special handling
            self.logger.info("DeepSpeed checkpoint detected - manual loading required")
            return None, tokenizer, metadata
        else:
            # Regular PyTorch checkpoint
            checkpoint = torch.load(model_path, map_location=device or 'cpu')
            
            # Reconstruct model
            from subword_transformer import ModernSubwordTransformer
            model_config = ModelConfig(**metadata.model_config)
            model = ModernSubwordTransformer(model_config)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            if device:
                model = model.to(device)
            
            # Load optimizer and scheduler if requested
            optimizer = None
            scheduler = None
            
            if load_optimizer:
                if 'optimizer_state_dict' in checkpoint:
                    from torch.optim import AdamW
                    optimizer = AdamW(model.parameters())
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
                if 'scheduler_state_dict' in checkpoint:
                    from torch.optim.lr_scheduler import OneCycleLR
                    # Note: Scheduler reconstruction might need more work
                    pass
            
            self.logger.info(f"Model loaded successfully: {model_id}")
            return model, tokenizer, metadata, optimizer, scheduler
    
    def _load_tokenizer(self, tokenizer_dir: Path):
        """Load tokenizer from directory."""
        vocab_path = tokenizer_dir / "vocab.json"
        merges_path = tokenizer_dir / "merges.txt"
        pickle_path = tokenizer_dir / "tokenizer.pkl"
        
        if vocab_path.exists() and merges_path.exists():
            from subword_transformer import SubwordTokenizer
            tokenizer = SubwordTokenizer()
            tokenizer.load_vocab(str(vocab_path), str(merges_path))
            return tokenizer
        elif pickle_path.exists():
            with open(pickle_path, 'rb') as f:
                return pickle.load(f)
        else:
            raise FileNotFoundError("No tokenizer files found")
    
    def _save_model_info(self, info_path: Path, metadata: ModelMetadata, model: nn.Module):
        """Save human-readable model information."""
        with open(info_path, 'w', encoding='utf-8') as f:
            f.write(f"Model Information\n")
            f.write(f"=" * 50 + "\n\n")
            
            f.write(f"Name: {metadata.model_name}\n")
            f.write(f"Version: {metadata.version}\n")
            f.write(f"Created: {metadata.created_at}\n")
            f.write(f"Last Modified: {metadata.last_modified}\n\n")
            
            f.write(f"Architecture\n")
            f.write(f"-" * 20 + "\n")
            if metadata.model_config:
                config = metadata.model_config
                f.write(f"Type: {config.get('model_type', 'Unknown')}\n")
                f.write(f"Vocab Size: {config.get('vocab_size', 0):,}\n")
                f.write(f"Hidden Size: {config.get('hidden_size', 0)}\n")
                f.write(f"Layers: {config.get('num_layers', 0)}\n")
                f.write(f"Heads: {config.get('num_heads', 0)}\n")
                f.write(f"Sequence Length: {config.get('seq_length', 0)}\n")
                f.write(f"RoPE: {config.get('use_rotary_pos_emb', False)}\n")
                f.write(f"RMS Norm: {config.get('use_rms_norm', False)}\n")
                f.write(f"GQA: {config.get('use_grouped_query_attention', False)}\n")
                f.write(f"GLU Variant: {config.get('glu_variant', 'None')}\n")
            f.write(f"\n")
            
            f.write(f"Model Statistics\n")
            f.write(f"-" * 20 + "\n")
            f.write(f"Total Parameters: {metadata.total_parameters:,}\n")
            f.write(f"Trainable Parameters: {metadata.trainable_parameters:,}\n")
            f.write(f"Model Size: {metadata.model_size_mb:.2f} MB\n")
            f.write(f"Epochs Trained: {metadata.epochs_trained}\n")
            f.write(f"Best Loss: {metadata.best_loss:.6f}\n")
            f.write(f"Best Perplexity: {metadata.best_perplexity:.2f}\n\n")
            
            if metadata.performance_metrics:
                f.write(f"Performance Metrics\n")
                f.write(f"-" * 20 + "\n")
                for metric, value in metadata.performance_metrics.items():
                    f.write(f"{metric}: {value}\n")
                f.write(f"\n")
            
            f.write(f"Environment\n")
            f.write(f"-" * 20 + "\n")
            f.write(f"Hardware: {metadata.hardware_used}\n")
            f.write(f"PyTorch: {metadata.pytorch_version}\n")
            f.write(f"CUDA: {metadata.cuda_version or 'N/A'}\n\n")
            
            if metadata.notes:
                f.write(f"Notes\n")
                f.write(f"-" * 20 + "\n")
                f.write(f"{metadata.notes}\n\n")
            
            if metadata.tags:
                f.write(f"Tags: {', '.join(metadata.tags)}\n")
    
    def _update_model_index(self, model_id: str, metadata: ModelMetadata):
        """Update the model index for easy lookup."""
        index_path = self.base_path / "model_index.json"
        
        # Load existing index
        if index_path.exists():
            with open(index_path, 'r', encoding='utf-8') as f:
                index = json.load(f)
        else:
            index = {"models": {}, "last_updated": None}
        
        # Add/update entry
        index["models"][model_id] = {
            "name": metadata.model_name,
            "version": metadata.version,
            "created_at": metadata.created_at,
            "last_modified": metadata.last_modified,
            "parameters": metadata.total_parameters,
            "size_mb": metadata.model_size_mb,
            "best_loss": metadata.best_loss,
            "tags": metadata.tags,
            "path": f"checkpoints/{model_id}"
        }
        index["last_updated"] = datetime.now().isoformat()
        
        # Save updated index
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(index, f, indent=2, default=str)
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all saved models."""
        index_path = self.base_path / "model_index.json"
        
        if not index_path.exists():
            return []
        
        with open(index_path, 'r', encoding='utf-8') as f:
            index = json.load(f)
        
        models = []
        for model_id, info in index.get("models", {}).items():
            models.append({
                "id": model_id,
                **info
            })
        
        # Sort by creation date (newest first)
        models.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return models
    
    def delete_model(self, model_id: str) -> bool:
        """Delete a model and its associated files."""
        try:
            model_dir = self.base_path / "checkpoints" / model_id
            if model_dir.exists():
                shutil.rmtree(model_dir)
            
            # Remove from index
            index_path = self.base_path / "model_index.json"
            if index_path.exists():
                with open(index_path, 'r', encoding='utf-8') as f:
                    index = json.load(f)
                
                if model_id in index.get("models", {}):
                    del index["models"][model_id]
                    index["last_updated"] = datetime.now().isoformat()
                    
                    with open(index_path, 'w', encoding='utf-8') as f:
                        json.dump(index, f, indent=2, default=str)
            
            self.logger.info(f"Model {model_id} deleted successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete model {model_id}: {e}")
            return False
    
    def print_model_summary(self):
        """Print a summary of all models."""
        models = self.list_models()
        
        if not models:
            print("No models found.")
            return
        
        print(f"\n📚 Model Summary ({len(models)} models)")
        print("=" * 80)
        
        for model in models:
            print(f"🔹 {model['name']} ({model['id']})")
            print(f"   Version: {model['version']}")
            print(f"   Parameters: {model['parameters']:,}")
            print(f"   Size: {model['size_mb']:.1f}MB")
            print(f"   Best Loss: {model['best_loss']:.4f}")
            print(f"   Created: {model['created_at']}")
            if model.get('tags'):
                print(f"   Tags: {', '.join(model['tags'])}")
            print()

# Configuration Presets
class ConfigPresets:
    """Predefined configurations for different scenarios."""
    
    @staticmethod
    def auto_detect() -> Tuple[ModelConfig, TrainingConfig, PrecisionConfig, DataConfig]:
        """Automatically detect optimal configuration."""
        return auto_select_config()
    
    @staticmethod
    def tiny_debug() -> Tuple[ModelConfig, TrainingConfig, PrecisionConfig, DataConfig]:
        """Minimal config for debugging and testing."""
        model = ModelConfig(
            vocab_size=1000,
            hidden_size=128,
            num_layers=2,
            num_heads=4,
            seq_length=256,
            use_grouped_query_attention=False,
            gradient_checkpointing=False
        )
        
        training = TrainingConfig(
            batch_size=2,
            gradient_accumulation_steps=2,
            max_epochs=2,
            learning_rate=1e-3,
            num_workers=0
        )
        
        precision = PrecisionConfig(precision_type="fp32", use_mixed_precision=False)
        data = DataConfig(max_samples_train=100, max_samples_eval=20)
        
        return model, training, precision, data
    
    @staticmethod
    def research_7b() -> Tuple[ModelConfig, TrainingConfig, PrecisionConfig, DataConfig]:
        """Configuration for a research-grade 7B model."""
        model = ModelConfig(
            vocab_size=50000,
            hidden_size=4096,
            num_layers=32,
            num_heads=32,
            seq_length=8192,
            num_key_value_heads=8,
            intermediate_size=11008,
            gradient_checkpointing=True
        )
        
        training = TrainingConfig(
            batch_size=2,
            gradient_accumulation_steps=16,
            max_epochs=1,
            learning_rate=1e-4,
            warmup_ratio=0.01,
            beta2=0.95
        )
        
        precision = PrecisionConfig()  # Auto-detect
        data = DataConfig()  # No limits
        
        return model, training, precision, data

def auto_select_config() -> Tuple[ModelConfig, TrainingConfig, PrecisionConfig, DataConfig]:
    """Automatically select optimal configuration based on hardware."""
    
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        total_memory_gb = props.total_memory / (1024**3)
        
        if total_memory_gb >= 40:  # A100, H100
            return ConfigPresets.research_7b()
        elif total_memory_gb >= 20:  # RTX 4090, A6000
            model = ModelConfig(
                vocab_size=32000, hidden_size=2048, num_layers=24, num_heads=16,
                seq_length=4096, num_key_value_heads=4, gradient_checkpointing=True
            )
            training = TrainingConfig(batch_size=4, gradient_accumulation_steps=8, max_epochs=3)
            precision = PrecisionConfig()
            data = DataConfig(max_samples_train=100000)
        elif total_memory_gb >= 10:  # RTX 3080, 4070
            model = ModelConfig(
                vocab_size=16000, hidden_size=1024, num_layers=12, num_heads=8,
                seq_length=2048, num_key_value_heads=2, gradient_checkpointing=True
            )
            training = TrainingConfig(batch_size=2, gradient_accumulation_steps=16, max_epochs=5)
            precision = PrecisionConfig()
            data = DataConfig(max_samples_train=50000)
        else:  # Low VRAM
            return ConfigPresets.tiny_debug()
        
        return model, training, precision, data
    
    else:  # CPU/MPS fallback
        return ConfigPresets.tiny_debug()