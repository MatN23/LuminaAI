# Copyright (c) 2025 Matias Nielsen. All rights reserved.
# Licensed under the Custom License below.

import os
import json
import time
import hashlib
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict, field
import torch
import torch.nn as nn
from contextlib import contextmanager

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    vocab_size: int = 32000
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    seq_length: int = 512
    dropout: float = 0.1
    model_type: str = "SubwordTransformer"
    tokenizer_type: str = "subword"
    intermediate_size: Optional[int] = None
    activation: str = "gelu"
    layer_norm_eps: float = 1e-5
    tie_word_embeddings: bool = True
    
    def __post_init__(self):
        if self.intermediate_size is None:
            self.intermediate_size = 4 * self.hidden_size

@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 16
    gradient_accumulation_steps: int = 4
    max_epochs: int = 50
    warmup_ratio: float = 0.1
    save_every: int = 1000
    eval_every: int = 500
    max_grad_norm: float = 1.0
    label_smoothing: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    scheduler_type: str = "cosine"
    min_lr_ratio: float = 0.1

@dataclass
class ModelMetadata:
    """Comprehensive metadata for saved models with JSON serialization support."""
    model_name: str
    version: str
    created_at: str
    last_modified: str
    model_config: Dict[str, Any] = field(default_factory=dict)
    training_config: Dict[str, Any] = field(default_factory=dict)
    dataset_info: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    model_size_mb: float = 0.0
    total_parameters: int = 0
    trainable_parameters: int = 0
    training_time_hours: float = 0.0
    epochs_trained: int = 0
    best_loss: float = float('inf')
    best_perplexity: float = float('inf')
    hardware_used: str = ""
    pytorch_version: str = ""
    cuda_version: Optional[str] = None
    model_hash: str = ""
    tokenizer_hash: str = ""
    notes: str = ""
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Convert dataclass objects to dictionaries for JSON serialization."""
        # Convert ModelConfig to dict if it's a dataclass
        if hasattr(self.model_config, '__dataclass_fields__'):
            self.model_config = asdict(self.model_config)
        
        # Convert TrainingConfig to dict if it's a dataclass
        if hasattr(self.training_config, '__dataclass_fields__'):
            self.training_config = asdict(self.training_config)
        
        # Ensure all numeric values are JSON serializable
        if isinstance(self.best_loss, torch.Tensor):
            self.best_loss = float(self.best_loss.item())
        elif not isinstance(self.best_loss, (int, float)):
            self.best_loss = float(self.best_loss)
        
        if isinstance(self.best_perplexity, torch.Tensor):
            self.best_perplexity = float(self.best_perplexity.item())
        elif not isinstance(self.best_perplexity, (int, float)):
            self.best_perplexity = float(self.best_perplexity)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with JSON-safe values."""
        data = asdict(self)
        
        # Ensure all values are JSON serializable
        def make_json_safe(obj):
            if isinstance(obj, torch.Tensor):
                return float(obj.item()) if obj.numel() == 1 else obj.tolist()
            elif isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            elif isinstance(obj, dict):
                return {k: make_json_safe(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [make_json_safe(item) for item in obj]
            else:
                return str(obj)
        
        return make_json_safe(data)

class ModelManager:
    """Enhanced model management system with robust saving and loading."""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.models_dir / "checkpoints").mkdir(exist_ok=True)
        (self.models_dir / "metadata").mkdir(exist_ok=True)
        (self.models_dir / "tokenizers").mkdir(exist_ok=True)
        
        logger.info(f"ModelManager initialized with directory: {self.models_dir}")
    
    @contextmanager
    def safe_cuda_context(self):
        """Context manager for safe CUDA operations."""
        try:
            yield
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
    
    def _generate_model_id(self, model_name: str) -> str:
        """Generate unique model ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        clean_name = "".join(c for c in model_name if c.isalnum() or c in "_-").lower()
        return f"{clean_name}_{timestamp}"
    
    def _calculate_hash(self, file_path: Path) -> str:
        """Calculate file hash for integrity checking."""
        try:
            hash_sha256 = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()[:16]  # First 16 chars
        except Exception as e:
            logger.warning(f"Could not calculate hash for {file_path}: {e}")
            return ""
    
    def save_model(self, model: nn.Module, tokenizer, metadata: ModelMetadata, 
                   optimizer=None, scheduler=None, force_cpu_save: bool = True) -> str:
        """
        Save model with comprehensive error handling and proper serialization.
        
        Args:
            model: The PyTorch model to save
            tokenizer: The tokenizer (SubwordTokenizer)
            metadata: Model metadata
            optimizer: Optional optimizer state
            scheduler: Optional scheduler state
            force_cpu_save: Whether to move model to CPU before saving
        
        Returns:
            str: The generated model ID
        """
        try:
            # Generate unique model ID
            model_id = self._generate_model_id(metadata.model_name)
            
            logger.info(f"💾 Saving model: {model_id}")
            logger.info(f"   Model name: {metadata.model_name}")
            logger.info(f"   Version: {metadata.version}")
            
            # Create model directory
            model_dir = self.models_dir / model_id
            model_dir.mkdir(exist_ok=True)
            
            # File paths
            model_path = model_dir / "model.pt"
            tokenizer_vocab_path = model_dir / "vocab.json"
            tokenizer_merges_path = model_dir / "merges.txt"
            metadata_path = model_dir / "metadata.json"
            
            # Save model checkpoint
            logger.info("   Saving model checkpoint...")
            with self.safe_cuda_context():
                # Prepare model for saving
                if force_cpu_save:
                    # Move model to CPU for stable saving
                    original_device = next(model.parameters()).device
                    model_cpu = model.cpu()
                    
                    checkpoint = {
                        'model_state_dict': model_cpu.state_dict(),
                        'model_config': metadata.model_config,
                        'model_type': metadata.model_config.get('model_type', 'SubwordTransformer'),
                        'vocab_size': metadata.model_config.get('vocab_size', 32000),
                        'created_at': metadata.created_at,
                        'version': metadata.version
                    }
                    
                    # Add optimizer and scheduler if provided
                    if optimizer is not None:
                        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
                    
                    if scheduler is not None:
                        if hasattr(scheduler, 'state_dict'):
                            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
                        else:
                            # Handle custom scheduler
                            checkpoint['scheduler_state_dict'] = {
                                'current_step': getattr(scheduler, 'current_step', 0),
                                'warmup_steps': getattr(scheduler, 'warmup_steps', 0),
                                'total_steps': getattr(scheduler, 'total_steps', 0),
                            }
                    
                    # Save checkpoint
                    torch.save(checkpoint, model_path)
                    
                    # Move model back to original device
                    model.to(original_device)
                    
                else:
                    # Save without moving to CPU
                    checkpoint = {
                        'model_state_dict': model.state_dict(),
                        'model_config': metadata.model_config,
                        'model_type': metadata.model_config.get('model_type', 'SubwordTransformer'),
                        'vocab_size': metadata.model_config.get('vocab_size', 32000),
                        'created_at': metadata.created_at,
                        'version': metadata.version
                    }
                    
                    if optimizer is not None:
                        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
                    
                    if scheduler is not None:
                        if hasattr(scheduler, 'state_dict'):
                            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
                        else:
                            checkpoint['scheduler_state_dict'] = {
                                'current_step': getattr(scheduler, 'current_step', 0),
                                'warmup_steps': getattr(scheduler, 'warmup_steps', 0),
                                'total_steps': getattr(scheduler, 'total_steps', 0),
                            }
                    
                    torch.save(checkpoint, model_path)
            
            logger.info(f"   ✅ Model checkpoint saved: {model_path}")
            
            # Save tokenizer
            logger.info("   Saving tokenizer...")
            try:
                if hasattr(tokenizer, 'save_vocab'):
                    tokenizer.save_vocab(str(tokenizer_vocab_path), str(tokenizer_merges_path))
                else:
                    # Manual tokenizer saving
                    with open(tokenizer_vocab_path, 'w', encoding='utf-8') as f:
                        json.dump(tokenizer.vocab, f, indent=2, ensure_ascii=False)
                    
                    with open(tokenizer_merges_path, 'w', encoding='utf-8') as f:
                        for pair in tokenizer.merges:
                            f.write(f"{pair[0]} {pair[1]}\n")
                
                logger.info(f"   ✅ Tokenizer saved: vocab={tokenizer_vocab_path}, merges={tokenizer_merges_path}")
                
            except Exception as tokenizer_error:
                logger.error(f"   ❌ Failed to save tokenizer: {tokenizer_error}")
                # Try alternative saving method
                try:
                    tokenizer_data = {
                        'vocab': tokenizer.vocab,
                        'merges': tokenizer.merges,
                        'vocab_size': tokenizer.vocab_size(),
                        'tokenizer_type': 'SubwordTokenizer'
                    }
                    with open(model_dir / "tokenizer.json", 'w', encoding='utf-8') as f:
                        json.dump(tokenizer_data, f, indent=2, ensure_ascii=False)
                    logger.info("   ✅ Tokenizer saved as single JSON file")
                except Exception as alt_error:
                    logger.error(f"   ❌ Alternative tokenizer save also failed: {alt_error}")
                    raise
            
            # Calculate file hashes
            logger.info("   Calculating file hashes...")
            metadata.model_hash = self._calculate_hash(model_path)
            if tokenizer_vocab_path.exists():
                metadata.tokenizer_hash = self._calculate_hash(tokenizer_vocab_path)
            
            # Save metadata with proper JSON serialization
            logger.info("   Saving metadata...")
            try:
                # Ensure metadata is properly serializable
                metadata.last_modified = datetime.now().isoformat()
                
                # Convert to dict and ensure JSON compatibility
                metadata_dict = metadata.to_dict()
                
                # Additional safety checks
                def ensure_json_serializable(obj):
                    """Recursively ensure all values are JSON serializable."""
                    if isinstance(obj, dict):
                        return {k: ensure_json_serializable(v) for k, v in obj.items()}
                    elif isinstance(obj, (list, tuple)):
                        return [ensure_json_serializable(item) for item in obj]
                    elif isinstance(obj, (int, float, str, bool, type(None))):
                        return obj
                    elif hasattr(obj, 'item'):  # torch.Tensor
                        return float(obj.item())
                    elif hasattr(obj, '__dict__'):  # Custom objects
                        return str(obj)
                    else:
                        return str(obj)
                
                safe_metadata = ensure_json_serializable(metadata_dict)
                
                # Write metadata
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(safe_metadata, f, indent=2, ensure_ascii=False, default=str)
                
                logger.info(f"   ✅ Metadata saved: {metadata_path}")
                
            except Exception as metadata_error:
                logger.error(f"   ❌ Failed to save metadata: {metadata_error}")
                logger.error(f"   Metadata error traceback: {traceback.format_exc()}")
                
                # Try saving minimal metadata
                try:
                    minimal_metadata = {
                        'model_name': metadata.model_name,
                        'version': metadata.version,
                        'created_at': metadata.created_at,
                        'last_modified': datetime.now().isoformat(),
                        'model_config': metadata.model_config,
                        'best_loss': float(metadata.best_loss),
                        'total_parameters': int(metadata.total_parameters),
                        'model_hash': metadata.model_hash,
                        'tokenizer_hash': metadata.tokenizer_hash,
                        'notes': str(metadata.notes)
                    }
                    
                    with open(metadata_path, 'w', encoding='utf-8') as f:
                        json.dump(minimal_metadata, f, indent=2, ensure_ascii=False)
                    
                    logger.info("   ✅ Minimal metadata saved successfully")
                    
                except Exception as minimal_error:
                    logger.error(f"   ❌ Even minimal metadata save failed: {minimal_error}")
                    raise
            
            # Create model registry entry
            logger.info("   Updating model registry...")
            self._update_model_registry(model_id, metadata)
            
            # Final validation
            if not self._validate_saved_model(model_id):
                logger.warning(f"   ⚠️ Model validation failed for {model_id}")
            
            logger.info(f"✅ Model saved successfully: {model_id}")
            logger.info(f"   Location: {model_dir}")
            logger.info(f"   Files: model.pt, vocab.json, merges.txt, metadata.json")
            
            return model_id
            
        except Exception as e:
            logger.error(f"❌ Failed to save model: {e}")
            logger.error(f"Save error traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Model saving failed: {e}")
    
    def _update_model_registry(self, model_id: str, metadata: ModelMetadata):
        """Update the global model registry."""
        registry_path = self.models_dir / "registry.json"
        
        try:
            # Load existing registry
            if registry_path.exists():
                with open(registry_path, 'r', encoding='utf-8') as f:
                    registry = json.load(f)
            else:
                registry = {"models": {}, "created_at": datetime.now().isoformat()}
            
            # Add model entry
            registry["models"][model_id] = {
                "model_name": metadata.model_name,
                "version": metadata.version,
                "created_at": metadata.created_at,
                "last_modified": metadata.last_modified,
                "best_loss": float(metadata.best_loss),
                "total_parameters": int(metadata.total_parameters),
                "model_size_mb": float(metadata.model_size_mb),
                "epochs_trained": int(metadata.epochs_trained),
                "tags": list(metadata.tags),
                "model_hash": metadata.model_hash,
                "tokenizer_hash": metadata.tokenizer_hash
            }
            
            registry["last_updated"] = datetime.now().isoformat()
            registry["total_models"] = len(registry["models"])
            
            # Save registry
            with open(registry_path, 'w', encoding='utf-8') as f:
                json.dump(registry, f, indent=2, ensure_ascii=False)
            
            logger.info(f"   ✅ Registry updated: {len(registry['models'])} total models")
            
        except Exception as e:
            logger.error(f"   ❌ Failed to update registry: {e}")
    
    def _validate_saved_model(self, model_id: str) -> bool:
        """Validate that the model was saved correctly."""
        try:
            model_dir = self.models_dir / model_id
            
            required_files = ["model.pt", "metadata.json"]
            optional_files = ["vocab.json", "merges.txt", "tokenizer.json"]
            
            # Check required files
            for file_name in required_files:
                file_path = model_dir / file_name
                if not file_path.exists():
                    logger.error(f"Missing required file: {file_path}")
                    return False
                if file_path.stat().st_size == 0:
                    logger.error(f"Empty required file: {file_path}")
                    return False
            
            # Check at least one tokenizer file exists
            tokenizer_exists = any((model_dir / file_name).exists() for file_name in optional_files)
            if not tokenizer_exists:
                logger.warning(f"No tokenizer files found for {model_id}")
            
            # Try to load metadata
            try:
                with open(model_dir / "metadata.json", 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                if not isinstance(metadata, dict):
                    logger.error(f"Invalid metadata format for {model_id}")
                    return False
            except Exception as e:
                logger.error(f"Could not load metadata for {model_id}: {e}")
                return False
            
            logger.info(f"   ✅ Model validation passed for {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Model validation error for {model_id}: {e}")
            return False
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all available models with comprehensive information."""
        models = []
        
        try:
            # Try to load from registry first
            registry_path = self.models_dir / "registry.json"
            if registry_path.exists():
                with open(registry_path, 'r', encoding='utf-8') as f:
                    registry = json.load(f)
                
                for model_id, info in registry["models"].items():
                    model_info = {
                        "id": model_id,
                        "model_name": info.get("model_name", "Unknown"),
                        "version": info.get("version", "Unknown"),
                        "created_at": info.get("created_at", "Unknown"),
                        "best_loss": info.get("best_loss", float('inf')),
                        "total_parameters": info.get("total_parameters", 0),
                        "model_size_mb": info.get("model_size_mb", 0.0),
                        "epochs_trained": info.get("epochs_trained", 0),
                        "tags": info.get("tags", []),
                        "validated": self._validate_saved_model(model_id)
                    }
                    models.append(model_info)
            
            else:
                # Fallback: scan directories
                for model_dir in self.models_dir.iterdir():
                    if model_dir.is_dir() and not model_dir.name.startswith('.'):
                        try:
                            metadata_path = model_dir / "metadata.json"
                            if metadata_path.exists():
                                with open(metadata_path, 'r', encoding='utf-8') as f:
                                    metadata = json.load(f)
                                
                                model_info = {
                                    "id": model_dir.name,
                                    "model_name": metadata.get("model_name", "Unknown"),
                                    "version": metadata.get("version", "Unknown"),
                                    "created_at": metadata.get("created_at", "Unknown"),
                                    "best_loss": metadata.get("best_loss", float('inf')),
                                    "total_parameters": metadata.get("total_parameters", 0),
                                    "model_size_mb": metadata.get("model_size_mb", 0.0),
                                    "epochs_trained": metadata.get("epochs_trained", 0),
                                    "tags": metadata.get("tags", []),
                                    "validated": self._validate_saved_model(model_dir.name)
                                }
                                models.append(model_info)
                        except Exception as e:
                            logger.warning(f"Could not load model info for {model_dir.name}: {e}")
            
            # Sort by creation date (newest first)
            models.sort(key=lambda x: x.get("created_at", ""), reverse=True)
            
        except Exception as e:
            logger.error(f"Error listing models: {e}")
        
        return models
    
    def load_model(self, model_id: str) -> Tuple[nn.Module, Any, ModelMetadata]:
        """Load a saved model with error handling."""
        try:
            model_dir = self.models_dir / model_id
            if not model_dir.exists():
                raise FileNotFoundError(f"Model directory not found: {model_dir}")
            
            logger.info(f"Loading model: {model_id}")
            
            # Load metadata
            metadata_path = model_dir / "metadata.json"
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata_dict = json.load(f)
            
            # Reconstruct metadata object
            metadata = ModelMetadata(**metadata_dict)
            
            # Load tokenizer
            tokenizer = self._load_tokenizer(model_dir)
            
            # Load model
            model_path = model_dir / "model.pt"
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Create model instance
            from subword_transformer import SubwordTransformer
            
            # Convert dict back to ModelConfig if needed
            if isinstance(metadata.model_config, dict):
                model_config = ModelConfig(**metadata.model_config)
            else:
                model_config = metadata.model_config
            
            model = SubwordTransformer(model_config)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            logger.info(f"✅ Model loaded successfully: {model_id}")
            
            return model, tokenizer, metadata
            
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            raise
    
    def _load_tokenizer(self, model_dir: Path):
        """Load tokenizer from model directory."""
        try:
            from subword_transformer import SubwordTokenizer
            
            vocab_path = model_dir / "vocab.json"
            merges_path = model_dir / "merges.txt"
            tokenizer_path = model_dir / "tokenizer.json"
            
            if vocab_path.exists() and merges_path.exists():
                # Load standard format
                tokenizer = SubwordTokenizer()
                tokenizer.load_vocab(str(vocab_path), str(merges_path))
                return tokenizer
            
            elif tokenizer_path.exists():
                # Load single file format
                with open(tokenizer_path, 'r', encoding='utf-8') as f:
                    tokenizer_data = json.load(f)
                
                tokenizer = SubwordTokenizer(
                    vocab=tokenizer_data['vocab'],
                    merges=tokenizer_data['merges']
                )
                return tokenizer
            
            else:
                raise FileNotFoundError("No tokenizer files found")
                
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise
    
    def delete_model(self, model_id: str) -> bool:
        """Delete a model and all its files."""
        try:
            model_dir = self.models_dir / model_id
            if not model_dir.exists():
                logger.warning(f"Model directory not found: {model_dir}")
                return False
            
            # Remove directory and all files
            import shutil
            shutil.rmtree(model_dir)
            
            # Update registry
            registry_path = self.models_dir / "registry.json"
            if registry_path.exists():
                with open(registry_path, 'r', encoding='utf-8') as f:
                    registry = json.load(f)
                
                if model_id in registry["models"]:
                    del registry["models"][model_id]
                    registry["total_models"] = len(registry["models"])
                    registry["last_updated"] = datetime.now().isoformat()
                    
                    with open(registry_path, 'w', encoding='utf-8') as f:
                        json.dump(registry, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Model deleted: {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete model {model_id}: {e}")
            return False
    
    def validate_model(self, model_id: str) -> Dict[str, Any]:
        """Comprehensive model validation."""
        validation_result = {
            "valid": False,
            "issues": [],
            "files_found": [],
            "files_missing": [],
            "model_loadable": False,
            "tokenizer_loadable": False,
            "metadata_valid": False
        }
        
        try:
            model_dir = self.models_dir / model_id
            if not model_dir.exists():
                validation_result["issues"].append(f"Model directory not found: {model_dir}")
                return validation_result
            
            # Check files
            required_files = ["model.pt", "metadata.json"]
            optional_files = ["vocab.json", "merges.txt", "tokenizer.json"]
            
            for file_name in required_files + optional_files:
                file_path = model_dir / file_name
                if file_path.exists():
                    validation_result["files_found"].append(file_name)
                elif file_name in required_files:
                    validation_result["files_missing"].append(file_name)
                    validation_result["issues"].append(f"Missing required file: {file_name}")
            
            # Validate metadata
            try:
                metadata_path = model_dir / "metadata.json"
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                validation_result["metadata_valid"] = isinstance(metadata, dict)
            except Exception as e:
                validation_result["issues"].append(f"Metadata validation failed: {e}")
            
            # Try loading tokenizer
            try:
                tokenizer = self._load_tokenizer(model_dir)
                validation_result["tokenizer_loadable"] = True
            except Exception as e:
                validation_result["issues"].append(f"Tokenizer loading failed: {e}")
            
            # Try loading model (basic check)
            try:
                model_path = model_dir / "model.pt"
                checkpoint = torch.load(model_path, map_location='cpu')
                if 'model_state_dict' in checkpoint:
                    validation_result["model_loadable"] = True
                else:
                    validation_result["issues"].append("No model_state_dict in checkpoint")
            except Exception as e:
                validation_result["issues"].append(f"Model checkpoint loading failed: {e}")
            
            # Overall validation
            validation_result["valid"] = (
                len(validation_result["files_missing"]) == 0 and
                validation_result["metadata_valid"] and
                validation_result["tokenizer_loadable"] and
                validation_result["model_loadable"]
            )
            
        except Exception as e:
            validation_result["issues"].append(f"Validation error: {e}")
        
        return validation_result
    
    def print_model_summary(self):
        """Print a comprehensive summary of all models."""
        models = self.list_models()
        
        print("=" * 80)
        print("🤖  MODEL MANAGER SUMMARY")
        print("=" * 80)
        print(f"📊  Total Models: {len(models)}")
        print(f"💾  Storage Location: {self.models_dir}")
        
        if not models:
            print("⚠️  No models found")
        else:
            print(f"🏆  Best Model (by loss): {min(models, key=lambda x: x.get('best_loss', float('inf')))['model_name']}")
            print(f"📈  Total Parameters: {sum(m.get('total_parameters', 0) for m in models):,}")
            print(f"💽  Total Storage: {sum(m.get('model_size_mb', 0) for m in models):.1f} MB")
            print()
            
            # List models
            print("📁  AVAILABLE MODELS:")
            print("-" * 80)
            for i, model in enumerate(models[:10], 1):  # Show top 10
                status = "✅" if model.get('validated', False) else "⚠️"
                print(f"{i:2d}. {status} {model['model_name']} ({model['id']})")
                print(f"     📊 Loss: {model.get('best_loss', 'N/A'):.4f} | "
                      f"📐 Params: {model.get('total_parameters', 0):,} | "
                      f"🏋️ Size: {model.get('model_size_mb', 0):.1f}MB")
                print(f"     📅 Created: {model.get('created_at', 'Unknown')[:19]}")
                if model.get('tags'):
                    print(f"     🏷️  Tags: {', '.join(model['tags'][:5])}")
                print()
            
            if len(models) > 10:
                print(f"... and {len(models) - 10} more models")
        
        print("=" * 80)
    
    def get_storage_info(self) -> Dict[str, Any]:
        """Get detailed storage information."""
        storage_info = {
            "total_models": 0,
            "total_size_mb": 0.0,
            "total_parameters": 0,
            "storage_path": str(self.models_dir),
            "disk_usage": {},
            "file_breakdown": {}
        }
        
        try:
            models = self.list_models()
            storage_info["total_models"] = len(models)
            
            for model in models:
                storage_info["total_size_mb"] += model.get("model_size_mb", 0.0)
                storage_info["total_parameters"] += model.get("total_parameters", 0)
            
            # Get disk usage
            if self.models_dir.exists():
                total_size = 0
                file_counts = {"model.pt": 0, "metadata.json": 0, "vocab.json": 0, "merges.txt": 0}
                
                for model_dir in self.models_dir.iterdir():
                    if model_dir.is_dir():
                        for file_path in model_dir.rglob("*"):
                            if file_path.is_file():
                                size = file_path.stat().st_size
                                total_size += size
                                
                                if file_path.name in file_counts:
                                    file_counts[file_path.name] += 1
                
                storage_info["disk_usage"] = {
                    "total_bytes": total_size,
                    "total_mb": total_size / (1024 * 1024),
                    "total_gb": total_size / (1024 * 1024 * 1024)
                }
                storage_info["file_breakdown"] = file_counts
        
        except Exception as e:
            logger.error(f"Error getting storage info: {e}")
        
        return storage_info
    
    def cleanup_invalid_models(self) -> int:
        """Remove invalid or corrupted models."""
        removed_count = 0
        
        try:
            models = self.list_models()
            
            for model in models:
                validation = self.validate_model(model["id"])
                
                if not validation["valid"]:
                    logger.info(f"Removing invalid model: {model['id']}")
                    logger.info(f"Issues: {validation['issues']}")
                    
                    if self.delete_model(model["id"]):
                        removed_count += 1
        
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        
        logger.info(f"Cleanup completed: {removed_count} models removed")
        return removed_count
    
    def export_model_info(self, output_file: str = "model_export.json"):
        """Export comprehensive model information to JSON."""
        try:
            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "manager_info": {
                    "models_directory": str(self.models_dir),
                    "pytorch_version": torch.__version__
                },
                "storage_info": self.get_storage_info(),
                "models": []
            }
            
            models = self.list_models()
            for model in models:
                model_info = model.copy()
                
                # Add validation info
                validation = self.validate_model(model["id"])
                model_info["validation"] = validation
                
                # Add detailed file info
                model_dir = self.models_dir / model["id"]
                if model_dir.exists():
                    file_info = {}
                    for file_path in model_dir.iterdir():
                        if file_path.is_file():
                            file_info[file_path.name] = {
                                "size_bytes": file_path.stat().st_size,
                                "size_mb": file_path.stat().st_size / (1024 * 1024),
                                "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                            }
                    model_info["files"] = file_info
                
                export_data["models"].append(model_info)
            
            # Write export file
            output_path = Path(output_file)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"✅ Model information exported to: {output_path}")
            return str(output_path)
        
        except Exception as e:
            logger.error(f"Failed to export model info: {e}")
            raise