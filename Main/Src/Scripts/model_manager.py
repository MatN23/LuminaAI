# Copyright (c) 2025 Matias Nielsen. All rights reserved.
# Licensed under the Custom License below.

import os
import json
import hashlib
import pickle
import shutil
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for the transformer model."""
    vocab_size: int = 32000
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    seq_length: int = 1024
    dropout: float = 0.1
    model_type: str = "WordTransformer"
    tokenizer_type: str = "word"

@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    max_epochs: int = 50
    warmup_ratio: float = 0.1
    save_every: int = 1000
    eval_every: int = 500
    max_grad_norm: float = 1.0
    label_smoothing: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95

@dataclass
class ModelMetadata:
    """Comprehensive metadata for trained models."""
    # Basic info
    model_name: str
    version: str
    created_at: str
    last_modified: str
    
    # Configuration
    model_config: ModelConfig
    training_config: TrainingConfig
    
    # Dataset information
    dataset_info: Dict[str, Any]
    
    # Performance metrics
    performance_metrics: Dict[str, float]
    
    # Model details
    model_size_mb: float
    total_parameters: int
    trainable_parameters: int
    training_time_hours: float
    epochs_trained: int
    best_loss: float
    best_perplexity: float
    
    # Technical details
    hardware_used: str
    pytorch_version: str
    cuda_version: Optional[str]
    model_hash: str
    tokenizer_hash: str
    
    # Additional info
    notes: str = ""
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []

class ModelManager:
    """Professional model management system with versioning and metadata."""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.models_dir / "checkpoints").mkdir(exist_ok=True)
        (self.models_dir / "metadata").mkdir(exist_ok=True)
        (self.models_dir / "tokenizers").mkdir(exist_ok=True)
        
        logger.info(f"ModelManager initialized with directory: {self.models_dir}")
    
    def _calculate_hash(self, obj: Any) -> str:
        """Calculate SHA256 hash of an object."""
        try:
            if hasattr(obj, 'state_dict'):
                # For PyTorch models/optimizers
                state_bytes = pickle.dumps(obj.state_dict())
            else:
                # For other objects
                state_bytes = pickle.dumps(obj)
            return hashlib.sha256(state_bytes).hexdigest()[:16]
        except Exception as e:
            logger.warning(f"Could not calculate hash: {e}")
            return "unknown"
    
    def _get_model_size(self, model_path: Path) -> float:
        """Calculate model file size in MB."""
        try:
            return model_path.stat().st_size / (1024 * 1024)
        except Exception:
            return 0.0
    
    def _generate_model_id(self, metadata: ModelMetadata) -> str:
        """Generate unique model ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name_clean = metadata.model_name.replace(" ", "_").lower()
        version_clean = metadata.version.replace(" ", "_").replace(".", "_").lower()
        return f"{name_clean}_{version_clean}_{timestamp}"
    
    def save_model(self, model: nn.Module, tokenizer, metadata: ModelMetadata,
                   optimizer=None, scheduler=None) -> str:
        """Save model with comprehensive metadata and versioning."""
        try:
            # Generate model ID
            model_id = self._generate_model_id(metadata)
            
            # Create model directory
            model_dir = self.models_dir / "checkpoints" / model_id
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model
            model_path = model_dir / "model.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_config': asdict(metadata.model_config),
                'model_id': model_id
            }, model_path)
            
            # Save tokenizer
            tokenizer_path = model_dir / "tokenizer.pkl"
            with open(tokenizer_path, 'wb') as f:
                pickle.dump(tokenizer, f)
            
            # Save optimizer and scheduler if provided
            if optimizer:
                optimizer_path = model_dir / "optimizer.pt"
                torch.save(optimizer.state_dict(), optimizer_path)
            
            if scheduler:
                scheduler_path = model_dir / "scheduler.pt"
                torch.save(scheduler.state_dict(), scheduler_path)
            
            # Calculate hashes and file size
            metadata.model_hash = self._calculate_hash(model)
            metadata.tokenizer_hash = self._calculate_hash(tokenizer)
            metadata.model_size_mb = self._get_model_size(model_path)
            metadata.last_modified = datetime.now().isoformat()
            
            # Save metadata
            metadata_path = self.models_dir / "metadata" / f"{model_id}.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(asdict(metadata), f, indent=2, default=str)
            
            # Create symlink to latest model if this is the best
            if "best" in metadata.tags:
                latest_link = self.models_dir / "latest_best"
                if latest_link.exists():
                    latest_link.unlink()
                try:
                    latest_link.symlink_to(model_dir.name)
                except OSError:
                    # Fallback for systems that don't support symlinks
                    with open(latest_link, 'w') as f:
                        f.write(model_id)
            
            logger.info(f"✅ Model saved successfully: {model_id}")
            logger.info(f"   Size: {metadata.model_size_mb:.2f} MB")
            logger.info(f"   Parameters: {metadata.total_parameters:,}")
            logger.info(f"   Location: {model_dir}")
            
            return model_id
            
        except Exception as e:
            logger.error(f"❌ Failed to save model: {e}")
            raise
    
    def load_model(self, model_id: str) -> Tuple[nn.Module, Any, ModelMetadata]:
        """Load model with tokenizer and metadata."""
        try:
            # Handle special case for 'latest' or 'best'
            if model_id in ['latest', 'best', 'latest_best']:
                latest_link = self.models_dir / "latest_best"
                if latest_link.exists():
                    if latest_link.is_symlink():
                        model_id = latest_link.readlink().name
                    else:
                        with open(latest_link, 'r') as f:
                            model_id = f.read().strip()
                else:
                    # Find best model by loss
                    models = self.list_models()
                    if not models:
                        raise FileNotFoundError("No models available")
                    model_id = min(models, key=lambda x: x['best_loss'])['id']
            
            model_dir = self.models_dir / "checkpoints" / model_id
            if not model_dir.exists():
                raise FileNotFoundError(f"Model not found: {model_id}")
            
            # Load metadata
            metadata_path = self.models_dir / "metadata" / f"{model_id}.json"
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata_dict = json.load(f)
            
            # Reconstruct metadata object
            metadata_dict['model_config'] = ModelConfig(**metadata_dict['model_config'])
            metadata_dict['training_config'] = TrainingConfig(**metadata_dict['training_config'])
            metadata = ModelMetadata(**metadata_dict)
            
            # Load tokenizer
            tokenizer_path = model_dir / "tokenizer.pkl"
            with open(tokenizer_path, 'rb') as f:
                tokenizer = pickle.load(f)
            
            # Load model
            from word_transformer import WordTransformer
            model = WordTransformer(metadata.model_config)
            
            model_path = model_dir / "model.pt"
            checkpoint = torch.load(model_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            
            logger.info(f"✅ Model loaded successfully: {model_id}")
            logger.info(f"   Name: {metadata.model_name} {metadata.version}")
            logger.info(f"   Parameters: {metadata.total_parameters:,}")
            logger.info(f"   Best Loss: {metadata.best_loss:.4f}")
            
            return model, tokenizer, metadata
            
        except Exception as e:
            logger.error(f"❌ Failed to load model {model_id}: {e}")
            raise
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all available models with their metadata."""
        models = []
        
        try:
            metadata_dir = self.models_dir / "metadata"
            if not metadata_dir.exists():
                return models
            
            for metadata_file in metadata_dir.glob("*.json"):
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata_dict = json.load(f)
                    
                    model_info = {
                        'id': metadata_file.stem,
                        'name': metadata_dict['model_name'],
                        'version': metadata_dict['version'],
                        'created_at': metadata_dict['created_at'],
                        'best_loss': metadata_dict['best_loss'],
                        'best_perplexity': metadata_dict['best_perplexity'],
                        'size_mb': metadata_dict['model_size_mb'],
                        'parameters': metadata_dict['total_parameters'],
                        'epochs': metadata_dict['epochs_trained'],
                        'training_hours': metadata_dict['training_time_hours'],
                        'tags': metadata_dict.get('tags', []),
                        'hardware': metadata_dict['hardware_used'],
                        'notes': metadata_dict.get('notes', '')
                    }
                    
                    models.append(model_info)
                    
                except Exception as e:
                    logger.warning(f"Could not load metadata for {metadata_file}: {e}")
                    continue
            
            # Sort by best loss (ascending)
            models.sort(key=lambda x: x['best_loss'])
            
        except Exception as e:
            logger.error(f"Error listing models: {e}")
        
        return models
    
    def delete_model(self, model_id: str) -> bool:
        """Delete a model and all its associated files."""
        try:
            model_dir = self.models_dir / "checkpoints" / model_id
            metadata_path = self.models_dir / "metadata" / f"{model_id}.json"
            
            # Remove model directory
            if model_dir.exists():
                shutil.rmtree(model_dir)
            
            # Remove metadata
            if metadata_path.exists():
                metadata_path.unlink()
            
            # Update latest link if this was the latest
            latest_link = self.models_dir / "latest_best"
            if latest_link.exists():
                try:
                    if latest_link.is_symlink():
                        current_latest = latest_link.readlink().name
                    else:
                        with open(latest_link, 'r') as f:
                            current_latest = f.read().strip()
                    
                    if current_latest == model_id:
                        latest_link.unlink()
                        # Find new best model
                        remaining_models = self.list_models()
                        if remaining_models:
                            best_model = min(remaining_models, key=lambda x: x['best_loss'])
                            try:
                                latest_link.symlink_to(best_model['id'])
                            except OSError:
                                with open(latest_link, 'w') as f:
                                    f.write(best_model['id'])
                except Exception:
                    pass
            
            logger.info(f"✅ Model deleted: {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to delete model {model_id}: {e}")
            return False
    
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific model."""
        try:
            metadata_path = self.models_dir / "metadata" / f"{model_id}.json"
            if not metadata_path.exists():
                return None
            
            with open(metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
                
        except Exception as e:
            logger.error(f"Error getting model info for {model_id}: {e}")
            return None
    
    def cleanup_old_models(self, keep_best: int = 5) -> int:
        """Clean up old models, keeping only the best N models."""
        try:
            models = self.list_models()
            if len(models) <= keep_best:
                return 0
            
            # Sort by loss and keep only the best
            models_to_delete = models[keep_best:]
            deleted_count = 0
            
            for model in models_to_delete:
                if self.delete_model(model['id']):
                    deleted_count += 1
            
            logger.info(f"🧹 Cleaned up {deleted_count} old models, kept best {keep_best}")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return 0
    
    def export_model(self, model_id: str, export_path: str) -> bool:
        """Export model to a specific path for sharing or deployment."""
        try:
            model_dir = self.models_dir / "checkpoints" / model_id
            metadata_path = self.models_dir / "metadata" / f"{model_id}.json"
            
            if not model_dir.exists() or not metadata_path.exists():
                raise FileNotFoundError(f"Model {model_id} not found")
            
            export_path = Path(export_path)
            export_path.mkdir(parents=True, exist_ok=True)
            
            # Copy model files
            shutil.copytree(model_dir, export_path / "model", dirs_exist_ok=True)
            shutil.copy2(metadata_path, export_path / "metadata.json")
            
            # Create export info
            export_info = {
                'model_id': model_id,
                'exported_at': datetime.now().isoformat(),
                'export_version': '1.0'
            }
            
            with open(export_path / "export_info.json", 'w') as f:
                json.dump(export_info, f, indent=2)
            
            logger.info(f"✅ Model exported to: {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to export model {model_id}: {e}")
            return False