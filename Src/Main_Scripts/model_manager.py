# Fixed Model Manager - Corrected Saving System
# Copyright (c) 2025 Matias Nielsen. All rights reserved.

import os
import json
import hashlib
import pickle
import shutil
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

# HuggingFace imports with fallback
try:
    import safetensors.torch as st
    from transformers import PreTrainedModel, PreTrainedTokenizer, AutoConfig, PretrainedConfig
    from transformers.modeling_outputs import CausalLMOutputWithPast
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    PreTrainedModel = object
    PreTrainedTokenizer = object
    PretrainedConfig = object
    logging.warning("HuggingFace transformers/safetensors not available. LM Studio compatibility disabled.")

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for the transformer model."""
    vocab_size: int = 32000
    hidden_size: int = 1024
    num_layers: int = 12
    num_heads: int = 16
    seq_length: int = 1024
    dropout: float = 0.1
    model_type: str = "SubwordTransformer"
    tokenizer_type: str = "subword"

@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    batch_size: int = 32
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
    """Fixed model management system with proper saving/loading."""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.models_dir / "checkpoints").mkdir(exist_ok=True)
        (self.models_dir / "metadata").mkdir(exist_ok=True)
        (self.models_dir / "tokenizers").mkdir(exist_ok=True)
        if HF_AVAILABLE:
            (self.models_dir / "lm_studio").mkdir(exist_ok=True)
        
        logger.info(f"ModelManager initialized with directory: {self.models_dir}")
        if HF_AVAILABLE:
            logger.info("✅ LM Studio compatibility enabled")
        else:
            logger.warning("⚠️ LM Studio compatibility disabled")
    
    def _calculate_hash(self, obj: Any) -> str:
        """Calculate SHA256 hash of an object."""
        try:
            if hasattr(obj, 'state_dict'):
                # For PyTorch models
                state_bytes = pickle.dumps({k: v.cpu() for k, v in obj.state_dict().items()})
            elif hasattr(obj, 'vocab') and hasattr(obj, 'merges'):
                # For tokenizers
                state_bytes = pickle.dumps({
                    'vocab': obj.vocab,
                    'merges': obj.merges,
                    'vocab_size': getattr(obj, 'vocab_size', lambda: len(obj.vocab))()
                })
            else:
                state_bytes = pickle.dumps(obj)
            return hashlib.sha256(state_bytes).hexdigest()[:16]
        except Exception as e:
            logger.warning(f"Could not calculate hash: {e}")
            return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def _get_model_size(self, model_path: Path) -> float:
        """Calculate model file size in MB."""
        try:
            if model_path.exists():
                return model_path.stat().st_size / (1024 * 1024)
            return 0.0
        except Exception:
            return 0.0
    
    def _generate_model_id(self, metadata: ModelMetadata) -> str:
        """Generate unique model ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name_clean = metadata.model_name.replace(" ", "_").lower()
        name_clean = "".join(c for c in name_clean if c.isalnum() or c == "_")
        version_clean = metadata.version.replace(".", "_").replace(" ", "_").lower()
        version_clean = "".join(c for c in version_clean if c.isalnum() or c == "_")
        return f"{name_clean}_{version_clean}_{timestamp}"
    
    def _save_tokenizer_safely(self, tokenizer, tokenizer_dir: Path) -> bool:
        """Save tokenizer with comprehensive error handling."""
        try:
            tokenizer_dir.mkdir(exist_ok=True)
            
            # Method 1: Save as pickle (most reliable)
            tokenizer_pkl_path = tokenizer_dir / "tokenizer.pkl"
            with open(tokenizer_pkl_path, 'wb') as f:
                pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"✅ Tokenizer saved as pickle: {tokenizer_pkl_path}")
            
            # Method 2: Save vocabulary and merges separately
            if hasattr(tokenizer, 'vocab') and hasattr(tokenizer, 'merges'):
                vocab_path = tokenizer_dir / "vocab.json"
                merges_path = tokenizer_dir / "merges.txt"
                
                # Save vocabulary
                with open(vocab_path, 'w', encoding='utf-8') as f:
                    json.dump(tokenizer.vocab, f, indent=2, ensure_ascii=False)
                
                # Save merges
                with open(merges_path, 'w', encoding='utf-8') as f:
                    for merge in tokenizer.merges:
                        if isinstance(merge, (list, tuple)) and len(merge) == 2:
                            f.write(f"{merge[0]} {merge[1]}\n")
                
                logger.info(f"✅ Tokenizer vocab saved: {vocab_path}")
                logger.info(f"✅ Tokenizer merges saved: {merges_path}")
            
            # Method 3: Save tokenizer metadata
            tokenizer_info = {
                'type': type(tokenizer).__name__,
                'vocab_size': getattr(tokenizer, 'vocab_size', lambda: len(getattr(tokenizer, 'vocab', {})))(),
                'special_tokens': {
                    'pad_token': getattr(tokenizer, 'vocab', {}).get('<pad>', 0),
                    'unk_token': getattr(tokenizer, 'vocab', {}).get('<unk>', 1),
                    'bos_token': getattr(tokenizer, 'vocab', {}).get('<s>', 2),
                    'eos_token': getattr(tokenizer, 'vocab', {}).get('</s>', 3),
                },
                'num_merges': len(getattr(tokenizer, 'merges', [])),
                'saved_at': datetime.now().isoformat()
            }
            
            tokenizer_info_path = tokenizer_dir / "tokenizer_info.json"
            with open(tokenizer_info_path, 'w', encoding='utf-8') as f:
                json.dump(tokenizer_info, f, indent=2)
            
            logger.info(f"✅ Tokenizer info saved: {tokenizer_info_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to save tokenizer: {e}")
            return False
    
    def _load_tokenizer_safely(self, tokenizer_dir: Path):
        """Load tokenizer with multiple fallback methods."""
        if not tokenizer_dir.exists():
            raise FileNotFoundError(f"Tokenizer directory not found: {tokenizer_dir}")
        
        # Method 1: Try loading pickle first
        tokenizer_pkl_path = tokenizer_dir / "tokenizer.pkl"
        if tokenizer_pkl_path.exists():
            try:
                with open(tokenizer_pkl_path, 'rb') as f:
                    tokenizer = pickle.load(f)
                logger.info(f"✅ Tokenizer loaded from pickle: {tokenizer_pkl_path}")
                return tokenizer
            except Exception as e:
                logger.warning(f"Failed to load tokenizer pickle: {e}")
        
        # Method 2: Reconstruct from vocab and merges
        vocab_path = tokenizer_dir / "vocab.json"
        merges_path = tokenizer_dir / "merges.txt"
        
        if vocab_path.exists() and merges_path.exists():
            try:
                # Import here to avoid circular imports
                from subword_transformer import SubwordTokenizer
                
                # Load vocabulary
                with open(vocab_path, 'r', encoding='utf-8') as f:
                    vocab = json.load(f)
                
                # Load merges
                merges = []
                with open(merges_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and ' ' in line:
                            parts = line.split(' ', 1)
                            if len(parts) == 2:
                                merges.append((parts[0], parts[1]))
                
                # Create tokenizer
                tokenizer = SubwordTokenizer(vocab=vocab, merges=merges)
                logger.info(f"✅ Tokenizer reconstructed from vocab/merges")
                return tokenizer
                
            except Exception as e:
                logger.error(f"Failed to reconstruct tokenizer: {e}")
        
        # Method 3: Check if there's a tokenizer info file for debugging
        tokenizer_info_path = tokenizer_dir / "tokenizer_info.json"
        if tokenizer_info_path.exists():
            try:
                with open(tokenizer_info_path, 'r', encoding='utf-8') as f:
                    info = json.load(f)
                logger.info(f"Tokenizer info available: {info}")
            except:
                pass
        
        raise FileNotFoundError(f"Could not load tokenizer from {tokenizer_dir}")
    
    def save_model(self, model: nn.Module, tokenizer, metadata: ModelMetadata,
                   optimizer=None, scheduler=None, force_cpu_save: bool = True) -> str:
        """Save model with comprehensive error handling and validation."""
        try:
            logger.info("💾 Starting model save process...")
            
            # Generate model ID
            model_id = self._generate_model_id(metadata)
            logger.info(f"Generated model ID: {model_id}")
            
            # Create model directory
            model_dir = self.models_dir / "checkpoints" / model_id
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Move model to CPU for saving if requested (more stable)
            original_device = next(model.parameters()).device
            if force_cpu_save and original_device.type != 'cpu':
                logger.info("Moving model to CPU for saving...")
                model = model.cpu()
            
            # Save model state dict
            model_path = model_dir / "model.pt"
            model_save_data = {
                'model_state_dict': model.state_dict(),
                'model_config': asdict(metadata.model_config) if hasattr(metadata.model_config, '__dict__') else metadata.model_config,
                'model_id': model_id,
                'model_type': metadata.model_config.model_type,
                'vocab_size': metadata.model_config.vocab_size,
                'pytorch_version': torch.__version__,
                'saved_at': datetime.now().isoformat()
            }
            
            torch.save(model_save_data, model_path)
            logger.info(f"✅ Model state saved: {model_path}")
            
            # Move model back to original device
            if force_cpu_save and original_device.type != 'cpu':
                model = model.to(original_device)
            
            # Save tokenizer
            tokenizer_dir = model_dir / "tokenizer"
            if not self._save_tokenizer_safely(tokenizer, tokenizer_dir):
                logger.error("Failed to save tokenizer - model save incomplete")
                # Don't fail completely, but warn
            
            # Save optimizer state if provided
            if optimizer is not None:
                try:
                    optimizer_path = model_dir / "optimizer.pt"
                    optimizer_state = optimizer.state_dict()
                    torch.save(optimizer_state, optimizer_path)
                    logger.info(f"✅ Optimizer state saved: {optimizer_path}")
                except Exception as e:
                    logger.warning(f"Failed to save optimizer: {e}")
            
            # Save scheduler state if provided
            if scheduler is not None:
                try:
                    scheduler_path = model_dir / "scheduler.pt"
                    if hasattr(scheduler, 'state_dict'):
                        scheduler_state = scheduler.state_dict()
                    else:
                        # Custom scheduler - save what we can
                        scheduler_state = {
                            'current_step': getattr(scheduler, 'current_step', 0),
                            'warmup_steps': getattr(scheduler, 'warmup_steps', 0),
                            'total_steps': getattr(scheduler, 'total_steps', 0)
                        }
                    torch.save(scheduler_state, scheduler_path)
                    logger.info(f"✅ Scheduler state saved: {scheduler_path}")
                except Exception as e:
                    logger.warning(f"Failed to save scheduler: {e}")
            
            # Calculate and update metadata
            metadata.model_hash = self._calculate_hash(model)
            metadata.tokenizer_hash = self._calculate_hash(tokenizer)
            metadata.model_size_mb = self._get_model_size(model_path)
            metadata.last_modified = datetime.now().isoformat()
            
            # Save comprehensive metadata
            metadata_path = self.models_dir / "metadata" / f"{model_id}.json"
            metadata_dict = asdict(metadata)
            
            # Convert non-serializable objects
            if isinstance(metadata_dict.get('model_config'), object):
                metadata_dict['model_config'] = asdict(metadata.model_config)
            if isinstance(metadata_dict.get('training_config'), object):
                metadata_dict['training_config'] = asdict(metadata.training_config)
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata_dict, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"✅ Metadata saved: {metadata_path}")
            
            # Create model summary
            summary_path = model_dir / "model_summary.txt"
            summary_content = f"""Model Summary: {model_id}
{'='*50}

Basic Information:
- Name: {metadata.model_name}
- Version: {metadata.version}
- Type: {metadata.model_config.model_type}
- Created: {metadata.created_at}
- Model ID: {model_id}

Architecture:
- Vocabulary Size: {metadata.model_config.vocab_size:,}
- Hidden Size: {metadata.model_config.hidden_size}
- Layers: {metadata.model_config.num_layers}
- Attention Heads: {metadata.model_config.num_heads}
- Sequence Length: {metadata.model_config.seq_length}
- Dropout: {metadata.model_config.dropout}

Model Statistics:
- Total Parameters: {metadata.total_parameters:,}
- Trainable Parameters: {metadata.trainable_parameters:,}
- Model Size: {metadata.model_size_mb:.2f} MB
- Best Loss: {metadata.best_loss:.4f}
- Best Perplexity: {metadata.best_perplexity:.2f}

Training Information:
- Epochs Trained: {metadata.epochs_trained}
- Training Time: {metadata.training_time_hours:.2f} hours
- Hardware: {metadata.hardware_used}
- PyTorch Version: {metadata.pytorch_version}

Dataset:
- Source: {metadata.dataset_info.get('source', 'Unknown')}
- Samples: {metadata.dataset_info.get('num_samples', 'Unknown')}

Files:
- Model: model.pt
- Tokenizer: tokenizer/
- Metadata: ../metadata/{model_id}.json
- Optimizer: {'optimizer.pt (saved)' if optimizer else 'optimizer.pt (not saved)'}
- Scheduler: {'scheduler.pt (saved)' if scheduler else 'scheduler.pt (not saved)'}

Notes:
{metadata.notes}

Tags: {', '.join(metadata.tags)}
"""
            
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(summary_content)
            
            logger.info(f"✅ Model summary saved: {summary_path}")
            
            # Handle "best" model linking
            if "best" in metadata.tags:
                self._update_best_model_link(model_id)
            
            # Success message
            logger.info("=" * 50)
            logger.info(f"✅ Model saved successfully!")
            logger.info(f"   Model ID: {model_id}")
            logger.info(f"   Location: {model_dir}")
            logger.info(f"   Size: {metadata.model_size_mb:.2f} MB")
            logger.info(f"   Parameters: {metadata.total_parameters:,}")
            logger.info(f"   Loss: {metadata.best_loss:.4f}")
            logger.info(f"   Perplexity: {metadata.best_perplexity:.2f}")
            logger.info("=" * 50)
            
            return model_id
            
        except Exception as e:
            logger.error(f"❌ Failed to save model: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Cleanup on failure
            if 'model_dir' in locals() and model_dir.exists():
                try:
                    shutil.rmtree(model_dir)
                    logger.info("Cleaned up incomplete model directory")
                except:
                    pass
            
            raise
    
    def _update_best_model_link(self, model_id: str):
        """Update symlink/reference to best model."""
        try:
            latest_link = self.models_dir / "latest_best"
            
            # Remove existing link
            if latest_link.exists():
                if latest_link.is_symlink():
                    latest_link.unlink()
                else:
                    latest_link.unlink()
            
            # Create new link
            try:
                # Try symlink first
                latest_link.symlink_to(f"checkpoints/{model_id}")
                logger.info(f"✅ Best model symlink updated: {latest_link}")
            except OSError:
                # Fallback to text file
                with open(latest_link, 'w') as f:
                    f.write(model_id)
                logger.info(f"✅ Best model reference updated: {latest_link}")
                
        except Exception as e:
            logger.warning(f"Failed to update best model link: {e}")
    
    def load_model(self, model_id: str, device: Optional[torch.device] = None):
        """Load model with comprehensive error handling."""
        try:
            logger.info(f"🔄 Loading model: {model_id}")
            
            # Handle special model IDs
            if model_id in ['latest', 'best', 'latest_best']:
                model_id = self._resolve_special_model_id(model_id)
            
            # Check if model exists
            model_dir = self.models_dir / "checkpoints" / model_id
            if not model_dir.exists():
                available_models = [d.name for d in (self.models_dir / "checkpoints").iterdir() if d.is_dir()]
                raise FileNotFoundError(f"Model '{model_id}' not found. Available models: {available_models}")
            
            # Load metadata
            metadata_path = self.models_dir / "metadata" / f"{model_id}.json"
            if not metadata_path.exists():
                raise FileNotFoundError(f"Metadata not found: {metadata_path}")
            
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata_dict = json.load(f)
            
            # Reconstruct metadata object
            model_config = ModelConfig(**metadata_dict['model_config'])
            training_config = TrainingConfig(**metadata_dict['training_config'])
            
            metadata_dict['model_config'] = model_config
            metadata_dict['training_config'] = training_config
            metadata = ModelMetadata(**metadata_dict)
            
            logger.info(f"✅ Metadata loaded: {metadata.model_name} {metadata.version}")
            
            # Load tokenizer
            tokenizer_dir = model_dir / "tokenizer"
            tokenizer = self._load_tokenizer_safely(tokenizer_dir)
            
            logger.info(f"✅ Tokenizer loaded: vocab_size={tokenizer.vocab_size()}")
            
            # Load model
            model_path = model_dir / "model.pt"
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Import model class
            if metadata.model_config.model_type == "SubwordTransformer":
                from subword_transformer import SubwordTransformer
                model = SubwordTransformer(model_config)
            else:
                raise ValueError(f"Unknown model type: {metadata.model_config.model_type}")
            
            # Load checkpoint
            checkpoint_device = 'cpu' if device is None else device
            checkpoint = torch.load(model_path, map_location=checkpoint_device)
            
            # Load state dict
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Load with error handling
            try:
                missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=True)
                if missing_keys:
                    logger.warning(f"Missing keys in checkpoint: {missing_keys}")
                if unexpected_keys:
                    logger.warning(f"Unexpected keys in checkpoint: {unexpected_keys}")
            except Exception as e:
                logger.error(f"Failed to load state dict: {e}")
                raise
            
            # Move to device if specified
            if device is not None:
                model = model.to(device)
            
            logger.info("=" * 50)
            logger.info(f"✅ Model loaded successfully!")
            logger.info(f"   Model ID: {model_id}")
            logger.info(f"   Name: {metadata.model_name} {metadata.version}")
            logger.info(f"   Parameters: {metadata.total_parameters:,}")
            logger.info(f"   Vocab Size: {model_config.vocab_size:,}")
            logger.info(f"   Best Loss: {metadata.best_loss:.4f}")
            logger.info(f"   Device: {next(model.parameters()).device}")
            logger.info("=" * 50)
            
            return model, tokenizer, metadata
            
        except Exception as e:
            logger.error(f"❌ Failed to load model '{model_id}': {e}")
            raise
    
    def _resolve_special_model_id(self, special_id: str) -> str:
        """Resolve special model IDs like 'latest', 'best'."""
        latest_link = self.models_dir / "latest_best"
        
        if latest_link.exists():
            try:
                if latest_link.is_symlink():
                    # Resolve symlink
                    target = latest_link.readlink()
                    return target.name if target.name != "checkpoints" else target.parent.name
                else:
                    # Read from text file
                    with open(latest_link, 'r') as f:
                        return f.read().strip()
            except Exception as e:
                logger.warning(f"Failed to resolve {special_id}: {e}")
        
        # Fallback: find best model by loss
        models = self.list_models()
        if not models:
            raise FileNotFoundError("No models available")
        
        best_model = min(models, key=lambda x: x['best_loss'])
        return best_model['id']
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all available models."""
        models = []
        
        try:
            metadata_dir = self.models_dir / "metadata"
            if not metadata_dir.exists():
                return models
            
            for metadata_file in metadata_dir.glob("*.json"):
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata_dict = json.load(f)
                    
                    model_id = metadata_file.stem
                    
                    model_info = {
                        'id': model_id,
                        'name': metadata_dict.get('model_name', 'Unknown'),
                        'version': metadata_dict.get('version', 'Unknown'),
                        'created_at': metadata_dict.get('created_at', 'Unknown'),
                        'best_loss': metadata_dict.get('best_loss', float('inf')),
                        'best_perplexity': metadata_dict.get('best_perplexity', float('inf')),
                        'size_mb': metadata_dict.get('model_size_mb', 0.0),
                        'parameters': metadata_dict.get('total_parameters', 0),
                        'epochs': metadata_dict.get('epochs_trained', 0),
                        'training_hours': metadata_dict.get('training_time_hours', 0.0),
                        'tags': metadata_dict.get('tags', []),
                        'hardware': metadata_dict.get('hardware_used', 'Unknown'),
                        'notes': metadata_dict.get('notes', ''),
                        'tokenizer_type': metadata_dict.get('model_config', {}).get('tokenizer_type', 'unknown'),
                        'model_type': metadata_dict.get('model_config', {}).get('model_type', 'Unknown')
                    }
                    
                    models.append(model_info)
                    
                except Exception as e:
                    logger.warning(f"Could not load metadata for {metadata_file}: {e}")
                    continue
            
            # Sort by best loss
            models.sort(key=lambda x: x['best_loss'])
            
        except Exception as e:
            logger.error(f"Error listing models: {e}")
        
        return models
    
    def validate_model(self, model_id: str) -> Dict[str, Any]:
        """Validate a saved model and return status."""
        try:
            model_dir = self.models_dir / "checkpoints" / model_id
            metadata_path = self.models_dir / "metadata" / f"{model_id}.json"
            
            validation_result = {
                'model_id': model_id,
                'valid': True,
                'issues': [],
                'files': {}
            }
            
            # Check model directory
            if not model_dir.exists():
                validation_result['valid'] = False
                validation_result['issues'].append("Model directory missing")
                return validation_result
            
            # Check metadata
            if metadata_path.exists():
                validation_result['files']['metadata'] = "✅ Present"
            else:
                validation_result['valid'] = False
                validation_result['files']['metadata'] = "❌ Missing"
                validation_result['issues'].append("Metadata file missing")
            
            # Check model file
            model_path = model_dir / "model.pt"
            if model_path.exists():
                validation_result['files']['model'] = f"✅ Present ({model_path.stat().st_size / 1024**2:.1f} MB)"
            else:
                validation_result['valid'] = False
                validation_result['files']['model'] = "❌ Missing"
                validation_result['issues'].append("Model file missing")
            
            # Check tokenizer
            tokenizer_dir = model_dir / "tokenizer"
            if tokenizer_dir.exists():
                tokenizer_files = list(tokenizer_dir.iterdir())
                validation_result['files']['tokenizer'] = f"✅ Present ({len(tokenizer_files)} files)"
            else:
                validation_result['valid'] = False
                validation_result['files']['tokenizer'] = "❌ Missing"
                validation_result['issues'].append("Tokenizer directory missing")
            
            # Check optional files
            optional_files = ['optimizer.pt', 'scheduler.pt', 'model_summary.txt']
            for optional_file in optional_files:
                file_path = model_dir / optional_file
                if file_path.exists():
                    validation_result['files'][optional_file] = "✅ Present"
                else:
                    validation_result['files'][optional_file] = "⚠️ Missing (optional)"
            
            return validation_result
            
        except Exception as e:
            return {
                'model_id': model_id,
                'valid': False,
                'issues': [f"Validation error: {e}"],
                'files': {}
            }
    
    def delete_model(self, model_id: str) -> bool:
        """Delete a model and all associated files."""
        try:
            logger.info(f"🗑️ Deleting model: {model_id}")
            
            model_dir = self.models_dir / "checkpoints" / model_id
            metadata_path = self.models_dir / "metadata" / f"{model_id}.json"
            
            deleted_items = []
            
            # Remove model directory
            if model_dir.exists():
                shutil.rmtree(model_dir)
                deleted_items.append("model directory")
                logger.info(f"   Removed model directory: {model_dir}")
            
            # Remove metadata
            if metadata_path.exists():
                metadata_path.unlink()
                deleted_items.append("metadata")
                logger.info(f"   Removed metadata: {metadata_path}")
            
            # Update latest links if this was the latest model
            self._cleanup_best_model_links(model_id)
            
            if deleted_items:
                logger.info(f"✅ Model '{model_id}' deleted successfully (removed: {', '.join(deleted_items)})")
                return True
            else:
                logger.warning(f"⚠️ No files found for model '{model_id}'")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to delete model '{model_id}': {e}")
            return False
    
    def _cleanup_best_model_links(self, deleted_model_id: str):
        """Clean up best model links if the deleted model was the best."""
        try:
            latest_link = self.models_dir / "latest_best"
            
            if latest_link.exists():
                # Check if the deleted model was the current best
                current_best_id = None
                try:
                    if latest_link.is_symlink():
                        target = latest_link.readlink()
                        current_best_id = target.name if target.name != "checkpoints" else target.parent.name
                    else:
                        with open(latest_link, 'r') as f:
                            current_best_id = f.read().strip()
                except:
                    pass
                
                if current_best_id == deleted_model_id:
                    # Remove the old link
                    latest_link.unlink()
                    logger.info("   Removed outdated best model link")
                    
                    # Find new best model
                    remaining_models = self.list_models()
                    if remaining_models:
                        new_best = min(remaining_models, key=lambda x: x['best_loss'])
                        self._update_best_model_link(new_best['id'])
                        logger.info(f"   Updated best model link to: {new_best['id']}")
        except Exception as e:
            logger.warning(f"Failed to cleanup best model links: {e}")
    
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific model."""
        try:
            metadata_path = self.models_dir / "metadata" / f"{model_id}.json"
            if not metadata_path.exists():
                return None
            
            with open(metadata_path, 'r', encoding='utf-8') as f:
                model_info = json.load(f)
            
            # Add validation info
            validation = self.validate_model(model_id)
            model_info['validation'] = validation
            
            # Add file paths
            model_dir = self.models_dir / "checkpoints" / model_id
            model_info['paths'] = {
                'model_dir': str(model_dir),
                'metadata': str(metadata_path),
                'model_file': str(model_dir / "model.pt"),
                'tokenizer_dir': str(model_dir / "tokenizer"),
                'summary': str(model_dir / "model_summary.txt")
            }
            
            return model_info
                
        except Exception as e:
            logger.error(f"Error getting model info for {model_id}: {e}")
            return None
    
    def export_model(self, model_id: str, export_path: str) -> bool:
        """Export model for sharing or deployment."""
        try:
            logger.info(f"📦 Exporting model: {model_id}")
            
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
                'export_version': '1.0',
                'contents': {
                    'model': 'Complete model checkpoint',
                    'metadata': 'Model metadata and configuration',
                    'tokenizer': 'Tokenizer files and vocabulary',
                    'summary': 'Human-readable model summary'
                }
            }
            
            with open(export_path / "export_info.json", 'w') as f:
                json.dump(export_info, f, indent=2)
            
            # Create README
            readme_content = f"""# Exported Model: {model_id}

This export contains a complete model checkpoint with all necessary files.

## Contents

- `model/` - Complete model checkpoint directory
  - `model.pt` - PyTorch model state dict
  - `tokenizer/` - Tokenizer files
  - `model_summary.txt` - Human-readable summary
  - Additional training files (optimizer, scheduler if available)

- `metadata.json` - Complete model metadata
- `export_info.json` - Export information
- `README.md` - This file

## Loading the Model

```python
from model_manager import ModelManager

# Create model manager
manager = ModelManager()

# Method 1: Copy to models directory and load
# (Copy the 'model' folder to your models/checkpoints/ directory)
model, tokenizer, metadata = manager.load_model('{model_id}')

# Method 2: Load directly from export
# (Requires manual setup - see documentation)
```

## Export Information

- **Exported At**: {export_info['exported_at']}
- **Export Version**: {export_info['export_version']}
- **Original Model ID**: {model_id}

## Notes

This is a complete model export containing all files necessary for loading and using the model.
The model was trained using the subword transformer architecture with BPE tokenization.
"""
            
            with open(export_path / "README.md", "w") as f:
                f.write(readme_content)
            
            logger.info(f"✅ Model exported successfully to: {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to export model {model_id}: {e}")
            return False
    
    def cleanup_old_models(self, keep_best: int = 5) -> int:
        """Clean up old models, keeping only the best N models."""
        try:
            logger.info(f"🧹 Cleaning up old models (keeping best {keep_best})...")
            
            models = self.list_models()
            if len(models) <= keep_best:
                logger.info(f"Only {len(models)} models found, no cleanup needed")
                return 0
            
            # Sort by loss and keep only the best
            models_to_delete = models[keep_best:]
            deleted_count = 0
            
            for model in models_to_delete:
                try:
                    if self.delete_model(model['id']):
                        deleted_count += 1
                        logger.info(f"   Deleted: {model['name']} (loss: {model['best_loss']:.4f})")
                except Exception as e:
                    logger.warning(f"   Failed to delete {model['id']}: {e}")
                    continue
            
            logger.info(f"✅ Cleanup complete: removed {deleted_count} models, kept best {keep_best}")
            return deleted_count
            
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")
            return 0
    
    def print_model_summary(self):
        """Print a comprehensive summary of all models."""
        print("\n" + "="*80)
        print("🤖 MODEL MANAGER SUMMARY")
        print("="*80)
        
        models = self.list_models()
        
        print(f"📊 Total Models: {len(models)}")
        print(f"💾 Storage Location: {self.models_dir}")
        
        if models:
            print(f"\n🏆 BEST MODEL (by loss):")
            best = models[0]
            print(f"   ID: {best['id']}")
            print(f"   Name: {best['name']} {best['version']}")
            print(f"   Loss: {best['best_loss']:.4f} (PPL: {best['best_perplexity']:.2f})")
            print(f"   Parameters: {best['parameters']:,}")
            print(f"   Size: {best['size_mb']:.1f} MB")
            print(f"   Type: {best['model_type']} ({best['tokenizer_type']})")
            print(f"   Training: {best['epochs']} epochs, {best['training_hours']:.1f}h")
            
            print(f"\n📋 ALL MODELS:")
            for i, model in enumerate(models, 1):
                status = "🏆" if i == 1 else "📄"
                type_info = f"({model['tokenizer_type']})"
                print(f"   {i:2d}. {status} {model['name']} {type_info}")
                print(f"       Loss: {model['best_loss']:.4f} | {model['parameters']:,} params | {model['size_mb']:.1f}MB")
        else:
            print("\n⚠️  No models found")
        
        print("="*80)
    
    def repair_model(self, model_id: str) -> bool:
        """Attempt to repair a corrupted model."""
        try:
            logger.info(f"🔧 Attempting to repair model: {model_id}")
            
            validation = self.validate_model(model_id)
            if validation['valid']:
                logger.info("✅ Model appears to be valid, no repair needed")
                return True
            
            logger.info(f"Issues found: {validation['issues']}")
            
            model_dir = self.models_dir / "checkpoints" / model_id
            metadata_path = self.models_dir / "metadata" / f"{model_id}.json"
            
            repairs_made = []
            
            # Attempt to regenerate missing metadata
            if not metadata_path.exists() and model_dir.exists():
                model_path = model_dir / "model.pt"
                if model_path.exists():
                    try:
                        # Load model checkpoint to extract info
                        checkpoint = torch.load(model_path, map_location='cpu')
                        
                        # Create minimal metadata
                        from datetime import datetime
                        minimal_metadata = {
                            'model_name': f'Repaired_Model_{model_id}',
                            'version': 'v1.0_repaired',
                            'created_at': datetime.now().isoformat(),
                            'last_modified': datetime.now().isoformat(),
                            'model_config': checkpoint.get('model_config', {}),
                            'training_config': {
                                'learning_rate': 3e-4,
                                'batch_size': 32,
                                'max_epochs': 50
                            },
                            'dataset_info': {'source': 'Unknown (repaired)'},
                            'performance_metrics': {},
                            'model_size_mb': model_path.stat().st_size / 1024**2,
                            'total_parameters': 0,
                            'trainable_parameters': 0,
                            'training_time_hours': 0.0,
                            'epochs_trained': 0,
                            'best_loss': float('inf'),
                            'best_perplexity': float('inf'),
                            'hardware_used': 'Unknown',
                            'pytorch_version': torch.__version__,
                            'cuda_version': None,
                            'model_hash': 'repaired',
                            'tokenizer_hash': 'repaired',
                            'notes': f'Repaired model from checkpoint. Original metadata was missing.',
                            'tags': ['repaired']
                        }
                        
                        with open(metadata_path, 'w') as f:
                            json.dump(minimal_metadata, f, indent=2, default=str)
                        
                        repairs_made.append("regenerated metadata")
                        logger.info("   ✅ Regenerated minimal metadata")
                        
                    except Exception as e:
                        logger.error(f"   ❌ Failed to regenerate metadata: {e}")
            
            # Create summary if missing
            summary_path = model_dir / "model_summary.txt"
            if not summary_path.exists() and metadata_path.exists():
                try:
                    summary_content = f"""Model Summary: {model_id} (REPAIRED)
{'='*50}

⚠️  WARNING: This model was repaired and may have incomplete information.

Model ID: {model_id}
Status: Repaired
Repairs Made: {', '.join(repairs_made) if repairs_made else 'None'}

Files Present:
{chr(10).join(f"- {f.name}" for f in model_dir.iterdir())}

⚠️  Please verify this model works correctly before using in production.
"""
                    with open(summary_path, 'w') as f:
                        f.write(summary_content)
                    
                    repairs_made.append("created summary")
                    logger.info("   ✅ Created model summary")
                    
                except Exception as e:
                    logger.error(f"   ❌ Failed to create summary: {e}")
            
            if repairs_made:
                logger.info(f"✅ Model repair completed. Repairs made: {', '.join(repairs_made)}")
                return True
            else:
                logger.error("❌ No repairs could be made")
                return False
                
        except Exception as e:
            logger.error(f"❌ Model repair failed: {e}")
            return False