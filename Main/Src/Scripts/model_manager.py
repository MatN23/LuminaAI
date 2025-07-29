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
import torch.nn.functional as F

# LM Studio compatibility imports
try:
    import safetensors.torch as st
    from transformers import PreTrainedModel, PreTrainedTokenizer, AutoConfig
    from transformers.modeling_outputs import CausalLMOutputWithPast
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logging.warning("HuggingFace transformers/safetensors not available. LM Studio compatibility disabled.")

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

class HuggingFaceCompatibleModel(PreTrainedModel):
    """Wrapper to make our model compatible with HuggingFace/LM Studio."""
    
    def __init__(self, config, model):
        super().__init__(config)
        self.model = model
        self.config = config
        
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        logits = self.model(input_ids)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )
    
    def generate(self, input_ids, max_length=50, temperature=0.8, top_k=50, top_p=0.9, **kwargs):
        """Generation method for compatibility."""
        self.eval()
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length - input_ids.size(1)):
                if generated.size(1) >= self.config.max_position_embeddings:
                    break
                
                logits = self.model(generated)
                next_token_logits = logits[0, -1, :] / temperature
                
                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
                
                # Stop on EOS token
                if next_token.item() == self.config.eos_token_id:
                    break
        
        return generated

class HuggingFaceCompatibleTokenizer(PreTrainedTokenizer):
    """Wrapper to make our tokenizer compatible with HuggingFace."""
    
    def __init__(self, word_tokenizer, **kwargs):
        self.word_tokenizer = word_tokenizer
        super().__init__(
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
            pad_token="<pad>",
            **kwargs
        )
    
    @property
    def vocab_size(self):
        return self.word_tokenizer.vocab_size()
    
    def get_vocab(self):
        return self.word_tokenizer.vocab
    
    def _tokenize(self, text):
        return self.word_tokenizer.tokenize(text)
    
    def _convert_token_to_id(self, token):
        return self.word_tokenizer.vocab.get(token, self.word_tokenizer.vocab.get("<unk>", 0))
    
    def _convert_id_to_token(self, index):
        return self.word_tokenizer.reverse_vocab.get(index, "<unk>")
    
    def convert_tokens_to_string(self, tokens):
        return " ".join(tokens).replace(" ##", "")
    
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        bos = [self.bos_token_id] if self.bos_token_id is not None else []
        eos = [self.eos_token_id] if self.eos_token_id is not None else []
        
        if token_ids_1 is None:
            return bos + token_ids_0 + eos
        return bos + token_ids_0 + eos + token_ids_1 + eos
    
    def save_vocabulary(self, save_directory, filename_prefix=None):
        """Save vocabulary files."""
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory)
        
        vocab_file = os.path.join(save_directory, "vocab.json")
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self.word_tokenizer.vocab, f, ensure_ascii=False, indent=2)
        
        return (vocab_file,)

class ModelManager:
    """Enhanced model management system with LM Studio compatibility."""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.models_dir / "checkpoints").mkdir(exist_ok=True)
        (self.models_dir / "metadata").mkdir(exist_ok=True)
        (self.models_dir / "tokenizers").mkdir(exist_ok=True)
        (self.models_dir / "lm_studio").mkdir(exist_ok=True)  # New LM Studio directory
        
        logger.info(f"ModelManager initialized with directory: {self.models_dir}")
        if HF_AVAILABLE:
            logger.info("✅ LM Studio compatibility enabled")
        else:
            logger.warning("⚠️ LM Studio compatibility disabled - install transformers and safetensors")
    
    def _calculate_hash(self, obj: Any) -> str:
        """Calculate SHA256 hash of an object."""
        try:
            if hasattr(obj, 'state_dict'):
                state_bytes = pickle.dumps(obj.state_dict())
            else:
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
    
    def _save_for_lm_studio(self, model: nn.Module, tokenizer, metadata: ModelMetadata, 
                           model_id: str) -> bool:
        """Save model in LM Studio compatible format."""
        if not HF_AVAILABLE:
            logger.warning("Skipping LM Studio save - dependencies not available")
            return False
        
        try:
            # Create LM Studio model directory
            lm_studio_dir = self.models_dir / "lm_studio" / model_id
            lm_studio_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"💾 Saving LM Studio compatible model: {model_id}")
            
            # Create HuggingFace compatible config
            hf_config = AutoConfig.from_dict({
                "model_type": "custom_word_transformer",
                "vocab_size": metadata.model_config.vocab_size,
                "hidden_size": metadata.model_config.hidden_size,
                "num_hidden_layers": metadata.model_config.num_layers,
                "num_attention_heads": metadata.model_config.num_heads,
                "max_position_embeddings": metadata.model_config.seq_length,
                "hidden_dropout_prob": metadata.model_config.dropout,
                "attention_probs_dropout_prob": metadata.model_config.dropout,
                "initializer_range": 0.02,
                "layer_norm_eps": 1e-12,
                "pad_token_id": tokenizer.vocab.get("<pad>", 0),
                "bos_token_id": tokenizer.vocab.get("<s>", 1),
                "eos_token_id": tokenizer.vocab.get("</s>", 2),
                "unk_token_id": tokenizer.vocab.get("<unk>", 3),
                "use_cache": True,
                "is_decoder": True,
                "architectures": ["WordTransformerForCausalLM"],
                "torch_dtype": "float32",
                "transformers_version": "4.21.0"
            })
            
            # Save config
            hf_config.save_pretrained(lm_studio_dir)
            
            # Create and save HuggingFace compatible model
            hf_model = HuggingFaceCompatibleModel(hf_config, model)
            
            # Save model weights using safetensors
            try:
                state_dict = {}
                for name, param in hf_model.named_parameters():
                    state_dict[name] = param.detach().cpu()
                
                # Save as safetensors (preferred by LM Studio)
                safetensors_path = lm_studio_dir / "model.safetensors"
                st.save_file(state_dict, safetensors_path)
                logger.info(f"✅ Saved safetensors: {safetensors_path}")
                
                # Also save as regular PyTorch (backup)
                torch.save(state_dict, lm_studio_dir / "pytorch_model.bin")
                
            except Exception as e:
                logger.warning(f"Failed to save as safetensors: {e}. Using PyTorch format.")
                hf_model.save_pretrained(lm_studio_dir, safe_serialization=False)
            
            # Create and save HuggingFace compatible tokenizer
            hf_tokenizer = HuggingFaceCompatibleTokenizer(tokenizer)
            hf_tokenizer.save_pretrained(lm_studio_dir)
            
            # Save comprehensive model card
            model_card = {
                "model_name": metadata.model_name,
                "model_type": "causal-lm",
                "architecture": "WordTransformer",
                "vocab_size": metadata.model_config.vocab_size,
                "context_length": metadata.model_config.seq_length,
                "hidden_size": metadata.model_config.hidden_size,
                "num_layers": metadata.model_config.num_layers,
                "num_heads": metadata.model_config.num_heads,
                "parameters": f"{metadata.total_parameters:,}",
                "tokenizer_type": "word-level",
                "training_data": metadata.dataset_info.get("source", "Unknown"),
                "license": "Custom License",
                "usage": "Text generation and conversation",
                "prompt_format": "<user> {prompt} <bot>",
                "stop_tokens": ["</s>", "<user>"],
                "recommended_settings": {
                    "temperature": 0.8,
                    "top_k": 50,
                    "top_p": 0.9,
                    "repetition_penalty": 1.1,
                    "max_tokens": 512
                },
                "performance": {
                    "loss": metadata.best_loss,
                    "perplexity": metadata.best_perplexity,
                    "training_time_hours": metadata.training_time_hours,
                    "epochs_trained": metadata.epochs_trained
                },
                "technical_details": {
                    "pytorch_version": metadata.pytorch_version,
                    "hardware_used": metadata.hardware_used,
                    "created_at": metadata.created_at,
                    "model_id": model_id
                }
            }
            
            with open(lm_studio_dir / "model_card.json", "w") as f:
                json.dump(model_card, f, indent=2)
            
            # Create README for LM Studio users
            readme_content = f"""# {metadata.model_name}

A word-level transformer model for conversational AI.

## Model Details
- **Architecture**: Word-level Transformer
- **Parameters**: {metadata.total_parameters:,}
- **Context Length**: {metadata.model_config.seq_length}
- **Vocabulary Size**: {metadata.model_config.vocab_size:,}
- **Training Loss**: {metadata.best_loss:.4f}
- **Perplexity**: {metadata.best_perplexity:.2f}

## Usage in LM Studio

### Loading the Model
1. Open LM Studio
2. Click "Load Model" 
3. Navigate to this folder
4. Select and load the model

### Prompt Format
```
<user> Your question or message here <bot>
```

### Recommended Settings
- **Temperature**: 0.8
- **Top-K**: 50
- **Top-P**: 0.9  
- **Repetition Penalty**: 1.1
- **Max Tokens**: 512
- **Stop Tokens**: `</s>`, `<user>`

## Example Conversation
```
<user> Hello! How are you today? <bot>
Hello! I'm doing well, thank you for asking. How can I help you today?

<user> Can you explain what artificial intelligence is? <bot>
Artificial intelligence (AI) refers to computer systems that can perform tasks typically requiring human intelligence...
```

## Training Details
- **Dataset**: {metadata.dataset_info.get('source', 'Custom dataset')}
- **Training Time**: {metadata.training_time_hours:.2f} hours
- **Epochs**: {metadata.epochs_trained}
- **Hardware**: {metadata.hardware_used}
- **Created**: {metadata.created_at}

## Notes
{metadata.notes}

## License
Custom License - See original training repository for details.
"""
            
            with open(lm_studio_dir / "README.md", "w") as f:
                f.write(readme_content)
            
            # Create GGUF conversion script
            conversion_script = f"""#!/bin/bash
# Convert {metadata.model_name} to GGUF format for optimized inference

echo "Converting {metadata.model_name} to GGUF format..."

# Requires llama.cpp tools to be installed and in PATH
# Download from: https://github.com/ggerganov/llama.cpp

# Convert to GGUF
if command -v convert_hf_to_gguf.py &> /dev/null; then
    python convert_hf_to_gguf.py . --outfile {model_id}.gguf --outtype f16
    echo "✅ Base GGUF created: {model_id}.gguf"
    
    # Create quantized versions
    if command -v quantize &> /dev/null; then
        echo "Creating quantized versions..."
        ./quantize {model_id}.gguf {model_id}_q4_0.gguf q4_0
        ./quantize {model_id}.gguf {model_id}_q4_1.gguf q4_1  
        ./quantize {model_id}.gguf {model_id}_q5_0.gguf q5_0
        ./quantize {model_id}.gguf {model_id}_q5_1.gguf q5_1
        ./quantize {model_id}.gguf {model_id}_q8_0.gguf q8_0
        echo "✅ Quantized models created!"
    else
        echo "⚠️ 'quantize' tool not found - skipping quantization"
    fi
else
    echo "❌ convert_hf_to_gguf.py not found"
    echo "Please install llama.cpp tools first"
fi

echo "Conversion complete!"
"""
            
            with open(lm_studio_dir / "convert_to_gguf.sh", "w") as f:
                f.write(conversion_script)
            
            # Make script executable
            os.chmod(lm_studio_dir / "convert_to_gguf.sh", 0o755)
            
            logger.info(f"✅ LM Studio model saved: {lm_studio_dir}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to save LM Studio format: {e}")
            return False
    
    def save_model(self, model: nn.Module, tokenizer, metadata: ModelMetadata,
                   optimizer=None, scheduler=None) -> str:
        """Save model with comprehensive metadata and LM Studio compatibility."""
        try:
            # Generate model ID
            model_id = self._generate_model_id(metadata)
            
            # Create model directory
            model_dir = self.models_dir / "checkpoints" / model_id
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Save original model format
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
            
            # Save for LM Studio compatibility
            lm_studio_success = self._save_for_lm_studio(model, tokenizer, metadata, model_id)
            
            # Create symlink to latest model if this is the best
            if "best" in metadata.tags:
                latest_link = self.models_dir / "latest_best"
                if latest_link.exists():
                    latest_link.unlink()
                try:
                    latest_link.symlink_to(f"checkpoints/{model_id}")
                except OSError:
                    # Fallback for systems that don't support symlinks
                    with open(latest_link, 'w') as f:
                        f.write(model_id)
                
                # Also create LM Studio latest link
                if lm_studio_success:
                    lm_latest_link = self.models_dir / "lm_studio" / "latest_best"
                    if lm_latest_link.exists():
                        lm_latest_link.unlink()
                    try:
                        lm_latest_link.symlink_to(model_id)
                    except OSError:
                        with open(lm_latest_link, 'w') as f:
                            f.write(model_id)
            
            logger.info(f"✅ Model saved successfully: {model_id}")
            logger.info(f"   Size: {metadata.model_size_mb:.2f} MB")
            logger.info(f"   Parameters: {metadata.total_parameters:,}")
            logger.info(f"   Original format: {model_dir}")
            if lm_studio_success:
                logger.info(f"   LM Studio format: {self.models_dir / 'lm_studio' / model_id}")
                logger.info("   🚀 Ready for LM Studio!")
            
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
                        model_path = latest_link.readlink()
                        model_id = model_path.name if model_path.name != "checkpoints" else model_path.parent.name
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
            
            # Check if LM Studio version exists
            lm_studio_path = self.models_dir / "lm_studio" / model_id
            if lm_studio_path.exists():
                logger.info(f"   🚀 LM Studio version available at: {lm_studio_path}")
            
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
                    
                    model_id = metadata_file.stem
                    lm_studio_available = (self.models_dir / "lm_studio" / model_id).exists()
                    
                    model_info = {
                        'id': model_id,
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
                        'notes': metadata_dict.get('notes', ''),
                        'lm_studio_ready': lm_studio_available
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
            lm_studio_dir = self.models_dir / "lm_studio" / model_id
            
            # Remove model directory
            if model_dir.exists():
                shutil.rmtree(model_dir)
            
            # Remove LM Studio directory
            if lm_studio_dir.exists():
                shutil.rmtree(lm_studio_dir)
                logger.info(f"   Removed LM Studio version")
            
            # Remove metadata
            if metadata_path.exists():
                metadata_path.unlink()
            
            # Update latest links if this was the latest
            for latest_link_name in ["latest_best"]:
                latest_link = self.models_dir / latest_link_name
                lm_latest_link = self.models_dir / "lm_studio" / latest_link_name
                
                for link in [latest_link, lm_latest_link]:
                    if link.exists():
                        try:
                            if link.is_symlink():
                                current_latest = link.readlink().name
                                if "checkpoints" in str(link.readlink()):
                                    current_latest = link.readlink().name
                            else:
                                with open(link, 'r') as f:
                                    current_latest = f.read().strip()
                            
                            if current_latest == model_id:
                                link.unlink()
                                # Find new best model
                                remaining_models = self.list_models()
                                if remaining_models:
                                    best_model = min(remaining_models, key=lambda x: x['best_loss'])
                                    try:
                                        if link == lm_latest_link:
                                            link.symlink_to(best_model['id'])
                                        else:
                                            link.symlink_to(f"checkpoints/{best_model['id']}")
                                    except OSError:
                                        with open(link, 'w') as f:
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
                model_info = json.load(f)
            
            # Add LM Studio availability info
            lm_studio_path = self.models_dir / "lm_studio" / model_id
            model_info['lm_studio_available'] = lm_studio_path.exists()
            model_info['lm_studio_path'] = str(lm_studio_path) if lm_studio_path.exists() else None
            
            return model_info
                
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
    
    def export_model(self, model_id: str, export_path: str, format: str = "both") -> bool:
        """Export model for sharing or deployment.
        
        Args:
            model_id: ID of the model to export
            export_path: Path to export to
            format: 'original', 'lm_studio', or 'both'
        """
        try:
            model_dir = self.models_dir / "checkpoints" / model_id
            metadata_path = self.models_dir / "metadata" / f"{model_id}.json"
            lm_studio_dir = self.models_dir / "lm_studio" / model_id
            
            if not model_dir.exists() or not metadata_path.exists():
                raise FileNotFoundError(f"Model {model_id} not found")
            
            export_path = Path(export_path)
            export_path.mkdir(parents=True, exist_ok=True)
            
            exported_formats = []
            
            # Export original format
            if format in ["original", "both"]:
                original_export = export_path / "original"
                shutil.copytree(model_dir, original_export, dirs_exist_ok=True)
                shutil.copy2(metadata_path, original_export / "metadata.json")
                exported_formats.append("original")
                logger.info(f"   Exported original format to: {original_export}")
            
            # Export LM Studio format
            if format in ["lm_studio", "both"] and lm_studio_dir.exists():
                lm_studio_export = export_path / "lm_studio"
                shutil.copytree(lm_studio_dir, lm_studio_export, dirs_exist_ok=True)
                exported_formats.append("lm_studio")
                logger.info(f"   Exported LM Studio format to: {lm_studio_export}")
            elif format in ["lm_studio", "both"]:
                logger.warning(f"   LM Studio format not available for model {model_id}")
            
            # Create export info
            export_info = {
                'model_id': model_id,
                'exported_at': datetime.now().isoformat(),
                'export_version': '2.0',
                'exported_formats': exported_formats,
                'original_available': format in ["original", "both"],
                'lm_studio_available': format in ["lm_studio", "both"] and lm_studio_dir.exists()
            }
            
            with open(export_path / "export_info.json", 'w') as f:
                json.dump(export_info, f, indent=2)
            
            # Create comprehensive README
            readme_content = f"""# Exported Model: {model_id}

This export contains a trained word-level transformer model.

## Export Contents

"""
            
            if "original" in exported_formats:
                readme_content += """### Original Format (`original/`)
- Compatible with the original training framework
- Contains: model.pt, tokenizer.pkl, metadata.json
- Use with: Your custom inference scripts

"""
            
            if "lm_studio" in exported_formats:
                readme_content += """### LM Studio Format (`lm_studio/`)
- Ready to use with LM Studio
- Contains: HuggingFace compatible files
- Use with: LM Studio, Ollama, or other HF-compatible tools

#### LM Studio Usage:
1. Open LM Studio
2. Load model from the `lm_studio/` folder
3. Use prompt format: `<user> Your message <bot>`

"""
            
            readme_content += f"""## Model Details
- **Export Date**: {export_info['exported_at']}
- **Model ID**: {model_id}
- **Formats**: {', '.join(exported_formats)}

## License
Custom License - See original training repository for details.
"""
            
            with open(export_path / "README.md", "w") as f:
                f.write(readme_content)
            
            logger.info(f"✅ Model exported to: {export_path}")
            logger.info(f"   Formats: {', '.join(exported_formats)}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to export model {model_id}: {e}")
            return False
    
    def convert_existing_to_lm_studio(self, model_id: str) -> bool:
        """Convert an existing model to LM Studio format."""
        try:
            if not HF_AVAILABLE:
                logger.error("Cannot convert - HuggingFace dependencies not available")
                return False
            
            # Check if already exists
            lm_studio_dir = self.models_dir / "lm_studio" / model_id
            if lm_studio_dir.exists():
                logger.info(f"LM Studio version already exists for {model_id}")
                return True
            
            # Load the model
            model, tokenizer, metadata = self.load_model(model_id)
            
            # Save in LM Studio format
            success = self._save_for_lm_studio(model, tokenizer, metadata, model_id)
            
            if success:
                logger.info(f"✅ Converted {model_id} to LM Studio format")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Failed to convert {model_id} to LM Studio format: {e}")
            return False
    
    def get_lm_studio_models(self) -> List[Dict[str, Any]]:
        """Get list of models available in LM Studio format."""
        lm_studio_models = []
        
        try:
            lm_studio_dir = self.models_dir / "lm_studio"
            if not lm_studio_dir.exists():
                return lm_studio_models
            
            for model_dir in lm_studio_dir.iterdir():
                if model_dir.is_dir() and model_dir.name != "latest_best":
                    try:
                        # Load model card if available
                        model_card_path = model_dir / "model_card.json"
                        if model_card_path.exists():
                            with open(model_card_path, 'r') as f:
                                model_card = json.load(f)
                            
                            lm_studio_models.append({
                                'id': model_dir.name,
                                'name': model_card.get('model_name', 'Unknown'),
                                'path': str(model_dir),
                                'parameters': model_card.get('parameters', 'Unknown'),
                                'context_length': model_card.get('context_length', 'Unknown'),
                                'architecture': model_card.get('architecture', 'Unknown'),
                                'performance': model_card.get('performance', {}),
                                'ready_for_lm_studio': True
                            })
                    except Exception as e:
                        logger.warning(f"Could not load LM Studio model info for {model_dir.name}: {e}")
                        continue
            
            # Sort by performance (loss)
            lm_studio_models.sort(key=lambda x: x.get('performance', {}).get('loss', float('inf')))
            
        except Exception as e:
            logger.error(f"Error listing LM Studio models: {e}")
        
        return lm_studio_models
    
    def print_model_summary(self):
        """Print a comprehensive summary of all models."""
        print("\n" + "="*80)
        print("🤖 MODEL MANAGER SUMMARY")
        print("="*80)
        
        models = self.list_models()
        lm_studio_models = self.get_lm_studio_models()
        
        print(f"📊 Total Models: {len(models)}")
        print(f"🚀 LM Studio Ready: {len(lm_studio_models)}")
        print(f"💾 Storage Location: {self.models_dir}")
        
        if models:
            print(f"\n🏆 BEST MODEL (by loss):")
            best = models[0]
            print(f"   ID: {best['id']}")
            print(f"   Name: {best['name']} {best['version']}")
            print(f"   Loss: {best['best_loss']:.4f} (PPL: {best['best_perplexity']:.2f})")
            print(f"   Parameters: {best['parameters']:,}")
            print(f"   LM Studio Ready: {'✅' if best['lm_studio_ready'] else '❌'}")
            
            print(f"\n📋 ALL MODELS:")
            for i, model in enumerate(models, 1):
                status = "🚀" if model['lm_studio_ready'] else "⚠️"
                print(f"   {i}. {status} {model['name']} - Loss: {model['best_loss']:.4f} - {model['parameters']:,} params")
        
        if HF_AVAILABLE:
            print(f"\n✅ LM Studio compatibility: ENABLED")
        else:
            print(f"\n❌ LM Studio compatibility: DISABLED")
            print("   Install: pip install transformers safetensors")
        
        print("="*80)