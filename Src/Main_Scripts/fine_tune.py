# Copyright (c) 2025 Matias Nielsen. All rights reserved.
# Licensed under the Custom License below.

import os
import sys
import json
import time
import math
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import gc
from contextlib import contextmanager

# NEW: DeepSpeed integration
try:
    import deepspeed
    from deepspeed.ops.adam import FusedAdam
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False

# NEW: Mixed Precision Training
try:
    from torch.cuda.amp import GradScaler, autocast
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False

# NEW: Gradient Checkpointing
try:
    from torch.utils.checkpoint import checkpoint
    CHECKPOINT_AVAILABLE = True
except ImportError:
    CHECKPOINT_AVAILABLE = False

# Enhanced logging setup
def setup_logging():
    """Setup comprehensive logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('finetuning_debug.log', mode='w')
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info("🔧 Fine-tuning debug logging initialized")
    return logger

logger = setup_logging()

# Fine-tuning specific configuration classes
@dataclass
class ModelConfig:
    vocab_size: int = 50000
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    seq_length: int = 512
    dropout: float = 0.1
    model_type: str = "transformer"
    tokenizer_type: str = "subword"
    # NEW: Memory optimization options
    gradient_checkpointing: bool = True
    use_flash_attention: bool = False  # For future compatibility

@dataclass
class FineTuningConfig:
    # Base model configuration
    base_model_path: str = ""
    model_id: str = ""
    
    # Fine-tuning specific parameters
    learning_rate: float = 5e-5  # Lower than pre-training
    weight_decay: float = 0.01
    batch_size: int = 4  # Smaller for fine-tuning
    gradient_accumulation_steps: int = 8
    max_epochs: int = 10  # Fewer epochs for fine-tuning
    warmup_ratio: float = 0.05  # Less warmup needed
    save_every: int = 500
    eval_every: int = 100
    max_grad_norm: float = 1.0
    label_smoothing: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    
    # Fine-tuning specific options
    freeze_embeddings: bool = False
    freeze_first_n_layers: int = 0
    lora_rank: int = 0  # 0 means no LoRA, >0 enables LoRA fine-tuning
    lora_alpha: float = 16.0
    lora_dropout: float = 0.1
    target_modules: List[str] = None  # For LoRA
    
    # Mixed precision and memory optimization
    use_mixed_precision: bool = True
    precision_type: str = "fp16"  # "fp16", "bf16", or "fp32"
    use_loss_scaling: bool = True  # Automatically set based on precision_type
    amp_opt_level: str = "O1"  # O0, O1, O2, O3
    
    # DataLoader settings
    dataloader_num_workers: int = 0
    pin_memory: bool = False
    
    # DeepSpeed options
    use_deepspeed: bool = True
    deepspeed_config: str = "finetune_deepspeed_config.json"
    zero_stage: int = 2  # ZeRO stage (1, 2, or 3)
    
    # Data configuration
    dataset_path: str = ""
    max_samples: int = 10000
    validation_split: float = 0.1
    
    def __post_init__(self):
        """Automatically adjust settings based on precision type."""
        if self.precision_type == "bf16":
            self.use_loss_scaling = False  # BF16 doesn't need loss scaling
        elif self.precision_type == "fp16":
            self.use_loss_scaling = True   # FP16 needs loss scaling
        else:  # fp32
            self.use_mixed_precision = False
            self.use_loss_scaling = False
        
        if self.target_modules is None:
            self.target_modules = ["qkv", "out_proj", "fc1", "fc2"]  # Default LoRA targets

@dataclass
class FineTuningMetadata:
    model_name: str = "finetuned_transformer"
    version: str = "v1.0"
    base_model_id: str = ""
    created_at: str = ""
    last_modified: str = ""
    finetune_config: FineTuningConfig = None
    dataset_info: dict = None
    performance_metrics: dict = None
    model_size_mb: float = 0.0
    total_parameters: int = 0
    trainable_parameters: int = 0
    finetuning_time_hours: float = 0.0
    epochs_trained: int = 0
    best_loss: float = float('inf')
    best_perplexity: float = float('inf')
    hardware_used: str = ""
    pytorch_version: str = ""
    cuda_version: str = None
    notes: str = ""
    tags: list = None
    lora_enabled: bool = False
    frozen_layers: int = 0

# Import model classes from the original training script
# (These would normally be in a separate module, but including here for completeness)

class CheckpointWrapper(nn.Module):
    """Wrapper to apply gradient checkpointing to any module."""
    
    def __init__(self, module):
        super().__init__()
        self.module = module
    
    def forward(self, *args, **kwargs):
        if self.training and CHECKPOINT_AVAILABLE:
            return checkpoint(self.module, *args, **kwargs)
        else:
            return self.module(*args, **kwargs)

class MiniTransformer(nn.Module):
    """Transformer model compatible with the training script."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        
        # Positional embeddings
        self.pos_embeddings = nn.Parameter(torch.zeros(config.seq_length, config.hidden_size))
        
        # Transformer layers
        self.layers = nn.ModuleList()
        for _ in range(config.num_layers):
            layer = MiniTransformerBlock(config)
            if config.gradient_checkpointing and CHECKPOINT_AVAILABLE:
                layer = CheckpointWrapper(layer)
            self.layers.append(layer)
        
        # Output layers
        self.ln_final = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Tie embeddings to reduce parameters
        self.lm_head.weight = self.embeddings.weight
    
    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        token_embeddings = self.embeddings(input_ids)
        pos_embeddings = self.pos_embeddings[:seq_len].unsqueeze(0).expand(batch_size, -1, -1)
        
        hidden_states = token_embeddings + pos_embeddings
        
        # Apply layers
        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states)
            if i % 2 == 1:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        hidden_states = self.ln_final(hidden_states)
        logits = self.lm_head(hidden_states)
        
        return logits

class MiniTransformerBlock(nn.Module):
    """Transformer block."""
    
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        
        self.attn = MiniAttention(config)
        self.mlp = MiniMLP(config)
        
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        residual = x
        x_norm = self.ln1(x)
        attn_out = self.attn(x_norm)
        del x_norm
        x = residual + self.dropout(attn_out)
        del residual, attn_out
        
        residual = x
        x_norm = self.ln2(x)
        mlp_out = self.mlp(x_norm)
        del x_norm
        x = residual + self.dropout(mlp_out)
        del residual, mlp_out
        
        return x

class MiniAttention(nn.Module):
    """Multi-head attention."""
    
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(config.seq_length, config.seq_length), diagonal=1).bool()
        )
    
    def forward(self, x):
        batch_size, seq_len, hidden_size = x.shape
        
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        del qkv
        
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        del q, k
        
        scores.masked_fill_(self.causal_mask[:seq_len, :seq_len], float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        del scores
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        del attn_weights, v
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        output = self.out_proj(attn_output)
        del attn_output
        
        return output

class MiniMLP(nn.Module):
    """MLP block."""
    
    def __init__(self, config):
        super().__init__()
        intermediate_size = max(config.hidden_size * 2, 128)
        
        self.fc1 = nn.Linear(config.hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Tokenizer class (from original script)
class ImprovedTokenizer:
    """Tokenizer compatible with the training script."""
    
    def __init__(self):
        self.vocab = {
            "<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3, 
            "<user>": 4, "<assistant>": 5, "\n": 6, " ": 7
        }
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.trained = False
    
    def vocab_size(self):
        return len(self.vocab)
    
    def encode(self, text):
        """Encode text with fallback to character-level."""
        if not self.trained:
            raise ValueError("Tokenizer not trained")
        
        tokens = []
        words = text.split()
        
        for word in words:
            if word.lower() in self.vocab:
                tokens.append(self.vocab[word.lower()])
            else:
                for char in word:
                    if char in self.vocab:
                        tokens.append(self.vocab[char])
                    else:
                        tokens.append(self.vocab["<unk>"])
            
            if " " in self.vocab:
                tokens.append(self.vocab[" "])
        
        if tokens and tokens[-1] == self.vocab.get(" ", -1):
            tokens.pop()
        
        return tokens
    
    def decode(self, token_ids):
        """Decode with better text reconstruction."""
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                if token not in ["<pad>", "<bos>", "<eos>"]:
                    tokens.append(token)
        
        text = ""
        for token in tokens:
            if token == " ":
                text += " "
            elif len(token) == 1:
                text += token
            else:
                if text and not text.endswith(" "):
                    text += " "
                text += token
        
        return text.strip()

# Utility functions from original script
@contextmanager
def ultra_memory_cleanup():
    """Ultra-aggressive memory cleanup."""
    try:
        yield
    finally:
        for _ in range(3):
            gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.ipc_collect()
            if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                torch.cuda.reset_peak_memory_stats()
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()

def setup_device():
    """Setup device for fine-tuning."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using device: CUDA ({torch.cuda.get_device_name()})")
        torch.cuda.set_per_process_memory_fraction(0.70)
        torch.cuda.empty_cache()
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using device: MPS (Apple Silicon)")
        torch.mps.empty_cache()
    else:
        device = torch.device("cpu")
        logger.info("Using device: CPU")
        torch.set_num_threads(min(4, os.cpu_count() or 1))
    
    return device

device = setup_device()

def get_optimal_precision_config(device):
    """Determine optimal precision based on hardware capabilities."""
    supports_bf16 = False
    precision_type = "fp16"  # Default fallback
    
    if device.type == 'cuda' and torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        
        if major >= 8 or (major == 7 and minor >= 5):
            supports_bf16 = True
            precision_type = "bf16"
            
        if supports_bf16:
            try:
                test_tensor = torch.tensor([1.0], dtype=torch.bfloat16, device=device)
                del test_tensor
                logger.info("✅ BF16 support detected and verified")
            except:
                supports_bf16 = False
                precision_type = "fp16"
                logger.info("⚠️ BF16 hardware detected but PyTorch support unavailable, using FP16")
    
    return supports_bf16, precision_type

class ModelLoader:
    """Load models saved by the training script."""
    
    def __init__(self):
        pass
    
    def load_model(self, model_path: str):
        """Load model, tokenizer, and metadata from training script output."""
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model path not found: {model_path}")
        
        logger.info(f"Loading model from: {model_path}")
        
        # Load metadata
        metadata_path = model_path / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Extract model config
        model_config_dict = metadata.get('model_config', {})
        model_config = ModelConfig(**model_config_dict)
        
        # Load tokenizer
        tokenizer_path = model_path / "tokenizer.json"
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
        
        with open(tokenizer_path, 'r') as f:
            tokenizer_data = json.load(f)
        
        tokenizer = ImprovedTokenizer()
        tokenizer.vocab = tokenizer_data['vocab']
        tokenizer.id_to_token = tokenizer_data['id_to_token']
        tokenizer.trained = True
        
        # Create model
        model = MiniTransformer(model_config)
        
        # Load model weights
        model_weights_path = model_path / "model.pth"
        if model_weights_path.exists():
            # Regular model loading
            state_dict = torch.load(model_weights_path, map_location='cpu')
            model.load_state_dict(state_dict)
            logger.info("Loaded regular model weights")
        else:
            # Try DeepSpeed checkpoint
            checkpoint_path = model_path / "checkpoint"
            if checkpoint_path.exists():
                logger.info("Found DeepSpeed checkpoint, will load during DeepSpeed initialization")
            else:
                raise FileNotFoundError(f"No model weights found in {model_path}")
        
        logger.info(f"Model loaded successfully: {metadata.get('model_name', 'Unknown')}")
        
        return model, tokenizer, metadata, model_config

class FineTuningDataset(Dataset):
    """Dataset for fine-tuning with task-specific formatting."""
    
    def __init__(self, texts: List[str], tokenizer, seq_length: int, task_type: str = "chat"):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.pad_token_id = tokenizer.vocab.get("<pad>", 0)
        self.task_type = task_type
        
        logger.info(f"Creating fine-tuning dataset with seq_length={seq_length}, task_type={task_type}")
        
        self.sequences = []
        
        for text in texts:
            if not text or len(text.strip()) < 5:
                continue
            
            try:
                # Format based on task type
                if self.task_type == "chat":
                    formatted_text = self._format_chat_text(text)
                elif self.task_type == "instruction":
                    formatted_text = self._format_instruction_text(text)
                else:
                    formatted_text = text.strip()
                
                tokens = tokenizer.encode(formatted_text)
                if len(tokens) < 3:
                    continue
                
                # Add special tokens
                bos_id = tokenizer.vocab.get("<bos>", 2)
                eos_id = tokenizer.vocab.get("<eos>", 3)
                full_sequence = [bos_id] + tokens[:seq_length-2] + [eos_id]
                
                # Pad to exact length
                if len(full_sequence) < seq_length + 1:
                    full_sequence.extend([self.pad_token_id] * (seq_length + 1 - len(full_sequence)))
                else:
                    full_sequence = full_sequence[:seq_length + 1]
                
                self.sequences.append(full_sequence)
                
            except Exception as e:
                continue
        
        if not self.sequences:
            raise ValueError("No valid sequences created!")
        
        logger.info(f"Created {len(self.sequences):,} fine-tuning sequences")
    
    def _format_chat_text(self, text):
        """Format text for chat fine-tuning."""
        # Assume text is already formatted with <user> and <assistant> tags
        return text
    
    def _format_instruction_text(self, text):
        """Format text for instruction fine-tuning."""
        # Simple instruction formatting
        if "instruction:" in text.lower() and "response:" in text.lower():
            parts = text.split("response:", 1)
            if len(parts) == 2:
                instruction = parts[0].replace("instruction:", "").strip()
                response = parts[1].strip()
                return f"<user> {instruction} <assistant> {response}"
        return text
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        input_ids = torch.tensor(sequence[:-1], dtype=torch.long)
        target_ids = torch.tensor(sequence[1:], dtype=torch.long)
        return input_ids, target_ids

def create_finetune_deepspeed_config(finetune_config: FineTuningConfig):
    """Create DeepSpeed config optimized for fine-tuning."""
    
    config = {
        "train_batch_size": finetune_config.batch_size * finetune_config.gradient_accumulation_steps,
        "train_micro_batch_size_per_gpu": finetune_config.batch_size,
        "gradient_accumulation_steps": finetune_config.gradient_accumulation_steps,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": finetune_config.learning_rate,
                "betas": [finetune_config.beta1, finetune_config.beta2],
                "eps": 1e-8,
                "weight_decay": finetune_config.weight_decay
            }
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": finetune_config.learning_rate,
                "warmup_num_steps": "auto",
                "total_num_steps": "auto"
            }
        },
        "gradient_clipping": finetune_config.max_grad_norm,
        "steps_per_print": 10,
        "wall_clock_breakdown": False
    }
    
    # Precision configuration
    if finetune_config.precision_type == "bf16":
        config["bf16"] = {"enabled": True}
        config["fp16"] = {"enabled": False}
        logger.info("DeepSpeed: BF16 enabled for fine-tuning")
        
    elif finetune_config.precision_type == "fp16":
        config["fp16"] = {
            "enabled": True,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1,
            "auto_cast": False
        }
        config["bf16"] = {"enabled": False}
        logger.info("DeepSpeed: FP16 enabled for fine-tuning")
        
    else:  # fp32
        config["fp16"] = {"enabled": False}
        config["bf16"] = {"enabled": False}
        logger.info("DeepSpeed: FP32 mode for fine-tuning")
    
    # ZeRO configuration (more conservative for fine-tuning)
    if finetune_config.zero_stage >= 1:
        config["zero_optimization"] = {
            "stage": finetune_config.zero_stage,
            "offload_optimizer": {
                "device": "cpu" if finetune_config.zero_stage >= 2 else "none",
                "pin_memory": True
            },
            "offload_param": {
                "device": "cpu" if finetune_config.zero_stage >= 3 else "none",
                "pin_memory": True
            },
            "overlap_comm": True,
            "contiguous_gradients": True,
            "sub_group_size": 1e9,
            "reduce_bucket_size": "auto"
        }
        
        if finetune_config.zero_stage >= 3:
            config["zero_optimization"].update({
                "stage3_prefetch_bucket_size": "auto",
                "stage3_param_persistence_threshold": "auto",
                "stage3_max_live_parameters": 1e9,
                "stage3_max_reuse_distance": 1e9,
                "stage3_gather_16bit_weights_on_model_save": True,
                "memory_efficient_linear": False
            })
    
    # Activation checkpointing
    config["activation_checkpointing"] = {
        "partition_activations": True,
        "cpu_checkpointing": True,
        "contiguous_memory_optimization": False,
        "number_checkpoints": 4,
        "synchronize_checkpoint_boundary": False,
        "profile": False
    }
    
    return config

def apply_layer_freezing(model, freeze_embeddings=False, freeze_first_n_layers=0):
    """Freeze specific layers for fine-tuning."""
    frozen_params = 0
    total_params = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        
        # Freeze embeddings if requested
        if freeze_embeddings and ("embeddings" in name or "pos_embeddings" in name):
            param.requires_grad = False
            frozen_params += param.numel()
            logger.info(f"Frozen embedding layer: {name}")
            continue
        
        # Freeze first N transformer layers if requested
        if freeze_first_n_layers > 0 and "layers." in name:
            layer_idx = int(name.split("layers.")[1].split(".")[0])
            if layer_idx < freeze_first_n_layers:
                param.requires_grad = False
                frozen_params += param.numel()
                continue
        
        # Keep everything else trainable
        param.requires_grad = True
    
    trainable_params = total_params - frozen_params
    
    logger.info(f"Layer freezing applied:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Frozen parameters: {frozen_params:,} ({frozen_params/total_params*100:.1f}%)")
    logger.info(f"  Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
    
    return trainable_params

def load_finetuning_data(data_path: str, max_samples: Optional[int] = None) -> List[str]:
    """Load fine-tuning data."""
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    logger.info(f"Loading fine-tuning data from: {data_path}")
    
    texts = []
    processed_count = 0
    
    try:
        if data_path.suffix == '.jsonl':
            # JSONL format
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if max_samples and processed_count >= max_samples:
                        break
                    
                    try:
                        line = line.strip()
                        if not line:
                            continue
                        
                        record = json.loads(line)
                        
                        # Extract text based on different formats
                        text = ""
                        if "conversations" in record:
                            # Multi-turn conversation format
                            for turn in record["conversations"]:
                                role = turn.get("from", "").lower()
                                content = turn.get("value", "").strip()
                                if role == "human" or role == "user":
                                    text += f"<user> {content} "
                                elif role == "gpt" or role == "assistant":
                                    text += f"<assistant> {content} "
                        elif "instruction" in record and "output" in record:
                            # Instruction-following format
                            instruction = record["instruction"].strip()
                            output = record["output"].strip()
                            text = f"<user> {instruction} <assistant> {output}"
                        elif "text" in record:
                            # Simple text format
                            text = record["text"].strip()
                        else:
                            continue
                        
                        if text and len(text.split()) >= 5:
                            texts.append(text)
                            processed_count += 1
                        
                    except (json.JSONDecodeError, KeyError):
                        continue
        
        elif data_path.suffix == '.json':
            # JSON format
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                if isinstance(data, list):
                    for record in data:
                        if max_samples and processed_count >= max_samples:
                            break
                        
                        text = ""
                        if "conversations" in record:
                            for turn in record["conversations"]:
                                role = turn.get("from", "").lower()
                                content = turn.get("value", "").strip()
                                if role == "human" or role == "user":
                                    text += f"<user> {content} "
                                elif role == "gpt" or role == "assistant":
                                    text += f"<assistant> {content} "
                        elif "instruction" in record and "output" in record:
                            instruction = record["instruction"].strip()
                            output = record["output"].strip()
                            text = f"<user> {instruction} <assistant> {output}"
                        elif "text" in record:
                            text = record["text"].strip()
                        else:
                            continue
                        
                        if text and len(text.split()) >= 5:
                            texts.append(text)
                            processed_count += 1
        
        else:
            # Plain text format
            with open(data_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                lines = content.split('\n')
                
                for line in lines:
                    if max_samples and processed_count >= max_samples:
                        break
                    
                    line = line.strip()
                    if line and len(line.split()) >= 5:
                        texts.append(line)
                        processed_count += 1
        
        gc.collect()
        
    except Exception as e:
        logger.error(f"Error loading fine-tuning data: {e}")
        raise
    
    logger.info(f"Loaded {len(texts):,} fine-tuning texts")
    return texts

def finetune_epoch_with_deepspeed(deepspeed_engine, dataloader, epoch, precision_type="fp16"):
    """Fine-tuning epoch with DeepSpeed."""
    deepspeed_engine.train()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    num_batches = 0
    
    logger.info(f"Starting DeepSpeed fine-tuning epoch {epoch} with {len(dataloader)} batches ({precision_type})")
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        try:
            if batch_idx > 0 and batch_idx % 3 == 0:
                with ultra_memory_cleanup():
                    pass
            
            if inputs.numel() == 0 or targets.numel() == 0:
                continue
            
            inputs = inputs.to(deepspeed_engine.local_rank)
            targets = targets.to(deepspeed_engine.local_rank)
            
            if torch.isnan(inputs).any() or torch.isinf(inputs).any():
                del inputs, targets
                continue
            
            # Forward pass
            logits = deepspeed_engine(inputs)
            
            # Compute loss
            criterion = nn.CrossEntropyLoss(ignore_index=0)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            
            if torch.isnan(loss) or torch.isinf(loss):
                del inputs, targets, logits, loss
                continue
            
            # DeepSpeed backward and step
            deepspeed_engine.backward(loss)
            deepspeed_engine.step()
            
            # Calculate accuracy
            with torch.no_grad():
                predictions = torch.argmax(logits, dim=-1)
                pad_mask = (targets != 0)
                correct = ((predictions == targets) & pad_mask).sum().item()
                valid_tokens = pad_mask.sum().item()
                
                total_correct += correct
                total_tokens += valid_tokens
                total_loss += loss.item()
                num_batches += 1
            
            del logits, predictions, pad_mask, inputs, targets, loss
            
            if batch_idx % 10 == 0 and num_batches > 0:
                current_loss = total_loss / num_batches
                current_acc = total_correct / max(total_tokens, 1)
                current_lr = deepspeed_engine.get_lr()[0]
                
                logger.info(f"Fine-tune Epoch {epoch} | Batch {batch_idx} | "
                           f"Loss: {current_loss:.4f} | Acc: {current_acc:.3f} | "
                           f"LR: {current_lr:.6f}")
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning(f"OOM at batch {batch_idx}, skipping...")
                with ultra_memory_cleanup():
                    pass
                continue
            else:
                raise e
        except Exception as e:
            logger.warning(f"Error at batch {batch_idx}: {e}")
            continue
    
    if num_batches == 0:
        return float('inf'), 0.0
    
    avg_loss = total_loss / num_batches
    avg_acc = total_correct / max(total_tokens, 1)
    
    return avg_loss, avg_acc

def finetune_epoch_regular(model, dataloader, criterion, optimizer, scheduler, epoch, 
                          gradient_accumulation_steps=1, max_grad_norm=1.0, 
                          use_mixed_precision=False, scaler=None, precision_type="fp16"):
    """Regular fine-tuning epoch."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    num_batches = 0
    accumulation_count = 0
    
    optimizer.zero_grad()
    
    logger.info(f"Starting fine-tuning epoch {epoch} with {len(dataloader)} batches ({precision_type})")
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        try:
            if batch_idx > 0 and batch_idx % 5 == 0:
                with ultra_memory_cleanup():
                    pass
            
            if inputs.numel() == 0 or targets.numel() == 0:
                continue
                
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            if torch.isnan(inputs).any() or torch.isinf(inputs).any():
                del inputs, targets
                continue
            
            # Forward pass with mixed precision
            if use_mixed_precision and scaler is not None:
                if precision_type == "bf16":
                    with autocast(dtype=torch.bfloat16):
                        logits = model(inputs)
                        loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
                else:  # fp16
                    with autocast(dtype=torch.float16):
                        logits = model(inputs)
                        loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            else:
                logits = model(inputs)
                loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            
            if torch.isnan(loss) or torch.isinf(loss):
                del inputs, targets, logits, loss
                continue
            
            scaled_loss = loss / gradient_accumulation_steps
            
            if use_mixed_precision and scaler is not None:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            
            with torch.no_grad():
                predictions = torch.argmax(logits, dim=-1)
                pad_mask = (targets != criterion.ignore_index)
                correct = ((predictions == targets) & pad_mask).sum().item()
                valid_tokens = pad_mask.sum().item()
                
                total_correct += correct
                total_tokens += valid_tokens
                total_loss += loss.item()
                num_batches += 1
            
            del logits, predictions, pad_mask, scaled_loss, inputs, targets
            
            accumulation_count += 1
            
            if accumulation_count >= gradient_accumulation_steps:
                if use_mixed_precision and scaler is not None:
                    if max_grad_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                
                scheduler.step()
                optimizer.zero_grad()
                accumulation_count = 0
                
                with ultra_memory_cleanup():
                    pass
            
            if batch_idx % 10 == 0 and num_batches > 0:
                current_loss = total_loss / num_batches
                current_acc = total_correct / max(total_tokens, 1)
                current_lr = optimizer.param_groups[0]['lr']
                
                logger.info(f"Fine-tune Epoch {epoch} | Batch {batch_idx} | "
                           f"Loss: {current_loss:.4f} | Acc: {current_acc:.3f} | "
                           f"LR: {current_lr:.6f}")
            
            del loss
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning(f"OOM at batch {batch_idx}, skipping...")
                optimizer.zero_grad()
                with ultra_memory_cleanup():
                    pass
                continue
            else:
                raise e
        except Exception as e:
            logger.warning(f"Error at batch {batch_idx}: {e}")
            continue
    
    if accumulation_count > 0:
        if use_mixed_precision and scaler is not None:
            if max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        optimizer.zero_grad()
    
    if num_batches == 0:
        return float('inf'), 0.0
    
    avg_loss = total_loss / num_batches
    avg_acc = total_correct / max(total_tokens, 1)
    
    return avg_loss, avg_acc

class FineTuningManager:
    """Manage fine-tuning process and model saving."""
    
    def __init__(self, save_dir):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
    
    def save_finetuned_model(self, model, tokenizer, metadata, base_model_id, deepspeed_engine=None):
        """Save fine-tuned model."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_id = f"finetuned_{base_model_id}_{timestamp}"
        model_path = self.save_dir / model_id
        model_path.mkdir(exist_ok=True)
        
        try:
            with ultra_memory_cleanup():
                if deepspeed_engine is not None and DEEPSPEED_AVAILABLE:
                    # DeepSpeed model saving
                    deepspeed_engine.save_checkpoint(str(model_path), tag="checkpoint")
                    logger.info(f"Fine-tuned DeepSpeed checkpoint saved to: {model_path}")
                else:
                    # Regular model saving
                    original_device = next(model.parameters()).device
                    model.cpu()
                    
                    state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    torch.save(state_dict, model_path / "model.pth")
                    del state_dict
                    
                    model.to(original_device)
            
            # Save tokenizer data
            tokenizer_data = {
                'vocab': tokenizer.vocab,
                'id_to_token': tokenizer.id_to_token,
                'vocab_size': tokenizer.vocab_size()
            }
            with open(model_path / "tokenizer.json", 'w') as f:
                json.dump(tokenizer_data, f, indent=2)
            
            # Save metadata
            if hasattr(metadata, '__dict__'):
                metadata_dict = asdict(metadata) if hasattr(metadata, '__dataclass_fields__') else metadata.__dict__
            else:
                metadata_dict = metadata
                
            with open(model_path / "metadata.json", 'w') as f:
                json.dump(metadata_dict, f, indent=2, default=str)
            
            logger.info(f"Fine-tuned model saved to: {model_path}")
            return model_id
            
        except Exception as e:
            logger.error(f"Error saving fine-tuned model: {e}")
            return None

class ImprovedScheduler:
    """Simple scheduler for fine-tuning."""
    
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.1):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lr = optimizer.param_groups[0]['lr']
        self.min_lr = self.base_lr * min_lr_ratio
        self.current_step = 0
    
    def step(self):
        self.current_step += 1
        
        if self.current_step <= self.warmup_steps:
            lr = self.base_lr * self.current_step / self.warmup_steps
        else:
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr

def get_memory_usage():
    """Get current memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        return f"GPU: {allocated:.2f}GB allocated, {cached:.2f}GB cached, {max_allocated:.2f}GB peak"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        allocated = torch.mps.current_allocated_memory() / 1024**3
        return f"MPS: {allocated:.2f}GB allocated"
    else:
        return "CPU mode"

def count_parameters(model):
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def evaluate_model_minimal(model, dataloader, criterion, max_batches=3, use_mixed_precision=False, 
                          deepspeed_engine=None, precision_type="fp16"):
    """Minimal evaluation."""
    if deepspeed_engine is not None:
        deepspeed_engine.eval()
        model = deepspeed_engine
    else:
        model.eval()
    
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    num_batches = 0
    
    try:
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                if batch_idx >= max_batches:
                    break
                
                try:
                    if deepspeed_engine is not None:
                        inputs = inputs.to(deepspeed_engine.local_rank)
                        targets = targets.to(deepspeed_engine.local_rank)
                    else:
                        inputs = inputs.to(device)
                        targets = targets.to(device)
                    
                    if use_mixed_precision and AMP_AVAILABLE and deepspeed_engine is None:
                        if precision_type == "bf16":
                            with autocast(dtype=torch.bfloat16):
                                logits = model(inputs)
                                loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
                        else:  # fp16
                            with autocast(dtype=torch.float16):
                                logits = model(inputs)
                                loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
                    else:
                        logits = model(inputs)
                        loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
                    
                    predictions = torch.argmax(logits, dim=-1)
                    pad_mask = (targets != criterion.ignore_index)
                    correct = ((predictions == targets) & pad_mask).sum().item()
                    valid_tokens = pad_mask.sum().item()
                    
                    total_loss += loss.item()
                    total_correct += correct
                    total_tokens += valid_tokens
                    num_batches += 1
                    
                    del logits, loss, predictions, pad_mask, inputs, targets
                    
                except Exception as e:
                    continue
                
                with ultra_memory_cleanup():
                    pass
    
    finally:
        if deepspeed_engine is not None:
            deepspeed_engine.train()
        else:
            model.train()
    
    if num_batches == 0:
        return {'avg_loss': float('inf'), 'accuracy': 0.0, 'perplexity': float('inf')}
    
    avg_loss = total_loss / num_batches
    accuracy = total_correct / max(total_tokens, 1)
    perplexity = math.exp(min(avg_loss, 10))
    
    return {
        'avg_loss': avg_loss,
        'accuracy': accuracy,
        'perplexity': perplexity
    }

def generate_sample_text_minimal(model, tokenizer, prompt="<user> Hello", max_length=20, 
                                use_mixed_precision=False, deepspeed_engine=None, precision_type="fp16"):
    """Generate sample text."""
    if deepspeed_engine is not None:
        deepspeed_engine.eval()
        model = deepspeed_engine
    else:
        model.eval()
    
    try:
        with torch.no_grad():
            input_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0)
            
            if deepspeed_engine is not None:
                input_ids = input_ids.to(deepspeed_engine.local_rank)
            else:
                input_ids = input_ids.to(device)
            
            generated = input_ids.clone()
            
            for _ in range(max_length):
                if generated.size(1) >= model.config.seq_length:
                    break
                
                if use_mixed_precision and AMP_AVAILABLE and deepspeed_engine is None:
                    if precision_type == "bf16":
                        with autocast(dtype=torch.bfloat16):
                            logits = model(generated)
                    else:  # fp16
                        with autocast(dtype=torch.float16):
                            logits = model(generated)
                else:
                    logits = model(generated)
                
                next_token_logits = logits[0, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1)
                
                generated = torch.cat([generated, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
                
                if next_token.item() == tokenizer.vocab.get("<eos>", -1):
                    break
                
                del logits, next_token_logits
            
            response_ids = generated[0][input_ids.size(1):].tolist()
            response = tokenizer.decode(response_ids)
            
            del input_ids, generated
            
            return response.strip()
    
    except Exception as e:
        return "Generation failed"
    finally:
        if deepspeed_engine is not None:
            deepspeed_engine.train()
        else:
            model.train()

def get_finetune_config():
    """Get fine-tuning configuration based on hardware."""
    supports_bf16, precision_type = get_optimal_precision_config(device)
    
    if device.type == 'cuda':
        config = FineTuningConfig(
            learning_rate=3e-5,  # Lower learning rate for fine-tuning
            batch_size=2,        # Smaller batches
            gradient_accumulation_steps=16,
            max_epochs=5,        # Fewer epochs
            warmup_ratio=0.03,
            max_grad_norm=1.0,
            precision_type=precision_type,
            use_mixed_precision=AMP_AVAILABLE,
            use_deepspeed=DEEPSPEED_AVAILABLE,
            zero_stage=2,
            max_samples=5000,    # Smaller dataset
            validation_split=0.1,
            freeze_embeddings=False,
            freeze_first_n_layers=0
        )
    elif device.type == 'mps':
        config = FineTuningConfig(
            learning_rate=1e-5,
            batch_size=1,
            gradient_accumulation_steps=32,
            max_epochs=3,
            warmup_ratio=0.05,
            precision_type="fp32",
            use_mixed_precision=False,
            use_deepspeed=False,
            max_samples=1000,
            validation_split=0.1,
            freeze_embeddings=True,
            freeze_first_n_layers=1
        )
    else:  # CPU
        config = FineTuningConfig(
            learning_rate=1e-5,
            batch_size=1,
            gradient_accumulation_steps=64,
            max_epochs=2,
            warmup_ratio=0.1,
            precision_type="fp32",
            use_mixed_precision=False,
            use_deepspeed=False,
            max_samples=500,
            validation_split=0.1,
            freeze_embeddings=True,
            freeze_first_n_layers=2
        )
    
    return config

def create_optimized_dataloader(dataset, batch_size, training_config, shuffle=True):
    """Create memory-optimized dataloader."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=training_config.dataloader_num_workers,
        pin_memory=training_config.pin_memory,
        drop_last=True,
        persistent_workers=False,
        prefetch_factor=2 if training_config.dataloader_num_workers > 0 else 2
    )

def main():
    """Main fine-tuning function."""
    logger.info("🚀 Starting Model Fine-Tuning")
    logger.info("=" * 60)
    
    # Load configuration from JSON
    config_path = Path("finetune_config.json")
    if config_path.exists():
        logger.info(f"Loading configuration from {config_path}")
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # Override defaults with loaded config
        finetune_config = get_finetune_config()
        for key, value in config_dict.items():
            if hasattr(finetune_config, key):
                setattr(finetune_config, key, value)
                logger.info(f"Config override: {key} = {value}")
    else:
        logger.info("No config file found, using defaults")
        finetune_config = get_finetune_config()
    
    # Validate required paths
    if not finetune_config.base_model_path:
        logger.error("base_model_path must be specified in config")
        return 1
    
    if not finetune_config.dataset_path:
        logger.error("dataset_path must be specified in config")
        return 1
    
    logger.info(f"Fine-tuning Configuration:")
    logger.info(f"  Base model: {finetune_config.base_model_path}")
    logger.info(f"  Dataset: {finetune_config.dataset_path}")
    logger.info(f"  Learning rate: {finetune_config.learning_rate}")
    logger.info(f"  Batch size: {finetune_config.batch_size}")
    logger.info(f"  Max epochs: {finetune_config.max_epochs}")
    logger.info(f"  Precision: {finetune_config.precision_type}")
    logger.info(f"  Freeze embeddings: {finetune_config.freeze_embeddings}")
    logger.info(f"  Freeze first N layers: {finetune_config.freeze_first_n_layers}")
    logger.info(f"  Use DeepSpeed: {finetune_config.use_deepspeed}")
    
    try:
        # Load base model
        logger.info("📥 Loading base model...")
        model_loader = ModelLoader()
        model, tokenizer, base_metadata, model_config = model_loader.load_model(finetune_config.base_model_path)
        model = model.to(device)
        
        base_model_id = finetune_config.model_id or Path(finetune_config.base_model_path).name
        
        logger.info(f"Base model loaded: {base_metadata.get('model_name', 'Unknown')}")
        
        # Apply layer freezing
        if finetune_config.freeze_embeddings or finetune_config.freeze_first_n_layers > 0:
            trainable_params = apply_layer_freezing(
                model, 
                finetune_config.freeze_embeddings, 
                finetune_config.freeze_first_n_layers
            )
        else:
            total_params, trainable_params = count_parameters(model)
        
        # Load fine-tuning data
        logger.info("📚 Loading fine-tuning data...")
        texts = load_finetuning_data(finetune_config.dataset_path, finetune_config.max_samples)
        
        # Create dataset
        task_type = "chat"  # Default to chat format
        dataset = FineTuningDataset(texts, tokenizer, model_config.seq_length, task_type)
        
        # Split into train/validation
        val_size = int(len(dataset) * finetune_config.validation_split)
        train_size = len(dataset) - val_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        # Create dataloaders
        train_dataloader = create_optimized_dataloader(
            train_dataset, finetune_config.batch_size, finetune_config, shuffle=True
        )
        val_dataloader = create_optimized_dataloader(
            val_dataset, finetune_config.batch_size, finetune_config, shuffle=False
        )
        
        logger.info(f"Training samples: {len(train_dataset):,}")
        logger.info(f"Validation samples: {len(val_dataset):,}")
        
        # Initialize DeepSpeed or regular training
        deepspeed_engine = None
        optimizer = None
        scheduler = None
        criterion = None
        scaler = None
        
        if finetune_config.use_deepspeed and DEEPSPEED_AVAILABLE:
            # Create DeepSpeed config
            ds_config = create_finetune_deepspeed_config(finetune_config)
            config_path = "finetune_deepspeed_config.json"
            with open(config_path, 'w') as f:
                json.dump(ds_config, f, indent=2)
            
            # Initialize DeepSpeed
            deepspeed_engine, optimizer, _, scheduler = deepspeed.initialize(
                model=model,
                config=config_path,
                model_parameters=[p for p in model.parameters() if p.requires_grad]
            )
            logger.info("DeepSpeed initialized for fine-tuning")
        else:
            # Regular training setup
            optimizer = optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=finetune_config.learning_rate,
                weight_decay=finetune_config.weight_decay,
                eps=1e-8,
                betas=(finetune_config.beta1, finetune_config.beta2)
            )
            
            criterion = nn.CrossEntropyLoss(
                ignore_index=tokenizer.vocab.get("<pad>", 0),
                label_smoothing=finetune_config.label_smoothing
            )
            
            total_steps = len(train_dataloader) * finetune_config.max_epochs // finetune_config.gradient_accumulation_steps
            warmup_steps = int(total_steps * finetune_config.warmup_ratio)
            
            scheduler = ImprovedScheduler(optimizer, warmup_steps, total_steps)
            
            if finetune_config.use_mixed_precision and AMP_AVAILABLE:
                if finetune_config.precision_type == "bf16":
                    scaler = GradScaler(enabled=False)
                else:
                    scaler = GradScaler()
        
        logger.info(f"Memory before fine-tuning: {get_memory_usage()}")
        
        # Fine-tuning loop
        logger.info("🎯 Starting fine-tuning...")
        finetuning_start = time.time()
        best_loss = float('inf')
        best_epoch = 0
        
        finetune_manager = FineTuningManager("finetuned_models")
        
        for epoch in range(1, finetune_config.max_epochs + 1):
            epoch_start = time.time()
            
            logger.info(f"=== Fine-tuning Epoch {epoch}/{finetune_config.max_epochs} ===")
            
            try:
                # Training
                if finetune_config.use_deepspeed and deepspeed_engine is not None:
                    train_loss, train_acc = finetune_epoch_with_deepspeed(
                        deepspeed_engine, train_dataloader, epoch, finetune_config.precision_type
                    )
                else:
                    train_loss, train_acc = finetune_epoch_regular(
                        model, train_dataloader, criterion, optimizer, scheduler, epoch,
                        finetune_config.gradient_accumulation_steps,
                        finetune_config.max_grad_norm,
                        finetune_config.use_mixed_precision,
                        scaler,
                        finetune_config.precision_type
                    )
                
                epoch_time = time.time() - epoch_start
                perplexity = math.exp(min(train_loss, 10))
                
                logger.info(f"Training - Loss: {train_loss:.4f}, Acc: {train_acc:.3f}, PPL: {perplexity:.2f}, Time: {epoch_time:.1f}s")
                
                # Validation
                eval_results = None
                if epoch % 2 == 0 or epoch == 1:
                    with ultra_memory_cleanup():
                        eval_criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.vocab.get("<pad>", 0))
                        eval_results = evaluate_model_minimal(
                            model, val_dataloader, eval_criterion, max_batches=5,
                            use_mixed_precision=finetune_config.use_mixed_precision and not finetune_config.use_deepspeed,
                            deepspeed_engine=deepspeed_engine if finetune_config.use_deepspeed else None,
                            precision_type=finetune_config.precision_type
                        )
                    logger.info(f"Validation - Loss: {eval_results['avg_loss']:.4f}, Acc: {eval_results['accuracy']:.3f}")
                
                # Sample generation
                if epoch % 3 == 0:
                    sample = generate_sample_text_minimal(
                        model, tokenizer, prompt="<user> What is machine learning?",
                        use_mixed_precision=finetune_config.use_mixed_precision and not finetune_config.use_deepspeed,
                        deepspeed_engine=deepspeed_engine if finetune_config.use_deepspeed else None,
                        precision_type=finetune_config.precision_type
                    )
                    logger.info(f"Sample: {sample}")
                
                # Track best model
                is_best = train_loss < best_loss
                if is_best:
                    best_loss = train_loss
                    best_epoch = epoch
                    logger.info(f"🏆 New best fine-tuning loss: {best_loss:.4f}")
                
                # Save model
                if is_best or epoch == finetune_config.max_epochs:
                    performance_metrics = {
                        "train_loss": float(train_loss),
                        "train_accuracy": float(train_acc),
                        "epoch": int(epoch),
                        "is_best": is_best,
                        "best_epoch": int(best_epoch),
                        "precision_type": finetune_config.precision_type,
                        "mixed_precision_used": finetune_config.use_mixed_precision,
                        "deepspeed_used": finetune_config.use_deepspeed,
                        "zero_stage": finetune_config.zero_stage if finetune_config.use_deepspeed else None,
                        "frozen_layers": finetune_config.freeze_first_n_layers,
                        "frozen_embeddings": finetune_config.freeze_embeddings,
                        "trainable_parameters": int(trainable_params),
                        "base_model_id": base_model_id
                    }
                    
                    if eval_results:
                        performance_metrics.update({
                            "val_loss": float(eval_results['avg_loss']),
                            "val_accuracy": float(eval_results['accuracy']),
                            "val_perplexity": float(eval_results['perplexity'])
                        })
                    
                    metadata = FineTuningMetadata(
                        model_name=f"FineTuned_{base_model_id}",
                        version=f"ft_v1.0_epoch_{epoch}",
                        base_model_id=base_model_id,
                        created_at=datetime.now().isoformat(),
                        finetune_config=finetune_config,
                        dataset_info={
                            "dataset_path": finetune_config.dataset_path,
                            "total_samples": len(texts),
                            "train_samples": len(train_dataset),
                            "val_samples": len(val_dataset),
                            "task_type": task_type
                        },
                        performance_metrics=performance_metrics,
                        total_parameters=sum(p.numel() for p in model.parameters()),
                        trainable_parameters=int(trainable_params),
                        epochs_trained=int(epoch),
                        best_loss=float(best_loss),
                        best_perplexity=float(math.exp(min(best_loss, 10))),
                        hardware_used=device.type.upper(),
                        pytorch_version=torch.__version__,
                        notes=f"Fine-tuned from {base_model_id} for {epoch} epochs with {finetune_config.precision_type.upper()} precision",
                        tags=["finetuned", f"base_{base_model_id}", f"{finetune_config.precision_type}_precision", f"epoch_{epoch}"] + 
                             (["best"] if is_best else []) + 
                             (["frozen_embeddings"] if finetune_config.freeze_embeddings else []) +
                             (["frozen_layers"] if finetune_config.freeze_first_n_layers > 0 else []) +
                             (["deepspeed"] if finetune_config.use_deepspeed else []),
                        lora_enabled=finetune_config.lora_rank > 0,
                        frozen_layers=finetune_config.freeze_first_n_layers
                    )
                    
                    try:
                        with ultra_memory_cleanup():
                            model_id = finetune_manager.save_finetuned_model(
                                model, tokenizer, metadata, base_model_id,
                                deepspeed_engine=deepspeed_engine if finetune_config.use_deepspeed else None
                            )
                            if model_id:
                                logger.info(f"💾 Fine-tuned model saved: {model_id}")
                    except Exception as save_error:
                        logger.error(f"Save failed: {save_error}")
                
                logger.info(f"Memory: {get_memory_usage()}")
                
                with ultra_memory_cleanup():
                    pass
                    
            except Exception as e:
                logger.error(f"Error in epoch {epoch}: {e}")
                
                if "out of memory" in str(e).lower():
                    logger.info("OOM detected during fine-tuning, attempting recovery...")
                    
                    if finetune_config.use_deepspeed and deepspeed_engine is not None:
                        with ultra_memory_cleanup():
                            pass
                    else:
                        optimizer.zero_grad()
                        if scaler is not None:
                            if finetune_config.precision_type == "bf16":
                                scaler = GradScaler(enabled=False)
                            else:
                                scaler = GradScaler()
                        
                        with ultra_memory_cleanup():
                            pass
                        
                        # Reduce batch size if needed
                        if finetune_config.batch_size > 1:
                            finetune_config.batch_size = 1
                            finetune_config.gradient_accumulation_steps *= 2
                            logger.info("Reduced fine-tuning batch size to 1")
                            
                            # Recreate dataloaders
                            train_dataloader = create_optimized_dataloader(
                                train_dataset, 1, finetune_config, shuffle=True
                            )
                            val_dataloader = create_optimized_dataloader(
                                val_dataset, 1, finetune_config, shuffle=False
                            )
                    
                    continue
                else:
                    raise e
        
        # Fine-tuning completion
        total_time = time.time() - finetuning_start
        
        logger.info("=" * 60)
        logger.info("✅ Fine-tuning completed!")
        logger.info(f"Base model: {base_model_id}")
        logger.info(f"Best loss: {best_loss:.4f} (epoch {best_epoch})")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Fine-tuning time: {total_time/60:.1f} minutes")
        logger.info(f"Final memory: {get_memory_usage()}")
        logger.info(f"Precision used: {finetune_config.precision_type.upper()}")
        logger.info(f"DeepSpeed used: {finetune_config.use_deepspeed}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Fine-tuning interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Fine-tuning failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return 1
    finally:
        with ultra_memory_cleanup():
            pass

if __name__ == "__main__":
    exit(main())