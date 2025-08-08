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

# Enhanced logging setup
def setup_logging():
    """Setup comprehensive logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('training_debug.log', mode='w')
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info("🔧 Debug logging initialized")
    return logger

logger = setup_logging()

def check_environment():
    """Check environment and dependencies."""
    logger.info("🔍 Checking environment...")
    
    try:
        # Python version
        logger.info(f"Python version: {sys.version}")
        
        # PyTorch version and setup
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"CUDA device count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                logger.info(f"GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
        
        # MPS availability
        if hasattr(torch.backends, 'mps'):
            logger.info(f"MPS available: {torch.backends.mps.is_available()}")
            logger.info(f"MPS built: {torch.backends.mps.is_built()}")
        
        # Check current directory and files
        cwd = Path.cwd()
        logger.info(f"Current directory: {cwd}")
        
        # Check for required files
        required_files = [
            "oasst1_data/oasst1_train.jsonl"
        ]
        
        for file_path in required_files:
            path = Path(file_path)
            exists = path.exists()
            logger.info(f"Required file {file_path}: {'✅ EXISTS' if exists else '❌ MISSING'}")
            if exists and path.is_file():
                size = path.stat().st_size / 1024**2
                logger.info(f"  Size: {size:.2f} MB")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Environment check failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

# Try to import required modules with fallbacks
try:
    from model_manager import ModelManager, ModelConfig, TrainingConfig, ModelMetadata
    from subword_transformer import SubwordTransformer, SubwordTokenizer
    IMPORTS_SUCCESSFUL = True
    logger.info("✅ Successfully imported required modules")
except ImportError as e:
    logger.warning(f"⚠️ Could not import custom modules: {e}")
    logger.info("Creating fallback implementations...")
    IMPORTS_SUCCESSFUL = False
    
    # Fallback implementations
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
    
    @dataclass
    class TrainingConfig:
        learning_rate: float = 5e-5
        weight_decay: float = 0.01
        batch_size: int = 8
        gradient_accumulation_steps: int = 4
        max_epochs: int = 10
        warmup_ratio: float = 0.1
        save_every: int = 1000
        eval_every: int = 500
        max_grad_norm: float = 1.0
        label_smoothing: float = 0.0
        beta1: float = 0.9
        beta2: float = 0.95
    
    @dataclass
    class ModelMetadata:
        model_name: str = "transformer"
        version: str = "v1.0"
        created_at: str = ""
        last_modified: str = ""
        model_config: ModelConfig = None
        training_config: TrainingConfig = None
        dataset_info: dict = None
        performance_metrics: dict = None
        model_size_mb: float = 0.0
        total_parameters: int = 0
        trainable_parameters: int = 0
        training_time_hours: float = 0.0
        epochs_trained: int = 0
        best_loss: float = float('inf')
        best_perplexity: float = float('inf')
        hardware_used: str = ""
        pytorch_version: str = ""
        cuda_version: str = None
        notes: str = ""
        tags: list = None
    
    class SimpleTokenizer:
        """Simple word-level tokenizer as fallback."""
        
        def __init__(self):
            self.vocab = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3, "<user>": 4, "<assistant>": 5}
            self.id_to_token = {v: k for k, v in self.vocab.items()}
            self.target_vocab_size = 10000
            self.trained = False
        
        def train_from_text(self, text, vocab_size=None, min_freq=1):
            """Train tokenizer on text."""
            if vocab_size:
                self.target_vocab_size = vocab_size
            
            # Simple word frequency counting
            word_freq = {}
            for line in text.split('\n'):
                words = line.lower().split()
                for word in words:
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # Add most frequent words to vocabulary
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            current_id = len(self.vocab)
            
            for word, freq in sorted_words:
                if freq >= min_freq and current_id < self.target_vocab_size and word not in self.vocab:
                    self.vocab[word] = current_id
                    self.id_to_token[current_id] = word
                    current_id += 1
            
            self.trained = True
            logger.info(f"Simple tokenizer trained with {len(self.vocab)} tokens")
        
        def encode(self, text):
            """Encode text to token IDs."""
            if not self.trained:
                raise ValueError("Tokenizer not trained")
            
            words = text.lower().split()
            token_ids = []
            for word in words:
                token_ids.append(self.vocab.get(word, self.vocab["<unk>"]))
            return token_ids
        
        def decode(self, token_ids):
            """Decode token IDs to text."""
            tokens = []
            for token_id in token_ids:
                if token_id in self.id_to_token:
                    token = self.id_to_token[token_id]
                    if token not in ["<pad>", "<bos>", "<eos>"]:
                        tokens.append(token)
            return " ".join(tokens)
        
        def vocab_size(self):
            return len(self.vocab)
    
    # Alias for compatibility
    SubwordTokenizer = SimpleTokenizer
    
    class SimpleTransformer(nn.Module):
        """Simple transformer model as fallback."""
        
        def __init__(self, config):
            super().__init__()
            self.config = config
            
            self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
            self.pos_embeddings = nn.Embedding(config.seq_length, config.hidden_size)
            
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_heads,
                dim_feedforward=config.hidden_size * 4,
                dropout=config.dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
            
            self.layer_norm = nn.LayerNorm(config.hidden_size)
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
            self.dropout = nn.Dropout(config.dropout)
            
            self.apply(self._init_weights)
        
        def _init_weights(self, module):
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
        def forward(self, input_ids, attention_mask=None):
            batch_size, seq_len = input_ids.shape
            
            # Create position IDs
            position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            
            # Embeddings
            token_embeddings = self.embeddings(input_ids)
            position_embeddings = self.pos_embeddings(position_ids)
            embeddings = self.dropout(token_embeddings + position_embeddings)
            
            # Create causal mask
            if attention_mask is None:
                attention_mask = torch.triu(torch.ones(seq_len, seq_len, device=input_ids.device), diagonal=1).bool()
            
            # Transformer
            hidden_states = self.transformer(embeddings, mask=attention_mask, is_causal=True)
            hidden_states = self.layer_norm(hidden_states)
            
            # Language modeling head
            logits = self.lm_head(hidden_states)
            
            return logits
    
    # Alias for compatibility
    SubwordTransformer = SimpleTransformer
    
    class ModelManager:
        """Simple model manager as fallback."""
        
        def __init__(self, save_dir):
            self.save_dir = Path(save_dir)
            self.save_dir.mkdir(exist_ok=True)
        
        def save_model(self, model, tokenizer, metadata, optimizer=None, scheduler=None, force_cpu_save=False):
            """Save model to disk."""
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_id = f"model_{timestamp}"
            model_path = self.save_dir / model_id
            model_path.mkdir(exist_ok=True)
            
            # Save model state dict
            if force_cpu_save:
                model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            else:
                model_state = model.state_dict()
            
            torch.save(model_state, model_path / "model.pth")
            
            # Save tokenizer
            tokenizer_data = {
                'vocab': tokenizer.vocab,
                'id_to_token': tokenizer.id_to_token,
                'vocab_size': tokenizer.vocab_size()
            }
            with open(model_path / "tokenizer.json", 'w') as f:
                json.dump(tokenizer_data, f, indent=2)
            
            # Save metadata
            metadata_dict = asdict(metadata) if hasattr(metadata, '__dict__') else metadata.__dict__
            with open(model_path / "metadata.json", 'w') as f:
                json.dump(metadata_dict, f, indent=2, default=str)
            
            logger.info(f"Model saved to: {model_path}")
            return model_id
        
        def validate_model(self, model_id):
            """Validate saved model."""
            model_path = self.save_dir / model_id
            if not model_path.exists():
                return {'valid': False, 'issues': ['Model directory does not exist']}
            
            issues = []
            if not (model_path / "model.pth").exists():
                issues.append("Model weights file missing")
            if not (model_path / "tokenizer.json").exists():
                issues.append("Tokenizer file missing")
            if not (model_path / "metadata.json").exists():
                issues.append("Metadata file missing")
            
            return {'valid': len(issues) == 0, 'issues': issues}
        
        def print_model_summary(self):
            """Print summary of saved models."""
            models = list(self.save_dir.glob("model_*"))
            logger.info(f"Found {len(models)} saved models in {self.save_dir}")
            for model_path in sorted(models):
                logger.info(f"  - {model_path.name}")

@contextmanager
def memory_cleanup():
    """Context manager for automatic memory cleanup."""
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()

def get_memory_usage():
    """Get current memory usage for monitoring."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        return f"CUDA: {allocated:.2f}GB allocated, {cached:.2f}GB cached"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        allocated = torch.mps.current_allocated_memory() / 1024**3
        return f"MPS: {allocated:.2f}GB allocated"
    else:
        return "CPU mode"

def setup_device():
    """Setup the best available device with conservative memory settings."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using device: CUDA ({torch.cuda.get_device_name()})")
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"CUDA Memory: {total_memory:.1f} GB")
        torch.cuda.set_per_process_memory_fraction(0.8)
        torch.cuda.empty_cache()
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        logger.info("Using device: MPS (Apple Silicon)")
        torch.mps.empty_cache()
    else:
        device = torch.device("cpu")
        logger.info("Using device: CPU")
        torch.set_num_threads(min(4, os.cpu_count() or 1))
    
    return device

device = setup_device()

class StableDataset(Dataset):
    """Stable dataset implementation with proper validation."""
    
    def __init__(self, texts: List[str], tokenizer, seq_length: int, max_sequences: int = 10000):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        
        logger.info(f"Creating stable dataset with seq_length={seq_length}...")
        
        vocab_size = tokenizer.vocab_size()
        pad_token_id = tokenizer.vocab.get("<pad>", 0)
        
        logger.info(f"Tokenizer vocab size: {vocab_size}")
        logger.info(f"Pad token ID: {pad_token_id}")
        
        # Pre-tokenize all texts and extract valid sequences
        self.sequences = []
        
        for text_idx, text in enumerate(texts[:max_sequences // 5]):
            if not text or not text.strip():
                continue
            
            try:
                tokens = tokenizer.encode(text.strip())
                
                if not tokens or len(tokens) < 10:
                    continue
                
                # Validate token IDs
                invalid_tokens = [t for t in tokens if t >= vocab_size or t < 0]
                if invalid_tokens:
                    # Replace invalid tokens with UNK
                    unk_id = tokenizer.vocab.get("<unk>", 1)
                    tokens = [t if 0 <= t < vocab_size else unk_id for t in tokens]
                
                # Extract sequences with sliding window
                step_size = seq_length // 2
                for start_pos in range(0, len(tokens) - seq_length, step_size):
                    if start_pos + seq_length + 1 <= len(tokens):
                        sequence = tokens[start_pos:start_pos + seq_length + 1]
                        
                        # Final validation
                        if len(sequence) == seq_length + 1 and all(0 <= t < vocab_size for t in sequence):
                            self.sequences.append(sequence)
                            
                            if len(self.sequences) >= max_sequences:
                                break
                
                if len(self.sequences) >= max_sequences:
                    break
                    
            except Exception as e:
                logger.warning(f"Error processing text {text_idx}: {e}")
                continue
        
        if not self.sequences:
            raise ValueError("No valid sequences created!")
        
        self.pad_token_id = pad_token_id
        self.vocab_size = vocab_size
        
        logger.info(f"Created {len(self.sequences):,} sequences")
        
        # Final validation
        invalid_count = 0
        for i, seq in enumerate(self.sequences):
            if len(seq) != seq_length + 1 or any(t >= vocab_size or t < 0 for t in seq):
                invalid_count += 1
        
        if invalid_count > 0:
            raise ValueError(f"Found {invalid_count} invalid sequences!")
        
        logger.info(f"All {len(self.sequences):,} sequences validated successfully")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        
        input_ids = torch.tensor(sequence[:-1], dtype=torch.long)
        target_ids = torch.tensor(sequence[1:], dtype=torch.long)
        
        return input_ids, target_ids

def load_and_process_data(data_path: str, max_samples: Optional[int] = None) -> List[str]:
    """Load and process OASST1 dataset."""
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    logger.info(f"Loading OASST1 training data from: {data_path}")
    
    role_tokens = {
        "prompter": "<user>",
        "assistant": "<assistant>"
    }
    
    texts = []
    processed_count = 0
    skipped_count = 0
    
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if max_samples and processed_count >= max_samples:
                    break
                
                try:
                    line = line.strip()
                    if not line:
                        continue
                    
                    record = json.loads(line)
                    
                    # Basic filtering
                    if record.get("deleted", False):
                        skipped_count += 1
                        continue
                    
                    if record.get("lang") != "en":
                        skipped_count += 1
                        continue
                    
                    text = record.get("text", "").strip()
                    if not text:
                        skipped_count += 1
                        continue
                    
                    # Filter by word count
                    word_count = len(text.split())
                    if word_count < 3 or word_count > 200:
                        skipped_count += 1
                        continue
                    
                    # Add role formatting
                    role = record.get("role", "").lower()
                    if role in role_tokens:
                        formatted_text = f"{role_tokens[role]} {text}"
                    else:
                        formatted_text = text
                    
                    texts.append(formatted_text)
                    processed_count += 1
                    
                    if processed_count % 1000 == 0:
                        logger.info(f"Processed {processed_count:,} samples...")
                    
                except (json.JSONDecodeError, KeyError, ValueError):
                    skipped_count += 1
                    continue
    
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise
    
    if not texts:
        raise ValueError(f"No valid text data found in {data_path}")
    
    logger.info(f"Successfully processed: {processed_count:,}")
    logger.info(f"Skipped: {skipped_count:,}")
    logger.info(f"Final dataset size: {len(texts):,} texts")
    
    return texts

class SimpleLRScheduler:
    """Simple learning rate scheduler."""
    
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, min_lr: float = 1e-7):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_step = 0
    
    def step(self, loss: Optional[float] = None, grad_norm: Optional[float] = None) -> float:
        """Update learning rate."""
        self.current_step += 1
        
        if self.current_step < self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * self.current_step / self.warmup_steps
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        
        # Apply to optimizer
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
    
    def state_dict(self):
        return {
            'current_step': self.current_step,
            'warmup_steps': self.warmup_steps,
            'total_steps': self.total_steps,
            'min_lr': self.min_lr,
            'base_lr': self.base_lr
        }
    
    def load_state_dict(self, state_dict):
        for key, value in state_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)

def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def calculate_perplexity(loss: float) -> float:
    """Calculate perplexity with overflow protection."""
    if math.isinf(loss) or math.isnan(loss) or loss > 20:
        return float('inf')
    return math.exp(min(loss, 20))

def calculate_token_accuracy(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = 0) -> Tuple[float, int]:
    """Calculate token-level accuracy."""
    with torch.no_grad():
        if logits.numel() == 0 or targets.numel() == 0:
            return 0.0, 0
        
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            return 0.0, 0
        
        flat_logits = logits.view(-1, logits.size(-1))
        flat_targets = targets.view(-1)
        
        predictions = torch.argmax(flat_logits, dim=-1)
        valid_mask = (flat_targets != ignore_index)
        
        if valid_mask.sum() == 0:
            return 0.0, 0
        
        valid_predictions = predictions[valid_mask]
        valid_targets = flat_targets[valid_mask]
        
        correct = (valid_predictions == valid_targets)
        total_correct = correct.sum().item()
        total_valid = valid_targets.numel()
        
        accuracy = total_correct / total_valid if total_valid > 0 else 0.0
        return accuracy, total_valid

def train_epoch(model, dataloader, criterion, optimizer, scheduler, epoch: int, 
                gradient_accumulation_steps: int = 1, max_grad_norm: float = 1.0,
                log_interval: int = 50, ignore_index: int = 0) -> Tuple[float, float, float]:
    """Training loop for one epoch."""
    model.train()
    total_loss = 0.0
    total_tokens = 0
    total_correct_tokens = 0
    total_valid_tokens = 0
    accumulation_steps = 0
    batch_times = []
    
    optimizer.zero_grad()
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        batch_start = time.time()
        
        try:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            # Validate inputs
            if torch.isnan(inputs).any() or torch.isinf(inputs).any():
                logger.warning(f"Invalid inputs at batch {batch_idx}, skipping")
                continue
            
            # Forward pass
            with torch.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                logits = model(inputs)
                loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            
            # Check for invalid loss
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"Invalid loss at batch {batch_idx}: {loss.item()}, skipping")
                optimizer.zero_grad()
                continue
            
            if loss.item() > 15.0:
                logger.warning(f"Very high loss at batch {batch_idx}: {loss.item()}")
                # Reduce learning rate
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.9
            
            # Calculate accuracy
            batch_accuracy, batch_valid_tokens = calculate_token_accuracy(logits, targets, ignore_index)
            total_correct_tokens += batch_accuracy * batch_valid_tokens
            total_valid_tokens += batch_valid_tokens
            
            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            accumulation_steps += 1
            
            if accumulation_steps >= gradient_accumulation_steps:
                # Gradient clipping
                if max_grad_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                else:
                    grad_norm = 0.0
                
                optimizer.step()
                current_lr = scheduler.step()
                optimizer.zero_grad()
                accumulation_steps = 0
            
            # Statistics
            batch_loss = loss.item() * gradient_accumulation_steps
            if not math.isnan(batch_loss) and not math.isinf(batch_loss):
                total_loss += batch_loss * inputs.size(0)
                total_tokens += targets.numel()
            
            batch_times.append(time.time() - batch_start)
            
            # Logging
            if batch_idx % log_interval == 0 and batch_idx > 0:
                current_loss = total_loss / max(total_tokens, 1)
                current_accuracy = total_correct_tokens / max(total_valid_tokens, 1)
                avg_batch_time = sum(batch_times[-log_interval:]) / len(batch_times[-log_interval:])
                tokens_per_sec = targets.numel() / avg_batch_time if avg_batch_time > 0 else 0
                
                logger.info(f"Epoch {epoch} | Batch {batch_idx}/{len(dataloader)} | "
                           f"Loss: {current_loss:.4f} | Acc: {current_accuracy:.3f} ({current_accuracy*100:.1f}%) | "
                           f"LR: {current_lr:.2e} | Speed: {tokens_per_sec:.0f} tok/s")
            
            # Memory cleanup
            if batch_idx % 10 == 0:
                with memory_cleanup():
                    pass
        
        except RuntimeError as e:
            error_str = str(e).lower()
            if "out of memory" in error_str:
                logger.error(f"OOM at batch {batch_idx}. Clearing cache and skipping...")
                optimizer.zero_grad()
                with memory_cleanup():
                    pass
                continue
            else:
                logger.error(f"Runtime error at batch {batch_idx}: {e}")
                raise e
    
    # Calculate final metrics
    avg_loss = total_loss / max(total_tokens, 1)
    avg_accuracy = total_correct_tokens / max(total_valid_tokens, 1)
    avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0.0
    
    # Validate final metrics
    if math.isnan(avg_loss) or math.isinf(avg_loss):
        avg_loss = 100.0
    
    if math.isnan(avg_accuracy) or math.isinf(avg_accuracy):
        avg_accuracy = 0.0
    
    return avg_loss, avg_accuracy, avg_batch_time

def generate_sample_text(model, tokenizer, prompt: str = "<user> Hello", 
                        max_length: int = 20, temperature: float = 0.8) -> str:
    """Generate sample text for evaluation."""
    model.eval()
    
    try:
        with torch.no_grad():
            input_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
            generated = input_ids.clone()
            
            for step in range(max_length):
                if generated.size(1) >= 100:  # Limit context
                    break
                
                try:
                    logits = model(generated)
                    
                    if torch.isnan(logits).any() or torch.isinf(logits).any():
                        break
                    
                    next_token_logits = logits[0, -1, :] / temperature
                    probs = F.softmax(next_token_logits, dim=-1)
                    
                    if torch.isnan(probs).any():
                        break
                    
                    next_token = torch.multinomial(probs, 1)
                    generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
                    
                    # Stop on end token or padding
                    if next_token.item() in [tokenizer.vocab.get("</s>", -1), tokenizer.vocab.get("<pad>", 0)]:
                        break
                        
                except Exception:
                    break
            
            response_ids = generated[0][input_ids.size(1):].tolist()
            response = tokenizer.decode(response_ids)
            return response.strip()
    
    except Exception as e:
        logger.warning(f"Error generating sample: {e}")
        return "Error during generation"
    finally:
        model.train()

def get_conservative_config():
    """Get conservative configuration for stable training."""
    
    if device.type == 'cuda':
        model_config = ModelConfig(
            vocab_size=80000,
            hidden_size=2048,
            num_layers=24,
            num_heads=16,
            seq_length=1024,
            dropout=0.1,
            model_type="SimpleTransformer",
            tokenizer_type="simple"
        )
        batch_size = 16
        max_samples = 8000
        
    elif device.type == 'mps':
        model_config = ModelConfig(
            vocab_size=4000,
            hidden_size=256,
            num_layers=4,
            num_heads=4,
            seq_length=128,
            dropout=0.1,
            model_type="SimpleTransformer",
            tokenizer_type="simple"
        )
        batch_size = 8
        max_samples = 2000
        
    else:  # CPU
        model_config = ModelConfig(
            vocab_size=2000,
            hidden_size=128,
            num_layers=3,
            num_heads=2,
            seq_length=64,
            dropout=0.1,
            model_type="SimpleTransformer",
            tokenizer_type="simple"
        )
        batch_size = 4
        max_samples = 1000
    
    training_config = TrainingConfig(
        learning_rate=1e-5,
        weight_decay=0.01,
        batch_size=batch_size,
        gradient_accumulation_steps=4,
        max_epochs=20,
        warmup_ratio=0.1,
        save_every=500,
        eval_every=250,
        max_grad_norm=1.0,
        label_smoothing=0.0,
        beta1=0.9,
        beta2=0.999
    )
    
    return model_config, training_config, max_samples

def validate_training_setup():
    """Validate that required files exist."""
    required_files = ["oasst1_data/oasst1_train.jsonl"]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        logger.error("Missing required files:")
        for file_path in missing_files:
            logger.error(f"  - {file_path}")
        return False
    
    return True

def save_model_safely(model_manager, model, tokenizer, metadata, optimizer=None, scheduler=None):
    """Save model with error handling."""
    try:
        logger.info("💾 Saving model...")
        
        with memory_cleanup():
            pass
        
        model_id = model_manager.save_model(
            model=model, 
            tokenizer=tokenizer, 
            metadata=metadata, 
            optimizer=optimizer, 
            scheduler=scheduler,
            force_cpu_save=True
        )
        
        logger.info(f"✅ Model saved successfully: {model_id}")
        return model_id
        
    except Exception as save_error:
        logger.error(f"❌ Failed to save model: {save_error}")
        return None

def evaluate_model(model, dataloader, criterion, tokenizer, max_batches: int = 10):
    """Evaluate model performance."""
    model.eval()
    
    total_loss = 0.0
    total_tokens = 0
    total_correct_tokens = 0
    total_valid_tokens = 0
    batch_count = 0
    
    ignore_index = tokenizer.vocab.get("<pad>", 0)
    
    try:
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                if batch_idx >= max_batches:
                    break
                
                try:
                    inputs = inputs.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    
                    if torch.isnan(inputs).any() or torch.isinf(inputs).any():
                        continue
                    
                    with torch.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                        logits = model(inputs)
                        loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
                    
                    if torch.isnan(loss) or torch.isinf(loss):
                        continue
                    
                    batch_accuracy, batch_valid_tokens = calculate_token_accuracy(logits, targets, ignore_index)
                    
                    total_loss += loss.item() * inputs.size(0)
                    total_tokens += targets.numel()
                    total_correct_tokens += batch_accuracy * batch_valid_tokens
                    total_valid_tokens += batch_valid_tokens
                    batch_count += 1
                    
                except Exception:
                    continue
    
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
    
    finally:
        model.train()
    
    # Calculate metrics
    if total_tokens > 0:
        avg_loss = total_loss / total_tokens
        perplexity = calculate_perplexity(avg_loss)
    else:
        avg_loss = float('inf')
        perplexity = float('inf')
    
    if total_valid_tokens > 0:
        accuracy = total_correct_tokens / total_valid_tokens
    else:
        accuracy = 0.0
    
    return {
        'avg_loss': avg_loss,
        'accuracy': accuracy,
        'perplexity': perplexity,
        'batch_count': batch_count
    }

def create_tokenizer_safely(vocab_size):
    """Create tokenizer with proper error handling."""
    try:
        # Try to create with the expected interface first
        if IMPORTS_SUCCESSFUL:
            # Check if the imported SubwordTokenizer expects vocab_size parameter
            try:
                tokenizer = SubwordTokenizer(vocab_size=vocab_size)
                return tokenizer
            except TypeError:
                # If that fails, try without parameters
                try:
                    tokenizer = SubwordTokenizer()
                    return tokenizer
                except Exception:
                    # Fall back to simple tokenizer
                    logger.warning("SubwordTokenizer creation failed, using SimpleTokenizer fallback")
                    return SimpleTokenizer()
        else:
            # Use fallback implementation
            return SimpleTokenizer()
    except Exception as e:
        logger.warning(f"Tokenizer creation failed: {e}, using SimpleTokenizer fallback")
        return SimpleTokenizer()

def main():
    """Main training function."""
    
    logger.info("🚀 Starting OASST1 Transformer Training")
    logger.info("=" * 80)
    logger.info(f"Initial memory: {get_memory_usage()}")
    
    # Check environment
    if not check_environment():
        logger.error("❌ Environment check failed")
        return 1
    
    # Validate setup
    if not validate_training_setup():
        logger.error("❌ Training setup validation failed!")
        return 1
    
    # Get configuration
    model_config, training_config, max_samples = get_conservative_config()
    
    logger.info(f"Configuration:")
    logger.info(f"  Model: {model_config.hidden_size}x{model_config.num_layers}")
    logger.info(f"  Vocab size: {model_config.vocab_size}")
    logger.info(f"  Batch size: {training_config.batch_size}")
    logger.info(f"  Max samples: {max_samples}")
    logger.info(f"  Learning rate: {training_config.learning_rate}")
    
    # Initialize model manager
    model_manager = ModelManager("models")
    
    try:
        # Load data
        logger.info("📚 Loading OASST1 dataset...")
        texts = load_and_process_data("oasst1_data/oasst1_train.jsonl", max_samples)
        
        if len(texts) == 0:
            raise ValueError("No training data loaded!")
        
        logger.info(f"Memory after data loading: {get_memory_usage()}")
        
        # Create tokenizer
        logger.info("🔤 Training tokenizer...")
        tokenizer = create_tokenizer_safely(model_config.vocab_size)
        
        # Use subset for tokenizer training
        sample_size = min(1000, len(texts))
        sample_texts = texts[:sample_size]
        all_text = "\n".join(sample_texts)
        
        tokenizer.train_from_text(all_text, vocab_size=model_config.vocab_size, min_freq=2)
        actual_vocab_size = tokenizer.vocab_size()
        model_config.vocab_size = actual_vocab_size
        
        logger.info(f"Tokenizer trained - Vocabulary size: {actual_vocab_size:,}")
        
        # Test tokenizer
        test_text = "Hello, this is a test!"
        test_tokens = tokenizer.encode(test_text)
        test_decoded = tokenizer.decode(test_tokens)
        
        logger.info(f"Tokenizer test:")
        logger.info(f"  Original: {test_text}")
        logger.info(f"  Tokens: {test_tokens}")
        logger.info(f"  Decoded: {test_decoded}")
        
        # Create dataset
        logger.info("📦 Creating dataset...")
        dataset = StableDataset(
            texts, 
            tokenizer, 
            model_config.seq_length,
            max_sequences=min(15000, len(texts) * 3)
        )
        
        logger.info(f"Memory after dataset creation: {get_memory_usage()}")
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=training_config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            drop_last=True
        )
        
        # Create evaluation dataloader
        eval_dataset_size = min(500, len(dataset) // 10)
        eval_indices = torch.randperm(len(dataset))[:eval_dataset_size]
        eval_dataset = torch.utils.data.Subset(dataset, eval_indices)
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=training_config.batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False
        )
        
        logger.info(f"Dataset: {len(dataset):,} sequences, {len(dataloader):,} batches/epoch")
        logger.info(f"Eval dataset: {len(eval_dataset):,} sequences")
        
        # Initialize model
        logger.info("🧠 Initializing transformer model...")
        with memory_cleanup():
            model = SubwordTransformer(model_config)
            model = model.to(device)
        
        total_params, trainable_params = count_parameters(model)
        model_size_mb = total_params * 4 / 1024**2
        logger.info(f"Model parameters: {total_params:,} (~{model_size_mb:.1f}MB)")
        logger.info(f"Memory after model creation: {get_memory_usage()}")
        
        # Training components
        optimizer = optim.AdamW(
            model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
            betas=(training_config.beta1, training_config.beta2)
        )
        
        pad_token_id = tokenizer.vocab.get("<pad>", 0)
        criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
        
        total_steps = len(dataloader) * training_config.max_epochs // training_config.gradient_accumulation_steps
        warmup_steps = int(total_steps * training_config.warmup_ratio)
        scheduler = SimpleLRScheduler(optimizer, warmup_steps, total_steps)
        
        logger.info(f"Training setup: {total_steps:,} steps, {warmup_steps:,} warmup")
        logger.info(f"Memory before training: {get_memory_usage()}")
        
        # Training loop
        logger.info("🚀 Starting training...")
        training_start = time.time()
        best_loss = float('inf')
        best_accuracy = 0.0
        models_saved = 0
        
        for epoch in range(1, training_config.max_epochs + 1):
            epoch_start = time.time()
            
            try:
                logger.info(f"Starting epoch {epoch}/{training_config.max_epochs}")
                
                # Train epoch
                avg_loss, avg_accuracy, avg_batch_time = train_epoch(
                    model, dataloader, criterion, optimizer, scheduler, epoch,
                    training_config.gradient_accumulation_steps,
                    training_config.max_grad_norm,
                    ignore_index=pad_token_id
                )
                
                # Validate results
                if math.isnan(avg_loss) or math.isinf(avg_loss):
                    logger.warning(f"Invalid loss in epoch {epoch}: {avg_loss}")
                    # Try to recover
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= 0.5
                    continue
                
                perplexity = calculate_perplexity(avg_loss)
                epoch_time = time.time() - epoch_start
                
                # Evaluation
                eval_metrics = None
                if epoch % 3 == 0 or epoch == 1 or epoch == training_config.max_epochs:
                    logger.info("📊 Running evaluation...")
                    eval_metrics = evaluate_model(model, eval_dataloader, criterion, tokenizer)
                    
                    if not math.isnan(eval_metrics['avg_loss']) and not math.isinf(eval_metrics['avg_loss']):
                        logger.info(f"Evaluation results:")
                        logger.info(f"  Loss: {eval_metrics['avg_loss']:.4f}")
                        logger.info(f"  Accuracy: {eval_metrics['accuracy']:.3f} ({eval_metrics['accuracy']*100:.1f}%)")
                        logger.info(f"  Perplexity: {eval_metrics['perplexity']:.2f}")
                
                # Logging
                logger.info("=" * 60)
                logger.info(f"Epoch {epoch}/{training_config.max_epochs} Summary:")
                logger.info(f"  Train Loss: {avg_loss:.4f} | Train Accuracy: {avg_accuracy:.3f} ({avg_accuracy*100:.1f}%)")
                logger.info(f"  Perplexity: {perplexity:.2f}")
                if eval_metrics and not math.isnan(eval_metrics['avg_loss']):
                    logger.info(f"  Eval Loss: {eval_metrics['avg_loss']:.4f} | Eval Accuracy: {eval_metrics['accuracy']:.3f}")
                logger.info(f"  Time: {epoch_time:.1f}s | {get_memory_usage()}")
                
                # Update best metrics
                is_best_loss = avg_loss < best_loss and not math.isinf(avg_loss)
                is_best_accuracy = avg_accuracy > best_accuracy
                
                if is_best_loss:
                    best_loss = avg_loss
                    logger.info(f"🏆 New best loss: {best_loss:.4f}")
                
                if is_best_accuracy:
                    best_accuracy = avg_accuracy
                    logger.info(f"🏆 New best accuracy: {best_accuracy:.3f}")
                
                # Sample generation
                if epoch % 5 == 0:
                    try:
                        sample = generate_sample_text(model, tokenizer, "<user> Hello")
                        logger.info(f"  Sample: <user> Hello → {sample}")
                    except Exception:
                        pass
                
                # Model saving
                should_save = (
                    is_best_loss or 
                    is_best_accuracy or
                    epoch % 5 == 0 or 
                    epoch == training_config.max_epochs
                )
                
                if should_save:
                    performance_metrics = {
                        "train_loss": float(avg_loss),
                        "train_accuracy": float(avg_accuracy),
                        "train_perplexity": float(perplexity),
                        "epoch": int(epoch),
                        "learning_rate": float(optimizer.param_groups[0]['lr']),
                        "is_best_loss": is_best_loss,
                        "is_best_accuracy": is_best_accuracy,
                    }
                    
                    if eval_metrics and not math.isnan(eval_metrics['avg_loss']):
                        performance_metrics.update({
                            "eval_loss": float(eval_metrics['avg_loss']),
                            "eval_accuracy": float(eval_metrics['accuracy']),
                            "eval_perplexity": float(eval_metrics['perplexity']),
                        })
                    
                    metadata = ModelMetadata(
                        model_name="OASST1_Transformer",
                        version=f"v1.0_epoch_{epoch}",
                        created_at=datetime.now().isoformat(),
                        last_modified=datetime.now().isoformat(),
                        model_config=model_config,
                        training_config=training_config,
                        dataset_info={
                            "name": "OpenAssistant OASST1",
                            "source": "oasst1_train.jsonl",
                            "num_samples": len(texts),
                            "vocab_size": int(actual_vocab_size),
                            "seq_length": model_config.seq_length,
                            "train_sequences": len(dataset),
                            "eval_sequences": len(eval_dataset),
                        },
                        performance_metrics=performance_metrics,
                        model_size_mb=float(model_size_mb),
                        total_parameters=int(total_params),
                        trainable_parameters=int(trainable_params),
                        epochs_trained=int(epoch),
                        best_loss=float(best_loss),
                        best_perplexity=float(calculate_perplexity(best_loss)),
                        hardware_used=f"{device.type.upper()}",
                        pytorch_version=torch.__version__,
                        notes=f"OASST1 transformer training epoch {epoch}. "
                              f"Train: loss={avg_loss:.4f}, acc={avg_accuracy:.3f}. "
                              f"Best: loss={best_loss:.4f}, acc={best_accuracy:.3f}.",
                        tags=["oasst1", "transformer", f"epoch_{epoch}"] + 
                             (["best_loss"] if is_best_loss else []) +
                             (["best_accuracy"] if is_best_accuracy else [])
                    )
                    
                    model_id = save_model_safely(model_manager, model, tokenizer, metadata, optimizer, scheduler)
                    if model_id:
                        models_saved += 1
                        logger.info(f"💾 Model saved: {model_id} (#{models_saved})")
                
                logger.info("=" * 60)
                
                # Memory cleanup
                with memory_cleanup():
                    pass
                
            except Exception as e:
                logger.error(f"❌ Error in epoch {epoch}: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                
                # Try to recover from OOM
                if "out of memory" in str(e).lower():
                    logger.info("Attempting OOM recovery...")
                    with memory_cleanup():
                        pass
                    continue
                else:
                    raise e
        
        # Training completion
        total_time = time.time() - training_start
        logger.info("=" * 80)
        logger.info("✅ Training completed successfully!")
        logger.info(f"🎯 Best loss: {best_loss:.4f}")
        logger.info(f"🎯 Best accuracy: {best_accuracy:.3f} ({best_accuracy*100:.1f}%)")
        logger.info(f"🎯 Best perplexity: {calculate_perplexity(best_loss):.2f}")
        logger.info(f"💾 Models saved: {models_saved}")
        logger.info(f"⏱️  Training time: {total_time/3600:.2f} hours")
        logger.info(f"💾 Final memory: {get_memory_usage()}")
        
        # Final save if no models saved
        if models_saved == 0:
            logger.warning("⚠️ No models saved! Performing final save...")
            
            final_metadata = ModelMetadata(
                model_name="OASST1_Transformer_FINAL",
                version="v1.0_FINAL",
                created_at=datetime.now().isoformat(),
                model_config=model_config,
                performance_metrics={
                    "final_train_loss": float(avg_loss) if 'avg_loss' in locals() else float('inf'),
                    "final_train_accuracy": float(avg_accuracy) if 'avg_accuracy' in locals() else 0.0,
                    "best_train_loss": float(best_loss),
                    "best_train_accuracy": float(best_accuracy),
                },
                notes="Final save after training completion",
                tags=["oasst1", "final"]
            )
            
            final_id = save_model_safely(model_manager, model, tokenizer, final_metadata)
            if final_id:
                models_saved += 1
                logger.info(f"✅ Final save successful: {final_id}")
        
        # Print model summary
        logger.info("\n" + "=" * 80)
        logger.info("📊 MODEL SUMMARY")
        model_manager.print_model_summary()
        logger.info("=" * 80)
        
        return 0 if models_saved > 0 else 1
        
    except KeyboardInterrupt:
        logger.info("❌ Training interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return 1
    finally:
        with memory_cleanup():
            pass

if __name__ == "__main__":
    exit(main())