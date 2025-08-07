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
    """Comprehensive environment check."""
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
        logger.info(f"Directory contents: {list(cwd.iterdir())}")
        
        # Check for required files
        required_files = [
            "model_manager.py",
            "subword_transformer.py",
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

def test_imports():
    """Test all required imports."""
    logger.info("📦 Testing imports...")
    
    import_results = {}
    
    # Test basic imports
    basic_imports = {
        'torch': lambda: __import__('torch'),
        'torch.nn': lambda: __import__('torch.nn'),
        'torch.optim': lambda: __import__('torch.optim'),
    }
    
    for name, import_func in basic_imports.items():
        try:
            import_func()
            import_results[name] = "✅ SUCCESS"
            logger.info(f"  {name}: ✅")
        except Exception as e:
            import_results[name] = f"❌ FAILED: {e}"
            logger.error(f"  {name}: ❌ {e}")
    
    # Test custom imports
    custom_imports = {
        'model_manager': 'model_manager',
        'subword_transformer': 'subword_transformer'
    }
    
    for name, module in custom_imports.items():
        try:
            if module == 'model_manager':
                from model_manager import ModelManager, ModelConfig, TrainingConfig, ModelMetadata
                import_results[name] = "✅ SUCCESS"
                logger.info(f"  {name}: ✅")
            elif module == 'subword_transformer':
                from subword_transformer import SubwordTransformer, SubwordTokenizer
                import_results[name] = "✅ SUCCESS"
                logger.info(f"  {name}: ✅")
        except Exception as e:
            import_results[name] = f"❌ FAILED: {e}"
            logger.error(f"  {name}: ❌ {e}")
            logger.error(f"    Traceback: {traceback.format_exc()}")
    
    # Check if all critical imports succeeded
    critical_failed = [name for name, result in import_results.items() if "FAILED" in result]
    
    if critical_failed:
        logger.error(f"❌ Critical imports failed: {critical_failed}")
        return False, import_results
    else:
        logger.info("✅ All imports successful")
        return True, import_results

def test_data_loading():
    """Test data loading functionality."""
    logger.info("📚 Testing data loading...")
    
    data_path = Path("oasst1_data/oasst1_train.jsonl")
    
    if not data_path.exists():
        logger.error(f"❌ Data file not found: {data_path}")
        return False
    
    try:
        # Try to read first few lines
        line_count = 0
        valid_lines = 0
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line_count += 1
                if i >= 10:  # Just test first 10 lines
                    break
                
                try:
                    line = line.strip()
                    if line:
                        record = json.loads(line)
                        if record.get("text") and record.get("lang") == "en":
                            valid_lines += 1
                except:
                    continue
        
        logger.info(f"  Total lines tested: {line_count}")
        logger.info(f"  Valid lines: {valid_lines}")
        
        if valid_lines > 0:
            logger.info("✅ Data loading test passed")
            return True
        else:
            logger.error("❌ No valid data found")
            return False
            
    except Exception as e:
        logger.error(f"❌ Data loading test failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def test_tokenizer():
    """Test subword tokenizer functionality."""
    logger.info("🔤 Testing subword tokenizer...")
    
    try:
        from subword_transformer import SubwordTokenizer
        
        # Create a simple tokenizer
        tokenizer = SubwordTokenizer()
        
        # Test with small vocabulary
        test_text = "hello world this is a test of subword tokenization"
        tokenizer.train_from_text(test_text, vocab_size=100, min_freq=1)
        
        # Test encoding/decoding
        test_sentence = "hello test"
        encoded = tokenizer.encode(test_sentence)
        decoded = tokenizer.decode(encoded)
        
        logger.info(f"  Test sentence: '{test_sentence}'")
        logger.info(f"  Encoded: {encoded}")
        logger.info(f"  Decoded: '{decoded}'")
        logger.info(f"  Vocabulary size: {tokenizer.vocab_size()}")
        logger.info(f"  Number of merges: {len(tokenizer.merges)}")
        
        if encoded and decoded:
            logger.info("✅ Subword tokenizer test passed")
            return True
        else:
            logger.error("❌ Tokenizer encoding/decoding failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Tokenizer test failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def run_comprehensive_debug():
    """Run comprehensive debugging."""
    logger.info("🚀 Starting comprehensive debug check...")
    logger.info("=" * 70)
    
    # Step 1: Environment check
    if not check_environment():
        logger.error("❌ Environment check failed - stopping")
        return False
    
    # Step 2: Import test
    imports_ok, import_results = test_imports()
    if not imports_ok:
        logger.error("❌ Import test failed - stopping")
        logger.error("Failed imports:")
        for name, result in import_results.items():
            if "FAILED" in result:
                logger.error(f"  {name}: {result}")
        return False
    
    # Step 3: Data loading test
    if not test_data_loading():
        logger.error("❌ Data loading test failed - stopping")
        return False
    
    # Step 4: Tokenizer test
    if not test_tokenizer():
        logger.error("❌ Tokenizer test failed - stopping")
        return False
    
    logger.info("=" * 70)
    logger.info("✅ All debug checks passed! Proceeding with training.")
    logger.info("=" * 70)
    return True

# Import shared components with error handling
try:
    from model_manager import ModelManager, ModelConfig, TrainingConfig, ModelMetadata
    from subword_transformer import SubwordTransformer, SubwordTokenizer
    IMPORTS_SUCCESSFUL = True
    logger.info("✅ Successfully imported required modules")
except ImportError as e:
    logger.error(f"❌ Import error: {e}")
    logger.error("Running debug checks to identify the issue...")
    IMPORTS_SUCCESSFUL = False

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
            torch.mps.synchronize()
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
    """Setup the best available device with very conservative memory settings."""
    # Set very conservative memory limits FIRST
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        logger.info("Using device: MPS (Apple Silicon)")
        # Very conservative for MPS - it has memory issues
        torch.mps.set_per_process_memory_fraction(0.7)
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using device: CUDA ({torch.cuda.get_device_name()})")
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"CUDA Memory: {total_memory:.1f} GB")
        # Very conservative memory settings
        torch.cuda.set_per_process_memory_fraction(0.5)  # Only use 50%
        torch.cuda.empty_cache()
        # Disable optimizations that use more memory
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    else:
        device = torch.device("cpu")
        logger.info("Using device: CPU")
        torch.set_num_threads(min(2, os.cpu_count() // 4 or 1))  # Very conservative
        torch.set_num_interop_threads(1)
    
    return device

device = setup_device()

class FixedSubwordDataset(Dataset):
    """Fixed memory-efficient dataset that pre-tokenizes and stores sequences properly."""
    
    def __init__(self, texts: List[str], tokenizer: SubwordTokenizer, seq_length: int, 
                 overlap_ratio: float = 0.5, min_seq_length: int = 16, max_sequences: int = 50000):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.min_seq_length = min_seq_length
        
        logger.info("Creating fixed subword-level dataset...")
        
        # Pre-tokenize all texts and extract valid sequences
        self.sequences = []
        valid_texts = 0
        
        for text_idx, text in enumerate(texts[:max_sequences // 10]):  # Limit input texts
            if not text or not text.strip():
                continue
                
            # Tokenize the text
            tokens = tokenizer.encode(text.strip())
            if len(tokens) < self.min_seq_length:
                continue
            
            # Extract overlapping sequences from this text
            step_size = max(1, int(seq_length * overlap_ratio))
            sequences_from_text = 0
            max_sequences_per_text = 15  # Limit sequences per text
            
            for start_pos in range(0, len(tokens) - seq_length, step_size):
                if start_pos + seq_length + 1 <= len(tokens):
                    # Extract the sequence (input + target)
                    sequence = tokens[start_pos:start_pos + seq_length + 1]
                    self.sequences.append(sequence)
                    sequences_from_text += 1
                    
                    if sequences_from_text >= max_sequences_per_text:
                        break
                    
                    # Stop if we have enough sequences total
                    if len(self.sequences) >= max_sequences:
                        break
            
            if sequences_from_text > 0:
                valid_texts += 1
            
            # Progress logging
            if valid_texts % 100 == 0 and valid_texts > 0:
                logger.info(f"Processed {valid_texts} texts, created {len(self.sequences):,} sequences")
            
            # Stop if we have enough sequences
            if len(self.sequences) >= max_sequences:
                break
        
        if not self.sequences:
            raise ValueError("No valid sequences created from the input texts!")
        
        # Add padding token for consistency
        self.pad_token_id = tokenizer.vocab.get("<pad>", 0)
        
        logger.info(f"✅ Created {len(self.sequences):,} sequences from {valid_texts} texts")
        logger.info(f"✅ Sequence length: {seq_length}, Min length: {min_seq_length}")
        logger.info(f"✅ Memory usage: ~{len(self.sequences) * (seq_length + 1) * 4 / 1024**2:.2f} MB")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        
        # Ensure sequence is the right length
        if len(sequence) < self.seq_length + 1:
            # Pad if too short
            sequence = sequence + [self.pad_token_id] * (self.seq_length + 1 - len(sequence))
        elif len(sequence) > self.seq_length + 1:
            # Truncate if too long
            sequence = sequence[:self.seq_length + 1]
        
        # Split into input and target
        input_ids = torch.tensor(sequence[:-1], dtype=torch.long)
        target_ids = torch.tensor(sequence[1:], dtype=torch.long)
        
        return input_ids, target_ids

def load_and_process_data(data_path: str, max_samples: Optional[int] = None) -> List[str]:
    """Enhanced data loading for OASST2 dataset with aggressive memory management."""
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    logger.info(f"Loading OASST2 training data from: {data_path}")
    
    # OASST2 specific role tokens
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
                    
                    # Skip deleted entries
                    if record.get("deleted", False):
                        skipped_count += 1
                        continue
                    
                    # Filter for English language
                    if record.get("lang") != "en":
                        skipped_count += 1
                        continue
                    
                    # Skip entries that failed review
                    if not record.get("review_result", True):
                        skipped_count += 1
                        continue
                    
                    # Only use entries from ready conversation trees
                    if record.get("tree_state") != "ready_for_export":
                        skipped_count += 1
                        continue
                    
                    # Extract text content
                    text = record.get("text", "").strip()
                    if not text:
                        skipped_count += 1
                        continue
                    
                    # Filter very short texts and very long texts
                    word_count = len(text.split())
                    if word_count < 3:
                        skipped_count += 1
                        continue
                    
                    # Truncate very long texts to save memory
                    if word_count > 150:  # Slightly longer for subword tokenization
                        text = ' '.join(text.split()[:150])
                    
                    # Get role and add appropriate tokens
                    role = record.get("role", "").lower()
                    
                    # Add role-specific formatting
                    if role in role_tokens:
                        formatted_text = f"{role_tokens[role]} {text}"
                    else:
                        formatted_text = text
                    
                    texts.append(formatted_text)
                    processed_count += 1
                    
                    # Progress logging with aggressive memory cleanup
                    if processed_count % 1000 == 0:
                        logger.info(f"Processed {processed_count:,} samples... {get_memory_usage()}")
                        gc.collect()  # Frequent cleanup
                    
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
    logger.info(f"{get_memory_usage()}")
    
    return texts

class AdaptiveLRScheduler:
    """Simplified learning rate scheduler."""
    
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, 
                 min_lr: float = 1e-6, decay_type: str = "cosine"):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_step = 0
        self.decay_type = decay_type
        
    def step(self, loss: Optional[float] = None) -> float:
        """Update learning rate."""
        self.current_step += 1
        
        # Calculate learning rate
        if self.current_step < self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * self.current_step / self.warmup_steps
        else:
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            
            if self.decay_type == "cosine":
                lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
            else:  # linear
                lr = self.base_lr * (1 - progress) + self.min_lr * progress
        
        # Update optimizer
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
    
    def state_dict(self):
        return {
            'current_step': self.current_step,
            'warmup_steps': self.warmup_steps,
            'total_steps': self.total_steps,
            'min_lr': self.min_lr,
            'base_lr': self.base_lr,
            'decay_type': self.decay_type
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
    return math.exp(min(loss, 20))

def calculate_token_accuracy(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = 0) -> Tuple[float, int]:
    """
    FIXED: Calculate token-level accuracy for language modeling.
    
    The key fix is to flatten the tensors BEFORE comparison and ensure proper masking.
    
    Args:
        logits: Model predictions [batch_size, seq_len, vocab_size]
        targets: Ground truth tokens [batch_size, seq_len]
        ignore_index: Token ID to ignore (usually padding token)
    
    Returns:
        Tuple of (accuracy, total_valid_tokens)
    """
    with torch.no_grad():
        # Flatten logits and targets for easier processing
        flat_logits = logits.view(-1, logits.size(-1))  # [batch_size * seq_len, vocab_size]
        flat_targets = targets.view(-1)  # [batch_size * seq_len]
        
        # Get predictions - argmax along vocab dimension
        predictions = torch.argmax(flat_logits, dim=-1)  # [batch_size * seq_len]
        
        # Create mask for valid tokens (not padding/ignore tokens)
        valid_mask = (flat_targets != ignore_index)
        
        # Only compare predictions for valid tokens
        valid_predictions = predictions[valid_mask]
        valid_targets = flat_targets[valid_mask]
        
        # Calculate correct predictions
        if valid_targets.numel() == 0:  # No valid tokens
            return 0.0, 0
        
        correct = (valid_predictions == valid_targets)
        total_correct = correct.sum().item()
        total_valid = valid_targets.numel()
        
        accuracy = total_correct / total_valid if total_valid > 0 else 0.0
        
        # Debug information for first few batches (remove after confirming fix)
        if hasattr(calculate_token_accuracy, '_debug_count'):
            calculate_token_accuracy._debug_count += 1
        else:
            calculate_token_accuracy._debug_count = 1
            
        if calculate_token_accuracy._debug_count <= 3:  # Only log first few batches
            logger.debug(f"[ACCURACY DEBUG] Batch {calculate_token_accuracy._debug_count}:")
            logger.debug(f"  Logits shape: {logits.shape}")
            logger.debug(f"  Targets shape: {targets.shape}")
            logger.debug(f"  Valid tokens: {total_valid}")
            logger.debug(f"  Correct predictions: {total_correct}")
            logger.debug(f"  Accuracy: {accuracy:.4f}")
            logger.debug(f"  Sample predictions: {valid_predictions[:10].tolist()}")
            logger.debug(f"  Sample targets: {valid_targets[:10].tolist()}")
            logger.debug(f"  Sample matches: {correct[:10].tolist()}")
        
        return accuracy, total_valid

def train_epoch(model, dataloader, criterion, optimizer, scheduler, epoch: int, 
                gradient_accumulation_steps: int = 1, max_grad_norm: float = 1.0,
                log_interval: int = 25, ignore_index: int = 0) -> Tuple[float, float, float]:
    """Ultra memory-optimized training loop with FIXED accuracy calculation."""
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
            # Move to device
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            # Forward pass - be very careful with mixed precision
            if device.type == 'cuda':
                # Use bfloat16 for better stability
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
                    logits = model(inputs)
                    loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            else:
                # No autocast for MPS/CPU - it can cause issues
                logits = model(inputs)
                loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            
            # FIXED: Calculate accuracy with proper handling
            batch_accuracy, batch_valid_tokens = calculate_token_accuracy(logits, targets, ignore_index)
            
            # Accumulate accuracy correctly
            total_correct_tokens += batch_accuracy * batch_valid_tokens  # This is the number of correct tokens
            total_valid_tokens += batch_valid_tokens
            
            loss = loss / gradient_accumulation_steps
            loss.backward()
            
            # Clear logits immediately
            del logits
            
            accumulation_steps += 1
            
            if accumulation_steps >= gradient_accumulation_steps:
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                
                optimizer.step()
                current_lr = scheduler.step(loss.item() * gradient_accumulation_steps)
                optimizer.zero_grad()
                accumulation_steps = 0
            
            # Statistics
            batch_loss = loss.item() * gradient_accumulation_steps
            total_loss += batch_loss * inputs.size(0)
            total_tokens += targets.numel()
            
            # Clear tensors
            del inputs, targets, loss
            
            batch_times.append(time.time() - batch_start)
            
            # Aggressive memory cleanup every few batches
            if batch_idx % 5 == 0:
                with memory_cleanup():
                    pass
            
            # Enhanced logging with accuracy details
            if batch_idx % log_interval == 0 and batch_idx > 0:
                current_loss = total_loss / total_tokens
                current_accuracy = total_correct_tokens / total_valid_tokens if total_valid_tokens > 0 else 0.0
                avg_batch_time = sum(batch_times[-log_interval:]) / len(batch_times[-log_interval:])
                tokens_per_sec = targets.numel() / avg_batch_time if batch_times else 0
                
                logger.info(f"Epoch {epoch} | Batch {batch_idx}/{len(dataloader)} | "
                           f"Loss: {current_loss:.4f} | Acc: {current_accuracy:.3f} ({current_accuracy*100:.1f}%) | "
                           f"Valid tokens: {total_valid_tokens:,} | Correct: {int(total_correct_tokens):,} | "
                           f"LR: {current_lr:.2e} | Speed: {tokens_per_sec:.0f} tok/s | {get_memory_usage()}")
        
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error(f"OOM at batch {batch_idx}. Clearing cache and skipping batch...")
                
                # Aggressive cleanup
                optimizer.zero_grad()
                with memory_cleanup():
                    pass
                
                # Skip this batch
                continue
            else:
                raise e
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    avg_accuracy = total_correct_tokens / total_valid_tokens if total_valid_tokens > 0 else 0.0
    avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0.0
    
    return avg_loss, avg_accuracy, avg_batch_time

def calculate_top_k_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int = 5, ignore_index: int = 0) -> float:
    """
    FIXED: Calculate top-k accuracy for more detailed evaluation.
    
    Args:
        logits: Model predictions [batch_size, seq_len, vocab_size]
        targets: Ground truth tokens [batch_size, seq_len] 
        k: Top-k value
        ignore_index: Token ID to ignore
    
    Returns:
        Top-k accuracy
    """
    with torch.no_grad():
        # Flatten tensors for easier processing
        flat_logits = logits.view(-1, logits.size(-1))  # [batch_size * seq_len, vocab_size]
        flat_targets = targets.view(-1)  # [batch_size * seq_len]
        
        # Get top-k predictions
        _, top_k_preds = torch.topk(flat_logits, k, dim=-1)  # [batch_size * seq_len, k]
        
        # Create mask for valid tokens
        valid_mask = (flat_targets != ignore_index)
        
        if valid_mask.sum() == 0:
            return 0.0
        
        # Only consider valid tokens
        valid_top_k_preds = top_k_preds[valid_mask]  # [num_valid_tokens, k]
        valid_targets = flat_targets[valid_mask]  # [num_valid_tokens]
        
        # Check if target is in top-k predictions for each valid token
        # Expand targets to match top-k predictions
        targets_expanded = valid_targets.unsqueeze(-1)  # [num_valid_tokens, 1]
        correct = (valid_top_k_preds == targets_expanded).any(dim=-1)  # [num_valid_tokens]
        
        total_correct = correct.sum().item()
        total_valid = valid_targets.numel()
        
        return total_correct / total_valid if total_valid > 0 else 0.0

def generate_sample_text(model, tokenizer, prompt: str = "<user> Hello", 
                        max_length: int = 20, temperature: float = 0.8) -> str:
    """Ultra lightweight text generation with subword tokenizer."""
    model.eval()
    
    try:
        with torch.no_grad():
            input_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
            generated = input_ids.clone()
            
            for _ in range(max_length):
                if generated.size(1) >= 64:  # Very short context
                    break
                
                logits = model(generated)
                next_token_logits = logits[0, -1, :] / temperature
                
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
                
                # Stop on end token
                if next_token.item() == tokenizer.vocab.get("</s>", -1):
                    break
            
            response_ids = generated[0][input_ids.size(1):].tolist()
            response = tokenizer.decode(response_ids)
            return response.strip()
    
    except Exception as e:
        logger.error(f"Error generating sample: {e}")
        return "Error during generation"
    finally:
        model.train()
        with memory_cleanup():
            pass

def get_subword_conservative_config():
    """Get conservative configuration optimized for subword tokenization."""
    
    if device.type == 'cuda':
        # Medium model for CUDA - subword allows smaller vocab
        model_config = ModelConfig(
            vocab_size=16000,      # Smaller vocab due to subword efficiency
            hidden_size=2048,       # Slightly smaller
            num_layers=24,
            num_heads=16,
            seq_length=1024,        # Shorter context
            dropout=0.1,
            model_type="SubwordTransformer",
            tokenizer_type="subword"
        )
        batch_size = 32        # Smaller due to longer sequences from subwords
        max_samples = 8000
        
    elif device.type == 'mps':
        # Small model for MPS
        model_config = ModelConfig(
            vocab_size=8000,       # Very compact vocab
            hidden_size=256,
            num_layers=6,
            num_heads=8,
            seq_length=256,
            dropout=0.1,
            model_type="SubwordTransformer",
            tokenizer_type="subword"
        )
        batch_size = 8
        max_samples = 3000
        
    else:  # CPU
        # Minimal model for CPU
        model_config = ModelConfig(
            vocab_size=4000,
            hidden_size=128,
            num_layers=4,
            num_heads=4,
            seq_length=128,
            dropout=0.1,
            model_type="SubwordTransformer",
            tokenizer_type="subword"
        )
        batch_size = 4
        max_samples = 1500
    
    training_config = TrainingConfig(
        learning_rate=1e-6,     
        weight_decay=0.01,
        batch_size=batch_size,
        gradient_accumulation_steps=16,
        max_epochs=200,          
        warmup_ratio=0.1,
        save_every=1000,
        eval_every=500,
        max_grad_norm=1.0,
        label_smoothing=0.0,
        beta1=0.9,
        beta2=0.95
    )
    
    return model_config, training_config, max_samples

def validate_training_setup():
    """Validate that all required files and dependencies are available."""
    required_files = [
        "oasst1_data/oasst1_train.jsonl",
        "model_manager.py",
        "subword_transformer.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        logger.error("Missing required files:")
        for file_path in missing_files:
            logger.error(f"  - {file_path}")
        
        if "oasst1_data/oasst1_train.jsonl" in missing_files:
            logger.info("To download the dataset, run: python Dataset_download.py")
        
        return False
    
    return True

def save_model_with_retries(model_manager, model, tokenizer, metadata, optimizer, scheduler, max_retries=3):
    """Save model with multiple retry attempts and detailed error reporting."""
    
    for attempt in range(max_retries):
        try:
            logger.info(f"💾 Attempting to save model (attempt {attempt + 1}/{max_retries})...")
            
            # Force garbage collection before save
            with memory_cleanup():
                pass
            
            # Try saving with force_cpu_save=True for stability
            model_id = model_manager.save_model(
                model=model, 
                tokenizer=tokenizer, 
                metadata=metadata, 
                optimizer=optimizer, 
                scheduler=scheduler,
                force_cpu_save=True
            )
            
            logger.info(f"✅ Model saved successfully: {model_id}")
            
            # Validate the saved model
            validation = model_manager.validate_model(model_id)
            if validation['valid']:
                logger.info(f"✅ Model validation passed")
                return model_id
            else:
                logger.warning(f"⚠️ Model validation issues: {validation['issues']}")
                if attempt < max_retries - 1:
                    logger.info("Retrying save due to validation issues...")
                    continue
                else:
                    logger.warning("Model saved but with validation warnings")
                    return model_id
            
        except Exception as save_error:
            logger.error(f"❌ Save attempt {attempt + 1} failed: {save_error}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in 2 seconds...")
                time.sleep(2)
                continue
            else:
                logger.error(f"❌ All save attempts failed. Final error: {save_error}")
                logger.error(f"Save error traceback: {traceback.format_exc()}")
                return None
    
    return None

def evaluate_model_detailed(model, dataloader, criterion, tokenizer, max_batches: int = 50):
    """
    Perform detailed evaluation of the model with multiple metrics.
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader for evaluation data
        criterion: Loss criterion
        tokenizer: Tokenizer for text generation
        max_batches: Maximum number of batches to evaluate (for speed)
    
    Returns:
        Dict with detailed evaluation metrics
    """
    model.eval()
    
    eval_metrics = {
        'total_loss': 0.0,
        'total_tokens': 0,
        'total_correct_tokens': 0,
        'total_valid_tokens': 0,
        'top5_correct_tokens': 0,
        'batch_count': 0,
        'sample_generations': []
    }
    
    ignore_index = tokenizer.vocab.get("<pad>", 0)
    
    try:
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                if batch_idx >= max_batches:
                    break
                
                # Move to device
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                
                # Forward pass
                if device.type == 'cuda':
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
                        logits = model(inputs)
                        loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
                else:
                    logits = model(inputs)
                    loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
                
                # FIXED: Calculate metrics with proper handling
                batch_accuracy, batch_valid_tokens = calculate_token_accuracy(logits, targets, ignore_index)
                top5_accuracy = calculate_top_k_accuracy(logits, targets, k=5, ignore_index=ignore_index)
                
                # Accumulate metrics correctly
                eval_metrics['total_loss'] += loss.item() * inputs.size(0)
                eval_metrics['total_tokens'] += targets.numel()
                eval_metrics['total_correct_tokens'] += batch_accuracy * batch_valid_tokens
                eval_metrics['total_valid_tokens'] += batch_valid_tokens
                eval_metrics['top5_correct_tokens'] += top5_accuracy * batch_valid_tokens
                eval_metrics['batch_count'] += 1
                
                # Generate sample text from first few batches
                if batch_idx < 3:
                    try:
                        # Use first sequence in batch as prompt
                        prompt_ids = inputs[0][:min(10, inputs.size(1))].tolist()
                        prompt_text = tokenizer.decode(prompt_ids)
                        sample_text = generate_sample_text(model, tokenizer, prompt_text[:20], max_length=15)
                        eval_metrics['sample_generations'].append({
                            'prompt': prompt_text,
                            'generated': sample_text
                        })
                    except:
                        pass
                
                # Clean up
                del inputs, targets, logits, loss
                
                # Memory cleanup every few batches
                if batch_idx % 10 == 0:
                    with memory_cleanup():
                        pass
    
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
    
    finally:
        model.train()
        with memory_cleanup():
            pass
    
    # Calculate final metrics
    if eval_metrics['total_tokens'] > 0:
        eval_metrics['avg_loss'] = eval_metrics['total_loss'] / eval_metrics['total_tokens']
        eval_metrics['perplexity'] = calculate_perplexity(eval_metrics['avg_loss'])
    else:
        eval_metrics['avg_loss'] = float('inf')
        eval_metrics['perplexity'] = float('inf')
    
    if eval_metrics['total_valid_tokens'] > 0:
        eval_metrics['accuracy'] = eval_metrics['total_correct_tokens'] / eval_metrics['total_valid_tokens']
        eval_metrics['top5_accuracy'] = eval_metrics['top5_correct_tokens'] / eval_metrics['total_valid_tokens']
    else:
        eval_metrics['accuracy'] = 0.0
        eval_metrics['top5_accuracy'] = 0.0
    
    return eval_metrics

def main():
    """Main training function with guaranteed model saving and FIXED accuracy tracking."""
    
    logger.info("🚀 Starting Subword-Level OASST1 Transformer Training with FIXED Accuracy Tracking")
    logger.info("=" * 80)
    logger.info(f"Initial memory: {get_memory_usage()}")
    
    # Handle imports - if they failed initially, run debug and try again
    if not IMPORTS_SUCCESSFUL:
        logger.error("❌ Imports failed initially - running debug checks")
        if not run_comprehensive_debug():
            logger.error("❌ Debug checks failed - cannot proceed")
            return 1
        
        # Try to import again after debug
        try:
            global ModelManager, ModelConfig, TrainingConfig, ModelMetadata
            global SubwordTransformer, SubwordTokenizer
            
            from model_manager import ModelManager, ModelConfig, TrainingConfig, ModelMetadata
            from subword_transformer import SubwordTransformer, SubwordTokenizer
            logger.info("✅ Successfully imported modules after debug")
            
        except ImportError as e:
            logger.error(f"❌ Still cannot import after debug: {e}")
            return 1
    
    # Validate setup
    if not validate_training_setup():
        logger.error("❌ Training setup validation failed!")
        return 1
    
    # Get conservative configuration for subword tokenization
    model_config, training_config, max_samples = get_subword_conservative_config()
    
    logger.info(f"Using subword-optimized config with FIXED accuracy tracking:")
    logger.info(f"  Model size: {model_config.hidden_size}x{model_config.num_layers}")
    logger.info(f"  Vocab size: {model_config.vocab_size} (subword)")
    logger.info(f"  Batch size: {training_config.batch_size}")
    logger.info(f"  Max samples: {max_samples}")
    logger.info(f"  Sequence length: {model_config.seq_length}")
    
    # Initialize model manager with the FIXED version
    model_manager = ModelManager("models")
    
    try:
        # Load and process OASST1 training data with strict limits
        logger.info("📚 Loading OASST1 dataset (subword-optimized)...")
        texts = load_and_process_data("oasst1_data/oasst1_train.jsonl", max_samples)
        
        if len(texts) == 0:
            raise ValueError("No training data loaded!")
        
        logger.info(f"Memory after data loading: {get_memory_usage()}")
        
        # Create and train BPE tokenizer
        logger.info("🔤 Training BPE subword tokenizer...")
        tokenizer = SubwordTokenizer()
        
        # Use subset for tokenizer training
        sample_size = min(3000, len(texts))
        sample_texts = texts[:sample_size]
        all_text = "\n".join(sample_texts)
        
        # Train BPE with target vocabulary size
        tokenizer.train_from_text(all_text, vocab_size=model_config.vocab_size, min_freq=2)
        model_config.vocab_size = tokenizer.vocab_size()
        
        logger.info(f"✅ BPE tokenizer trained - Vocabulary size: {model_config.vocab_size:,}")
        logger.info(f"✅ Number of BPE merges: {len(tokenizer.merges):,}")
        logger.info(f"Memory after tokenizer: {get_memory_usage()}")
        
        # Test tokenizer with sample
        test_text = "Hello, this is a test of subword tokenization!"
        test_tokens = tokenizer.tokenize(test_text)
        test_encoded = tokenizer.encode(test_text)
        test_decoded = tokenizer.decode(test_encoded)
        
        logger.info(f"📝 Tokenizer test:")
        logger.info(f"   Original: {test_text}")
        logger.info(f"   Subwords: {test_tokens[:10]}{'...' if len(test_tokens) > 10 else ''}")
        logger.info(f"   Decoded: {test_decoded}")
        
        # FIXED: Add debug information about special tokens
        logger.info(f"🔧 Special tokens in vocabulary:")
        special_tokens = ["<pad>", "<unk>", "<s>", "</s>", "<user>", "<assistant>"]
        for token in special_tokens:
            token_id = tokenizer.vocab.get(token, -1)
            logger.info(f"   {token}: {token_id}")
        
        # Create fixed dataset with pre-tokenized sequences
        logger.info("📦 Creating fixed subword training dataset...")
        dataset = FixedSubwordDataset(
            texts, 
            tokenizer, 
            model_config.seq_length,
            overlap_ratio=0.5,
            max_sequences=min(50000, len(texts) * 10)  # Reasonable limit
        )
        
        logger.info(f"Memory after dataset creation: {get_memory_usage()}")
        
        # FIXED: Add dataset validation to ensure we have valid sequences
        if len(dataset) == 0:
            raise ValueError("Dataset is empty! Check tokenizer and data processing.")
        
        # Test a few samples from dataset
        logger.info("🔧 Dataset validation:")
        for i in range(min(3, len(dataset))):
            input_ids, target_ids = dataset[i]
            logger.info(f"   Sample {i+1}: input_shape={input_ids.shape}, target_shape={target_ids.shape}")
            logger.info(f"   Input tokens: {input_ids[:10].tolist()}")
            logger.info(f"   Target tokens: {target_ids[:10].tolist()}")
            
            # Decode sample
            sample_input = tokenizer.decode(input_ids.tolist())
            sample_target = tokenizer.decode(target_ids.tolist())
            logger.info(f"   Decoded input: {sample_input[:50]}...")
            logger.info(f"   Decoded target: {sample_target[:50]}...")
        
        # Conservative dataloader settings
        dataloader = DataLoader(
            dataset,
            batch_size=training_config.batch_size,
            shuffle=True,
            num_workers=0,              # Critical: no multiprocessing
            pin_memory=False,           # Critical: disable pin memory
            drop_last=True,
            persistent_workers=False
        )
        
        # Create evaluation dataloader (smaller subset)
        eval_dataset_size = min(1000, len(dataset) // 10)
        eval_indices = torch.randperm(len(dataset))[:eval_dataset_size]
        eval_dataset = torch.utils.data.Subset(dataset, eval_indices)
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=training_config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=False
        )
        
        logger.info(f"📊 Dataset ready: {len(dataset):,} sequences, {len(dataloader):,} batches/epoch")
        logger.info(f"📊 Eval dataset: {len(eval_dataset):,} sequences, {len(eval_dataloader):,} batches")
        
        # Initialize model
        logger.info("🧠 Initializing subword transformer model...")
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
            betas=(training_config.beta1, training_config.beta2),
            eps=1e-8
        )
        
        # FIXED: Use correct ignore_index from tokenizer
        pad_token_id = tokenizer.vocab.get("<pad>", 0)
        criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
        
        logger.info(f"🔧 Using pad_token_id={pad_token_id} for loss calculation")
        
        # Simple scheduler
        total_steps = len(dataloader) * training_config.max_epochs // training_config.gradient_accumulation_steps
        warmup_steps = int(total_steps * training_config.warmup_ratio)
        scheduler = AdaptiveLRScheduler(optimizer, warmup_steps, total_steps, decay_type="cosine")
        
        logger.info(f"🎯 Training setup: {total_steps:,} steps, {warmup_steps:,} warmup")
        logger.info(f"Memory before training: {get_memory_usage()}")
        
        # FIXED: Reset the debug counter for accuracy calculation
        if hasattr(calculate_token_accuracy, '_debug_count'):
            del calculate_token_accuracy._debug_count
        
        # Training loop with GUARANTEED model saving and FIXED accuracy tracking
        logger.info("🚀 Starting subword-optimized training with FIXED accuracy tracking...")
        training_start = time.time()
        best_loss = float('inf')
        best_accuracy = 0.0
        models_saved = 0
        save_interval = 5  # Save every 5 epochs regardless of improvement
        eval_interval = 3  # Evaluate every 3 epochs
        
        # Track training history
        training_history = {
            'epochs': [],
            'train_loss': [],
            'train_accuracy': [],
            'eval_loss': [],
            'eval_accuracy': [],
            'eval_top5_accuracy': [],
            'perplexity': []
        }
        
        for epoch in range(1, training_config.max_epochs + 1):
            epoch_start = time.time()
            
            try:
                logger.info(f"Starting epoch {epoch}, memory: {get_memory_usage()}")
                
                # Train epoch with FIXED accuracy calculation
                avg_loss, avg_accuracy, avg_batch_time = train_epoch(
                    model, dataloader, criterion, optimizer, scheduler, epoch,
                    training_config.gradient_accumulation_steps,
                    training_config.max_grad_norm,
                    ignore_index=pad_token_id  # Use correct pad token
                )
                
                # Calculate metrics
                perplexity = calculate_perplexity(avg_loss)
                epoch_time = time.time() - epoch_start
                
                # Update training history
                training_history['epochs'].append(epoch)
                training_history['train_loss'].append(avg_loss)
                training_history['train_accuracy'].append(avg_accuracy)
                training_history['perplexity'].append(perplexity)
                
                # Evaluation with FIXED accuracy calculation
                eval_metrics = None
                if epoch % eval_interval == 0 or epoch == 1 or epoch == training_config.max_epochs:
                    logger.info("📊 Running detailed evaluation...")
                    eval_start = time.time()
                    eval_metrics = evaluate_model_detailed(model, eval_dataloader, criterion, tokenizer)
                    eval_time = time.time() - eval_start
                    
                    # Update evaluation history
                    training_history['eval_loss'].append(eval_metrics['avg_loss'])
                    training_history['eval_accuracy'].append(eval_metrics['accuracy'])
                    training_history['eval_top5_accuracy'].append(eval_metrics['top5_accuracy'])
                    
                    logger.info(f"📊 Evaluation completed in {eval_time:.1f}s:")
                    logger.info(f"   Eval Loss: {eval_metrics['avg_loss']:.4f}")
                    logger.info(f"   Eval Accuracy: {eval_metrics['accuracy']:.3f} ({eval_metrics['accuracy']*100:.1f}%)")
                    logger.info(f"   Eval Top-5 Accuracy: {eval_metrics['top5_accuracy']:.3f} ({eval_metrics['top5_accuracy']*100:.1f}%)")
                    logger.info(f"   Eval Perplexity: {eval_metrics['perplexity']:.2f}")
                    
                    if eval_metrics['sample_generations']:
                        logger.info("   Sample generations:")
                        for i, gen in enumerate(eval_metrics['sample_generations'][:2]):
                            logger.info(f"     {i+1}. '{gen['prompt'][:30]}...' → '{gen['generated']}'")
                
                # Enhanced logging with percentage accuracy
                logger.info("=" * 60)
                logger.info(f"📊 Epoch {epoch}/{training_config.max_epochs} Summary:")
                logger.info(f"   Train Loss: {avg_loss:.4f} | Train Accuracy: {avg_accuracy:.3f} ({avg_accuracy*100:.1f}%)")
                logger.info(f"   Perplexity: {perplexity:.2f}")
                if eval_metrics:
                    logger.info(f"   Eval Loss: {eval_metrics['avg_loss']:.4f} | Eval Accuracy: {eval_metrics['accuracy']:.3f} ({eval_metrics['accuracy']*100:.1f}%)")
                logger.info(f"   Time: {epoch_time:.1f}s | {get_memory_usage()}")
                
                # Update best metrics
                is_best_loss = avg_loss < best_loss
                is_best_accuracy = avg_accuracy > best_accuracy
                
                if is_best_loss:
                    best_loss = avg_loss
                    logger.info(f"🏆 New best train loss: {best_loss:.4f}")
                
                if is_best_accuracy:
                    best_accuracy = avg_accuracy
                    logger.info(f"🏆 New best train accuracy: {best_accuracy:.3f} ({best_accuracy*100:.1f}%)")
                
                # Sample generation every few epochs
                if epoch % 3 == 0:
                    with memory_cleanup():
                        sample = generate_sample_text(model, tokenizer, "<user> Hello", 15)
                        logger.info(f"   Sample: <user> Hello → {sample}")
                
                # GUARANTEED MODEL SAVING - save if best OR every save_interval epochs OR last epoch
                should_save = (
                    is_best_loss or 
                    is_best_accuracy or
                    epoch % save_interval == 0 or 
                    epoch == training_config.max_epochs or
                    epoch == 1  # Always save first epoch as baseline
                )
                
                if should_save:
                    with memory_cleanup():
                        # Create comprehensive metadata
                        save_reason = []
                        if is_best_loss:
                            save_reason.append("best_loss")
                        if is_best_accuracy:
                            save_reason.append("best_accuracy")
                        if epoch % save_interval == 0:
                            save_reason.append("regular_interval")
                        if epoch == training_config.max_epochs:
                            save_reason.append("final_epoch")
                        if epoch == 1:
                            save_reason.append("initial_checkpoint")
                        
                        performance_metrics = {
                            "train_loss": float(avg_loss),
                            "train_accuracy": float(avg_accuracy),
                            "train_accuracy_percent": float(avg_accuracy * 100),
                            "train_perplexity": float(perplexity),
                            "batch_time_ms": float(avg_batch_time * 1000),
                            "epoch": int(epoch),
                            "learning_rate": float(scheduler.optimizer.param_groups[0]['lr']),
                            "gradient_norm": 0.0,
                            "is_best_loss": is_best_loss,
                            "is_best_accuracy": is_best_accuracy,
                            "save_reason": ",".join(save_reason),
                            "training_time_hours": float((time.time() - training_start) / 3600),
                            "pad_token_id": int(pad_token_id),
                            "accuracy_calculation_method": "fixed_flattened_comparison",
                        }
                        
                        # Add evaluation metrics if available
                        if eval_metrics:
                            performance_metrics.update({
                                "eval_loss": float(eval_metrics['avg_loss']),
                                "eval_accuracy": float(eval_metrics['accuracy']),
                                "eval_accuracy_percent": float(eval_metrics['accuracy'] * 100),
                                "eval_top5_accuracy": float(eval_metrics['top5_accuracy']),
                                "eval_top5_accuracy_percent": float(eval_metrics['top5_accuracy'] * 100),
                                "eval_perplexity": float(eval_metrics['perplexity']),
                                "eval_batches": int(eval_metrics['batch_count']),
                            })
                        
                        metadata = ModelMetadata(
                            model_name="OASST1_SubwordTransformer",
                            version=f"v1.0_epoch_{epoch}{'_BEST' if (is_best_loss or is_best_accuracy) else ''}",
                            created_at=datetime.now().isoformat(),
                            last_modified=datetime.now().isoformat(),
                            model_config=model_config,
                            training_config=training_config,
                            dataset_info={
                                "name": "OpenAssistant OASST1 (Subword-Optimized with FIXED Accuracy)",
                                "source": "oasst1_train.jsonl",
                                "num_samples": len(texts),
                                "vocab_size": model_config.vocab_size,
                                "seq_length": model_config.seq_length,
                                "preprocessing": "BPE subword tokenization",
                                "num_merges": len(tokenizer.merges),
                                "tokenizer_type": "BPE (Byte Pair Encoding)",
                                "language": "English",
                                "dataset_version": "OASST1",
                                "train_sequences": len(dataset),
                                "eval_sequences": len(eval_dataset),
                                "pad_token_id": int(pad_token_id),
                            },
                            performance_metrics=performance_metrics,
                            model_size_mb=float(model_size_mb),
                            total_parameters=int(total_params),
                            trainable_parameters=int(trainable_params),
                            training_time_hours=float((time.time() - training_start) / 3600),
                            epochs_trained=int(epoch),
                            best_loss=float(best_loss),
                            best_perplexity=float(calculate_perplexity(best_loss)),
                            hardware_used=f"{device.type.upper()}" + (f" ({torch.cuda.get_device_name()})" if device.type == 'cuda' else ""),
                            pytorch_version=torch.__version__,
                            cuda_version=torch.version.cuda if device.type == 'cuda' else None,
                            model_hash="",
                            tokenizer_hash="",
                            notes=f"Subword-level OASST1 transformer with BPE tokenization and FIXED accuracy tracking. "
                                  f"Epoch {epoch}/{training_config.max_epochs}. "
                                  f"Save reason: {', '.join(save_reason)}. "
                                  f"Train: loss={avg_loss:.4f}, acc={avg_accuracy:.3f} ({avg_accuracy*100:.1f}%). "
                                  f"Best: loss={best_loss:.4f}, acc={best_accuracy:.3f} ({best_accuracy*100:.1f}%). "
                                  f"Training on {len(texts):,} samples with {model_config.vocab_size:,} vocab using {len(tokenizer.merges):,} BPE merges. "
                                  f"Model: {model_config.hidden_size}D x {model_config.num_layers}L x {model_config.num_heads}H. "
                                  f"Hardware: {device.type.upper()}. "
                                  f"Accuracy calculation uses flattened tensor comparison with proper masking.",
                            tags=["oasst1", "subword", "bpe", "transformer", "conversational", "fixed_accuracy", f"epoch_{epoch}"] + 
                                 (["best_loss"] if is_best_loss else []) +
                                 (["best_accuracy"] if is_best_accuracy else []) +
                                 save_reason
                        )
                        
                        # Use the retry mechanism for robust saving
                        model_id = save_model_with_retries(
                            model_manager, model, tokenizer, metadata, optimizer, scheduler
                        )
                        
                        if model_id:
                            models_saved += 1
                            logger.info(f"💾 Model saved successfully: {model_id} (#{models_saved})")
                        else:
                            logger.error(f"❌ Failed to save model for epoch {epoch}")
                
                logger.info("=" * 60)
                
                # Cleanup after each epoch
                with memory_cleanup():
                    pass
                
            except Exception as e:
                logger.error(f"❌ Error in epoch {epoch}: {e}")
                logger.info(f"Memory state: {get_memory_usage()}")
                
                if "out of memory" in str(e).lower():
                    logger.info("Attempting to recover from OOM...")
                    with memory_cleanup():
                        pass
                    continue
                else:
                    raise e
        
        # Training completion
        total_time = time.time() - training_start
        logger.info("=" * 80)
        logger.info("✅ Subword-optimized training with FIXED accuracy tracking completed successfully!")
        logger.info(f"🎯 Best train loss achieved: {best_loss:.4f}")
        logger.info(f"🎯 Best train accuracy achieved: {best_accuracy:.3f} ({best_accuracy*100:.1f}%)")
        logger.info(f"🎯 Best perplexity: {calculate_perplexity(best_loss):.2f}")
        logger.info(f"💾 Total models saved: {models_saved}")
        logger.info(f"⏱️  Total training time: {total_time/3600:.2f} hours")
        logger.info(f"🔤 Final vocabulary: {model_config.vocab_size:,} tokens with {len(tokenizer.merges):,} BPE merges")
        logger.info(f"🚀 Average processing speed: {len(dataset) * model_config.seq_length * training_config.max_epochs / total_time:.0f} tokens/sec")
        logger.info(f"💾 Final memory state: {get_memory_usage()}")
        
        # Print training history summary
        logger.info("\n📈 Training History Summary:")
        logger.info(f"   Final train loss: {training_history['train_loss'][-1]:.4f}")
        logger.info(f"   Final train accuracy: {training_history['train_accuracy'][-1]:.3f} ({training_history['train_accuracy'][-1]*100:.1f}%)")
        if training_history['eval_loss']:
            logger.info(f"   Final eval loss: {training_history['eval_loss'][-1]:.4f}")
            logger.info(f"   Final eval accuracy: {training_history['eval_accuracy'][-1]:.3f} ({training_history['eval_accuracy'][-1]*100:.1f}%)")
            logger.info(f"   Final eval top-5 accuracy: {training_history['eval_top5_accuracy'][-1]:.3f} ({training_history['eval_top5_accuracy'][-1]*100:.1f}%)")
        
        # FINAL GUARANTEED SAVE - save one more time as final model if no models were saved
        if models_saved == 0:
            logger.warning("⚠️ No models were saved during training! Performing emergency final save...")
            
            final_metadata = ModelMetadata(
                model_name="OASST1_SubwordTransformer_FINAL",
                version="v1.0_EMERGENCY_SAVE",
                created_at=datetime.now().isoformat(),
                last_modified=datetime.now().isoformat(),
                model_config=model_config,
                training_config=training_config,
                dataset_info={
                    "name": "OpenAssistant OASST1 (Emergency Save)",
                    "source": "oasst1_train.jsonl",
                    "num_samples": len(texts),
                    "vocab_size": model_config.vocab_size,
                    "preprocessing": "BPE subword tokenization with FIXED accuracy tracking",
                },
                performance_metrics={
                    "final_train_loss": float(avg_loss) if 'avg_loss' in locals() else float('inf'),
                    "final_train_accuracy": float(avg_accuracy) if 'avg_accuracy' in locals() else 0.0,
                    "final_train_accuracy_percent": float(avg_accuracy * 100) if 'avg_accuracy' in locals() else 0.0,
                    "training_completed": True,
                    "best_train_loss": float(best_loss),
                    "best_train_accuracy": float(best_accuracy),
                    "best_train_accuracy_percent": float(best_accuracy * 100),
                    "accuracy_calculation_method": "fixed_flattened_comparison",
                },
                model_size_mb=float(model_size_mb),
                total_parameters=int(total_params),
                trainable_parameters=int(trainable_params),
                training_time_hours=float(total_time / 3600),
                epochs_trained=int(training_config.max_epochs),
                best_loss=float(best_loss),
                best_perplexity=float(calculate_perplexity(best_loss)),
                hardware_used=f"{device.type.upper()}",
                pytorch_version=torch.__version__,
                notes="Emergency save after training completion to ensure model with FIXED accuracy tracking is preserved.",
                tags=["oasst1", "subword", "emergency_save", "final", "fixed_accuracy"]
            )
            
            emergency_model_id = save_model_with_retries(
                model_manager, model, tokenizer, final_metadata, optimizer, scheduler
            )
            
            if emergency_model_id:
                models_saved += 1
                logger.info(f"✅ Emergency final save successful: {emergency_model_id}")
            else:
                logger.error(f"❌ Even emergency save failed!")
        
        # Save training history to a JSON file
        try:
            history_file = Path("training_history.json")
            with open(history_file, 'w') as f:
                json.dump(training_history, f, indent=2)
            logger.info(f"📊 Training history saved to: {history_file}")
        except Exception as e:
            logger.error(f"Failed to save training history: {e}")
        
        # Print final model summary
        logger.info("\n" + "=" * 80)
        logger.info("📊 FINAL MODEL SUMMARY WITH FIXED ACCURACY METRICS")
        model_manager.print_model_summary()
        logger.info("=" * 80)
        
        # Print detailed accuracy summary
        logger.info("🎯 FIXED ACCURACY SUMMARY:")
        logger.info(f"   Best Training Loss: {best_loss:.4f}")
        logger.info(f"   Best Training Accuracy: {best_accuracy:.3f} ({best_accuracy*100:.1f}%)")
        logger.info(f"   Final Training Loss: {training_history['train_loss'][-1]:.4f}")
        logger.info(f"   Final Training Accuracy: {training_history['train_accuracy'][-1]:.3f} ({training_history['train_accuracy'][-1]*100:.1f}%)")
        
        if training_history['eval_loss']:
            logger.info(f"   Final Eval Loss: {training_history['eval_loss'][-1]:.4f}")
            logger.info(f"   Final Eval Accuracy: {training_history['eval_accuracy'][-1]:.3f} ({training_history['eval_accuracy'][-1]*100:.1f}%)")
            logger.info(f"   Final Eval Top-5 Accuracy: {training_history['eval_top5_accuracy'][-1]:.3f} ({training_history['eval_top5_accuracy'][-1]*100:.1f}%)")
        
        logger.info("🔧 ACCURACY CALCULATION FIXES APPLIED:")
        logger.info("   - Fixed tensor flattening before comparison")
        logger.info("   - Proper masking for padding tokens")
        logger.info("   - Corrected accumulation of accuracy metrics")
        logger.info("   - Added debug logging for accuracy calculation")
        logger.info("   - Verified pad_token_id usage in loss and accuracy")
        logger.info("=" * 80)
        
        if models_saved > 0:
            logger.info(f"✅ Training completed with {models_saved} models saved successfully!")
            logger.info(f"📊 All models include comprehensive FIXED accuracy metrics and evaluation data")
            return 0
        else:
            logger.error(f"❌ Training completed but NO MODELS WERE SAVED!")
            return 1
        
    except KeyboardInterrupt:
        logger.info("❌ Training interrupted by user")
        # Try to save current state before exiting
        logger.info("Attempting to save current model state before exit...")
        if 'model' in locals() and 'tokenizer' in locals():
            try:
                interrupt_metadata = ModelMetadata(
                    model_name="OASST1_SubwordTransformer_INTERRUPTED",
                    version="v1.0_INTERRUPTED",
                    created_at=datetime.now().isoformat(),
                    last_modified=datetime.now().isoformat(),
                    model_config=model_config if 'model_config' in locals() else {},
                    performance_metrics={
                        "interrupted_at_epoch": epoch if 'epoch' in locals() else 0,
                        "last_train_loss": float(avg_loss) if 'avg_loss' in locals() else float('inf'),
                        "last_train_accuracy": float(avg_accuracy) if 'avg_accuracy' in locals() else 0.0,
                        "last_train_accuracy_percent": float(avg_accuracy * 100) if 'avg_accuracy' in locals() else 0.0,
                        "best_train_loss": float(best_loss) if 'best_loss' in locals() else float('inf'),
                        "best_train_accuracy": float(best_accuracy) if 'best_accuracy' in locals() else 0.0,
                        "best_train_accuracy_percent": float(best_accuracy * 100) if 'best_accuracy' in locals() else 0.0,
                        "accuracy_calculation_method": "fixed_flattened_comparison",
                    },
                    notes="Model saved after training interruption - includes FIXED accuracy tracking up to interruption point",
                    tags=["interrupted", "partial", "fixed_accuracy"]
                )
                interrupt_id = save_model_with_retries(
                    model_manager, model, tokenizer, interrupt_metadata, 
                    optimizer if 'optimizer' in locals() else None, 
                    scheduler if 'scheduler' in locals() else None
                )
                if interrupt_id:
                    logger.info(f"✅ Interrupted model saved: {interrupt_id}")
            except:
                logger.error("Failed to save interrupted model")
        return 1
    except Exception as e:
        logger.error(f"❌ Training failed with error: {e}")
        logger.info(f"Memory state at failure: {get_memory_usage()}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return 1
    finally:
        # Final cleanup
        logger.info("Performing final memory cleanup...")
        with memory_cleanup():
            pass

if __name__ == "__main__":
    exit(main())