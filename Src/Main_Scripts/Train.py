#!/usr/bin/env python3
"""
Enhanced ModernSubwordTransformer Training Script
Fixed version with proper regex handling
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
import math
import random
from collections import defaultdict, Counter

# Core ML imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Use regex module for Unicode support instead of re
try:
    import regex as re
except ImportError:
    print("Installing regex module for Unicode support...")
    os.system("pip install regex")
    import regex as re

import numpy as np
from tqdm import tqdm

# Optional imports
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from torch.cuda.amp import GradScaler, autocast
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False

# Configuration
@dataclass
class ModelConfig:
    """Model configuration"""
    vocab_size: int = 16000
    hidden_size: int = 2048
    num_layers: int = 24
    num_heads: int = 16
    intermediate_size: int = 4096
    max_seq_length: int = 2048
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    pad_token_id: int = 0
    
@dataclass
class TrainingConfig:
    """Training configuration"""
    batch_size: int = 4
    gradient_accumulation_steps: int = 16
    learning_rate: float = 5e-4
    num_epochs: int = 200
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    fp16: bool = True
    save_steps: int = 1000
    eval_steps: int = 500
    logging_steps: int = 100
    output_dir: str = "models"
    experiment_name: str = "auto"
    
class SubwordTokenizer:
    """Fixed SubwordTokenizer with proper regex handling"""
    
    def __init__(self, vocab_size: int = 16000):
        self.vocab_size = vocab_size
        self.word_freqs = defaultdict(int)
        self.vocab = {}
        self.merges = []
        
        # Fixed regex patterns - no Unicode properties
        self.word_pattern = re.compile(r'[a-zA-ZÀ-ÿĀ-žА-я]+|[0-9]+|[^\w\s]', re.UNICODE)
        self.split_pattern = re.compile(r'(\s+)', re.UNICODE)
        
        # Special tokens
        self.special_tokens = {
            '<pad>': 0,
            '<unk>': 1, 
            '<s>': 2,
            '</s>': 3,
        }
        
    def _get_word_tokens(self, word: str) -> List[str]:
        """Split word into character tokens with end marker"""
        if not word:
            return []
        return list(word[:-1]) + [word[-1] + '</w>']
    
    def _get_pairs(self, word_tokens: List[str]) -> set:
        """Get all adjacent pairs in word tokens"""
        pairs = set()
        prev_char = word_tokens[0]
        for char in word_tokens[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs
    
    def train(self, texts: List[str], progress_callback=None):
        """Train BPE tokenizer on texts"""
        logging.info("🔤 Training SubwordTokenizer...")
        
        # Count word frequencies
        for i, text in enumerate(texts):
            if progress_callback and i % 10000 == 0:
                progress_callback(f"Processing text {i+1}/{len(texts)}")
                
            words = self.word_pattern.findall(text.lower())
            for word in words:
                self.word_freqs[word] += 1
        
        logging.info(f"Found {len(self.word_freqs)} unique words")
        
        # Initialize vocabulary with characters
        vocab = set()
        word_splits = {}
        
        for word in self.word_freqs:
            word_tokens = self._get_word_tokens(word)
            word_splits[word] = word_tokens
            vocab.update(word_tokens)
        
        # Add special tokens
        vocab.update(self.special_tokens.keys())
        
        # Learn merges
        num_merges = self.vocab_size - len(vocab)
        logging.info(f"Learning {num_merges} merges...")
        
        for i in tqdm(range(num_merges), desc="Learning BPE merges"):
            pairs = defaultdict(int)
            
            # Count all pairs
            for word, freq in self.word_freqs.items():
                word_tokens = word_splits[word]
                word_pairs = self._get_pairs(word_tokens)
                for pair in word_pairs:
                    pairs[pair] += freq
            
            if not pairs:
                break
                
            # Find most frequent pair
            best_pair = max(pairs, key=pairs.get)
            self.merges.append(best_pair)
            
            # Merge best pair in all words
            new_word_splits = {}
            for word in word_splits:
                new_tokens = self._merge_tokens(word_splits[word], best_pair)
                new_word_splits[word] = new_tokens
                vocab.add(''.join(best_pair))
            
            word_splits = new_word_splits
        
        # Build final vocabulary
        self.vocab = self.special_tokens.copy()
        for i, token in enumerate(sorted(vocab - set(self.special_tokens.keys()))):
            self.vocab[token] = len(self.special_tokens) + i
        
        logging.info(f"✅ Tokenizer trained with {len(self.vocab)} tokens")
        
    def _merge_tokens(self, tokens: List[str], pair: Tuple[str, str]) -> List[str]:
        """Merge a specific pair in token list"""
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                new_tokens.append(pair[0] + pair[1])
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        return new_tokens
        
    def encode(self, text: str) -> List[int]:
        """Encode text to token ids"""
        if not hasattr(self, 'vocab') or not self.vocab:
            raise ValueError("Tokenizer not trained yet")
            
        words = self.word_pattern.findall(text.lower())
        ids = []
        
        for word in words:
            word_tokens = self._get_word_tokens(word)
            
            # Apply learned merges
            for merge_pair in self.merges:
                word_tokens = self._merge_tokens(word_tokens, merge_pair)
            
            # Convert to ids
            for token in word_tokens:
                ids.append(self.vocab.get(token, self.special_tokens['<unk>']))
                
        return ids
    
    def decode(self, ids: List[int]) -> str:
        """Decode token ids to text"""
        if not hasattr(self, 'vocab') or not self.vocab:
            raise ValueError("Tokenizer not trained yet")
            
        id_to_token = {v: k for k, v in self.vocab.items()}
        tokens = [id_to_token.get(id, '<unk>') for id in ids]
        
        # Join tokens and clean up
        text = ''.join(tokens)
        text = text.replace('</w>', ' ')
        return text.strip()

class TransformerDataset(Dataset):
    """Dataset for transformer training"""
    
    def __init__(self, texts: List[str], tokenizer: SubwordTokenizer, max_length: int = 2048):
        self.texts = texts
        self.tokenizer = tokenizer  
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize
        tokens = self.tokenizer.encode(text)
        
        # Truncate or pad
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens.extend([self.tokenizer.special_tokens['<pad>']] * (self.max_length - len(tokens)))
            
        # Create input and target (shifted for language modeling)
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'labels': labels
        }

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism"""
    
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        
        assert hidden_size % num_heads == 0
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)  
        self.value = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        batch_size, seq_len, hidden_size = x.size()
        
        # Linear projections
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_size)
        
        # Apply causal mask
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
        scores = scores.masked_fill(causal_mask == 0, float('-inf'))
        
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))
            
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        
        return self.output(context)

class TransformerBlock(nn.Module):
    """Transformer block with attention and feedforward"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = MultiHeadAttention(config.hidden_size, config.num_heads, config.dropout)
        self.ln1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.dropout)
        )
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        # Attention with residual connection
        attn_out = self.attention(self.ln1(x), attention_mask)
        x = x + attn_out
        
        # MLP with residual connection  
        mlp_out = self.mlp(self.ln2(x))
        x = x + mlp_out
        
        return x

class ModernSubwordTransformer(nn.Module):
    """Modern Transformer model"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_seq_length, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])
        
        # Output layer
        self.ln_final = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
            
    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None):
        batch_size, seq_len = input_ids.size()
        device = input_ids.device
        
        # Create position ids
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_embeds = self.token_embeddings(input_ids)
        pos_embeds = self.position_embeddings(position_ids)
        x = self.dropout(token_embeds + pos_embeds)
        
        # Create attention mask
        attention_mask = (input_ids != self.config.pad_token_id).unsqueeze(1).unsqueeze(2)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, attention_mask)
            
        # Final layer norm and output projection
        x = self.ln_final(x)
        logits = self.lm_head(x)
        
        loss = None
        if labels is not None:
            # Compute cross entropy loss
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id)
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
            
        return {'logits': logits, 'loss': loss}

class ModelManager:
    """Manages model saving and loading"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def save_model(self, model: nn.Module, tokenizer: SubwordTokenizer, config: ModelConfig, 
                   step: int, loss: float):
        """Save model, tokenizer, and config"""
        save_dir = self.output_dir / f"checkpoint-{step}"
        save_dir.mkdir(exist_ok=True)
        
        # Save model
        torch.save(model.state_dict(), save_dir / "model.pt")
        
        # Save tokenizer
        tokenizer_data = {
            'vocab': tokenizer.vocab,
            'merges': tokenizer.merges,
            'special_tokens': tokenizer.special_tokens
        }
        with open(save_dir / "tokenizer.json", 'w') as f:
            json.dump(tokenizer_data, f, indent=2)
            
        # Save config
        with open(save_dir / "config.json", 'w') as f:
            json.dump(config.__dict__, f, indent=2)
            
        logging.info(f"💾 Model saved to {save_dir} (loss: {loss:.4f})")

class AdvancedTrainer:
    """Advanced trainer with modern features"""
    
    def __init__(self, model_config: ModelConfig, training_config: TrainingConfig, 
                 device: str = "auto"):
        self.model_config = model_config
        self.training_config = training_config
        
        # Device setup
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        # Initialize model
        self.model = ModernSubwordTransformer(model_config).to(self.device)
        self.model_manager = ModelManager(training_config.output_dir)
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        
        # Mixed precision
        self.use_amp = training_config.fp16 and AMP_AVAILABLE
        if self.use_amp:
            self.scaler = GradScaler()
            
        # Wandb setup
        self.use_wandb = WANDB_AVAILABLE and hasattr(training_config, 'wandb_project')
        
        logging.info(f"🚀 AdvancedTrainer initialized:")
        logging.info(f"   Experiment: {training_config.experiment_name}")
        logging.info(f"   Device: {self.device}")
        logging.info(f"   Precision: {'fp16' if self.use_amp else 'fp32'}")
        logging.info(f"   Wandb: {self.use_wandb}")
        
    def train(self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None):
        """Main training loop"""
        # Data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
        
        eval_loader = None
        if eval_dataset:
            eval_loader = DataLoader(
                eval_dataset,
                batch_size=self.training_config.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True
            )
        
        # Optimizer and scheduler
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay
        )
        
        total_steps = len(train_loader) * self.training_config.num_epochs // self.training_config.gradient_accumulation_steps
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
        
        # Training loop
        logging.info("🏋️ Starting training...")
        
        for epoch in range(self.training_config.num_epochs):
            self.current_epoch = epoch
            self._train_epoch(train_loader, optimizer, scheduler, eval_loader)
            
        logging.info("✅ Training completed!")
        
    def _train_epoch(self, train_loader: DataLoader, optimizer, scheduler, eval_loader=None):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for step, batch in enumerate(progress_bar):
            # Move to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    outputs = self.model(**batch)
                    loss = outputs['loss'] / self.training_config.gradient_accumulation_steps
                    
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(**batch)
                loss = outputs['loss'] / self.training_config.gradient_accumulation_steps
                loss.backward()
                
            total_loss += loss.item()
            
            # Gradient accumulation
            if (step + 1) % self.training_config.gradient_accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.training_config.max_grad_norm)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.training_config.max_grad_norm)
                    optimizer.step()
                    
                scheduler.step()
                optimizer.zero_grad()
                self.global_step += 1
                
                # Logging
                if self.global_step % self.training_config.logging_steps == 0:
                    avg_loss = total_loss / self.training_config.logging_steps
                    learning_rate = scheduler.get_last_lr()[0]
                    
                    logging.info(f"Step {self.global_step}: loss={avg_loss:.4f}, lr={learning_rate:.2e}")
                    total_loss = 0
                    
                # Evaluation
                if eval_loader and self.global_step % self.training_config.eval_steps == 0:
                    eval_loss = self._evaluate(eval_loader)
                    logging.info(f"📊 Eval loss: {eval_loss:.4f}")
                    
                # Save checkpoint
                if self.global_step % self.training_config.save_steps == 0:
                    self.model_manager.save_model(
                        self.model, self.tokenizer, self.model_config, 
                        self.global_step, avg_loss
                    )
                    
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
    def _evaluate(self, eval_loader: DataLoader) -> float:
        """Evaluate model"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                if self.use_amp:
                    with autocast():
                        outputs = self.model(**batch)
                        loss = outputs['loss']
                else:
                    outputs = self.model(**batch)
                    loss = outputs['loss']
                    
                total_loss += loss.item()
                num_batches += 1
                
        self.model.train()
        return total_loss / num_batches if num_batches > 0 else 0

def load_data(file_path: str) -> List[str]:
    """Load training data from JSONL file"""
    texts = []
    duplicates = set()
    
    logging.info(f"📂 Loading data from: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        processed_lines = 0
        valid_texts = 0
        
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                
                # Extract text (adjust based on your data format)
                text = None
                if 'text' in data:
                    text = data['text']
                elif 'content' in data:
                    text = data['content']
                elif 'message' in data:
                    text = data['message']
                    
                if text and isinstance(text, str) and len(text.strip()) > 0:
                    text = text.strip()
                    
                    # Remove duplicates
                    if text not in duplicates:
                        texts.append(text)
                        duplicates.add(text)
                        valid_texts += 1
                    
                processed_lines += 1
                
                if processed_lines % 10000 == 0:
                    logging.info(f"   Processed {processed_lines:,} lines, {valid_texts:,} valid texts")
                    
            except json.JSONDecodeError:
                continue
            except Exception as e:
                logging.warning(f"Error processing line {line_num}: {e}")
                continue
    
    num_duplicates = processed_lines - len(texts)
    if num_duplicates > 0:
        logging.info(f"   Removed {num_duplicates} duplicates")
        
    logging.info(f"✅ Loaded {len(texts):,} texts from {file_path}")
    return texts

def setup_logging(log_dir: str = "logs") -> str:
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return log_file

def get_device_info():
    """Get device information"""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        compute_capability = torch.cuda.get_device_capability(0)
        num_gpus = torch.cuda.device_count()
        
        logging.info("🔥 CUDA Device Detected:")
        logging.info(f"   GPU: {device_name} (Compute {compute_capability[0]}.{compute_capability[1]})")
        logging.info(f"   Memory: {memory_gb:.1f}GB")
        logging.info(f"   GPUs Available: {num_gpus}")
    else:
        logging.info("💻 Using CPU")

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train ModernSubwordTransformer")
    parser.add_argument("--config", default="auto", help="Model configuration")
    parser.add_argument("--output", default="models", help="Output directory")
    parser.add_argument("--experiment", default="auto", help="Experiment name")
    
    args = parser.parse_args()
    
    # HARDCODED DATA PATH - Change this to your training data file
    DATA_PATH = "oasst1_data/oasst1_train.jsonl"  # <-- CHANGE THIS TO YOUR DATA FILE PATH
    
    # Setup logging
    log_file = setup_logging()
    logging.info("🔧 Logging initialized - Log file: %s", log_file)
    
    # Header
    logging.info("🚀 Enhanced ModernSubwordTransformer Training")
    logging.info("=" * 80)
    
    # Environment check
    logging.info("🔍 Environment Check:")
    logging.info(f"   Python: {sys.version}")
    logging.info(f"   PyTorch: {torch.__version__}")
    logging.info(f"   CUDA Available: {torch.cuda.is_available()}")
    logging.info(f"   AMP Available: {AMP_AVAILABLE}")
    logging.info(f"   DeepSpeed Available: False")  # Not implemented in this version
    logging.info(f"   Wandb Available: {WANDB_AVAILABLE}")
    
    get_device_info()
    
    # Configuration
    if args.config == "auto":
        model_config = ModelConfig()
        training_config = TrainingConfig(
            output_dir=args.output,
            experiment_name=args.experiment if args.experiment != "auto" else f"auto_{random.randint(1000000, 9999999)}"
        )
    else:
        # Load custom config if provided
        with open(args.config) as f:
            config_data = json.load(f)
        model_config = ModelConfig(**config_data.get('model', {}))
        training_config = TrainingConfig(**config_data.get('training', {}))
    
    logging.info(f"📊 Configuration: {args.config}")
    logging.info(f"   Model: {model_config.hidden_size}d × {model_config.num_layers}L")
    logging.info(f"   Vocabulary: {model_config.vocab_size:,}")
    logging.info(f"   Sequence Length: {model_config.max_seq_length}")
    logging.info(f"   Precision: {'fp16' if training_config.fp16 else 'fp32'}")
    logging.info(f"   Batch Size: {training_config.batch_size}")
    logging.info(f"   Gradient Accumulation: {training_config.gradient_accumulation_steps}")
    
    # Initialize trainer
    trainer = AdvancedTrainer(model_config, training_config)
    
    try:
        # Load and prepare data
        logging.info("📦 Loading and preparing data...")
        texts = load_data(DATA_PATH)
        
        # Split data
        split_idx = int(0.9 * len(texts))
        train_texts = texts[:split_idx]
        eval_texts = texts[split_idx:]
        
        logging.info("📊 Data split:")
        logging.info(f"   Training: {len(train_texts):,} texts")
        logging.info(f"   Evaluation: {len(eval_texts):,} texts")
        
        # Train tokenizer
        tokenizer = SubwordTokenizer(model_config.vocab_size)
        tokenizer.train(train_texts)
        trainer.tokenizer = tokenizer  # Store for saving
        
        # Update model config with actual vocab size
        model_config.vocab_size = len(tokenizer.vocab)
        trainer.model_config = model_config
        
        # Recreate model with correct vocab size
        trainer.model = ModernSubwordTransformer(model_config).to(trainer.device)
        
        # Create datasets
        logging.info("🔄 Creating datasets...")
        train_dataset = TransformerDataset(train_texts, tokenizer, model_config.max_seq_length)
        eval_dataset = TransformerDataset(eval_texts, tokenizer, model_config.max_seq_length)
        
        logging.info(f"✅ Datasets created:")
        logging.info(f"   Training samples: {len(train_dataset):,}")
        logging.info(f"   Evaluation samples: {len(eval_dataset):,}")
        
        # Start training
        logging.info("🏋️ Starting training...")
        trainer.train(train_dataset, eval_dataset)
        
        # Save final model
        final_loss = 0.0  # You might want to track this during training
        trainer.model_manager.save_model(
            trainer.model, tokenizer, model_config, 
            trainer.global_step, final_loss
        )
        
        logging.info("🎉 Training completed successfully!")
        
    except Exception as e:
        logging.error(f"❌ Training failed: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()