# Conversational Transformer Training System
# Optimized for OASST1 conversational dataset format

import math
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from collections import Counter, defaultdict
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np

# Optional high-performance imports
try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False

try:
    from torch.cuda.amp import autocast, GradScaler
    HAS_AMP = True
except ImportError:
    HAS_AMP = False

# Configure for performance
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# =============================================================================
# CONFIGURATION SYSTEM
# =============================================================================

@dataclass
class ModelConfig:
    """Model architecture configuration."""
    vocab_size: int = 32000
    hidden_size: int = 2048
    num_layers: int = 24
    num_heads: int = 16
    num_kv_heads: int = 8  # For GQA
    seq_length: int = 2048
    intermediate_size: int = 5504  # SwiGLU expansion
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    dropout: float = 0.0
    
    def __post_init__(self):
        if self.intermediate_size is None:
            self.intermediate_size = int(8 * self.hidden_size / 3)
            # Round to multiple of 256 for efficiency
            self.intermediate_size = ((self.intermediate_size + 255) // 256) * 256
        
        assert self.hidden_size % self.num_heads == 0
        assert self.num_heads % self.num_kv_heads == 0

@dataclass  
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_steps: int = 10000
    warmup_steps: int = 1000
    eval_interval: int = 1000
    save_interval: int = 2000
    max_grad_norm: float = 1.0
    precision: str = "bf16"  # bf16, fp16, fp32
    compile: bool = True
    
@dataclass
class DataConfig:
    """Data processing configuration."""
    train_data_path: str = "oasst1_data/oasst1_train_conversations.jsonl"
    eval_data_path: str = "oasst1_data/oasst1_validation_conversations.jsonl"
    seq_length: int = 2048
    num_workers: int = 4
    vocab_size: int = 32000
    min_frequency: int = 2
    max_conversations: Optional[int] = None  # Limit for debugging

# =============================================================================
# CONVERSATION-AWARE TOKENIZER
# =============================================================================

class ConversationTokenizer:
    """Tokenizer specifically designed for conversational data."""
    
    def __init__(self):
        # Special tokens for conversation structure
        self.special_tokens = {
            "<pad>": 0,
            "<unk>": 1,
            "<s>": 2,
            "</s>": 3,
            "<|user|>": 4,
            "<|assistant|>": 5,
            "<|system|>": 6,
            "<|end|>": 7,
            "<|conversation_start|>": 8,
            "<|conversation_end|>": 9,
            "<|turn|>": 10,
        }
        
        self.vocab = self.special_tokens.copy()
        self.merges = []
        self._id_to_token = {v: k for k, v in self.vocab.items()}
        self._merge_cache = {}
        
        # Regex pattern for tokenization
        import re
        self.pattern = re.compile(
            r"'(?:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{2,}|[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
        )
    
    def train_from_conversations(self, conversations: List[Dict], vocab_size: int = 32000, min_freq: int = 2):
        """Train tokenizer specifically on conversational data."""
        logging.info(f"Training conversation tokenizer to vocab size {vocab_size}")
        
        # Extract all text from conversations
        texts = []
        for conv in conversations:
            for message in conv.get('messages', []):
                content = message.get('content', '').strip()
                if content:
                    texts.append(content)
        
        logging.info(f"Training on {len(texts)} message contents")
        
        # Standard BPE training
        word_freqs = Counter()
        for text in texts:
            words = self.pattern.findall(text)
            for word in words:
                if word.strip():
                    word_freqs[word + "</w>"] += 1
        
        # Filter by frequency
        word_freqs = {w: f for w, f in word_freqs.items() if f >= min_freq}
        logging.info(f"Found {len(word_freqs)} unique words after filtering")
        
        # Add characters to vocab
        chars = set()
        for word in word_freqs:
            chars.update(word)
        
        for char in sorted(chars):
            if char not in self.vocab and len(self.vocab) < vocab_size:
                self.vocab[char] = len(self.vocab)
                self._id_to_token[self.vocab[char]] = char
        
        # BPE training
        word_splits = {word: list(word) for word in word_freqs}
        target_merges = vocab_size - len(self.vocab)
        
        for merge_step in range(target_merges):
            pair_counts = defaultdict(int)
            
            for word, freq in word_freqs.items():
                splits = word_splits[word]
                for i in range(len(splits) - 1):
                    pair = (splits[i], splits[i + 1])
                    pair_counts[pair] += freq
            
            if not pair_counts:
                break
            
            best_pair = max(pair_counts.items(), key=lambda x: x[1])[0]
            merged_token = best_pair[0] + best_pair[1]
            
            if merged_token not in self.vocab:
                self.vocab[merged_token] = len(self.vocab)
                self._id_to_token[self.vocab[merged_token]] = merged_token
            
            self.merges.append(best_pair)
            
            # Apply merge
            for word in word_freqs:
                splits = word_splits[word]
                new_splits = []
                i = 0
                while i < len(splits):
                    if (i < len(splits) - 1 and 
                        (splits[i], splits[i + 1]) == best_pair):
                        new_splits.append(merged_token)
                        i += 2
                    else:
                        new_splits.append(splits[i])
                        i += 1
                word_splits[word] = new_splits
            
            if merge_step % 1000 == 0:
                logging.info(f"BPE step {merge_step}, vocab size: {len(self.vocab)}")
        
        logging.info(f"Tokenizer training complete. Final vocab size: {len(self.vocab)}")
    
    def encode_conversation(self, conversation: Dict, max_length: int = 2048) -> List[int]:
        """Encode a full conversation with special formatting."""
        tokens = [self.special_tokens["<|conversation_start|>"]]
        
        messages = conversation.get('messages', [])
        
        for i, message in enumerate(messages):
            role = message.get('role', '').lower()
            content = message.get('content', '').strip()
            
            if not content:
                continue
            
            # Add role token
            if role == 'prompter':
                tokens.append(self.special_tokens["<|user|>"])
            elif role == 'assistant':
                tokens.append(self.special_tokens["<|assistant|>"])
            else:
                tokens.append(self.special_tokens["<|system|>"])
            
            # Encode content
            content_tokens = self._encode_text(content)
            tokens.extend(content_tokens)
            
            # Add turn separator (except for last message)
            if i < len(messages) - 1:
                tokens.append(self.special_tokens["<|turn|>"])
        
        tokens.append(self.special_tokens["<|conversation_end|>"])
        
        # Truncate if too long
        if len(tokens) > max_length:
            tokens = tokens[:max_length - 1] + [self.special_tokens["</s>"]]
        
        return tokens
    
    def _encode_text(self, text: str) -> List[int]:
        """Encode regular text content."""
        if not text.strip():
            return []
        
        words = self.pattern.findall(text)
        tokens = []
        
        for word in words:
            if not word.strip():
                continue
            word_with_end = word + "</w>"
            subwords = self._apply_bpe(word_with_end)
            for subword in subwords:
                tokens.append(self.vocab.get(subword, self.special_tokens["<unk>"]))
        
        return tokens
    
    def _apply_bpe(self, word: str) -> List[str]:
        """Apply BPE merges to a word."""
        if word in self._merge_cache:
            return self._merge_cache[word]
        
        chars = list(word)
        if len(chars) <= 1:
            return chars
        
        for merge_pair in self.merges:
            if len(chars) <= 1:
                break
            
            i = 0
            new_chars = []
            while i < len(chars):
                if (i < len(chars) - 1 and 
                    (chars[i], chars[i + 1]) == merge_pair):
                    new_chars.append(chars[i] + chars[i + 1])
                    i += 2
                else:
                    new_chars.append(chars[i])
                    i += 1
            chars = new_chars
        
        if len(self._merge_cache) < 10000:
            self._merge_cache[word] = chars
        
        return chars
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text."""
        tokens = []
        for token_id in token_ids:
            if skip_special_tokens and token_id in self.special_tokens.values():
                continue
            tokens.append(self._id_to_token.get(token_id, "<unk>"))
        
        text = "".join(tokens)
        text = text.replace("</w>", " ")
        return text.strip()

# =============================================================================
# MODEL ARCHITECTURE (Same as before but included for completeness)
# =============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 8192, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len
        self._cos_cached = None
        self._sin_cached = None
        self._seq_len_cached = 0
    
    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self._seq_len_cached or self._cos_cached is None:
            self._seq_len_cached = max(seq_len, self._seq_len_cached)
            t = torch.arange(self._seq_len_cached, device=device, dtype=torch.float32)
            freqs = torch.outer(t, self.inv_freq.to(device))
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = emb.cos()
            self._sin_cached = emb.sin()
        return self._cos_cached[:seq_len], self._sin_cached[:seq_len]

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, 
                        cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class GroupedQueryAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        
        self.rope = RotaryEmbedding(self.head_dim, config.seq_length, config.rope_theta)
        
        for proj in [self.q_proj, self.k_proj, self.v_proj]:
            nn.init.xavier_uniform_(proj.weight)
        nn.init.xavier_uniform_(self.o_proj.weight, gain=1 / math.sqrt(2))
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, D = x.shape
        
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        cos, sin = self.rope(L, x.device)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        if self.num_queries_per_kv > 1:
            k = k[:, :, None, :, :].expand(B, self.num_kv_heads, self.num_queries_per_kv, L, self.head_dim)
            v = v[:, :, None, :, :].expand(B, self.num_kv_heads, self.num_queries_per_kv, L, self.head_dim)
            k = k.reshape(B, self.num_heads, L, self.head_dim)
            v = v.reshape(B, self.num_heads, L, self.head_dim)
        
        if HAS_FLASH_ATTN and x.is_cuda and mask is None:
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            out = flash_attn_func(q, k, v, causal=True)
            out = out.reshape(B, L, self.hidden_size)
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            
            if mask is None:
                mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1)
                scores.masked_fill_(mask, float('-inf'))
            else:
                scores = scores + mask
            
            attn = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
            out = torch.matmul(attn, v)
            out = out.transpose(1, 2).reshape(B, L, self.hidden_size)
        
        return self.o_proj(out)

class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        
        nn.init.xavier_uniform_(self.gate_proj.weight)
        nn.init.xavier_uniform_(self.up_proj.weight)
        nn.init.xavier_uniform_(self.down_proj.weight, gain=1 / math.sqrt(2))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)

class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.input_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = GroupedQueryAttention(config)
        self.post_attn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = SwiGLU(config)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.self_attn(self.input_norm(x), mask)
        x = x + self.mlp(self.post_attn_norm(x))
        return x

class TransformerModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        nn.init.normal_(self.embed_tokens.weight, std=0.02)
        nn.init.normal_(self.lm_head.weight, std=0.02)
        
        n_params = sum(p.numel() for p in self.parameters())
        logging.info(f"Model initialized with {n_params:,} parameters")
    
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        
        for layer in self.layers:
            x = layer(x, attention_mask)
        
        x = self.norm(x)
        return self.lm_head(x)

# =============================================================================
# CONVERSATIONAL DATASET
# =============================================================================

class ConversationDataset(Dataset):
    """Dataset specifically designed for conversational training."""
    
    def __init__(self, conversations: List[Dict], tokenizer: ConversationTokenizer, 
                 seq_length: int = 2048):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.examples = []
        
        logging.info("Processing conversations for training...")
        
        for conv in conversations:
            # Skip empty conversations
            if not conv.get('messages') or len(conv['messages']) < 2:
                continue
            
            # Encode the conversation
            tokens = tokenizer.encode_conversation(conv, max_length=seq_length)
            
            if len(tokens) >= 10:  # Minimum meaningful length
                self.examples.append(tokens)
        
        logging.info(f"Created dataset with {len(self.examples)} conversation examples")
        
        # Dataset statistics
        if self.examples:
            lengths = [len(ex) for ex in self.examples]
            logging.info(f"Token lengths - Mean: {np.mean(lengths):.1f}, "
                        f"Std: {np.std(lengths):.1f}, "
                        f"Min: {min(lengths)}, Max: {max(lengths)}")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        tokens = self.examples[idx]
        
        # Pad to sequence length
        if len(tokens) < self.seq_length:
            tokens = tokens + [0] * (self.seq_length - len(tokens))
        
        tokens = torch.tensor(tokens[:self.seq_length], dtype=torch.long)
        
        return {
            'input_ids': tokens[:-1],
            'labels': tokens[1:],
            'attention_mask': (tokens[:-1] != 0).float()
        }

def load_conversations(file_path: str, max_conversations: Optional[int] = None) -> List[Dict]:
    """Load conversations from JSONL file."""
    logging.info(f"Loading conversations from {file_path}")
    
    conversations = []
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        return conversations
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            try:
                conv = json.loads(line.strip())
                
                # Validate conversation structure
                if 'messages' not in conv or not conv['messages']:
                    continue
                
                # Check message format
                valid = True
                for msg in conv['messages']:
                    if 'role' not in msg or 'content' not in msg:
                        valid = False
                        break
                
                if valid:
                    conversations.append(conv)
                    
                if max_conversations and len(conversations) >= max_conversations:
                    break
                    
            except json.JSONDecodeError:
                logging.warning(f"Skipping invalid JSON at line {line_no}")
                continue
    
    logging.info(f"Loaded {len(conversations)} valid conversations")
    return conversations

# =============================================================================
# CONVERSATIONAL TRAINER
# =============================================================================

class ConversationTrainer:
    """Trainer specifically optimized for conversational data."""
    
    def __init__(self, model: TransformerModel, tokenizer: ConversationTokenizer,
                 config: TrainingConfig, device: torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        
        self.model = self.model.to(device)
        
        # Optimizer with different learning rates for different components
        param_groups = [
            {'params': [p for n, p in model.named_parameters() if 'embed' in n], 'lr': config.learning_rate * 0.1},
            {'params': [p for n, p in model.named_parameters() if 'embed' not in n], 'lr': config.learning_rate}
        ]
        
        self.optimizer = AdamW(
            param_groups,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95),
            fused=torch.cuda.is_available()
        )
        
        self.use_amp = HAS_AMP and config.precision != "fp32"
        self.scaler = GradScaler() if self.use_amp else None
        
        # Compile model
        if config.compile and hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model)
                logging.info("Model compiled successfully")
            except Exception as e:
                logging.warning(f"Model compilation failed: {e}")
        
        self.step = 0
        self.scheduler = None
    
    def train(self, train_dataset: ConversationDataset, eval_dataset: Optional[ConversationDataset] = None):
        """Main conversational training loop."""
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=[pg['lr'] for pg in self.optimizer.param_groups],
            total_steps=self.config.max_steps,
            pct_start=self.config.warmup_steps / self.config.max_steps
        )
        
        logging.info("Starting conversational training...")
        logging.info(f"Training steps: {self.config.max_steps}")
        logging.info(f"Batch size: {self.config.batch_size}")
        logging.info(f"Gradient accumulation: {self.config.gradient_accumulation_steps}")
        logging.info(f"Effective batch size: {self.config.batch_size * self.config.gradient_accumulation_steps}")
        
        self.model.train()
        accumulation_loss = 0.0
        conversation_count = 0
        
        while self.step < self.config.max_steps:
            for batch in train_loader:
                if self.step >= self.config.max_steps:
                    break
                
                batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
                
                with autocast(enabled=self.use_amp, 
                             dtype=torch.bfloat16 if self.config.precision == "bf16" else torch.float16):
                    logits = self.model(batch['input_ids'], batch['attention_mask'])
                    
                    # Compute loss with special token weighting
                    loss = self.compute_conversation_loss(logits, batch['labels'])
                    loss = loss / self.config.gradient_accumulation_steps
                
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                accumulation_loss += loss.item()
                conversation_count += batch['input_ids'].size(0)
                
                # Update weights
                if (self.step + 1) % self.config.gradient_accumulation_steps == 0:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                        self.optimizer.step()
                    
                    self.optimizer.zero_grad()
                    self.scheduler.step()
                    
                    # Logging
                    if (self.step + 1) % 100 == 0:
                        lr = self.scheduler.get_last_lr()[0]
                        conversations_per_sec = conversation_count / 100 if (self.step + 1) % 100 == 0 else 0
                        logging.info(f"Step {self.step + 1}: loss={accumulation_loss:.6f}, "
                                   f"lr={lr:.2e}, conv/s={conversations_per_sec:.1f}")
                        conversation_count = 0
                    
                    # Evaluation
                    if eval_dataset and (self.step + 1) % self.config.eval_interval == 0:
                        eval_loss = self.evaluate(eval_dataset)
                        logging.info(f"Eval loss: {eval_loss:.6f}")
                        self.model.train()
                    
                    # Save checkpoint
                    if (self.step + 1) % self.config.save_interval == 0:
                        self.save_checkpoint()
                    
                    accumulation_loss = 0.0
                
                self.step += 1
        
        logging.info("Conversational training completed!")
        self.save_checkpoint(final=True)
    
    def compute_conversation_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute loss with special weighting for conversational tokens."""
        # Standard cross-entropy loss
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=0,  # Ignore padding
            reduction='none'
        ).view_as(labels)
        
        # Apply weights to focus on important tokens
        weights = torch.ones_like(labels, dtype=torch.float)
        
        # Higher weight for assistant responses (tokens after <|assistant|>)
        assistant_token = self.tokenizer.special_tokens.get("<|assistant|>", -1)
        if assistant_token != -1:
            assistant_mask = (labels == assistant_token).float()
            # Create a forward-fill mask for assistant responses
            for i in range(1, labels.size(1)):
                assistant_mask[:, i] = torch.max(assistant_mask[:, i], assistant_mask[:, i-1] * 0.9)
            weights = weights + assistant_mask * 0.5  # 1.5x weight for assistant tokens
        
        # Apply weights and compute final loss
        weighted_loss = loss * weights
        mask = (labels != 0).float()  # Don't count padding tokens
        
        return weighted_loss.sum() / mask.sum().clamp(min=1)
    
    @torch.no_grad()
    def evaluate(self, eval_dataset: ConversationDataset) -> float:
        """Evaluate model on conversational eval dataset."""
        self.model.eval()
        
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        total_loss = 0.0
        num_batches = 0
        
        for batch in eval_loader:
            if num_batches >= 50:  # Limit eval time
                break
                
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
            
            with autocast(enabled=self.use_amp, 
                         dtype=torch.bfloat16 if self.config.precision == "bf16" else torch.float16):
                logits = self.model(batch['input_ids'], batch['attention_mask'])
                loss = self.compute_conversation_loss(logits, batch['labels'])
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def save_checkpoint(self, final: bool = False):
        """Save model checkpoint."""
        os.makedirs("checkpoints", exist_ok=True)
        suffix = "final" if final else f"step_{self.step}"
        path = f"checkpoints/conversation_model_{suffix}.pt"
        
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler else None,
            'step': self.step,
            'config': self.config,
            'tokenizer_vocab': self.tokenizer.vocab,
            'tokenizer_merges': self.tokenizer.merges
        }
        
        torch.save(checkpoint, path)
        logging.info(f"Saved checkpoint: {path}")
    
    @torch.no_grad()
    def generate_response(self, prompt: str, max_new_tokens: int = 256, 
                         temperature: float = 0.8, top_p: float = 0.9) -> str:
        """Generate a conversational response to a prompt."""
        self.model.eval()
        
        # Format prompt as conversation
        conversation = {
            'messages': [
                {'role': 'prompter', 'content': prompt}
            ]
        }
        
        # Encode input
        input_tokens = self.tokenizer.encode_conversation(conversation)
        
        # Add assistant token to start response
        input_tokens.append(self.tokenizer.special_tokens["<|assistant|>"])
        
        input_ids = torch.tensor([input_tokens], dtype=torch.long, device=self.device)
        
        # Generate
        generated = input_ids.clone()
        
        for _ in range(max_new_tokens):
            with autocast(enabled=self.use_amp, 
                         dtype=torch.bfloat16 if self.config.precision == "bf16" else torch.float16):
                logits = self.model(generated)
            
            next_token_logits = logits[0, -1, :] / temperature
            
            # Top-p sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
            
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
            
            # Stop at conversation end or turn token
            if next_token.item() in [
                self.tokenizer.special_tokens.get("<|conversation_end|>", -1),
                self.tokenizer.special_tokens.get("<|turn|>", -1),
                self.tokenizer.special_tokens.get("</s>", -1)
            ]:
                break
        
        # Extract response
        response_tokens = generated[0][len(input_tokens):].cpu().tolist()
        response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
        
        return response.strip()

# =============================================================================
# MAIN TRAINING SCRIPT
# =============================================================================

def main():
    """Main conversational training function."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('conversation_training.log'),
            logging.StreamHandler()
        ]
    )
    
    # Configuration
    model_config = ModelConfig(
        vocab_size=32000,  # Will be updated after tokenizer training
        hidden_size=1024,  # Smaller for faster training
        num_layers=12,
        num_heads=16,
        num_kv_heads=4,
        seq_length=2048,
        intermediate_size=2816
    )
    
    training_config = TrainingConfig(
        batch_size=2,  # Small batch for conversation data
        gradient_accumulation_steps=8,  # Larger effective batch
        learning_rate=5e-4,
        max_steps=20000,
        warmup_steps=2000,
        eval_interval=1000,
        save_interval=2500,
        precision="bf16" if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else "fp16"
    )
    
    data_config = DataConfig(
        max_conversations=10000  # Limit for faster experimentation
    )
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    if torch.cuda.is_available():
        logging.info(f"GPU: {torch.cuda.get_device_name()}")
        logging.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    # Load conversational data
    train_conversations = load_conversations(data_config.train_data_path, data_config.max_conversations)
    eval_conversations = load_conversations(data_config.eval_data_path, data_config.max_conversations // 10)
    
    if not train_conversations:
        logging.error("No training conversations loaded! Check your data paths.")
        logging.error(f"Expected: {data_config.train_data_path}")
        logging.error("Make sure you've run the dataset download script first.")
        return 1
    
    logging.info(f"Loaded {len(train_conversations)} training conversations")
    logging.info(f"Loaded {len(eval_conversations)} evaluation conversations")
    
    # Analyze conversation statistics
    analyze_conversation_data(train_conversations)
    
    # Initialize and train tokenizer
    tokenizer = ConversationTokenizer()
    tokenizer.train_from_conversations(
        train_conversations, 
        vocab_size=data_config.vocab_size, 
        min_freq=data_config.min_frequency
    )
    
    # Update model config with actual vocab size
    model_config.vocab_size = len(tokenizer.vocab)
    logging.info(f"Updated model vocab size to: {model_config.vocab_size}")
    
    # Create datasets
    train_dataset = ConversationDataset(train_conversations, tokenizer, data_config.seq_length)
    eval_dataset = ConversationDataset(eval_conversations, tokenizer, data_config.seq_length) if eval_conversations else None
    
    # Initialize model
    model = TransformerModel(model_config)
    
    # Initialize trainer
    trainer = ConversationTrainer(model, tokenizer, training_config, device)
    
    # Start training
    logging.info("="*60)
    logging.info("STARTING CONVERSATIONAL TRAINING")
    logging.info("="*60)
    
    trainer.train(train_dataset, eval_dataset)
    
    # Save tokenizer
    save_tokenizer(tokenizer)
    
    # Test generation
    test_generation(trainer)
    
    logging.info("Conversational training completed successfully!")
    return 0

def analyze_conversation_data(conversations: List[Dict]):
    """Analyze the structure of conversational data."""
    logging.info("Analyzing conversational data...")
    
    total_messages = 0
    role_counts = Counter()
    turn_counts = []
    
    for conv in conversations:
        messages = conv.get('messages', [])
        turn_counts.append(len(messages))
        total_messages += len(messages)
        
        for msg in messages:
            role_counts[msg.get('role', 'unknown')] += 1
    
    logging.info(f"Total conversations: {len(conversations)}")
    logging.info(f"Total messages: {total_messages}")
    logging.info(f"Average turns per conversation: {np.mean(turn_counts):.1f}")
    logging.info(f"Turn distribution - Min: {min(turn_counts)}, Max: {max(turn_counts)}")
    
    logging.info("Role distribution:")
    for role, count in role_counts.most_common():
        logging.info(f"  {role}: {count:,} ({count/total_messages*100:.1f}%)")

def save_tokenizer(tokenizer: ConversationTokenizer):
    """Save tokenizer vocabulary and merges."""
    os.makedirs("checkpoints", exist_ok=True)
    
    with open('checkpoints/conversation_tokenizer_vocab.json', 'w') as f:
        json.dump(tokenizer.vocab, f, indent=2)
    
    with open('checkpoints/conversation_tokenizer_merges.txt', 'w') as f:
        f.write("#version: 0.2\n")
        for merge in tokenizer.merges:
            f.write(f"{merge[0]} {merge[1]}\n")
    
    logging.info("Saved conversation tokenizer files")

def test_generation(trainer: ConversationTrainer):
    """Test text generation with the trained model."""
    logging.info("Testing conversation generation...")
    
    test_prompts = [
        "How can I learn Python programming?",
        "What's the weather like today?",
        "Explain quantum computing in simple terms.",
        "How do I make a good impression in a job interview?"
    ]
    
    for prompt in test_prompts:
        logging.info(f"\nPrompt: {prompt}")
        response = trainer.generate_response(prompt, max_new_tokens=128, temperature=0.8)
        logging.info(f"Response: {response}")

# =============================================================================
# CONFIGURATION PRESETS FOR CONVERSATIONS
# =============================================================================

class ConversationConfigPresets:
    """Predefined configurations optimized for conversational training."""
    
    @staticmethod
    def debug_tiny() -> Tuple[ModelConfig, TrainingConfig, DataConfig]:
        """Tiny model for debugging conversational training."""
        model_config = ModelConfig(
            vocab_size=5000,
            hidden_size=256,
            num_layers=4,
            num_heads=8,
            num_kv_heads=2,
            seq_length=512,
            intermediate_size=688
        )
        
        training_config = TrainingConfig(
            batch_size=1,
            gradient_accumulation_steps=2,
            max_steps=500,
            warmup_steps=50,
            eval_interval=100,
            save_interval=250,
            precision="fp32",
            compile=False
        )
        
        data_config = DataConfig(
            seq_length=512,
            vocab_size=5000,
            max_conversations=100
        )
        
        return model_config, training_config, data_config
    
    @staticmethod
    def small_conversational() -> Tuple[ModelConfig, TrainingConfig, DataConfig]:
        """Small model optimized for conversation."""
        model_config = ModelConfig(
            vocab_size=32000,
            hidden_size=1024,
            num_layers=12,
            num_heads=16,
            num_kv_heads=4,
            seq_length=2048,
            intermediate_size=2816
        )
        
        training_config = TrainingConfig(
            batch_size=2,
            gradient_accumulation_steps=8,
            learning_rate=5e-4,
            max_steps=20000,
            warmup_steps=2000,
            eval_interval=1000,
            save_interval=2500,
            precision="bf16"
        )
        
        data_config = DataConfig(
            seq_length=2048,
            vocab_size=32000,
            max_conversations=10000
        )
        
        return model_config, training_config, data_config
    
    @staticmethod
    def production_conversational() -> Tuple[ModelConfig, TrainingConfig, DataConfig]:
        """Production-ready conversational model."""
        model_config = ModelConfig(
            vocab_size=32000,
            hidden_size=2048,
            num_layers=24,
            num_heads=16,
            num_kv_heads=8,
            seq_length=4096,
            intermediate_size=5504
        )
        
        training_config = TrainingConfig(
            batch_size=1,
            gradient_accumulation_steps=16,
            learning_rate=3e-4,
            max_steps=100000,
            warmup_steps=5000,
            eval_interval=2000,
            save_interval=10000,
            precision="bf16"
        )
        
        data_config = DataConfig(
            seq_length=4096,
            vocab_size=32000,
            max_conversations=None  # Use all data
        )
        
        return model_config, training_config, data_config

# =============================================================================
# UTILITIES AND HELPERS
# =============================================================================

def load_checkpoint(checkpoint_path: str, model: TransformerModel, 
                   optimizer: torch.optim.Optimizer = None) -> Dict:
    """Load model checkpoint."""
    logging.info(f"Loading checkpoint from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    
    if optimizer and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    return checkpoint

def estimate_training_time(num_conversations: int, config: TrainingConfig, 
                          device_name: str = "unknown") -> str:
    """Estimate training time based on configuration."""
    # Rough estimates based on empirical data
    base_time_per_step = {
        'cpu': 2.0,
        'gpu_slow': 0.5,  # Older GPUs
        'gpu_fast': 0.1   # Modern GPUs like A100
    }
    
    device_type = 'cpu'
    if 'cuda' in str(device_name).lower():
        if any(gpu in device_name.lower() for gpu in ['a100', 'h100', '4090']):
            device_type = 'gpu_fast'
        else:
            device_type = 'gpu_slow'
    
    time_per_step = base_time_per_step[device_type]
    total_seconds = config.max_steps * time_per_step
    
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    
    return f"Estimated training time: {hours:.0f}h {minutes:.0f}m (on {device_type})"

def create_conversation_from_text(user_text: str, assistant_text: str) -> Dict:
    """Helper to create a conversation dict from text pairs."""
    return {
        'conversation_id': 'synthetic',
        'messages': [
            {'role': 'prompter', 'content': user_text, 'turn': 1},
            {'role': 'assistant', 'content': assistant_text, 'turn': 2}
        ],
        'total_turns': 2
    }

if __name__ == "__main__":
    exit(main())