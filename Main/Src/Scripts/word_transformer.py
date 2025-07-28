# Copyright (c) 2025 Matias Nielsen. All rights reserved.
# Licensed under the Custom License below.

import math
import re
import logging
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class WordTokenizer:
    """Professional word-level tokenizer with vocabulary management."""
    
    def __init__(self, vocab: Optional[Dict[str, int]] = None):
        if vocab is None:
            # Initialize with basic special tokens
            self.vocab = {
                "<pad>": 0,
                "<unk>": 1,
                "<s>": 2,    # Start token
                "</s>": 3,   # End token
                "<user>": 4, # User message
                "<bot>": 5,  # Bot message
            }
            self.next_id = 6
        else:
            self.vocab = vocab.copy()
            self.next_id = max(vocab.values()) + 1 if vocab else 6
        
        # Create reverse mapping
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        logger.info(f"WordTokenizer initialized with {len(self.vocab)} tokens")
    
    def train_from_text(self, text: str, vocab_size: int = 32000, min_freq: int = 2) -> None:
        """Train tokenizer on text corpus with frequency-based vocabulary."""
        logger.info(f"Training tokenizer on {len(text):,} characters...")
        
        # Basic text cleaning and tokenization
        text = text.lower()
        
        # Extract words using regex (handles punctuation separately)
        words = re.findall(r'\w+|[^\w\s]', text)
        
        # Count word frequencies
        word_counts = Counter(words)
        logger.info(f"Found {len(word_counts):,} unique tokens")
        
        # Filter by minimum frequency and sort by frequency
        filtered_words = [(word, count) for word, count in word_counts.items() 
                         if count >= min_freq and word not in self.vocab]
        filtered_words.sort(key=lambda x: x[1], reverse=True)
        
        # Add most frequent words to vocabulary
        available_slots = vocab_size - len(self.vocab)
        words_to_add = filtered_words[:available_slots]
        
        for word, count in words_to_add:
            self.vocab[word] = self.next_id
            self.id_to_token[self.next_id] = word
            self.next_id += 1
        
        logger.info(f"Final vocabulary size: {len(self.vocab):,}")
        logger.info(f"Added {len(words_to_add):,} new words")
        
        if words_to_add:
            logger.info(f"Most frequent new words: {[w for w, _ in words_to_add[:10]]}")
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        if not text:
            return []
        
        # Tokenize text
        text = text.lower()
        tokens = re.findall(r'\w+|[^\w\s]', text)
        
        # Convert to IDs
        token_ids = []
        unk_id = self.vocab.get("<unk>", 1)
        
        for token in tokens:
            token_id = self.vocab.get(token, unk_id)
            token_ids.append(token_id)
        
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text."""
        if not token_ids:
            return ""
        
        tokens = []
        for token_id in token_ids:
            token = self.id_to_token.get(token_id, "<unk>")
            # Skip special tokens except spaces
            if token not in ["<pad>", "<s>", "</s>"]:
                tokens.append(token)
        
        # Reconstruct text with proper spacing
        text = ""
        for i, token in enumerate(tokens):
            if i > 0 and token.isalnum() and tokens[i-1].isalnum():
                text += " "
            text += token
        
        return text.strip()
    
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return len(self.vocab)
    
    def get_vocab(self) -> Dict[str, int]:
        """Return vocabulary dictionary."""
        return self.vocab.copy()
    
    def save_vocab(self, path: str) -> None:
        """Save vocabulary to file."""
        import json
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, indent=2, ensure_ascii=False)
        logger.info(f"Vocabulary saved to {path}")
    
    def load_vocab(self, path: str) -> None:
        """Load vocabulary from file."""
        import json
        with open(path, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.next_id = max(self.vocab.values()) + 1 if self.vocab else 0
        logger.info(f"Vocabulary loaded from {path}: {len(self.vocab)} tokens")

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism with scaled dot-product attention."""
    
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert hidden_size % num_heads == 0
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.scale = math.sqrt(self.head_size)
        
        # Linear projections for Q, K, V
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with proper scaling."""
        for module in [self.query, self.key, self.value, self.out_proj]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, hidden_size = x.shape
        
        # Linear projections
        q = self.query(x)  # [batch_size, seq_len, hidden_size]
        k = self.key(x)
        v = self.value(x)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        # Apply causal mask (for autoregressive generation)
        if mask is None:
            # Create causal mask
            causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            causal_mask = causal_mask.to(x.device)
            scores.masked_fill_(causal_mask, float('-inf'))
        else:
            scores.masked_fill_(mask, float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        output = self.out_proj(attn_output)
        
        return output

class FeedForward(nn.Module):
    """Position-wise feed-forward network with GELU activation."""
    
    def __init__(self, hidden_size: int, intermediate_size: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        if intermediate_size is None:
            intermediate_size = 4 * hidden_size
        
        self.linear1 = nn.Linear(hidden_size, intermediate_size)
        self.linear2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.zeros_(self.linear2.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class TransformerBlock(nn.Module):
    """Transformer block with multi-head attention and feed-forward layers."""
    
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        
        # Multi-head attention
        self.attention = MultiHeadAttention(hidden_size, num_heads, dropout)
        
        # Feed-forward network
        self.feed_forward = FeedForward(hidden_size, dropout=dropout)
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(hidden_size, eps=1e-5)
        self.ln2 = nn.LayerNorm(hidden_size, eps=1e-5)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual connection and layer norm
        residual = x
        x = self.ln1(x)
        x = self.attention(x, mask)
        x = self.dropout(x)
        x = residual + x
        
        # Feed-forward with residual connection and layer norm
        residual = x
        x = self.ln2(x)
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = residual + x
        
        return x

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer models."""
    
    def __init__(self, hidden_size: int, max_seq_length: int = 10000):
        super().__init__()
        
        pe = torch.zeros(max_seq_length, hidden_size)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * 
                           (-math.log(10000.0) / hidden_size))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :].unsqueeze(0)

class WordTransformer(nn.Module):
    """Word-level transformer model for language generation."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token embedding
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(config.hidden_size, config.seq_length)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config.hidden_size, config.num_heads, config.dropout)
            for _ in range(config.num_layers)
        ])
        
        # Final layer norm
        self.ln_final = nn.LayerNorm(config.hidden_size, eps=1e-5)
        
        # Language modeling head (tied with token embedding)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Tie weights between embedding and lm_head
        self.lm_head.weight = self.token_embedding.weight
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialize weights
        self._init_weights()
        
        logger.info(f"WordTransformer initialized:")
        logger.info(f"  Vocabulary: {config.vocab_size:,}")
        logger.info(f"  Hidden size: {config.hidden_size}")
        logger.info(f"  Layers: {config.num_layers}")
        logger.info(f"  Attention heads: {config.num_heads}")
        logger.info(f"  Sequence length: {config.seq_length}")
        logger.info(f"  Parameters: {self.count_parameters():,}")
    
    def _init_weights(self):
        """Initialize model weights with proper scaling."""
        # Token embedding
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        
        # Apply initialization to all modules
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def count_parameters(self) -> int:
        """Count total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the transformer model.
        
        Args:
            input_ids: Token IDs of shape [batch_size, seq_len]
            attention_mask: Optional attention mask of shape [batch_size, seq_len]
        
        Returns:
            Logits of shape [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape
        
        # Check sequence length
        if seq_len > self.config.seq_length:
            raise ValueError(f"Input sequence length ({seq_len}) exceeds maximum ({self.config.seq_length})")
        
        # Token embeddings
        x = self.token_embedding(input_ids)  # [batch_size, seq_len, hidden_size]
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x, attention_mask)
        
        # Final layer normalization
        x = self.ln_final(x)
        
        # Language modeling head
        logits = self.lm_head(x)  # [batch_size, seq_len, vocab_size]
        
        return logits
    
    def generate(self, input_ids: torch.Tensor, max_length: int = 100, 
                temperature: float = 1.0, top_k: int = 50, top_p: float = 0.9,
                do_sample: bool = True, pad_token_id: int = 0) -> torch.Tensor:
        """
        Generate text using the model.
        
        Args:
            input_ids: Starting token IDs [batch_size, seq_len]
            max_length: Maximum length to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling or greedy decoding
            pad_token_id: ID of padding token
        
        Returns:
            Generated token IDs [batch_size, generated_length]
        """
        self.eval()
        
        with torch.no_grad():
            batch_size = input_ids.size(0)
            generated = input_ids.clone()
            
            for _ in range(max_length):
                # Get model predictions
                logits = self.forward(generated)
                next_token_logits = logits[:, -1, :] / temperature
                
                if do_sample:
                    # Apply top-k filtering
                    if top_k > 0:
                        indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                        next_token_logits[indices_to_remove] = float('-inf')
                    
                    # Apply nucleus (top-p) filtering
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        
                        # Remove tokens with cumulative probability above the threshold
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        
                        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                        next_token_logits[indices_to_remove] = float('-inf')
                    
                    # Sample from the filtered distribution
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1)
                else:
                    # Greedy decoding
                    next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_tokens], dim=1)
                
                # Check if we've hit sequence length limit
                if generated.size(1) >= self.config.seq_length:
                    break
        
        return generated
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        return {
            'model_type': 'WordTransformer',
            'vocab_size': self.config.vocab_size,
            'hidden_size': self.config.hidden_size,
            'num_layers': self.config.num_layers,
            'num_heads': self.config.num_heads,
            'seq_length': self.config.seq_length,
            'dropout': self.config.dropout,
            'total_parameters': self.count_parameters(),
            'model_size_mb': self.count_parameters() * 4 / (1024 * 1024),  # Assuming float32
        }
    
    def load_pretrained_weights(self, pretrained_path: str, strict: bool = True):
        """Load pretrained weights from a checkpoint."""
        try:
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Load weights
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=strict)
            
            if missing_keys:
                logger.warning(f"Missing keys in checkpoint: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys in checkpoint: {unexpected_keys}")
            
            logger.info(f"Loaded pretrained weights from {pretrained_path}")
            
        except Exception as e:
            logger.error(f"Failed to load pretrained weights: {e}")
            raise
    
    def save_checkpoint(self, path: str, optimizer=None, scheduler=None, epoch=None, loss=None):
        """Save model checkpoint with additional training information."""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'model_config': self.config.__dict__ if hasattr(self.config, '__dict__') else self.config,
            'model_info': self.get_model_info(),
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        if epoch is not None:
            checkpoint['epoch'] = epoch
        
        if loss is not None:
            checkpoint['loss'] = loss
        
        torch.save(checkpoint, path)
        logger.info(f"Model checkpoint saved to {path}")
    
    @classmethod
    def from_pretrained(cls, model_path: str, config=None):
        """Load model from pretrained checkpoint."""
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Get config from checkpoint or use provided config
            if config is None:
                if 'model_config' in checkpoint:
                    from model_manager import ModelConfig
                    config_dict = checkpoint['model_config']
                    config = ModelConfig(**config_dict)
                else:
                    raise ValueError("No config provided and none found in checkpoint")
            
            # Create model
            model = cls(config)
            
            # Load state dict
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            logger.info(f"Model loaded from {model_path}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            raise