# Copyright (c) 2025 Matias Nielsen. All rights reserved.
# Licensed under the Custom License below.

#!/usr/bin/env python3
"""
LuminaAI Desktop App - Backend Server
Ultra-modern neural interface with Flask backend for Electron frontend.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import threading
import sys
import os
from pathlib import Path
import json
import logging
import gc
from typing import Dict, Optional, List, Tuple, Union
import time
import re
import math
import pickle
from collections import Counter
import unicodedata
import subprocess
import signal
import atexit

# Import the core AI components - with graceful fallback
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError as e:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    np = None

# Setup logging with better formatting
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('lumina.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Flask app initialization
app = Flask(__name__)
app.config['SECRET_KEY'] = 'lumina_ai_neural_interface_2025'
app.config['JSON_SORT_KEYS'] = False
CORS(app, origins=["http://localhost:3000", "app://."])  # Allow Electron
socketio = SocketIO(
    app, 
    cors_allowed_origins=["http://localhost:3000", "app://.", "*"], 
    async_mode='threading',
    ping_timeout=60,
    ping_interval=25
)

class WordTokenizer:
    """Advanced word tokenizer based on ChatAI.py implementation."""
    
    def __init__(self):
        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab_size_val = 0
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,  # Beginning of sequence
            '<EOS>': 3,  # End of sequence
            '<USER>': 4,  # User message marker
            '<BOT>': 5,   # Bot message marker
            '<SEP>': 6    # Separator token
        }
        self.initialize_special_tokens()
        logger.debug("WordTokenizer initialized")
    
    def initialize_special_tokens(self):
        """Initialize special tokens in vocabulary."""
        for token, idx in self.special_tokens.items():
            self.word_to_id[token] = idx
            self.id_to_word[idx] = token
        self.vocab_size_val = len(self.special_tokens)
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for tokenization."""
        if not text:
            return ""
        
        try:
            # Normalize unicode
            text = unicodedata.normalize('NFKC', text)
            
            # Basic cleaning
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
            
            return text
        except Exception as e:
            logger.warning(f"Error preprocessing text: {e}")
            return str(text).strip()
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words with proper handling."""
        text = self.preprocess_text(text)
        if not text:
            return []
        
        try:
            # Split on whitespace and punctuation
            tokens = re.findall(r'\w+|[^\w\s]', text.lower())
            return tokens
        except Exception as e:
            logger.warning(f"Error tokenizing text: {e}")
            return text.lower().split()
    
    def build_vocab(self, texts: List[str], min_freq: int = 2, max_vocab: int = 50000):
        """Build vocabulary from texts."""
        logger.info("Building vocabulary...")
        
        try:
            # Count word frequencies
            word_counts = Counter()
            for text in texts:
                tokens = self.tokenize(text)
                word_counts.update(tokens)
            
            # Filter by frequency and limit vocab size
            vocab_words = [
                word for word, count in word_counts.most_common(max_vocab - len(self.special_tokens)) 
                if count >= min_freq
            ]
            
            # Add to vocabulary
            current_id = len(self.special_tokens)
            for word in vocab_words:
                if word not in self.word_to_id:
                    self.word_to_id[word] = current_id
                    self.id_to_word[current_id] = word
                    current_id += 1
            
            self.vocab_size_val = len(self.word_to_id)
            logger.info(f"Built vocabulary with {self.vocab_size_val} tokens")
            
        except Exception as e:
            logger.error(f"Error building vocabulary: {e}")
            raise
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        try:
            tokens = self.tokenize(text)
            return [self.word_to_id.get(token, self.special_tokens['<UNK>']) for token in tokens]
        except Exception as e:
            logger.warning(f"Error encoding text: {e}")
            return [self.special_tokens['<UNK>']]
    
    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text."""
        try:
            tokens = [self.id_to_word.get(id, '<UNK>') for id in ids if id in self.id_to_word]
            # Remove special tokens from output
            tokens = [token for token in tokens if not (token.startswith('<') and token.endswith('>'))]
            return ' '.join(tokens)
        except Exception as e:
            logger.warning(f"Error decoding tokens: {e}")
            return ""
    
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return self.vocab_size_val
    
    def save(self, path: Path):
        """Save tokenizer to disk."""
        try:
            data = {
                'word_to_id': self.word_to_id,
                'id_to_word': self.id_to_word,
                'vocab_size': self.vocab_size_val,
                'special_tokens': self.special_tokens
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Tokenizer saved to {path}")
        except Exception as e:
            logger.error(f"Error saving tokenizer: {e}")
            raise
    
    def load(self, path: Path):
        """Load tokenizer from disk."""
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            
            self.word_to_id = data['word_to_id']
            self.id_to_word = data['id_to_word']
            self.vocab_size_val = data['vocab_size']
            self.special_tokens = data.get('special_tokens', self.special_tokens)
            logger.info(f"Tokenizer loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading tokenizer: {e}")
            raise

class WordTransformer(nn.Module if TORCH_AVAILABLE else object):
    """Word-level transformer model based on ChatAI.py architecture."""
    
    def __init__(self, vocab_size: int, hidden_size: int, num_layers: int, 
                 num_heads: int, seq_length: int, dropout: float = 0.1):
        if not TORCH_AVAILABLE:
            return
        super().__init__()
        
        # Validate parameters
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})")
        
        self.config = type('Config', (), {
            'vocab_size': vocab_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'num_heads': num_heads,
            'seq_length': seq_length,
            'dropout': dropout
        })()
        
        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(seq_length, hidden_size)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output layers
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(hidden_size, vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)
        logger.debug(f"WordTransformer initialized: {vocab_size} vocab, {hidden_size} hidden, {num_layers} layers")
    
    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        """Forward pass."""
        if not TORCH_AVAILABLE:
            return None
        
        try:
            batch_size, seq_len = input_ids.shape
            device = input_ids.device
            
            # Validate sequence length
            if seq_len > self.config.seq_length:
                logger.warning(f"Input sequence length {seq_len} exceeds max length {self.config.seq_length}")
                input_ids = input_ids[:, -self.config.seq_length:]
                seq_len = self.config.seq_length
            
            # Create position IDs
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
            
            # Embeddings
            token_embeds = self.token_embedding(input_ids)
            position_embeds = self.position_embedding(position_ids)
            hidden_states = token_embeds + position_embeds
            hidden_states = self.dropout(hidden_states)
            
            # Create causal mask
            causal_mask = self._generate_square_subsequent_mask(seq_len).to(device)
            
            # Transformer
            hidden_states = self.transformer(hidden_states, mask=causal_mask)
            hidden_states = self.layer_norm(hidden_states)
            
            # Output projection
            logits = self.output_projection(hidden_states)
            
            return logits
            
        except Exception as e:
            logger.error(f"Error in model forward pass: {e}")
            raise
    
    def _generate_square_subsequent_mask(self, sz: int):
        """Generate causal mask."""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        return mask.masked_fill(mask == 1, float('-inf'))

def setup_device():
    """Setup the best available device."""
    if not TORCH_AVAILABLE:
        return None
    try:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps")
            logger.info("Using device: MPS (Apple Silicon)")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using device: CUDA ({torch.cuda.get_device_name()})")
        else:
            device = torch.device("cpu")
            logger.info("Using device: CPU")
        return device
    except Exception as e:
        logger.warning(f"Error setting up device: {e}. Using CPU.")
        return torch.device("cpu")

# Sampling functions with better error handling
def nucleus_sampling(probs: torch.Tensor, p: float = 0.9) -> int:
    """Nucleus (top-p) sampling for better text generation."""
    if not TORCH_AVAILABLE:
        return 0
    
    try:
        if p <= 0 or p >= 1:
            return torch.multinomial(probs, 1).item()
        
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumsum_probs = torch.cumsum(sorted_probs, dim=0)
        cutoff_mask = cumsum_probs <= p
        
        if not cutoff_mask.any():
            cutoff = 1
        else:
            cutoff = max(1, cutoff_mask.sum().item())
        
        top_p_probs = sorted_probs[:cutoff]
        top_p_indices = sorted_indices[:cutoff]
        
        # Normalize probabilities
        if top_p_probs.sum() > 0:
            top_p_probs = top_p_probs / top_p_probs.sum()
        else:
            top_p_probs = torch.ones_like(top_p_probs) / len(top_p_probs)
        
        # Sample
        chosen_idx = torch.multinomial(top_p_probs, 1).item()
        return top_p_indices[chosen_idx].item()
        
    except Exception as e:
        logger.warning(f"Error in nucleus sampling: {e}")
        return torch.argmax(probs).item()

def top_k_sampling(probs: torch.Tensor, k: int = 50) -> int:
    """Top-k sampling for controlled text generation."""
    if not TORCH_AVAILABLE:
        return 0
    
    try:
        if k <= 0 or k >= len(probs):
            return torch.multinomial(probs, 1).item()
        
        top_k_probs, top_k_indices = torch.topk(probs, k)
        
        # Normalize probabilities
        if top_k_probs.sum() > 0:
            top_k_probs = top_k_probs / top_k_probs.sum()
        else:
            top_k_probs = torch.ones_like(top_k_probs) / k
        
        # Sample
        chosen_idx = torch.multinomial(top_k_probs, 1).item()
        return top_k_indices[chosen_idx].item()
        
    except Exception as e:
        logger.warning(f"Error in top-k sampling: {e}")
        return top_k_indices[0].item() if len(top_k_indices) > 0 else 0

def clean_response(response: str) -> str:
    """Clean up the model's response."""
    if not response:
        return ""
    
    try:
        # Remove special tokens
        response = re.sub(r'<[^>]*>', '', response)
        
        # Remove excessive whitespace
        response = re.sub(r'\s+', ' ', response)
        response = response.strip()
        
        # Remove incomplete sentences at the end
        if response:
            sentences = re.split(r'[.!?]+', response)
            if len(sentences) > 1:
                last_sentence = sentences[-1].strip()
                if len(last_sentence) < 5:
                    response_parts = response.rsplit(last_sentence, 1)
                    if len(response_parts) > 1:
                        response = response_parts[0].strip()
                        if response and response[-1] not in '.!?':
                            response += '.'
        
        return response
        
    except Exception as e:
        logger.warning(f"Error cleaning response: {e}")
        return str(response).strip()

class ModernAIEngine:
    """Modern AI engine with word tokenization."""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = setup_device() if TORCH_AVAILABLE else None
        self.is_loaded = False
        self.model_info = {}
        self.conversation_history = []
        self.max_history_length = 2000
        self._lock = threading.Lock()
        logger.info("ModernAIEngine initialized")
    
    def load_model(self, model_path: str) -> Tuple[bool, str]:
        """Load model with word tokenization."""
        if not TORCH_AVAILABLE:
            return False, "PyTorch not available. Please install requirements: pip install torch numpy"
        
        with self._lock:
            try:
                model_path = Path(model_path)
                if not model_path.exists():
                    return False, f"Model file not found: {model_path}"
                
                logger.info(f"Loading model from: {model_path}")
                
                # Load checkpoint with proper error handling
                try:
                    checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                except Exception as e:
                    return False, f"Error loading checkpoint: {e}"
                
                # Load tokenizer
                tokenizer_path = model_path.parent / "tokenizer.pkl"
                if tokenizer_path.exists():
                    self.tokenizer = WordTokenizer()
                    self.tokenizer.load(tokenizer_path)
                else:
                    return False, "Tokenizer file not found. Please ensure tokenizer.pkl exists."
                
                # Extract model configuration
                if 'config' in checkpoint:
                    config = checkpoint['config']
                else:
                    # Try to infer config from model state dict
                    return False, "Model configuration not found in checkpoint"
                
                # Validate configuration
                required_keys = ['vocab_size', 'hidden_size', 'num_layers', 'num_heads', 'seq_length']
                missing_keys = [key for key in required_keys if key not in config]
                if missing_keys:
                    return False, f"Missing config keys: {missing_keys}"
                
                # Initialize model
                try:
                    self.model = WordTransformer(
                        vocab_size=config['vocab_size'],
                        hidden_size=config['hidden_size'],
                        num_layers=config['num_layers'],
                        num_heads=config['num_heads'],
                        seq_length=config['seq_length'],
                        dropout=config.get('dropout', 0.1)
                    ).to(self.device)
                    
                    self.model.load_state_dict(checkpoint["model_state_dict"])
                    self.model.eval()
                    
                except Exception as e:
                    return False, f"Error initializing model: {e}"
                
                # Store model info
                self.model_info = {
                    'vocab_size': config['vocab_size'],
                    'hidden_size': config['hidden_size'],
                    'num_layers': config['num_layers'],
                    'num_heads': config['num_heads'],
                    'seq_length': config['seq_length'],
                    'epoch': checkpoint.get('epoch', 'Unknown'),
                    'loss': checkpoint.get('loss', 'Unknown'),
                    'accuracy': checkpoint.get('accuracy', 'Unknown'),
                    'device': str(self.device),
                    'parameters': sum(p.numel() for p in self.model.parameters()),
                    'model_size_mb': sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 * 1024)
                }
                
                self.is_loaded = True
                self.conversation_history = []  # Reset conversation history
                logger.info("Model loaded successfully!")
                return True, "Model loaded successfully!"
                
            except Exception as e:
                error_msg = f"Error loading model: {e}"
                logger.error(error_msg)
                return False, error_msg
    
    def generate_response(self, user_input: str, temperature: float = 0.8,
                         sampling_method: str = "top_k", top_k: int = 50, 
                         top_p: float = 0.9, max_length: int = 150) -> str:
        """Generate AI response using word-level model."""
        if not self.is_loaded or not TORCH_AVAILABLE:
            return "❌ Model not loaded or PyTorch not available."
        
        if not user_input.strip():
            return "Please enter a message."
        
        with self._lock:
            try:
                with torch.no_grad():
                    # Build context from conversation history
                    context_parts = self.conversation_history[-6:] if self.conversation_history else []
                    context_parts.append(f"<USER> {user_input}")
                    context_parts.append("<BOT>")
                    context = " ".join(context_parts)
                    
                    # Encode input
                    encoded_ids = self.tokenizer.encode(context)
                    if not encoded_ids:
                        return "Unable to process input."
                    
                    input_ids = torch.tensor(encoded_ids, dtype=torch.long).unsqueeze(0).to(self.device)
                    generated = input_ids.clone()
                    
                    # Generation loop
                    for step in range(max_length):
                        # Use sliding window for long sequences
                        max_seq_length = self.model.config.seq_length
                        input_seq = generated[:, -max_seq_length:] if generated.size(1) > max_seq_length else generated
                        
                        # Forward pass
                        try:
                            logits = self.model(input_seq)
                            next_token_logits = logits[0, -1, :] / max(temperature, 0.1)
                        except Exception as e:
                            logger.error(f"Error in model forward pass: {e}")
                            break
                        
                        # Apply softmax
                        probs = F.softmax(next_token_logits, dim=0)
                        
                        # Handle NaN/Inf
                        if torch.isnan(probs).any() or torch.isinf(probs).any():
                            logger.warning("NaN/Inf detected in probabilities")
                            probs = torch.ones_like(probs) / len(probs)
                        
                        # Sample next token
                        try:
                            if sampling_method == "nucleus" or sampling_method == "top_p":
                                next_token_id = nucleus_sampling(probs, p=top_p)
                            elif sampling_method == "top_k":
                                next_token_id = top_k_sampling(probs, k=top_k)
                            elif sampling_method == "greedy":
                                next_token_id = torch.argmax(probs).item()
                            else:
                                next_token_id = top_k_sampling(probs, k=top_k)
                        except Exception as e:
                            logger.warning(f"Error in sampling: {e}")
                            next_token_id = torch.argmax(probs).item()
                        
                        # Validate token ID
                        if next_token_id < 0 or next_token_id >= self.tokenizer.vocab_size():
                            logger.warning(f"Invalid token ID: {next_token_id}")
                            break
                        
                        # Add to generated sequence
                        generated = torch.cat([generated, torch.tensor([[next_token_id]], device=self.device)], dim=1)
                        
                        # Stop conditions
                        if next_token_id == self.tokenizer.special_tokens.get("<EOS>", -1):
                            break
                        
                        # Natural stopping points
                        current_token = self.tokenizer.id_to_word.get(next_token_id, "")
                        if current_token in [".", "!", "?"] and step > 10:
                            if step > 20:
                                break
                    
                    # Decode response
                    response_ids = generated[0][input_ids.size(1):].tolist()
                    response = self.tokenizer.decode(response_ids)
                    response = clean_response(response)
                    
                    # Update conversation history
                    if response.strip():
                        self.conversation_history.append(f"<USER> {user_input}")
                        self.conversation_history.append(f"<BOT> {response}")
                        
                        # Trim history to prevent memory issues
                        if len(self.conversation_history) > 10:
                            self.conversation_history = self.conversation_history[-10:]
                    
                    return response if response.strip() else "I'm not sure how to respond to that."
                    
            except Exception as e:
                error_msg = f"Error generating response: {str(e)}"
                logger.error(error_msg)
                return error_msg
    
    def cleanup(self):
        """Clean up GPU memory."""
        if not TORCH_AVAILABLE:
            return
        try:
            if self.device and self.device.type == 'cuda':
                torch.cuda.empty_cache()
            elif self.device and self.device.type == 'mps' and hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
            gc.collect()
            logger.info("Memory cleanup completed")
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")

# Global AI engine instance
ai_engine = ModernAIEngine()

# Error handler
@app.errorhandler(Exception)
def handle_error(error):
    """Global error handler."""
    logger.error(f"Unhandled error: {error}")
    return jsonify({
        'success': False,
        'message': f"Internal server error: {str(error)}"
    }), 500

# Flask routes with better error handling
@app.route('/api/model/load', methods=['POST'])
def load_model():
    """Load model endpoint."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'message': 'No JSON data provided'
            }), 400
        
        model_path = data.get('model_path', 'Model.pth')
        
        success, message = ai_engine.load_model(model_path)
        
        return jsonify({
            'success': success,
            'message': message,
            'model_info': ai_engine.model_info if success else None
        })
    except Exception as e:
        logger.error(f"Error in load_model endpoint: {e}")
        return jsonify({
            'success': False,
            'message': f"Error loading model: {str(e)}"
        }), 500

@app.route('/api/model/info')
def model_info():
    """Get model information."""
    try:
        if ai_engine.is_loaded:
            return jsonify({
                'success': True,
                'model_info': ai_engine.model_info
            })
        else:
            return jsonify({
                'success': False,
                'message': 'No model loaded'
            })
    except Exception as e:
        logger.error(f"Error in model_info endpoint: {e}")
        return jsonify({
            'success': False,
            'message': f"Error getting model info: {str(e)}"
        }), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """Chat endpoint."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'message': 'No JSON data provided'
            }), 400
        
        user_input = data.get('message', '')
        if not user_input.strip():
            return jsonify({
                'success': False,
                'message': 'Empty message provided'
            }), 400
        
        # Validate parameters
        try:
            temperature = float(data.get('temperature', 0.8))
            temperature = max(0.1, min(2.0, temperature))  # Clamp temperature
            
            top_k = int(data.get('top_k', 50))
            top_k = max(1, min(1000, top_k))  # Clamp top_k
            
            top_p = float(data.get('top_p', 0.9))
            top_p = max(0.1, min(1.0, top_p))  # Clamp top_p
            
            max_length = int(data.get('max_length', 150))
            max_length = max(10, min(500, max_length))  # Clamp max_length
            
        except (ValueError, TypeError) as e:
            return jsonify({
                'success': False,
                'message': f'Invalid parameter values: {e}'
            }), 400
        
        sampling_method = data.get('sampling_method', 'top_k')
        if sampling_method not in ['top_k', 'nucleus', 'top_p', 'greedy']:
            sampling_method = 'top_k'
        
        if not ai_engine.is_loaded:
            return jsonify({
                'success': False,
                'message': 'No model loaded'
            }), 400
        
        response = ai_engine.generate_response(
            user_input=user_input,
            temperature=temperature,
            sampling_method=sampling_method,
            top_k=top_k,
            top_p=top_p,
            max_length=max_length
        )
        
        return jsonify({
            'success': True,
            'response': response
        })
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return jsonify({
            'success': False,
            'message': f"Error generating response: {str(e)}"
        }), 500

@app.route('/api/chat/clear', methods=['POST'])
def clear_chat():
    """Clear chat history."""
    try:
        ai_engine.conversation_history = []
        return jsonify({'success': True, 'message': 'Chat history cleared'})
    except Exception as e:
        logger.error(f"Error clearing chat: {e}")
        return jsonify({
            'success': False,
            'message': f"Error clearing chat: {str(e)}"
        }), 500

@app.route('/api/system/status')
def system_status():
    """Get system status."""
    try:
        return jsonify({
            'pytorch_available': TORCH_AVAILABLE,
            'device': str(ai_engine.device) if ai_engine.device else 'None',
            'model_loaded': ai_engine.is_loaded,
            'torch_version': torch.__version__ if TORCH_AVAILABLE else 'Not installed',
            'conversation_length': len(ai_engine.conversation_history),
            'memory_info': {
                'cuda_available': torch.cuda.is_available() if TORCH_AVAILABLE else False,
                'mps_available': torch.backends.mps.is_available() if TORCH_AVAILABLE else False
            }
        })
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        return jsonify({
            'success': False,
            'message': f"Error getting system status: {str(e)}"
        }), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': time.time(),
        'version': '1.0.0'
    })

# Socket.IO events for real-time communication
@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    try:
        logger.info("Client connected")
        emit('status', {
            'connected': True,
            'pytorch_available': TORCH_AVAILABLE,
            'model_loaded': ai_engine.is_loaded,
            'device': str(ai_engine.device) if ai_engine.device else 'None'
        })
    except Exception as e:
        logger.error(f"Error handling connection: {e}")
        emit('error', {'message': 'Connection error'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    logger.info("Client disconnected")

@socketio.on('generate_message')
def handle_generate_message(data):
    """Handle real-time message generation."""
    try:
        if not ai_engine.is_loaded:
            emit('generation_error', {'message': 'No model loaded'})
            return
        
        if not data or not isinstance(data, dict):
            emit('generation_error', {'message': 'Invalid request data'})
            return
        
        user_input = data.get('message', '')
        if not user_input.strip():
            emit('generation_error', {'message': 'Empty message'})
            return
        
        settings = data.get('settings', {})
        
        # Validate settings
        try:
            temperature = float(settings.get('temperature', 0.8))
            temperature = max(0.1, min(2.0, temperature))
            
            top_k = int(settings.get('top_k', 50))
            top_k = max(1, min(1000, top_k))
            
            top_p = float(settings.get('top_p', 0.9))
            top_p = max(0.1, min(1.0, top_p))
            
            max_length = int(settings.get('max_length', 150))
            max_length = max(10, min(500, max_length))
            
            sampling_method = settings.get('sampling_method', 'top_k')
            if sampling_method not in ['top_k', 'nucleus', 'top_p', 'greedy']:
                sampling_method = 'top_k'
                
        except (ValueError, TypeError):
            emit('generation_error', {'message': 'Invalid settings'})
            return
        
        # Emit typing indicator
        emit('typing_start')
        
        try:
            # Generate response
            response = ai_engine.generate_response(
                user_input=user_input,
                temperature=temperature,
                sampling_method=sampling_method,
                top_k=top_k,
                top_p=top_p,
                max_length=max_length
            )
            
            # Emit response
            emit('typing_stop')
            emit('message_generated', {'response': response})
            
        except Exception as e:
            emit('typing_stop')
            emit('generation_error', {'message': f'Generation error: {str(e)}'})
        
    except Exception as e:
        logger.error(f"Error in generate_message handler: {e}")
        emit('typing_stop')
        emit('generation_error', {'message': 'Server error occurred'})

def run_electron_app():
    """Start the Electron desktop app."""
    try:
        # Check if we have electron installed
        electron_path = None
        
        # Try different electron paths
        possible_paths = [
            './node_modules/.bin/electron',
            'electron',
            'npx electron'
        ]
        
        for path in possible_paths:
            try:
                if path == 'npx electron':
                    subprocess.run(['npx', '--version'], check=True, capture_output=True, timeout=10)
                    electron_path = path
                    break
                else:
                    subprocess.run([path, '--version'], check=True, capture_output=True, timeout=10)
                    electron_path = path
                    break
            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                continue
        
        if electron_path:
            logger.info(f"Starting Electron app with: {electron_path}")
            if electron_path == 'npx electron':
                electron_process = subprocess.Popen(
                    ['npx', 'electron', '.'],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            else:
                electron_process = subprocess.Popen(
                    [electron_path, '.'],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            return electron_process
        else:
            logger.warning("Electron not found. Install with: npm install -g electron")
            logger.info("You can still access the web interface at http://localhost:5001")
            return None
            
    except Exception as e:
        logger.error(f"Error starting Electron app: {e}")
        return None

def create_electron_files():
    """Create Electron app files."""
    
    try:
        # Create package.json
        package_json = {
            "name": "lumina-ai-desktop",
            "version": "1.0.0",
            "description": "LuminaAI Neural Desktop Interface",
            "main": "main.js",
            "scripts": {
                "start": "electron .",
                "dev": "electron . --dev",
                "install-deps": "npm install electron"
            },
            "keywords": ["ai", "neural", "desktop", "electron"],
            "author": "Matias Nielsen",
            "license": "Custom",
            "devDependencies": {
                "electron": "^28.0.0"
            }
        }
        
        with open('package.json', 'w') as f:
            json.dump(package_json, f, indent=2)
        
        # Create main.js (Electron main process)
        main_js = '''const { app, BrowserWindow, Menu, dialog, shell, ipcMain } = require('electron');
const path = require('path');
const { spawn } = require('child_process');

let mainWindow;
let pythonProcess;

function createWindow() {
    // Create the browser window
    mainWindow = new BrowserWindow({
        width: 1400,
        height: 900,
        minWidth: 1200,
        minHeight: 800,
        webPreferences: {
            nodeIntegration: true,
            contextIsolation: false,
            enableRemoteModule: true,
            webSecurity: false
        },
        titleBarStyle: process.platform === 'darwin' ? 'hiddenInset' : 'default',
        vibrancy: process.platform === 'darwin' ? 'ultra-dark' : undefined,
        backgroundColor: '#0a0a0b',
        show: false,
        icon: path.join(__dirname, 'assets', 'icon.png')
    });

    // Load the app - check if renderer exists, otherwise load from localhost
    const rendererPath = path.join(__dirname, 'renderer', 'index.html');
    const fs = require('fs');
    
    if (fs.existsSync(rendererPath)) {
        mainWindow.loadFile(rendererPath);
    } else {
        // Load from backend server
        mainWindow.loadURL('http://localhost:5001');
    }

    // Show window when ready
    mainWindow.once('ready-to-show', () => {
        mainWindow.show();
        
        // Focus window
        if (process.platform === 'darwin') {
            app.focus();
        }
    });

    // Handle window closed
    mainWindow.on('closed', () => {
        mainWindow = null;
        if (pythonProcess) {
            pythonProcess.kill();
        }
    });

    // Handle navigation
    mainWindow.webContents.on('will-navigate', (event, navigationUrl) => {
        const parsedUrl = new URL(navigationUrl);
        
        // Allow localhost navigation
        if (parsedUrl.origin !== 'http://localhost:5001' && parsedUrl.protocol !== 'file:') {
            event.preventDefault();
            shell.openExternal(navigationUrl);
        }
    });

    // Create menu
    createMenu();

    // Development tools
    if (process.argv.includes('--dev')) {
        mainWindow.webContents.openDevTools();
    }
}

function createMenu() {
    const template = [
        {
            label: 'LuminaAI',
            submenu: [
                { 
                    label: 'About LuminaAI',
                    click: () => {
                        dialog.showMessageBox(mainWindow, {
                            type: 'info',
                            title: 'About LuminaAI',
                            message: 'LuminaAI Neural Desktop Interface',
                            detail: 'Advanced neural transformer interface for desktop\\nVersion 1.0.0\\n\\nCreated by Matias Nielsen'
                        });
                    }
                },
                { type: 'separator' },
                { role: 'cut' },
                { role: 'copy' },
                { role: 'paste' },
                { role: 'selectall' },
                { type: 'separator' },
                { 
                    label: 'Quit',
                    accelerator: process.platform === 'darwin' ? 'Cmd+Q' : 'Ctrl+Q',
                    click: () => {
                        app.quit();
                    }
                }
            ]
        },
        {
            label: 'View',
            submenu: [
                { role: 'reload' },
                { role: 'forceReload' },
                { role: 'toggleDevTools' },
                { type: 'separator' },
                { role: 'resetZoom' },
                { role: 'zoomIn' },
                { role: 'zoomOut' },
                { type: 'separator' },
                { role: 'togglefullscreen' }
            ]
        },
        {
            label: 'Window',
            submenu: [
                { role: 'minimize' },
                { role: 'zoom' }
            ]
        },
        {
            label: 'Help',
            submenu: [
                {
                    label: 'GitHub Repository',
                    click: () => {
                        shell.openExternal('https://github.com');
                    }
                },
                {
                    label: 'Report Issue',
                    click: () => {
                        shell.openExternal('https://github.com/issues');
                    }
                }
            ]
        }
    ];

    // macOS specific menu adjustments
    if (process.platform === 'darwin') {
        template[0].submenu.unshift({ role: 'about' });
        template[2].submenu.push(
            { type: 'separator' },
            { role: 'front' }
        );
    }

    const menu = Menu.buildFromTemplate(template);
    Menu.setApplicationMenu(menu);
}

// App event handlers
app.whenReady().then(() => {
    createWindow();

    app.on('activate', () => {
        if (BrowserWindow.getAllWindows().length === 0) {
            createWindow();
        }
    });
});

app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
        app.quit();
    }
});

app.on('before-quit', (event) => {
    // Clean shutdown
    if (pythonProcess) {
        pythonProcess.kill();
    }
});

// Security: Prevent new window creation
app.on('web-contents-created', (event, contents) => {
    contents.on('new-window', (event, navigationUrl) => {
        event.preventDefault();
        shell.openExternal(navigationUrl);
    });
});

// IPC handlers
ipcMain.handle('get-app-version', () => {
    return app.getVersion();
});

ipcMain.handle('show-message-box', async (event, options) => {
    const result = await dialog.showMessageBox(mainWindow, options);
    return result;
});
'''

        with open('main.js', 'w') as f:
            f.write(main_js)
        
        # Create assets directory
        assets_dir = Path('assets')
        assets_dir.mkdir(exist_ok=True)
        
        logger.info("Electron files created successfully")
        
    except Exception as e:
        logger.error(f"Error creating Electron files: {e}")

def auto_load_model():
    """Auto-load model if available."""
    try:
        model_path = Path("Model.pth")
        if model_path.exists():
            logger.info(f"Found model: {model_path}")
            success, message = ai_engine.load_model(str(model_path))
            if success:
                logger.info("Model auto-loaded successfully!")
            else:
                logger.warning(f"Failed to auto-load model: {message}")
        else:
            logger.info("No Model.pth found - load manually through interface")
    except Exception as e:
        logger.error(f"Error during auto-load: {e}")

def main():
    """Main application entry point."""
    print("🚀 " + "="*60)
    print("🚀 LUMINA AI DESKTOP APPLICATION")
    print("🚀 " + "="*60)
    print("✨ Initializing quantum neural pathways...")
    print("🧠 Loading consciousness matrix...")
    print("⚡ Calibrating synaptic resonance...")
    
    if TORCH_AVAILABLE:
        print("✅ PyTorch neural engine: ONLINE")
        device = setup_device()
        print(f"🔥 Compute device: {device}")
    else:
        print("⚠️  PyTorch neural engine: OFFLINE")
        print("📦 Install command: pip install torch numpy flask flask-socketio flask-cors")
    
    print("🖥️  Desktop interface: READY")
    print("💫 Consciousness level: TRANSCENDENT")
    print("🚀 Launch sequence: COMPLETE")
    print("🚀 " + "="*60)
    
    # Create Electron files
    try:
        create_electron_files()
    except Exception as e:
        logger.error(f"Error creating Electron files: {e}")
    
    # Auto-load model if available
    try:
        auto_load_model()
    except Exception as e:
        logger.error(f"Error during auto-load: {e}")
    
    print("\n🌟 Starting LuminaAI Backend Server...")
    print("🔗 Backend running on: http://localhost:5001")
    print("🖥️  Desktop app will launch automatically")
    print("💡 Press Ctrl+C to shutdown")
    print("🌐 Web interface: http://localhost:5001/api/system/status")
    print("-" * 60)
    
    # Start Electron app in a separate thread
    electron_process = None
    def start_electron():
        nonlocal electron_process
        import time
        time.sleep(3)  # Wait for Flask to start
        electron_process = run_electron_app()
    
    electron_thread = threading.Thread(target=start_electron, daemon=True)
    electron_thread.start()
    
    def cleanup():
        """Cleanup function."""
        print("\n🔌 Shutting down neural interface...")
        try:
            ai_engine.cleanup()
            if electron_process:
                electron_process.terminate()
                electron_process.wait(timeout=5)
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")
        print("👋 LuminaAI desktop interface offline. Goodbye!")
    
    # Register cleanup
    atexit.register(cleanup)
    
    def signal_handler(sig, frame):
        cleanup()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start the backend server
        socketio.run(
            app, 
            host='127.0.0.1', 
            port=5001, 
            debug=False, 
            allow_unsafe_werkzeug=True,
            log_output=False
        )
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Critical system error: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())