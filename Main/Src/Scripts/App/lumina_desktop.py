# Copyright (c) 2025 Matias Nielsen. All rights reserved.
# Licensed under the Custom License below.

#!/usr/bin/env python3
"""
Modern LuminaAI Web Interface
Ultra-modern dark theme with glassmorphism, animations, and word tokenization.
Web-based interface replacing Tkinter with Flask + modern frontend.
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
import threading
import queue
import sys
import os
from pathlib import Path
import json
import logging
import gc
from typing import Dict, Optional, List, Tuple
import time
import re
import math
import pickle
from collections import Counter
import unicodedata
import webbrowser

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

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Flask app initialization
app = Flask(__name__)
app.config['SECRET_KEY'] = 'lumina_ai_neural_interface_2025'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

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
        
        # Normalize unicode
        text = unicodedata.normalize('NFKC', text)
        
        # Basic cleaning
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words with proper handling."""
        text = self.preprocess_text(text)
        if not text:
            return []
        
        # Split on whitespace and punctuation
        tokens = re.findall(r'\w+|[^\w\s]', text.lower())
        return tokens
    
    def build_vocab(self, texts: List[str], min_freq: int = 2, max_vocab: int = 50000):
        """Build vocabulary from texts."""
        logger.info("Building vocabulary...")
        
        # Count word frequencies
        word_counts = Counter()
        for text in texts:
            tokens = self.tokenize(text)
            word_counts.update(tokens)
        
        # Filter by frequency and limit vocab size
        vocab_words = [word for word, count in word_counts.most_common(max_vocab - len(self.special_tokens)) 
                      if count >= min_freq]
        
        # Add to vocabulary
        current_id = len(self.special_tokens)
        for word in vocab_words:
            if word not in self.word_to_id:
                self.word_to_id[word] = current_id
                self.id_to_word[current_id] = word
                current_id += 1
        
        self.vocab_size_val = len(self.word_to_id)
        logger.info(f"Built vocabulary with {self.vocab_size_val} tokens")
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        tokens = self.tokenize(text)
        return [self.word_to_id.get(token, self.special_tokens['<UNK>']) for token in tokens]
    
    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text."""
        tokens = [self.id_to_word.get(id, '<UNK>') for id in ids if id in self.id_to_word]
        # Remove special tokens from output
        tokens = [token for token in tokens if not token.startswith('<') or not token.endswith('>')]
        return ' '.join(tokens)
    
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return self.vocab_size_val
    
    def save(self, path: Path):
        """Save tokenizer to disk."""
        data = {
            'word_to_id': self.word_to_id,
            'id_to_word': self.id_to_word,
            'vocab_size': self.vocab_size_val,
            'special_tokens': self.special_tokens
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, path: Path):
        """Load tokenizer from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.word_to_id = data['word_to_id']
        self.id_to_word = data['id_to_word']
        self.vocab_size_val = data['vocab_size']
        self.special_tokens = data.get('special_tokens', self.special_tokens)

class WordTransformer(nn.Module if TORCH_AVAILABLE else object):
    """Word-level transformer model based on ChatAI.py architecture."""
    
    def __init__(self, vocab_size: int, hidden_size: int, num_layers: int, 
                 num_heads: int, seq_length: int, dropout: float = 0.1):
        if not TORCH_AVAILABLE:
            return
        super().__init__()
        
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
        
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
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

# Sampling functions from ChatAI.py
def nucleus_sampling(probs: torch.Tensor, p: float = 0.9) -> int:
    """Nucleus (top-p) sampling for better text generation."""
    if not TORCH_AVAILABLE:
        return 0
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
    
    if top_p_probs.sum() > 0:
        top_p_probs = top_p_probs / top_p_probs.sum()
    else:
        top_p_probs = torch.ones_like(top_p_probs) / len(top_p_probs)
    
    try:
        chosen_idx = torch.multinomial(top_p_probs, 1).item()
        return top_p_indices[chosen_idx].item()
    except RuntimeError:
        return top_p_indices[0].item()

def top_k_sampling(probs: torch.Tensor, k: int = 50) -> int:
    """Top-k sampling for controlled text generation."""
    if not TORCH_AVAILABLE:
        return 0
    if k <= 0 or k >= len(probs):
        return torch.multinomial(probs, 1).item()
    
    top_k_probs, top_k_indices = torch.topk(probs, k)
    
    if top_k_probs.sum() > 0:
        top_k_probs = top_k_probs / top_k_probs.sum()
    else:
        top_k_probs = torch.ones_like(top_k_probs) / k
    
    try:
        chosen_idx = torch.multinomial(top_k_probs, 1).item()
        return top_k_indices[chosen_idx].item()
    except RuntimeError:
        return top_k_indices[0].item()

def clean_response(response: str) -> str:
    """Clean up the model's response."""
    if not response:
        return ""
    
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
    
    def load_model(self, model_path: str) -> Tuple[bool, str]:
        """Load model with word tokenization."""
        if not TORCH_AVAILABLE:
            return False, "PyTorch not available. Please install requirements: pip install torch numpy"
        
        try:
            model_path = Path(model_path)
            if not model_path.exists():
                return False, f"Model file not found: {model_path}"
            
            logger.info(f"Loading model from: {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            
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
                # Try to infer from model state
                return False, "Model configuration not found in checkpoint"
            
            # Initialize model
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
                'device': str(self.device)
            }
            
            self.is_loaded = True
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
        
        try:
            with torch.no_grad():
                # Build context from conversation history
                context_parts = self.conversation_history[-6:] if self.conversation_history else []
                context_parts.append(f"<USER> {user_input}")
                context_parts.append("<BOT>")
                context = " ".join(context_parts)
                
                # Encode input
                input_ids = torch.tensor(self.tokenizer.encode(context), dtype=torch.long).unsqueeze(0).to(self.device)
                
                if input_ids.size(1) == 0:
                    return "Unable to process input."
                
                generated = input_ids.clone()
                
                for step in range(max_length):
                    # Use sliding window for long sequences
                    max_seq_length = self.model.config.seq_length
                    input_seq = generated[:, -max_seq_length:] if generated.size(1) > max_seq_length else generated
                    
                    # Forward pass
                    logits = self.model(input_seq)
                    next_token_logits = logits[0, -1, :] / max(temperature, 0.1)
                    
                    # Apply softmax
                    probs = F.softmax(next_token_logits, dim=0)
                    
                    # Handle NaN/Inf
                    if torch.isnan(probs).any() or torch.isinf(probs).any():
                        probs = torch.ones_like(probs) / len(probs)
                    
                    # Sample next token
                    if sampling_method == "nucleus" or sampling_method == "top_p":
                        next_token_id = nucleus_sampling(probs, p=top_p)
                    elif sampling_method == "top_k":
                        next_token_id = top_k_sampling(probs, k=top_k)
                    elif sampling_method == "greedy":
                        next_token_id = torch.argmax(probs).item()
                    else:
                        next_token_id = top_k_sampling(probs, k=top_k)
                    
                    # Validate token ID
                    if next_token_id < 0 or next_token_id >= self.tokenizer.vocab_size():
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
                    
                    # Trim history
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
        except Exception:
            pass

# Global AI engine instance
ai_engine = ModernAIEngine()

# Flask routes
@app.route('/')
def index():
    """Main page route."""
    return render_template('index.html')

@app.route('/api/model/load', methods=['POST'])
def load_model():
    """Load model endpoint."""
    try:
        data = request.get_json()
        model_path = data.get('model_path', 'Model.pth')
        
        success, message = ai_engine.load_model(model_path)
        
        return jsonify({
            'success': success,
            'message': message,
            'model_info': ai_engine.model_info if success else None
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f"Error loading model: {str(e)}"
        })

@app.route('/api/model/info')
def model_info():
    """Get model information."""
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

@app.route('/api/chat', methods=['POST'])
def chat():
    """Chat endpoint."""
    try:
        data = request.get_json()
        user_input = data.get('message', '')
        temperature = float(data.get('temperature', 0.8))
        sampling_method = data.get('sampling_method', 'top_k')
        top_k = int(data.get('top_k', 50))
        top_p = float(data.get('top_p', 0.9))
        max_length = int(data.get('max_length', 150))
        
        if not ai_engine.is_loaded:
            return jsonify({
                'success': False,
                'message': 'No model loaded'
            })
        
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
        return jsonify({
            'success': False,
            'message': f"Error generating response: {str(e)}"
        })

@app.route('/api/chat/clear', methods=['POST'])
def clear_chat():
    """Clear chat history."""
    ai_engine.conversation_history = []
    return jsonify({'success': True, 'message': 'Chat history cleared'})

@app.route('/api/system/status')
def system_status():
    """Get system status."""
    return jsonify({
        'pytorch_available': TORCH_AVAILABLE,
        'device': str(ai_engine.device) if ai_engine.device else 'None',
        'model_loaded': ai_engine.is_loaded,
        'torch_version': torch.__version__ if TORCH_AVAILABLE else 'Not installed'
    })

# Socket.IO events for real-time communication
@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    emit('status', {
        'connected': True,
        'pytorch_available': TORCH_AVAILABLE,
        'model_loaded': ai_engine.is_loaded
    })

@socketio.on('generate_message')
def handle_generate_message(data):
    """Handle real-time message generation."""
    try:
        if not ai_engine.is_loaded:
            emit('generation_error', {'message': 'No model loaded'})
            return
        
        user_input = data.get('message', '')
        settings = data.get('settings', {})
        
        # Emit typing indicator
        emit('typing_start')
        
        # Generate response
        response = ai_engine.generate_response(
            user_input=user_input,
            temperature=settings.get('temperature', 0.8),
            sampling_method=settings.get('sampling_method', 'top_k'),
            top_k=settings.get('top_k', 50),
            top_p=settings.get('top_p', 0.9),
            max_length=settings.get('max_length', 150)
        )
        
        # Emit response
        emit('typing_stop')
        emit('message_generated', {'response': response})
        
    except Exception as e:
        emit('typing_stop')
        emit('generation_error', {'message': str(e)})

def create_html_template():
    """Create modern HTML template."""
    template_dir = Path("templates")
    template_dir.mkdir(exist_ok=True)
    
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>✨ LuminaAI - Neural Transformer Interface</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.2/socket.io.js"></script>
    <style>
        :root {
            --bg-primary: #0a0a0b;
            --bg-secondary: #1a1a1d;
            --bg-tertiary: #2d2d30;
            --bg-glass: rgba(26, 26, 29, 0.8);
            --accent-primary: #00d4ff;
            --accent-secondary: #7c3aed;
            --text-primary: #ffffff;
            --text-secondary: #b4b4b8;
            --text-success: #10b981;
            --text-warning: #f59e0b;
            --text-error: #ef4444;
            --border-color: #404040;
            --shadow-color: rgba(0, 0, 0, 0.3);
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            overflow-x: hidden;
        }

        .container {
            display: flex;
            height: 100vh;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            gap: 20px;
        }

        .sidebar {
            width: 320px;
            background: var(--bg-glass);
            backdrop-filter: blur(20px);
            border: 1px solid var(--border-color);
            border-radius: 20px;
            padding: 25px;
            box-shadow: 0 20px 40px var(--shadow-color);
        }

        .chat-area {
            flex: 1;
            display: flex;
            flex-direction: column;
            background: var(--bg-glass);
            backdrop-filter: blur(20px);
            border: 1px solid var(--border-color);
            border-radius: 20px;
            overflow: hidden;
            box-shadow: 0 20px 40px var(--shadow-color);
        }

        .header {
            background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
            padding: 20px 30px;
            text-align: center;
            position: relative;
        }

        .header h1 {
            font-size: 24px;
            font-weight: 700;
            margin: 0;
        }

        .header .subtitle {
            font-size: 14px;
            opacity: 0.9;
            margin-top: 5px;
        }

        .status-indicator {
            position: absolute;
            right: 20px;
            top: 50%;
            transform: translateY(-50%);
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: var(--text-warning);
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 0.5; }
            50% { opacity: 1; }
        }

        .section-title {
            font-size: 16px;
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .btn {
            background: var(--accent-primary);
            color: var(--text-primary);
            border: none;
            padding: 12px 20px;
            border-radius: 12px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            font-size: 14px;
            width: 100%;
            margin-bottom: 10px;
        }

        .btn:hover {
            background: #0099cc;
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(0, 212, 255, 0.3);
        }

        .btn:disabled {
            background: #333;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }

        .slider-container {
            margin-bottom: 20px;
        }

        .slider-label {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
            font-size: 14px;
            color: var(--text-secondary);
        }

        .slider-value {
            color: var(--accent-primary);
            font-weight: 600;
        }

        .slider {
            width: 100%;
            height: 6px;
            border-radius: 3px;
            background: var(--bg-tertiary);
            outline: none;
            -webkit-appearance: none;
            margin-bottom: 15px;
        }

        .slider::-webkit-slider-thumb {
            -webkit-appearance: none;
            appearance: none;
            width: 18px;
            height: 18px;
            border-radius: 50%;
            background: var(--accent-primary);
            cursor: pointer;
            box-shadow: 0 0 10px rgba(0, 212, 255, 0.5);
        }

        .slider::-moz-range-thumb {
            width: 18px;
            height: 18px;
            border-radius: 50%;
            background: var(--accent-primary);
            cursor: pointer;
            border: none;
            box-shadow: 0 0 10px rgba(0, 212, 255, 0.5);
        }

        .sampling-buttons {
            display: flex;
            gap: 8px;
            margin-bottom: 15px;
        }

        .sampling-btn {
            flex: 1;
            padding: 8px 12px;
            border: 1px solid var(--border-color);
            background: var(--bg-tertiary);
            color: var(--text-secondary);
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s ease;
            font-size: 12px;
        }

        .sampling-btn.active {
            background: var(--accent-secondary);
            color: var(--text-primary);
            border-color: var(--accent-secondary);
        }

        .sampling-btn:hover {
            border-color: var(--accent-primary);
            color: var(--text-primary);
        }

        .input-group {
            margin-bottom: 15px;
        }

        .input-label {
            display: block;
            margin-bottom: 6px;
            font-size: 14px;
            color: var(--text-secondary);
        }

        .input-field {
            width: 100%;
            padding: 10px;
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            color: var(--text-primary);
            font-size: 14px;
            transition: border-color 0.3s ease;
        }

        .input-field:focus {
            outline: none;
            border-color: var(--accent-primary);
            box-shadow: 0 0 10px rgba(0, 212, 255, 0.2);
        }

        .status-panel {
            background: var(--bg-tertiary);
            border-radius: 12px;
            padding: 15px;
            margin-top: 20px;
            border: 1px solid var(--border-color);
        }

        .status-item {
            display: flex;
            justify-content: space-between;
            margin-bottom: 8px;
            font-size: 13px;
        }

        .status-item:last-child {
            margin-bottom: 0;
        }

        .status-key {
            color: var(--text-secondary);
        }

        .status-value {
            color: var(--text-primary);
            font-weight: 500;
        }

        .chat-messages {
            flex: 1;
            padding: 20px;
            overflow-y: auto;
            background: var(--bg-secondary);
        }

        .message {
            margin-bottom: 20px;
            animation: slideIn 0.3s ease;
        }

        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .message-header {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 8px;
        }

        .message-avatar {
            width: 32px;
            height: 32px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 16px;
            color: var(--text-primary);
        }

        .user-avatar {
            background: var(--accent-primary);
        }

        .ai-avatar {
            background: var(--text-success);
        }

        .system-avatar {
            background: var(--text-warning);
        }

        .message-name {
            font-weight: 600;
            font-size: 14px;
        }

        .message-time {
            font-size: 12px;
            color: var(--text-secondary);
            margin-left: auto;
        }

        .message-content {
            background: var(--bg-tertiary);
            padding: 15px;
            border-radius: 12px;
            margin-left: 42px;
            border: 1px solid var(--border-color);
            line-height: 1.5;
        }

        .user-message .message-content {
            background: rgba(0, 212, 255, 0.1);
            border-color: var(--accent-primary);
        }

        .ai-message .message-content {
            background: rgba(16, 185, 129, 0.1);
            border-color: var(--text-success);
        }

        .system-message .message-content {
            background: rgba(245, 158, 11, 0.1);
            border-color: var(--text-warning);
            font-style: italic;
        }

        .typing-indicator {
            display: none;
            margin-left: 42px;
            padding: 15px;
            background: var(--bg-tertiary);
            border-radius: 12px;
            border: 1px solid var(--border-color);
        }

        .typing-dots {
            display: flex;
            gap: 4px;
        }

        .typing-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--text-secondary);
            animation: typingPulse 1.5s infinite;
        }

        .typing-dot:nth-child(2) {
            animation-delay: 0.2s;
        }

        .typing-dot:nth-child(3) {
            animation-delay: 0.4s;
        }

        @keyframes typingPulse {
            0%, 60%, 100% {
                opacity: 0.3;
                transform: scale(1);
            }
            30% {
                opacity: 1;
                transform: scale(1.2);
            }
        }

        .chat-input-area {
            padding: 20px;
            background: var(--bg-secondary);
            border-top: 1px solid var(--border-color);
        }

        .input-container {
            display: flex;
            gap: 15px;
            align-items: flex-end;
        }

        .message-input {
            flex: 1;
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 15px;
            color: var(--text-primary);
            font-size: 14px;
            resize: none;
            min-height: 60px;
            max-height: 120px;
            transition: border-color 0.3s ease;
            font-family: inherit;
        }

        .message-input:focus {
            outline: none;
            border-color: var(--accent-primary);
            box-shadow: 0 0 20px rgba(0, 212, 255, 0.2);
        }

        .message-input::placeholder {
            color: var(--text-secondary);
        }

        .send-btn {
            background: var(--accent-primary);
            border: none;
            border-radius: 12px;
            padding: 15px 20px;
            color: var(--text-primary);
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 14px;
        }

        .send-btn:hover:not(:disabled) {
            background: #0099cc;
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(0, 212, 255, 0.3);
        }

        .send-btn:disabled {
            background: #333;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }

        .model-info-modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.8);
            backdrop-filter: blur(10px);
            z-index: 1000;
            align-items: center;
            justify-content: center;
        }

        .modal-content {
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 20px;
            padding: 30px;
            max-width: 600px;
            width: 90%;
            max-height: 80vh;
            overflow-y: auto;
            box-shadow: 0 30px 60px var(--shadow-color);
        }

        .modal-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 1px solid var(--border-color);
        }

        .modal-title {
            font-size: 20px;
            font-weight: 700;
            color: var(--text-primary);
        }

        .close-btn {
            background: none;
            border: none;
            font-size: 24px;
            color: var(--text-secondary);
            cursor: pointer;
            padding: 5px;
            transition: color 0.3s ease;
        }

        .close-btn:hover {
            color: var(--text-primary);
        }

        .info-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }

        .info-card {
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 15px;
        }

        .info-card-title {
            font-size: 14px;
            color: var(--text-secondary);
            margin-bottom: 8px;
        }

        .info-card-value {
            font-size: 18px;
            font-weight: 600;
            color: var(--text-primary);
        }

        .toast {
            position: fixed;
            top: 20px;
            right: 20px;
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 15px 20px;
            box-shadow: 0 10px 30px var(--shadow-color);
            z-index: 1001;
            transform: translateX(400px);
            transition: transform 0.3s ease;
        }

        .toast.show {
            transform: translateX(0);
        }

        .toast.success {
            border-left: 4px solid var(--text-success);
        }

        .toast.error {
            border-left: 4px solid var(--text-error);
        }

        .toast.warning {
            border-left: 4px solid var(--text-warning);
        }

        /* Responsive design */
        @media (max-width: 768px) {
            .container {
                flex-direction: column;
                padding: 10px;
                gap: 10px;
            }

            .sidebar {
                width: 100%;
                order: 2;
            }

            .chat-area {
                order: 1;
                min-height: 60vh;
            }

            .input-container {
                flex-direction: column;
                gap: 10px;
            }

            .send-btn {
                width: 100%;
                justify-content: center;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Sidebar -->
        <div class="sidebar">
            <!-- Model Section -->
            <div class="section-title">🧠 Model Configuration</div>
            <button class="btn" id="loadModelBtn">📁 Load Model</button>
            <button class="btn" id="modelInfoBtn" disabled>ℹ️ Model Info</button>
            
            <!-- Generation Settings -->
            <div class="section-title" style="margin-top: 30px;">⚙️ Generation Settings</div>
            
            <div class="slider-container">
                <div class="slider-label">
                    <span>🌡️ Temperature</span>
                    <span class="slider-value" id="tempValue">0.8</span>
                </div>
                <input type="range" class="slider" id="temperatureSlider" min="0.1" max="2.0" step="0.1" value="0.8">
            </div>
            
            <div class="slider-container">
                <div class="slider-label">
                    <span>🎯 Sampling Method</span>
                </div>
                <div class="sampling-buttons">
                    <button class="sampling-btn active" data-method="top_k">Top-K</button>
                    <button class="sampling-btn" data-method="nucleus">Nucleus</button>
                    <button class="sampling-btn" data-method="greedy">Greedy</button>
                </div>
            </div>
            
            <div class="input-group">
                <label class="input-label">🔢 Top-K Value</label>
                <input type="number" class="input-field" id="topKInput" min="1" max="100" value="50">
            </div>
            
            <div class="slider-container">
                <div class="slider-label">
                    <span>📊 Top-P (Nucleus)</span>
                    <span class="slider-value" id="topPValue">0.9</span>
                </div>
                <input type="range" class="slider" id="topPSlider" min="0.1" max="1.0" step="0.1" value="0.9">
            </div>
            
            <div class="input-group">
                <label class="input-label">📏 Max Length</label>
                <input type="number" class="input-field" id="maxLengthInput" min="50" max="500" value="150">
            </div>
            
            <!-- Controls -->
            <div class="section-title" style="margin-top: 30px;">🎮 Controls</div>
            <button class="btn" id="clearChatBtn" style="background: var(--text-warning);">🗑️ Clear Chat</button>
            
            <!-- Status Panel -->
            <div class="status-panel">
                <div class="status-item">
                    <span class="status-key">PyTorch</span>
                    <span class="status-value" id="pytorchStatus">Checking...</span>
                </div>
                <div class="status-item">
                    <span class="status-key">Device</span>
                    <span class="status-value" id="deviceStatus">Unknown</span>
                </div>
                <div class="status-item">
                    <span class="status-key">Model</span>
                    <span class="status-value" id="modelStatus">Not loaded</span>
                </div>
            </div>
        </div>

        <!-- Chat Area -->
        <div class="chat-area">
            <div class="header">
                <h1>✨ LuminaAI</h1>
                <div class="subtitle">Neural Transformer Interface</div>
                <div class="status-indicator" id="connectionIndicator"></div>
            </div>
            
            <div class="chat-messages" id="chatMessages">
                <!-- Messages will be added here -->
                <div class="message system-message">
                    <div class="message-header">
                        <div class="message-avatar system-avatar">⚡</div>
                        <div class="message-name">System</div>
                        <div class="message-time" id="currentTime"></div>
                    </div>
                    <div class="message-content">
                        ✨ Welcome to LuminaAI Neural Interface! Load a model to begin chatting with your AI.
                    </div>
                </div>
            </div>
            
            <div class="typing-indicator" id="typingIndicator">
                <div class="typing-dots">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
            </div>
            
            <div class="chat-input-area">
                <div class="input-container">
                    <textarea class="message-input" id="messageInput" placeholder="Type your message... (Ctrl+Enter to send)" rows="3"></textarea>
                    <button class="send-btn" id="sendBtn" disabled>
                        🚀 Send
                    </button>
                </div>
            </div>
        </div>
    </div>

    <!-- Model Info Modal -->
    <div class="model-info-modal" id="modelInfoModal">
        <div class="modal-content">
            <div class="modal-header">
                <h2 class="modal-title">🧠 Neural Model Architecture</h2>
                <button class="close-btn" id="closeModalBtn">&times;</button>
            </div>
            <div id="modelInfoContent">
                <!-- Model info will be populated here -->
            </div>
        </div>
    </div>

    <script>
        // Initialize Socket.IO
        const socket = io();
        
        // Global state
        let isModelLoaded = false;
        let isGenerating = false;
        let currentSamplingMethod = 'top_k';
        
        // DOM elements
        const elements = {
            loadModelBtn: document.getElementById('loadModelBtn'),
            modelInfoBtn: document.getElementById('modelInfoBtn'),
            temperatureSlider: document.getElementById('temperatureSlider'),
            tempValue: document.getElementById('tempValue'),
            topPSlider: document.getElementById('topPSlider'),
            topPValue: document.getElementById('topPValue'),
            topKInput: document.getElementById('topKInput'),
            maxLengthInput: document.getElementById('maxLengthInput'),
            clearChatBtn: document.getElementById('clearChatBtn'),
            chatMessages: document.getElementById('chatMessages'),
            messageInput: document.getElementById('messageInput'),
            sendBtn: document.getElementById('sendBtn'),
            typingIndicator: document.getElementById('typingIndicator'),
            connectionIndicator: document.getElementById('connectionIndicator'),
            modelInfoModal: document.getElementById('modelInfoModal'),
            closeModalBtn: document.getElementById('closeModalBtn'),
            modelInfoContent: document.getElementById('modelInfoContent'),
            pytorchStatus: document.getElementById('pytorchStatus'),
            deviceStatus: document.getElementById('deviceStatus'),
            modelStatus: document.getElementById('modelStatus')
        };
        
        // Initialize app
        document.addEventListener('DOMContentLoaded', function() {
            updateCurrentTime();
            setupEventListeners();
            checkSystemStatus();
            
            // Try to auto-load Model.pth if it exists
            setTimeout(() => {
                loadModel('Model.pth');
            }, 1000);
        });
        
        function updateCurrentTime() {
            const now = new Date();
            const timeStr = now.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
            const timeElement = document.getElementById('currentTime');
            if (timeElement) {
                timeElement.textContent = timeStr;
            }
        }
        
        function setupEventListeners() {
            // Temperature slider
            elements.temperatureSlider.addEventListener('input', function() {
                elements.tempValue.textContent = this.value;
            });
            
            // Top-P slider
            elements.topPSlider.addEventListener('input', function() {
                elements.topPValue.textContent = this.value;
            });
            
            // Sampling method buttons
            document.querySelectorAll('.sampling-btn').forEach(btn => {
                btn.addEventListener('click', function() {
                    document.querySelectorAll('.sampling-btn').forEach(b => b.classList.remove('active'));
                    this.classList.add('active');
                    currentSamplingMethod = this.dataset.method;
                });
            });
            
            // Load model button
            elements.loadModelBtn.addEventListener('click', function() {
                const modelPath = prompt('Enter model path:', 'Model.pth');
                if (modelPath) {
                    loadModel(modelPath);
                }
            });
            
            // Model info button
            elements.modelInfoBtn.addEventListener('click', showModelInfo);
            
            // Clear chat button
            elements.clearChatBtn.addEventListener('click', clearChat);
            
            // Send button
            elements.sendBtn.addEventListener('click', sendMessage);
            
            // Message input
            elements.messageInput.addEventListener('keydown', function(e) {
                if (e.ctrlKey && e.key === 'Enter') {
                    e.preventDefault();
                    sendMessage();
                } else if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    sendMessage();
                }
            });
            
            // Modal close
            elements.closeModalBtn.addEventListener('click', function() {
                elements.modelInfoModal.style.display = 'none';
            });
            
            // Close modal on backdrop click
            elements.modelInfoModal.addEventListener('click', function(e) {
                if (e.target === this) {
                    this.style.display = 'none';
                }
            });
        }
        
        function checkSystemStatus() {
            fetch('/api/system/status')
                .then(response => response.json())
                .then(data => {
                    elements.pytorchStatus.textContent = data.pytorch_available ? '✅ Available' : '❌ Missing';
                    elements.deviceStatus.textContent = data.device || 'Unknown';
                    elements.modelStatus.textContent = data.model_loaded ? '✅ Loaded' : '❌ Not loaded';
                    
                    if (data.pytorch_available) {
                        elements.pytorchStatus.style.color = 'var(--text-success)';
                    } else {
                        elements.pytorchStatus.style.color = 'var(--text-error)';
                        showToast('PyTorch not available. Install with: pip install torch numpy', 'error');
                    }
                })
                .catch(error => {
                    console.error('Error checking system status:', error);
                });
        }
        
        function loadModel(modelPath) {
            elements.loadModelBtn.disabled = true;
            elements.loadModelBtn.textContent = '🔄 Loading...';
            
            addSystemMessage('🧠 Loading neural model: ' + modelPath);
            
            fetch('/api/model/load', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    model_path: modelPath
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    isModelLoaded = true;
                    elements.sendBtn.disabled = false;
                    elements.modelInfoBtn.disabled = false;
                    elements.modelStatus.textContent = '✅ Loaded';
                    elements.modelStatus.style.color = 'var(--text-success)';
                    elements.connectionIndicator.style.background = 'var(--text-success)';
                    
                    addSystemMessage('✅ Neural model loaded successfully!');
                    addSystemMessage('🚀 Ready for neural inference • All systems online');
                    showToast('Model loaded successfully!', 'success');
                } else {
                    addSystemMessage('❌ Failed to load model: ' + data.message, 'error');
                    showToast('Failed to load model: ' + data.message, 'error');
                }
            })
            .catch(error => {
                console.error('Error loading model:', error);
                addSystemMessage('❌ Error loading model: ' + error.message, 'error');
                showToast('Error loading model', 'error');
            })
            .finally(() => {
                elements.loadModelBtn.disabled = false;
                elements.loadModelBtn.textContent = '📁 Load Model';
            });
        }
        
        function showModelInfo() {
            if (!isModelLoaded) {
                showToast('No model loaded', 'warning');
                return;
            }
            
            fetch('/api/model/info')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        const info = data.model_info;
                        const content = `
                            <div class="info-grid">
                                <div class="info-card">
                                    <div class="info-card-title">Vocabulary Size</div>
                                    <div class="info-card-value">${info.vocab_size?.toLocaleString() || 'Unknown'}</div>
                                </div>
                                <div class="info-card">
                                    <div class="info-card-title">Hidden Size</div>
                                    <div class="info-card-value">${info.hidden_size || 'Unknown'}</div>
                                </div>
                                <div class="info-card">
                                    <div class="info-card-title">Layers</div>
                                    <div class="info-card-value">${info.num_layers || 'Unknown'}</div>
                                </div>
                                <div class="info-card">
                                    <div class="info-card-title">Attention Heads</div>
                                    <div class="info-card-value">${info.num_heads || 'Unknown'}</div>
                                </div>
                                <div class="info-card">
                                    <div class="info-card-title">Sequence Length</div>
                                    <div class="info-card-value">${info.seq_length || 'Unknown'}</div>
                                </div>
                                <div class="info-card">
                                    <div class="info-card-title">Device</div>
                                    <div class="info-card-value">${info.device || 'Unknown'}</div>
                                </div>
                                <div class="info-card">
                                    <div class="info-card-title">Training Epoch</div>
                                    <div class="info-card-value">${info.epoch || 'Unknown'}</div>
                                </div>
                                <div class="info-card">
                                    <div class="info-card-title">Loss</div>
                                    <div class="info-card-value">${typeof info.loss === 'number' ? info.loss.toFixed(4) : 'Unknown'}</div>
                                </div>
                            </div>
                            <div style="margin-top: 20px; padding: 15px; background: var(--bg-tertiary); border-radius: 12px; border: 1px solid var(--border-color);">
                                <h3 style="margin-bottom: 10px; color: var(--accent-primary);">🏗️ Architecture Details</h3>
                                <p style="color: var(--text-secondary); line-height: 1.5;">
                                    This is a word-level transformer model with ${info.num_layers || 'unknown'} layers, 
                                    ${info.num_heads || 'unknown'} attention heads, and a vocabulary of ${info.vocab_size?.toLocaleString() || 'unknown'} tokens. 
                                    The model uses GELU activation and layer normalization for optimal performance.
                                </p>
                            </div>
                        `;
                        elements.modelInfoContent.innerHTML = content;
                        elements.modelInfoModal.style.display = 'flex';
                    }
                })
                .catch(error => {
                    console.error('Error fetching model info:', error);
                    showToast('Error fetching model info', 'error');
                });
        }
        
        function sendMessage() {
            if (isGenerating || !isModelLoaded) return;
            
            const message = elements.messageInput.value.trim();
            if (!message) return;
            
            // Add user message
            addUserMessage(message);
            elements.messageInput.value = '';
            
            // Show typing indicator
            showTypingIndicator();
            isGenerating = true;
            elements.sendBtn.disabled = true;
            elements.sendBtn.textContent = '⏳ Generating...';
            
            // Prepare settings
            const settings = {
                temperature: parseFloat(elements.temperatureSlider.value),
                sampling_method: currentSamplingMethod,
                top_k: parseInt(elements.topKInput.value),
                top_p: parseFloat(elements.topPSlider.value),
                max_length: parseInt(elements.maxLengthInput.value)
            };
            
            // Send to server
            fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    message: message,
                    ...settings
                })
            })
            .then(response => response.json())
            .then(data => {
                hideTypingIndicator();
                
                if (data.success) {
                    addAIMessage(data.response);
                } else {
                    addSystemMessage('❌ Error: ' + data.message, 'error');
                }
            })
            .catch(error => {
                hideTypingIndicator();
                console.error('Error sending message:', error);
                addSystemMessage('❌ Error sending message: ' + error.message, 'error');
            })
            .finally(() => {
                isGenerating = false;
                elements.sendBtn.disabled = !isModelLoaded;
                elements.sendBtn.textContent = '🚀 Send';
            });
        }
        
        function addUserMessage(message) {
            const messageElement = createMessageElement('user', '👤', 'You', message);
            elements.chatMessages.appendChild(messageElement);
            scrollToBottom();
        }
        
        function addAIMessage(message) {
            const messageElement = createMessageElement('ai', '🤖', 'LuminaAI', message);
            elements.chatMessages.appendChild(messageElement);
            scrollToBottom();
        }
        
        function addSystemMessage(message, type = 'system') {
            const icon = type === 'error' ? '❌' : type === 'warning' ? '⚠️' : '⚡';
            const name = type === 'error' ? 'Error' : type === 'warning' ? 'Warning' : 'System';
            const messageElement = createMessageElement(type, icon, name, message);
            elements.chatMessages.appendChild(messageElement);
            scrollToBottom();
        }
        
        function createMessageElement(type, icon, name, content) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${type}-message`;
            
            const currentTime = new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
            
            messageDiv.innerHTML = `
                <div class="message-header">
                    <div class="message-avatar ${type}-avatar">${icon}</div>
                    <div class="message-name">${name}</div>
                    <div class="message-time">${currentTime}</div>
                </div>
                <div class="message-content">${content}</div>
            `;
            
            return messageDiv;
        }
        
        function showTypingIndicator() {
            elements.typingIndicator.style.display = 'block';
            scrollToBottom();
        }
        
        function hideTypingIndicator() {
            elements.typingIndicator.style.display = 'none';
        }
        
        function scrollToBottom() {
            setTimeout(() => {
                elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
            }, 100);
        }
        
        function clearChat() {
            if (confirm('Clear all conversation history? This will reset the neural context.')) {
                // Clear UI
                const systemMessages = elements.chatMessages.querySelectorAll('.system-message');
                elements.chatMessages.innerHTML = '';
                
                // Keep the welcome message
                if (systemMessages.length > 0) {
                    elements.chatMessages.appendChild(systemMessages[0]);
                }
                
                // Clear server-side history
                fetch('/api/chat/clear', {
                    method: 'POST'
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        addSystemMessage('🧹 Neural memory cleared • Context reset');
                        showToast('Chat history cleared', 'success');
                    }
                })
                .catch(error => {
                    console.error('Error clearing chat:', error);
                });
            }
        }
        
        function showToast(message, type = 'success') {
            // Remove existing toasts
            const existingToasts = document.querySelectorAll('.toast');
            existingToasts.forEach(toast => toast.remove());
            
            const toast = document.createElement('div');
            toast.className = `toast ${type}`;
            toast.textContent = message;
            
            document.body.appendChild(toast);
            
            // Show toast
            setTimeout(() => {
                toast.classList.add('show');
            }, 100);
            
            // Hide toast after 3 seconds
            setTimeout(() => {
                toast.classList.remove('show');
                setTimeout(() => toast.remove(), 300);
            }, 3000);
        }
        
        // Socket.IO event handlers
        socket.on('connect', function() {
            elements.connectionIndicator.style.background = 'var(--text-success)';
            console.log('Connected to server');
        });
        
        socket.on('disconnect', function() {
            elements.connectionIndicator.style.background = 'var(--text-error)';
            console.log('Disconnected from server');
        });
        
        socket.on('status', function(data) {
            console.log('Server status:', data);
            if (data.model_loaded) {
                isModelLoaded = true;
                elements.sendBtn.disabled = false;
                elements.modelInfoBtn.disabled = false;
            }
        });
        
        socket.on('typing_start', function() {
            showTypingIndicator();
        });
        
        socket.on('typing_stop', function() {
            hideTypingIndicator();
        });
        
        socket.on('message_generated', function(data) {
            addAIMessage(data.response);
            isGenerating = false;
            elements.sendBtn.disabled = !isModelLoaded;
            elements.sendBtn.textContent = '🚀 Send';
        });
        
        socket.on('generation_error', function(data) {
            addSystemMessage('❌ Generation error: ' + data.message, 'error');
            isGenerating = false;
            elements.sendBtn.disabled = !isModelLoaded;
            elements.sendBtn.textContent = '🚀 Send';
        });
        
        // Auto-resize textarea
        elements.messageInput.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 120) + 'px';
        });
        
        // Focus message input on page load
        elements.messageInput.focus();
    </script>
</body>
</html>"""
    
    with open(template_dir / "index.html", "w", encoding="utf-8") as f:
        f.write(html_content)

def main():
    """Main application entry point."""
    print("🚀 " + "="*60)
    print("🚀 LUMINA AI MODERN WEB INTERFACE")
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
        print("📦 Install command: pip install torch numpy flask flask-socketio")
    
    print("🌐 Neural web interface: READY")
    print("💫 Consciousness level: TRANSCENDENT")
    print("🚀 Launch sequence: COMPLETE")
    print("🚀 " + "="*60)
    
    # Create HTML template
    create_html_template()
    
    # Auto-load model if available
    model_path = Path("Model.pth")
    if model_path.exists():
        print(f"🔍 Found model: {model_path}")
        print("🤖 Model will auto-load when interface starts")
    else:
        print("⚠️  No Model.pth found - load manually through interface")
    
    print("\n🌟 Starting LuminaAI Web Interface...")
    print("🔗 Open your browser to: http://localhost:5000")
    print("💡 Press Ctrl+C to shutdown")
    print("-" * 60)
    
    try:
        # Try to open browser automatically
        import threading
        def open_browser():
            import time
            time.sleep(1.5)
            try:
                webbrowser.open('http://localhost:5000')
            except:
                pass
        
        threading.Thread(target=open_browser, daemon=True).start()
        
        # Start the web server
        socketio.run(app, host='0.0.0.0', port=5000, debug=False, 
                    allow_unsafe_werkzeug=True)
    
    except KeyboardInterrupt:
        print("\n🔌 Shutting down neural interface...")
        ai_engine.cleanup()
        print("👋 LuminaAI neural interface offline. Goodbye!")
    except Exception as e:
        print(f"\n❌ Critical system error: {e}")
        ai_engine.cleanup()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())