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
from typing import Dict, Optional, List, Tuple
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

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Flask app initialization
app = Flask(__name__)
app.config['SECRET_KEY'] = 'lumina_ai_neural_interface_2025'
CORS(app, origins=["http://localhost:3000", "app://."])  # Allow Electron
socketio = SocketIO(app, cors_allowed_origins=["http://localhost:3000", "app://.", "*"], async_mode='threading')

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

# Sampling functions
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
                    subprocess.run(['npx', '--version'], check=True, capture_output=True)
                    electron_path = path
                    break
                else:
                    subprocess.run([path, '--version'], check=True, capture_output=True)
                    electron_path = path
                    break
            except (subprocess.CalledProcessError, FileNotFoundError):
                continue
        
        if electron_path:
            logger.info(f"Starting Electron app with: {electron_path}")
            if electron_path == 'npx electron':
                electron_process = subprocess.Popen(['npx', 'electron', '.'])
            else:
                electron_process = subprocess.Popen([electron_path, '.'])
            return electron_process
        else:
            logger.warning("Electron not found. Please install with: npm install -g electron")
            return None
            
    except Exception as e:
        logger.error(f"Error starting Electron app: {e}")
        return None

def create_electron_files():
    """Create Electron app files."""
    
    # Create package.json
    package_json = {
        "name": "lumina-ai-desktop",
        "version": "1.0.0",
        "description": "LuminaAI Neural Desktop Interface",
        "main": "main.js",
        "scripts": {
            "start": "electron .",
            "dev": "electron . --dev"
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
    main_js = """const { app, BrowserWindow, Menu, dialog, shell, ipcMain } = require('electron');
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
            enableRemoteModule: true
        },
        titleBarStyle: 'hiddenInset',
        vibrancy: 'ultra-dark',
        backgroundColor: '#0a0a0b',
        show: false,
        icon: path.join(__dirname, 'assets', 'icon.png')
    });

    // Load the app
    mainWindow.loadFile('renderer/index.html');

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

    // Create menu
    createMenu();
}

function createMenu() {
    const template = [
        {
            label: 'LuminaAI',
            submenu: [
                { role: 'about' },
                { type: 'separator' },
                { role: 'cut' },
                { role: 'copy' },
                { role: 'paste' },
                { role: 'selectall' }
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
                { role: 'zoom' },
                { type: 'separator' },
                { role: 'front' }
            ]
        },
        {
            label: 'Help',
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
                {
                    label: 'Learn More',
                    click: () => {
                        shell.openExternal('https://github.com');
                    }
                }
            ]
        }
    ];

    const menu = Menu.buildFromTemplate(template);
    Menu.setApplicationMenu(menu);
}

function startPythonBackend() {
    // Start the Python Flask backend
    pythonProcess = spawn('python', ['lumina_desktop.py'], {
        stdio: 'pipe'
    });

    pythonProcess.stdout.on('data', (data) => {
        console.log(`Python: ${data}`);
    });

    pythonProcess.stderr.on('data', (data) => {
        console.error(`Python Error: ${data}`);
    });

    pythonProcess.on('close', (code) => {
        console.log(`Python process exited with code ${code}`);
    });
}

// App event handlers
app.whenReady().then(() => {
    createWindow();
    startPythonBackend();

    app.on('activate', () => {
        if (BrowserWindow.getAllWindows().length === 0) {
            createWindow();
        }
    });
});

app.on('window-all-closed', () => {
    if (pythonProcess) {
        pythonProcess.kill();
    }
    
    if (process.platform !== 'darwin') {
        app.quit();
    }
});

app.on('before-quit', () => {
    if (pythonProcess) {
        pythonProcess.kill();
    }
});

// IPC handlers
ipcMain.handle('get-app-version', () => {
    return app.getVersion();
});
"""

    with open('main.js', 'w') as f:
        f.write(main_js)

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
    create_electron_files()
    
    # Auto-load model if available
    model_path = Path("Model.pth")
    if model_path.exists():
        print(f"🔍 Found model: {model_path}")
        print("🤖 Model will auto-load when interface starts")
    else:
        print("⚠️  No Model.pth found - load manually through interface")
    
    print("\n🌟 Starting LuminaAI Backend Server...")
    print("🔗 Backend running on: http://localhost:5001")
    print("🖥️  Desktop app will launch automatically")
    print("💡 Press Ctrl+C to shutdown")
    print("-" * 60)
    
    # Start Electron app in a separate thread
    electron_process = None
    def start_electron():
        nonlocal electron_process
        import time
        time.sleep(2)  # Wait for Flask to start
        electron_process = run_electron_app()
    
    electron_thread = threading.Thread(target=start_electron, daemon=True)
    electron_thread.start()
    
    def cleanup():
        """Cleanup function."""
        print("\n🔌 Shutting down neural interface...")
        ai_engine.cleanup()
        if electron_process:
            try:
                electron_process.terminate()
            except:
                pass
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
        socketio.run(app, host='127.0.0.1', port=5001, debug=False, 
                    allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"\n❌ Critical system error: {e}")
        return 1
    
    return 0
