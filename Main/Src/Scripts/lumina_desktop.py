#!/usr/bin/env python3
"""
LuminaAI Desktop GUI Application
Cross-platform desktop interface for the character-level transformer chatbot.
Compatible with existing ChatAI.py and Model.pth files.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import threading
import queue
import sys
import os
from pathlib import Path
import json
import logging
import gc
from typing import Dict, Optional
import time
import re
import math

# Import the core AI components
try:
    import torch
    import torch.nn as nn
    import numpy as np
except ImportError as e:
    messagebox.showerror("Missing Dependencies", 
                        f"Required packages not installed: {e}\n\n"
                        "Please install requirements.txt:\n"
                        "pip install -r requirements.txt")
    sys.exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer models."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class CharTransformer(nn.Module):
    """Character-level transformer model - matches your existing architecture."""
    
    def __init__(self, vocab_size: int, hidden_size: int, seq_length: int, 
                 num_layers: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_enc = PositionalEncoding(hidden_size, seq_length, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, 
            nhead=nhead, 
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.fc_out = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate causal mask for autoregressive generation."""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        return mask.masked_fill(mask == 1, float('-inf'))

    def forward(self, x):
        seq_len = x.size(1)
        mask = self.generate_square_subsequent_mask(seq_len).to(x.device)
        
        x = self.embedding(x) * math.sqrt(self.hidden_size)
        x = self.pos_enc(x)
        x = self.transformer(x, mask=mask)
        x = self.layer_norm(x)
        x = self.dropout(x)
        
        return self.fc_out(x)

class AIEngine:
    """AI Engine that matches your existing ChatAI.py functionality."""
    
    def __init__(self):
        self.model = None
        self.char_to_ix = {}
        self.ix_to_char = {}
        self.device = self.setup_device()
        self.is_loaded = False
        self.model_info = {}
        
    def setup_device(self):
        """Setup the best available device."""
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
    
    def load_model(self, model_path: str) -> tuple[bool, str]:
        """Load the AI model from .pth file - matches your existing format."""
        try:
            model_path = Path(model_path)
            if not model_path.exists():
                return False, f"Model file not found: {model_path}"
            
            logger.info(f"Loading model from: {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Validate checkpoint structure (same as your ChatAI.py)
            required_keys = ["char_to_ix", "ix_to_char", "model_state_dict"]
            missing_keys = [key for key in required_keys if key not in checkpoint]
            if missing_keys:
                return False, f"Missing required keys in checkpoint: {missing_keys}"
            
            # Extract vocabularies
            self.char_to_ix = checkpoint["char_to_ix"]
            self.ix_to_char = checkpoint["ix_to_char"]
            
            # Validate vocabulary consistency
            if len(self.char_to_ix) != len(self.ix_to_char):
                return False, "Vocabulary mappings are inconsistent"
            
            # Extract model configuration
            if 'config' in checkpoint:
                config = checkpoint['config']
                hidden_size = config['hidden_size']
                num_layers = config['num_layers']
                nhead = config.get('nhead', 8)
                seq_length = config.get('seq_length', 512)
                dropout = config.get('dropout', 0.1)
            else:
                # Legacy format
                hidden_size = checkpoint.get("hidden_size", 512)
                num_layers = checkpoint.get("num_layers", 6)
                nhead = checkpoint.get("nhead", 8)
                seq_length = checkpoint.get("seq_length", 512)
                dropout = 0.1
            
            vocab_size = len(self.char_to_ix)
            
            # Store model info for display
            self.model_info = {
                'vocab_size': vocab_size,
                'hidden_size': hidden_size,
                'num_layers': num_layers,
                'nhead': nhead,
                'seq_length': seq_length,
                'epoch': checkpoint.get('epoch', 'Unknown'),
                'loss': checkpoint.get('loss', 'Unknown'),
                'accuracy': checkpoint.get('accuracy', 'Unknown'),
                'device': str(self.device)
            }
            
            # Initialize model
            self.model = CharTransformer(
                vocab_size, hidden_size, seq_length, num_layers, nhead, dropout
            ).to(self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()
            
            self.is_loaded = True
            logger.info("Model loaded successfully!")
            return True, "Model loaded successfully!"
            
        except Exception as e:
            error_msg = f"Error loading model: {e}"
            logger.error(error_msg)
            return False, error_msg
    
    def nucleus_sampling(self, probs: torch.Tensor, p: float = 0.9) -> int:
        """Nucleus sampling - matches your ChatAI.py implementation."""
        if p <= 0 or p > 1:
            p = 0.9
        
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
    
    def top_k_sampling(self, probs: torch.Tensor, k: int = 5) -> int:
        """Top-k sampling - matches your ChatAI.py implementation."""
        if k <= 0:
            k = 5
        
        if k >= len(probs):
            try:
                return torch.multinomial(probs, 1).item()
            except RuntimeError:
                return torch.argmax(probs).item()
        
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
    
    def generate_response(self, user_input: str, conversation_history: str = "", 
                         temperature: float = 0.7, sampling_method: str = "top_k",
                         top_k: int = 8, top_p: float = 0.9, max_length: int = 300) -> str:
        """Generate AI response - matches your ChatAI.py sample_text function."""
        if not self.is_loaded:
            return "❌ Model not loaded. Please load a model first."
        
        if not user_input.strip():
            return "Please enter a message."
        
        try:
            with torch.no_grad():
                # Build context with chat tokens
                context = conversation_history + f"<|user|>{user_input}<|bot|>"
                
                # Handle unknown characters gracefully
                input_ix = []
                for ch in context:
                    if ch in self.char_to_ix:
                        input_ix.append(self.char_to_ix[ch])
                    else:
                        fallback_char = ' ' if ' ' in self.char_to_ix else list(self.char_to_ix.keys())[0]
                        input_ix.append(self.char_to_ix[fallback_char])
                
                if not input_ix:
                    input_ix = [self.char_to_ix.get(" ", 0)]
                
                generated = torch.tensor(input_ix, dtype=torch.long).unsqueeze(0).to(self.device)
                
                for step in range(max_length):
                    try:
                        # Use last seq_length tokens to match training
                        max_seq_length = getattr(self.model, 'seq_length', 512)
                        input_seq = generated[:, -max_seq_length:] if generated.size(1) > max_seq_length else generated
                        
                        outputs = self.model(input_seq)
                        next_token_logits = outputs[0, -1, :] / temperature
                        
                        # Apply softmax to get probabilities
                        probs = torch.softmax(next_token_logits, dim=0)
                        
                        # Handle potential NaN/Inf values
                        if torch.isnan(probs).any() or torch.isinf(probs).any():
                            probs = torch.ones_like(probs) / len(probs)
                        
                        # Choose sampling method
                        if sampling_method == "nucleus" or sampling_method == "top_p":
                            next_ix = self.nucleus_sampling(probs, p=top_p)
                        elif sampling_method == "top_k":
                            next_ix = self.top_k_sampling(probs, k=top_k)
                        elif sampling_method == "greedy":
                            next_ix = torch.argmax(probs).item()
                        else:
                            next_ix = self.top_k_sampling(probs, k=top_k)
                        
                        # Validate next_ix
                        if next_ix < 0 or next_ix >= len(self.ix_to_char):
                            next_ix = 0
                        
                        next_char = self.ix_to_char.get(next_ix, "")
                        if not next_char:
                            continue
                        
                        # Add to generated sequence
                        generated = torch.cat([generated, torch.tensor([[next_ix]], device=self.device)], dim=1)
                        
                        # Stop conditions
                        if next_char == "\n" and generated.size(1) > len(input_ix) + 10:
                            break
                    
                    except Exception as e:
                        logger.error(f"Error during generation step {step}: {e}")
                        break
                
                # Decode generated text
                output_chars = []
                for idx in generated[0][len(input_ix):]:
                    char = self.ix_to_char.get(idx.item(), '')
                    if char:
                        output_chars.append(char)
                
                response = ''.join(output_chars).strip()
                
                # Extract just the bot's response
                if "<|bot|>" in response:
                    response = response.split("<|bot|>")[-1]
                
                # Clean response (matches your clean_response function)
                response = re.sub(r'<\|[^|]*\|>', '', response)
                response = re.sub(r'\n+', '\n', response)
                response = re.sub(r' +', ' ', response)
                response = response.strip()
                
                return response if response else "I'm not sure how to respond to that."
                
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def cleanup(self):
        """Clean up GPU memory."""
        try:
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            elif self.device.type == 'mps' and hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
            gc.collect()
        except Exception:
            pass

class LuminaAIApp:
    """Main GUI Application Class."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("🤖 LuminaAI - Character-Level Transformer Chatbot")
        self.root.geometry("1000x700")
        
        # Configure style
        style = ttk.Style()
        if "clam" in style.theme_names():
            style.theme_use("clam")
        
        # Initialize AI engine
        self.ai_engine = AIEngine()
        self.conversation_history = ""
        self.max_history_length = 2000
        
        # Settings
        self.temperature = tk.DoubleVar(value=0.7)
        self.sampling_method = tk.StringVar(value="top_k")
        self.top_k = tk.IntVar(value=8)
        self.top_p = tk.DoubleVar(value=0.9)
        
        # Message queue for threading
        self.message_queue = queue.Queue()
        
        self.setup_ui()
        self.check_queue()
        
        # Auto-load model if Model.pth exists
        if Path("Model.pth").exists():
            self.load_model("Model.pth")
    
    def setup_ui(self):
        """Setup the user interface."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Header
        header_frame = ttk.Frame(main_frame)
        header_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        title_label = ttk.Label(header_frame, text="🤖 LuminaAI Chatbot", 
                               font=("Arial", 16, "bold"))
        title_label.pack(side=tk.LEFT)
        
        # Model controls
        model_frame = ttk.Frame(header_frame)
        model_frame.pack(side=tk.RIGHT)
        
        ttk.Button(model_frame, text="Load Model", command=self.browse_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(model_frame, text="Model Info", command=self.show_model_info).pack(side=tk.LEFT)
        
        # Settings panel
        settings_frame = ttk.LabelFrame(main_frame, text="Settings", padding="10")
        settings_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Temperature
        ttk.Label(settings_frame, text="Temperature:").grid(row=0, column=0, sticky=tk.W, pady=2)
        temp_scale = ttk.Scale(settings_frame, from_=0.1, to=2.0, variable=self.temperature, 
                              orient=tk.HORIZONTAL, length=200)
        temp_scale.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5, pady=2)
        temp_label = ttk.Label(settings_frame, text="0.7")
        temp_label.grid(row=0, column=2, sticky=tk.W, pady=2)
        
        # Update temperature label
        def update_temp_label(*args):
            temp_label.config(text=f"{self.temperature.get():.1f}")
        self.temperature.trace('w', update_temp_label)
        
        # Sampling method
        ttk.Label(settings_frame, text="Sampling:").grid(row=1, column=0, sticky=tk.W, pady=2)
        sampling_combo = ttk.Combobox(settings_frame, textvariable=self.sampling_method, 
                                     values=["top_k", "nucleus", "greedy"], state="readonly", width=15)
        sampling_combo.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Top-K
        ttk.Label(settings_frame, text="Top-K:").grid(row=2, column=0, sticky=tk.W, pady=2)
        topk_spin = ttk.Spinbox(settings_frame, from_=1, to=50, textvariable=self.top_k, width=10)
        topk_spin.grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Top-P
        ttk.Label(settings_frame, text="Top-P:").grid(row=3, column=0, sticky=tk.W, pady=2)
        topp_scale = ttk.Scale(settings_frame, from_=0.1, to=1.0, variable=self.top_p, 
                              orient=tk.HORIZONTAL, length=200)
        topp_scale.grid(row=3, column=1, sticky=(tk.W, tk.E), padx=5, pady=2)
        topp_label = ttk.Label(settings_frame, text="0.9")
        topp_label.grid(row=3, column=2, sticky=tk.W, pady=2)
        
        def update_topp_label(*args):
            topp_label.config(text=f"{self.top_p.get():.1f}")
        self.top_p.trace('w', update_topp_label)
        
        # Clear button
        ttk.Button(settings_frame, text="Clear Chat", command=self.clear_chat).grid(row=4, column=0, columnspan=2, pady=10)
        
        # Status
        self.status_var = tk.StringVar(value="No model loaded")
        status_label = ttk.Label(settings_frame, textvariable=self.status_var, foreground="red")
        status_label.grid(row=5, column=0, columnspan=3, pady=5)
        
        # Chat area
        chat_frame = ttk.Frame(main_frame)
        chat_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        chat_frame.columnconfigure(0, weight=1)
        chat_frame.rowconfigure(0, weight=1)
        
        # Chat display
        self.chat_display = scrolledtext.ScrolledText(chat_frame, wrap=tk.WORD, width=60, height=25,
                                                     font=("Consolas", 10))
        self.chat_display.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # Configure text tags for styling
        self.chat_display.tag_configure("user", foreground="blue", font=("Consolas", 10, "bold"))
        self.chat_display.tag_configure("ai", foreground="green", font=("Consolas", 10))
        self.chat_display.tag_configure("system", foreground="gray", font=("Consolas", 9, "italic"))
        
        # Input area
        input_frame = ttk.Frame(chat_frame)
        input_frame.grid(row=1, column=0, sticky=(tk.W, tk.E))
        input_frame.columnconfigure(0, weight=1)
        
        self.user_input = tk.Text(input_frame, height=3, wrap=tk.WORD, font=("Consolas", 10))
        self.user_input.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 10))
        
        button_frame = ttk.Frame(input_frame)
        button_frame.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        self.send_button = ttk.Button(button_frame, text="Send", command=self.send_message)
        self.send_button.pack(fill=tk.X, pady=(0, 5))
        
        # Bind Enter key
        self.user_input.bind('<Control-Return>', lambda e: self.send_message())
        
        # Initial welcome message
        self.add_message("system", "🤖 Welcome to LuminaAI! Load a model to start chatting.")
        self.add_message("system", "💡 Tip: Use Ctrl+Enter to send messages quickly.")
    
    def add_message(self, sender: str, message: str):
        """Add a message to the chat display."""
        self.chat_display.config(state=tk.NORMAL)
        
        if sender == "user":
            self.chat_display.insert(tk.END, "🧑 You: ", "user")
            self.chat_display.insert(tk.END, f"{message}\n\n")
        elif sender == "ai":
            self.chat_display.insert(tk.END, "🤖 AI: ", "ai")
            self.chat_display.insert(tk.END, f"{message}\n\n")
        else:  # system
            self.chat_display.insert(tk.END, f"{message}\n", "system")
        
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)
    
    def browse_model(self):
        """Browse for model file."""
        file_path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("PyTorch Models", "*.pth"), ("All Files", "*.*")],
            initialdir="."
        )
        
        if file_path:
            self.load_model(file_path)
    
    def load_model(self, model_path: str):
        """Load the AI model."""
        self.add_message("system", f"Loading model from: {Path(model_path).name}")
        self.status_var.set("Loading model...")
        self.root.config(cursor="wait")
        
        def load_thread():
            success, message = self.ai_engine.load_model(model_path)
            self.message_queue.put(("model_loaded", success, message))
        
        threading.Thread(target=load_thread, daemon=True).start()
    
    def send_message(self):
        """Send user message and get AI response."""
        user_text = self.user_input.get("1.0", tk.END).strip()
        if not user_text:
            return
        
        if not self.ai_engine.is_loaded:
            messagebox.showwarning("No Model", "Please load a model first!")
            return
        
        # Add user message
        self.add_message("user", user_text)
        self.user_input.delete("1.0", tk.END)
        
        # Update conversation history
        self.conversation_history += f"<|user|>{user_text}<|bot|>"
        
        # Truncate history if too long
        if len(self.conversation_history) > self.max_history_length:
            lines = self.conversation_history.split("<|user|>")
            self.conversation_history = "<|user|>" + "<|user|>".join(lines[-3:])
        
        # Show thinking message
        self.add_message("system", "🤔 Thinking...")
        self.send_button.config(state="disabled")
        
        def generate_thread():
            try:
                response = self.ai_engine.generate_response(
                    user_text,
                    conversation_history=self.conversation_history.rsplit("<|bot|>", 1)[0] if "<|bot|>" in self.conversation_history else "",
                    temperature=self.temperature.get(),
                    sampling_method=self.sampling_method.get(),
                    top_k=self.top_k.get(),
                    top_p=self.top_p.get(),
                    max_length=300
                )
                self.message_queue.put(("response", response))
            except Exception as e:
                self.message_queue.put(("error", str(e)))
        
        threading.Thread(target=generate_thread, daemon=True).start()
    
    def clear_chat(self):
        """Clear the chat history."""
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.delete("1.0", tk.END)
        self.chat_display.config(state=tk.DISABLED)
        self.conversation_history = ""
        self.add_message("system", "Chat cleared!")
    
    def show_model_info(self):
        """Show model information."""
        if not self.ai_engine.is_loaded:
            messagebox.showinfo("No Model", "No model is currently loaded.")
            return
        
        info = self.ai_engine.model_info
        info_text = f"""🤖 Model Information
        
Architecture: Character-level Transformer
Vocabulary Size: {info.get('vocab_size', 'Unknown'):,}
Hidden Dimension: {info.get('hidden_size', 'Unknown')}
Number of Layers: {info.get('num_layers', 'Unknown')}
Attention Heads: {info.get('nhead', 'Unknown')}
Sequence Length: {info.get('seq_length', 'Unknown')}
Device: {info.get('device', 'Unknown')}

Training Info:
Final Epoch: {info.get('epoch', 'Unknown')}
Final Loss: {info.get('loss', 'Unknown')}
Final Accuracy: {f"{info.get('accuracy', 0)*100:.2f}%" if isinstance(info.get('accuracy'), (int, float)) else 'Unknown'}
"""
        
        messagebox.showinfo("Model Information", info_text)
    
    def check_queue(self):
        """Check for messages from background threads."""
        try:
            while True:
                message_type, *args = self.message_queue.get_nowait()
                
                if message_type == "model_loaded":
                    success, message = args
                    self.root.config(cursor="")
                    
                    if success:
                        self.status_var.set("✅ Model loaded successfully")
                        self.add_message("system", "✅ Model loaded! You can now start chatting.")
                    else:
                        self.status_var.set("❌ Failed to load model")
                        self.add_message("system", f"❌ Error loading model: {message}")
                
                elif message_type == "response":
                    response = args[0]
                    # Remove the "thinking" message
                    self.chat_display.config(state=tk.NORMAL)
                    # Find and remove the last "Thinking..." line
                    content = self.chat_display.get("1.0", tk.END)
                    lines = content.split('\n')
                    if lines and "🤔 Thinking..." in lines[-2]:
                        # Remove the thinking line
                        self.chat_display.delete("end-2l linestart", "end-1l linestart")
                    self.chat_display.config(state=tk.DISABLED)
                    
                    self.add_message("ai", response)
                    self.conversation_history += response + "\n"
                    self.send_button.config(state="normal")
                
                elif message_type == "error":
                    error = args[0]
                    # Remove the "thinking" message
                    self.chat_display.config(state=tk.NORMAL)
                    content = self.chat_display.get("1.0", tk.END)
                    lines = content.split('\n')
                    if lines and "🤔 Thinking..." in lines[-2]:
                        self.chat_display.delete("end-2l linestart", "end-1l linestart")
                    self.chat_display.config(state=tk.DISABLED)
                    
                    self.add_message("system", f"❌ Error: {error}")
                    self.send_button.config(state="normal")
        
        except queue.Empty:
            pass
        
        # Schedule next check
        self.root.after(100, self.check_queue)
    
    def on_closing(self):
        """Handle application closing."""
        if messagebox.askokcancel("Quit", "Do you want to quit LuminaAI?"):
            self.ai_engine.cleanup()
            self.root.destroy()

def main():
    """Main application entry point."""
    root = tk.Tk()
    
    # Set application icon (if available)
    try:
        # You can add an icon file here
        # root.iconbitmap("icon.ico")  # Windows
        # root.iconphoto(True, tk.PhotoImage(file="icon.png"))  # Cross-platform
        pass
    except:
        pass
    
    app = LuminaAIApp(root)
    
    # Handle window closing
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    # Center window on screen
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f"{width}x{height}+{x}+{y}")
    
    # Start the application
    print("🚀 Starting LuminaAI Desktop Application...")
    print("🤖 Compatible with your existing Model.pth files")
    print("⚡ GPU acceleration supported (CUDA/MPS/CPU)")
    
    root.mainloop()

if __name__ == "__main__":
    main()