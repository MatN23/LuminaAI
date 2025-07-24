# Copywrite (c) 2025 Matias Nielsen. All rights reserved.
# Licensed under the Custom License below.

import torch
import torch.nn as nn
import numpy as np
import re
import math
from pathlib import Path

# Device selection with better detection
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
    print("Using device: MPS (Apple Silicon)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using device: CUDA ({torch.cuda.get_device_name()})")
else:
    device = torch.device("cpu")
    print("Using device: CPU")

# --- Positional Encoding ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
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

# --- Transformer Model ---
class CharTransformer(nn.Module):
    def __init__(self, vocab_size, hidden_size, seq_length, num_layers, nhead, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_enc = PositionalEncoding(hidden_size, seq_length, dropout)
        
        # Use decoder layers for causal language modeling
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size, 
            nhead=nhead, 
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.fc_out = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_square_subsequent_mask(self, sz):
        """Generate causal mask for autoregressive generation"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        return mask.masked_fill(mask == 1, float('-inf'))

    def forward(self, x):
        seq_len = x.size(1)
        mask = self.generate_square_subsequent_mask(seq_len).to(x.device)
        
        # Embedding and positional encoding
        x = self.embedding(x) * math.sqrt(self.hidden_size)  # Scale embeddings
        x = self.pos_enc(x)
        
        # Transformer decoder (for causal modeling)
        x = self.decoder(x, x, tgt_mask=mask)
        x = self.layer_norm(x)
        x = self.dropout(x)
        
        return self.fc_out(x)

def load_model(model_path="best_model.pth"):
    """Load model with error handling"""
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file '{model_path}' not found!")
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        required_keys = ["char_to_ix", "ix_to_char", "hidden_size", "num_layers", "vocab_size"]
        
        for key in required_keys:
            if key not in checkpoint:
                raise KeyError(f"Missing required key '{key}' in checkpoint")
        
        return checkpoint
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")

def nucleus_sampling(probs, p=0.9):
    """Nucleus (top-p) sampling - often better than top-k"""
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumsum_probs = torch.cumsum(sorted_probs, dim=0)
    
    # Find cutoff index
    cutoff = torch.searchsorted(cumsum_probs, p)
    cutoff = max(1, cutoff.item())  # Keep at least one token
    
    # Keep only top-p tokens
    top_p_probs = sorted_probs[:cutoff]
    top_p_indices = sorted_indices[:cutoff]
    
    # Renormalize
    top_p_probs = top_p_probs / top_p_probs.sum()
    
    # Sample
    chosen_idx = torch.multinomial(top_p_probs, 1).item()
    return top_p_indices[chosen_idx].item()

def top_k_sampling(probs, k=5):
    """Improved top-k sampling with torch operations"""
    if k >= len(probs):
        return torch.multinomial(probs, 1).item()
    
    top_k_probs, top_k_indices = torch.topk(probs, k)
    top_k_probs = top_k_probs / top_k_probs.sum()
    chosen_idx = torch.multinomial(top_k_probs, 1).item()
    return top_k_indices[chosen_idx].item()

def sample_text(model, char_to_ix, ix_to_char, start_str, max_length=300, 
                temperature=1.0, sampling_method="top_k", top_k=5, top_p=0.9):
    """Enhanced text sampling with multiple methods"""
    model.eval()
    
    with torch.no_grad():
        # Handle unknown characters gracefully
        input_ix = []
        for ch in start_str:
            if ch in char_to_ix:
                input_ix.append(char_to_ix[ch])
            else:
                # Use space or most common character as fallback
                input_ix.append(char_to_ix.get(" ", 0))
        
        if not input_ix:  # Empty input
            input_ix = [char_to_ix.get(" ", 0)]
        
        generated = torch.tensor(input_ix, dtype=torch.long).unsqueeze(0).to(device)
        
        for _ in range(max_length):
            # Only use last 512 tokens to avoid memory issues and match training seq_length
            input_seq = generated[:, -512:] if generated.size(1) > 512 else generated
            
            outputs = model(input_seq)
            next_token_logits = outputs[0, -1, :] / temperature
            probs = torch.softmax(next_token_logits, dim=0)
            
            # Choose sampling method
            if sampling_method == "nucleus" or sampling_method == "top_p":
                next_ix = nucleus_sampling(probs, p=top_p)
            elif sampling_method == "top_k":
                next_ix = top_k_sampling(probs, k=top_k)
            else:  # greedy
                next_ix = torch.argmax(probs).item()
            
            next_char = ix_to_char.get(next_ix, "")
            if not next_char:  # Skip invalid characters
                continue
                
            # Add to generated sequence
            generated = torch.cat([generated, torch.tensor([[next_ix]], device=device)], dim=1)
            
            # Stop conditions
            if next_char == "\n" and generated.size(1) > len(input_ix) + 10:
                break
        
        # Decode generated text
        output_str = ''.join([ix_to_char.get(idx.item(), '') for idx in generated[0][len(input_ix):]])
        return output_str.strip()

def clean_response(response):
    """Clean up the model's response"""
    # Remove any remaining chat tokens
    response = re.sub(r'<\|[^|]*\|>', '', response)
    
    # Remove extra whitespace and newlines
    response = re.sub(r'\n+', '\n', response).strip()
    
    # Remove incomplete sentences at the end
    sentences = response.split('.')
    if len(sentences) > 1 and len(sentences[-1].strip()) < 5:
        response = '.'.join(sentences[:-1]) + '.'
    
    return response

def print_model_info(checkpoint):
    """Print model information"""
    vocab_size = len(checkpoint["char_to_ix"])
    print(f"\nModel Information:")
    print(f"├── Architecture: Transformer")
    print(f"├── Vocabulary size: {vocab_size:,}")
    print(f"├── Hidden size: {checkpoint['hidden_size']}")
    print(f"├── Number of layers: {checkpoint['num_layers']}")
    print(f"├── Number of heads: {checkpoint.get('nhead', 'Unknown')}")
    if 'epoch' in checkpoint:
        print(f"├── Training epoch: {checkpoint['epoch']}")
    if 'loss' in checkpoint:
        print(f"└── Final loss: {checkpoint['loss']:.4f}")
    print()

def main():
    # Load model and vocab
    try:
        checkpoint = load_model("best_model.pth")
    except (FileNotFoundError, RuntimeError) as e:
        print(f"Error: {e}")
        return
    
    char_to_ix = checkpoint["char_to_ix"]
    ix_to_char = checkpoint["ix_to_char"]
    hidden_size = checkpoint["hidden_size"]
    num_layers = checkpoint["num_layers"]
    nhead = checkpoint.get("nhead", 8)  # Default to 8 heads if not specified
    seq_length = checkpoint.get("seq_length", 512)
    vocab_size = len(char_to_ix)
    
    # Print model info
    print_model_info(checkpoint)
    
    # Initialize model
    model = CharTransformer(vocab_size, hidden_size, seq_length, num_layers, nhead).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    print("🤖 Chat with the Transformer AI!")
    print("Commands:")
    print("  'exit' or 'quit' - Exit the chat")
    print("  'clear' - Clear conversation history")
    print("  'temp X' - Set temperature (e.g., 'temp 0.8')")
    print("  'topk X' - Set top-k value (e.g., 'topk 10')")
    print("  'nucleus' - Switch to nucleus sampling")
    print("  'topk_mode' - Switch to top-k sampling")
    print("-" * 50)
    
    conversation_history = ""
    temperature = 0.7
    sampling_method = "top_k"
    top_k = 8
    top_p = 0.9
    max_history_length = 2000  # Prevent context from getting too long
    
    while True:
        try:
            user_input = input("\n🧑 You: ").strip()
            
            if not user_input:
                continue
                
            # Handle commands
            if user_input.lower() in ["exit", "quit"]:
                print("👋 Goodbye!")
                break
            elif user_input.lower() == "clear":
                conversation_history = ""
                print("🗑️  Conversation history cleared!")
                continue
            elif user_input.startswith("temp "):
                try:
                    temperature = float(user_input.split()[1])
                    temperature = max(0.1, min(2.0, temperature))  # Clamp between 0.1 and 2.0
                    print(f"🌡️  Temperature set to {temperature}")
                    continue
                except (IndexError, ValueError):
                    print("❌ Invalid temperature. Use: temp 0.8")
                    continue
            elif user_input.startswith("topk "):
                try:
                    top_k = int(user_input.split()[1])
                    top_k = max(1, min(50, top_k))  # Clamp between 1 and 50
                    sampling_method = "top_k"
                    print(f"🔢 Top-k set to {top_k}")
                    continue
                except (IndexError, ValueError):
                    print("❌ Invalid top-k value. Use: topk 10")
                    continue
            elif user_input.lower() == "nucleus":
                sampling_method = "nucleus"
                print(f"🎯 Switched to nucleus sampling (p={top_p})")
                continue
            elif user_input.lower() == "topk_mode":
                sampling_method = "top_k"
                print(f"🔢 Switched to top-k sampling (k={top_k})")
                continue
            
            # Build context with chat tokens
            conversation_history += f"<|user|>{user_input}<|bot|>"
            
            # Truncate history if too long
            if len(conversation_history) > max_history_length:
                # Keep the last part of the conversation
                lines = conversation_history.split("<|user|>")
                conversation_history = "<|user|>" + "<|user|>".join(lines[-3:])
            
            # Generate response
            print("🤖 AI: ", end="", flush=True)
            response = sample_text(
                model, char_to_ix, ix_to_char, 
                conversation_history,
                max_length=300,
                temperature=temperature,
                sampling_method=sampling_method,
                top_k=top_k,
                top_p=top_p
            )
            
            # Extract just the bot's response (after the last <|bot|> token)
            if "<|bot|>" in response:
                response = response.split("<|bot|>")[-1]
            
            # Clean up response
            response = clean_response(response)
            
            # Print response
            if response:
                print(response)
                # Add response to history
                conversation_history += response + "\n"
            else:
                print("(No response generated)")
        
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Continuing...")

if __name__ == "__main__":
    main()