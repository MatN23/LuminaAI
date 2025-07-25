# Copyright (c) 2025 Matias Nielsen. All rights reserved.
# Licensed under the Custom License below.

import torch
import torch.nn as nn
import numpy as np
import re
import math
import logging
import gc
from pathlib import Path
from typing import Dict, Optional, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Device selection with better detection and error handling
def setup_device():
    """Setup the best available device with comprehensive error handling."""
    try:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps")
            logger.info("Using device: MPS (Apple Silicon)")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using device: CUDA ({torch.cuda.get_device_name()})")
            logger.info(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            device = torch.device("cpu")
            logger.info("Using device: CPU")
        return device
    except Exception as e:
        logger.warning(f"Error setting up device: {e}. Falling back to CPU.")
        return torch.device("cpu")

device = setup_device()

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
    """Fixed character-level transformer model using proper encoder architecture."""
    
    def __init__(self, vocab_size: int, hidden_size: int, seq_length: int, 
                 num_layers: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_enc = PositionalEncoding(hidden_size, seq_length, dropout)
        
        # Use TransformerEncoder with causal mask for decoder-only model
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
        
        # Embedding and positional encoding
        x = self.embedding(x) * math.sqrt(self.hidden_size)  # Scale embeddings
        x = self.pos_enc(x)
        
        # Transformer with causal mask
        x = self.transformer(x, mask=mask)
        x = self.layer_norm(x)
        x = self.dropout(x)
        
        return self.fc_out(x)

def load_model(model_path: str = "Model.pth") -> Dict:
    """Load model with comprehensive error handling and validation."""
    model_path = Path(model_path)
    
    if not model_path.exists():
        available_models = list(Path.cwd().glob("*.pth"))
        if available_models:
            logger.error(f"Model file '{model_path}' not found!")
            logger.error(f"Available models: {[str(p.name) for p in available_models]}")
        else:
            logger.error(f"Model file '{model_path}' not found and no .pth files in current directory!")
        raise FileNotFoundError(f"Model file '{model_path}' not found!")
    
    try:
        logger.info(f"Loading model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        
        # Validate required keys
        required_keys = ["char_to_ix", "ix_to_char", "model_state_dict"]
        missing_keys = [key for key in required_keys if key not in checkpoint]
        
        if missing_keys:
            raise KeyError(f"Missing required keys in checkpoint: {missing_keys}")
        
        # Validate vocabulary consistency
        char_to_ix = checkpoint["char_to_ix"]
        ix_to_char = checkpoint["ix_to_char"]
        
        if len(char_to_ix) != len(ix_to_char):
            raise ValueError("Vocabulary mappings are inconsistent")
        
        # Check for model configuration
        config_keys = ['hidden_size', 'num_layers', 'vocab_size']
        if 'config' in checkpoint:
            # New format with config dict
            config = checkpoint['config']
            for key in config_keys:
                if key not in config:
                    logger.warning(f"Missing config key '{key}', using default")
        else:
            # Legacy format - individual keys
            for key in config_keys:
                if key not in checkpoint:
                    logger.warning(f"Missing key '{key}' in checkpoint")
        
        logger.info("Model loaded successfully!")
        return checkpoint
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise RuntimeError(f"Error loading model: {e}")

def nucleus_sampling(probs: torch.Tensor, p: float = 0.9) -> int:
    """
    Improved nucleus (top-p) sampling with better error handling.
    """
    if p <= 0 or p > 1:
        raise ValueError(f"p must be in (0, 1], got {p}")
    
    # Sort probabilities in descending order
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumsum_probs = torch.cumsum(sorted_probs, dim=0)
    
    # Find cutoff index where cumulative probability exceeds p
    cutoff_mask = cumsum_probs <= p
    
    # Ensure we keep at least one token
    if not cutoff_mask.any():
        cutoff = 1
    else:
        cutoff = cutoff_mask.sum().item()
        cutoff = max(1, cutoff)  # Ensure at least 1 token
    
    # Keep only top-p tokens
    top_p_probs = sorted_probs[:cutoff]
    top_p_indices = sorted_indices[:cutoff]
    
    # Renormalize probabilities
    if top_p_probs.sum() > 0:
        top_p_probs = top_p_probs / top_p_probs.sum()
    else:
        # Fallback to uniform distribution
        top_p_probs = torch.ones_like(top_p_probs) / len(top_p_probs)
    
    # Sample from the filtered distribution
    try:
        chosen_idx = torch.multinomial(top_p_probs, 1).item()
        return top_p_indices[chosen_idx].item()
    except RuntimeError:
        # Fallback to most likely token
        return top_p_indices[0].item()

def top_k_sampling(probs: torch.Tensor, k: int = 5) -> int:
    """Improved top-k sampling with better error handling."""
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    
    if k >= len(probs):
        # If k is larger than vocabulary, use all tokens
        try:
            return torch.multinomial(probs, 1).item()
        except RuntimeError:
            return torch.argmax(probs).item()
    
    # Get top-k probabilities and indices
    top_k_probs, top_k_indices = torch.topk(probs, k)
    
    # Renormalize
    if top_k_probs.sum() > 0:
        top_k_probs = top_k_probs / top_k_probs.sum()
    else:
        top_k_probs = torch.ones_like(top_k_probs) / k
    
    # Sample
    try:
        chosen_idx = torch.multinomial(top_k_probs, 1).item()
        return top_k_indices[chosen_idx].item()
    except RuntimeError:
        return top_k_indices[0].item()

def sample_text(model, char_to_ix: Dict[str, int], ix_to_char: Dict[int, str], 
                start_str: str, max_length: int = 300, temperature: float = 1.0, 
                sampling_method: str = "top_k", top_k: int = 5, top_p: float = 0.9) -> str:
    """
    Enhanced text sampling with multiple methods and comprehensive error handling.
    """
    if not start_str:
        logger.warning("Empty start string, using default")
        start_str = "<|user|>"
    
    if temperature <= 0:
        logger.warning(f"Invalid temperature {temperature}, using 0.7")
        temperature = 0.7
    
    if max_length <= 0:
        logger.warning(f"Invalid max_length {max_length}, using 300")
        max_length = 300
    
    model.eval()
    
    try:
        with torch.no_grad():
            # Handle unknown characters gracefully
            input_ix = []
            for ch in start_str:
                if ch in char_to_ix:
                    input_ix.append(char_to_ix[ch])
                else:
                    # Use space or most common character as fallback
                    fallback_char = ' ' if ' ' in char_to_ix else list(char_to_ix.keys())[0]
                    input_ix.append(char_to_ix[fallback_char])
                    logger.debug(f"Unknown character '{ch}' replaced with '{fallback_char}'")
            
            if not input_ix:  # Empty input
                input_ix = [char_to_ix.get(" ", 0)]
            
            generated = torch.tensor(input_ix, dtype=torch.long).unsqueeze(0).to(device)
            
            for step in range(max_length):
                try:
                    # Use last seq_length tokens to match training
                    max_seq_length = getattr(model, 'seq_length', 512)
                    input_seq = generated[:, -max_seq_length:] if generated.size(1) > max_seq_length else generated
                    
                    outputs = model(input_seq)
                    next_token_logits = outputs[0, -1, :] / temperature
                    
                    # Apply softmax to get probabilities
                    probs = torch.softmax(next_token_logits, dim=0)
                    
                    # Handle potential NaN/Inf values
                    if torch.isnan(probs).any() or torch.isinf(probs).any():
                        logger.warning("NaN/Inf in probabilities, using uniform distribution")
                        probs = torch.ones_like(probs) / len(probs)
                    
                    # Choose sampling method
                    if sampling_method == "nucleus" or sampling_method == "top_p":
                        next_ix = nucleus_sampling(probs, p=top_p)
                    elif sampling_method == "top_k":
                        next_ix = top_k_sampling(probs, k=top_k)
                    elif sampling_method == "greedy":
                        next_ix = torch.argmax(probs).item()
                    else:
                        logger.warning(f"Unknown sampling method '{sampling_method}', using top_k")
                        next_ix = top_k_sampling(probs, k=top_k)
                    
                    # Validate next_ix
                    if next_ix < 0 or next_ix >= len(ix_to_char):
                        logger.warning(f"Invalid token index {next_ix}, using 0")
                        next_ix = 0
                    
                    next_char = ix_to_char.get(next_ix, "")
                    if not next_char:  # Skip invalid characters
                        continue
                        
                    # Add to generated sequence
                    generated = torch.cat([generated, torch.tensor([[next_ix]], device=device)], dim=1)
                    
                    # Stop conditions
                    if next_char == "\n" and generated.size(1) > len(input_ix) + 10:
                        break
                        
                    # Prevent infinite loops
                    if step > 0 and step % 100 == 0:
                        logger.debug(f"Generation step {step}/{max_length}")
                
                except Exception as e:
                    logger.error(f"Error during generation step {step}: {e}")
                    break
            
            # Decode generated text
            try:
                output_chars = []
                for idx in generated[0][len(input_ix):]:
                    char = ix_to_char.get(idx.item(), '')
                    if char:  # Only add valid characters
                        output_chars.append(char)
                
                output_str = ''.join(output_chars).strip()
                return output_str if output_str else "Unable to generate text"
                
            except Exception as e:
                logger.error(f"Error decoding generated text: {e}")
                return "Error during text decoding"
    
    except Exception as e:
        logger.error(f"Error in text generation: {e}")
        return f"Generation error: {str(e)}"
    
    finally:
        # Memory cleanup
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        elif device.type == 'mps' and hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
        gc.collect()

def clean_response(response: str) -> str:
    """Clean up the model's response with improved processing."""
    if not response:
        return ""
    
    # Remove any remaining chat tokens
    response = re.sub(r'<\|[^|]*\|>', '', response)
    
    # Remove excessive whitespace and newlines
    response = re.sub(r'\n+', '\n', response)
    response = re.sub(r' +', ' ', response)
    response = response.strip()
    
    # Remove incomplete sentences at the end (more sophisticated)
    if response:
        sentences = re.split(r'[.!?]+', response)
        if len(sentences) > 1:
            # Check if last sentence is too short or incomplete
            last_sentence = sentences[-1].strip()
            if len(last_sentence) < 5 or not last_sentence:
                # Remove the incomplete last sentence
                response = re.sub(r'[.!?]*\s*' + re.escape(last_sentence) + r'$', '', response)
                # Ensure proper ending punctuation
                if response and not response[-1] in '.!?':
                    response += '.'
    
    return response

def print_model_info(checkpoint: Dict):
    """Print comprehensive model information."""
    try:
        if 'config' in checkpoint:
            config = checkpoint['config']
            vocab_size = config.get('vocab_size', len(checkpoint.get("char_to_ix", {})))
            hidden_size = config.get('hidden_size', 'Unknown')
            num_layers = config.get('num_layers', 'Unknown')
            nhead = config.get('nhead', 'Unknown')
            seq_length = config.get('seq_length', 'Unknown')
        else:
            # Legacy format
            vocab_size = checkpoint.get('vocab_size', len(checkpoint.get("char_to_ix", {})))
            hidden_size = checkpoint.get("hidden_size", 'Unknown')
            num_layers = checkpoint.get("num_layers", 'Unknown')
            nhead = checkpoint.get("nhead", 'Unknown')
            seq_length = checkpoint.get("seq_length", 'Unknown')
        
        print(f"\n{'='*50}")
        print(f"🤖 MODEL INFORMATION")
        print(f"{'='*50}")
        print(f"├── Architecture: Character-level Transformer")
        print(f"├── Vocabulary size: {vocab_size:,}")
        print(f"├── Hidden dimension: {hidden_size}")
        print(f"├── Number of layers: {num_layers}")
        print(f"├── Attention heads: {nhead}")
        print(f"├── Sequence length: {seq_length}")
        
        if 'epoch' in checkpoint:
            print(f"├── Training epoch: {checkpoint['epoch']}")
        if 'loss' in checkpoint:
            print(f"├── Final loss: {checkpoint['loss']:.4f}")
        if 'accuracy' in checkpoint:
            print(f"├── Final accuracy: {checkpoint['accuracy']*100:.2f}%")
        if checkpoint.get('fine_tuned', False):
            print(f"├── Status: Fine-tuned model")
            print(f"├── Original vocab: {checkpoint.get('original_vocab_size', 'Unknown')}")
            print(f"└── Extended vocab: {checkpoint.get('extended_vocab_size', 'Unknown')}")
        else:
            print(f"└── Status: Base trained model")
        
        print(f"{'='*50}\n")
        
    except Exception as e:
        logger.error(f"Error printing model info: {e}")
        print(f"\n🤖 Model loaded (detailed info unavailable: {e})\n")

def validate_model_checkpoint(checkpoint: Dict) -> bool:
    """Validate that the loaded checkpoint is complete and usable."""
    try:
        # Check essential components
        if 'model_state_dict' not in checkpoint:
            logger.error("Missing model_state_dict in checkpoint")
            return False
        
        if 'char_to_ix' not in checkpoint or 'ix_to_char' not in checkpoint:
            logger.error("Missing vocabulary mappings in checkpoint")
            return False
        
        char_to_ix = checkpoint['char_to_ix']
        ix_to_char = checkpoint['ix_to_char']
        
        # Validate vocabulary consistency
        if len(char_to_ix) != len(ix_to_char):
            logger.error("Vocabulary mappings have inconsistent sizes")
            return False
        
        # Check if vocabularies are properly mapped
        for char, idx in char_to_ix.items():
            if idx not in ix_to_char or ix_to_char[idx] != char:
                logger.error(f"Vocabulary mapping inconsistency for '{char}' -> {idx}")
                return False
        
        # Check for minimum vocabulary size
        if len(char_to_ix) < 10:
            logger.warning(f"Very small vocabulary size: {len(char_to_ix)}")
        
        logger.info("✅ Model checkpoint validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Error validating checkpoint: {e}")
        return False

def main():
    """Main chat function with comprehensive error handling and user experience."""
    print("🚀 Initializing Character-level Transformer Chatbot...")
    
    # Load model and vocab
    try:
        checkpoint = load_model("Model.pth")
        
        if not validate_model_checkpoint(checkpoint):
            print("❌ Invalid model checkpoint. Please check your model file.")
            return 1
        
    except (FileNotFoundError, RuntimeError) as e:
        print(f"❌ Error: {e}")
        print("\n📝 Make sure you have:")
        print("   1. Trained a model using Train.py")
        print("   2. The model file 'Model.pth' exists in the current directory")
        print("   3. The model file is not corrupted")
        return 1
    
    # Extract model configuration
    char_to_ix = checkpoint["char_to_ix"]
    ix_to_char = checkpoint["ix_to_char"]
    
    if 'config' in checkpoint:
        config = checkpoint['config']
        hidden_size = config['hidden_size']
        num_layers = config['num_layers']
        nhead = config.get('nhead', 8)
        seq_length = config.get('seq_length', 512)
    else:
        # Legacy format
        hidden_size = checkpoint.get("hidden_size", 512)
        num_layers = checkpoint.get("num_layers", 6)
        nhead = checkpoint.get("nhead", 8)
        seq_length = checkpoint.get("seq_length", 512)
    
    vocab_size = len(char_to_ix)
    
    # Print model info
    print_model_info(checkpoint)
    
    # Initialize model
    try:
        model = CharTransformer(vocab_size, hidden_size, seq_length, num_layers, nhead).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info("Model initialized and weights loaded successfully")
    except Exception as e:
        print(f"❌ Error initializing model: {e}")
        return 1
    
    # Chat interface
    print("🤖 Character-level Transformer AI is ready!")
    print("\n" + "="*60)
    print("💬 CHAT COMMANDS:")
    print("="*60)
    print("  'exit' or 'quit'     - Exit the chat")
    print("  'clear'              - Clear conversation history")
    print("  'temp X'             - Set temperature (0.1-2.0, e.g., 'temp 0.8')")
    print("  'topk X'             - Set top-k value (1-50, e.g., 'topk 10')")
    print("  'nucleus' or 'topp'  - Switch to nucleus sampling")
    print("  'topk_mode'          - Switch to top-k sampling")
    print("  'greedy'             - Switch to greedy sampling")
    print("  'help'               - Show this help message")
    print("="*60)
    
    # Chat state
    conversation_history = ""
    temperature = 0.7
    sampling_method = "top_k"
    top_k = 8
    top_p = 0.9
    max_history_length = 2000  # Prevent context from getting too long
    
    # Display current settings
    print(f"⚙️  Current settings: temp={temperature}, method={sampling_method}", end="")
    if sampling_method == "top_k":
        print(f", k={top_k}")
    elif sampling_method in ["nucleus", "top_p"]:
        print(f", p={top_p}")
    else:
        print()
    print("-" * 60)
    
    while True:
        try:
            user_input = input("\n🧑 You: ").strip()
            
            if not user_input:
                continue
                
            # Handle commands
            if user_input.lower() in ["exit", "quit", "q"]:
                print("👋 Goodbye! Thanks for chatting!")
                break
                
            elif user_input.lower() == "clear":
                conversation_history = ""
                print("🗑️  Conversation history cleared!")
                continue
                
            elif user_input.lower() == "help":
                print("\n💬 Available commands:")
                print("  exit/quit - Exit chat")
                print("  clear - Clear history")
                print("  temp 0.8 - Set temperature")
                print("  topk 10 - Set top-k sampling")
                print("  nucleus - Use nucleus sampling")
                print("  greedy - Use greedy sampling")
                continue
                
            elif user_input.startswith("temp "):
                try:
                    new_temp = float(user_input.split()[1])
                    if 0.1 <= new_temp <= 2.0:
                        temperature = new_temp
                        print(f"🌡️  Temperature set to {temperature}")
                    else:
                        print("❌ Temperature must be between 0.1 and 2.0")
                except (IndexError, ValueError):
                    print("❌ Invalid temperature. Use: temp 0.8")
                continue
                
            elif user_input.startswith("topk "):
                try:
                    new_topk = int(user_input.split()[1])
                    if 1 <= new_topk <= 50:
                        top_k = new_topk
                        sampling_method = "top_k"
                        print(f"🔢 Top-k set to {top_k}")
                    else:
                        print("❌ Top-k must be between 1 and 50")
                except (IndexError, ValueError):
                    print("❌ Invalid top-k value. Use: topk 10")
                continue
                
            elif user_input.lower() in ["nucleus", "topp"]:
                sampling_method = "nucleus"
                print(f"🎯 Switched to nucleus sampling (p={top_p})")
                continue
                
            elif user_input.lower() == "topk_mode":
                sampling_method = "top_k"
                print(f"🔢 Switched to top-k sampling (k={top_k})")
                continue
                
            elif user_input.lower() == "greedy":
                sampling_method = "greedy"
                print("🎯 Switched to greedy sampling")
                continue
            
            # Build context with chat tokens
            conversation_history += f"<|user|>{user_input}<|bot|>"
            
            # Truncate history if too long
            if len(conversation_history) > max_history_length:
                # Keep the most recent parts of the conversation
                lines = conversation_history.split("<|user|>")
                conversation_history = "<|user|>" + "<|user|>".join(lines[-3:])
            
            # Generate response
            print("🤖 AI: ", end="", flush=True)
            
            try:
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
                if response and response.strip():
                    print(response)
                    # Add response to history
                    conversation_history += response + "\n"
                else:
                    print("(No response generated - try adjusting temperature or sampling method)")
                    
            except Exception as e:
                print(f"Error generating response: {e}")
                logger.error(f"Generation error: {e}")
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! (Interrupted by user)")
            break
        except EOFError:
            print("\n\n👋 Goodbye! (End of input)")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            logger.error(f"Unexpected error in main loop: {e}")
            print("Continuing chat...")
    
    # Cleanup
    try:
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        elif device.type == 'mps' and hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
        gc.collect()
    except Exception:
        pass
    
    return 0

if __name__ == "__main__":
    exit(main())