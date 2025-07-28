# Copyright (c) 2025 Matias Nielsen. All rights reserved.
# Licensed under the Custom License below.

import torch
import torch.nn.functional as F
import logging
import re
import gc
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Import shared components
from model_manager import ModelManager, ModelMetadata
from word_transformer import WordTransformer, WordTokenizer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_device():
    """Setup the best available device with proper error handling."""
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
        logger.warning(f"Error setting up device: {e}. Falling back to CPU.")
        return torch.device("cpu")

device = setup_device()

def nucleus_sampling(probs: torch.Tensor, p: float = 0.9) -> int:
    """
    Nucleus (top-p) sampling for better text generation.
    """
    if p <= 0 or p >= 1:
        return torch.multinomial(probs, 1).item()
    
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
        cutoff = max(1, cutoff)
    
    # Keep only top-p tokens
    top_p_probs = sorted_probs[:cutoff]
    top_p_indices = sorted_indices[:cutoff]
    
    # Renormalize probabilities
    if top_p_probs.sum() > 0:
        top_p_probs = top_p_probs / top_p_probs.sum()
    else:
        top_p_probs = torch.ones_like(top_p_probs) / len(top_p_probs)
    
    # Sample from the filtered distribution
    try:
        chosen_idx = torch.multinomial(top_p_probs, 1).item()
        return top_p_indices[chosen_idx].item()
    except RuntimeError:
        return top_p_indices[0].item()

def top_k_sampling(probs: torch.Tensor, k: int = 50) -> int:
    """Top-k sampling for controlled text generation."""
    if k <= 0 or k >= len(probs):
        return torch.multinomial(probs, 1).item()
    
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

def generate_response(model: WordTransformer, tokenizer: WordTokenizer, 
                     prompt: str, max_length: int = 150, temperature: float = 0.8,
                     sampling_method: str = "top_k", top_k: int = 50, top_p: float = 0.9) -> str:
    """
    Generate response using the word-level transformer with advanced sampling.
    """
    if not prompt.strip():
        return "I need some input to respond to."
    
    model.eval()
    
    try:
        with torch.no_grad():
            # Encode prompt
            input_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
            
            if input_ids.size(1) == 0:
                return "Unable to process input."
            
            generated = input_ids.clone()
            
            for step in range(max_length):
                # Use sliding window for long sequences
                max_seq_length = model.config.seq_length
                input_seq = generated[:, -max_seq_length:] if generated.size(1) > max_seq_length else generated
                
                # Forward pass
                logits = model(input_seq)
                next_token_logits = logits[0, -1, :] / max(temperature, 0.1)
                
                # Apply softmax to get probabilities
                probs = F.softmax(next_token_logits, dim=0)
                
                # Handle potential NaN/Inf values
                if torch.isnan(probs).any() or torch.isinf(probs).any():
                    logger.warning("NaN/Inf in probabilities, using uniform distribution")
                    probs = torch.ones_like(probs) / len(probs)
                
                # Choose sampling method
                if sampling_method == "nucleus" or sampling_method == "top_p":
                    next_token_id = nucleus_sampling(probs, p=top_p)
                elif sampling_method == "top_k":
                    next_token_id = top_k_sampling(probs, k=top_k)
                elif sampling_method == "greedy":
                    next_token_id = torch.argmax(probs).item()
                else:
                    next_token_id = top_k_sampling(probs, k=top_k)
                
                # Validate token ID
                if next_token_id < 0 or next_token_id >= tokenizer.vocab_size():
                    logger.warning(f"Invalid token ID {next_token_id}, stopping generation")
                    break
                
                # Add to generated sequence
                generated = torch.cat([generated, torch.tensor([[next_token_id]], device=device)], dim=1)
                
                # Stop conditions
                if next_token_id == tokenizer.vocab.get("</s>", -1):
                    break
                
                # Check for natural stopping points
                current_token = tokenizer.id_to_token.get(next_token_id, "")
                if current_token in [".", "!", "?"] and step > 10:
                    # Look ahead a bit to see if this is a good stopping point
                    if step > 20:
                        break
            
            # Decode generated text
            response_ids = generated[0][input_ids.size(1):].tolist()
            response = tokenizer.decode(response_ids)
            
            # Clean up response
            response = clean_response(response)
            
            return response if response.strip() else "I'm not sure how to respond to that."
    
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return "Sorry, I encountered an error while generating a response."
    
    finally:
        # Memory cleanup
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        elif device.type == 'mps' and hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
        gc.collect()

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
                # Remove incomplete last sentence
                response_parts = response.rsplit(last_sentence, 1)
                if len(response_parts) > 1:
                    response = response_parts[0].strip()
                    # Ensure proper ending punctuation
                    if response and response[-1] not in '.!?':
                        response += '.'
    
    return response

class WordAIChat:
    """Professional chat interface for the word-level AI."""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.model = None
        self.tokenizer = None
        self.metadata = None
        self.conversation_history = []
        self.max_history_length = 2000  # Token limit for conversation history
    
    def list_available_models(self) -> List[Dict]:
        """List all available models."""
        return self.model_manager.list_models()
    
    def load_model(self, model_id: str) -> bool:
        """Load a specific model for chatting."""
        try:
            self.model, self.tokenizer, self.metadata = self.model_manager.load_model(model_id)
            self.model.eval()
            logger.info(f"✅ Chat ready with model: {self.metadata.model_name} {self.metadata.version}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            return False
    
    def print_model_info(self):
        """Print comprehensive model information."""
        if not self.metadata:
            print("❌ No model loaded.")
            return
        
        print(f"\n{'='*70}")
        print(f"🤖 {self.metadata.model_name} {self.metadata.version}")
        print(f"{'='*70}")
        print(f"📊 Architecture:")
        print(f"   • Parameters: {self.metadata.total_parameters:,}")
        print(f"   • Hidden Size: {self.metadata.model_config.hidden_size}")
        print(f"   • Layers: {self.metadata.model_config.num_layers}")
        print(f"   • Attention Heads: {self.metadata.model_config.num_heads}")
        print(f"   • Sequence Length: {self.metadata.model_config.seq_length}")
        print(f"   • Vocabulary Size: {self.metadata.model_config.vocab_size:,}")
        
        print(f"\n🎯 Performance:")
        print(f"   • Best Loss: {self.metadata.best_loss:.4f}")
        print(f"   • Best Perplexity: {self.metadata.best_perplexity:.2f}")
        if 'accuracy' in self.metadata.performance_metrics:
            print(f"   • Accuracy: {self.metadata.performance_metrics['accuracy']*100:.2f}%")
        
        print(f"\n💾 Details:")
        print(f"   • Size: {self.metadata.model_size_mb:.2f} MB")
        print(f"   • Training Time: {self.metadata.training_time_hours:.2f} hours")
        print(f"   • Epochs Trained: {self.metadata.epochs_trained}")
        print(f"   • Hardware: {self.metadata.hardware_used}")
        
        if self.metadata.tags:
            print(f"   • Tags: {', '.join(self.metadata.tags)}")
        
        print(f"{'='*70}\n")
    
    def manage_conversation_history(self, user_input: str, ai_response: str):
        """Manage conversation history with token limits."""
        # Add to history
        self.conversation_history.append(f"<user> {user_input}")
        self.conversation_history.append(f"<bot> {ai_response}")
        
        # Trim history if too long
        if len(self.conversation_history) > 10:  # Keep last 5 exchanges
            self.conversation_history = self.conversation_history[-10:]
    
    def build_context(self, user_input: str) -> str:
        """Build conversation context for the model."""
        # Create context from recent history + current input
        context_parts = self.conversation_history[-6:] if self.conversation_history else []
        context_parts.append(f"<user> {user_input}")
        context_parts.append("<bot>")
        
        return " ".join(context_parts)
    
    def chat(self):
        """Interactive chat interface with advanced features."""
        if not self.model:
            # Auto-load the best available model
            models = self.list_available_models()
            if not models:
                print("❌ No models available. Please train a model first using Train.py")
                return
            
            print("🔍 Available models:")
            for i, model in enumerate(models[:5]):  # Show top 5
                print(f"   {i+1}. {model['name']} {model['version']} "
                      f"(Loss: {model['best_loss']:.4f}, Size: {model['size_mb']:.2f}MB)")
            
            try:
                choice = input(f"\nSelect model (1-{min(5, len(models))}) or press Enter for best: ").strip()
                if choice:
                    model_idx = int(choice) - 1
                    if 0 <= model_idx < len(models):
                        selected_model = models[model_idx]
                    else:
                        print("Invalid selection, using best model.")
                        selected_model = min(models, key=lambda x: x['best_loss'])
                else:
                    selected_model = min(models, key=lambda x: x['best_loss'])
                
                if not self.load_model(selected_model['id']):
                    print("❌ Failed to load model.")
                    return
                    
            except (ValueError, KeyboardInterrupt):
                print("❌ Invalid selection or interrupted.")
                return
        
        # Print model information
        self.print_model_info()
        
        print("💬 Word-Level AI Chat Interface")
        print("=" * 70)
        print("🔧 Commands:")
        print("   • 'exit', 'quit', 'q' - Exit chat")
        print("   • 'clear' - Clear conversation history")
        print("   • 'info' - Show model information")
        print("   • 'models' - List available models")
        print("   • 'temp X' - Set temperature (0.1-2.0)")
        print("   • 'topk X' - Set top-k value (1-100)")
        print("   • 'nucleus' or 'topp' - Switch to nucleus sampling")
        print("   • 'topk_mode' - Switch to top-k sampling")
        print("   • 'greedy' - Switch to greedy sampling")
        print("   • 'help' - Show this help")
        print("=" * 70)
        
        # Chat settings
        temperature = 0.8
        sampling_method = "top_k"
        top_k = 50
        top_p = 0.9
        
        print(f"⚙️  Settings: temp={temperature}, method={sampling_method}", end="")
        if sampling_method == "top_k":
            print(f", k={top_k}")
        elif sampling_method in ["nucleus", "top_p"]:
            print(f", p={top_p}")
        else:
            print()
        print("-" * 70)
        
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
                    self.conversation_history = []
                    print("🗑️  Conversation history cleared!")
                    continue
                
                elif user_input.lower() == "info":
                    self.print_model_info()
                    continue
                
                elif user_input.lower() == "models":
                    print("\n📋 Available models:")
                    for model in self.list_available_models():
                        print(f"   {model['id']}: {model['name']} {model['version']} "
                              f"(Loss: {model['best_loss']:.4f})")
                    continue
                
                elif user_input.lower() == "help":
                    print("\n💬 Available commands:")
                    print("  exit/quit - Exit chat")
                    print("  clear - Clear conversation history")
                    print("  info - Show model information")
                    print("  models - List available models")
                    print("  temp 0.8 - Set temperature")
                    print("  topk 50 - Set top-k sampling")
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
                        if 1 <= new_topk <= 100:
                            top_k = new_topk
                            sampling_method = "top_k"
                            print(f"🔢 Top-k set to {top_k}")
                        else:
                            print("❌ Top-k must be between 1 and 100")
                    except (IndexError, ValueError):
                        print("❌ Invalid top-k value. Use: topk 50")
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
                
                # Generate response
                print("🤖 AI: ", end="", flush=True)
                
                try:
                    # Build context from conversation history
                    context = self.build_context(user_input)
                    
                    # Generate response
                    response = generate_response(
                        self.model, self.tokenizer, context,
                        max_length=200,
                        temperature=temperature,
                        sampling_method=sampling_method,
                        top_k=top_k,
                        top_p=top_p
                    )
                    
                    if response and response.strip():
                        print(response)
                        # Add to conversation history
                        self.manage_conversation_history(user_input, response)
                    else:
                        print("(No response generated - try adjusting settings)")
                
                except Exception as e:
                    print(f"Error generating response: {e}")
                    logger.error(f"Generation error: {e}")
            
            except KeyboardInterrupt:
                print("\n\n👋 Chat interrupted!")
                break
            except EOFError:
                print("\n\n👋 End of input!")
                break
            except Exception as e:
                print(f"\n❌ Unexpected error: {e}")
                logger.error(f"Unexpected error in chat loop: {e}")
                continue
        
        # Cleanup
        try:
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            elif device.type == 'mps' and hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
            gc.collect()
        except Exception:
            pass

def print_welcome():
    """Print welcome message and system info."""
    print("\n" + "="*70)
    print("🤖 WORD-LEVEL AI CHAT SYSTEM")
    print("="*70)
    print(f"🔧 Device: {device}")
    print(f"🐍 PyTorch: {torch.__version__}")
    if device.type == 'cuda':
        print(f"🎮 CUDA: {torch.version.cuda}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print("="*70)

def main():
    """Main chat function with model management."""
    print_welcome()
    
    # Initialize model manager
    model_manager = ModelManager("models")
    
    # Check for available models
    models = model_manager.list_models()
    if not models:
        print("❌ No trained models found!")
        print("\n📝 To get started:")
        print("   1. Run 'python Train.py' to train a model")
        print("   2. Or run 'python fine_tune.py' to fine-tune an existing model")
        print("   3. Then run 'python ChatAI.py' to start chatting")
        return 1
    
    print(f"✅ Found {len(models)} trained model(s)")
    
    # Initialize chat interface
    chat = WordAIChat(model_manager)
    
    try:
        chat.chat()
        return 0
    except Exception as e:
        logger.error(f"Chat system error: {e}")
        print(f"❌ Chat system error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())