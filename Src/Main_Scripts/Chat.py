#!/usr/bin/env python3
"""
Complete Interactive Chat Interface for DeepSeek Transformer
Fixed to properly load multi-shard ZeRO checkpoints
"""

import os
import sys
import logging
import argparse
import signal
import json
import time
import re
import glob
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.cuda import get_device_properties

from config.config_manager import ConfigPresets

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from core.model import DeepSeekTransformer, DeepSeekConfig
    from core.tokenizer import ConversationTokenizer
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure model.py and tokenizer.py are in the core/ directory")
    sys.exit(1)

# Alternative import fallback
try:
    import tiktoken
except ImportError:
    print("tiktoken not found, using basic tokenizer fallback")
    tiktoken = None


class SimpleTokenizer:
    """Fallback tokenizer if tiktoken is not available"""
    
    def __init__(self):
        # Basic vocabulary - just for fallback
        self.vocab = {chr(i): i for i in range(32, 127)}  # Printable ASCII
        self.vocab.update({f"<|special_{i}|>": 127 + i for i in range(100)})
        self.vocab_size = len(self.vocab)
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
    
    def encode(self, text: str) -> List[int]:
        """Simple character-based encoding"""
        return [self.vocab.get(c, 0) for c in text[:1000]]  # Limit length
    
    def decode(self, tokens: List[int]) -> str:
        """Simple decoding"""
        return ''.join(self.reverse_vocab.get(t, '?') for t in tokens)


@dataclass
class ConversationStats:
    """Session statistics tracking"""
    messages_sent: int = 0
    messages_received: int = 0
    session_start: datetime = None
    total_tokens_generated: int = 0
    total_input_tokens: int = 0
    avg_response_time: float = 0.0
    generation_attempts: int = 0
    failed_generations: int = 0

    def __post_init__(self):
        if self.session_start is None:
            self.session_start = datetime.now()


def load_zero_checkpoint_shards(checkpoint_dir: str) -> Dict[str, torch.Tensor]:
    """Load and merge all ZeRO checkpoint shards from a directory"""
    checkpoint_dir = Path(checkpoint_dir)
    
    if checkpoint_dir.is_file():
        # Single checkpoint file
        checkpoint_files = [checkpoint_dir]
    else:
        # Directory with multiple shards
        patterns = [
            "*model_states.pt",
            "*mp_rank_*.pt", 
            "zero_pp_rank_*_mp_rank_*.pt",
            "pytorch_model*.pt"
        ]
        
        checkpoint_files = []
        for pattern in patterns:
            files = list(checkpoint_dir.glob(pattern))
            if files:
                checkpoint_files.extend(files)
                break
        
        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")
    
    print(f"Loading {len(checkpoint_files)} checkpoint shard(s)...")
    for f in checkpoint_files:
        print(f"  - {f.name}")
    
    # Load and merge all shards
    merged_state_dict = {}
    
    for i, shard_file in enumerate(sorted(checkpoint_files)):
        print(f"Loading shard {i+1}/{len(checkpoint_files)}: {shard_file.name}")
        
        try:
            # Load shard
            shard_data = torch.load(shard_file, map_location='cpu', weights_only=False)
            
            # Extract state dict from various possible formats
            if isinstance(shard_data, dict):
                if 'module' in shard_data:
                    shard_state = shard_data['module']
                elif 'model_state_dict' in shard_data:
                    shard_state = shard_data['model_state_dict']
                elif 'state_dict' in shard_data:
                    shard_state = shard_data['state_dict']
                else:
                    shard_state = shard_data
            else:
                shard_state = shard_data
            
            # Merge parameters
            for key, tensor in shard_state.items():
                if key in merged_state_dict:
                    # Handle parameter sharding/concatenation if needed
                    existing_tensor = merged_state_dict[key]
                    if existing_tensor.shape != tensor.shape:
                        print(f"Warning: Shape mismatch for {key}: {existing_tensor.shape} vs {tensor.shape}")
                        # Keep the newer tensor
                        merged_state_dict[key] = tensor
                    else:
                        # Use the newer tensor (last shard wins for identical keys)
                        merged_state_dict[key] = tensor
                else:
                    merged_state_dict[key] = tensor
            
        except Exception as e:
            print(f"Warning: Failed to load shard {shard_file}: {e}")
            continue
    
    if not merged_state_dict:
        raise RuntimeError("Failed to load any checkpoint data")
    
    print(f"Successfully merged {len(merged_state_dict)} parameters")
    return merged_state_dict


def load_config_from_checkpoint(checkpoint_path: str) -> DeepSeekConfig:
    """Load configuration from checkpoint file with comprehensive inference"""
    try:
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"Loading configuration from checkpoint: {checkpoint_path}")
        
        # Load merged checkpoint state
        if checkpoint_path.is_dir():
            merged_state_dict = load_zero_checkpoint_shards(checkpoint_path)
            checkpoint = {'merged_state': merged_state_dict}
        else:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Try to extract existing config first
        config_dict = None
        
        if isinstance(checkpoint, dict):
            for key in ['config', 'model_config', 'hyperparameters']:
                if key in checkpoint:
                    config_dict = checkpoint[key]
                    break
            
            if config_dict is None:
                print("No explicit config found, inferring from model structure...")
                state_dict = checkpoint.get('merged_state', checkpoint)
                config_dict = infer_config_from_checkpoint(state_dict)
        else:
            config_dict = infer_config_from_checkpoint(checkpoint)
        
        if isinstance(config_dict, dict):
            config = DeepSeekConfig(**config_dict)
        else:
            raise ValueError(f"Unsupported config format: {type(config_dict)}")
        
        print(f"Loaded config - Hidden: {config.hidden_size}, Layers: {config.num_layers}, Vocab: {config.vocab_size}")
        return config
        
    except Exception as e:
        print(f"Failed to load config: {e}")
        print("Using fallback configuration...")
        
        # Safe fallback configuration
        return DeepSeekConfig(
            hidden_size=128,
            num_layers=12,
            num_heads=8,
            num_kv_heads=4,
            seq_length=1024,
            vocab_size=50000,
            intermediate_size=512,
            rms_norm_eps=1e-6,
            rope_base=10000,
            rope_scaling=None,
            attention_dropout=0.0,
            hidden_dropout=0.0,
        )


def infer_config_from_checkpoint(checkpoint) -> Dict[str, Any]:
    """Infer model configuration from checkpoint structure"""
    
    # Extract state dict
    if isinstance(checkpoint, dict):
        if 'module' in checkpoint:
            state_dict = checkpoint['module']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'merged_state' in checkpoint:
            state_dict = checkpoint['merged_state']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    config_params = {}
    
    print(f"Analyzing {len(state_dict)} parameters...")
    
    # Find vocabulary size and hidden size from embeddings/lm_head
    for key, tensor in state_dict.items():
        if 'lm_head.weight' in key and tensor.dim() == 2:
            config_params['vocab_size'] = tensor.shape[0]
            config_params['hidden_size'] = tensor.shape[1]
            print(f"From lm_head: vocab_size={tensor.shape[0]}, hidden_size={tensor.shape[1]}")
            break
        elif 'embed_tokens.weight' in key and tensor.dim() == 2:
            config_params['vocab_size'] = tensor.shape[0]
            config_params['hidden_size'] = tensor.shape[1]
            print(f"From embeddings: vocab_size={tensor.shape[0]}, hidden_size={tensor.shape[1]}")
    
    # Count transformer layers
    layer_count = 0
    for key in state_dict.keys():
        if 'layers.' in key:
            parts = key.split('.')
            for i, part in enumerate(parts):
                if part == 'layers' and i + 1 < len(parts) and parts[i + 1].isdigit():
                    layer_num = int(parts[i + 1])
                    layer_count = max(layer_count, layer_num + 1)
    
    if layer_count > 0:
        config_params['num_layers'] = layer_count
        print(f"Found {layer_count} transformer layers")
    
    # Infer attention configuration
    hidden_size = config_params.get('hidden_size', 128)
    
    # Find attention heads from query projection
    for key, tensor in state_dict.items():
        if 'self_attn.q_proj.weight' in key and tensor.dim() == 2:
            q_out_features = tensor.shape[0]
            
            # Standard attention: q_proj output equals hidden_size
            if q_out_features == hidden_size:
                # Try common head counts
                for heads in [8, 16, 12, 32, 4, 6]:
                    if hidden_size % heads == 0:
                        config_params['num_heads'] = heads
                        print(f"Inferred num_heads: {heads} (head_dim: {hidden_size // heads})")
                        break
            break
    
    # Find KV heads for GQA
    for key, tensor in state_dict.items():
        if 'self_attn.k_proj.weight' in key and tensor.dim() == 2:
            k_out_features = tensor.shape[0]
            num_heads = config_params.get('num_heads', 8)
            head_dim = hidden_size // num_heads
            
            config_params['num_kv_heads'] = max(1, k_out_features // head_dim)
            print(f"Inferred num_kv_heads: {config_params['num_kv_heads']}")
            break
    
    # Find MLP intermediate size
    for key, tensor in state_dict.items():
        if ('mlp.gate_proj.weight' in key or 'mlp.up_proj.weight' in key) and tensor.dim() == 2:
            config_params['intermediate_size'] = tensor.shape[0]
            print(f"Found MLP intermediate size: {tensor.shape[0]}")
            break
    
    # Apply sensible defaults
    defaults = {
        'hidden_size': 128,
        'num_layers': 12,
        'num_heads': 8,
        'num_kv_heads': 4,
        'seq_length': 1024,
        'vocab_size': 50000,
        'intermediate_size': None,  # Will be calculated
        'rms_norm_eps': 1e-6,
        'rope_base': 10000,
        'rope_scaling': None,
        'attention_dropout': 0.0,
        'hidden_dropout': 0.0,
    }
    
    for key, default_value in defaults.items():
        if key not in config_params:
            config_params[key] = default_value
    
    # Calculate intermediate_size if not found
    if config_params['intermediate_size'] is None:
        config_params['intermediate_size'] = 4 * config_params['hidden_size']
    
    # Ensure num_kv_heads is reasonable
    if config_params['num_kv_heads'] <= 0:
        config_params['num_kv_heads'] = config_params['num_heads']
    
    print(f"Final inferred config: {config_params}")
    return config_params


def find_best_checkpoint() -> Optional[str]:
    """Find the best available checkpoint automatically"""
    print("Searching for checkpoints...")
    
    search_locations = [
        Path("checkpoints"),
        Path("experiments"),
        Path("models"),
        Path("."),
    ]
    
    best_checkpoint = None
    best_time = 0
    
    for location in search_locations:
        if not location.exists():
            continue
            
        # Look for checkpoint directories first (ZeRO shards)
        for subdir in location.iterdir():
            if subdir.is_dir() and any(subdir.glob("*model_states.pt")):
                try:
                    mtime = subdir.stat().st_mtime
                    if mtime > best_time:
                        best_checkpoint = str(subdir)
                        best_time = mtime
                        print(f"Found checkpoint directory: {best_checkpoint}")
                except:
                    continue
        
        # Look for individual checkpoint files (prioritize model states over optimizer states)
        patterns = ["*model_states.pt", "best_*.pt", "*_best.pt", "checkpoint_*.pt", "model_*.pt", "*.pt", "*.pth"]
        
        for pattern in patterns:
            for checkpoint in location.glob(pattern):
                if checkpoint.is_file():
                    try:
                        mtime = checkpoint.stat().st_mtime
                        if mtime > best_time:
                            best_checkpoint = str(checkpoint)
                            best_time = mtime
                            print(f"Found checkpoint: {best_checkpoint}")
                    except:
                        continue
    
    if best_checkpoint:
        print(f"Using checkpoint: {best_checkpoint}")
    else:
        print("ERROR: No checkpoints found!")
        print("Please train a model first or specify a checkpoint with --checkpoint")
    
    return best_checkpoint


class ChatInterface:
    """Main chat interface with proper ZeRO checkpoint loading"""
    
    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path
        
        # Load config from checkpoint
        self.config = load_config_from_checkpoint(checkpoint_path)
        
        # Initialize tracking
        self.conversation_history = []
        self.stats = ConversationStats()

        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Chat settings
        self.max_history_length = 15
        self.show_token_count = False
        self.show_timing = False
        self.system_prompt = None

        # Generation settings
        self.generation_configs = {
            'standard': {'temperature': 0.8, 'top_p': 0.9, 'top_k': 50},
            'creative': {'temperature': 1.0, 'top_p': 0.95, 'top_k': 100},
            'analytical': {'temperature': 0.3, 'top_p': 0.7, 'top_k': 20},
            'precise': {'temperature': 0.1, 'top_p': 0.5, 'top_k': 10}
        }
        self.conversation_mode = "standard"

        # Initialize components
        self.tokenizer = None
        self.model = None

        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize tokenizer and model"""
        print("Initializing chat system...")
        
        try:
            # Initialize tokenizer
            if tiktoken is not None:
                print(f"Using tiktoken tokenizer for vocab size: {self.config.vocab_size}")
                try:
                    # Try to use appropriate encoding based on vocab size
                    if self.config.vocab_size > 100000:
                        encoding_name = "cl100k_base"  # GPT-4 style
                    elif self.config.vocab_size > 50000:
                        encoding_name = "gpt2"
                    else:
                        encoding_name = "gpt2"
                    
                    self.tokenizer = tiktoken.get_encoding(encoding_name)
                    print(f"Loaded {encoding_name} tokenizer")
                except:
                    print("Using simple fallback tokenizer")
                    self.tokenizer = SimpleTokenizer()
            else:
                print("Using simple fallback tokenizer")
                self.tokenizer = SimpleTokenizer()
            
            # Initialize model
            self.model = DeepSeekTransformer(self.config).to(self.device)
            param_count = sum(p.numel() for p in self.model.parameters())
            print(f"Model initialized ({param_count:,} parameters)")
            
            # Load checkpoint
            self._load_checkpoint()
            
            # Set to eval mode
            self.model.eval()
            
            print("Chat system ready!")
            
        except Exception as e:
            print(f"Failed to initialize chat system: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _load_checkpoint(self):
        """Load model weights from checkpoint with ZeRO support"""
        try:
            print(f"Loading checkpoint: {self.checkpoint_path}")
            
            checkpoint_path = Path(self.checkpoint_path)
            
            if checkpoint_path.is_dir():
                # Handle ZeRO checkpoint directory
                print("Detected ZeRO checkpoint directory")
                state_dict = load_zero_checkpoint_shards(checkpoint_path)
            else:
                # Handle single checkpoint file
                print("Loading single checkpoint file")
                if self.device.type == 'cuda':
                    checkpoint = torch.load(checkpoint_path, map_location='cuda', weights_only=False)
                else:
                    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                
                # Extract model state dict
                if isinstance(checkpoint, dict):
                    if 'module' in checkpoint:
                        state_dict = checkpoint['module']
                    elif 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                    elif 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    else:
                        state_dict = checkpoint
                else:
                    state_dict = checkpoint
            
            # Filter and load state dict
            model_keys = set(self.model.state_dict().keys())
            filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_keys}
            
            missing_keys, unexpected_keys = self.model.load_state_dict(filtered_state_dict, strict=False)
            
            if missing_keys:
                print(f"Warning: {len(missing_keys)} missing keys")
                for key in missing_keys[:5]:  # Show first 5
                    print(f"  Missing: {key}")
            if unexpected_keys:
                print(f"Warning: {len(unexpected_keys)} unexpected keys")
                for key in unexpected_keys[:5]:  # Show first 5
                    print(f"  Unexpected: {key}")
            
            # Check if we actually loaded meaningful weights
            total_params = sum(p.numel() for p in self.model.parameters())
            loaded_params = sum(p.numel() for name, p in self.model.named_parameters() if name in filtered_state_dict)
            
            print(f"Loaded {loaded_params:,}/{total_params:,} parameters ({100*loaded_params/total_params:.1f}%)")
            
            if loaded_params == 0:
                print("WARNING: No parameters were loaded! Model will use random weights.")
                print("This will result in gibberish output and token ID warnings.")
            else:
                print("Checkpoint loaded successfully")
            
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            print("Model will use random weights")
            import traceback
            traceback.print_exc()
    
    def _generate_response(self, user_input: str) -> Tuple[str, Dict[str, Any]]:
        """Generate response to user input"""
        start_time = time.time()
        
        try:
            # Add user message to history
            self.conversation_history.append({'role': 'user', 'content': user_input})
            
            # Prepare input for model
            conversation_text = self._format_conversation()
            
            # Tokenize
            try:
                if hasattr(self.tokenizer, 'encode'):
                    input_tokens = self.tokenizer.encode(conversation_text)
                else:
                    input_tokens = [1, 2, 3]  # Fallback dummy tokens
            except:
                input_tokens = [1, 2, 3]  # Fallback
            
            if not input_tokens:
                return "I couldn't process your message.", {}
            
            # Convert to tensor
            input_ids = torch.tensor([input_tokens], dtype=torch.long, device=self.device)
            
            # Limit input length
            max_input_length = self.config.seq_length - 100  # Reserve space for output
            if input_ids.shape[1] > max_input_length:
                input_ids = input_ids[:, -max_input_length:]
            
            # Generate response
            gen_config = self.generation_configs[self.conversation_mode]
            generated_tokens = self._generate_tokens(
                input_ids,
                max_new_tokens=100,
                temperature=gen_config['temperature'],
                top_p=gen_config['top_p'],
                top_k=gen_config['top_k']
            )
            
            # Decode response
            if generated_tokens:
                try:
                    if hasattr(self.tokenizer, 'decode'):
                        response = self.tokenizer.decode(generated_tokens)
                    else:
                        response = ''.join(chr(max(32, min(126, t))) for t in generated_tokens[:50])
                    response = response.strip()
                except:
                    response = "Generated response (decode error)"
            else:
                response = "I couldn't generate a response."
            
            # Calculate metrics
            response_time = time.time() - start_time
            metrics = {
                'response_time': response_time,
                'input_tokens': len(input_tokens),
                'output_tokens': len(generated_tokens),
                'tokens_per_second': len(generated_tokens) / response_time if response_time > 0 else 0
            }
            
            # Update conversation history
            self.conversation_history.append({'role': 'assistant', 'content': response})
            
            # Update stats
            self._update_stats(metrics)
            
            return response, metrics
            
        except Exception as e:
            logging.error(f"Generation failed: {e}")
            return f"Error generating response: {str(e)}", {}
    
    def _generate_tokens(self, input_ids: torch.Tensor, max_new_tokens: int,
                        temperature: float, top_p: float, top_k: int) -> List[int]:
        """Generate tokens using the model"""
        self.model.eval()
        generated = []
        
        with torch.no_grad():
            current_ids = input_ids.clone()
            
            for step in range(max_new_tokens):
                try:
                    # Forward pass
                    logits = self.model(current_ids)
                    if isinstance(logits, tuple):
                        logits = logits[0]
                    
                    # Get next token logits
                    next_token_logits = logits[0, -1, :].clone()
                    
                    # Apply temperature
                    if temperature != 1.0:
                        next_token_logits = next_token_logits / max(temperature, 0.1)
                    
                    # Apply top-k filtering
                    if top_k > 0:
                        top_k_actual = min(top_k, next_token_logits.size(-1))
                        values, _ = torch.topk(next_token_logits, top_k_actual)
                        next_token_logits[next_token_logits < values[-1]] = -float('inf')
                    
                    # Apply top-p filtering
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                        sorted_indices_to_remove[0] = 0
                        
                        indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
                        next_token_logits[indices_to_remove] = -float('inf')
                    
                    # Sample next token
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).item()
                    
                    # Basic stopping criteria
                    if next_token == 0 or (hasattr(self.tokenizer, 'vocab_size') and next_token >= self.tokenizer.vocab_size):
                        break
                    
                    generated.append(next_token)
                    
                    # Update input for next iteration
                    current_ids = torch.cat([current_ids, torch.tensor([[next_token]], device=self.device)], dim=1)
                    
                    # Check context length
                    if current_ids.shape[1] >= self.config.seq_length:
                        break
                        
                except Exception as e:
                    logging.error(f"Token generation failed at step {step}: {e}")
                    break
        
        return generated
    
    def _format_conversation(self) -> str:
        """Format conversation history for model input"""
        parts = []
        
        if self.system_prompt:
            parts.append(f"System: {self.system_prompt}")
        
        # Include recent conversation history
        history = self.conversation_history[-self.max_history_length:]
        
        for msg in history:
            role = msg['role'].capitalize()
            content = msg['content']
            parts.append(f"{role}: {content}")
        
        return '\n'.join(parts) + '\nAssistant:'
    
    def _update_stats(self, metrics: Dict[str, Any]):
        """Update session statistics"""
        self.stats.messages_sent += 1
        self.stats.messages_received += 1
        self.stats.total_tokens_generated += metrics.get('output_tokens', 0)
        self.stats.total_input_tokens += metrics.get('input_tokens', 0)
        self.stats.generation_attempts += 1
        
        # Update average response time
        count = self.stats.messages_received
        self.stats.avg_response_time = ((self.stats.avg_response_time * (count - 1)) + metrics.get('response_time', 0)) / count
    
    def _handle_command(self, command: str) -> bool:
        """Handle special commands"""
        parts = command.strip().split(None, 1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        
        if cmd in ['/quit', '/exit']:
            return False
        elif cmd == '/help':
            self._show_help()
        elif cmd == '/stats':
            self._show_stats()
        elif cmd == '/clear':
            self._clear_history()
        elif cmd == '/mode':
            self._set_mode(args)
        elif cmd == '/system':
            self._set_system_prompt(args)
        elif cmd == '/save':
            self._save_conversation(args)
        else:
            print(f"Unknown command: {cmd}. Type /help for available commands.")
        
        return True
    
    def _show_help(self):
        """Show help information"""
        print("\nAvailable Commands:")
        print("  /help          - Show this help")
        print("  /quit, /exit   - Exit chat")
        print("  /clear         - Clear conversation history")
        print("  /stats         - Show session statistics")
        print("  /mode <mode>   - Set conversation mode (standard/creative/analytical/precise)")
        print("  /system <text> - Set system prompt")
        print("  /save [name]   - Save conversation")
        print()
    
    def _show_stats(self):
        """Show session statistics"""
        duration = datetime.now() - self.stats.session_start
        print(f"\nSession Statistics:")
        print(f"Duration: {duration}")
        print(f"Messages exchanged: {self.stats.messages_sent}")
        print(f"Total tokens generated: {self.stats.total_tokens_generated:,}")
        print(f"Average response time: {self.stats.avg_response_time:.2f}s")
        print(f"Current mode: {self.conversation_mode}")
        print()
    
    def _clear_history(self):
        """Clear conversation history"""
        self.conversation_history.clear()
        print("Conversation history cleared.")
    
    def _set_mode(self, mode: str):
        """Set conversation mode"""
        mode = mode.strip().lower()
        if mode in self.generation_configs:
            self.conversation_mode = mode
            config = self.generation_configs[mode]
            print(f"Mode set to: {mode}")
            print(f"  Temperature: {config['temperature']}")
            print(f"  Top-p: {config['top_p']}")
            print(f"  Top-k: {config['top_k']}")
        else:
            print(f"Invalid mode. Available: {', '.join(self.generation_configs.keys())}")
    
    def _set_system_prompt(self, prompt: str):
        """Set system prompt"""
        if prompt.strip():
            self.system_prompt = prompt.strip()
            print(f"System prompt set: {prompt[:50]}...")
        else:
            self.system_prompt = None
            print("System prompt cleared")
    
    def _save_conversation(self, name: str = ""):
        """Save conversation to file"""
        if not self.conversation_history:
            print("No conversation to save")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if not name.strip():
            name = f"chat_{timestamp}"
        
        try:
            Path("conversations").mkdir(exist_ok=True)
            filepath = Path("conversations") / f"{name}.json"
            
            data = {
                'name': name,
                'timestamp': timestamp,
                'conversation': self.conversation_history,
                'mode': self.conversation_mode,
                'system_prompt': self.system_prompt
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            print(f"Conversation saved: {filepath}")
        except Exception as e:
            print(f"Failed to save conversation: {e}")
    
    def _print_header(self):
        """Print chat header"""
        print("\n" + "="*60)
        print("DEEPSEEK TRANSFORMER CHAT INTERFACE")
        print("="*60)
        print(f"Model: DeepSeek Transformer")
        print(f"Config: {self.config.hidden_size}d, {self.config.num_layers} layers, {self.config.num_heads} heads")
        print(f"Vocab: {self.config.vocab_size:,} tokens")
        print(f"Device: {self.device}")
        print(f"Mode: {self.conversation_mode}")
        print(f"Checkpoint: {self.checkpoint_path}")
        print("\nCommands: /help /stats /clear /mode /system /save /quit")
        print("="*60)
        print("Start chatting! (Type your message and press Enter)\n")
    
    def run_chat(self):
        """Run the interactive chat loop"""
        self._print_header()
        
        # Setup graceful exit
        def signal_handler(signum, frame):
            print("\n\nChat session ended")
            self._show_stats()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        
        try:
            while True:
                try:
                    user_input = input(f"[{self.conversation_mode[0].upper()}] You: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\n\nChat session ended")
                    break
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    if not self._handle_command(user_input):
                        break
                    continue
                
                # Generate response
                print("Assistant: ", end="", flush=True)
                
                response, metrics = self._generate_response(user_input)
                print(response)
                
                # Show metrics if enabled
                if self.show_timing or self.show_token_count:
                    info_parts = []
                    if self.show_timing:
                        info_parts.append(f"Time: {metrics.get('response_time', 0):.2f}s")
                    if self.show_token_count:
                        info_parts.append(f"Tokens: {metrics.get('output_tokens', 0)}")
                    if info_parts:
                        print(f"   [{' | '.join(info_parts)}]")
                
                print()  # Extra line for readability
        
        except Exception as e:
            print(f"\nChat error: {e}")
        
        finally:
            print("\nFinal Statistics:")
            self._show_stats()
            print("Goodbye!")


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description='Interactive Chat Interface for DeepSeek Transformer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python chat.py                                    # Auto-detect checkpoint
  python chat.py --checkpoint model.pt             # Specific checkpoint
  python chat.py --checkpoint ./checkpoints/       # ZeRO checkpoint directory
  python chat.py --mode creative --show-timing     # Creative mode with timing
        """
    )
    
    parser.add_argument('--checkpoint', type=str, help='Checkpoint file or directory to load')
    parser.add_argument('--mode', choices=['standard', 'creative', 'analytical', 'precise'],
                       default='standard', help='Conversation mode')
    parser.add_argument('--system-prompt', type=str, help='Initial system prompt')
    parser.add_argument('--show-tokens', action='store_true', help='Show token counts')
    parser.add_argument('--show-timing', action='store_true', help='Show response timing')
    parser.add_argument('--max-history', type=int, default=15, help='Max conversation history')
    
    args = parser.parse_args()
    
    # Find checkpoint
    checkpoint_path = args.checkpoint
    if not checkpoint_path:
        checkpoint_path = find_best_checkpoint()
        if not checkpoint_path:
            return 1
    
    try:
        # Initialize chat
        print("Initializing chat interface...")
        chat = ChatInterface(checkpoint_path)
        
        # Apply settings
        chat.conversation_mode = args.mode
        chat.show_token_count = args.show_tokens
        chat.show_timing = args.show_timing
        chat.max_history_length = args.max_history
        if args.system_prompt:
            chat.system_prompt = args.system_prompt
        
        # Run chat
        chat.run_chat()
        return 0
        
    except KeyboardInterrupt:
        print("\nChat interrupted by user")
        return 0
    except Exception as e:
        print(f"Chat failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())