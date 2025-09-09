# Copyright (c) 2025 Matias Nielsen. All rights reserved.
# Licensed under the Custom License below.

import os
import sys
import logging
import argparse
import signal
import json
import pickle
import threading
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict
import re

import torch
import torch.nn.functional as F
from torch.cuda import get_device_properties

# Import the actual modules from the provided files
sys.path.insert(0, str(Path(__file__).parent))

try:
    from core.model import DeepSeekTransformer, DeepSeekConfig, create_deepseek_model
    from core.tokenizer import ConversationTokenizer, TokenizationMode
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure model.py and tokenizer.py are in the same directory as this script.")
    sys.exit(1)


@dataclass
class ConversationBranch:
    """Represents a conversation branch for advanced conversation management."""
    id: str
    parent_id: Optional[str]
    created_at: datetime
    messages: List[Dict[str, str]]
    metadata: Dict[str, Any]


class ModelAnalyzer:
    """Analyzes model behavior and provides insights."""
    
    def __init__(self, model: DeepSeekTransformer, tokenizer: ConversationTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.attention_patterns = []
        self.generation_stats = defaultdict(list)
    
    def analyze_attention(self, input_ids: torch.Tensor) -> Dict[str, Any]:
        """Analyze attention patterns in the model."""
        self.model.eval()
        with torch.no_grad():
            # Get logits and try to extract hidden states if model supports it
            try:
                outputs = self.model(input_ids, return_hidden_states=True)
                if isinstance(outputs, tuple):
                    logits, hidden_states = outputs[0], outputs[1]
                else:
                    logits = outputs
                    hidden_states = None
                
                # Basic analysis
                analysis = {
                    'sequence_length': input_ids.shape[1],
                    'vocab_size': logits.shape[-1],
                    'batch_size': input_ids.shape[0],
                }
                
                if hidden_states:
                    analysis.update({
                        'num_layers': len(hidden_states),
                        'hidden_size': hidden_states[0].shape[-1],
                        'layer_activations': [h.abs().mean().item() for h in hidden_states],
                        'layer_variances': [h.var().item() for h in hidden_states]
                    })
                
                return analysis
            except Exception as e:
                logging.error(f"Attention analysis failed: {e}")
                return {'error': str(e)}
    
    def analyze_generation_quality(self, prompt: str, responses: List[str]) -> Dict[str, Any]:
        """Analyze quality metrics for generated responses."""
        try:
            metrics = {
                'diversity': self._calculate_diversity(responses),
                'coherence': self._calculate_coherence(responses),
                'length_stats': self._calculate_length_stats(responses),
                'repetition_score': self._calculate_repetition(responses)
            }
            return metrics
        except Exception as e:
            logging.error(f"Generation quality analysis failed: {e}")
            return {'error': str(e)}
    
    def _calculate_diversity(self, responses: List[str]) -> float:
        """Calculate diversity score based on unique token usage."""
        if not responses:
            return 0.0
        
        all_tokens = []
        for response in responses:
            try:
                tokens = self.tokenizer.tokenizer.encode(response)
                all_tokens.extend(tokens)
            except:
                continue
        
        if not all_tokens:
            return 0.0
        
        unique_ratio = len(set(all_tokens)) / len(all_tokens)
        return unique_ratio
    
    def _calculate_coherence(self, responses: List[str]) -> float:
        """Simple coherence metric based on sentence structure."""
        if not responses:
            return 0.0
        
        coherence_scores = []
        for response in responses:
            # Basic coherence: ratio of complete sentences
            sentences = re.split(r'[.!?]+', response.strip())
            non_empty_sentences = [s for s in sentences if s.strip()]
            
            if len(non_empty_sentences) == 0:
                coherence_scores.append(0.0)
            else:
                # Penalize very short "sentences"
                valid_sentences = [s for s in non_empty_sentences if len(s.strip()) > 3]
                coherence = len(valid_sentences) / len(non_empty_sentences)
                coherence_scores.append(coherence)
        
        return sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0.0
    
    def _calculate_length_stats(self, responses: List[str]) -> Dict[str, float]:
        """Calculate length statistics."""
        if not responses:
            return {'mean': 0, 'min': 0, 'max': 0, 'std': 0}
        
        lengths = [len(response) for response in responses]
        mean_length = sum(lengths) / len(lengths)
        return {
            'mean': mean_length,
            'min': min(lengths),
            'max': max(lengths),
            'std': (sum((l - mean_length)**2 for l in lengths) / len(lengths))**0.5
        }
    
    def _calculate_repetition(self, responses: List[str]) -> float:
        """Calculate repetition score."""
        if not responses:
            return 0.0
        
        repetition_scores = []
        for response in responses:
            words = response.lower().split()
            if not words:
                repetition_scores.append(0.0)
                continue
            
            # Calculate bigram repetition
            bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
            if not bigrams:
                repetition_scores.append(0.0)
                continue
            
            unique_bigrams = len(set(bigrams))
            total_bigrams = len(bigrams)
            repetition = 1.0 - (unique_bigrams / total_bigrams)
            repetition_scores.append(repetition)
        
        return sum(repetition_scores) / len(repetition_scores) if repetition_scores else 0.0


def find_best_checkpoint() -> Optional[str]:
    """Find the best available checkpoint automatically."""
    print("Searching for checkpoints...")
    
    # List of checkpoint locations to search (in order of preference)
    search_locations = [
        Path("checkpoints"),
        Path("experiments"),
        Path("."),
    ]
    
    best_checkpoint = None
    best_time = 0
    
    for location in search_locations:
        if not location.exists():
            continue
            
        # Look for checkpoint files
        checkpoint_patterns = [
            "best_*.pt",
            "*_best.pt", 
            "checkpoint_*.pt",
            "model_*.pt",
            "*.pt",
            "*.pth",
            "*.bin"
        ]
        
        for pattern in checkpoint_patterns:
            checkpoints = list(location.glob(pattern))
            for checkpoint in checkpoints:
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


class AdvancedChat:
    """Advanced chat interface with comprehensive features."""
    
    def __init__(self, config: DeepSeekConfig, checkpoint_path: str):
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.conversation_history = []
        self.conversation_branches = {}
        self.current_branch_id = "main"
        self.session_stats = {
            'messages_sent': 0,
            'messages_received': 0,
            'session_start': datetime.now(),
            'total_tokens_generated': 0,
            'total_input_tokens': 0,
            'avg_response_time': 0.0,
            'generation_attempts': 0,
            'failed_generations': 0
        }
        
        # Setup device and check capabilities
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._check_system_capabilities()
        
        # Chat settings
        self.max_history_length = 15
        self.show_token_count = False
        self.show_timing = False
        self.show_analysis = False
        self.auto_save = False
        self.system_prompt = None
        self.conversation_mode = "standard"
        
        # Generation settings
        self.generation_configs = {
            'standard': {'temperature': 0.8, 'top_p': 0.9, 'top_k': 50},
            'creative': {'temperature': 1.0, 'top_p': 0.95, 'top_k': 100},
            'analytical': {'temperature': 0.3, 'top_p': 0.7, 'top_k': 20},
            'precise': {'temperature': 0.1, 'top_p': 0.5, 'top_k': 10}
        }
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.analyzer = None
        
        # Setup logging
        logging.basicConfig(
            level=logging.WARNING,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        self._initialize_chat_system()
    
    def _check_system_capabilities(self):
        """Check and display system capabilities."""
        self.system_info = {
            'device': str(self.device),
            'cuda_available': torch.cuda.is_available(),
            'torch_version': torch.__version__,
            'python_version': sys.version.split()[0]
        }
        
        if torch.cuda.is_available():
            props = get_device_properties(0)
            self.system_info.update({
                'gpu_name': props.name,
                'gpu_memory': f"{props.total_memory / 1024**3:.1f} GB",
                'gpu_compute_capability': f"{props.major}.{props.minor}"
            })
    
    def _initialize_chat_system(self):
        """Initialize tokenizer and model."""
        print("Initializing chat system...")
        
        try:
            # Initialize tokenizer - use GPT-2 to match model vocabulary
            self.tokenizer = ConversationTokenizer(
                model_name="gpt2",  # Changed to match model vocabulary
                max_context_length=self.config.seq_length,
                enable_caching=True,
                thread_safe=False
            )
            print(f"Tokenizer loaded (vocab size: {self.tokenizer.vocab_size:,})")
            
            # Initialize model
            self.model = DeepSeekTransformer(self.config).to(self.device)
            param_count = sum(p.numel() for p in self.model.parameters())
            print(f"Model initialized ({param_count:,} parameters)")
            
            # Initialize analyzer
            self.analyzer = ModelAnalyzer(self.model, self.tokenizer)
            
            # Load checkpoint
            self._load_checkpoint()
            
            # Set model to evaluation mode
            self.model.eval()
            
            print("Chat system ready!")
            
        except Exception as e:
            print(f"Failed to initialize chat system: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _load_checkpoint(self):
        """Load model checkpoint."""
        try:
            checkpoint_path = Path(self.checkpoint_path)
            
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
            print(f"Loading checkpoint: {checkpoint_path}")
            
            # Load checkpoint with proper device mapping
            if self.device.type == 'cuda':
                checkpoint = torch.load(checkpoint_path, map_location='cuda')
            else:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                if 'epoch' in checkpoint:
                    print(f"Loaded from epoch {checkpoint['epoch']}")
            elif 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                # Assume the checkpoint is just the state dict
                self.model.load_state_dict(checkpoint)
            
            print("Checkpoint loaded successfully")
            
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _generate_response(self, user_input: str) -> Tuple[str, Dict[str, Any]]:
        """Generate response with advanced sampling options."""
        start_time = time.time()
        
        try:
            # Add user message to history
            self.conversation_history.append({'role': 'user', 'content': user_input})
            
            # Format conversation for model
            conversation = self._format_conversation_for_model()
            
            # Tokenize input
            input_tokens = self.tokenizer.encode_conversation(conversation)
            
            if not input_tokens:
                return "I couldn't process your message. Please try again.", {}
            
            # Convert to tensor
            input_ids = torch.tensor([input_tokens], dtype=torch.long, device=self.device)
            
            # Truncate if too long - use config.seq_length instead of max_new_tokens
            max_new_tokens = getattr(self.config, 'max_new_tokens', 100)
            if input_ids.shape[1] > self.config.seq_length - max_new_tokens:
                input_ids = input_ids[:, -(self.config.seq_length - max_new_tokens):]
            
            # Get current generation config
            gen_config = self.generation_configs[self.conversation_mode]
            
            # Generate response
            generated_tokens = self._generate_tokens(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=gen_config['temperature'],
                top_p=gen_config['top_p'],
                top_k=gen_config['top_k']
            )
            
            # Decode response
            if generated_tokens:
                response = self.tokenizer.decode(
                    generated_tokens,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
                # Clean up response
                response = response.strip()
            else:
                response = "I'm sorry, I couldn't generate a response."
            
            # Calculate metrics
            response_time = time.time() - start_time
            input_token_count = len(input_tokens)
            output_token_count = len(generated_tokens)
            
            metrics = {
                'response_time': response_time,
                'input_tokens': input_token_count,
                'output_tokens': output_token_count,
                'tokens_per_second': output_token_count / response_time if response_time > 0 else 0,
                'total_tokens': input_token_count + output_token_count,
                'conversation_length': len(self.conversation_history)
            }
            
            # Update conversation history
            self.conversation_history.append({'role': 'assistant', 'content': response})
            
            # Update session stats
            self._update_session_stats(metrics)
            
            return response, metrics
            
        except Exception as e:
            logging.error(f"Generation failed: {e}")
            self.session_stats['failed_generations'] += 1
            return f"I apologize, but I encountered an error: {str(e)}", {}
    
    def _generate_tokens(self, input_ids: torch.Tensor, max_new_tokens: int, 
                        temperature: float, top_p: float, top_k: int) -> List[int]:
        """Advanced token generation with multiple sampling strategies."""
        self.model.eval()
        generated = []
        
        with torch.no_grad():
            current_ids = input_ids.clone()
            
            for _ in range(max_new_tokens):
                # Forward pass
                try:
                    logits = self.model(current_ids)
                    if isinstance(logits, tuple):
                        logits = logits[0]  # If model returns tuple, take first element
                    
                    next_token_logits = logits[0, -1, :]  # Last token predictions
                    
                    # Apply temperature
                    if temperature != 1.0:
                        next_token_logits = next_token_logits / temperature
                    
                    # Apply top-k filtering
                    if top_k > 0:
                        top_k_actual = min(top_k, next_token_logits.size(-1))
                        indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k_actual)[0][..., -1, None]
                        next_token_logits[indices_to_remove] = -float('Inf')
                    
                    # Apply top-p (nucleus) sampling
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        
                        # Remove tokens with cumulative probability above the threshold
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        
                        indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
                        next_token_logits[indices_to_remove] = -float('Inf')
                    
                    # Sample next token
                    probs = F.softmax(next_token_logits, dim=-1, dtype=torch.float32).to(next_token_logits.dtype)
                    next_token = torch.multinomial(probs, num_samples=1).item()
                    
                    # Check for end of sequence tokens
                    if next_token == self.tokenizer.tokenizer.eot_token:  # End of text token
                        break
                    
                    generated.append(next_token)
                    
                    # Add token to current sequence for next iteration
                    current_ids = torch.cat([current_ids, torch.tensor([[next_token]], device=self.device)], dim=1)
                    
                    # Check for context length limit
                    if current_ids.shape[1] >= self.config.seq_length:
                        break
                        
                except Exception as e:
                    logging.error(f"Token generation failed: {e}")
                    break
        
        return generated
    
    def _format_conversation_for_model(self) -> Dict[str, Any]:
        """Format conversation history for model input."""
        messages = []
        
        # Add system prompt if provided
        if self.system_prompt:
            messages.append({
                'role': 'system',
                'content': self.system_prompt
            })
        
        # Add conversation history (limit to max_history_length)
        history_to_include = self.conversation_history[-self.max_history_length:]
        
        for message in history_to_include:
            messages.append({
                'role': message['role'],
                'content': message['content']
            })
        
        return {'messages': messages}
    
    def _update_session_stats(self, metrics: Dict[str, Any]):
        """Update session statistics."""
        self.session_stats['messages_sent'] += 1
        self.session_stats['messages_received'] += 1
        self.session_stats['total_tokens_generated'] += metrics.get('output_tokens', 0)
        self.session_stats['total_input_tokens'] += metrics.get('input_tokens', 0)
        self.session_stats['generation_attempts'] += 1
        
        # Update average response time
        current_avg = self.session_stats['avg_response_time']
        new_time = metrics.get('response_time', 0)
        msg_count = self.session_stats['messages_received']
        self.session_stats['avg_response_time'] = ((current_avg * (msg_count - 1)) + new_time) / msg_count
    
    def _handle_command(self, command: str) -> bool:
        """Handle special commands."""
        parts = command.strip().split(None, 1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        
        if cmd in ['/quit', '/exit']:
            return False
        
        elif cmd == '/help':
            self._show_help()
        
        elif cmd == '/stats':
            self._show_stats()
        
        elif cmd == '/analysis':
            self._toggle_analysis()
        
        elif cmd == '/clear':
            self._clear_history()
        
        elif cmd == '/save':
            self._save_conversation(args)
        
        elif cmd == '/load':
            self._load_conversation(args)
        
        elif cmd == '/export':
            self._export_conversation(args)
        
        elif cmd == '/settings':
            self._show_settings()
        
        elif cmd == '/mode':
            self._set_conversation_mode(args)
        
        elif cmd == '/system':
            self._set_system_prompt(args)
        
        elif cmd == '/regenerate':
            self._regenerate_last_response()
        
        elif cmd == '/compare':
            self._compare_responses(args)
        
        elif cmd == '/benchmark':
            self._run_benchmark()
        
        elif cmd == '/inspect':
            self._inspect_model()
        
        elif cmd == '/token':
            self._analyze_tokenization(args)
        
        else:
            print(f"Unknown command: {cmd}")
            print("Type /help for available commands")
        
        return True
    
    def _show_help(self):
        """Show comprehensive help."""
        print("\nHELP - Advanced Chat Interface")
        print("=" * 60)
        print("BASIC COMMANDS:")
        print("  /help             - Show this help")
        print("  /quit, /exit      - Exit chat")
        print("  /clear            - Clear conversation history")
        print("  /stats            - Show detailed session statistics")
        print("\nCONVERSATION MANAGEMENT:")
        print("  /save [name]      - Save conversation (auto-named if no name)")
        print("  /load [name]      - Load saved conversation")
        print("  /export [format]  - Export conversation (json/txt/md)")
        print("  /regenerate       - Regenerate last response")
        print("\nSETTINGS:")
        print("  /settings         - Show/modify chat settings")
        print("  /mode <mode>      - Set conversation mode:")
        print("    • standard      - Balanced responses")
        print("    • creative      - More creative/diverse responses")
        print("    • analytical    - Precise, analytical responses")
        print("    • precise       - Very focused responses")
        print("  /system <prompt>  - Set system prompt")
        print("\nANALYSIS TOOLS:")
        print("  /analysis         - Toggle response analysis")
        print("  /compare <n>      - Compare N different responses")
        print("  /benchmark        - Run generation benchmark")
        print("  /inspect          - Inspect model internals")
        print("  /token <text>     - Analyze tokenization of text")
        print("=" * 60)
    
    def _show_stats(self):
        """Show comprehensive session statistics."""
        duration = datetime.now() - self.session_stats['session_start']
        
        print("\nSESSION STATISTICS")
        print("=" * 50)
        print(f"Duration: {duration}")
        print(f"Messages exchanged: {self.session_stats['messages_sent']}")
        print(f"Total input tokens: {self.session_stats['total_input_tokens']:,}")
        print(f"Total output tokens: {self.session_stats['total_tokens_generated']:,}")
        print(f"Generation attempts: {self.session_stats['generation_attempts']}")
        print(f"Failed generations: {self.session_stats['failed_generations']}")
        print(f"Average response time: {self.session_stats['avg_response_time']:.2f}s")
        
        # Calculate tokens per minute
        total_minutes = duration.total_seconds() / 60
        if total_minutes > 0:
            tokens_per_minute = self.session_stats['total_tokens_generated'] / total_minutes
            print(f"Tokens per minute: {tokens_per_minute:.1f}")
        
        print(f"\nCONVERSATION INFO:")
        print(f"Exchanges: {len(self.conversation_history)}")
        print(f"Current mode: {self.conversation_mode}")
        if self.system_prompt:
            print(f"System prompt: {self.system_prompt[:50]}...")
        
        print(f"\nSYSTEM INFO:")
        for key, value in self.system_info.items():
            print(f"{key}: {value}")
        print("=" * 50)
    
    def _toggle_analysis(self):
        """Toggle response analysis."""
        self.show_analysis = not self.show_analysis
        status = "enabled" if self.show_analysis else "disabled"
        print(f"Response analysis {status}")
    
    def _clear_history(self):
        """Clear conversation history with confirmation."""
        if not self.conversation_history:
            print("No conversation to clear")
            return
        
        confirm = input(f"Clear {len(self.conversation_history)} exchanges? (y/N): ").strip().lower()
        if confirm in ['y', 'yes']:
            self.conversation_history.clear()
            print("Conversation history cleared")
    
    def _save_conversation(self, name: str = ""):
        """Save conversation."""
        if not self.conversation_history:
            print("No conversation to save")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if not name:
            name = f"chat_{timestamp}"
        
        try:
            conversation_data = {
                'name': name,
                'timestamp': timestamp,
                'system_prompt': self.system_prompt,
                'conversation_mode': self.conversation_mode,
                'conversation': self.conversation_history,
                'session_stats': self.session_stats,
                'system_info': self.system_info
            }
            
            Path("conversations").mkdir(exist_ok=True)
            filepath = Path("conversations") / f"{name}.json"
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(conversation_data, f, indent=2, default=str)
            
            print(f"Conversation saved: {filepath}")
            
        except Exception as e:
            print(f"Failed to save conversation: {e}")
    
    def _load_conversation(self, name: str = ""):
        """Load conversation."""
        conversations_dir = Path("conversations")
        if not conversations_dir.exists():
            print("No conversations directory found")
            return
        
        if name:
            filepath = conversations_dir / f"{name}.json"
            if not filepath.exists():
                print(f"Conversation '{name}' not found")
                return
            conversation_files = [filepath]
        else:
            conversation_files = list(conversations_dir.glob("*.json"))
            
            if not conversation_files:
                print("No saved conversations found")
                return
            
            print("\nSaved conversations:")
            for i, filepath in enumerate(conversation_files, 1):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    timestamp = data.get('timestamp', 'Unknown')
                    exchanges = len(data.get('conversation', []))
                    mode = data.get('conversation_mode', 'unknown')
                    print(f"  {i}. {filepath.stem} - {timestamp} ({exchanges} exchanges, {mode} mode)")
                except Exception:
                    print(f"  {i}. {filepath.stem} - (corrupted)")
            
            try:
                choice = input("\nEnter number to load (or Enter to cancel): ").strip()
                if not choice:
                    return
                
                file_index = int(choice) - 1
                if 0 <= file_index < len(conversation_files):
                    filepath = conversation_files[file_index]
                else:
                    print("Invalid selection")
                    return
            except ValueError:
                print("Invalid input")
                return
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.conversation_history = data.get('conversation', [])
            if data.get('system_prompt'):
                self.system_prompt = data['system_prompt']
            if data.get('conversation_mode'):
                self.conversation_mode = data['conversation_mode']
            
            print(f"Conversation loaded: {filepath.name}")
            print(f"  Exchanges: {len(self.conversation_history)}")
            print(f"  Mode: {self.conversation_mode}")
            if self.system_prompt:
                print(f"  System prompt: {self.system_prompt[:50]}...")
                
        except Exception as e:
            print(f"Failed to load conversation: {e}")
    
    def _export_conversation(self, format_type: str = ""):
        """Export conversation in different formats."""
        if not self.conversation_history:
            print("No conversation to export")
            return
        
        if not format_type:
            format_type = input("Export format (json/txt/md): ").strip().lower()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            Path("exports").mkdir(exist_ok=True)
            
            if format_type == "json":
                filepath = Path("exports") / f"chat_{timestamp}.json"
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(self.conversation_history, f, indent=2, default=str)
            
            elif format_type == "txt":
                filepath = Path("exports") / f"chat_{timestamp}.txt"
                with open(filepath, 'w', encoding='utf-8') as f:
                    for exchange in self.conversation_history:
                        if exchange['role'] == 'user':
                            f.write(f"User: {exchange['content']}\n")
                        elif exchange['role'] == 'assistant':
                            f.write(f"Assistant: {exchange['content']}\n")
                        f.write("\n")
            
            elif format_type == "md":
                filepath = Path("exports") / f"chat_{timestamp}.md"
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write("# Chat Conversation\n\n")
                    f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"**Mode:** {self.conversation_mode}\n\n")
                    
                    for i, exchange in enumerate(self.conversation_history, 1):
                        if exchange['role'] == 'user':
                            f.write(f"## User (Exchange {i})\n\n")
                            f.write(f"{exchange['content']}\n\n")
                        elif exchange['role'] == 'assistant':
                            f.write(f"## Assistant (Exchange {i})\n\n")
                            f.write(f"{exchange['content']}\n\n")
            
            else:
                print("Invalid export format. Use: json, txt, or md")
                return
            
            print(f"Conversation exported: {filepath}")
            
        except Exception as e:
            print(f"Failed to export conversation: {e}")
    
    def _show_settings(self):
        """Show and modify comprehensive settings."""
        print("\nCURRENT SETTINGS")
        print("=" * 40)
        print(f"Conversation mode: {self.conversation_mode}")
        print(f"Max history length: {self.max_history_length}")
        print(f"Show token count: {self.show_token_count}")
        print(f"Show timing: {self.show_timing}")
        print(f"Show analysis: {self.show_analysis}")
        print(f"Auto save: {self.auto_save}")
        
        gen_config = self.generation_configs[self.conversation_mode]
        print(f"\nGENERATION SETTINGS ({self.conversation_mode} mode):")
        print(f"Temperature: {gen_config['temperature']}")
        print(f"Top-p: {gen_config['top_p']}")
        print(f"Top-k: {gen_config['top_k']}")
        
        if self.system_prompt:
            print(f"\nSystem prompt: {self.system_prompt}")
        print("=" * 40)
        
        modify = input("Modify settings? (y/N): ").strip().lower()
        if modify in ['y', 'yes']:
            self._modify_settings()
    
    def _modify_settings(self):
        """Interactive settings modification."""
        print("\nModify settings (press Enter to keep current value):")
        
        try:
            # Max history length
            new_history = input(f"Max history length ({self.max_history_length}): ").strip()
            if new_history and new_history.isdigit():
                self.max_history_length = int(new_history)
            
            # Toggle settings
            for setting, desc in [
                ('show_token_count', 'Show token count'),
                ('show_timing', 'Show timing'),
                ('show_analysis', 'Show analysis'),
                ('auto_save', 'Auto save')
            ]:
                current = getattr(self, setting)
                new_value = input(f"{desc} ({current}) [y/n]: ").strip().lower()
                if new_value in ['y', 'yes', 'true']:
                    setattr(self, setting, True)
                elif new_value in ['n', 'no', 'false']:
                    setattr(self, setting, False)
            
            print("Settings updated")
            
        except KeyboardInterrupt:
            print("\nSettings modification cancelled")
    
    def _set_conversation_mode(self, mode: str):
        """Set conversation mode."""
        mode = mode.strip().lower()
        if mode in self.generation_configs:
            self.conversation_mode = mode
            gen_config = self.generation_configs[mode]
            print(f"Conversation mode set to: {mode}")
            print(f"  Temperature: {gen_config['temperature']}")
            print(f"  Top-p: {gen_config['top_p']}")
            print(f"  Top-k: {gen_config['top_k']}")
        else:
            print(f"Invalid mode. Available modes: {', '.join(self.generation_configs.keys())}")
    
    def _set_system_prompt(self, prompt: str):
        """Set or clear system prompt."""
        if prompt.strip():
            self.system_prompt = prompt.strip()
            print(f"System prompt set: {prompt[:50]}...")
        else:
            self.system_prompt = None
            print("System prompt cleared")
    
    def _regenerate_last_response(self):
        """Regenerate the last assistant response."""
        if not self.conversation_history:
            print("No conversation history to regenerate from")
            return
        
        # Find the last user message
        last_user_idx = None
        for i in range(len(self.conversation_history)-1, -1, -1):
            if self.conversation_history[i]['role'] == 'user':
                last_user_idx = i
                break
        
        if last_user_idx is None:
            print("No user message to respond to")
            return
        
        user_message = self.conversation_history[last_user_idx]['content']
        
        # Remove all messages after the user message
        self.conversation_history = self.conversation_history[:last_user_idx+1]
        
        print("Regenerating last response...")
        print(f"User: {user_message}")
        print("Assistant: ", end="", flush=True)
        
        response, metrics = self._generate_response(user_message)
        print(response)
        
        if self.show_timing or self.show_token_count:
            self._show_metrics(metrics)
    
    def _compare_responses(self, args: str):
        """Generate multiple responses for comparison."""
        try:
            count = int(args) if args.isdigit() else 3
            count = min(count, 5)  # Limit to 5 responses
        except:
            count = 3
        
        if not self.conversation_history:
            print("No conversation history for comparison")
            return
        
        # Find the last user message
        last_user_idx = None
        for i in range(len(self.conversation_history)-1, -1, -1):
            if self.conversation_history[i]['role'] == 'user':
                last_user_idx = i
                break
        
        if last_user_idx is None:
            print("No user message to respond to")
            return
        
        user_message = self.conversation_history[last_user_idx]['content']
        print(f"\nGenerating {count} different responses to: '{user_message[:50]}...'")
        
        responses = []
        original_history = self.conversation_history.copy()
        
        for i in range(count):
            print(f"\nResponse {i+1}:")
            
            # Restore original history (without previous assistant responses)
            self.conversation_history = original_history.copy()
            
            response, metrics = self._generate_response(user_message)
            responses.append(response)
            print(response)
            
            if self.show_analysis:
                print(f"   [Tokens: {metrics.get('output_tokens', 0)}, Time: {metrics.get('response_time', 0):.2f}s]")
        
        # Restore original history
        self.conversation_history = original_history
        
        # Analyze responses
        if self.show_analysis:
            analysis = self.analyzer.analyze_generation_quality(user_message, responses)
            print(f"\nQuality Analysis:")
            print(f"  Diversity: {analysis['diversity']:.3f}")
            print(f"  Coherence: {analysis['coherence']:.3f}")
            print(f"  Avg length: {analysis['length_stats']['mean']:.1f} chars")
            print(f"  Repetition: {analysis['repetition_score']:.3f}")
    
    def _run_benchmark(self):
        """Run generation benchmark."""
        print("Running generation benchmark...")
        
        test_prompts = [
            "Hello, how are you?",
            "What is artificial intelligence?",
            "Tell me a short story.",
            "Explain quantum physics in simple terms.",
            "What's the weather like?"
        ]
        
        results = []
        original_history = self.conversation_history.copy()
        
        for prompt in test_prompts:
            print(f"Testing: '{prompt[:30]}...'")
            
            # Clear history for clean test
            self.conversation_history = [{'role': 'user', 'content': prompt}]
            
            start_time = time.time()
            response, metrics = self._generate_response(prompt)
            end_time = time.time()
            
            results.append({
                'prompt': prompt,
                'response_length': len(response),
                'response_time': end_time - start_time,
                'tokens_generated': metrics.get('output_tokens', 0),
                'tokens_per_second': metrics.get('tokens_per_second', 0)
            })
            
        # Restore original history
        self.conversation_history = original_history
        
        # Display results
        print("\nBenchmark Results:")
        print("-" * 60)
        total_time = sum(r['response_time'] for r in results)
        total_tokens = sum(r['tokens_generated'] for r in results)
        avg_tokens_per_sec = total_tokens / total_time if total_time > 0 else 0
        
        print(f"Average response time: {total_time / len(results):.2f}s")
        print(f"Average tokens/second: {avg_tokens_per_sec:.1f}")
        print(f"Total tokens generated: {total_tokens}")
        print("-" * 60)
    
    def _inspect_model(self):
        """Inspect model internals."""
        print("\nMODEL INSPECTION")
        print("=" * 40)
        print(f"Model class: {self.model.__class__.__name__}")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Trainable params: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        print(f"Model device: {next(self.model.parameters()).device}")
        print(f"Model dtype: {next(self.model.parameters()).dtype}")
        
        # Memory usage
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_cached = torch.cuda.memory_reserved() / 1024**3
            print(f"GPU memory allocated: {memory_allocated:.2f} GB")
            print(f"GPU memory cached: {memory_cached:.2f} GB")
        
        # Architecture details
        print(f"\nArchitecture:")
        print(f"  Hidden size: {self.config.hidden_size}")
        print(f"  Layers: {self.config.num_layers}")
        print(f"  Attention heads: {self.config.num_heads}")
        print(f"  KV heads: {self.config.num_kv_heads}")
        print(f"  Sequence length: {self.config.seq_length}")
        print(f"  Vocab size: {self.config.vocab_size:,}")
        print("=" * 40)
    
    def _analyze_tokenization(self, text: str):
        """Analyze tokenization of given text."""
        if not text.strip():
            text = input("Enter text to analyze: ").strip()
        
        if not text:
            return
        
        print(f"\nTokenization Analysis for: '{text}'")
        print("-" * 50)
        
        # Basic tokenization
        try:
            tokens = self.tokenizer.tokenizer.encode(text)
            print(f"Token count: {len(tokens)}")
            print(f"Character count: {len(text)}")
            print(f"Chars per token: {len(text) / len(tokens) if tokens else 0:.2f}")
            
            # Show token breakdown
            print(f"\nToken breakdown:")
            for i, token_id in enumerate(tokens[:20]):  # Show first 20 tokens
                try:
                    token_text = self.tokenizer.tokenizer.decode([token_id])
                    print(f"  {i:2d}: {token_id:5d} -> '{token_text}'")
                except:
                    print(f"  {i:2d}: {token_id:5d} -> [decode error]")
            
            if len(tokens) > 20:
                print(f"  ... and {len(tokens) - 20} more tokens")
        except Exception as e:
            print(f"Tokenization failed: {e}")
        
        print("-" * 50)
    
    def _show_metrics(self, metrics: Dict[str, Any]):
        """Display response metrics."""
        info_parts = []
        if self.show_timing and 'response_time' in metrics:
            info_parts.append(f"Time: {metrics['response_time']:.2f}s")
        if self.show_token_count:
            if 'output_tokens' in metrics:
                info_parts.append(f"Out: {metrics['output_tokens']} tokens")
            if 'input_tokens' in metrics:
                info_parts.append(f"In: {metrics['input_tokens']} tokens")
            if 'tokens_per_second' in metrics:
                info_parts.append(f"Speed: {metrics['tokens_per_second']:.1f} tok/s")
        
        if info_parts:
            print(f"   [{' | '.join(info_parts)}]")
    
    def _print_header(self):
        """Print enhanced chat header."""
        print("\n" + "="*80)
        print("CONVERSATIONAL TRANSFORMER CHAT - ENHANCED")
        print("="*80)
        print(f"Model: DeepSeek Transformer")
        print(f"Parameters: ~{sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Device: {self.device}")
        print(f"Mode: {self.conversation_mode}")
        
        gen_config = self.generation_configs[self.conversation_mode]
        print(f"Generation settings - Temp: {gen_config['temperature']}, Top-p: {gen_config['top_p']}, Top-k: {gen_config['top_k']}")
        
        print(f"Checkpoint: {self.checkpoint_path}")
        if self.system_prompt:
            print(f"System prompt: {self.system_prompt[:50]}...")
        
        print("\nQuick Commands: /help /stats /save /load /mode /settings /quit")
        print("="*80)
        print("Start chatting! (Type your message and press Enter)\n")
    
    def run_chat(self):
        """Run the enhanced interactive chat loop."""
        self._print_header()
        
        # Setup graceful exit handler
        def signal_handler(signum, frame):
            print("\n\nChat session ended by user")
            if self.auto_save and self.conversation_history:
                self._save_conversation()
            self._show_stats()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        
        try:
            while True:
                # Get user input with enhanced prompt
                try:
                    mode_indicator = self.conversation_mode[0].upper()
                    user_input = input(f"[{mode_indicator}] You: ").strip()
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
                
                # Show metrics
                if self.show_token_count or self.show_timing or self.show_analysis:
                    self._show_metrics(metrics)
                    
                    if self.show_analysis and len(self.conversation_history) > 0:
                        # Analyze this specific response
                        last_response = self.conversation_history[-1]['content'] if self.conversation_history[-1]['role'] == 'assistant' else ''
                        if last_response:
                            analysis = self.analyzer.analyze_generation_quality(user_input, [last_response])
                            print(f"   [Quality - Coherence: {analysis['coherence']:.2f}, Length: {len(last_response)} chars]")
                
                # Auto-save if enabled
                if self.auto_save and len(self.conversation_history) % 5 == 0:
                    self._save_conversation(f"auto_save_{datetime.now().strftime('%H%M%S')}")
                
                print()  # Extra line for readability
        
        except Exception as e:
            print(f"\nChat error: {e}")
            logging.error(f"Chat loop error: {e}")
        
        finally:
            if self.auto_save and self.conversation_history:
                self._save_conversation("final_auto_save")
            print("\nFinal Statistics:")
            self._show_stats()
            print("\nGoodbye!")


def main():
    """Enhanced main function with comprehensive options."""
    parser = argparse.ArgumentParser(
        description='Enhanced Interactive Chat Interface for DeepSeek Transformer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic chat with auto-detected best model
  python chat.py
  
  # Chat with specific checkpoint
  python chat.py --checkpoint checkpoints/best_checkpoint.pt
  
  # Chat in creative mode with analysis
  python chat.py --config medium --mode creative --show-analysis
  
  # Chat with auto-save and custom settings
  python chat.py --checkpoint latest --auto-save
  
  # Benchmark mode for testing
  python chat.py --config small --benchmark
        """
    )
    
    # Configuration options
    parser.add_argument('--config', choices=['small', 'medium', 'large'],
                       default='medium', help='Configuration preset')
    
    # Model options
    parser.add_argument('--checkpoint', type=str, help='Checkpoint to load')
    
    # Generation settings
    parser.add_argument('--temperature', type=float, help='Sampling temperature')
    parser.add_argument('--top-p', type=float, help='Top-p sampling threshold')
    parser.add_argument('--top-k', type=int, help='Top-k sampling threshold')
    parser.add_argument('--max-tokens', type=int, default=100, help='Maximum tokens to generate')
    
    # Chat settings
    parser.add_argument('--mode', choices=['standard', 'creative', 'analytical', 'precise'],
                       default='standard', help='Conversation mode')
    parser.add_argument('--system-prompt', type=str, help='Initial system prompt')
    parser.add_argument('--max-history', type=int, default=15, help='Max conversation history length')
    
    # Display options
    parser.add_argument('--show-tokens', action='store_true', help='Show token counts')
    parser.add_argument('--show-timing', action='store_true', help='Show response timing')
    parser.add_argument('--show-analysis', action='store_true', help='Show response analysis')
    parser.add_argument('--auto-save', action='store_true', help='Auto-save conversations')
    
    # Special modes
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark and exit')
    parser.add_argument('--inspect', action='store_true', help='Inspect model and exit')
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config_map = {
            'small': DeepSeekConfig.small,
            'medium': DeepSeekConfig.medium,
            'large': DeepSeekConfig.large,
        }
        config = config_map[args.config]()
        
        # Add missing attributes that the chat system expects
        config.max_new_tokens = args.max_tokens
        config.experiment_name = f"deepseek_{args.config}"
        
    except Exception as e:
        print(f"Configuration error: {e}")
        return 1
    
    # Determine checkpoint path
    checkpoint_path = args.checkpoint
    if not checkpoint_path:
        # Auto-detect checkpoint
        checkpoint_path = find_best_checkpoint()
        if not checkpoint_path:
            return 1
    
    # Initialize chat
    try:
        chat = AdvancedChat(config, checkpoint_path)
        
        # Apply settings from CLI
        chat.conversation_mode = args.mode
        if args.system_prompt:
            chat.system_prompt = args.system_prompt
        chat.show_token_count = args.show_tokens
        chat.show_timing = args.show_timing
        chat.show_analysis = args.show_analysis
        chat.auto_save = args.auto_save
        chat.max_history_length = args.max_history
        
        # Override generation configs if specified
        if args.temperature is not None:
            for mode_config in chat.generation_configs.values():
                mode_config['temperature'] = args.temperature
        if args.top_p is not None:
            for mode_config in chat.generation_configs.values():
                mode_config['top_p'] = args.top_p
        if args.top_k is not None:
            for mode_config in chat.generation_configs.values():
                mode_config['top_k'] = args.top_k
        
        # Handle special modes
        if args.benchmark:
            chat._run_benchmark()
            return 0
        
        if args.inspect:
            chat._inspect_model()
            return 0
        
        # Run normal chat
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