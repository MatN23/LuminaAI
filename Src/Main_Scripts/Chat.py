#!/usr/bin/env python3
"""
Enhanced interactive chat interface for conversational transformer models.
Includes advanced features like conversation branching, export options, and model analysis.
"""

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

# Add the current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import our modules
try:
    from config.config_manager import Config, ConfigPresets
    from core.tokenizer import ConversationTokenizer, TokenizationMode
    from core.model import TransformerModel, estimate_parameters
    from training.checkpoint import CheckpointManager
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all required modules are present in the correct directory structure.")
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
    
    def __init__(self, model: TransformerModel, tokenizer: ConversationTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.attention_patterns = []
        self.generation_stats = defaultdict(list)
    
    def analyze_attention(self, input_ids: torch.Tensor) -> Dict[str, Any]:
        """Analyze attention patterns in the model."""
        self.model.eval()
        with torch.no_grad():
            # Get hidden states and attention weights
            logits, hidden_states = self.model(input_ids, return_hidden_states=True)
            
            # Basic analysis
            analysis = {
                'num_layers': len(hidden_states),
                'sequence_length': input_ids.shape[1],
                'hidden_size': hidden_states[0].shape[-1],
                'layer_activations': [h.abs().mean().item() for h in hidden_states],
                'layer_variances': [h.var().item() for h in hidden_states]
            }
            
            return analysis
    
    def analyze_generation_quality(self, prompt: str, responses: List[str]) -> Dict[str, Any]:
        """Analyze quality metrics for generated responses."""
        metrics = {
            'diversity': self._calculate_diversity(responses),
            'coherence': self._calculate_coherence(responses),
            'length_stats': self._calculate_length_stats(responses),
            'repetition_score': self._calculate_repetition(responses)
        }
        return metrics
    
    def _calculate_diversity(self, responses: List[str]) -> float:
        """Calculate diversity score based on unique token usage."""
        if not responses:
            return 0.0
        
        all_tokens = []
        for response in responses:
            tokens = self.tokenizer.tokenizer.encode(response)
            all_tokens.extend(tokens)
        
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
        
        return sum(coherence_scores) / len(coherence_scores)
    
    def _calculate_length_stats(self, responses: List[str]) -> Dict[str, float]:
        """Calculate length statistics."""
        if not responses:
            return {'mean': 0, 'min': 0, 'max': 0, 'std': 0}
        
        lengths = [len(response) for response in responses]
        return {
            'mean': sum(lengths) / len(lengths),
            'min': min(lengths),
            'max': max(lengths),
            'std': (sum((l - sum(lengths)/len(lengths))**2 for l in lengths) / len(lengths))**0.5
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
        
        return sum(repetition_scores) / len(repetition_scores)


def find_best_checkpoint() -> Optional[str]:
    """Find the best available checkpoint automatically."""
    print("Searching for checkpoints...")
    
    # List of checkpoint locations to search (in order of preference)
    search_locations = [
        # First try the CheckpointManager structure
        Path("checkpoints"),
        # Also check current directory for loose checkpoint files
        Path("."),
    ]
    
    best_checkpoint = None
    best_info = None
    
    for location in search_locations:
        if not location.exists():
            continue
            
        print(f"Checking {location}...")
        
        # Look for experiment directories with checkpoint history
        for exp_dir in location.iterdir():
            if not exp_dir.is_dir():
                continue
                
            history_file = exp_dir / "checkpoint_history.json"
            if history_file.exists():
                try:
                    with open(history_file, 'r') as f:
                        data = json.load(f)
                    
                    best_path = data.get('best_checkpoint_path')
                    if best_path and Path(best_path).exists():
                        best_metric = data.get('best_metric_value', float('inf'))
                        if best_checkpoint is None or best_metric < best_info['metric']:
                            best_checkpoint = best_path
                            best_info = {
                                'metric': best_metric,
                                'experiment': exp_dir.name,
                                'type': 'managed_best'
                            }
                            print(f"Found managed best checkpoint: {exp_dir.name} (loss: {best_metric:.6f})")
                except Exception as e:
                    print(f"Error reading checkpoint history in {exp_dir}: {e}")
            
            # Also check for direct best_checkpoint.pt symlink
            best_link = exp_dir / "best_checkpoint.pt"
            if best_link.exists():
                if best_checkpoint is None:
                    best_checkpoint = str(best_link)
                    best_info = {'experiment': exp_dir.name, 'type': 'symlink_best'}
                    print(f"Found best checkpoint symlink: {exp_dir.name}")
        
        # Look for standalone checkpoint files
        checkpoint_patterns = [
            "best_*.pt",
            "*_best.pt", 
            "checkpoint_*.pt",
            "model_*.pt"
        ]
        
        for pattern in checkpoint_patterns:
            checkpoints = list(location.glob(pattern))
            if checkpoints:
                # Sort by modification time, newest first
                checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                latest = checkpoints[0]
                
                if best_checkpoint is None:
                    best_checkpoint = str(latest)
                    best_info = {'type': 'standalone', 'pattern': pattern}
                    print(f"Found standalone checkpoint: {latest.name}")
                    break
    
    if best_checkpoint:
        print(f"Selected checkpoint: {best_checkpoint}")
        return best_checkpoint
    else:
        print("ERROR: No checkpoints found!")
        return None


class AdvancedChat:
    """Advanced chat interface with comprehensive features."""
    
    def __init__(self, config: Config, checkpoint_path: str):
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.conversation_history = []
        self.conversation_branches = {}  # For conversation branching
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
        self.conversation_mode = "standard"  # standard, creative, analytical
        
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
        self.checkpoint_manager = None
        
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
            # Initialize tokenizer with enhanced settings
            self.tokenizer = ConversationTokenizer(
                model_name="gpt-4",
                max_context_length=self.config.seq_length,
                enable_caching=True,
                thread_safe=False
            )
            self.config.vocab_size = self.tokenizer.vocab_size
            print(f"Tokenizer loaded (vocab size: {self.tokenizer.vocab_size:,})")
            
            # Initialize model
            self.model = TransformerModel(self.config).to(self.device)
            param_count = estimate_parameters(self.config)
            print(f"Model initialized (~{param_count:,} parameters)")
            
            # Initialize checkpoint manager
            self.checkpoint_manager = CheckpointManager(self.config)
            
            # Initialize analyzer
            self.analyzer = ModelAnalyzer(self.model, self.tokenizer)
            
            # Load checkpoint - this is now required
            self._load_checkpoint()
            
            print("Chat system ready!")
            
        except Exception as e:
            print(f"Failed to initialize chat system: {e}")
            raise
    
    def _load_checkpoint(self):
        """Load model checkpoint with enhanced error handling."""
        try:
            checkpoint_path = Path(self.checkpoint_path)
            
            # Handle special keywords using CheckpointManager
            if self.checkpoint_path == "best":
                best_path = self.checkpoint_manager.get_best_checkpoint()
                if best_path:
                    checkpoint_path = best_path
                else:
                    print("No best checkpoint found in checkpoint manager")
                    # Fallback to global search
                    fallback = find_best_checkpoint()
                    if fallback:
                        checkpoint_path = Path(fallback)
                    else:
                        raise FileNotFoundError("No checkpoint available")
                    
            elif self.checkpoint_path == "latest":
                latest_path = self.checkpoint_manager.get_latest_checkpoint()
                if latest_path:
                    checkpoint_path = latest_path
                else:
                    print("No latest checkpoint found in checkpoint manager")
                    raise FileNotFoundError("No checkpoint available")
            
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
            print(f"Loading checkpoint: {checkpoint_path}")
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                # Enhanced checkpoint format
                self.model.load_state_dict(checkpoint['model_state_dict'])
                
                # Display checkpoint info
                if 'current_epoch' in checkpoint:
                    print(f"Loaded from epoch {checkpoint['current_epoch']}")
                if 'global_step' in checkpoint:
                    print(f"Global step: {checkpoint['global_step']}")
                if 'metrics' in checkpoint:
                    metrics = checkpoint['metrics']
                    if 'eval_losses' in metrics and metrics['eval_losses']:
                        print(f"Best eval loss: {min(metrics['eval_losses']):.6f}")
                
                # Validate model compatibility
                if 'model_config' in checkpoint:
                    self._validate_checkpoint_config(checkpoint['model_config'])
                    
            else:
                # Assume the checkpoint is just the state dict
                self.model.load_state_dict(checkpoint)
            
            print("Checkpoint loaded successfully")
            
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            raise
    
    def _validate_checkpoint_config(self, checkpoint_config: Dict[str, Any]):
        """Validate that checkpoint config matches current config."""
        current_config = {
            'hidden_size': self.config.hidden_size,
            'num_layers': self.config.num_layers,
            'num_heads': self.config.num_heads,
            'vocab_size': self.config.vocab_size,
        }
        
        mismatches = []
        for key, current_value in current_config.items():
            checkpoint_value = checkpoint_config.get(key)
            if checkpoint_value is not None and checkpoint_value != current_value:
                mismatches.append(f"{key}: checkpoint={checkpoint_value}, current={current_value}")
        
        if mismatches:
            print("WARNING: Config mismatches detected:")
            for mismatch in mismatches:
                print(f"  {mismatch}")
            
            # Auto-update config to match checkpoint if reasonable
            if len(mismatches) <= 2:  # Only auto-update for minor mismatches
                for key in current_config:
                    if key in checkpoint_config:
                        setattr(self.config, key, checkpoint_config[key])
                print("Config auto-updated to match checkpoint")
    
    def _generate_response(self, user_input: str) -> Tuple[str, Dict[str, Any]]:
        """Generate response with advanced sampling options."""
        start_time = time.time()
        
        try:
            # Add user message to history
            if not self.conversation_history or 'assistant' in self.conversation_history[-1]:
                self.conversation_history.append({'user': user_input})
            
            # Format conversation for model
            conversation = self._format_conversation_for_model()
            
            # Tokenize input
            input_tokens = self.tokenizer.encode_conversation(conversation)
            
            if not input_tokens:
                return "I couldn't process your message. Please try again.", {}
            
            # Convert to tensor
            input_ids = torch.tensor([input_tokens], dtype=torch.long, device=self.device)
            
            # Truncate if too long
            if input_ids.shape[1] > self.config.seq_length - self.config.max_new_tokens:
                input_ids = input_ids[:, -(self.config.seq_length - self.config.max_new_tokens):]
            
            # Get current generation config
            gen_config = self.generation_configs[self.conversation_mode]
            
            # Generate response
            generated_tokens = self._generate_tokens(
                input_ids,
                max_new_tokens=self.config.max_new_tokens,
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
            self._update_conversation_history(user_input, response)
            
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
                logits = self.model(current_ids)
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
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
                
                # Check for special tokens that indicate end of response
                if self.tokenizer.is_special_token(next_token):
                    if next_token == self.tokenizer.special_tokens.get("<|im_end|>"):
                        break
                
                generated.append(next_token)
                
                # Add token to current sequence for next iteration
                current_ids = torch.cat([current_ids, torch.tensor([[next_token]], device=self.device)], dim=1)
                
                # Check for context length limit
                if current_ids.shape[1] >= self.config.seq_length:
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
        
        # Add conversation history
        history_to_include = self.conversation_history[-self.max_history_length:]
        
        for exchange in history_to_include:
            messages.append({
                'role': 'user',
                'content': exchange['user']
            })
            if 'assistant' in exchange:
                messages.append({
                    'role': 'assistant',
                    'content': exchange['assistant']
                })
        
        return {'messages': messages}
    
    def _update_conversation_history(self, user_message: str, assistant_response: str):
        """Update conversation history."""
        if self.conversation_history and 'assistant' not in self.conversation_history[-1]:
            self.conversation_history[-1]['assistant'] = assistant_response
        else:
            self.conversation_history.append({
                'user': user_message,
                'assistant': assistant_response,
                'timestamp': datetime.now().isoformat(),
                'mode': self.conversation_mode
            })
    
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
        """Handle special commands with enhanced functionality."""
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
        
        elif cmd == '/branch':
            self._handle_branching(args)
        
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
        
        elif cmd == '/checkpoints':
            self._list_checkpoints()
        
        else:
            print(f"Unknown command: {cmd}")
            print("Type /help for available commands")
        
        return True
    
    def _list_checkpoints(self):
        """List available checkpoints."""
        print("\nAVAILABLE CHECKPOINTS")
        print("=" * 50)
        
        if self.checkpoint_manager:
            checkpoints = self.checkpoint_manager.list_checkpoints()
            if checkpoints:
                print(f"Current experiment ({self.config.experiment_name}):")
                for cp in sorted(checkpoints, key=lambda x: x['step'], reverse=True)[:10]:
                    status = " (BEST)" if cp.get('is_best', False) else ""
                    eval_loss = cp.get('eval_loss', 'N/A')
                    print(f"  Epoch {cp['epoch']:3d}, Step {cp['step']:6d}: {eval_loss}{status}")
            else:
                print("No checkpoints found in current experiment")
        
        # Also show global checkpoint search results
        print(f"\nGlobal search results:")
        found_checkpoint = find_best_checkpoint()
        if found_checkpoint:
            print(f"Best available: {found_checkpoint}")
        else:
            print("No checkpoints found globally")
        
        print("=" * 50)
    
    def _show_help(self):
        """Show comprehensive help."""
        print("\nHELP - Advanced Chat Interface")
        print("=" * 60)
        print("BASIC COMMANDS:")
        print("  /help             - Show this help")
        print("  /quit, /exit      - Exit chat")
        print("  /clear            - Clear conversation history")
        print("  /stats            - Show detailed session statistics")
        print("  /checkpoints      - List available checkpoints")
        print("\nCONVERSATION MANAGEMENT:")
        print("  /save [name]      - Save conversation (auto-named if no name)")
        print("  /load [name]      - Load saved conversation")
        print("  /export [format]  - Export conversation (json/txt/md)")
        print("  /branch <name>    - Create conversation branch")
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
        """Save conversation with enhanced metadata."""
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
                'model_config': {
                    'experiment_name': self.config.experiment_name,
                    'hidden_size': self.config.hidden_size,
                    'num_layers': self.config.num_layers,
                    'parameters': estimate_parameters(self.config)
                },
                'system_prompt': self.system_prompt,
                'conversation_mode': self.conversation_mode,
                'conversation': self.conversation_history,
                'session_stats': self.session_stats,
                'system_info': self.system_info
            }
            
            # Create conversations directory
            Path("conversations").mkdir(exist_ok=True)
            filepath = Path("conversations") / f"{name}.json"
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(conversation_data, f, indent=2, default=str)
            
            print(f"Conversation saved: {filepath}")
            
        except Exception as e:
            print(f"Failed to save conversation: {e}")
    
    def _load_conversation(self, name: str = ""):
        """Load conversation with search functionality."""
        conversations_dir = Path("conversations")
        if not conversations_dir.exists():
            print("No conversations directory found")
            return
        
        if name:
            # Load specific conversation
            filepath = conversations_dir / f"{name}.json"
            if not filepath.exists():
                print(f"Conversation '{name}' not found")
                return
            conversation_files = [filepath]
        else:
            # List all conversations
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
        
        # Load the conversation
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
                        f.write(f"User: {exchange['user']}\n")
                        if 'assistant' in exchange:
                            f.write(f"Assistant: {exchange['assistant']}\n")
                        f.write("\n")
            
            elif format_type == "md":
                filepath = Path("exports") / f"chat_{timestamp}.md"
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write("# Chat Conversation\n\n")
                    f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"**Model:** {self.config.experiment_name}\n")
                    f.write(f"**Mode:** {self.conversation_mode}\n\n")
                    
                    for i, exchange in enumerate(self.conversation_history, 1):
                        f.write(f"## Exchange {i}\n\n")
                        f.write(f"**User:** {exchange['user']}\n\n")
                        if 'assistant' in exchange:
                            f.write(f"**Assistant:** {exchange['assistant']}\n\n")
            
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
        print(f"Max new tokens: {self.config.max_new_tokens}")
        
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
            
            # Max new tokens
            new_max_tokens = input(f"Max new tokens ({self.config.max_new_tokens}): ").strip()
            if new_max_tokens and new_max_tokens.isdigit():
                self.config.max_new_tokens = int(new_max_tokens)
            
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
    
    def _handle_branching(self, args: str):
        """Handle conversation branching."""
        print("Conversation branching not fully implemented in this simplified version")
        print("Use /save and /load for conversation management")
    
    def _regenerate_last_response(self):
        """Regenerate the last assistant response."""
        if not self.conversation_history:
            print("No conversation history to regenerate from")
            return
        
        last_exchange = self.conversation_history[-1]
        if 'assistant' not in last_exchange:
            print("No assistant response to regenerate")
            return
        
        user_message = last_exchange['user']
        
        # Remove the assistant response
        del last_exchange['assistant']
        
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
        
        last_exchange = self.conversation_history[-1]
        if 'user' not in last_exchange:
            print("No user message to respond to")
            return
        
        user_message = last_exchange['user']
        print(f"\nGenerating {count} different responses to: '{user_message[:50]}...'")
        
        responses = []
        for i in range(count):
            print(f"\nResponse {i+1}:")
            
            # Temporarily remove last assistant response if exists
            had_assistant = 'assistant' in last_exchange
            if had_assistant:
                old_response = last_exchange['assistant']
                del last_exchange['assistant']
            
            response, metrics = self._generate_response(user_message)
            responses.append(response)
            print(response)
            
            if self.show_analysis:
                print(f"   [Tokens: {metrics.get('output_tokens', 0)}, Time: {metrics.get('response_time', 0):.2f}s]")
        
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
        
        for prompt in test_prompts:
            print(f"Testing: '{prompt[:30]}...'")
            
            # Clear history for clean test
            original_history = self.conversation_history.copy()
            self.conversation_history = [{'user': prompt}]
            
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
        print(f"Model: {self.config.experiment_name}")
        print(f"Parameters: ~{estimate_parameters(self.config):,}")
        print(f"Device: {self.device}")
        print(f"Mode: {self.conversation_mode}")
        print(f"Max tokens per response: {self.config.max_new_tokens}")
        
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
                        last_response = self.conversation_history[-1].get('assistant', '')
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
        description='Enhanced Interactive Chat Interface for Conversational Transformer',
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
  python chat.py --checkpoint latest --auto-save --max-tokens 512
  
  # Benchmark mode for testing
  python chat.py --config debug --benchmark
        """
    )
    
    # Configuration options
    parser.add_argument('--config', choices=['debug', 'small', 'medium', 'large'],
                       default='medium', help='Configuration preset')
    parser.add_argument('--config-file', type=str, help='Load config from YAML file')
    
    # Model options
    parser.add_argument('--checkpoint', type=str, help='Checkpoint to load (best/latest/path)')
    parser.add_argument('--experiment-name', type=str, help='Experiment name override')
    
    # Generation settings
    parser.add_argument('--temperature', type=float, help='Sampling temperature')
    parser.add_argument('--top-p', type=float, help='Top-p sampling threshold')
    parser.add_argument('--top-k', type=int, help='Top-k sampling threshold')
    parser.add_argument('--max-tokens', type=int, help='Maximum tokens to generate')
    
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
        if args.config_file:
            config = Config.load(args.config_file)
        else:
            config_map = {
                'debug': ConfigPresets.debug,
                'small': ConfigPresets.small,
                'medium': ConfigPresets.medium,
                'large': ConfigPresets.large,
            }
            config = config_map[args.config]()
        
        # Apply CLI overrides
        if args.experiment_name:
            config.experiment_name = args.experiment_name
        if args.temperature is not None:
            config.temperature = args.temperature
        if args.top_p is not None:
            config.top_p = args.top_p
        if args.top_k is not None:
            config.top_k = args.top_k
        if args.max_tokens is not None:
            config.max_new_tokens = args.max_tokens
        
        config.validate()
        
    except Exception as e:
        print(f"Configuration error: {e}")
        return 1
    
    # Determine checkpoint path - now required
    checkpoint_path = args.checkpoint
    if not checkpoint_path:
        # Auto-detect checkpoint
        checkpoint_path = find_best_checkpoint()
        if not checkpoint_path:
            print("ERROR: No checkpoint found!")
            print("The chat interface requires a trained model checkpoint to function.")
            print("Please:")
            print("1. Train a model first using train.py")
            print("2. Or specify a checkpoint path with --checkpoint")
            print("3. Or place a checkpoint file in the checkpoints/ directory")
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
        logging.error(f"Chat initialization failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())