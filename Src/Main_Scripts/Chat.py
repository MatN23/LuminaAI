# Copyright (c) 2025 Matias Nielsen. All rights reserved.
# Licensed under the Custom License below.

import os
import sys
import logging
import argparse
import signal
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

# Add the current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import our modules
try:
    from config.config_manager import Config, ConfigPresets
    from core.tokenizer import ConversationTokenizer
    from core.model import TransformerModel, estimate_parameters
    from training.trainer import EnhancedConversationTrainer
    from monitoring.logger import ProductionLogger
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all required modules are present in the correct directory structure.")
    sys.exit(1)

import torch


class ConversationalChat:
    """Interactive chat interface for conversational transformer models."""
    
    def __init__(self, config: Config, checkpoint_path: Optional[str] = None):
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.conversation_history = []
        self.session_stats = {
            'messages_sent': 0,
            'messages_received': 0,
            'session_start': datetime.now(),
            'total_tokens_generated': 0,
            'avg_response_time': 0.0
        }
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
        # Chat settings
        self.max_history_length = 10  # Keep last 10 exchanges
        self.show_token_count = False
        self.show_timing = False
        self.system_prompt = None
        
        # Initialize the chat system
        self._initialize_chat_system()
    
    def _initialize_chat_system(self):
        """Initialize tokenizer, model, and trainer."""
        print("🤖 Initializing chat system...")
        
        try:
            # Initialize tokenizer
            self.tokenizer = ConversationTokenizer()
            self.config.vocab_size = self.tokenizer.vocab_size
            print(f"✅ Tokenizer loaded (vocab size: {self.tokenizer.vocab_size:,})")
            
            # Initialize model
            self.model = TransformerModel(self.config)
            param_count = estimate_parameters(self.config)
            print(f"✅ Model initialized (~{param_count:,} parameters)")
            
            # Initialize trainer
            logger = ProductionLogger(log_level="WARNING")  # Reduce logging noise
            self.trainer = EnhancedConversationTrainer(
                self.model, self.tokenizer, self.config, logger
            )
            
            # Load checkpoint if provided
            if self.checkpoint_path:
                self._load_checkpoint()
            else:
                print("⚠️  No checkpoint loaded - using randomly initialized model")
            
            print("🚀 Chat system ready!")
            
        except Exception as e:
            print(f"❌ Failed to initialize chat system: {e}")
            raise
    
    def _load_checkpoint(self):
        """Load model checkpoint."""
        try:
            checkpoint_path = Path(self.checkpoint_path)
            
            # Handle special keywords
            if self.checkpoint_path == "best":
                best_checkpoint = self.trainer.checkpoint_manager.get_best_checkpoint()
                if best_checkpoint:
                    checkpoint_path = best_checkpoint
                else:
                    print("❌ No best checkpoint found")
                    return
            elif self.checkpoint_path == "latest":
                latest_checkpoint = self.trainer.checkpoint_manager.get_latest_checkpoint()
                if latest_checkpoint:
                    checkpoint_path = latest_checkpoint
                else:
                    print("❌ No latest checkpoint found")
                    return
            
            if not checkpoint_path.exists():
                print(f"❌ Checkpoint not found: {checkpoint_path}")
                return
            
            print(f"📂 Loading checkpoint: {checkpoint_path}")
            epoch = self.trainer.load_checkpoint(str(checkpoint_path))
            print(f"✅ Checkpoint loaded successfully (epoch {epoch})")
            
        except Exception as e:
            print(f"❌ Failed to load checkpoint: {e}")
            print("⚠️  Continuing with randomly initialized model")
    
    def _format_conversation_for_model(self, include_system: bool = True) -> Dict[str, Any]:
        """Format conversation history for model input."""
        messages = []
        
        # Add system prompt if provided and requested
        if include_system and self.system_prompt:
            messages.append({
                'role': 'system',
                'content': self.system_prompt
            })
        
        # Add conversation history (keep last N exchanges)
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
            # Complete the last exchange
            self.conversation_history[-1]['assistant'] = assistant_response
        else:
            # Start new exchange
            self.conversation_history.append({
                'user': user_message,
                'assistant': assistant_response
            })
        
        # Update stats
        self.session_stats['messages_sent'] += 1
        self.session_stats['messages_received'] += 1
    
    def generate_response(self, user_input: str) -> tuple[str, Dict[str, Any]]:
        """Generate response to user input."""
        import time
        
        start_time = time.time()
        
        try:
            # Add user message to history
            if not self.conversation_history or 'assistant' in self.conversation_history[-1]:
                self.conversation_history.append({'user': user_input})
            
            # Format conversation for model
            conversation = self._format_conversation_for_model()
            
            # Generate response
            response = self.trainer.generate(
                user_input,  # We pass just the current input to the generate method
                max_new_tokens=self.config.max_new_tokens
            )
            
            # Calculate metrics
            response_time = time.time() - start_time
            token_count = len(self.tokenizer.tokenizer.encode(response)) if response else 0
            
            # Update history and stats
            self._update_conversation_history(user_input, response)
            self.session_stats['total_tokens_generated'] += token_count
            self.session_stats['avg_response_time'] = (
                (self.session_stats['avg_response_time'] * (self.session_stats['messages_received'] - 1) + response_time)
                / self.session_stats['messages_received']
            )
            
            metrics = {
                'response_time': response_time,
                'token_count': token_count,
                'conversation_length': len(self.conversation_history)
            }
            
            return response, metrics
            
        except Exception as e:
            logging.error(f"Generation failed: {e}")
            return f"I apologize, but I encountered an error: {str(e)}", {}
    
    def _print_header(self):
        """Print chat header."""
        print("\n" + "="*80)
        print("🤖 CONVERSATIONAL TRANSFORMER CHAT")
        print("="*80)
        print(f"Model: {self.config.experiment_name}")
        print(f"Parameters: ~{estimate_parameters(self.config):,}")
        print(f"Max tokens per response: {self.config.max_new_tokens}")
        print(f"Temperature: {self.config.temperature}")
        print(f"Top-p: {self.config.top_p}")
        if self.checkpoint_path:
            print(f"Checkpoint: {self.checkpoint_path}")
        print("\nCommands:")
        print("  /help          - Show help")
        print("  /stats         - Show session statistics")
        print("  /clear         - Clear conversation history") 
        print("  /save          - Save conversation")
        print("  /load          - Load conversation")
        print("  /settings      - Show/modify settings")
        print("  /system <msg>  - Set system prompt")
        print("  /quit or /exit - Exit chat")
        print("="*80)
        print("💬 Start chatting! (Type your message and press Enter)\n")
    
    def _handle_command(self, command: str) -> bool:
        """Handle special commands. Returns True if should continue, False if should exit."""
        command = command.strip().lower()
        
        if command in ['/quit', '/exit']:
            return False
        
        elif command == '/help':
            self._show_help()
        
        elif command == '/stats':
            self._show_stats()
        
        elif command == '/clear':
            self._clear_history()
        
        elif command == '/save':
            self._save_conversation()
        
        elif command == '/load':
            self._load_conversation()
        
        elif command == '/settings':
            self._show_settings()
        
        elif command.startswith('/system '):
            system_msg = command[8:].strip()
            if system_msg:
                self.system_prompt = system_msg
                print(f"✅ System prompt set: {system_msg}")
            else:
                self.system_prompt = None
                print("✅ System prompt cleared")
        
        else:
            print(f"❌ Unknown command: {command}")
            print("Type /help for available commands")
        
        return True
    
    def _show_help(self):
        """Show help information."""
        print("\n📖 HELP")
        print("-" * 40)
        print("Commands:")
        print("  /help          - Show this help")
        print("  /stats         - Show session statistics")
        print("  /clear         - Clear conversation history")
        print("  /save          - Save current conversation")
        print("  /load          - Load saved conversation")
        print("  /settings      - Show/modify chat settings")
        print("  /system <msg>  - Set system prompt")
        print("  /quit, /exit   - Exit chat")
        print("\nTips:")
        print("  • Conversation history is automatically maintained")
        print("  • The model sees the last 10 exchanges by default")
        print("  • Use /clear to start fresh if context becomes confusing")
        print("  • System prompts help guide the model's behavior")
        print("-" * 40)
    
    def _show_stats(self):
        """Show session statistics."""
        duration = datetime.now() - self.session_stats['session_start']
        
        print("\n📊 SESSION STATISTICS")
        print("-" * 40)
        print(f"Session duration: {duration}")
        print(f"Messages sent: {self.session_stats['messages_sent']}")
        print(f"Messages received: {self.session_stats['messages_received']}")
        print(f"Total tokens generated: {self.session_stats['total_tokens_generated']:,}")
        print(f"Average response time: {self.session_stats['avg_response_time']:.2f}s")
        print(f"Conversation exchanges: {len(self.conversation_history)}")
        if self.system_prompt:
            print(f"System prompt: {self.system_prompt[:50]}...")
        print("-" * 40)
    
    def _clear_history(self):
        """Clear conversation history."""
        self.conversation_history.clear()
        print("✅ Conversation history cleared")
    
    def _save_conversation(self):
        """Save conversation to file."""
        if not self.conversation_history:
            print("❌ No conversation to save")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"conversation_{timestamp}.json"
        
        try:
            conversation_data = {
                'timestamp': timestamp,
                'model_config': self.config.experiment_name,
                'system_prompt': self.system_prompt,
                'conversation': self.conversation_history,
                'stats': self.session_stats
            }
            
            # Create conversations directory if it doesn't exist
            Path("conversations").mkdir(exist_ok=True)
            filepath = Path("conversations") / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(conversation_data, f, indent=2, default=str)
            
            print(f"✅ Conversation saved: {filepath}")
            
        except Exception as e:
            print(f"❌ Failed to save conversation: {e}")
    
    def _load_conversation(self):
        """Load conversation from file."""
        conversations_dir = Path("conversations")
        if not conversations_dir.exists():
            print("❌ No conversations directory found")
            return
        
        # List available conversations
        conversation_files = list(conversations_dir.glob("conversation_*.json"))
        if not conversation_files:
            print("❌ No saved conversations found")
            return
        
        print("\n📁 Available conversations:")
        for i, filepath in enumerate(conversation_files, 1):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                timestamp = data.get('timestamp', 'Unknown')
                exchanges = len(data.get('conversation', []))
                print(f"  {i}. {filepath.name} - {timestamp} ({exchanges} exchanges)")
            except Exception:
                print(f"  {i}. {filepath.name} - (corrupted file)")
        
        try:
            choice = input("\nEnter number to load (or Enter to cancel): ").strip()
            if not choice:
                return
            
            file_index = int(choice) - 1
            if 0 <= file_index < len(conversation_files):
                filepath = conversation_files[file_index]
                
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                self.conversation_history = data.get('conversation', [])
                if data.get('system_prompt'):
                    self.system_prompt = data['system_prompt']
                
                print(f"✅ Conversation loaded: {filepath.name}")
                print(f"   Exchanges: {len(self.conversation_history)}")
                if self.system_prompt:
                    print(f"   System prompt: {self.system_prompt[:50]}...")
            else:
                print("❌ Invalid selection")
                
        except (ValueError, KeyError, FileNotFoundError) as e:
            print(f"❌ Failed to load conversation: {e}")
    
    def _show_settings(self):
        """Show and optionally modify settings."""
        print("\n⚙️  CURRENT SETTINGS")
        print("-" * 40)
        print(f"Max history length: {self.max_history_length}")
        print(f"Show token count: {self.show_token_count}")
        print(f"Show timing: {self.show_timing}")
        print(f"Temperature: {self.config.temperature}")
        print(f"Top-p: {self.config.top_p}")
        print(f"Top-k: {self.config.top_k}")
        print(f"Max new tokens: {self.config.max_new_tokens}")
        print("-" * 40)
        
        modify = input("Modify settings? (y/N): ").strip().lower()
        if modify in ['y', 'yes']:
            self._modify_settings()
    
    def _modify_settings(self):
        """Modify chat settings."""
        print("\nModify settings (press Enter to keep current value):")
        
        try:
            # Max history length
            new_history = input(f"Max history length ({self.max_history_length}): ").strip()
            if new_history and new_history.isdigit():
                self.max_history_length = int(new_history)
            
            # Show token count
            new_tokens = input(f"Show token count ({self.show_token_count}) [y/n]: ").strip().lower()
            if new_tokens in ['y', 'yes', 'true']:
                self.show_token_count = True
            elif new_tokens in ['n', 'no', 'false']:
                self.show_token_count = False
            
            # Show timing
            new_timing = input(f"Show timing ({self.show_timing}) [y/n]: ").strip().lower()
            if new_timing in ['y', 'yes', 'true']:
                self.show_timing = True
            elif new_timing in ['n', 'no', 'false']:
                self.show_timing = False
            
            # Temperature
            new_temp = input(f"Temperature ({self.config.temperature}): ").strip()
            if new_temp:
                try:
                    self.config.temperature = float(new_temp)
                except ValueError:
                    print("❌ Invalid temperature value")
            
            # Top-p
            new_top_p = input(f"Top-p ({self.config.top_p}): ").strip()
            if new_top_p:
                try:
                    self.config.top_p = float(new_top_p)
                except ValueError:
                    print("❌ Invalid top-p value")
            
            # Max new tokens
            new_max_tokens = input(f"Max new tokens ({self.config.max_new_tokens}): ").strip()
            if new_max_tokens and new_max_tokens.isdigit():
                self.config.max_new_tokens = int(new_max_tokens)
            
            print("✅ Settings updated")
            
        except KeyboardInterrupt:
            print("\n❌ Settings modification cancelled")
    
    def run_chat(self):
        """Run the interactive chat loop."""
        self._print_header()
        
        # Setup signal handler for graceful exit
        def signal_handler(signum, frame):
            print("\n\n👋 Chat session ended by user")
            self._show_stats()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        
        try:
            while True:
                # Get user input
                try:
                    user_input = input("👤 You: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\n\n👋 Chat session ended")
                    break
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    if not self._handle_command(user_input):
                        break
                    continue
                
                # Generate response
                print("🤖 Assistant: ", end="", flush=True)
                
                response, metrics = self.generate_response(user_input)
                print(response)
                
                # Show metrics if enabled
                if self.show_token_count or self.show_timing:
                    info_parts = []
                    if self.show_timing and 'response_time' in metrics:
                        info_parts.append(f"⏱️ {metrics['response_time']:.2f}s")
                    if self.show_token_count and 'token_count' in metrics:
                        info_parts.append(f"🔤 {metrics['token_count']} tokens")
                    
                    if info_parts:
                        print(f"   {' | '.join(info_parts)}")
                
                print()  # Extra line for readability
        
        except Exception as e:
            print(f"\n❌ Chat error: {e}")
        
        finally:
            print("\n📊 Final Statistics:")
            self._show_stats()
            print("\n👋 Goodbye!")


def main():
    """Main function for the chat script."""
    parser = argparse.ArgumentParser(
        description='Interactive Chat Interface for Conversational Transformer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Chat with a trained model
  python chat.py --checkpoint checkpoints/best_checkpoint.pt
  
  # Chat with specific configuration
  python chat.py --config small --checkpoint latest
  
  # Chat with random model (for testing)
  python chat.py --config debug
  
  # Chat with custom generation settings
  python chat.py --checkpoint best --temperature 0.7 --top-p 0.9 --max-tokens 256
        """
    )
    
    # Configuration options
    parser.add_argument('--config', choices=['debug', 'small', 'medium', 'large'],
                       default='small', help='Configuration preset')
    parser.add_argument('--config-file', type=str, help='Load config from YAML file')
    
    # Model options
    parser.add_argument('--checkpoint', type=str, help='Checkpoint to load (best/latest/path)')
    parser.add_argument('--experiment-name', type=str, help='Experiment name')
    
    # Generation settings
    parser.add_argument('--temperature', type=float, help='Sampling temperature')
    parser.add_argument('--top-p', type=float, help='Top-p sampling threshold')
    parser.add_argument('--top-k', type=int, help='Top-k sampling threshold')
    parser.add_argument('--max-tokens', type=int, help='Maximum tokens to generate')
    
    # Chat settings
    parser.add_argument('--system-prompt', type=str, help='Initial system prompt')
    parser.add_argument('--show-tokens', action='store_true', help='Show token counts')
    parser.add_argument('--show-timing', action='store_true', help='Show response timing')
    parser.add_argument('--max-history', type=int, default=10, help='Max conversation history length')
    
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
        
        # Validate configuration
        config.validate()
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return 1
    
    # Initialize and run chat
    try:
        chat = ConversationalChat(config, args.checkpoint)
        
        # Apply chat settings
        if args.system_prompt:
            chat.system_prompt = args.system_prompt
        chat.show_token_count = args.show_tokens
        chat.show_timing = args.show_timing
        chat.max_history_length = args.max_history
        
        # Run the chat
        chat.run_chat()
        
        return 0
        
    except KeyboardInterrupt:
        print("\n👋 Chat interrupted by user")
        return 0
    except Exception as e:
        print(f"❌ Chat failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())