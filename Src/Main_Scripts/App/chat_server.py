#!/usr/bin/env python3
"""
Flask server backend for LuminaAI Electron desktop app.
Provides REST API and WebSocket support for neural model inference.
"""

import os
import sys
import logging
import json
import time
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import argparse

# Flask and WebSocket imports
from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import torch

# Add the current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import your existing modules
try:
    from config.config_manager import Config, ConfigPresets
    from core.tokenizer import ConversationTokenizer, TokenizationMode
    from core.model import TransformerModel, estimate_parameters
    from checkpoint import CheckpointManager
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all required modules are present in the correct directory structure.")
    sys.exit(1)


class LuminaAIServer:
    """Flask server backend for LuminaAI desktop application."""
    
    def __init__(self, host='localhost', port=5001):
        self.host = host
        self.port = port
        
        # Initialize Flask app
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'luminaai-secret-key-change-in-production'
        
        # Enable CORS for Electron
        CORS(self.app, origins=['*'])
        
        # Initialize SocketIO
        self.socketio = SocketIO(
            self.app,
            cors_allowed_origins="*",
            async_mode='threading',
            ping_timeout=60,
            ping_interval=25
        )
        
        # Model components
        self.config = None
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_model_loaded = False
        
        # Chat state
        self.conversation_history = []
        
        # System info
        self.system_info = {
            'pytorch_available': self._check_pytorch(),
            'device': str(self.device),
            'model_loaded': False
        }
        
        # Setup routes and handlers
        self._setup_routes()
        self._setup_socketio_handlers()
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('LuminaAI-Server')
    
    def _check_pytorch(self) -> bool:
        """Check if PyTorch is available."""
        try:
            import torch
            return True
        except ImportError:
            return False
    
    def _setup_routes(self):
        """Setup Flask REST API routes."""
        
        @self.app.route('/api/system/status', methods=['GET'])
        def get_system_status():
            """Get system status information."""
            return jsonify({
                'success': True,
                'pytorch_available': self.system_info['pytorch_available'],
                'device': self.system_info['device'],
                'model_loaded': self.is_model_loaded,
                'cuda_available': torch.cuda.is_available(),
                'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
            })
        
        @self.app.route('/api/model/load', methods=['POST'])
        def load_model():
            """Load a PyTorch model from file path."""
            try:
                data = request.get_json()
                model_path = data.get('model_path')
                
                if not model_path or not os.path.exists(model_path):
                    return jsonify({
                        'success': False,
                        'message': f'Model file not found: {model_path}'
                    })
                
                success, message = self._load_model_from_path(model_path)
                
                if success:
                    self.socketio.emit('status', {'model_loaded': True})
                
                return jsonify({
                    'success': success,
                    'message': message
                })
                
            except Exception as e:
                self.logger.error(f"Error loading model: {e}")
                return jsonify({
                    'success': False,
                    'message': f'Error loading model: {str(e)}'
                })
        
        @self.app.route('/api/model/info', methods=['GET'])
        def get_model_info():
            """Get loaded model information."""
            if not self.is_model_loaded:
                return jsonify({
                    'success': False,
                    'message': 'No model loaded'
                })
            
            try:
                model_info = {
                    'vocab_size': getattr(self.config, 'vocab_size', None),
                    'hidden_size': getattr(self.config, 'hidden_size', None),
                    'num_layers': getattr(self.config, 'num_layers', None),
                    'num_heads': getattr(self.config, 'num_heads', None),
                    'seq_length': getattr(self.config, 'seq_length', None),
                    'device': str(self.device),
                    'parameters': self._count_parameters(),
                    'model_type': self.model.__class__.__name__ if self.model else None
                }
                
                return jsonify({
                    'success': True,
                    'model_info': model_info
                })
                
            except Exception as e:
                return jsonify({
                    'success': False,
                    'message': f'Error getting model info: {str(e)}'
                })
        
        @self.app.route('/api/chat', methods=['POST'])
        def chat():
            """Handle chat message and generate response."""
            if not self.is_model_loaded:
                return jsonify({
                    'success': False,
                    'message': 'No model loaded'
                })
            
            try:
                data = request.get_json()
                message = data.get('message', '').strip()
                
                if not message:
                    return jsonify({
                        'success': False,
                        'message': 'Empty message'
                    })
                
                # Generation settings
                settings = {
                    'temperature': data.get('temperature', 0.8),
                    'top_k': data.get('top_k', 50),
                    'top_p': data.get('top_p', 0.9),
                    'max_length': data.get('max_length', 150),
                    'sampling_method': data.get('sampling_method', 'top_k')
                }
                
                # Generate response
                response = self._generate_response(message, settings)
                
                return jsonify({
                    'success': True,
                    'response': response
                })
                
            except Exception as e:
                self.logger.error(f"Error in chat: {e}")
                return jsonify({
                    'success': False,
                    'message': f'Error generating response: {str(e)}'
                })
        
        @self.app.route('/api/chat/clear', methods=['POST'])
        def clear_chat():
            """Clear conversation history."""
            self.conversation_history = []
            return jsonify({
                'success': True,
                'message': 'Chat history cleared'
            })
        
        @self.app.route('/api/health', methods=['GET'])
        def health_check():
            """Health check endpoint."""
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'model_loaded': self.is_model_loaded
            })
    
    def _setup_socketio_handlers(self):
        """Setup WebSocket event handlers."""
        
        @self.socketio.on('connect')
        def handle_connect():
            self.logger.info('Client connected')
            emit('status', {
                'model_loaded': self.is_model_loaded,
                'device': str(self.device)
            })
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            self.logger.info('Client disconnected')
        
        @self.socketio.on('chat_message')
        def handle_chat_message(data):
            """Handle real-time chat message via WebSocket."""
            if not self.is_model_loaded:
                emit('generation_error', {'message': 'No model loaded'})
                return
            
            try:
                message = data.get('message', '').strip()
                if not message:
                    emit('generation_error', {'message': 'Empty message'})
                    return
                
                # Emit typing start
                emit('typing_start')
                
                # Generation settings
                settings = data.get('settings', {})
                
                # Generate response in background thread
                def generate_async():
                    try:
                        response = self._generate_response(message, settings)
                        self.socketio.emit('typing_stop')
                        self.socketio.emit('message_generated', {'response': response})
                    except Exception as e:
                        self.socketio.emit('typing_stop')
                        self.socketio.emit('generation_error', {'message': str(e)})
                
                thread = threading.Thread(target=generate_async)
                thread.daemon = True
                thread.start()
                
            except Exception as e:
                emit('generation_error', {'message': str(e)})
    
    def _load_model_from_path(self, model_path: str) -> tuple[bool, str]:
        """Load model from file path."""
        try:
            self.logger.info(f"Loading model from: {model_path}")
            
            # Initialize default config
            self.config = ConfigPresets.medium()
            
            # Load checkpoint
            checkpoint_path = Path(model_path)
            if not checkpoint_path.exists():
                return False, f"Model file not found: {model_path}"
            
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                # Enhanced checkpoint format
                if 'model_config' in checkpoint:
                    # Update config from checkpoint
                    model_config = checkpoint['model_config']
                    for key, value in model_config.items():
                        if hasattr(self.config, key):
                            setattr(self.config, key, value)
                
                model_state_dict = checkpoint['model_state_dict']
                epoch = checkpoint.get('current_epoch', 'Unknown')
                loss = checkpoint.get('best_loss', checkpoint.get('loss', 'Unknown'))
            else:
                # Assume the checkpoint is just the state dict
                model_state_dict = checkpoint
                epoch = 'Unknown'
                loss = 'Unknown'
            
            # Initialize tokenizer
            try:
                self.tokenizer = ConversationTokenizer(
                    model_name="gpt-4",
                    max_context_length=self.config.seq_length,
                    enable_caching=True,
                    thread_safe=True
                )
                self.config.vocab_size = self.tokenizer.vocab_size
                self.logger.info(f"Tokenizer initialized (vocab size: {self.tokenizer.vocab_size:,})")
            except Exception as e:
                self.logger.error(f"Failed to initialize tokenizer: {e}")
                return False, f"Failed to initialize tokenizer: {str(e)}"
            
            # Initialize model
            try:
                self.model = TransformerModel(self.config).to(self.device)
                
                # Load state dict
                self.model.load_state_dict(model_state_dict, strict=False)
                self.model.eval()
                
                param_count = self._count_parameters()
                self.logger.info(f"Model loaded successfully (~{param_count:,} parameters)")
                
                self.is_model_loaded = True
                self.system_info['model_loaded'] = True
                
                return True, f"Model loaded successfully (Epoch: {epoch}, Loss: {loss})"
                
            except Exception as e:
                self.logger.error(f"Failed to initialize model: {e}")
                return False, f"Failed to initialize model: {str(e)}"
                
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return False, f"Error loading model: {str(e)}"
    
    def _count_parameters(self) -> int:
        """Count model parameters."""
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters())
    
    def _generate_response(self, user_message: str, settings: Dict[str, Any]) -> str:
        """Generate response using the loaded model."""
        if not self.is_model_loaded:
            raise Exception("No model loaded")
        
        try:
            # Add user message to history
            self.conversation_history.append({'role': 'user', 'content': user_message})
            
            # Format conversation for model
            conversation = self._format_conversation()
            
            # Tokenize input
            input_tokens = self.tokenizer.encode_conversation(conversation)
            
            if not input_tokens:
                raise Exception("Failed to tokenize input")
            
            # Convert to tensor
            input_ids = torch.tensor([input_tokens], dtype=torch.long, device=self.device)
            
            # Truncate if too long
            max_input_length = self.config.seq_length - settings.get('max_length', 150)
            if input_ids.shape[1] > max_input_length:
                input_ids = input_ids[:, -max_input_length:]
            
            # Generate response
            generated_tokens = self._generate_tokens(input_ids, settings)
            
            # Decode response
            if generated_tokens:
                response = self.tokenizer.decode(
                    generated_tokens,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
            else:
                response = "I'm sorry, I couldn't generate a response."
            
            # Add to conversation history
            self.conversation_history.append({'role': 'assistant', 'content': response})
            
            # Keep history manageable (last 20 exchanges)
            if len(self.conversation_history) > 40:  # 20 user + 20 assistant messages
                self.conversation_history = self.conversation_history[-40:]
            
            return response
            
        except Exception as e:
            self.logger.error(f"Generation error: {e}")
            raise Exception(f"Generation failed: {str(e)}")
    
    def _generate_tokens(self, input_ids: torch.Tensor, settings: Dict[str, Any]) -> List[int]:
        """Generate tokens using the model."""
        self.model.eval()
        generated = []
        
        # Extract settings
        temperature = settings.get('temperature', 0.8)
        top_k = settings.get('top_k', 50)
        top_p = settings.get('top_p', 0.9)
        max_length = settings.get('max_length', 150)
        sampling_method = settings.get('sampling_method', 'top_k')
        
        with torch.no_grad():
            current_ids = input_ids.clone()
            
            for _ in range(max_length):
                # Forward pass
                logits = self.model(current_ids)
                next_token_logits = logits[0, -1, :]  # Last token predictions
                
                # Apply sampling based on method
                if sampling_method == 'greedy':
                    next_token = torch.argmax(next_token_logits).item()
                else:
                    # Apply temperature
                    if temperature != 1.0:
                        next_token_logits = next_token_logits / temperature
                    
                    # Apply top-k filtering
                    if sampling_method == 'top_k' and top_k > 0:
                        top_k_actual = min(top_k, next_token_logits.size(-1))
                        indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k_actual)[0][..., -1, None]
                        next_token_logits[indices_to_remove] = -float('Inf')
                    
                    # Apply nucleus (top-p) sampling
                    elif sampling_method == 'nucleus' and top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                        
                        # Remove tokens with cumulative probability above threshold
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        
                        indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
                        next_token_logits[indices_to_remove] = -float('Inf')
                    
                    # Sample next token
                    probs = torch.softmax(next_token_logits, dim=-1)
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
    
    def _format_conversation(self) -> Dict[str, Any]:
        """Format conversation history for model input."""
        messages = []
        
        # Convert conversation history to messages format
        for exchange in self.conversation_history[-20:]:  # Last 20 messages
            messages.append({
                'role': exchange['role'],
                'content': exchange['content']
            })
        
        return {'messages': messages}
    
    def run(self, debug=False):
        """Run the Flask server."""
        self.logger.info(f"Starting LuminaAI server on {self.host}:{self.port}")
        
        if debug:
            self.logger.info("Running in debug mode")
        
        try:
            self.socketio.run(
                self.app,
                host=self.host,
                port=self.port,
                debug=debug,
                use_reloader=False  # Disable reloader to prevent issues with threading
            )
        except Exception as e:
            self.logger.error(f"Failed to start server: {e}")
            raise


def create_server_from_args(args) -> LuminaAIServer:
    """Create server instance from command line arguments."""
    server = LuminaAIServer(host=args.host, port=args.port)
    
    # Auto-load model if specified
    if args.model_path:
        success, message = server._load_model_from_path(args.model_path)
        if success:
            server.logger.info(f"Model loaded: {message}")
        else:
            server.logger.error(f"Failed to load model: {message}")
    
    return server


def main():
    """Main entry point for server mode."""
    parser = argparse.ArgumentParser(
        description='LuminaAI Flask Server Backend',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--host', default='localhost', help='Server host (default: localhost)')
    parser.add_argument('--port', type=int, default=5001, help='Server port (default: 5001)')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    parser.add_argument('--model-path', type=str, help='Auto-load model from path')
    parser.add_argument('--server-mode', action='store_true', help='Run in server mode (for Electron)')
    
    args = parser.parse_args()
    
    # Create and run server
    try:
        server = create_server_from_args(args)
        
        print(f"🚀 LuminaAI Backend Server starting...")
        print(f"📡 Host: {args.host}")
        print(f"🔌 Port: {args.port}")
        print(f"💻 Device: {server.device}")
        print(f"🔥 PyTorch: {torch.__version__}")
        print("=" * 50)
        print("Backend ready for Electron app connection")
        
        server.run(debug=args.debug)
        
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Server error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())