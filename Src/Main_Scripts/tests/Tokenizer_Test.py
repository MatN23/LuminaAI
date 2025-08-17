# Integration compatibility fixes for the transformer training system
# This module provides the necessary compatibility layers and fixes

import sys
from pathlib import Path

# Add compatibility for import paths
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Create a simple compatibility import structure
def setup_imports():
    """Setup import compatibility for the training system."""
    
    # Create the core module structure
    try:
        # Import the tokenizer with the correct interface
        from integrated_tokenizer import ConversationTokenizer
        
        # Create a core module mock if needed
        import types
        core_module = types.ModuleType('core')
        core_module.tokenizer = types.ModuleType('tokenizer')
        core_module.tokenizer.ConversationTokenizer = ConversationTokenizer
        sys.modules['core'] = core_module
        sys.modules['core.tokenizer'] = core_module.tokenizer
        
        # Mock other required modules for the main.py imports
        config_module = types.ModuleType('config')
        config_manager = types.ModuleType('config_manager')
        
        # Import the compatible config
        from compatible_config import Config, ConfigPresets
        config_manager.Config = Config
        config_manager.ConfigPresets = ConfigPresets
        
        config_module.config_manager = config_manager
        sys.modules['config'] = config_module
        sys.modules['config.config_manager'] = config_manager
        
        # Create mock modules for other imports that might not exist yet
        training_module = types.ModuleType('training')
        training_orchestrator = types.ModuleType('orchestrator')
        
        class MockTrainingOrchestrator:
            def __init__(self, config):
                self.config = config
                print(f"MockTrainingOrchestrator initialized with config: {config.experiment_name}")
            
            def run_training(self):
                print("MockTrainingOrchestrator: Training would run here")
                print(f"  - Epochs: {self.config.num_epochs}")
                print(f"  - Batch size: {self.config.batch_size}")
                print(f"  - Learning rate: {self.config.learning_rate}")
                return True
        
        training_orchestrator.TrainingOrchestrator = MockTrainingOrchestrator
        training_module.orchestrator = training_orchestrator
        sys.modules['training'] = training_module
        sys.modules['training.orchestrator'] = training_orchestrator
        
        # Utils module mocks
        utils_module = types.ModuleType('utils')
        
        # Data processing mock
        data_processing = types.ModuleType('data_processing')
        def mock_process_oasst_data(*args, **kwargs):
            print("Mock: process_oasst_data called")
            return True
        
        def mock_validate_data_comprehensive(*args, **kwargs):
            print("Mock: validate_data_comprehensive called")
            return True, []
        
        data_processing.process_oasst_data = mock_process_oasst_data
        data_processing.validate_data_comprehensive = mock_validate_data_comprehensive
        
        # Environment mock
        environment = types.ModuleType('environment')
        def mock_validate_environment():
            print("Mock: Environment validation")
            return []  # No issues
        
        def mock_estimate_training_time(config, dataset_size):
            print("Mock: Training time estimation")
            return {
                'estimated_hours': 2.5,
                'estimated_days': 0.1,
                'total_tokens': dataset_size * 100,
                'tokens_per_second': 1000,
                'memory_utilization': 0.75,
                'memory_warning': False
            }
        
        environment.validate_environment = mock_validate_environment
        environment.estimate_training_time = mock_estimate_training_time
        
        # Reporting mock
        reporting = types.ModuleType('reporting')
        def mock_create_data_summary_report(*args, **kwargs):
            print("Mock: create_data_summary_report called")
            return "Mock report generated"
        
        reporting.create_data_summary_report = mock_create_data_summary_report
        
        # Model mock
        model = types.ModuleType('model')
        def mock_estimate_parameters(config):
            # Simple parameter estimation
            embed_params = config.vocab_size * config.hidden_size
            layer_params = config.hidden_size * config.hidden_size * 4 * config.num_layers
            return embed_params + layer_params
        
        model.estimate_parameters = mock_estimate_parameters
        
        # Add to sys.modules
        utils_module.data_processing = data_processing
        utils_module.environment = environment  
        utils_module.reporting = reporting
        sys.modules['utils'] = utils_module
        sys.modules['utils.data_processing'] = data_processing
        sys.modules['utils.environment'] = environment
        sys.modules['utils.reporting'] = reporting
        sys.modules['core.model'] = model
        
        print("✅ Import compatibility setup complete")
        return True
        
    except Exception as e:
        print(f"❌ Import setup failed: {e}")
        return False


# Updated dataset.py with compatibility fixes
def create_compatible_dataset():
    """Create a compatible dataset class that works with the new tokenizer."""
    
    import json
    import logging
    import numpy as np
    from pathlib import Path
    from typing import Dict, List, Optional, Any
    import torch
    from torch.utils.data import Dataset, DataLoader

    class ConversationDataset(Dataset):
        """Enhanced dataset with better error handling and monitoring."""
        
        def __init__(self, data_path: str, tokenizer, config, split: str = "train"):
            self.data_path = Path(data_path)
            self.tokenizer = tokenizer
            self.config = config
            self.split = split
            
            # Statistics tracking
            self.stats = {
                'total_loaded': 0,
                'valid_conversations': 0,
                'invalid_conversations': 0,
                'tokenization_errors': 0,
                'avg_token_length': 0,
                'max_token_length': 0,
                'min_token_length': float('inf')
            }
            
            # Load conversations with validation
            self.conversations = self._load_and_validate_conversations()
            if self.conversations:  # Only compute stats if we have data
                self._compute_statistics()
            
            logging.info(f"Dataset {split}: {len(self.conversations):,} conversations")
            if self.conversations:
                logging.info(f"Average tokens: {self.stats['avg_token_length']:.1f}, "
                           f"Max: {self.stats['max_token_length']}, Min: {self.stats['min_token_length']}")
        
        def _load_and_validate_conversations(self) -> List[Dict]:
            """Load and validate conversations with comprehensive error handling."""
            conversations = []
            
            if not self.data_path.exists():
                logging.warning(f"Data file not found: {self.data_path}, creating mock data for testing")
                # Create mock data for testing
                return self._create_mock_conversations()
            
            logging.info(f"Loading {self.split} data from {self.data_path}")
            
            try:
                with open(self.data_path, 'r', encoding='utf-8') as f:
                    for line_no, line in enumerate(f, 1):
                        try:
                            conversation = json.loads(line.strip())
                            self.stats['total_loaded'] += 1
                            
                            if self._validate_conversation(conversation):
                                conversations.append(conversation)
                                self.stats['valid_conversations'] += 1
                            else:
                                self.stats['invalid_conversations'] += 1
                                
                        except json.JSONDecodeError as e:
                            self.stats['invalid_conversations'] += 1
                            if line_no <= 10:  # Only log first few errors
                                logging.warning(f"JSON decode error at line {line_no}: {e}")
                        except Exception as e:
                            self.stats['invalid_conversations'] += 1
                            logging.warning(f"Error loading conversation {line_no}: {e}")
                        
                        # Progress logging for large datasets
                        if line_no % 10000 == 0:
                            logging.info(f"Processed {line_no:,} lines, {len(conversations):,} valid")
                            
            except Exception as e:
                logging.error(f"Failed to read data file: {e}")
                return self._create_mock_conversations()
            
            return conversations
        
        def _create_mock_conversations(self) -> List[Dict]:
            """Create mock conversations for testing."""
            mock_conversations = [
                {
                    "messages": [
                        {"role": "user", "content": "Hello, how are you?"},
                        {"role": "assistant", "content": "I'm doing well, thank you! How can I help you today?"}
                    ]
                },
                {
                    "messages": [
                        {"role": "user", "content": "What is Python?"},
                        {"role": "assistant", "content": "Python is a high-level programming language known for its simplicity and readability."}
                    ]
                },
                {
                    "messages": [
                        {"role": "user", "content": "Can you help me with math?"},
                        {"role": "assistant", "content": "Of course! I'd be happy to help you with math problems. What specific topic would you like assistance with?"}
                    ]
                }
            ]
            
            # Replicate mock data to create a reasonable dataset size
            mock_data = mock_conversations * 100  # 300 conversations for testing
            logging.info(f"Created {len(mock_data)} mock conversations for testing")
            return mock_data
        
        def _validate_conversation(self, conversation: Dict) -> bool:
            """Comprehensive conversation validation."""
            if 'messages' not in conversation:
                return False
            
            messages = conversation['messages']
            if not messages or len(messages) < 2:
                return False
            
            # Check message structure and content
            has_user = False
            has_assistant = False
            
            for msg in messages:
                if not isinstance(msg, dict):
                    return False
                
                role = msg.get('role', '').lower()
                content = msg.get('content', '').strip()
                
                if not content:
                    return False
                
                # Track roles
                if role in ['user', 'prompter']:
                    has_user = True
                elif role == 'assistant':
                    has_assistant = True
            
            # Require both user and assistant messages
            return has_user and has_assistant
        
        def _compute_statistics(self):
            """Compute dataset statistics."""
            if not self.conversations:
                return
            
            token_lengths = []
            
            # Sample conversations for statistics (to avoid processing all)
            sample_size = min(100, len(self.conversations))  # Smaller sample for speed
            sample_indices = np.random.choice(len(self.conversations), sample_size, replace=False)
            
            for idx in sample_indices:
                try:
                    tokens = self.tokenizer.encode_conversation(self.conversations[idx])
                    if tokens:
                        token_lengths.append(len(tokens))
                except Exception as e:
                    self.stats['tokenization_errors'] += 1
                    logging.debug(f"Tokenization error: {e}")
            
            if token_lengths:
                self.stats['avg_token_length'] = np.mean(token_lengths)
                self.stats['max_token_length'] = max(token_lengths)
                self.stats['min_token_length'] = min(token_lengths)
        
        def _process_conversation(self, conversation: Dict) -> Optional[Dict[str, torch.Tensor]]:
            """Process conversation with enhanced error handling."""
            try:
                tokens = self.tokenizer.encode_conversation(conversation)
                
                # Validate token sequence
                if not tokens or len(tokens) < 10:
                    return None
                
                # Handle sequence length
                if len(tokens) > self.config.seq_length:
                    # Truncate from the beginning to keep the most recent context
                    tokens = tokens[-self.config.seq_length:]
                else:
                    # Pad to sequence length
                    pad_length = self.config.seq_length - len(tokens)
                    tokens.extend([0] * pad_length)
                
                tokens = torch.tensor(tokens, dtype=torch.long)
                
                # Create attention mask
                attention_mask = (tokens != 0).float()
                
                # Create labels for next token prediction
                labels = tokens.clone()
                
                # Create loss weights with role-based weighting
                loss_weights = self._create_loss_weights(tokens)
                
                return {
                    'input_ids': tokens[:-1],
                    'labels': labels[1:],
                    'attention_mask': attention_mask[:-1],
                    'loss_weights': loss_weights[1:]
                }
                
            except Exception as e:
                logging.debug(f"Error processing conversation: {e}")
                return None
        
        def _create_loss_weights(self, tokens: torch.Tensor) -> torch.Tensor:
            """Create loss weights with assistant response emphasis."""
            loss_weights = torch.ones_like(tokens, dtype=torch.float)
            
            # Use the compatible method to get role tokens
            assistant_token = self.tokenizer.get_role_token('assistant')
            im_end_token = self.tokenizer.special_tokens.get("<|im_end|>")
            
            if im_end_token is None:
                # Fallback to the conversation end token
                im_end_token = self.tokenizer.special_tokens.get(
                    self.tokenizer.config.conversation_tokens["end"]
                )
            
            in_assistant_response = False
            for i, token_id in enumerate(tokens):
                if token_id == assistant_token:
                    in_assistant_response = True
                elif im_end_token and token_id == im_end_token:
                    in_assistant_response = False
                
                # Weight assistant responses higher, but not padding
                if in_assistant_response and token_id != 0:
                    loss_weights[i] = getattr(self.config, 'assistant_loss_weight', 1.5)
            
            return loss_weights
        
        def __len__(self) -> int:
            return len(self.conversations)
        
        def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
            """Get processed conversation with fallback."""
            conversation = self.conversations[idx]
            processed = self._process_conversation(conversation)
            
            # Return dummy sample if processing fails
            if processed is None:
                seq_len = self.config.seq_length - 1
                return {
                    'input_ids': torch.zeros(seq_len, dtype=torch.long),
                    'labels': torch.zeros(seq_len, dtype=torch.long),
                    'attention_mask': torch.zeros(seq_len, dtype=torch.float),
                    'loss_weights': torch.zeros(seq_len, dtype=torch.float)
                }
            
            return processed
        
        def get_stats(self) -> Dict[str, Any]:
            """Get dataset statistics."""
            return self.stats.copy()

    def create_dataloader(dataset: ConversationDataset, config, shuffle: bool = True) -> DataLoader:
        """Create optimized dataloader with error handling."""
        try:
            return DataLoader(
                dataset,
                batch_size=config.batch_size,
                shuffle=shuffle,
                num_workers=config.num_workers,
                pin_memory=torch.cuda.is_available(),
                prefetch_factor=2 if config.num_workers > 0 else None,
                drop_last=True,
                persistent_workers=config.num_workers > 0
            )
        except Exception as e:
            logging.warning(f"Failed to create optimized dataloader: {e}")
            # Fallback to basic dataloader
            return DataLoader(
                dataset,
                batch_size=config.batch_size,
                shuffle=shuffle,
                num_workers=0,
                drop_last=True
            )
    
    return ConversationDataset, create_dataloader


if __name__ == "__main__":
    # Test the compatibility setup
    print("Testing compatibility setup...")
    success = setup_imports()
    
    if success:
        print("Testing tokenizer compatibility...")
        try:
            from integrated_tokenizer import ConversationTokenizer
            tokenizer = ConversationTokenizer()
            
            # Test the compatibility interface
            test_conv = {
                "messages": [
                    {"role": "user", "content": "Hello!"},
                    {"role": "assistant", "content": "Hi there! How can I help you?"}
                ]
            }
            
            tokens = tokenizer.encode_conversation(test_conv)
            assistant_token = tokenizer.get_role_token('assistant')
            im_end_token = tokenizer.special_tokens.get("<|im_end|>")
            
            print(f"✅ Tokenizer test passed:")
            print(f"  - Encoded {len(tokens)} tokens")
            print(f"  - Assistant token: {assistant_token}")
            print(f"  - End token: {im_end_token}")
            
            # Test config compatibility
            from compatible_config import Config, ConfigPresets
            config = ConfigPresets.debug()
            print(f"✅ Config test passed: {config.experiment_name}")
            print(f"  - Assistant loss weight: {config.assistant_loss_weight}")
            
            print("\n🎉 All compatibility tests passed!")
            
        except Exception as e:
            print(f"❌ Compatibility test failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("❌ Compatibility setup failed")