# dataset.py - Fixed version with proper error handling
# Copyright (c) 2025 Matias Nielsen. All rights reserved.

import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
import torch
from torch.utils.data import Dataset, DataLoader


class ConversationDataset(Dataset):
    """Enhanced dataset with comprehensive error handling and validation."""
    
    def __init__(self, data_path: str, tokenizer, config, split: str = "train"):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.config = config
        self.split = split
        
        # Validate critical config attributes
        self._validate_config()
        
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
        
        # Ensure we have data
        if not self.conversations:
            raise ValueError(f"No valid conversations loaded from {data_path}")
        
        self._compute_statistics()
        
        logging.info(f"Dataset {split}: {len(self.conversations):,} conversations from {data_path}")
        logging.info(f"Average tokens: {self.stats['avg_token_length']:.1f}, "
                    f"Max: {self.stats['max_token_length']}, Min: {self.stats['min_token_length']}")
    
    def _validate_config(self):
        """Validate that config has required attributes."""
        required_attrs = ['seq_length', 'batch_size']
        for attr in required_attrs:
            if not hasattr(self.config, attr):
                raise ValueError(f"Config missing required attribute: {attr}")
            
            value = getattr(self.config, attr)
            if value is None:
                raise ValueError(f"Config attribute {attr} is None")
            
            if attr == 'seq_length' and value <= 0:
                raise ValueError(f"seq_length must be positive, got {value}")
            
            if attr == 'batch_size' and value <= 0:
                raise ValueError(f"batch_size must be positive, got {value}")
        
        # Check for assistant_loss_weight (with default)
        if not hasattr(self.config, 'assistant_loss_weight'):
            self.config.assistant_loss_weight = 2.0
            logging.warning("Config missing assistant_loss_weight, using default 2.0")
    
    def _load_and_validate_conversations(self) -> List[Dict]:
        """Load and validate conversations with comprehensive error handling."""
        conversations = []
        
        if not self.data_path.exists():
            logging.error(f"Data file not found: {self.data_path}")
            return conversations
        
        logging.info(f"Loading {self.split} data from {self.data_path}")
        
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                for line_no, line in enumerate(f, 1):
                    try:
                        line = line.strip()
                        if not line:  # Skip empty lines
                            continue
                            
                        conversation = json.loads(line)
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
                        logging.info(f"Processed {line_no:,} lines, {len(conversations):,} valid conversations")
        
        except FileNotFoundError:
            logging.error(f"Training data file not found: {self.data_path}")
        except Exception as e:
            logging.error(f"Error reading data file {self.data_path}: {e}")
        
        if not conversations:
            logging.warning(f"No valid conversations loaded from {self.data_path}")
        
        return conversations
    
    def _validate_conversation(self, conversation: Dict) -> bool:
        """Comprehensive conversation validation."""
        if not isinstance(conversation, dict):
            return False
            
        if 'messages' not in conversation:
            return False
        
        messages = conversation['messages']
        if not isinstance(messages, list) or len(messages) < 2:
            return False
        
        # Check message structure and content
        has_user = False
        has_assistant = False
        
        for msg in messages:
            if not isinstance(msg, dict):
                return False
            
            role = msg.get('role', '').lower().strip()
            content = msg.get('content', '').strip()
            
            if not content:  # Skip empty content messages
                continue
            
            # Track roles
            if role in ['user', 'prompter', 'human']:
                has_user = True
            elif role in ['assistant', 'bot', 'ai']:
                has_assistant = True
        
        # Require both user and assistant messages
        return has_user and has_assistant
    
    def _compute_statistics(self):
        """Compute dataset statistics with error handling."""
        if not self.conversations:
            return
        
        token_lengths = []
        
        # Sample conversations for statistics (to avoid processing all)
        sample_size = min(1000, len(self.conversations))
        if len(self.conversations) > sample_size:
            sample_indices = np.random.choice(len(self.conversations), sample_size, replace=False)
        else:
            sample_indices = range(len(self.conversations))
        
        for idx in sample_indices:
            try:
                tokens = self.tokenizer.encode_conversation(self.conversations[idx])
                if tokens and len(tokens) > 0:
                    token_lengths.append(len(tokens))
            except Exception as e:
                self.stats['tokenization_errors'] += 1
                if self.stats['tokenization_errors'] <= 5:  # Log first few errors
                    logging.warning(f"Tokenization error for conversation {idx}: {e}")
        
        if token_lengths:
            self.stats['avg_token_length'] = np.mean(token_lengths)
            self.stats['max_token_length'] = max(token_lengths)
            self.stats['min_token_length'] = min(token_lengths)
        else:
            logging.warning("No successful tokenizations for statistics")
    
    def _process_conversation(self, conversation: Dict) -> Optional[Dict[str, torch.Tensor]]:
        """Process conversation with enhanced error handling."""
        try:
            # Encode conversation
            tokens = self.tokenizer.encode_conversation(conversation)
            
            # Validate token sequence
            if not tokens or len(tokens) < 2:
                return None
            
            # Handle sequence length with proper truncation/padding
            target_length = self.config.seq_length
            
            if len(tokens) > target_length:
                # Truncate from the end to keep context, but ensure we have both input and target
                tokens = tokens[:target_length]
            elif len(tokens) < target_length:
                # Pad with pad token (should be 0)
                pad_token_id = getattr(self.tokenizer, 'pad_token_id', 0)
                tokens.extend([pad_token_id] * (target_length - len(tokens)))
            
            # Convert to tensor
            tokens = torch.tensor(tokens, dtype=torch.long)
            
            # Create attention mask (1 for real tokens, 0 for padding)
            pad_token_id = getattr(self.tokenizer, 'pad_token_id', 0)
            attention_mask = (tokens != pad_token_id).float()
            
            # For causal language modeling, input is tokens[:-1], target is tokens[1:]
            input_ids = tokens[:-1].clone()
            labels = tokens[1:].clone()
            attention_mask = attention_mask[:-1].clone()
            
            # Create loss weights with role-based weighting
            loss_weights = self._create_loss_weights(tokens[1:])
            
            return {
                'input_ids': input_ids,
                'labels': labels,
                'attention_mask': attention_mask,
                'loss_weights': loss_weights
            }
            
        except Exception as e:
            logging.debug(f"Error processing conversation: {e}")
            return None
    
    def _create_loss_weights(self, tokens: torch.Tensor) -> torch.Tensor:
        """Create loss weights with assistant response emphasis."""
        loss_weights = torch.ones_like(tokens, dtype=torch.float)
        
        try:
            # Get assistant and end tokens
            assistant_token = self.tokenizer.get_role_token('assistant')
            
            # Handle different ways to access special tokens
            if hasattr(self.tokenizer, 'special_tokens'):
                end_tokens = [
                    self.tokenizer.special_tokens.get("<|im_end|>", -1),
                    self.tokenizer.special_tokens.get("<|end|>", -1),
                ]
            else:
                end_tokens = []
            
            # Apply higher weights to assistant responses
            in_assistant_response = False
            for i, token_id in enumerate(tokens):
                if token_id == assistant_token:
                    in_assistant_response = True
                elif token_id.item() in end_tokens:
                    in_assistant_response = False
                
                # Weight assistant responses higher, but not padding
                if in_assistant_response and token_id != 0:
                    loss_weights[i] = self.config.assistant_loss_weight
                    
        except Exception as e:
            logging.debug(f"Error creating loss weights: {e}")
            # Return uniform weights as fallback
            pass
        
        return loss_weights
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.conversations)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get processed conversation with robust fallback."""
        if idx >= len(self.conversations):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.conversations)}")
        
        conversation = self.conversations[idx]
        processed = self._process_conversation(conversation)
        
        # Return properly formatted dummy sample if processing fails
        if processed is None:
            seq_len = self.config.seq_length - 1  # -1 for input_ids vs labels shift
            
            return {
                'input_ids': torch.zeros(seq_len, dtype=torch.long),
                'labels': torch.zeros(seq_len, dtype=torch.long),
                'attention_mask': torch.zeros(seq_len, dtype=torch.float),
                'loss_weights': torch.ones(seq_len, dtype=torch.float)  # Use ones, not zeros for fallback
            }
        
        return processed
    
    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        return self.stats.copy()


def create_dataloader(dataset: ConversationDataset, config, shuffle: bool = True) -> Optional[DataLoader]:
    """Create optimized dataloader with comprehensive error handling."""
    if dataset is None:
        logging.error("Cannot create DataLoader: dataset is None")
        return None
    
    if len(dataset) == 0:
        logging.error("Cannot create DataLoader: dataset is empty")
        return None
    
    logging.info(f"Creating DataLoader with dataset of length: {len(dataset)}")
    
    try:
        # Test dataset access
        test_item = dataset[0]
        logging.info(f"Dataset test successful. Sample shapes: "
                    f"input_ids: {test_item['input_ids'].shape}, "
                    f"labels: {test_item['labels'].shape}")
        
        # Get number of workers (be conservative)
        num_workers = getattr(config, 'num_workers', 2)
        # Reduce workers if we have memory constraints
        if num_workers > 2:
            num_workers = 2
            logging.info(f"Reduced num_workers to {num_workers} for stability")
        
        return DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
        )
        
    except Exception as e:
        logging.error(f"Failed to create optimized dataloader: {e}")
        logging.info("Attempting to create basic dataloader...")
        
        try:
            # Fallback to very basic dataloader
            return DataLoader(
                dataset,
                batch_size=config.batch_size,
                shuffle=shuffle,
                num_workers=0,
                drop_last=True
            )
        except Exception as e2:
            logging.error(f"Failed to create basic dataloader: {e2}")
            return None


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Custom collate function for handling variable-length sequences."""
    # Stack tensors
    collated = {}
    
    for key in batch[0].keys():
        tensors = [item[key] for item in batch]
        collated[key] = torch.stack(tensors, dim=0)
    
    return collated


# Utility functions for dataset validation and debugging
def validate_dataset_config(config) -> bool:
    """Validate dataset configuration."""
    required_attrs = ['seq_length', 'batch_size']
    
    for attr in required_attrs:
        if not hasattr(config, attr):
            print(f"❌ Config missing required attribute: {attr}")
            return False
        
        value = getattr(config, attr)
        if value is None:
            print(f"❌ Config attribute {attr} is None")
            return False
        
        if isinstance(value, int) and value <= 0:
            print(f"❌ Config attribute {attr} must be positive, got {value}")
            return False
    
    print("✅ Dataset config validation passed")
    return True


def test_dataset_loading(data_path: str, tokenizer, config, max_samples: int = 10) -> bool:
    """Test dataset loading with detailed diagnostics."""
    print(f"🧪 Testing dataset loading from: {data_path}")
    
    try:
        # Test dataset creation
        dataset = ConversationDataset(data_path, tokenizer, config, "test")
        print(f"✅ Dataset created successfully: {len(dataset)} conversations")
        
        # Test data access
        if len(dataset) == 0:
            print("❌ Dataset is empty")
            return False
        
        # Test first few samples
        for i in range(min(max_samples, len(dataset))):
            try:
                sample = dataset[i]
                print(f"  Sample {i}: input_ids={sample['input_ids'].shape}, "
                      f"labels={sample['labels'].shape}")
            except Exception as e:
                print(f"❌ Error accessing sample {i}: {e}")
                return False
        
        # Test dataloader creation
        dataloader = create_dataloader(dataset, config, shuffle=False)
        if dataloader is None:
            print("❌ Failed to create dataloader")
            return False
        
        # Test batch loading
        try:
            batch = next(iter(dataloader))
            print(f"✅ Batch loaded successfully: {batch['input_ids'].shape}")
        except Exception as e:
            print(f"❌ Error loading batch: {e}")
            return False
        
        print("✅ Dataset testing completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Dataset testing failed: {e}")
        return False


if __name__ == "__main__":
    # Basic testing functionality
    import sys
    from pathlib import Path
    
    # Add parent directory to path for imports
    sys.path.append(str(Path(__file__).parent))
    
    try:
        from tokenizer import ConversationTokenizer, TokenizerConfig
        from dataclasses import dataclass
        
        @dataclass
        class TestConfig:
            seq_length: int = 512
            batch_size: int = 2
            assistant_loss_weight: float = 2.0
        
        # Test with dummy data
        print("Testing dataset functionality...")
        
        # This would require actual test data to run
        print("Dataset module loaded successfully!")
        print("For full testing, run with actual training data.")
        
    except ImportError as e:
        print(f"Import error during testing: {e}")
        print("Ensure all dependencies are installed.")