# Copyright (c) 2025 Matias Nielsen. All rights reserved.
# Licensed under the Custom License below.

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
        self._compute_statistics()
        
        logging.info(f"Dataset {split}: {len(self.conversations):,} conversations from {data_path}")
        logging.info(f"Average tokens: {self.stats['avg_token_length']:.1f}, "
                    f"Max: {self.stats['max_token_length']}, Min: {self.stats['min_token_length']}")
    
    def _load_and_validate_conversations(self) -> List[Dict]:
        """Load and validate conversations with comprehensive error handling."""
        conversations = []
        
        if not self.data_path.exists():
            logging.error(f"Data file not found: {self.data_path}")
            return conversations
        
        logging.info(f"Loading {self.split} data from {self.data_path}")
        
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
                    logging.info(f"Processed {line_no:,} lines, {len(conversations):,} valid conversations")
        
        return conversations
    
    def _validate_conversation(self, conversation: Dict) -> bool:
        """Comprehensive conversation validation for OASST format."""
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
            
            # Track roles - handle both 'prompter' and 'user'
            if role in ['user', 'prompter', 'human']:
                has_user = True
            elif role in ['assistant', 'ai', 'bot']:
                has_assistant = True
        
        # Require both user and assistant messages
        return has_user and has_assistant
    
    def _compute_statistics(self):
        """Compute dataset statistics."""
        if not self.conversations:
            return
        
        token_lengths = []
        
        # Sample conversations for statistics (to avoid processing all)
        sample_size = min(1000, len(self.conversations))
        sample_indices = np.random.choice(len(self.conversations), sample_size, replace=False)
        
        for idx in sample_indices:
            try:
                tokens = self.tokenizer.encode_conversation(self.conversations[idx])
                if tokens:
                    token_lengths.append(len(tokens))
            except Exception:
                self.stats['tokenization_errors'] += 1
        
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
            
            # Create loss weights - train on ALL content tokens (both user and assistant)
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
        """Create loss weights - train on both user and assistant content."""
        loss_weights = torch.ones_like(tokens, dtype=torch.float)
        
        # Special tokens for tracking conversation structure
        im_start_token = self.tokenizer.special_tokens["<|im_start|>"]
        im_end_token = self.tokenizer.special_tokens["<|im_end|>"]
        
        # Handle both user/prompter and assistant roles
        user_token = self.tokenizer.get_role_token('user')
        prompter_token = self.tokenizer.get_role_token('user')  # Maps to same token
        assistant_token = self.tokenizer.get_role_token('assistant')
        system_token = self.tokenizer.get_role_token('system')
        
        in_content = False
        current_role_weight = 1.0
        
        for i, token_id in enumerate(tokens):
            if token_id == 0:  # Padding tokens
                loss_weights[i] = 0.0
            elif token_id == im_start_token:
                in_content = False
                loss_weights[i] = 0.0  # Don't train on structure tokens
            elif token_id in [user_token, prompter_token, assistant_token, system_token]:
                in_content = False
                loss_weights[i] = 0.0  # Don't train on role tokens
                # Set weight for upcoming content based on role
                if token_id == assistant_token:
                    current_role_weight = getattr(self.config, 'assistant_loss_weight', 2.0)
                else:
                    # Train on user/prompter content with normal weight
                    current_role_weight = 1.0
            elif token_id == im_end_token:
                in_content = False
                loss_weights[i] = 0.0  # Don't train on structure tokens
            else:
                # This is content - train on it with role-specific weight
                in_content = True
                loss_weights[i] = current_role_weight
        
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