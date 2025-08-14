# Copyright (c) 2025 Matias Nielsen. All rights reserved.
# Licensed under the Custom License below.

import logging
from typing import Dict, List, Any
import tiktoken


class ConversationTokenizer:
    """Production tokenizer with enhanced error handling and validation."""
    
    def __init__(self, model_name: str = "gpt2"):
        try:
            self.tokenizer = tiktoken.get_encoding(model_name)
        except Exception as e:
            logging.error(f"Failed to load tokenizer {model_name}: {e}")
            raise
            
        self.base_vocab_size = self.tokenizer.n_vocab
        
        # Special tokens for conversation structure
        self.special_tokens = {
            "<|im_start|>": self.base_vocab_size,
            "<|im_end|>": self.base_vocab_size + 1,
            "<|user|>": self.base_vocab_size + 2,
            "<|assistant|>": self.base_vocab_size + 3,
            "<|system|>": self.base_vocab_size + 4,
        }
        
        self.vocab_size = self.base_vocab_size + len(self.special_tokens)
        self._reverse_special_tokens = {v: k for k, v in self.special_tokens.items()}
        
        # Pad vocab size to be efficient
        if self.vocab_size % 64 != 0:
            self.vocab_size = ((self.vocab_size + 63) // 64) * 64
            
        logging.info(f"Tokenizer initialized with vocab size: {self.vocab_size}")
    
    def encode_conversation(self, conversation: Dict[str, Any]) -> List[int]:
        """Encode conversation with enhanced error handling."""
        try:
            tokens = []
            messages = conversation.get('messages', [])
            
            if not messages:
                return tokens
            
            for message in messages:
                role = message.get('role', '').lower()
                content = message.get('content', '').strip()
                
                if not content:
                    continue
                
                # Validate role
                if role not in ['user', 'prompter', 'assistant', 'system']:
                    role = 'user'  # Default fallback
                
                # Start message
                tokens.append(self.special_tokens["<|im_start|>"])
                
                # Add role
                if role == 'user' or role == 'prompter':
                    tokens.append(self.special_tokens["<|user|>"])
                elif role == 'assistant':
                    tokens.append(self.special_tokens["<|assistant|>"])
                else:
                    tokens.append(self.special_tokens["<|system|>"])
                
                # Add content with error handling
                try:
                    content_tokens = self.tokenizer.encode(content)
                    tokens.extend(content_tokens)
                except Exception as e:
                    logging.warning(f"Failed to encode content: {e}")
                    continue
                
                # End message
                tokens.append(self.special_tokens["<|im_end|>"])
            
            return tokens
            
        except Exception as e:
            logging.error(f"Conversation encoding failed: {e}")
            return []
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode with enhanced error handling."""
        try:
            # Filter out special tokens if requested
            if skip_special_tokens:
                filtered_tokens = []
                for token_id in token_ids:
                    if token_id not in self._reverse_special_tokens and token_id < self.base_vocab_size:
                        filtered_tokens.append(token_id)
                token_ids = filtered_tokens
            
            # Clamp invalid tokens
            token_ids = [max(0, min(token_id, self.base_vocab_size - 1)) for token_id in token_ids]
            
            return self.tokenizer.decode(token_ids)
        except Exception as e:
            logging.warning(f"Decode error: {e}")
            return "<decode_error>"
    
    def is_special_token(self, token_id: int) -> bool:
        """Check if token is a special token."""
        return token_id in self._reverse_special_tokens
    
    def get_role_token(self, role: str) -> int:
        """Get token ID for a role."""
        role_map = {
            'user': self.special_tokens["<|user|>"],
            'prompter': self.special_tokens["<|user|>"], 
            'assistant': self.special_tokens["<|assistant|>"],
            'system': self.special_tokens["<|system|>"]
        }
        return role_map.get(role.lower(), self.special_tokens["<|user|>"])