# Copyright (c) 2025 Matias Nielsen. All rights reserved.
# Licensed under the Custom License below.

import logging
import json
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
import tiktoken
import hashlib
from functools import lru_cache


class TokenizerModel(Enum):
    """Supported tokenizer models with their tiktoken names."""
    GPT4O = "o200k_base"  # Latest and most efficient
    GPT4 = "cl100k_base"  # GPT-3.5/4 standard
    GPT3 = "p50k_base"    # GPT-3 standard
    GPT2 = "r50k_base"    # Legacy GPT-2
    
    @classmethod
    def get_recommended(cls) -> 'TokenizerModel':
        """Get the recommended modern tokenizer."""
        return cls.GPT4O


@dataclass
class TokenizerConfig:
    """Configuration for the tokenizer."""
    model: TokenizerModel = TokenizerModel.GPT4O
    max_sequence_length: int = 32768
    padding_token: str = "<|pad|>"
    eos_token: str = "<|endoftext|>"
    bos_token: str = "<|startoftext|>"
    unk_token: str = "<|unk|>"
    
    # Conversation-specific tokens
    conversation_tokens: Dict[str, str] = None
    role_tokens: Dict[str, str] = None
    
    # Performance settings
    use_cache: bool = True
    cache_size: int = 10000
    
    # Validation settings
    strict_validation: bool = True
    allow_empty_messages: bool = False
    max_message_length: int = 16384
    
    def __post_init__(self):
        if self.conversation_tokens is None:
            self.conversation_tokens = {
                "start": "<|im_start|>",
                "end": "<|im_end|>",
                "separator": "<|sep|>",
                "continuation": "<|cont|>"
            }
        
        if self.role_tokens is None:
            self.role_tokens = {
                "user": "<|user|>",
                "assistant": "<|assistant|>", 
                "system": "<|system|>",
                "tool": "<|tool|>",
                "observation": "<|observation|>"
            }
    
    def save(self, path: Union[str, Path]) -> None:
        """Save configuration to file."""
        path = Path(path)
        with open(path, 'w') as f:
            # Convert enum to string for JSON serialization
            config_dict = asdict(self)
            config_dict['model'] = self.model.value
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'TokenizerConfig':
        """Load configuration from file."""
        path = Path(path)
        with open(path, 'r') as f:
            config_dict = json.load(f)
            # Convert string back to enum
            config_dict['model'] = TokenizerModel(config_dict['model'])
            return cls(**config_dict)


class ConversationTokenizer:
    """
    Advanced conversation tokenizer with compatibility for existing codebase.
    
    Features:
    - Modern tokenizer models (GPT-4o by default)
    - Comprehensive error handling and validation
    - Caching for performance
    - Flexible configuration
    - Full compatibility with existing dataset.py and model.py
    """
    
    def __init__(self, config: Optional[TokenizerConfig] = None, model_name: str = None):
        # Support legacy initialization
        if model_name is not None:
            model_map = {
                "gpt2": TokenizerModel.GPT2,
                "r50k_base": TokenizerModel.GPT2,
                "p50k_base": TokenizerModel.GPT3,
                "cl100k_base": TokenizerModel.GPT4,
                "o200k_base": TokenizerModel.GPT4O
            }
            if config is None:
                config = TokenizerConfig()
            config.model = model_map.get(model_name, TokenizerModel.GPT4O)
        
        self.config = config or TokenizerConfig()
        self.logger = self._setup_logger()
        
        # Initialize base tokenizer
        self._init_base_tokenizer()
        
        # Build special tokens vocabulary
        self._build_special_tokens()
        
        # Initialize caching
        if self.config.use_cache:
            self._init_cache()
        
        # Statistics tracking
        self._stats = {
            'encode_calls': 0,
            'decode_calls': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'validation_errors': 0,
            'encoding_errors': 0
        }
        
        self.logger.info(f"Tokenizer initialized successfully")
        self.logger.info(f"Base vocab: {self.base_vocab_size}, Total vocab: {self.vocab_size}")
        self.logger.info(f"Model: {self.config.model.value}")
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger with proper formatting."""
        logger = logging.getLogger(f"{__name__}.{id(self)}")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _init_base_tokenizer(self) -> None:
        """Initialize the base tiktoken tokenizer."""
        try:
            self.tokenizer = tiktoken.get_encoding(self.config.model.value)
            self.base_tokenizer = self.tokenizer  # Compatibility alias
            self.base_vocab_size = self.tokenizer.n_vocab
            # Legacy compatibility
            self.n_vocab = self.base_vocab_size
            self.logger.info(f"Loaded {self.config.model.value} tokenizer")
        except Exception as e:
            self.logger.error(f"Failed to load tokenizer {self.config.model.value}: {e}")
            # Fallback to GPT-2
            self.logger.warning("Falling back to GPT-2 tokenizer")
            self.tokenizer = tiktoken.get_encoding("r50k_base")
            self.base_tokenizer = self.tokenizer
            self.base_vocab_size = self.tokenizer.n_vocab
            self.n_vocab = self.base_vocab_size
    
    def _build_special_tokens(self) -> None:
        """Build the special tokens vocabulary."""
        self.special_tokens: Dict[str, int] = {}
        self.reverse_special_tokens: Dict[int, str] = {}
        self._reverse_special_tokens = self.reverse_special_tokens  # Legacy compatibility
        
        # Start from base vocab size
        current_id = self.base_vocab_size
        
        # Add core tokens
        core_tokens = [
            self.config.padding_token,
            self.config.eos_token, 
            self.config.bos_token,
            self.config.unk_token
        ]
        
        for token in core_tokens:
            if token not in self.special_tokens:
                self.special_tokens[token] = current_id
                self.reverse_special_tokens[current_id] = token
                current_id += 1
        
        # Add conversation tokens
        for token in self.config.conversation_tokens.values():
            if token not in self.special_tokens:
                self.special_tokens[token] = current_id
                self.reverse_special_tokens[current_id] = token
                current_id += 1
        
        # Add role tokens
        for token in self.config.role_tokens.values():
            if token not in self.special_tokens:
                self.special_tokens[token] = current_id
                self.reverse_special_tokens[current_id] = token
                current_id += 1
        
        # Set final vocab size (no arbitrary padding)
        self.vocab_size = current_id
        
        # Create easy access properties
        self.pad_token_id = self.special_tokens[self.config.padding_token]
        self.eos_token_id = self.special_tokens[self.config.eos_token]
        self.bos_token_id = self.special_tokens[self.config.bos_token]
        self.unk_token_id = self.special_tokens[self.config.unk_token]
        
        self.logger.debug(f"Built {len(self.special_tokens)} special tokens")
    
    def _init_cache(self) -> None:
        """Initialize LRU caches for performance."""
        self._encode_cache: Dict[str, List[int]] = {}
        self._decode_cache: Dict[str, str] = {}
    
    @lru_cache(maxsize=1000)
    def _get_content_hash(self, content: str) -> str:
        """Get hash for content caching."""
        return hashlib.md5(content.encode()).hexdigest()
    
    def validate_conversation(self, conversation: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate conversation structure and content.
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        if not isinstance(conversation, dict):
            errors.append("Conversation must be a dictionary")
            return False, errors
        
        messages = conversation.get('messages', [])
        if not isinstance(messages, list):
            errors.append("Messages must be a list")
            return False, errors
        
        if not messages and self.config.strict_validation:
            errors.append("Conversation must contain at least one message")
        
        for i, message in enumerate(messages):
            if not isinstance(message, dict):
                errors.append(f"Message {i} must be a dictionary")
                continue
            
            # Check required fields
            role = message.get('role', '').lower().strip()
            content = message.get('content', '').strip()
            
            if not role and self.config.strict_validation:
                errors.append(f"Message {i} missing role")
            
            if not content and not self.config.allow_empty_messages:
                errors.append(f"Message {i} has empty content")
            
            # Validate role
            valid_roles = set(self.config.role_tokens.keys()) | {'prompter', 'human'}
            if role and role not in valid_roles:
                if self.config.strict_validation:
                    errors.append(f"Message {i} has invalid role: {role}")
                else:
                    self.logger.warning(f"Unknown role '{role}' in message {i}, will use 'user'")
            
            # Check content length
            if len(content) > self.config.max_message_length:
                errors.append(f"Message {i} content too long: {len(content)} > {self.config.max_message_length}")
        
        is_valid = len(errors) == 0
        if errors:
            self._stats['validation_errors'] += len(errors)
        
        return is_valid, errors
    
    def encode_conversation(self, 
                          conversation: Dict[str, Any], 
                          add_generation_prompt: bool = False,
                          max_length: Optional[int] = None) -> List[int]:
        """
        Encode conversation with advanced features.
        
        Args:
            conversation: Conversation dictionary with messages
            add_generation_prompt: Whether to add prompt for generation
            max_length: Maximum sequence length (uses config default if None)
            
        Returns:
            List of token IDs
        """
        self._stats['encode_calls'] += 1
        max_length = max_length or self.config.max_sequence_length
        
        try:
            # Validate conversation
            is_valid, errors = self.validate_conversation(conversation)
            if not is_valid and self.config.strict_validation:
                raise ValueError(f"Conversation validation failed: {errors}")
            
            tokens = []
            messages = conversation.get('messages', [])
            
            for i, message in enumerate(messages):
                role = message.get('role', 'user').lower().strip()
                content = message.get('content', '').strip()
                
                if not content and not self.config.allow_empty_messages:
                    continue
                
                # Normalize role names
                role = self._normalize_role(role)
                
                # Check if adding this message would exceed max length
                estimated_tokens = len(content) // 3  # Rough estimate
                if len(tokens) + estimated_tokens + 10 > max_length:  # +10 for special tokens
                    self.logger.warning(f"Truncating conversation at message {i} due to length limit")
                    break
                
                # Start message marker
                tokens.append(self.special_tokens[self.config.conversation_tokens["start"]])
                
                # Add role token
                if role in self.config.role_tokens:
                    tokens.append(self.special_tokens[self.config.role_tokens[role]])
                else:
                    tokens.append(self.special_tokens[self.config.role_tokens["user"]])
                
                # Encode content with caching
                if content:
                    content_tokens = self._encode_content(content)
                    tokens.extend(content_tokens)
                
                # End message marker
                tokens.append(self.special_tokens[self.config.conversation_tokens["end"]])
            
            # Add generation prompt if requested
            if add_generation_prompt:
                tokens.append(self.special_tokens[self.config.conversation_tokens["start"]])
                tokens.append(self.special_tokens[self.config.role_tokens["assistant"]])
            
            # Truncate if necessary
            if len(tokens) > max_length:
                tokens = tokens[:max_length-1] + [self.eos_token_id]
                self.logger.warning(f"Truncated sequence to {max_length} tokens")
            
            return tokens
            
        except Exception as e:
            self._stats['encoding_errors'] += 1
            self.logger.error(f"Conversation encoding failed: {e}")
            if self.config.strict_validation:
                raise
            return [self.unk_token_id]  # Return unknown token as fallback
    
    def _normalize_role(self, role: str) -> str:
        """Normalize role names to standard format."""
        role_mapping = {
            'prompter': 'user',
            'human': 'user',
            'bot': 'assistant',
            'ai': 'assistant',
            'model': 'assistant'
        }
        return role_mapping.get(role, role)
    
    def _encode_content(self, content: str) -> List[int]:
        """Encode content with caching."""
        if not self.config.use_cache:
            return self.tokenizer.encode(content)
        
        content_hash = self._get_content_hash(content)
        
        if content_hash in self._encode_cache:
            self._stats['cache_hits'] += 1
            return self._encode_cache[content_hash]
        
        self._stats['cache_misses'] += 1
        tokens = self.tokenizer.encode(content)
        
        # Manage cache size
        if len(self._encode_cache) >= self.config.cache_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self._encode_cache))
            del self._encode_cache[oldest_key]
        
        self._encode_cache[content_hash] = tokens
        return tokens
    
    def decode(self, 
               token_ids: List[int], 
               skip_special_tokens: bool = True,
               clean_up_tokenization_spaces: bool = True) -> str:
        """
        Decode tokens with advanced options.
        
        Args:
            token_ids: List of token IDs to decode
            skip_special_tokens: Whether to skip special tokens
            clean_up_tokenization_spaces: Whether to clean up tokenization artifacts
            
        Returns:
            Decoded string
        """
        self._stats['decode_calls'] += 1
        
        try:
            if not token_ids:
                return ""
            
            # Create cache key if caching enabled
            if self.config.use_cache:
                cache_key = f"{hash(tuple(token_ids))}_{skip_special_tokens}"
                if cache_key in self._decode_cache:
                    self._stats['cache_hits'] += 1
                    return self._decode_cache[cache_key]
                self._stats['cache_misses'] += 1
            
            # Filter tokens
            filtered_tokens = []
            for token_id in token_ids:
                # Skip special tokens if requested
                if skip_special_tokens and self.is_special_token(token_id):
                    continue
                
                # Clamp to valid range
                if token_id < 0:
                    token_id = self.unk_token_id
                elif token_id >= self.base_vocab_size:
                    if not self.is_special_token(token_id):
                        continue  # Skip invalid tokens
                
                filtered_tokens.append(token_id)
            
            # Decode using base tokenizer
            if filtered_tokens:
                text = self.tokenizer.decode(filtered_tokens)
            else:
                text = ""
            
            # Clean up tokenization spaces
            if clean_up_tokenization_spaces:
                text = self._cleanup_tokenization_spaces(text)
            
            # Cache result
            if self.config.use_cache:
                if len(self._decode_cache) >= self.config.cache_size:
                    oldest_key = next(iter(self._decode_cache))
                    del self._decode_cache[oldest_key]
                self._decode_cache[cache_key] = text
            
            return text
            
        except Exception as e:
            self.logger.error(f"Decode error: {e}")
            return "<decode_error>"
    
    def _cleanup_tokenization_spaces(self, text: str) -> str:
        """Clean up common tokenization artifacts."""
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Fix common punctuation issues
        text = text.replace(' .', '.')
        text = text.replace(' ,', ',')
        text = text.replace(' !', '!')
        text = text.replace(' ?', '?')
        text = text.replace(' ;', ';')
        text = text.replace(' :', ':')
        
        return text.strip()
    
    def parse_conversation(self, token_ids: List[int]) -> Dict[str, Any]:
        """
        Parse tokens back into conversation structure.
        
        Returns:
            Parsed conversation dictionary
        """
        messages = []
        current_role = None
        current_content_tokens = []
        
        i = 0
        while i < len(token_ids):
            token_id = token_ids[i]
            
            if token_id == self.special_tokens.get(self.config.conversation_tokens["start"]):
                # Save previous message if exists
                if current_role is not None and current_content_tokens:
                    content = self.decode(current_content_tokens, skip_special_tokens=False)
                    messages.append({"role": current_role, "content": content})
                
                # Reset for new message
                current_role = None
                current_content_tokens = []
                
            elif token_id in [self.special_tokens.get(role_token) for role_token in self.config.role_tokens.values()]:
                # Identify role
                for role, role_token in self.config.role_tokens.items():
                    if token_id == self.special_tokens.get(role_token):
                        current_role = role
                        break
            
            elif token_id == self.special_tokens.get(self.config.conversation_tokens["end"]):
                # End of message - save it
                if current_role is not None:
                    content = self.decode(current_content_tokens, skip_special_tokens=False)
                    messages.append({"role": current_role, "content": content})
                current_role = None
                current_content_tokens = []
                
            else:
                # Regular content token
                if current_role is not None:
                    current_content_tokens.append(token_id)
            
            i += 1
        
        # Handle any remaining content
        if current_role is not None and current_content_tokens:
            content = self.decode(current_content_tokens, skip_special_tokens=False)
            messages.append({"role": current_role, "content": content})
        
        return {"messages": messages}
    
    def get_special_tokens_dict(self) -> Dict[str, int]:
        """Get dictionary of all special tokens."""
        return self.special_tokens.copy()
    
    def is_special_token(self, token_id: int) -> bool:
        """Check if token ID corresponds to a special token."""
        return token_id in self.reverse_special_tokens
    
    def get_role_token_id(self, role: str) -> int:
        """Get token ID for a specific role."""
        normalized_role = self._normalize_role(role.lower())
        role_token = self.config.role_tokens.get(normalized_role)
        if role_token:
            return self.special_tokens.get(role_token, self.unk_token_id)
        return self.special_tokens.get(self.config.role_tokens["user"], self.unk_token_id)
    
    # LEGACY COMPATIBILITY METHODS
    def get_role_token(self, role: str) -> int:
        """Legacy compatibility method."""
        return self.get_role_token_id(role)
    
    def encode(self, text: str) -> List[int]:
        """Legacy encode method for basic text."""
        return self.tokenizer.encode(text)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get tokenizer usage statistics."""
        stats = self._stats.copy()
        if self.config.use_cache:
            stats['cache_hit_rate'] = (
                stats['cache_hits'] / max(1, stats['cache_hits'] + stats['cache_misses'])
            )
            stats['encode_cache_size'] = len(self._encode_cache)
            stats['decode_cache_size'] = len(self._decode_cache)
        return stats
    
    def clear_cache(self) -> None:
        """Clear all caches."""
        if self.config.use_cache:
            self._encode_cache.clear()
            self._decode_cache.clear()
            self.logger.info("Cleared tokenizer caches")
    
    def save_tokenizer(self, path: Union[str, Path]) -> None:
        """Save tokenizer configuration and special tokens."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        self.config.save(path / "config.json")
        
        # Save special tokens
        with open(path / "special_tokens.pkl", 'wb') as f:
            pickle.dump({
                'special_tokens': self.special_tokens,
                'reverse_special_tokens': self.reverse_special_tokens,
                'vocab_size': self.vocab_size,
                'base_vocab_size': self.base_vocab_size
            }, f)
        
        self.logger.info(f"Saved tokenizer to {path}")
    
    @classmethod
    def load_tokenizer(cls, path: Union[str, Path]) -> 'ConversationTokenizer':
        """Load tokenizer from saved files."""
        path = Path(path)
        
        # Load configuration
        config = TokenizerConfig.load(path / "config.json")
        
        # Create instance
        tokenizer = cls(config)
        
        # Load special tokens
        with open(path / "special_tokens.pkl", 'rb') as f:
            saved_data = pickle.load(f)
            tokenizer.special_tokens = saved_data['special_tokens']
            tokenizer.reverse_special_tokens = saved_data['reverse_special_tokens']
            tokenizer.vocab_size = saved_data['vocab_size']
            tokenizer.base_vocab_size = saved_data['base_vocab_size']
        
        tokenizer.logger.info(f"Loaded tokenizer from {path}")
        return tokenizer
    
    def __repr__(self) -> str:
        return (f"ConversationTokenizer(model={self.config.model.value}, "
                f"vocab_size={self.vocab_size}, "
                f"special_tokens={len(self.special_tokens)})")


# Convenience functions
def create_tokenizer(model: str = "gpt4o", **kwargs) -> ConversationTokenizer:
    """Create a tokenizer with common configurations."""
    model_map = {
        "gpt4o": TokenizerModel.GPT4O,
        "gpt4": TokenizerModel.GPT4,
        "gpt3": TokenizerModel.GPT3,
        "gpt2": TokenizerModel.GPT2
    }
    
    config = TokenizerConfig(
        model=model_map.get(model.lower(), TokenizerModel.GPT4O),
        **kwargs
    )
    
    return ConversationTokenizer(config)


# Example usage and testing
if __name__ == "__main__":
    # Test compatibility with legacy interface
    tokenizer = ConversationTokenizer(model_name="gpt2")  # Legacy init
    
    # Example conversation
    conversation = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you today?"},
            {"role": "assistant", "content": "I'm doing well, thank you for asking! How can I help you?"}
        ]
    }
    
    # Test legacy methods expected by dataset.py
    tokens = tokenizer.encode_conversation(conversation)
    print(f"Encoded to {len(tokens)} tokens")
    
    # Test role token access (expected by dataset.py)
    assistant_token = tokenizer.get_role_token('assistant')
    print(f"Assistant token ID: {assistant_token}")
    
    # Test special token access (expected by dataset.py)
    im_end_token = tokenizer.special_tokens["<|im_end|>"]
    print(f"End token ID: {im_end_token}")
    
    # Test decode
    decoded = tokenizer.decode(tokens)
    print(f"Decoded: {decoded}")
    
    # Show stats
    print(f"Stats: {json.dumps(tokenizer.get_stats(), indent=2)}")