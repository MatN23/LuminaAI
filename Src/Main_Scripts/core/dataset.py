# Copyright (c) 2025 Matias Nielsen. All rights reserved.
# Licensed under the Custom License below.

import json
import logging
import numpy as np
import os
import pickle
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Iterator, Union
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset


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


class StreamingConversationDataset(IterableDataset):
    """Memory-efficient streaming dataset for very large conversation files."""
    
    def __init__(self, 
                 data_path: str, 
                 tokenizer, 
                 config, 
                 split: str = "train",
                 buffer_size: int = 1000,
                 shuffle_buffer: bool = True,
                 num_workers: int = 1,
                 worker_id: int = 0):
        """
        Initialize streaming dataset.
        
        Args:
            data_path: Path to JSONL file or directory with shard files
            tokenizer: Conversation tokenizer
            config: Training configuration
            split: Dataset split name
            buffer_size: Size of shuffle buffer
            shuffle_buffer: Whether to shuffle samples in buffer
            num_workers: Total number of workers (for sharding)
            worker_id: ID of current worker (for sharding)
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.config = config
        self.split = split
        self.buffer_size = buffer_size
        self.shuffle_buffer = shuffle_buffer
        self.num_workers = num_workers
        self.worker_id = worker_id
        
        # Find data files
        self.data_files = self._discover_data_files()
        
        # Statistics
        self.stats = {
            'total_processed': 0,
            'valid_conversations': 0,
            'invalid_conversations': 0,
            'files_processed': 0
        }
        
        logging.info(f"StreamingDataset {split}: Found {len(self.data_files)} files")
        if len(self.data_files) > 1:
            logging.info(f"Worker {worker_id}/{num_workers} will process {len(self._get_worker_files())} files")
    
    def _discover_data_files(self) -> List[Path]:
        """Discover all data files to process."""
        files = []
        
        if self.data_path.is_file():
            # Single file
            files.append(self.data_path)
        elif self.data_path.is_dir():
            # Directory with shard files
            pattern = f"*_shard_*.jsonl"
            files.extend(sorted(self.data_path.glob(pattern)))
            
            # Fallback to any .jsonl files
            if not files:
                files.extend(sorted(self.data_path.glob("*.jsonl")))
        else:
            # Check for shard files in parent directory
            parent = self.data_path.parent
            stem = self.data_path.stem
            pattern = f"{stem}_shard_*.jsonl"
            files.extend(sorted(parent.glob(pattern)))
        
        if not files:
            raise FileNotFoundError(f"No data files found for {self.data_path}")
        
        return files
    
    def _get_worker_files(self) -> List[Path]:
        """Get files assigned to current worker."""
        if self.num_workers <= 1:
            return self.data_files
        
        # Distribute files across workers
        worker_files = []
        for i, file_path in enumerate(self.data_files):
            if i % self.num_workers == self.worker_id:
                worker_files.append(file_path)
        
        return worker_files
    
    def _validate_conversation(self, conversation: Dict) -> bool:
        """Fast conversation validation."""
        messages = conversation.get('messages', [])
        if not messages or len(messages) < 2:
            return False
        
        # Quick validation - just check we have content
        for msg in messages:
            if not isinstance(msg, dict) or not msg.get('content', '').strip():
                return False
        
        return True
    
    def _process_conversation(self, conversation: Dict) -> Optional[Dict[str, torch.Tensor]]:
        """Process single conversation - same as ConversationDataset."""
        try:
            tokens = self.tokenizer.encode_conversation(conversation)
            
            if not tokens or len(tokens) < 10:
                return None
            
            # Handle sequence length
            if len(tokens) > self.config.seq_length:
                tokens = tokens[-self.config.seq_length:]
            else:
                pad_length = self.config.seq_length - len(tokens)
                tokens.extend([0] * pad_length)
            
            tokens = torch.tensor(tokens, dtype=torch.long)
            attention_mask = (tokens != 0).float()
            labels = tokens.clone()
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
        """Create loss weights - same as ConversationDataset."""
        loss_weights = torch.ones_like(tokens, dtype=torch.float)
        
        # Get special tokens
        try:
            im_start_token = self.tokenizer.special_tokens["<|im_start|>"]
            im_end_token = self.tokenizer.special_tokens["<|im_end|>"]
            user_token = self.tokenizer.get_role_token('user')
            assistant_token = self.tokenizer.get_role_token('assistant')
            system_token = self.tokenizer.get_role_token('system')
        except (KeyError, AttributeError):
            # Fallback if tokenizer doesn't have these methods
            return loss_weights
        
        current_role_weight = 1.0
        
        for i, token_id in enumerate(tokens):
            if token_id == 0:  # Padding
                loss_weights[i] = 0.0
            elif token_id in [im_start_token, im_end_token]:
                loss_weights[i] = 0.0  # Structure tokens
            elif token_id in [user_token, assistant_token, system_token]:
                loss_weights[i] = 0.0  # Role tokens
                current_role_weight = 2.0 if token_id == assistant_token else 1.0
            else:
                loss_weights[i] = current_role_weight
        
        return loss_weights
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate over conversations with buffering and shuffling."""
        worker_files = self._get_worker_files()
        
        # Shuffle files if requested
        if self.shuffle_buffer:
            import random
            random.shuffle(worker_files)
        
        buffer = []
        
        for file_path in worker_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line_no, line in enumerate(f, 1):
                        try:
                            conversation = json.loads(line.strip())
                            self.stats['total_processed'] += 1
                            
                            if not self._validate_conversation(conversation):
                                self.stats['invalid_conversations'] += 1
                                continue
                            
                            processed = self._process_conversation(conversation)
                            if processed is None:
                                continue
                            
                            self.stats['valid_conversations'] += 1
                            
                            # Add to buffer
                            buffer.append(processed)
                            
                            # Yield from buffer when full
                            if len(buffer) >= self.buffer_size:
                                if self.shuffle_buffer:
                                    np.random.shuffle(buffer)
                                
                                for item in buffer:
                                    yield item
                                buffer = []
                                
                        except (json.JSONDecodeError, Exception) as e:
                            self.stats['invalid_conversations'] += 1
                            if line_no <= 10:
                                logging.debug(f"Error processing line {line_no}: {e}")
                            continue
                
                self.stats['files_processed'] += 1
                
                # Log progress periodically
                if self.stats['files_processed'] % 10 == 0:
                    logging.info(f"StreamingDataset processed {self.stats['files_processed']} files, "
                               f"{self.stats['valid_conversations']} valid conversations")
                
            except Exception as e:
                logging.error(f"Error reading file {file_path}: {e}")
                continue
        
        # Yield remaining items in buffer
        if buffer:
            if self.shuffle_buffer:
                np.random.shuffle(buffer)
            for item in buffer:
                yield item
    
    def get_stats(self) -> Dict[str, Any]:
        """Get streaming statistics."""
        return self.stats.copy()


class ShardedConversationDataset(Dataset):
    """Dataset that automatically handles sharded data files."""
    
    def __init__(self, 
                 data_path: str, 
                 tokenizer, 
                 config, 
                 split: str = "train",
                 max_shard_size_mb: int = 512,
                 memory_map: bool = True):
        """
        Initialize sharded dataset.
        
        Args:
            data_path: Path to main data file or shard directory
            tokenizer: Conversation tokenizer
            config: Training configuration
            split: Dataset split
            max_shard_size_mb: Maximum shard size in MB
            memory_map: Use memory mapping for large files
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.config = config
        self.split = split
        self.max_shard_size_mb = max_shard_size_mb
        self.memory_map = memory_map
        
        # Find or create shards
        self.shard_files = self._discover_or_create_shards()
        
        # Load shard metadata
        self.shard_metadata = self._load_shard_metadata()
        
        # Calculate total length
        self.total_length = sum(meta['length'] for meta in self.shard_metadata)
        
        # Current shard cache
        self._current_shard_idx = -1
        self._current_shard_data = None
        self._shard_lock = threading.Lock()
        
        logging.info(f"ShardedDataset {split}: {len(self.shard_files)} shards, "
                    f"{self.total_length:,} total conversations")
    
    def _discover_or_create_shards(self) -> List[Path]:
        """Discover existing shards or create new ones."""
        # Check for existing shards
        if self.data_path.is_dir():
            shard_files = sorted(self.data_path.glob("*_shard_*.jsonl"))
            if shard_files:
                return shard_files
        else:
            parent = self.data_path.parent
            stem = self.data_path.stem
            shard_files = sorted(parent.glob(f"{stem}_shard_*.jsonl"))
            if shard_files:
                return shard_files
        
        # Need to create shards
        return self._create_shards()
    
    def _create_shards(self) -> List[Path]:
        """Create shards from source file."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Source file not found: {self.data_path}")
        
        logging.info(f"Creating shards from {self.data_path}")
        
        shard_dir = self.data_path.parent / "shards"
        shard_dir.mkdir(exist_ok=True)
        
        base_name = self.data_path.stem
        shard_files = []
        
        max_size_bytes = self.max_shard_size_mb * 1024 * 1024
        shard_idx = 0
        current_size = 0
        current_shard_file = None
        conversations_in_shard = 0
        
        with open(self.data_path, 'r', encoding='utf-8') as source:
            for line in source:
                line = line.strip()
                if not line:
                    continue
                
                line_size = len(line.encode('utf-8'))
                
                # Start new shard if needed
                if current_shard_file is None or current_size + line_size > max_size_bytes:
                    if current_shard_file is not None:
                        current_shard_file.close()
                        # Save metadata
                        self._save_shard_metadata(shard_files[-1], conversations_in_shard)
                    
                    shard_path = shard_dir / f"{base_name}_shard_{shard_idx:04d}.jsonl"
                    shard_files.append(shard_path)
                    current_shard_file = open(shard_path, 'w', encoding='utf-8')
                    current_size = 0
                    conversations_in_shard = 0
                    shard_idx += 1
                
                current_shard_file.write(line + '\n')
                current_size += line_size
                conversations_in_shard += 1
        
        if current_shard_file is not None:
            current_shard_file.close()
            self._save_shard_metadata(shard_files[-1], conversations_in_shard)
        
        logging.info(f"Created {len(shard_files)} shards in {shard_dir}")
        return shard_files
    
    def _save_shard_metadata(self, shard_path: Path, length: int):
        """Save metadata for a shard."""
        meta_path = shard_path.with_suffix('.meta')
        metadata = {
            'length': length,
            'created': time.time(),
            'source': str(self.data_path)
        }
        with open(meta_path, 'w') as f:
            json.dump(metadata, f)
    
    def _load_shard_metadata(self) -> List[Dict]:
        """Load metadata for all shards."""
        metadata = []
        for shard_file in self.shard_files:
            meta_path = shard_file.with_suffix('.meta')
            
            if meta_path.exists():
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
            else:
                # Calculate length by counting lines
                with open(shard_file, 'r') as f:
                    length = sum(1 for _ in f)
                meta = {'length': length}
                self._save_shard_metadata(shard_file, length)
            
            metadata.append(meta)
        
        return metadata
    
    def _load_shard(self, shard_idx: int) -> List[Dict]:
        """Load conversations from a specific shard."""
        shard_file = self.shard_files[shard_idx]
        conversations = []
        
        with open(shard_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    conversation = json.loads(line.strip())
                    conversations.append(conversation)
                except json.JSONDecodeError:
                    continue
        
        return conversations
    
    def _get_shard_and_offset(self, idx: int) -> tuple:
        """Get shard index and offset for global index."""
        current_offset = 0
        for shard_idx, meta in enumerate(self.shard_metadata):
            shard_length = meta['length']
            if idx < current_offset + shard_length:
                return shard_idx, idx - current_offset
            current_offset += shard_length
        
        raise IndexError(f"Index {idx} out of range")
    
    def __len__(self) -> int:
        return self.total_length
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get item by global index."""
        shard_idx, offset = self._get_shard_and_offset(idx)
        
        # Load shard if not cached
        with self._shard_lock:
            if self._current_shard_idx != shard_idx:
                self._current_shard_data = self._load_shard(shard_idx)
                self._current_shard_idx = shard_idx
            
            conversation = self._current_shard_data[offset]
        
        # Process conversation (same as ConversationDataset)
        processed = self._process_conversation(conversation)
        
        if processed is None:
            # Return dummy data
            seq_len = self.config.seq_length - 1
            return {
                'input_ids': torch.zeros(seq_len, dtype=torch.long),
                'labels': torch.zeros(seq_len, dtype=torch.long),
                'attention_mask': torch.zeros(seq_len, dtype=torch.float),
                'loss_weights': torch.zeros(seq_len, dtype=torch.float)
            }
        
        return processed
    
    def _process_conversation(self, conversation: Dict) -> Optional[Dict[str, torch.Tensor]]:
        """Process conversation - same as ConversationDataset."""
        # Reuse the processing logic from ConversationDataset
        dataset = ConversationDataset.__new__(ConversationDataset)
        dataset.tokenizer = self.tokenizer
        dataset.config = self.config
        return dataset._process_conversation(conversation)


def create_dataloader(dataset: Union[ConversationDataset, StreamingConversationDataset], 
                     config, 
                     shuffle: bool = True) -> DataLoader:
    """Create optimized dataloader with error handling."""
    try:
        # Different handling for streaming vs regular datasets
        if isinstance(dataset, StreamingConversationDataset):
            # Streaming datasets can't be shuffled by DataLoader
            return DataLoader(
                dataset,
                batch_size=config.batch_size,
                num_workers=0,  # Streaming datasets handle their own parallelism
                pin_memory=torch.cuda.is_available(),
                drop_last=True,
            )
        else:
            # Regular datasets
            return DataLoader(
                dataset,
                batch_size=config.batch_size,
                shuffle=shuffle,
                num_workers=config.num_workers,
                pin_memory=torch.cuda.is_available(),
                prefetch_factor=4 if config.num_workers > 0 else None,
                drop_last=True,
                persistent_workers=config.num_workers > 0
            )
    except Exception as e:
        logging.warning(f"Failed to create optimized dataloader: {e}")
        # Fallback to basic dataloader
        return DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=shuffle if not isinstance(dataset, StreamingConversationDataset) else False,
            num_workers=0,
            drop_last=True
        )


def create_memory_efficient_dataloader(data_path: str, 
                                     tokenizer, 
                                     config, 
                                     split: str = "train",
                                     strategy: str = "auto") -> DataLoader:
    """
    Create memory-efficient dataloader with automatic strategy selection.
    
    Args:
        data_path: Path to data file
        tokenizer: Conversation tokenizer
        config: Training configuration
        split: Dataset split
        strategy: Loading strategy ('auto', 'memory', 'streaming', 'sharded')
    """
    path = Path(data_path)
    
    # Auto-detect strategy if not specified
    if strategy == "auto":
        if path.exists():
            file_size_gb = path.stat().st_size / (1024**3)
            if file_size_gb < 0.5:
                strategy = "memory"
            elif file_size_gb < 10:
                strategy = "sharded"
            else:
                strategy = "streaming"
        else:
            # Check for existing shards
            shard_files = list(path.parent.glob(f"{path.stem}_shard_*.jsonl"))
            if shard_files:
                total_size = sum(f.stat().st_size for f in shard_files) / (1024**3)
                strategy = "streaming" if total_size > 10 else "sharded"
            else:
                strategy = "memory"  # Default
    
    logging.info(f"Using {strategy} loading strategy for {data_path}")
    
    # Create appropriate dataset
    if strategy == "streaming":
        dataset = StreamingConversationDataset(
            data_path, tokenizer, config, split,
            buffer_size=getattr(config, 'stream_buffer_size', 1000),
            shuffle_buffer=split == "train"
        )
    elif strategy == "sharded":
        dataset = ShardedConversationDataset(
            data_path, tokenizer, config, split,
            max_shard_size_mb=getattr(config, 'max_shard_size_mb', 512)
        )
    else:  # memory
        dataset = ConversationDataset(data_path, tokenizer, config, split)
    
    return create_dataloader(dataset, config, shuffle=(split == "train"))