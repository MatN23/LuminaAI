# Copyright (c) 2025 Matias Nielsen. All rights reserved.
# Licensed under the Custom License below.

import json
import logging
import numpy as np
import os
import gc
import mmap
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Optional, Any, Iterator, Union
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from threading import Lock
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import random
import time
import pickle
import hashlib


@dataclass
class ShardConfig:
    """Configuration for dataset sharding."""
    max_shard_size_mb: int = 512
    min_shard_size_mb: int = 50
    max_memory_usage_gb: float = 8.0
    num_workers: int = min(8, mp.cpu_count())
    buffer_size: int = 10000
    enable_memory_mapping: bool = True
    enable_compression: bool = False
    cache_shards: bool = True
    shard_shuffle: bool = True


class ShardManager:
    """Manages dataset sharding for both small and massive datasets."""
    
    def __init__(self, base_path: Path, config: ShardConfig):
        self.base_path = Path(base_path)
        self.config = config
        self.shards_dir = self.base_path / "shards"
        self.metadata_file = self.base_path / "shard_metadata.json"
        self.lock = Lock()
        
        self.stats = {
            'total_conversations': 0,
            'total_shards': 0,
            'total_size_mb': 0,
            'avg_shard_size_mb': 0,
            'load_strategy': 'unknown'
        }
        
        self.shards_dir.mkdir(parents=True, exist_ok=True)
        
    def estimate_dataset_size(self, data_path: Path) -> Dict[str, Any]:
        """Estimate dataset size and determine loading strategy."""
        if not data_path.exists():
            return {'size_mb': 0, 'estimated_conversations': 0, 'strategy': 'none'}
        
        file_size_mb = data_path.stat().st_size / (1024 * 1024)
        
        # Sample to estimate conversation count
        sample_size = 0
        sample_conversations = 0
        
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= 100:
                        break
                    sample_size += len(line.encode('utf-8'))
                    sample_conversations += 1
        except Exception:
            return {'size_mb': file_size_mb, 'estimated_conversations': 1000, 'strategy': 'sharded'}
        
        if sample_conversations == 0:
            return {'size_mb': file_size_mb, 'estimated_conversations': 0, 'strategy': 'none'}
        
        avg_conversation_size = sample_size / sample_conversations
        estimated_conversations = int(file_size_mb * 1024 * 1024 / avg_conversation_size)
        
        # Determine strategy
        if file_size_mb < 500:
            strategy = 'memory'
        elif file_size_mb < 5000:
            strategy = 'simple_shards'
        else:
            strategy = 'advanced_shards'
        
        return {
            'size_mb': file_size_mb,
            'estimated_conversations': estimated_conversations,
            'strategy': strategy,
            'avg_conversation_size': avg_conversation_size
        }
    
    def create_shards(self, data_path: Path, split: str = "train") -> List[Path]:
        """Create shards from a large dataset file."""
        info = self.estimate_dataset_size(data_path)
        strategy = info['strategy']
        
        logging.info(f"Dataset size: {info['size_mb']:.1f}MB, Strategy: {strategy}")
        
        if strategy == 'memory':
            # Small dataset - no sharding needed
            return [data_path]
        
        # Create shards
        shard_paths = []
        current_shard = []
        current_shard_size = 0
        shard_index = 0
        
        shard_size_limit = self.config.max_shard_size_mb * 1024 * 1024
        
        logging.info(f"Creating shards with {shard_size_limit / (1024*1024):.0f}MB limit...")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line_no, line in enumerate(f, 1):
                line_size = len(line.encode('utf-8'))
                
                # Check if we need to start a new shard
                if (current_shard_size + line_size > shard_size_limit and 
                    current_shard and len(current_shard) >= 100):
                    
                    # Save current shard
                    shard_path = self._save_shard(current_shard, split, shard_index)
                    shard_paths.append(shard_path)
                    
                    # Reset for next shard
                    current_shard = []
                    current_shard_size = 0
                    shard_index += 1
                    
                    if shard_index % 10 == 0:
                        logging.info(f"Created {shard_index} shards, processed {line_no:,} lines")
                
                current_shard.append(line.strip())
                current_shard_size += line_size
        
        # Save final shard
        if current_shard:
            shard_path = self._save_shard(current_shard, split, shard_index)
            shard_paths.append(shard_path)
        
        # Save metadata
        self._save_metadata(shard_paths, split, info)
        
        logging.info(f"Created {len(shard_paths)} shards for {split}")
        return shard_paths
    
    def _save_shard(self, conversations: List[str], split: str, shard_index: int) -> Path:
        """Save a shard to disk."""
        shard_path = self.shards_dir / f"{split}_shard_{shard_index:04d}.jsonl"
        
        with open(shard_path, 'w', encoding='utf-8') as f:
            for conv in conversations:
                f.write(conv + '\n')
        
        return shard_path
    
    def _save_metadata(self, shard_paths: List[Path], split: str, info: Dict):
        """Save shard metadata."""
        metadata = {
            'split': split,
            'total_shards': len(shard_paths),
            'shard_paths': [str(p) for p in shard_paths],
            'dataset_info': info,
            'config': {
                'max_shard_size_mb': self.config.max_shard_size_mb,
                'created_at': time.time()
            }
        }
        
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load_metadata(self) -> Optional[Dict]:
        """Load shard metadata."""
        if not self.metadata_file.exists():
            return None
        
        try:
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.warning(f"Failed to load metadata: {e}")
            return None


class ConversationDataset(Dataset):
    """Enhanced dataset with sharding support for datasets from 150MB to 16TB."""
    
    def __init__(self, data_path: str, tokenizer, config, split: str = "train"):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.config = config
        self.split = split
        
        # Sharding configuration
        self.shard_config = ShardConfig(
            max_shard_size_mb=getattr(config, 'max_shard_size_mb', 512),
            max_memory_usage_gb=getattr(config, 'max_memory_usage_gb', 8.0),
            num_workers=getattr(config, 'num_workers', min(8, mp.cpu_count())),
            enable_memory_mapping=getattr(config, 'enable_memory_mapping', True)
        )
        
        # Initialize shard manager
        self.shard_manager = ShardManager(self.data_path.parent, self.shard_config)
        
        # Statistics tracking
        self.stats = {
            'total_loaded': 0,
            'valid_conversations': 0,
            'invalid_conversations': 0,
            'tokenization_errors': 0,
            'avg_token_length': 0,
            'max_token_length': 0,
            'min_token_length': float('inf'),
            'sharding_strategy': 'unknown',
            'memory_usage_mb': 0,
            'total_shards': 0
        }
        
        # Determine loading strategy and load conversations
        self.conversations = self._load_and_validate_conversations()
        self._compute_statistics()
        
        logging.info(f"Dataset {split}: {len(self.conversations):,} conversations from {data_path}")
        logging.info(f"Strategy: {self.stats['sharding_strategy']}")
        logging.info(f"Memory usage: {self.stats['memory_usage_mb']:.1f}MB")
        if self.stats['total_shards'] > 0:
            logging.info(f"Total shards: {self.stats['total_shards']}")
        logging.info(f"Average tokens: {self.stats['avg_token_length']:.1f}, "
                    f"Max: {self.stats['max_token_length']}, Min: {self.stats['min_token_length']}")
    
    def _load_and_validate_conversations(self) -> List[Dict]:
        """Load and validate conversations with automatic sharding strategy."""
        if not self.data_path.exists():
            logging.error(f"Data file not found: {self.data_path}")
            return []
        
        # Estimate dataset characteristics
        info = self.shard_manager.estimate_dataset_size(self.data_path)
        strategy = info['strategy']
        self.stats['sharding_strategy'] = strategy
        
        logging.info(f"Loading {self.split} data from {self.data_path}")
        logging.info(f"Dataset size: {info['size_mb']:.1f}MB, Strategy: {strategy}")
        
        if strategy == 'memory':
            return self._load_small_dataset()
        elif strategy == 'simple_shards':
            return self._load_medium_dataset()
        else:  # advanced_shards
            return self._load_large_dataset()
    
    def _load_small_dataset(self) -> List[Dict]:
        """Load small datasets (< 500MB) directly into memory."""
        conversations = []
        
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
                    if line_no <= 10:
                        logging.warning(f"JSON decode error at line {line_no}: {e}")
                except Exception as e:
                    self.stats['invalid_conversations'] += 1
                    logging.warning(f"Error loading conversation {line_no}: {e}")
        
        # Estimate memory usage
        import sys
        self.stats['memory_usage_mb'] = sys.getsizeof(conversations) / (1024 * 1024)
        
        return conversations
    
    def _load_medium_dataset(self) -> List[Dict]:
        """Load medium datasets (500MB - 5GB) with simple sharding."""
        # Check if shards already exist
        metadata = self.shard_manager.load_metadata()
        
        if metadata and self._validate_existing_shards(metadata):
            logging.info("Using existing shards")
            return self._load_from_shards(metadata['shard_paths'])
        
        # Create new shards
        logging.info("Creating shards for medium dataset...")
        shard_paths = self.shard_manager.create_shards(self.data_path, self.split)
        self.stats['total_shards'] = len(shard_paths)
        
        return self._load_from_shards(shard_paths)
    
    def _load_large_dataset(self) -> List[Dict]:
        """Load large datasets (> 5GB) with advanced sharding and memory management."""
        # For very large datasets, we use a different approach
        # Load shard index only, then stream conversations as needed
        
        metadata = self.shard_manager.load_metadata()
        
        if metadata and self._validate_existing_shards(metadata):
            logging.info("Using existing shards for large dataset")
            shard_paths = [Path(p) for p in metadata['shard_paths']]
        else:
            logging.info("Creating shards for large dataset...")
            shard_paths = self.shard_manager.create_shards(self.data_path, self.split)
        
        self.stats['total_shards'] = len(shard_paths)
        
        # For large datasets, create a conversation index instead of loading all
        return self._create_conversation_index(shard_paths)
    
    def _validate_existing_shards(self, metadata: Dict) -> bool:
        """Validate that existing shards are complete and accessible."""
        try:
            for shard_path_str in metadata.get('shard_paths', []):
                shard_path = Path(shard_path_str)
                if not shard_path.exists() or shard_path.stat().st_size == 0:
                    return False
            return True
        except Exception:
            return False
    
    def _load_from_shards(self, shard_paths: List[Path]) -> List[Dict]:
        """Load conversations from shard files."""
        conversations = []
        total_shards = len(shard_paths)
        
        logging.info(f"Loading from {total_shards} shards...")
        
        for i, shard_path in enumerate(shard_paths):
            if i % 10 == 0 and i > 0:
                logging.info(f"Processed {i}/{total_shards} shards, {len(conversations):,} conversations")
            
            try:
                with open(shard_path, 'r', encoding='utf-8') as f:
                    for line_no, line in enumerate(f, 1):
                        try:
                            conversation = json.loads(line.strip())
                            self.stats['total_loaded'] += 1
                            
                            if self._validate_conversation(conversation):
                                conversations.append(conversation)
                                self.stats['valid_conversations'] += 1
                            else:
                                self.stats['invalid_conversations'] += 1
                                
                        except json.JSONDecodeError:
                            self.stats['invalid_conversations'] += 1
                        except Exception:
                            self.stats['invalid_conversations'] += 1
            
            except Exception as e:
                logging.warning(f"Failed to load shard {shard_path}: {e}")
                continue
        
        # Estimate memory usage
        import sys
        self.stats['memory_usage_mb'] = sys.getsizeof(conversations) / (1024 * 1024)
        
        return conversations
    
    def _create_conversation_index(self, shard_paths: List[Path]) -> List[Dict]:
        """Create a lightweight index for very large datasets."""
        # For massive datasets, we create an index that points to conversations
        # rather than loading them all into memory
        
        conversation_index = []
        
        for shard_idx, shard_path in enumerate(shard_paths):
            if shard_idx % 50 == 0:
                logging.info(f"Indexing shard {shard_idx}/{len(shard_paths)}")
            
            try:
                with open(shard_path, 'r', encoding='utf-8') as f:
                    for line_idx, line in enumerate(f):
                        try:
                            # Quick validation without full parsing
                            if line.strip() and '"messages"' in line:
                                conversation_index.append({
                                    'shard_path': str(shard_path),
                                    'shard_index': shard_idx,
                                    'line_index': line_idx,
                                    'estimated_size': len(line)
                                })
                                self.stats['valid_conversations'] += 1
                        except Exception:
                            self.stats['invalid_conversations'] += 1
                            continue
            
            except Exception as e:
                logging.warning(f"Failed to index shard {shard_path}: {e}")
                continue
        
        self.stats['memory_usage_mb'] = len(conversation_index) * 0.001  # Minimal memory usage
        logging.info(f"Created index for {len(conversation_index):,} conversations")
        
        return conversation_index
    
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
    
    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        return self.stats.copy()


class ShardedDataLoader:
    """Custom dataloader that efficiently handles sharded datasets."""
    
    def __init__(self, dataset: Union[ConversationDataset, StreamingConversationDataset], 
                 config, shuffle: bool = True):
        self.dataset = dataset
        self.config = config
        self.shuffle = shuffle
        self.current_epoch = 0
        
        # Determine if we need streaming for large datasets
        if hasattr(dataset, 'stats') and dataset.stats.get('sharding_strategy') == 'advanced_shards':
            self.use_streaming = True
        else:
            self.use_streaming = False
    
    def __iter__(self):
        """Create iterator based on dataset size."""
        if self.use_streaming:
            return self._create_streaming_loader()
        else:
            return self._create_standard_loader()
    
    def _create_streaming_loader(self):
        """Create streaming dataloader for massive datasets."""
        # Convert to streaming dataset if needed
        if isinstance(self.dataset, ConversationDataset):
            streaming_dataset = StreamingConversationDataset(
                str(self.dataset.data_path), 
                self.dataset.tokenizer, 
                self.dataset.config, 
                self.dataset.split
            )
        else:
            streaming_dataset = self.dataset
        
        dataloader = DataLoader(
            streaming_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
            persistent_workers=self.config.num_workers > 0
        )
        
        return iter(dataloader)
    
    def _create_standard_loader(self):
        """Create standard dataloader for smaller datasets."""
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=self.shuffle,
            num_workers=self.config.num_workers,
            pin_memory=torch.cuda.is_available(),
            prefetch_factor=2 if self.config.num_workers > 0 else None,
            drop_last=True,
            persistent_workers=self.config.num_workers > 0
        )
        
        return iter(dataloader)


def create_dataloader(dataset: ConversationDataset, config, shuffle: bool = True) -> DataLoader:
    """Create optimized dataloader with automatic sharding support."""
    try:
        # Check if we should use streaming for very large datasets
        if hasattr(dataset, 'stats') and dataset.stats.get('sharding_strategy') == 'advanced_shards':
            logging.info("Using streaming dataloader for large dataset")
            
            # Create streaming dataset
            streaming_dataset = StreamingConversationDataset(
                str(dataset.data_path), 
                dataset.tokenizer, 
                dataset.config, 
                dataset.split
            )
            
            return DataLoader(
                streaming_dataset,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
                pin_memory=torch.cuda.is_available(),
                drop_last=True,
                persistent_workers=config.num_workers > 0
            )
        
        # Standard dataloader for smaller datasets
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


# Enhanced dataset download functions with sharding support

def setup_output_directory(project_root: Optional[str] = None) -> Path:
    """Setup and create output directory for dataset files."""
    if project_root is None:
        script_dir = Path(__file__).parent.absolute()
        project_root = script_dir
    
    output_dir = Path(project_root) / "oasst1_data"
    
    # Ensure the directory exists
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created output directory: {output_dir}")
    except Exception as e:
        logging.error(f"Failed to create output directory: {e}")
        raise
    
    logging.info(f"Project root: {project_root}")
    logging.info(f"Output directory: {output_dir}")
    
    return output_dir


def build_conversation_tree(messages: List[Dict]) -> tuple[Dict[str, Dict], List[str]]:
    """Build a tree structure from messages using parent-child relationships."""
    message_map = {}
    
    # First pass: create message map
    for msg in messages:
        message_map[msg['message_id']] = {
            'data': msg,
            'children': []
        }
    
    # Second pass: build parent-child relationships
    root_messages = []
    for msg in messages:
        parent_id = msg.get('parent_id')
        if parent_id and parent_id in message_map:
            message_map[parent_id]['children'].append(msg['message_id'])
        else:
            # This is a root message (conversation starter)
            root_messages.append(msg['message_id'])
    
    return message_map, root_messages


def extract_conversation_paths(message_map: Dict, root_id: str) -> List[List[Dict]]:
    """Extract all possible conversation paths from a root message."""
    conversations = []
    
    def dfs_path(node_id: str, current_path: List[Dict]):
        if node_id not in message_map:
            return
        
        node = message_map[node_id]
        new_path = current_path + [node['data']]
        
        # Save conversation at multiple points to generate more data
        if len(new_path) >= 2:  # At least one exchange
            conversations.append(new_path.copy())
        
        # If this is a leaf node or has no valid children, we're done with this path
        if not node['children']:
            return
        else:
            # Continue down each child path
            for child_id in node['children']:
                dfs_path(child_id, new_path)
    
    dfs_path(root_id, [])
    return conversations


def format_conversation(messages: List[Dict]) -> Dict:
    """Format a conversation path into a structured format."""
    conversation = {
        'conversation_id': messages[0].get('message_tree_id', ''),
        'messages': [],
        'total_turns': len(messages),
        'languages': list(set(msg.get('lang', 'en') for msg in messages)),
        'created_date': messages[0].get('created_date', ''),
        'tree_state': messages[0].get('tree_state', '')
    }
    
    for i, msg in enumerate(messages):
        formatted_msg = {
            'turn': i + 1,
            'role': msg.get('role', '').lower(),
            'content': msg.get('text', '').strip(),
            'message_id': msg.get('message_id', ''),
            'review_result': msg.get('review_result', None),
            'rank': msg.get('rank', 0),
            'synthetic': msg.get('synthetic', False),
            'model_name': msg.get('model_name', '')
        }
        conversation['messages'].append(formatted_msg)
    
    return conversation


def filter_quality_conversations(conversations: List[Dict], strict_filtering: bool = False) -> List[Dict]:
    """Filter conversations based on quality criteria with sharding-aware processing."""
    quality_conversations = []
    
    for conv in conversations:
        # Quality checks
        valid = True
        
        # Must have at least 2 turns (one exchange)
        if len(conv['messages']) < 2:
            continue
            
        # Must start with a prompter (user)
        if conv['messages'][0]['role'] != 'prompter':
            continue
            
        # Must alternate between prompter and assistant
        expected_role = 'prompter'
        for msg in conv['messages']:
            if msg['role'] != expected_role:
                valid = False
                break
            expected_role = 'assistant' if expected_role == 'prompter' else 'prompter'
        
        if not valid:
            continue
        
        # Relaxed filtering to maximize dataset size
        has_content = True
        for msg in conv['messages']:
            if not msg['content'] or len(msg['content'].strip()) == 0:
                has_content = False
                break
                
        if not has_content:
            continue
            
        quality_conversations.append(conv)
    
    return quality_conversations


class ShardedDatasetProcessor:
    """Processes datasets with automatic sharding for any size."""
    
    def __init__(self, output_dir: Path, shard_config: Optional[ShardConfig] = None):
        self.output_dir = output_dir
        self.shard_config = shard_config or ShardConfig()
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def process_dataset_with_sharding(self, dataset, split_name: str) -> List[Path]:
        """Process dataset with automatic sharding based on size."""
        
        # Estimate memory requirements
        estimated_size_mb = self._estimate_dataset_memory(dataset)
        logging.info(f"Estimated dataset size: {estimated_size_mb:.1f}MB")
        
        if estimated_size_mb < 500:  # Small dataset
            return self._process_small_dataset(dataset, split_name)
        elif estimated_size_mb < 5000:  # Medium dataset  
            return self._process_medium_dataset(dataset, split_name)
        else:  # Large dataset
            return self._process_large_dataset(dataset, split_name)
    
    def _estimate_dataset_memory(self, dataset) -> float:
        """Estimate memory requirements for dataset."""
        try:
            # Sample a few items to estimate memory usage
            sample_size = min(100, len(dataset))
            total_size = 0
            
            for i in range(sample_size):
                item = dataset[i]
                total_size += len(json.dumps(item).encode('utf-8'))
            
            avg_size = total_size / sample_size
            return (avg_size * len(dataset)) / (1024 * 1024)
        
        except Exception:
            return 1000  # Default estimate
    
    def _process_small_dataset(self, dataset, split_name: str) -> List[Path]:
        """Process small datasets without sharding."""
        output_file = self.output_dir / f"oasst1_{split_name}.jsonl"
        
        logging.info(f"Processing small dataset: {len(dataset):,} items")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in dataset:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        return [output_file]
    
    def _process_medium_dataset(self, dataset, split_name: str) -> List[Path]:
        """Process medium datasets with simple sharding."""
        conversations_per_shard = self.shard_config.buffer_size
        shard_paths = []
        
        logging.info(f"Processing medium dataset with sharding: {len(dataset):,} items")
        
        for shard_idx in range(0, len(dataset), conversations_per_shard):
            end_idx = min(shard_idx + conversations_per_shard, len(dataset))
            shard_data = dataset[shard_idx:end_idx]
            
            shard_path = self.output_dir / f"oasst1_{split_name}_shard_{shard_idx//conversations_per_shard:04d}.jsonl"
            
            with open(shard_path, 'w', encoding='utf-8') as f:
                for item in shard_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            shard_paths.append(shard_path)
            
            if len(shard_paths) % 10 == 0:
                logging.info(f"Created {len(shard_paths)} shards...")
        
        return shard_paths
    
    def _process_large_dataset(self, dataset, split_name: str) -> List[Path]:
        """Process large datasets with advanced sharding and parallel processing."""
        
        def process_chunk(chunk_data):
            """Process a chunk of conversations in parallel."""
            chunk_idx, conversations_chunk = chunk_data
            processed_conversations = []
            
            for conv in conversations_chunk:
                try:
                    # Apply filtering
                    if self._validate_large_conversation(conv):
                        processed_conversations.append(conv)
                except Exception:
                    continue
            
            return chunk_idx, processed_conversations
        
        # Split dataset into chunks for parallel processing
        chunk_size = max(1000, len(dataset) // (self.shard_config.num_workers * 4))
        chunks = []
        
        for i in range(0, len(dataset), chunk_size):
            end_idx = min(i + chunk_size, len(dataset))
            chunks.append((i // chunk_size, dataset[i:end_idx]))
        
        logging.info(f"Processing large dataset: {len(dataset):,} items in {len(chunks)} chunks")
        
        # Process chunks in parallel
        processed_conversations = []
        
        with ProcessPoolExecutor(max_workers=self.shard_config.num_workers) as executor:
            future_to_chunk = {executor.submit(process_chunk, chunk): chunk[0] for chunk in chunks}
            
            for future in as_completed(future_to_chunk):
                try:
                    chunk_idx, chunk_conversations = future.result()
                    processed_conversations.extend(chunk_conversations)
                    
                    if len(processed_conversations) % 10000 == 0:
                        logging.info(f"Processed {len(processed_conversations):,} conversations...")
                        
                except Exception as e:
                    logging.warning(f"Chunk processing failed: {e}")
        
        # Create shards from processed conversations
        return self._create_shards_from_conversations(processed_conversations, split_name)
    
    def _validate_large_conversation(self, conversation: Dict) -> bool:
        """Fast validation for large dataset processing."""
        return ('messages' in conversation and 
                conversation['messages'] and 
                len(conversation['messages']) >= 2)
    
    def _create_shards_from_conversations(self, conversations: List[Dict], split_name: str) -> List[Path]:
        """Create shards from processed conversations."""
        shard_paths = []
        shard_size_limit = self.shard_config.max_shard_size_mb * 1024 * 1024
        
        current_shard = []
        current_shard_size = 0
        shard_index = 0
        
        for conv in conversations:
            conv_size = len(json.dumps(conv).encode('utf-8'))
            
            # Check if we need to start a new shard
            if (current_shard_size + conv_size > shard_size_limit and 
                current_shard and len(current_shard) >= 100):
                
                # Save current shard
                shard_path = self._save_conversation_shard(current_shard, split_name, shard_index)
                shard_paths.append(shard_path)
                
                # Reset for next shard
                current_shard = []
                current_shard_size = 0
                shard_index += 1
                
                if shard_index % 10 == 0:
                    logging.info(f"Created {shard_index} shards...")
            
            current_shard.append(conv)
            current_shard_size += conv_size
        
        # Save final shard
        if current_shard:
            shard_path = self._save_conversation_shard(current_shard, split_name, shard_index)
            shard_paths.append(shard_path)
        
        return shard_paths
    
    def _save_conversation_shard(self, conversations: List[Dict], split_name: str, shard_index: int) -> Path:
        """Save a shard of conversations."""
        shard_path = self.output_dir / f"oasst1_{split_name}_shard_{shard_index:04d}.jsonl"
        
        with open(shard_path, 'w', encoding='utf-8') as f:
            for conv in conversations:
                f.write(json.dumps(conv, ensure_ascii=False) + '\n')
        
        return shard_path


def analyze_conversations(conversations: List[Dict], split_name: str):
    """Analyze the structure and quality of conversations with memory efficiency."""
    if not conversations:
        logging.warning(f"No conversations found in {split_name}")
        return
    
    # For very large datasets, sample for analysis
    if len(conversations) > 10000:
        sample_size = 5000
        sample_indices = np.random.choice(len(conversations), sample_size, replace=False)
        analysis_conversations = [conversations[i] for i in sample_indices]
        logging.info(f"Analyzing sample of {sample_size:,} conversations from {len(conversations):,} total")
    else:
        analysis_conversations = conversations
    
    total_conversations = len(conversations)  # Use full count
    total_turns = sum(len(conv['messages']) for conv in analysis_conversations)
    avg_turns = total_turns / len(analysis_conversations) if analysis_conversations else 0
    
    from collections import defaultdict
    turn_distribution = defaultdict(int)
    role_counts = defaultdict(int)
    
    for conv in analysis_conversations:
        turn_distribution[len(conv['messages'])] += 1
        for msg in conv['messages']:
            role_counts[msg['role']] += 1
    
    logging.info(f"Conversation Analysis for {split_name}:")
    logging.info(f"  Total conversations: {total_conversations:,}")
    logging.info(f"  Analyzed conversations: {len(analysis_conversations):,}")
    logging.info(f"  Average turns per conversation: {avg_turns:.1f}")
    logging.info(f"  Turn distribution (top 10):")
    
    for turns, count in sorted(turn_distribution.items())[:10]:
        pct = count / len(analysis_conversations) * 100
        logging.info(f"    {turns} turns: {count:,} conversations ({pct:.1f}%)")
    
    logging.info(f"  Role distribution:")
    total_role_messages = sum(role_counts.values())
    for role, count in role_counts.items():
        pct = count / total_role_messages * 100 if total_role_messages > 0 else 0
        logging.info(f"    {role}: {count:,} messages ({pct:.1f}%)")


def download_and_process_conversations(output_dir: Path) -> bool:
    """Download OASST1 dataset and process with automatic sharding."""
    try:
        logging.info("Loading OpenAssistant dataset (oasst1)...")
        logging.info("This may take a few minutes for the first download...")
        
        # Load dataset with error handling
        try:
            from datasets import load_dataset
            ds = load_dataset("OpenAssistant/oasst1", trust_remote_code=True)
        except Exception as e:
            logging.error(f"Failed to load dataset: {e}")
            logging.info("Possible solutions:")
            logging.info("  1. Try running: huggingface-cli login")
            logging.info("  2. Install git-lfs if not installed")
            logging.info("  3. Check your internet connection")
            return False
        
        logging.info("Dataset loaded successfully!")
        logging.info(f"Available splits: {list(ds.keys())}")
        
        # Initialize sharded processor
        processor = ShardedDatasetProcessor(output_dir)
        
        # Process train, validation, and evaluation splits
        splits_to_process = ['train', 'validation']
        
        # Check for test/eval splits
        if 'test' in ds:
            splits_to_process.append('test')
        elif 'eval' in ds:
            splits_to_process.append('eval')
        elif 'evaluation' in ds:
            splits_to_process.append('evaluation')
        
        for split_name in splits_to_process:
            if split_name not in ds:
                logging.warning(f"Split '{split_name}' not found in dataset")
                continue
                
            logging.info(f"Processing {split_name} split...")
            
            # Get split data
            split_data = ds[split_name]
            logging.info(f"Total messages in {split_name}: {len(split_data):,}")
            
            # Process in chunks to manage memory for large datasets
            all_conversations = []
            chunk_size = min(50000, max(1000, len(split_data) // 20))  # Adaptive chunk size
            
            for chunk_start in range(0, len(split_data), chunk_size):
                chunk_end = min(chunk_start + chunk_size, len(split_data))
                chunk_data = split_data[chunk_start:chunk_end]
                
                logging.info(f"Processing chunk {chunk_start//chunk_size + 1}, "
                           f"messages {chunk_start:,}-{chunk_end:,}")
                
                # Filter valid messages in chunk
                valid_messages = []
                for msg in chunk_data:
                    if (msg.get('text', '').strip() and msg.get('message_tree_id')):
                        valid_messages.append(msg)
                
                # Group by conversation tree
                from collections import defaultdict
                tree_messages = defaultdict(list)
                for msg in valid_messages:
                    tree_id = msg.get('message_tree_id', '')
                    if tree_id:
                        tree_messages[tree_id].append(msg)
                
                # Build conversations from trees
                chunk_conversations = []
                for tree_id, messages in tree_messages.items():
                    message_map, root_messages = build_conversation_tree(messages)
                    
                    for root_id in root_messages:
                        paths = extract_conversation_paths(message_map, root_id)
                        for path in paths:
                            conv = format_conversation(path)
                            chunk_conversations.append(conv)
                            
                            # Create sub-conversations for longer paths
                            if len(path) > 4:
                                for start_idx in range(0, len(path) - 3, 2):
                                    if start_idx > 0:
                                        sub_path = path[start_idx:]
                                        if len(sub_path) >= 2:
                                            sub_conv = format_conversation(sub_path)
                                            sub_conv['conversation_id'] = f"{sub_conv['conversation_id']}_sub_{start_idx}"
                                            chunk_conversations.append(sub_conv)
                
                # Filter quality conversations
                filtered_chunk = filter_quality_conversations(chunk_conversations, strict_filtering=False)
                all_conversations.extend(filtered_chunk)
                
                # Clear memory
                del chunk_data, valid_messages, tree_messages, chunk_conversations, filtered_chunk
                gc.collect()
                
                if len(all_conversations) % 25000 == 0:
                    logging.info(f"Total conversations processed: {len(all_conversations):,}")
            
            logging.info(f"Total conversations extracted from {split_name}: {len(all_conversations):,}")
            
            # Create shards based on dataset size
            shard_paths = processor.process_dataset_with_sharding(all_conversations, split_name)
            
            # Analyze dataset (using sampling for large datasets)
            logging.info(f"Analysis for {split_name}")
            analyze_conversations(all_conversations, split_name)
            
            # Clear memory
            del all_conversations
            gc.collect()
            
            logging.info(f"Created {len(shard_paths)} shard files for {split_name}")
        
        return True
        
    except Exception as e:
        logging.error(f"Error processing conversations: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_conversation_files(output_dir: Path) -> bool:
    """Validate that conversation files exist and are readable."""
    # Get all conversation files in the directory (including shards)
    conversation_files = list(output_dir.glob("oasst1_*.jsonl"))
    
    if not conversation_files:
        logging.error("No conversation files found")
        return False
    
    for file_path in conversation_files:
        if not file_path.exists():
            logging.error(f"Missing file: {file_path}")
            return False
        
        if file_path.stat().st_size == 0:
            logging.error(f"Empty file: {file_path}")
            return False
        
        # Test reading and parsing (sample only for large files)
        try:
            line_count = 0
            sample_lines = 3 if file_path.stat().st_size < 100 * 1024 * 1024 else 1  # Sample fewer lines for large files
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    line_count += 1
                    
                    # Test parsing of sample lines
                    if i < sample_lines:
                        line = line.strip()
                        if not line:
                            continue
                        
                        try:
                            data = json.loads(line)
                            # Check conversation structure
                            required_fields = ['conversation_id', 'messages', 'total_turns']
                            for field in required_fields:
                                if field not in data:
                                    logging.error(f"Missing field '{field}' in {file_path}")
                                    return False
                            
                            # Check message structure
                            if data['messages']:
                                msg = data['messages'][0]
                                msg_fields = ['role', 'content', 'turn']
                                for field in msg_fields:
                                    if field not in msg:
                                        logging.error(f"Missing message field '{field}' in {file_path}")
                                        return False
                                        
                        except json.JSONDecodeError as e:
                            logging.error(f"Invalid JSON in {file_path} line {i+1}: {e}")
                            return False
            
            logging.info(f"{file_path.name}: {line_count:,} conversations validated")
                        
        except Exception as e:
            logging.error(f"Cannot read file {file_path}: {e}")
            return False
    
    logging.info("All conversation files validated successfully!")
    return True


def check_existing_files(output_dir: Path) -> bool:
    """Check if dataset files already exist and are valid."""
    # Look for any existing conversation files (including shards)
    existing_files = list(output_dir.glob("oasst1_*.jsonl"))
    
    if existing_files:
        logging.info("Found existing conversation files!")
        total_size_mb = 0
        for file_path in existing_files:
            size_mb = file_path.stat().st_size / (1024 * 1024)
            total_size_mb += size_mb
            logging.info(f"  - {file_path.name}: {size_mb:.1f} MB")
        
        logging.info(f"Total dataset size: {total_size_mb:.1f} MB")
        logging.info("Validating existing files...")
        return validate_conversation_files(output_dir)
    
    return False


def create_memory_efficient_dataloader(dataset_path: str, tokenizer, config, split: str = "train") -> Union[DataLoader, Iterator]:
    """Create the most appropriate dataloader based on dataset size."""
    
    # Estimate dataset size first
    data_path = Path(dataset_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    file_size_mb = data_path.stat().st_size / (1024 * 1024)
    
    if file_size_mb < 500:
        # Small dataset - use standard ConversationDataset
        logging.info(f"Using standard dataset loading for {file_size_mb:.1f}MB dataset")
        dataset = ConversationDataset(dataset_path, tokenizer, config, split)
        return create_dataloader(dataset, config, shuffle=(split == "train"))
    
    elif file_size_mb < 10000:  # 10GB
        # Medium dataset - use sharded ConversationDataset
        logging.info(f"Using sharded dataset loading for {file_size_mb:.1f}MB dataset")
        dataset = ConversationDataset(dataset_path, tokenizer, config, split)
        return create_dataloader(dataset, config, shuffle=(split == "train"))
    
    else:
        # Very large dataset - use streaming
        logging.info(f"Using streaming dataset loading for {file_size_mb:.1f}MB dataset")
        dataset = StreamingConversationDataset(dataset_path, tokenizer, config, split)
        return DataLoader(
            dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
            persistent_workers=config.num_workers > 0
        )


def main():
    """Enhanced main function with sharding support for datasets of any size."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    logger.info("Starting OASST1 Conversational Dataset Download with Sharding Support...")
    logger.info("Supports datasets from 150MB to 16TB+")
    logger.info("=" * 70)
    
    try:
        # Setup directories
        output_dir = setup_output_directory()
        
        # Check if files already exist and are valid
        if check_existing_files(output_dir):
            logger.info("Valid dataset files already exist!")
            logger.info("Delete files in the output directory if you want to re-download")
            logger.info("=" * 70)
            return 0
        else:
            logger.info("Downloading/reprocessing dataset with sharding...")
        
        # Download and process conversations with sharding
        success = download_and_process_conversations(output_dir)
        
        if not success:
            logger.error("Dataset processing failed!")
            return 1
        
        # Validate files
        if not validate_conversation_files(output_dir):
            logger.error("File validation failed!")
            return 1
        
        # Success summary
        logger.info("=" * 70)
        logger.info("Conversational dataset preparation completed!")
        logger.info(f"Files saved in: {output_dir}")
        logger.info("")
        logger.info("Generated Files:")
        
        # List all generated files with sizes and shard info
        total_size_mb = 0
        total_conversations = 0
        shard_files = []
        regular_files = []
        
        for file_path in sorted(output_dir.glob("*.jsonl")):
            size_mb = file_path.stat().st_size / (1024 * 1024)
            total_size_mb += size_mb
            
            try:
                line_count = sum(1 for _ in open(file_path, 'r', encoding='utf-8'))
                total_conversations += line_count
                
                if "_shard_" in file_path.name:
                    shard_files.append((file_path, size_mb, line_count))
                else:
                    regular_files.append((file_path, size_mb, line_count))
                    
            except Exception:
                logger.warning(f"Could not count lines in {file_path.name}")
                continue
        
        # Display regular files
        for file_path, size_mb, line_count in regular_files:
            logger.info(f"   {file_path.name}: {size_mb:.1f} MB, {line_count:,} conversations")
        
        # Display shard summary
        if shard_files:
            shard_count = len(shard_files)
            shard_size_mb = sum(size for _, size, _ in shard_files)
            shard_conversations = sum(count for _, _, count in shard_files)
            
            logger.info(f"   Sharded files: {shard_count} shards, {shard_size_mb:.1f} MB total, {shard_conversations:,} conversations")
            
            # Show first few shards as examples
            for file_path, size_mb, line_count in shard_files[:3]:
                logger.info(f"     {file_path.name}: {size_mb:.1f} MB, {line_count:,} conversations")
            
            if shard_count > 3:
                logger.info(f"     ... and {shard_count - 3} more shard files")
        
        logger.info("")
        logger.info(f"Dataset Summary:")
        logger.info(f"  Total size: {total_size_mb:.1f} MB ({total_size_mb/1024:.1f} GB)")
        logger.info(f"  Total conversations: {total_conversations:,}")
        logger.info(f"  Loading strategy: {'Sharded' if shard_files else 'In-memory'}")
        
        if total_size_mb > 1000:  # > 1GB
            logger.info(f"  Estimated training time: {total_conversations // 1000:.0f}+ hours")
            logger.info(f"  Memory efficient: Yes")
        
        logger.info("")
        logger.info("Dataset Format:")
        logger.info("   Each line contains a complete conversation with:")
        logger.info("   - conversation_id: Unique identifier")
        logger.info("   - messages: Array of turn-by-turn exchanges")
        logger.info("   - total_turns: Number of messages in conversation")
        logger.info("   - Each message has: role, content, turn number")
        logger.info("")
        
        if shard_files:
            logger.info("Sharding Information:")
            logger.info("   - Dataset automatically sharded for memory efficiency")
            logger.info("   - ConversationDataset will handle shards transparently")
            logger.info("   - StreamingConversationDataset available for massive datasets")
            logger.info("")
        
        logger.info("Ready for conversational training!")
        logger.info("Usage examples:")
        logger.info("   # Small datasets (auto-detected)")
        logger.info("   dataset = ConversationDataset('path/to/data.jsonl', tokenizer, config)")
        logger.info("")
        logger.info("   # Large datasets (auto-detected, uses sharding)")
        logger.info("   dataset = ConversationDataset('path/to/data.jsonl', tokenizer, config)")
        logger.info("")
        logger.info("   # Massive datasets (streaming)")
        logger.info("   dataset = StreamingConversationDataset('path/to/data.jsonl', tokenizer, config)")
        logger.info("")
        logger.info("   # Memory-efficient dataloader (auto-selects best strategy)")
        logger.info("   dataloader = create_memory_efficient_dataloader('path/to/data.jsonl', tokenizer, config)")
        logger.info("=" * 70)
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
                
    
    def _compute_statistics(self):
        """Compute dataset statistics."""
        if not self.conversations:
            return
        
        token_lengths = []
        
        # For large datasets, sample more efficiently
        if self.stats['sharding_strategy'] == 'advanced_shards':
            # Sample fewer conversations for massive datasets
            sample_size = min(500, len(self.conversations))
        else:
            sample_size = min(1000, len(self.conversations))
        
        if len(self.conversations) > sample_size:
            sample_indices = np.random.choice(len(self.conversations), sample_size, replace=False)
        else:
            sample_indices = range(len(self.conversations))
        
        for idx in sample_indices:
            try:
                if self.stats['sharding_strategy'] == 'advanced_shards':
                    # For indexed conversations, load on demand
                    conversation = self._load_conversation_by_index(idx)
                else:
                    conversation = self.conversations[idx]
                
                if conversation:
                    tokens = self.tokenizer.encode_conversation(conversation)
                    if tokens:
                        token_lengths.append(len(tokens))
            except Exception:
                self.stats['tokenization_errors'] += 1
        
        if token_lengths:
            self.stats['avg_token_length'] = np.mean(token_lengths)
            self.stats['max_token_length'] = max(token_lengths)
            self.stats['min_token_length'] = min(token_lengths)
    
    def _load_conversation_by_index(self, idx: int) -> Optional[Dict]:
        """Load a conversation by index for large datasets."""
        if self.stats['sharding_strategy'] != 'advanced_shards':
            return self.conversations[idx] if idx < len(self.conversations) else None
        
        # For indexed datasets, load the conversation from its shard
        if idx >= len(self.conversations):
            return None
        
        index_entry = self.conversations[idx]
        shard_path = Path(index_entry['shard_path'])
        line_idx = index_entry['line_index']
        
        try:
            with open(shard_path, 'r', encoding='utf-8') as f:
                for current_line_idx, line in enumerate(f):
                    if current_line_idx == line_idx:
                        return json.loads(line.strip())
        except Exception as e:
            logging.debug(f"Failed to load conversation {idx}: {e}")
            return None
        
        return None
    
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
        """Get processed conversation with fallback and smart loading."""
        # Load conversation based on strategy
        if self.stats['sharding_strategy'] == 'advanced_shards':
            conversation = self._load_conversation_by_index(idx)
        else:
            conversation = self.conversations[idx] if idx < len(self.conversations) else None
        
        if conversation is None:
            # Return dummy sample if loading fails
            seq_len = self.config.seq_length - 1
            return {
                'input_ids': torch.zeros(seq_len, dtype=torch.long),
                'labels': torch.zeros(seq_len, dtype=torch.long),
                'attention_mask': torch.zeros(seq_len, dtype=torch.float),
                'loss_weights': torch.zeros(seq_len, dtype=torch.float)
            }
        
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
    """Streaming dataset for massive datasets that don't fit in memory."""
    
    def __init__(self, data_path: str, tokenizer, config, split: str = "train"):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.config = config
        self.split = split
        
        # Initialize shard manager
        self.shard_config = ShardConfig(
            max_shard_size_mb=getattr(config, 'max_shard_size_mb', 512),
            max_memory_usage_gb=getattr(config, 'max_memory_usage_gb', 8.0),
            num_workers=getattr(config, 'num_workers', min(8, mp.cpu_count()))
        )
        
        self.shard_manager = ShardManager(self.data_path.parent, self.shard_config)
        
        # Get or create shards
        metadata = self.shard_manager.load_metadata()
        if metadata and self._validate_existing_shards(metadata):
            self.shard_paths = [Path(p) for p in metadata['shard_paths']]
        else:
            self.shard_paths = self.shard_manager.create_shards(self.data_path, self.split)
        
        self.total_conversations = self._count_total_conversations()
        
        logging.info(f"Streaming dataset {split}: {self.total_conversations:,} conversations "
                    f"across {len(self.shard_paths)} shards")
    
    def _validate_existing_shards(self, metadata: Dict) -> bool:
        """Validate existing shards."""
        try:
            for shard_path_str in metadata.get('shard_paths', []):
                shard_path = Path(shard_path_str)
                if not shard_path.exists() or shard_path.stat().st_size == 0:
                    return False
            return True
        except Exception:
            return False
    
    def _count_total_conversations(self) -> int:
        """Count total conversations across all shards."""
        total = 0
        for shard_path in self.shard_paths:
            try:
                with open(shard_path, 'r', encoding='utf-8') as f:
                    total += sum(1 for _ in f)
            except Exception:
                continue
        return total
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate through conversations across all shards."""
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is None:
            # Single process
            shard_paths = self.shard_paths
        else:
            # Multi-process: divide shards among workers
            shards_per_worker = len(self.shard_paths) // worker_info.num_workers
            start_idx = worker_info.id * shards_per_worker
            if worker_info.id == worker_info.num_workers - 1:
                # Last worker gets remaining shards
                end_idx = len(self.shard_paths)
            else:
                end_idx = start_idx + shards_per_worker
            
            shard_paths = self.shard_paths[start_idx:end_idx]
        
        # Shuffle shards if enabled
        if self.shard_config.shard_shuffle:
            shard_paths = shard_paths.copy()
            random.shuffle(shard_paths)
        
        for shard_path in shard_paths:
            yield from self._iter_shard(shard_path)
    
    def _iter_shard(self, shard_path: Path) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate through conversations in a single shard."""
        try:
            with open(shard_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        conversation = json.loads(line.strip())
                        
                        if self._validate_conversation(conversation):
                            processed = self._process_conversation(conversation)
                            if processed is not None:
                                yield processed
                    
                    except Exception:
                        continue
        
        except Exception as e:
            logging.debug(f"Error reading shard {shard_path}: {e}")
    
    def _validate_conversation(self, conversation: Dict) -> bool:
        """Validate conversation structure."""
        if 'messages' not in conversation:
            return False
        
        messages = conversation['messages']
        if not messages or len(messages) < 2:
            return False
        
        has_user = False
        has_assistant = False
        
        for msg in messages:
            if not isinstance(msg, dict):
                return False
            
            role = msg.get('role', '').lower()
            content = msg.get('content', '').strip()