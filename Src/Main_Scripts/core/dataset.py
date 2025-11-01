# Copyright (c) 2025 MatN23. All rights reserved.
# Licensed under the Custom License below.

"""
LuminaAI Dataset with C++/CUDA Acceleration
Drop-in replacement for dataset.py with transparent speedups

This file is API-compatible with the original dataset.py.
Simply replace 'from core.dataset import ...' with 'from core.dataset_accelerated import ...'
Or rename this file to dataset.py to automatically enable acceleration.
"""

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
import random

# Try to import acceleration backend
try:
    from dataset_accelerator import (
        FastFileReader,
        StreamingIterator as AcceleratedStreamingIterator,
        fast_shuffle,
        parallel_shuffle,
        fast_chunk_documents,
        prepare_batch as accelerated_prepare_batch,
        get_backend_info,
        ACCELERATOR_AVAILABLE,
        CUDA_BACKEND,
    )
    
    if ACCELERATOR_AVAILABLE:
        logging.info("✓ Dataset acceleration enabled")
        if CUDA_BACKEND:
            logging.info("  Using CUDA backend for GPU acceleration")
        else:
            logging.info("  Using C++ backend for CPU multi-threading")
except ImportError as e:
    logging.debug(f"Dataset accelerator not available: {e}")
    ACCELERATOR_AVAILABLE = False
    CUDA_BACKEND = False


class BaseTrainingDataset(Dataset):
    """
    Dataset for base/pre-training on raw text (like The Pile, C4, etc.).
    
    ACCELERATED: Uses C++/CUDA for file loading and chunking when available.
    """
    
    def __init__(self, data_path: str, tokenizer, config, split: str = "train"):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.config = config
        self.split = split
        self.seq_length = config.seq_length
        
        # Statistics tracking
        self.stats = {
            'total_loaded': 0,
            'total_chunks': 0,
            'total_tokens': 0,
            'documents_processed': 0,
            'acceleration_used': ACCELERATOR_AVAILABLE,
            'backend': 'CUDA' if CUDA_BACKEND else ('C++' if ACCELERATOR_AVAILABLE else 'Python'),
        }
        
        # Load and chunk documents
        self.chunks = self._load_and_chunk_documents()
        
        logging.info(f"Base Training Dataset {split}: {len(self.chunks):,} chunks from {data_path}")
        logging.info(f"Total tokens: {self.stats['total_tokens']:,}, Documents: {self.stats['documents_processed']:,}")
        if ACCELERATOR_AVAILABLE:
            logging.info(f"Acceleration: {self.stats['backend']} backend")
    
    def _load_and_chunk_documents(self) -> List[List[int]]:
        """Load documents and create fixed-length token chunks."""
        chunks = []
        current_tokens = []
        
        if not self.data_path.exists():
            logging.error(f"Data file not found: {self.data_path}")
            return chunks
        
        logging.info(f"Loading base training data from {self.data_path}")
        
        # Determine file format
        file_ext = self.data_path.suffix.lower()
        is_jsonl = file_ext == '.jsonl'
        is_txt = file_ext in ['.txt', '.text']
        
        # ACCELERATED: Use FastFileReader if available
        if ACCELERATOR_AVAILABLE:
            try:
                reader = FastFileReader(str(self.data_path))
                lines = reader.read_lines_parallel()
                logging.info(f"✓ Loaded {len(lines):,} lines using accelerated reader")
            except Exception as e:
                logging.warning(f"Accelerated reader failed: {e}, falling back to standard I/O")
                lines = self._read_file_standard()
        else:
            lines = self._read_file_standard()
        
        # Process lines
        texts = []
        for line_no, line in enumerate(lines, 1):
            try:
                # Extract text based on format
                if is_jsonl:
                    data = json.loads(line.strip())
                    text = data.get('text', '')
                elif is_txt:
                    text = line.strip()
                else:
                    text = line.strip()
                
                if not text:
                    continue
                
                self.stats['total_loaded'] += 1
                texts.append(text)
                
                # Progress logging
                if line_no % 10000 == 0:
                    logging.info(f"Processed {line_no:,} lines")
            
            except json.JSONDecodeError as e:
                if is_jsonl:
                    logging.warning(f"JSON decode error at line {line_no}: {e}")
                continue
            except Exception as e:
                logging.warning(f"Error processing line {line_no}: {e}")
                continue
        
        # ACCELERATED: Use fast chunking if available
        if ACCELERATOR_AVAILABLE and len(texts) > 0:
            try:
                chunk_result = fast_chunk_documents(texts, self.seq_length, overlap=True)
                chunks = chunk_result['chunks']
                self.stats['total_tokens'] = chunk_result['total_tokens']
                self.stats['documents_processed'] = chunk_result['documents_processed']
                self.stats['total_chunks'] = len(chunks)
                logging.info(f"✓ Created {len(chunks):,} chunks using accelerated chunking")
                return chunks
            except Exception as e:
                logging.warning(f"Accelerated chunking failed: {e}, using standard method")
        
        # Fallback to standard chunking
        for text in texts:
            tokens = self.tokenizer.tokenizer.encode(text)
            if not tokens:
                continue
            
            self.stats['documents_processed'] += 1
            self.stats['total_tokens'] += len(tokens)
            current_tokens.extend(tokens)
            
            while len(current_tokens) >= self.seq_length + 1:
                chunk = current_tokens[:self.seq_length + 1]
                chunks.append(chunk)
                current_tokens = current_tokens[self.seq_length:]
                self.stats['total_chunks'] += 1
        
        # Handle remaining tokens
        if current_tokens:
            if len(current_tokens) < self.seq_length + 1:
                current_tokens.extend([0] * (self.seq_length + 1 - len(current_tokens)))
            chunks.append(current_tokens[:self.seq_length + 1])
            self.stats['total_chunks'] += 1
        
        return chunks
    
    def _read_file_standard(self) -> List[str]:
        """Standard file reading fallback"""
        lines = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    lines.append(line)
        return lines
    
    def __len__(self) -> int:
        return len(self.chunks)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a tokenized chunk for base training."""
        chunk = self.chunks[idx]
        tokens = torch.tensor(chunk, dtype=torch.long)
        
        # Split into input and labels (next token prediction)
        input_ids = tokens[:-1]
        labels = tokens[1:]
        
        # Create attention mask (all ones for base training)
        attention_mask = torch.ones_like(input_ids, dtype=torch.float)
        
        # Loss weights (train on all tokens for base training)
        loss_weights = torch.ones_like(input_ids, dtype=torch.float)
        
        # Mask padding tokens
        padding_mask = (input_ids == 0)
        attention_mask[padding_mask] = 0.0
        loss_weights[padding_mask] = 0.0
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
            'loss_weights': loss_weights
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        return self.stats.copy()


class StreamingBaseTrainingDataset(IterableDataset):
    """
    Streaming version of base training dataset for very large datasets.
    
    ACCELERATED: Uses C++/CUDA streaming iterator when available.
    """
    
    def __init__(self, 
                 data_path: str, 
                 tokenizer, 
                 config, 
                 split: str = "train",
                 buffer_size: int = 10000):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.config = config
        self.split = split
        self.seq_length = config.seq_length
        self.buffer_size = buffer_size
        
        # Statistics
        self.stats = {
            'total_processed': 0,
            'total_chunks': 0,
            'documents_processed': 0,
            'acceleration_used': ACCELERATOR_AVAILABLE,
        }
        
        # Determine file format
        self.is_jsonl = self.data_path.suffix == '.jsonl'
        
        logging.info(f"StreamingBaseTrainingDataset {split}: {self.data_path}")
        if ACCELERATOR_AVAILABLE:
            logging.info(f"Using accelerated streaming iterator")
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Stream tokenized chunks."""
        # ACCELERATED: Try to use accelerated streaming
        if ACCELERATOR_AVAILABLE:
            try:
                return self._iter_accelerated()
            except Exception as e:
                logging.warning(f"Accelerated streaming failed: {e}, using standard method")
        
        return self._iter_standard()
    
    def _iter_accelerated(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Accelerated streaming using C++/CUDA backend"""
        iterator = AcceleratedStreamingIterator(
            str(self.data_path),
            self.seq_length,
            self.buffer_size
        )
        
        while iterator.has_next():
            chunk = iterator.next_chunk()
            
            if len(chunk) < self.seq_length + 1:
                continue
            
            tokens_tensor = torch.tensor(chunk[:self.seq_length + 1], dtype=torch.long)
            input_ids = tokens_tensor[:-1]
            labels = tokens_tensor[1:]
            attention_mask = torch.ones_like(input_ids, dtype=torch.float)
            loss_weights = torch.ones_like(input_ids, dtype=torch.float)
            
            padding_mask = (input_ids == 0)
            attention_mask[padding_mask] = 0.0
            loss_weights[padding_mask] = 0.0
            
            self.stats['total_chunks'] += 1
            
            yield {
                'input_ids': input_ids,
                'labels': labels,
                'attention_mask': attention_mask,
                'loss_weights': loss_weights
            }
    
    def _iter_standard(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Standard streaming fallback"""
        current_tokens = []
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line_no, line in enumerate(f, 1):
                try:
                    if self.is_jsonl:
                        data = json.loads(line.strip())
                        text = data.get('text', '')
                    else:
                        text = line.strip()
                    
                    if not text:
                        continue
                    
                    self.stats['total_processed'] += 1
                    
                    tokens = self.tokenizer.tokenizer.encode(text)
                    if not tokens:
                        continue
                    
                    self.stats['documents_processed'] += 1
                    current_tokens.extend(tokens)
                    
                    while len(current_tokens) >= self.seq_length + 1:
                        chunk = current_tokens[:self.seq_length + 1]
                        current_tokens = current_tokens[self.seq_length:]
                        
                        tokens_tensor = torch.tensor(chunk, dtype=torch.long)
                        input_ids = tokens_tensor[:-1]
                        labels = tokens_tensor[1:]
                        attention_mask = torch.ones_like(input_ids, dtype=torch.float)
                        loss_weights = torch.ones_like(input_ids, dtype=torch.float)
                        
                        padding_mask = (input_ids == 0)
                        attention_mask[padding_mask] = 0.0
                        loss_weights[padding_mask] = 0.0
                        
                        self.stats['total_chunks'] += 1
                        
                        yield {
                            'input_ids': input_ids,
                            'labels': labels,
                            'attention_mask': attention_mask,
                            'loss_weights': loss_weights
                        }
                    
                    if line_no % 50000 == 0:
                        logging.info(f"Streaming: processed {line_no:,} lines, "
                                   f"{self.stats['total_chunks']:,} chunks yielded")
                
                except Exception as e:
                    logging.debug(f"Error processing line {line_no}: {e}")
                    continue


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
                    if line_no <= 10:
                        logging.warning(f"JSON decode error at line {line_no}: {e}")
                except Exception as e:
                    self.stats['invalid_conversations'] += 1
                    logging.warning(f"Error loading conversation {line_no}: {e}")
                
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
        
        has_user = False
        has_assistant = False
        
        for msg in messages:
            if not isinstance(msg, dict):
                return False
            
            role = msg.get('role', '').lower()
            content = msg.get('content', '').strip()
            
            if not content:
                return False
            
            if role in ['user', 'prompter', 'human']:
                has_user = True
            elif role in ['assistant', 'ai', 'bot']:
                has_assistant = True
        
        return has_user and has_assistant
    
    def _compute_statistics(self):
        """Compute dataset statistics."""
        if not self.conversations:
            return
        
        token_lengths = []
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
            
            if not tokens or len(tokens) < 10:
                return None
            
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
        """Create loss weights - train on both user and assistant content."""
        loss_weights = torch.ones_like(tokens, dtype=torch.float)
        
        im_start_token = self.tokenizer.special_tokens["<|im_start|>"]
        im_end_token = self.tokenizer.special_tokens["<|im_end|>"]
        user_token = self.tokenizer.get_role_token('user')
        assistant_token = self.tokenizer.get_role_token('assistant')
        system_token = self.tokenizer.get_role_token('system')
        
        in_content = False
        current_role_weight = 1.0
        
        for i, token_id in enumerate(tokens):
            if token_id == 0:
                loss_weights[i] = 0.0
            elif token_id == im_start_token:
                in_content = False
                loss_weights[i] = 0.0
            elif token_id in [user_token, assistant_token, system_token]:
                in_content = False
                loss_weights[i] = 0.0
                if token_id == assistant_token:
                    current_role_weight = getattr(self.config, 'assistant_loss_weight', 2.0)
                else:
                    current_role_weight = 1.0
            elif token_id == im_end_token:
                in_content = False
                loss_weights[i] = 0.0
            else:
                in_content = True
                loss_weights[i] = current_role_weight
        
        return loss_weights
    
    def __len__(self) -> int:
        return len(self.conversations)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get processed conversation with fallback."""
        conversation = self.conversations[idx]
        processed = self._process_conversation(conversation)
        
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


class HybridDatasetManager:
    """Manages both base training and instruction tuning datasets."""
    
    def __init__(self, 
                 base_dataset: Optional[Dataset] = None,
                 instruction_dataset: Optional[Dataset] = None,
                 base_ratio: float = 0.7):
        self.base_dataset = base_dataset
        self.instruction_dataset = instruction_dataset
        self.base_ratio = base_ratio
        
        self.total_samples = 0
        if base_dataset is not None:
            self.total_samples += len(base_dataset)  # type: ignore
        if instruction_dataset is not None:
            self.total_samples += len(instruction_dataset)  # type: ignore
    
    def __len__(self) -> int:
        return self.total_samples
    
    def sample(self) -> Dict[str, torch.Tensor]:
        """Sample from either dataset based on ratio."""
        if random.random() < self.base_ratio and self.base_dataset is not None:
            idx = random.randint(0, len(self.base_dataset) - 1)  # type: ignore
            return self.base_dataset[idx]
        elif self.instruction_dataset is not None:
            idx = random.randint(0, len(self.instruction_dataset) - 1)  # type: ignore
            return self.instruction_dataset[idx]
        else:
            # Fallback
            dataset = self.base_dataset or self.instruction_dataset
            if dataset is not None:
                idx = random.randint(0, len(dataset) - 1)  # type: ignore
                return dataset[idx]
            # Return empty if no dataset available
            return {
                'input_ids': torch.zeros(512, dtype=torch.long),
                'labels': torch.zeros(512, dtype=torch.long),
                'attention_mask': torch.zeros(512, dtype=torch.float),
                'loss_weights': torch.zeros(512, dtype=torch.float)
            }


class InterleavedDataset(Dataset):
    """Interleaves base training and fine-tuning samples."""
    
    def __init__(self,
                 base_dataset: Optional[Dataset] = None,
                 instruction_dataset: Optional[Dataset] = None,
                 base_ratio: float = 0.7):
        self.base_dataset = base_dataset
        self.instruction_dataset = instruction_dataset
        self.base_ratio = base_ratio
        
        # Create interleaved indices
        self.indices = self._create_interleaved_indices()
    
    def _create_interleaved_indices(self) -> List[tuple]:
        """Create list of (dataset_type, index) tuples."""
        indices = []
        
        base_len = len(self.base_dataset) if self.base_dataset is not None else 0  # type: ignore
        inst_len = len(self.instruction_dataset) if self.instruction_dataset is not None else 0  # type: ignore
        
        total_len = base_len + inst_len
        num_base = int(total_len * self.base_ratio)
        num_inst = total_len - num_base
        
        # Add base indices
        if base_len > 0:
            for i in range(min(num_base, base_len)):
                indices.append(('base', i % base_len))
        
        # Add instruction indices
        if inst_len > 0:
            for i in range(min(num_inst, inst_len)):
                indices.append(('instruction', i % inst_len))
        
        # Shuffle
        random.shuffle(indices)
        
        return indices
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        dataset_type, data_idx = self.indices[idx]
        
        if dataset_type == 'base' and self.base_dataset is not None:
            return self.base_dataset[data_idx]  # type: ignore
        elif dataset_type == 'instruction' and self.instruction_dataset is not None:
            return self.instruction_dataset[data_idx]  # type: ignore
        else:
            # Fallback to empty sample
            return {
                'input_ids': torch.zeros(512, dtype=torch.long),
                'labels': torch.zeros(512, dtype=torch.long),
                'attention_mask': torch.zeros(512, dtype=torch.float),
                'loss_weights': torch.zeros(512, dtype=torch.float)
            }


def create_dataloader(dataset: Union[Dataset, IterableDataset], 
                     config, 
                     shuffle: bool = True) -> DataLoader:
    """Create optimized dataloader with error handling."""
    try:
        # Check if dataset is streaming
        is_streaming = isinstance(dataset, IterableDataset)
        
        # Also check for custom streaming classes by name
        is_custom_streaming = (
            hasattr(dataset, '__class__') and 
            'Streaming' in dataset.__class__.__name__
        )
        
        if is_streaming or is_custom_streaming:
            return DataLoader(
                dataset,
                batch_size=config.batch_size,
                num_workers=0,  # Streaming datasets don't support num_workers > 0
                pin_memory=torch.cuda.is_available(),
                drop_last=True,
            )
        else:
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
        # Safer fallback
        return DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=shuffle if not isinstance(dataset, IterableDataset) else False,
            num_workers=0,  # Safest option
            drop_last=True
        )


def setup_datasets(config, tokenizer):
    """Main entry point for setting up datasets."""
    train_dataset = None
    eval_dataset = None
    
    # Setup base training dataset if path provided
    if hasattr(config, 'base_train_data') and config.base_train_data:
        if hasattr(config, 'use_streaming') and config.use_streaming:
            train_dataset = StreamingBaseTrainingDataset(
                config.base_train_data,
                tokenizer,
                config,
                split='train'
            )
        else:
            train_dataset = BaseTrainingDataset(
                config.base_train_data,
                tokenizer,
                config,
                split='train'
            )
    
    # Setup instruction tuning dataset if path provided
    if hasattr(config, 'instruction_data') and config.instruction_data:
        instruction_dataset = ConversationDataset(
            config.instruction_data,
            tokenizer,
            config,
            split='train'
        )
        
        # If we also have base training, create hybrid dataset
        if train_dataset:
            base_ratio = getattr(config, 'base_ratio', 0.7)
            train_dataset = InterleavedDataset(
                train_dataset,
                instruction_dataset,
                base_ratio=base_ratio
            )
        else:
            train_dataset = instruction_dataset
    
    # Setup eval dataset if path provided
    if hasattr(config, 'eval_data') and config.eval_data:
        eval_dataset = ConversationDataset(
            config.eval_data,
            tokenizer,
            config,
            split='eval'
        )
    
    return train_dataset, eval_dataset


# Export everything
__all__ = [
    'BaseTrainingDataset',
    'StreamingBaseTrainingDataset',
    'ConversationDataset',
    'HybridDatasetManager',
    'InterleavedDataset',
    'create_dataloader',
    'setup_datasets',
    'ACCELERATOR_AVAILABLE',
    'CUDA_BACKEND',
]