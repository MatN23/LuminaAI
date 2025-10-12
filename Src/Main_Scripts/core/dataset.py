# Copyright (c) 2025 MatN23. All rights reserved.
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
import random


class BaseTrainingDataset(Dataset):
    """
    Dataset for base/pre-training on raw text (like The Pile, C4, etc.).
    
    This dataset:
    - Loads raw text files or JSONL with 'text' field
    - Tokenizes without conversation structure
    - Creates fixed-length chunks for efficient training
    - Supports document continuation across chunks
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
        }
        
        # Load and chunk documents
        self.chunks = self._load_and_chunk_documents()
        
        logging.info(f"Base Training Dataset {split}: {len(self.chunks):,} chunks from {data_path}")
        logging.info(f"Total tokens: {self.stats['total_tokens']:,}, Documents: {self.stats['documents_processed']:,}")
    
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
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line_no, line in enumerate(f, 1):
                try:
                    # Extract text based on format
                    if is_jsonl:
                        # JSONL format: {"text": "content"}
                        data = json.loads(line.strip())
                        text = data.get('text', '')
                    elif is_txt:
                        # Plain text: each line is content
                        # Empty lines can separate documents
                        text = line.strip()
                    else:
                        # Unknown format, treat as plain text
                        text = line.strip()
                    
                    if not text:
                        continue
                    
                    self.stats['total_loaded'] += 1
                    
                    # Tokenize
                    tokens = self.tokenizer.tokenizer.encode(text)
                    if not tokens:
                        continue
                    
                    self.stats['documents_processed'] += 1
                    self.stats['total_tokens'] += len(tokens)
                    
                    # Add tokens to buffer
                    current_tokens.extend(tokens)
                    
                    # Create chunks when we have enough tokens
                    while len(current_tokens) >= self.seq_length + 1:  # +1 for labels
                        chunk = current_tokens[:self.seq_length + 1]
                        chunks.append(chunk)
                        current_tokens = current_tokens[self.seq_length:]  # Sliding window
                        self.stats['total_chunks'] += 1
                    
                    # Progress logging
                    if line_no % 10000 == 0:
                        logging.info(f"Processed {line_no:,} lines, {len(chunks):,} chunks created")
                
                except json.JSONDecodeError as e:
                    if is_jsonl:
                        logging.warning(f"JSON decode error at line {line_no}: {e}")
                    continue
                except Exception as e:
                    logging.warning(f"Error processing line {line_no}: {e}")
                    continue
        
        # Handle remaining tokens (pad if needed)
        if current_tokens:
            if len(current_tokens) < self.seq_length + 1:
                # Pad to sequence length
                current_tokens.extend([0] * (self.seq_length + 1 - len(current_tokens)))
            chunks.append(current_tokens[:self.seq_length + 1])
            self.stats['total_chunks'] += 1
        
        return chunks
    
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
    
    Memory-efficient for huge pre-training corpora.
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
            'documents_processed': 0
        }
        
        # Determine file format
        self.is_jsonl = self.data_path.suffix == '.jsonl'
        
        logging.info(f"StreamingBaseTrainingDataset {split}: {self.data_path}")
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Stream tokenized chunks."""
        current_tokens = []
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line_no, line in enumerate(f, 1):
                try:
                    # Extract text
                    if self.is_jsonl:
                        data = json.loads(line.strip())
                        text = data.get('text', '')
                    else:
                        text = line.strip()
                    
                    if not text:
                        continue
                    
                    self.stats['total_processed'] += 1
                    
                    # Tokenize
                    tokens = self.tokenizer.tokenizer.encode(text)
                    if not tokens:
                        continue
                    
                    self.stats['documents_processed'] += 1
                    current_tokens.extend(tokens)
                    
                    # Yield chunks as they become ready
                    while len(current_tokens) >= self.seq_length + 1:
                        chunk = current_tokens[:self.seq_length + 1]
                        current_tokens = current_tokens[self.seq_length:]
                        
                        # Create tensors
                        tokens_tensor = torch.tensor(chunk, dtype=torch.long)
                        input_ids = tokens_tensor[:-1]
                        labels = tokens_tensor[1:]
                        attention_mask = torch.ones_like(input_ids, dtype=torch.float)
                        loss_weights = torch.ones_like(input_ids, dtype=torch.float)
                        
                        # Mask padding
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
                    
                    # Progress logging
                    if line_no % 50000 == 0:
                        logging.info(f"Streaming: processed {line_no:,} lines, "
                                   f"{self.stats['total_chunks']:,} chunks yielded")
                
                except Exception as e:
                    logging.debug(f"Error processing line {line_no}: {e}")
                    continue
        
        # Yield remaining tokens if any
        if len(current_tokens) >= 10:  # Minimum viable chunk
            if len(current_tokens) < self.seq_length + 1:
                current_tokens.extend([0] * (self.seq_length + 1 - len(current_tokens)))
            
            tokens_tensor = torch.tensor(current_tokens[:self.seq_length + 1], dtype=torch.long)
            input_ids = tokens_tensor[:-1]
            labels = tokens_tensor[1:]
            attention_mask = torch.ones_like(input_ids, dtype=torch.float)
            loss_weights = torch.ones_like(input_ids, dtype=torch.float)
            
            padding_mask = (input_ids == 0)
            attention_mask[padding_mask] = 0.0
            loss_weights[padding_mask] = 0.0
            
            yield {
                'input_ids': input_ids,
                'labels': labels,
                'attention_mask': attention_mask,
                'loss_weights': loss_weights
            }


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


# Keep all the other dataset classes (StreamingConversationDataset, ShardedConversationDataset, MultiDatasetManager)
# [Previous implementations remain unchanged...]


class HybridDatasetManager:
    """
    Manages both base training and instruction tuning datasets.
    
    Supports:
    - Pure base training (The Pile, C4, etc.)
    - Pure instruction tuning (OASST, conversations)
    - Hybrid: base training then instruction tuning
    - Interleaved: mix of both during training
    - Multiple datasets for each type
    """
    
    def __init__(self, config):
        self.config = config
        
        # Get base training paths (multiple files supported)
        base_paths = getattr(config, 'base_training_paths', None)
        if base_paths is None:
            base_paths = getattr(config, 'base_training_path', None)
        
        # Convert to list if string
        if isinstance(base_paths, str):
            self.base_training_paths = [base_paths] if base_paths else []
        else:
            self.base_training_paths = list(base_paths) if base_paths else []
        
        # Get fine-tuning/instruction paths (multiple files supported)
        ft_paths = getattr(config, 'finetuning_paths', None)
        if ft_paths is None:
            ft_paths = getattr(config, 'train_data_path', None)
        
        # Convert to list if string
        if isinstance(ft_paths, str):
            self.finetuning_paths = [ft_paths] if ft_paths else []
        else:
            self.finetuning_paths = list(ft_paths) if ft_paths else []
        
        # Eval paths
        base_eval = getattr(config, 'base_eval_paths', None)
        if base_eval is None:
            base_eval = getattr(config, 'base_eval_path', None)
        
        if isinstance(base_eval, str):
            self.base_eval_paths = [base_eval] if base_eval else []
        else:
            self.base_eval_paths = list(base_eval) if base_eval else []
        
        ft_eval = getattr(config, 'finetuning_eval_paths', None)
        if ft_eval is None:
            ft_eval = getattr(config, 'eval_data_path', None)
        
        if isinstance(ft_eval, str):
            self.finetuning_eval_paths = [ft_eval] if ft_eval else []
        else:
            self.finetuning_eval_paths = list(ft_eval) if ft_eval else []
        
        self.training_mode = self._detect_training_mode()
        
        logging.info(f"HybridDatasetManager: {self.training_mode} mode")
        if self.base_training_paths:
            logging.info(f"  Base training: {len(self.base_training_paths)} dataset(s)")
        if self.finetuning_paths:
            logging.info(f"  Fine-tuning: {len(self.finetuning_paths)} dataset(s)")
    
    def _detect_training_mode(self) -> str:
        """Detect which training mode to use."""
        # Check if any base training files exist
        has_base = any(Path(p).exists() for p in self.base_training_paths)
        
        # Check if any fine-tuning files exist
        has_finetuning = any(Path(p).exists() for p in self.finetuning_paths)
        
        if has_base and has_finetuning:
            # Check config preference
            mode = getattr(self.config, 'training_mode', 'hybrid')
            if mode not in ['base_only', 'finetuning_only', 'instruction_only', 'hybrid', 'interleaved']:
                logging.warning(f"Unknown training_mode '{mode}', defaulting to 'hybrid'")
                return 'hybrid'
            
            # Support legacy 'instruction_only' naming
            if mode == 'instruction_only':
                return 'finetuning_only'
            
            return mode
        elif has_base:
            return 'base_only'
        elif has_finetuning:
            return 'finetuning_only'
        else:
            raise ValueError("No training data found! Set base_training_paths or finetuning_paths")
    
    def get_datasets(self, tokenizer):
        """Get appropriate datasets based on training mode."""
        if self.training_mode == 'base_only':
            return self._get_base_training_datasets(tokenizer)
        elif self.training_mode in ['finetuning_only', 'instruction_only']:
            return self._get_finetuning_datasets(tokenizer)
        elif self.training_mode == 'hybrid':
            return self._get_hybrid_datasets(tokenizer)
        elif self.training_mode == 'interleaved':
            return self._get_interleaved_datasets(tokenizer)
    
    def _combine_datasets(self, dataset_class, paths, tokenizer, split_name):
        """Combine multiple dataset files into one."""
        if len(paths) == 1:
            return dataset_class(paths[0], tokenizer, self.config, split_name)
        
        # Create temporary combined file
        cache_dir = Path(getattr(self.config, 'data_cache_dir', 'data/cache'))
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        combined_path = cache_dir / f"combined_{split_name}_{hash(tuple(paths))}.jsonl"
        
        if not combined_path.exists():
            logging.info(f"Combining {len(paths)} {split_name} files...")
            with open(combined_path, 'w', encoding='utf-8') as out_f:
                total_lines = 0
                for path in paths:
                    if not Path(path).exists():
                        logging.warning(f"File not found, skipping: {path}")
                        continue
                    
                    with open(path, 'r', encoding='utf-8') as in_f:
                        for line in in_f:
                            out_f.write(line)
                            total_lines += 1
                
                logging.info(f"Combined {total_lines:,} lines from {len(paths)} files")
        
        return dataset_class(str(combined_path), tokenizer, self.config, split_name)
    
    def _get_base_training_datasets(self, tokenizer):
        """Get base training datasets."""
        logging.info("Setting up base/pre-training datasets")
        
        # Check total file size to determine if streaming is needed
        total_size_gb = sum(
            Path(p).stat().st_size / (1024**3) 
            for p in self.base_training_paths 
            if Path(p).exists()
        )
        
        use_streaming = total_size_gb > getattr(self.config, 'streaming_threshold_gb', 10.0)
        
        if use_streaming:
            logging.info(f"Using streaming for {total_size_gb:.1f}GB of base training data")
            # For streaming, use first file (or combine if needed)
            train_dataset = StreamingBaseTrainingDataset(
                self.base_training_paths[0], tokenizer, self.config, "train"
            )
        else:
            # Combine all base training files
            train_dataset = self._combine_datasets(
                BaseTrainingDataset, self.base_training_paths, tokenizer, "base_train"
            )
        
        # Eval dataset
        if self.base_eval_paths:
            eval_dataset = self._combine_datasets(
                BaseTrainingDataset, self.base_eval_paths, tokenizer, "base_eval"
            )
        else:
            eval_dataset = train_dataset
        
        return train_dataset, eval_dataset
    
    def _get_finetuning_datasets(self, tokenizer):
        """Get fine-tuning/instruction datasets."""
        logging.info("Setting up fine-tuning datasets")
        
        # Combine all fine-tuning files
        train_dataset = self._combine_datasets(
            ConversationDataset, self.finetuning_paths, tokenizer, "ft_train"
        )
        
        # Eval dataset
        if self.finetuning_eval_paths:
            eval_dataset = self._combine_datasets(
                ConversationDataset, self.finetuning_eval_paths, tokenizer, "ft_eval"
            )
        else:
            eval_dataset = train_dataset
        
        return train_dataset, eval_dataset
    
    def _get_hybrid_datasets(self, tokenizer):
        """Get hybrid datasets - sequential training."""
        logging.info("Setting up hybrid training (base → fine-tuning)")
        
        # For hybrid mode, return base training first
        # The training orchestrator should handle switching to fine-tuning
        base_train, base_eval = self._get_base_training_datasets(tokenizer)
        
        # Store fine-tuning datasets for later phase
        self.config._finetuning_datasets_ready = True
        self.config._finetuning_paths = self.finetuning_paths
        self.config._finetuning_eval_paths = self.finetuning_eval_paths
        
        return base_train, base_eval
    
    def _get_interleaved_datasets(self, tokenizer):
        """Get interleaved datasets - mixed training."""
        logging.info("Setting up interleaved training (base + fine-tuning mixed)")
        
        # Create combined base dataset
        base_dataset = self._combine_datasets(
            BaseTrainingDataset, self.base_training_paths, tokenizer, "base_train"
        )
        
        # Create combined fine-tuning dataset
        finetuning_dataset = self._combine_datasets(
            ConversationDataset, self.finetuning_paths, tokenizer, "ft_train"
        )
        
        # Combine with InterleavedDataset
        mix_ratio = getattr(self.config, 'base_finetuning_ratio', 0.5)
        train_dataset = InterleavedDataset(base_dataset, finetuning_dataset, mix_ratio)
        
        # Eval dataset (use fine-tuning eval)
        if self.finetuning_eval_paths:
            eval_dataset = self._combine_datasets(
                ConversationDataset, self.finetuning_eval_paths, tokenizer, "ft_eval"
            )
        else:
            eval_dataset = finetuning_dataset
        
        return train_dataset, eval_dataset


class InterleavedDataset(Dataset):
    """
    Interleaves base training and fine-tuning samples.
    
    Useful for maintaining both capabilities during training.
    """
    
    def __init__(self, base_dataset, finetuning_dataset, base_ratio: float = 0.5):
        self.base_dataset = base_dataset
        self.finetuning_dataset = finetuning_dataset
        self.base_ratio = base_ratio
        
        # Create index mapping
        total_samples = len(base_dataset) + len(finetuning_dataset)
        self.indices = self._create_interleaved_indices(total_samples)
        
        logging.info(f"InterleavedDataset: {len(self)} samples "
                    f"({base_ratio:.1%} base, {1-base_ratio:.1%} fine-tuning)")
    
    def _create_interleaved_indices(self, total_samples: int) -> List[tuple]:
        """Create interleaved indices."""
        indices = []
        
        base_count = int(total_samples * self.base_ratio)
        ft_count = total_samples - base_count
        
        # Create indices
        for i in range(min(len(self.base_dataset), base_count)):
            indices.append(('base', i))
        
        for i in range(min(len(self.finetuning_dataset), ft_count)):
            indices.append(('finetuning', i))
        
        # Shuffle
        random.shuffle(indices)
        
        return indices
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get sample from appropriate dataset."""
        dataset_type, dataset_idx = self.indices[idx]
        
        if dataset_type == 'base':
            return self.base_dataset[dataset_idx]
        else:
            return self.finetuning_dataset[dataset_idx]


# Add this to the END of your dataset.py file (after all class definitions)

def create_dataloader(dataset: Union[Dataset, IterableDataset], 
                     config, 
                     shuffle: bool = True) -> DataLoader:
    """Create optimized dataloader with error handling."""
    try:
        # FIXED: Check using isinstance with actual class objects
        is_streaming = isinstance(dataset, IterableDataset)
        
        # Also check for our custom streaming classes by name (safer)
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
        # SAFER FALLBACK - just create basic dataloader
        return DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=shuffle if not isinstance(dataset, IterableDataset) else False,
            num_workers=0,  # Safest option
            drop_last=True
        )


def setup_datasets(config, tokenizer):
    """
    Main entry point for setting up datasets.
    
    Automatically detects training mode and returns appropriate datasets.
    """
    manager = HybridDatasetManager(config)
    return manager.get_datasets(tokenizer)


# Make sure these are exported
__all__ = [
    'BaseTrainingDataset',
    'StreamingBaseTrainingDataset', 
    'ConversationDataset',
    'HybridDatasetManager',
    'InterleavedDataset',
    'create_dataloader',
    'setup_datasets'
]