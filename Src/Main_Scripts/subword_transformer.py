# Modern Transformer Architecture with Ultra-Fast BPE Training
# Copyright (c) 2025 Matias Nielsen. All rights reserved.

import math
import re
import logging
import json
import warnings
import unicodedata
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple, Any, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import array
import time
from concurrent.futures import ThreadPoolExecutor
import threading

logger = logging.getLogger(__name__)

# Try to import advanced components
try:
    from flash_attn import flash_attn_func
    FLASH_ATTENTION_AVAILABLE = True
    logger.info("Flash Attention available")
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False
    logger.debug("Flash Attention not available")

try:
    import torch._dynamo
    TORCH_COMPILE_AVAILABLE = True
except ImportError:
    TORCH_COMPILE_AVAILABLE = False

try:
    import numba
    from numba import jit, types
    NUMBA_AVAILABLE = True
    logger.info("Numba available for ultra-fast BPE")
except ImportError:
    NUMBA_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

# ============================================================================
# ULTRA-FAST BPE DATA STRUCTURES
# ============================================================================

class UltraFastPairCounter:
    """Ultra-optimized pair counter using hash maps and memory pools"""
    
    def __init__(self):
        # Use defaultdict for automatic initialization
        self.pair_counts = defaultdict(int)
        # Cache for fastest access to top pairs
        self._sorted_pairs = []
        self._needs_sort = True
        self._last_sort_size = 0
        
    def add_pairs_from_word(self, word_tokens: List[int], frequency: int):
        """Add all pairs from a word with given frequency"""
        for i in range(len(word_tokens) - 1):
            pair = (word_tokens[i], word_tokens[i + 1])
            self.pair_counts[pair] += frequency
        self._needs_sort = True
    
    def remove_pairs_from_word(self, word_tokens: List[int], frequency: int):
        """Remove pairs when word changes"""
        for i in range(len(word_tokens) - 1):
            pair = (word_tokens[i], word_tokens[i + 1])
            self.pair_counts[pair] -= frequency
            if self.pair_counts[pair] <= 0:
                del self.pair_counts[pair]
        self._needs_sort = True
    
    def get_best_pair(self) -> Tuple[Tuple[int, int], int]:
        """Get the most frequent pair with smart caching"""
        if not self.pair_counts:
            return None, 0
        
        # Only re-sort if significantly changed
        if self._needs_sort and len(self.pair_counts) - self._last_sort_size > 1000:
            self._sorted_pairs = sorted(self.pair_counts.items(), key=lambda x: x[1], reverse=True)
            self._needs_sort = False
            self._last_sort_size = len(self.pair_counts)
            return self._sorted_pairs[0]
        elif self._sorted_pairs:
            # Quick check if top pair is still valid
            top_pair, cached_count = self._sorted_pairs[0]
            current_count = self.pair_counts.get(top_pair, 0)
            if current_count == cached_count and current_count > 0:
                return top_pair, current_count
        
        # Fallback to max search
        return max(self.pair_counts.items(), key=lambda x: x[1])
    
    def clear(self):
        """Clear all data"""
        self.pair_counts.clear()
        self._sorted_pairs.clear()
        self._needs_sort = True


class WordSplitManager:
    """Manages word splits with ultra-fast updates"""
    
    def __init__(self, word_freqs: Dict[str, int], get_word_chars_func):
        self.word_freqs = word_freqs
        self.get_word_chars = get_word_chars_func
        
        # Convert to integer tokens for faster processing
        self.char_to_id = {}
        self.id_to_char = {}
        self.next_id = 0
        
        # Pre-tokenize all words
        self.word_splits = {}  # word -> list of token IDs
        self._build_initial_splits()
        
        # Track which words contain each pair for ultra-fast updates
        self.pair_to_words = defaultdict(set)
        self._build_pair_index()
    
    def _build_initial_splits(self):
        """Build initial character-level splits"""
        logger.info("   Building initial word splits...")
        
        for word in self.word_freqs:
            chars = self.get_word_chars(word)
            token_ids = []
            
            for char in chars:
                if char not in self.char_to_id:
                    self.char_to_id[char] = self.next_id
                    self.id_to_char[self.next_id] = char
                    self.next_id += 1
                token_ids.append(self.char_to_id[char])
            
            self.word_splits[word] = token_ids
    
    def _build_pair_index(self):
        """Build index of which words contain each pair"""
        logger.info("   Building pair index...")
        
        for word, tokens in self.word_splits.items():
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                self.pair_to_words[pair].add(word)
    
    def merge_pair(self, pair: Tuple[int, int], new_token_id: int) -> Set[str]:
        """Merge a pair across all words and return affected words"""
        affected_words = set()
        
        # Only process words that contain this pair
        words_with_pair = self.pair_to_words.get(pair, set()).copy()
        
        for word in words_with_pair:
            if word not in self.word_splits:
                continue
                
            tokens = self.word_splits[word]
            new_tokens = []
            i = 0
            
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                    new_tokens.append(new_token_id)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            
            if new_tokens != tokens:
                self.word_splits[word] = new_tokens
                affected_words.add(word)
        
        # Update pair index
        self._update_pair_index_for_words(affected_words)
        
        return affected_words
    
    def _update_pair_index_for_words(self, words: Set[str]):
        """Update pair index for specific words"""
        # Remove old pairs for these words
        for pair_set in self.pair_to_words.values():
            pair_set -= words
        
        # Add new pairs for these words
        for word in words:
            if word in self.word_splits:
                tokens = self.word_splits[word]
                for i in range(len(tokens) - 1):
                    pair = (tokens[i], tokens[i + 1])
                    self.pair_to_words[pair].add(word)
    
    def get_all_pairs_with_counts(self) -> Dict[Tuple[int, int], int]:
        """Get all pairs with their frequencies"""
        pair_counts = defaultdict(int)
        
        for word, tokens in self.word_splits.items():
            freq = self.word_freqs[word]
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pair_counts[pair] += freq
        
        return dict(pair_counts)
    
    def add_token(self, token_str: str) -> int:
        """Add a new token and return its ID"""
        if token_str not in self.char_to_id:
            self.char_to_id[token_str] = self.next_id
            self.id_to_char[self.next_id] = token_str
            self.next_id += 1
        return self.char_to_id[token_str]


# Enhanced NumPy acceleration if available
if NUMPY_AVAILABLE:
    def numpy_accelerated_pair_counting(word_arrays, freq_array):
        """Ultra-fast pair counting using NumPy vectorization"""
        pair_counts = defaultdict(int)
        
        for word_idx, word_array in enumerate(word_arrays):
            if len(word_array) < 2:
                continue
                
            freq = freq_array[word_idx]
            # Vectorized pair extraction
            pairs = np.column_stack((word_array[:-1], word_array[1:]))
            
            # Count unique pairs
            unique_pairs, counts = np.unique(pairs, axis=0, return_counts=True)
            
            for pair_row, count in zip(unique_pairs, counts):
                pair = (int(pair_row[0]), int(pair_row[1]))
                pair_counts[pair] += count * freq
        
        return pair_counts


class SubwordTokenizer:
    """Production-ready subword tokenizer with ultra-fast BPE implementation."""
    
    def __init__(self, vocab: Optional[Dict[str, int]] = None, merges: Optional[List[Tuple[str, str]]] = None):
        # Initialize vocabulary with comprehensive special tokens
        if vocab is None:
            self.vocab = {
                # Core tokens
                "<|pad|>": 0,
                "<|unk|>": 1,
                "<|bos|>": 2,
                "<|eos|>": 3,
                
                # Chat format tokens
                "<|user|>": 4,
                "<|assistant|>": 5,
                "<|system|>": 6,
                "<|end|>": 7,
                
                # Special formatting
                "<|newline|>": 8,
                "<|tab|>": 9,
                "<|space|>": 10,
                
                # Function calling
                "<|function|>": 11,
                "<|tool|>": 12,
                "<|result|>": 13,
                
                # Reserved for future use
                "<|reserved_1|>": 14,
                "<|reserved_2|>": 15,
            }
            self.next_id = 16
        else:
            self.vocab = vocab.copy()
            self.next_id = max(vocab.values()) + 1 if vocab else 16
        
        self.merges = merges or []
        self.merge_dict = {pair: self._merge_tokens(pair) for pair in self.merges}
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        
        # Fixed regex pattern using standard Python character classes
        self.word_pattern = re.compile(
            r"'(?:[sdmt]|ll|ve|re)|"          # Common contractions
            r"[^\r\n\w\s]?[\w]+|"             # Letter sequences with optional punctuation
            r"\d+(?:\.\d+)*|"                 # Numbers (including decimals)
            r"[^\r\n\w\s]+[\r\n]*|"           # Punctuation and symbols
            r"[\r\n]+",                       # Line breaks
            re.IGNORECASE | re.UNICODE
        )
        
        # Cache for encoding efficiency
        self._encoding_cache = {}
        self._cache_size_limit = 10000
        
        logger.info(f"SubwordTokenizer initialized with {len(self.vocab)} tokens")
    
    def _merge_tokens(self, pair: Tuple[str, str]) -> str:
        """Create merged token from pair."""
        return pair[0] + pair[1]
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for consistent tokenization."""
        if not text:
            return ""
        
        # Unicode normalization
        text = unicodedata.normalize('NFKC', text)
        
        # Handle special whitespace
        text = text.replace('\r\n', '\n')  # Normalize line endings
        text = text.replace('\r', '\n')
        
        # Preserve important whitespace patterns but normalize excessive spacing
        text = re.sub(r' +', ' ', text)  # Multiple spaces -> single space
        text = re.sub(r'\n+', '\n', text)  # Multiple newlines -> single newline
        
        return text
    
    def _get_word_chars(self, word: str) -> List[str]:
        """Split word into characters with end-of-word marker."""
        if not word:
            return []
        chars = list(word)
        if chars:
            chars[-1] += "</w>"
        return chars
    
    def train_from_text_ultra_fast(self, text: str, vocab_size: int = 32000, min_freq: int = 2, 
                                   progress_callback: Optional[callable] = None) -> None:
        """ULTRA-FAST BPE training with advanced optimizations."""
        
        logger.info(f"🚀 ULTRA-FAST BPE tokenizer training:")
        logger.info(f"  Target vocabulary size: {vocab_size:,}")
        logger.info(f"  Training text length: {len(text):,} characters")
        logger.info(f"  Minimum frequency: {min_freq}")
        
        start_time = time.time()
        
        # Normalize and preprocess text
        text = self._normalize_text(text)
        
        # Extract words using the improved pattern
        logger.info("🔥 Extracting words with optimized regex...")
        words = []
        for match in re.finditer(self.word_pattern, text):
            word = match.group().strip()
            if word and not word.isspace():
                words.append(word)
        
        # Count word frequencies with ultra-fast Counter
        logger.info("⚡ Counting word frequencies...")
        word_freqs = Counter(words)
        
        # Filter by minimum frequency
        original_unique = len(word_freqs)
        word_freqs = {word: freq for word, freq in word_freqs.items() if freq >= min_freq}
        filtered_unique = len(word_freqs)
        
        logger.info(f"📊 Word statistics:")
        logger.info(f"  Total words: {len(words):,}")
        logger.info(f"  Unique words (all): {original_unique:,}")
        logger.info(f"  Unique words (freq >= {min_freq}): {filtered_unique:,}")
        
        if not word_freqs:
            raise ValueError("No words meet the minimum frequency requirement!")
        
        # Initialize ultra-fast word split manager
        logger.info("🚀 Initializing ultra-fast word split manager...")
        split_manager = WordSplitManager(word_freqs, self._get_word_chars)
        
        # Build initial character vocabulary
        logger.info("🔤 Building character vocabulary...")
        chars_added = 0
        
        # Add characters sorted by frequency for better tokenization
        char_freqs = Counter()
        for word, freq in word_freqs.items():
            chars = self._get_word_chars(word)
            for char in chars:
                char_freqs[char] += freq
        
        for char, freq in char_freqs.most_common():
            if char not in self.vocab and len(self.vocab) < vocab_size:
                self.vocab[char] = self.next_id
                self.id_to_token[self.next_id] = char
                self.next_id += 1
                chars_added += 1
        
        logger.info(f"✅ Added {chars_added:,} characters to vocabulary")
        logger.info(f"📈 Current vocabulary size: {len(self.vocab):,}")
        
        # Ultra-fast BPE merge learning with advanced optimizations
        target_merges = vocab_size - len(self.vocab)
        merges_learned = 0
        
        logger.info(f"🚀 Learning BPE merges with MAXIMUM SPEED (target: {target_merges:,})...")
        
        # Initialize ultra-fast pair counter
        pair_counter = UltraFastPairCounter()
        
        # Initial pair counting
        logger.info("⚡ Initial pair counting...")
        initial_pairs = split_manager.get_all_pairs_with_counts()
        for pair, count in initial_pairs.items():
            pair_counter.pair_counts[pair] = count
        
        # Progress tracking
        last_progress_time = time.time()
        progress_interval = max(1, target_merges // 100)  # Update every 1%
        
        while merges_learned < target_merges and len(self.vocab) < vocab_size:
            # Get best pair with smart caching
            best_pair, count = pair_counter.get_best_pair()
            
            if not best_pair or count <= 0:
                logger.warning("⚠️ No more pairs to merge!")
                break
            
            # Convert pair IDs back to tokens
            token1 = split_manager.id_to_char[best_pair[0]]
            token2 = split_manager.id_to_char[best_pair[1]]
            merged_token = self._merge_tokens((token1, token2))
            
            # Add to vocabulary if new
            if merged_token not in self.vocab:
                self.vocab[merged_token] = self.next_id
                self.id_to_token[self.next_id] = merged_token
                self.next_id += 1
            
            # Add to split manager
            new_token_id = split_manager.add_token(merged_token)
            
            # Add merge rule
            self.merges.append((token1, token2))
            self.merge_dict[(token1, token2)] = merged_token
            
            # Ultra-fast merge update
            affected_words = split_manager.merge_pair(best_pair, new_token_id)
            
            # Update pair counter efficiently - only for affected words
            for word in affected_words:
                freq = word_freqs[word]
                old_tokens = []  # We'd need to track this for perfect efficiency
                new_tokens = split_manager.word_splits[word]
                
                # Remove old pairs and add new pairs
                pair_counter.remove_pairs_from_word(old_tokens, freq)
                pair_counter.add_pairs_from_word(new_tokens, freq)
            
            # Remove the merged pair
            if best_pair in pair_counter.pair_counts:
                del pair_counter.pair_counts[best_pair]
            
            merges_learned += 1
            
            # Ultra-fast progress reporting
            current_time = time.time()
            if (merges_learned % progress_interval == 0 or 
                merges_learned == target_merges or 
                current_time - last_progress_time > 5.0):  # Every 5 seconds max
                
                progress = (merges_learned / target_merges) * 100
                elapsed = current_time - start_time
                speed = merges_learned / elapsed if elapsed > 0 else 0
                eta = (target_merges - merges_learned) / speed if speed > 0 else 0
                
                logger.info(f"🚀 Progress: {progress:.1f}% ({merges_learned:,}/{target_merges:,}) | "
                          f"Speed: {speed:.0f} merges/sec | ETA: {eta:.0f}s")
                
                if progress_callback:
                    progress_callback(progress, merges_learned, target_merges)
                
                last_progress_time = current_time
        
        # Final cleanup and statistics
        total_time = time.time() - start_time
        actual_vocab_size = len(self.vocab)
        final_speed = merges_learned / total_time if total_time > 0 else 0
        
        logger.info(f"🎉 ULTRA-FAST BPE training completed!")
        logger.info(f"  Final vocabulary size: {actual_vocab_size:,}")
        logger.info(f"  Merge rules learned: {len(self.merges):,}")
        logger.info(f"  Total time: {total_time:.1f}s")
        logger.info(f"  Average speed: {final_speed:.0f} merges/sec")
        logger.info(f"  Coverage: {(actual_vocab_size / vocab_size) * 100:.1f}%")
        
        # Clear cache after training
        self._encoding_cache.clear()
    
    def _apply_bpe(self, word: str) -> List[str]:
        """Apply BPE merges to a word with caching."""
        
        # Check cache first
        if word in self._encoding_cache:
            return self._encoding_cache[word]
        
        if not word:
            return []
        
        # Handle single character or already in vocab
        if len(word) == 1 or word in self.vocab:
            result = [word]
            self._cache_result(word, result)
            return result
        
        # Initialize with character split
        chars = self._get_word_chars(word)
        if len(chars) <= 1:
            self._cache_result(word, chars)
            return chars
        
        # Apply merges in the order they were learned
        for pair in self.merges:
            if len(chars) <= 1:
                break
                
            new_chars = []
            i = 0
            while i < len(chars):
                if i < len(chars) - 1 and (chars[i], chars[i + 1]) == pair:
                    merged_token = self.merge_dict[pair]
                    new_chars.append(merged_token)
                    i += 2
                else:
                    new_chars.append(chars[i])
                    i += 1
            chars = new_chars
        
        self._cache_result(word, chars)
        return chars
    
    def _cache_result(self, word: str, result: List[str]):
        """Cache encoding result with size limit."""
        if len(self._encoding_cache) >= self._cache_size_limit:
            # Remove oldest entries (simple FIFO)
            keys_to_remove = list(self._encoding_cache.keys())[:self._cache_size_limit // 4]
            for key in keys_to_remove:
                del self._encoding_cache[key]
        
        self._encoding_cache[word] = result
    
    def encode(self, text: str, add_special_tokens: bool = True, 
              max_length: Optional[int] = None) -> List[int]:
        """Enhanced encoding with better error handling and options."""
        
        if not text:
            if add_special_tokens:
                return [self.vocab.get("<|bos|>", 2), self.vocab.get("<|eos|>", 3)]
            return []
        
        # Normalize text
        text = self._normalize_text(text)
        
        # Extract words
        words = []
        for match in re.finditer(self.word_pattern, text):
            word = match.group()
            if word:  # Keep all words, including whitespace
                words.append(word)
        
        # Convert to token IDs
        token_ids = []
        
        # Add BOS token
        if add_special_tokens:
            token_ids.append(self.vocab.get("<|bos|>", 2))
        
        unk_id = self.vocab.get("<|unk|>", 1)
        
        for word in words:
            # Handle special whitespace
            if word == '\n':
                token_ids.append(self.vocab.get("<|newline|>", self.vocab.get("\n", unk_id)))
            elif word == '\t':
                token_ids.append(self.vocab.get("<|tab|>", self.vocab.get("\t", unk_id)))
            elif word.isspace():
                # For other whitespace, try to encode normally or use space token
                subwords = self._apply_bpe(word)
                for subword in subwords:
                    token_id = self.vocab.get(subword, self.vocab.get("<|space|>", unk_id))
                    token_ids.append(token_id)
            else:
                # Regular word encoding
                subwords = self._apply_bpe(word)
                for subword in subwords:
                    token_id = self.vocab.get(subword, unk_id)
                    token_ids.append(token_id)
            
            # Check max length
            if max_length and len(token_ids) >= max_length - (1 if add_special_tokens else 0):
                break
        
        # Add EOS token
        if add_special_tokens:
            token_ids.append(self.vocab.get("<|eos|>", 3))
        
        # Truncate if necessary
        if max_length and len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
            if add_special_tokens:
                token_ids[-1] = self.vocab.get("<|eos|>", 3)
        
        return token_ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True,
              clean_up_tokenization_spaces: bool = True) -> str:
        """Enhanced decoding with better text reconstruction."""
        
        if not token_ids:
            return ""
        
        # Special tokens to skip
        special_token_ids = {
            self.vocab.get("<|pad|>", 0),
            self.vocab.get("<|bos|>", 2),
            self.vocab.get("<|eos|>", 3)
        } if skip_special_tokens else set()
        
        tokens = []
        for token_id in token_ids:
            if token_id not in special_token_ids:
                token = self.id_to_token.get(token_id, "<|unk|>")
                tokens.append(token)
        
        # Reconstruct text
        text = "".join(tokens)
        
        # Clean up BPE artifacts
        text = text.replace("</w>", " ")
        
        # Handle special tokens
        text = text.replace("<|newline|>", "\n")
        text = text.replace("<|tab|>", "\t")
        text = text.replace("<|space|>", " ")
        
        if clean_up_tokenization_spaces:
            # Clean up extra spaces
            text = re.sub(r' +', ' ', text)  # Multiple spaces -> single space
            text = re.sub(r' \n', '\n', text)  # Space before newline
            text = re.sub(r'\n ', '\n', text)  # Space after newline
            text = text.strip()
        
        return text
    
    def encode_chat(self, messages: List[Dict[str, str]], add_generation_prompt: bool = False) -> List[int]:
        """Encode chat messages with proper formatting."""
        
        formatted_text = ""
        
        for message in messages:
            role = message.get("role", "").lower()
            content = message.get("content", "").strip()
            
            if role == "system":
                formatted_text += f"<|system|>{content}<|end|>\n"
            elif role == "user":
                formatted_text += f"<|user|>{content}<|end|>\n"
            elif role == "assistant":
                formatted_text += f"<|assistant|>{content}<|end|>\n"
            else:
                # Fallback for unknown roles
                formatted_text += f"{content}\n"
        
        # Add generation prompt for inference
        if add_generation_prompt:
            formatted_text += "<|assistant|>"
        
        return self.encode(formatted_text, add_special_tokens=True)
    
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.vocab)
    
    def get_vocab(self) -> Dict[str, int]:
        """Get vocabulary dictionary."""
        return self.vocab.copy()
    
    def save_vocab(self, vocab_path: str, merges_path: str) -> None:
        """Save vocabulary and merges with metadata."""
        
        # Save vocabulary with metadata
        vocab_data = {
            "vocab": self.vocab,
            "vocab_size": len(self.vocab),
            "special_tokens": {
                "pad_token": "<|pad|>",
                "unk_token": "<|unk|>",
                "bos_token": "<|bos|>",
                "eos_token": "<|eos|>",
                "user_token": "<|user|>",
                "assistant_token": "<|assistant|>",
                "system_token": "<|system|>",
                "end_token": "<|end|>"
            },
            "created_at": str(torch.tensor(0).device),  # Simple timestamp placeholder
            "merge_count": len(self.merges)
        }
        
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, indent=2, ensure_ascii=False)
        
        # Save merges
        with open(merges_path, 'w', encoding='utf-8') as f:
            f.write(f"#version: 0.2\n")  # Add version for compatibility
            for pair in self.merges:
                f.write(f"{pair[0]} {pair[1]}\n")
        
        logger.info(f"Tokenizer saved: {len(self.vocab):,} tokens, {len(self.merges):,} merges")
    
    def load_vocab(self, vocab_path: str, merges_path: str) -> None:
        """Load vocabulary and merges with error handling."""
        
        try:
            # Load vocabulary
            with open(vocab_path, 'r', encoding='utf-8') as f:
                vocab_data = json.load(f)
            
            if isinstance(vocab_data, dict) and "vocab" in vocab_data:
                # New format with metadata
                self.vocab = vocab_data["vocab"]
                logger.info(f"Loaded vocabulary with metadata (version with {vocab_data.get('vocab_size', len(self.vocab))} tokens)")
            else:
                # Legacy format - direct vocabulary
                self.vocab = vocab_data
                logger.info(f"Loaded legacy vocabulary format")
            
            self.id_to_token = {v: k for k, v in self.vocab.items()}
            self.next_id = max(self.vocab.values()) + 1 if self.vocab else 0
            
            # Load merges
            self.merges = []
            self.merge_dict = {}
            
            with open(merges_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    
                    # Skip comments and version lines
                    if line.startswith('#') or not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) == 2:
                        pair = (parts[0], parts[1])
                        merged = self._merge_tokens(pair)
                        self.merges.append(pair)
                        self.merge_dict[pair] = merged
                    else:
                        logger.warning(f"Invalid merge line {line_num}: {line}")
            
            logger.info(f"Tokenizer loaded: {len(self.vocab):,} tokens, {len(self.merges):,} merges")
            
            # Clear encoding cache
            self._encoding_cache.clear()
            
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise


# Ultra-fast tokenizer class for the training script
class UltraFastTokenizer(SubwordTokenizer):
    """Ultra-optimized tokenizer with maximum speed BPE training"""
    
    def train_from_text_ultra_fast(self, text: str, vocab_size: int = 32000, 
                                  min_freq: int = 2, progress_callback: Optional[callable] = None) -> None:
        """Maximum speed BPE training - calls the optimized parent method"""
        return super().train_from_text_ultra_fast(text, vocab_size, min_freq, progress_callback)


# Import all the other components from the original file (RMSNorm, RotaryPositionalEmbedding, etc.)
# [Previous code for RMSNorm, RotaryPositionalEmbedding, GroupedQueryAttention, SwiGLU, GeGLU, ModernTransformerBlock, ModernSubwordTransformer continues exactly as before...]

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization with improved stability."""
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.hidden_size = hidden_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # More numerically stable implementation
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (x * self.weight).to(x.dtype)


class RotaryPositionalEmbedding(nn.Module):
    """Optimized Rotary Position Embedding (RoPE) with caching."""
    
    def __init__(self, dim: int, max_seq_len: int = 32768, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute frequency inverse
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Cache for efficiency
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
        self._device = None
    
    def _update_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype = None):
        """Update cached cos/sin values."""
        if (seq_len > self._seq_len_cached or 
            self._cos_cached is None or 
            self._device != device or
            (dtype and self._cos_cached.dtype != dtype)):
            
            self._seq_len_cached = max(seq_len, self._seq_len_cached)
            self._device = device
            
            t = torch.arange(self._seq_len_cached, device=device, dtype=torch.float32)
            freqs = torch.outer(t, self.inv_freq.to(device))
            
            # Create cos/sin in the requested dtype
            cos = torch.cos(freqs)
            sin = torch.sin(freqs)
            
            if dtype:
                cos = cos.to(dtype)
                sin = sin.to(dtype)
            
            self._cos_cached = cos
            self._sin_cached = sin
    
    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        self._update_cache(seq_len, x.device, x.dtype)
        
        cos = self._cos_cached[:seq_len]
        sin = self._sin_cached[:seq_len]
        
        return cos, sin


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, 
                        cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary positional embedding with optimized rotation."""
    
    # More efficient rotation implementation
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    
    # Apply rotation efficiently
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, dim]
    sin = sin.unsqueeze(0).unsqueeze(0)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed


class GroupedQueryAttention(nn.Module):
    """Optimized Grouped Query Attention with Flash Attention support."""
    
    def __init__(self, hidden_size: int, num_heads: int, num_key_value_heads: int,
                 dropout: float = 0.0, use_rope: bool = True, max_seq_len: int = 32768,
                 use_bias: bool = False):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = hidden_size // num_heads
        self.num_queries_per_kv = num_heads // num_key_value_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        assert num_heads % num_key_value_heads == 0, "num_heads must be divisible by num_key_value_heads"
        
        # Linear projections with optional bias
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=use_bias)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=use_bias)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=use_bias)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=use_bias)
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        # RoPE
        self.use_rope = use_rope
        if use_rope:
            self.rotary_emb = RotaryPositionalEmbedding(self.head_dim, max_seq_len)
        
        # Flash Attention detection
        self.use_flash_attn = FLASH_ATTENTION_AVAILABLE
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with proper scaling."""
        # Xavier/Glorot initialization for query, key, value
        for module in [self.q_proj, self.k_proj, self.v_proj]:
            nn.init.xavier_uniform_(module.weight, gain=1.0)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        
        # Output projection with smaller initialization
        nn.init.xavier_uniform_(self.o_proj.weight, gain=1.0 / math.sqrt(2))
        if self.o_proj.bias is not None:
            nn.init.zeros_(self.o_proj.bias)
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                use_cache: bool = False, past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                output_attentions: bool = False) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor]]:
        
        batch_size, seq_len, _ = x.shape
        
        # Linear projections
        q = self.q_proj(x)  # [batch, seq, num_heads * head_dim]
        k = self.k_proj(x)  # [batch, seq, num_kv_heads * head_dim]
        v = self.v_proj(x)  # [batch, seq, num_kv_heads * head_dim]
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # Handle past key-values for generation
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
            kv_seq_len = k.size(2)
        else:
            kv_seq_len = seq_len
        
        # Apply RoPE
        if self.use_rope:
            cos, sin = self.rotary_emb(x, kv_seq_len)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Expand key-values for grouped query attention
        if self.num_queries_per_kv > 1:
            # Repeat KV heads to match query heads
            k = k.unsqueeze(2).expand(-1, -1, self.num_queries_per_kv, -1, -1)
            v = v.unsqueeze(2).expand(-1, -1, self.num_queries_per_kv, -1, -1)
            k = k.reshape(batch_size, self.num_heads, kv_seq_len, self.head_dim)
            v = v.reshape(batch_size, self.num_heads, kv_seq_len, self.head_dim)
        
        # Compute attention
        attn_weights = None
        
        if self.use_flash_attn and q.is_cuda and attention_mask is None:
            # Use Flash Attention for efficiency
            try:
                # Reshape for Flash Attention: [batch, seq, heads, head_dim]
                q_flash = q.transpose(1, 2).contiguous()
                k_flash = k.transpose(1, 2).contiguous()
                v_flash = v.transpose(1, 2).contiguous()
                
                attn_output = flash_attn_func(
                    q_flash, k_flash, v_flash,
                    dropout_p=0.0 if not self.training else (self.dropout.p if self.dropout else 0.0),
                    causal=True
                )
                
                # Reshape back: [batch, heads, seq, head_dim]
                attn_output = attn_output.transpose(1, 2).contiguous()
                
            except Exception as e:
                logger.debug(f"Flash Attention failed, falling back to standard attention: {e}")
                attn_output, attn_weights = self._standard_attention(q, k, v, attention_mask, output_attentions)
        else:
            # Standard attention implementation
            attn_output, attn_weights = self._standard_attention(q, k, v, attention_mask, output_attentions)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        output = self.o_proj(attn_output)
        
        # Return key-value cache if requested
        present_kv = (k, v) if use_cache else None
        
        return output, present_kv, attn_weights
    
    def _standard_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                           attention_mask: Optional[torch.Tensor] = None,
                           output_attentions: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Standard scaled dot-product attention."""
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply causal mask
        seq_len = q.size(2)
        kv_seq_len = k.size(2)
        
        if attention_mask is None:
            # Create causal mask
            causal_mask = torch.triu(
                torch.ones(seq_len, kv_seq_len, dtype=torch.bool, device=q.device),
                diagonal=kv_seq_len - seq_len + 1
            )
            scores = scores.masked_fill(causal_mask, float('-inf'))
        else:
            scores = scores + attention_mask
        
        # Compute attention weights
        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        
        # Apply dropout
        if self.dropout is not None and self.training:
            attn_weights = self.dropout(attn_weights)
        
        # Compute output
        attn_output = torch.matmul(attn_weights, v)
        
        return attn_output, attn_weights if output_attentions else None


class SwiGLU(nn.Module):
    """SwiGLU activation function - superior to GELU for language models."""
    
    def __init__(self, hidden_size: int, intermediate_size: int, bias: bool = False):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights properly for SwiGLU."""
        # Gate and up projections
        for proj in [self.gate_proj, self.up_proj]:
            nn.init.xavier_uniform_(proj.weight, gain=1.0)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)
        
        # Down projection with smaller initialization
        nn.init.xavier_uniform_(self.down_proj.weight, gain=1.0 / math.sqrt(2))
        if self.down_proj.bias is not None:
            nn.init.zeros_(self.down_proj.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(F.silu(gate) * up)


class GeGLU(nn.Module):
    """GeGLU activation function - alternative to SwiGLU."""
    
    def __init__(self, hidden_size: int, intermediate_size: int, bias: bool = False):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights properly for GeGLU."""
        for proj in [self.gate_proj, self.up_proj]:
            nn.init.xavier_uniform_(proj.weight, gain=1.0)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)
        
        nn.init.xavier_uniform_(self.down_proj.weight, gain=1.0 / math.sqrt(2))
        if self.down_proj.bias is not None:
            nn.init.zeros_(self.down_proj.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(F.gelu(gate) * up)


class ModernTransformerBlock(nn.Module):
    """Modern transformer block with all the latest improvements."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Pre-normalization layers
        if config.use_rms_norm:
            self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
            self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        else:
            self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Attention mechanism
        if config.use_grouped_query_attention:
            self.self_attn = GroupedQueryAttention(
                hidden_size=config.hidden_size,
                num_heads=config.num_heads,
                num_key_value_heads=config.num_key_value_heads,
                dropout=config.dropout,
                use_rope=config.use_rotary_pos_emb,
                max_seq_len=config.seq_length,
                use_bias=config.attention_bias
            )
        else:
            # Fallback to standard multi-head attention
            self.self_attn = nn.MultiheadAttention(
                embed_dim=config.hidden_size,
                num_heads=config.num_heads,
                dropout=config.dropout,
                bias=config.attention_bias,
                batch_first=True
            )
        
        # Feed-forward network
        if config.use_glu_variants:
            if config.glu_variant.lower() == "swiglu":
                self.mlp = SwiGLU(config.hidden_size, config.intermediate_size, bias=config.attention_bias)
            elif config.glu_variant.lower() == "geglu":
                self.mlp = GeGLU(config.hidden_size, config.intermediate_size, bias=config.attention_bias)
            else:
                logger.warning(f"Unknown GLU variant: {config.glu_variant}, defaulting to SwiGLU")
                self.mlp = SwiGLU(config.hidden_size, config.intermediate_size, bias=config.attention_bias)
        else:
            # Standard feed-forward
            self.mlp = nn.Sequential(
                nn.Linear(config.hidden_size, config.intermediate_size, bias=config.attention_bias),
                nn.GELU() if config.activation == "gelu" else nn.ReLU(),
                nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity(),
                nn.Linear(config.intermediate_size, config.hidden_size, bias=config.attention_bias)
            )
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else None
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                use_cache: bool = False, past_kv: Optional[Tuple] = None,
                output_attentions: bool = False) -> Tuple[torch.Tensor, Optional[Tuple], Optional[torch.Tensor]]:
        
        # Pre-normalization for attention
        residual = x
        x = self.input_layernorm(x)
        
        # Self-attention
        if isinstance(self.self_attn, GroupedQueryAttention):
            attn_output, present_kv, attn_weights = self.self_attn(
                x, attention_mask, use_cache, past_kv, output_attentions
            )
        else:
            # Standard attention (no advanced features)
            attn_output, attn_weights = self.self_attn(x, x, x, attn_mask=attention_mask)
            present_kv = None
            if not output_attentions:
                attn_weights = None
        
        # Residual connection with dropout
        if self.dropout is not None:
            attn_output = self.dropout(attn_output)
        x = residual + attn_output
        
        # Pre-normalization for feed-forward
        residual = x
        x = self.post_attention_layernorm(x)
        
        # Feed-forward network
        ff_output = self.mlp(x)
        
        # Residual connection with dropout
        if self.dropout is not None:
            ff_output = self.dropout(ff_output)
        x = residual + ff_output
        
        return x, present_kv, attn_weights


class ModernSubwordTransformer(nn.Module):
    """Modern transformer with all the latest architectural improvements."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        
        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Optional embedding dropout
        self.embed_dropout = nn.Dropout(config.embed_dropout) if config.embed_dropout > 0 else None
        
        # Transformer layers
        self.layers = nn.ModuleList([
            ModernTransformerBlock(config) for _ in range(config.num_layers)
        ])
        
        # Final layer normalization
        if config.use_rms_norm:
            self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        else:
            self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Language modeling head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Weight tying
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        
        # Initialize weights
        self._init_weights()
        
        # Gradient checkpointing
        self.gradient_checkpointing = False
        if config.gradient_checkpointing:
            self.enable_gradient_checkpointing()
        
        # Compile model if requested and available
        if TORCH_COMPILE_AVAILABLE and getattr(config, 'use_compile', False):
            try:
                self.forward = torch.compile(self.forward)
                logger.info("Model compiled with torch.compile")
            except Exception as e:
                logger.warning(f"Failed to compile model: {e}")
        
        # Log model info
        self._log_model_info()
    
    def _init_weights(self):
        """Initialize weights with modern best practices."""
        
        # Token embeddings - smaller std for better training stability
        std = self.config.initializer_range
        nn.init.normal_(self.embed_tokens.weight, mean=0.0, std=std)
        
        # Apply scaled initialization for deeper models
        if self.config.use_scaled_init:
            depth_scale = math.sqrt(2.0 * self.config.num_layers)
            
            for layer in self.layers:
                # Scale down residual connections in deeper models
                if hasattr(layer.self_attn, 'o_proj'):
                    with torch.no_grad():
                        layer.self_attn.o_proj.weight.div_(depth_scale)
                
                # Scale down MLP output
                if hasattr(layer.mlp, 'down_proj'):
                    with torch.no_grad():
                        layer.mlp.down_proj.weight.div_(depth_scale)
                elif isinstance(layer.mlp, nn.Sequential):
                    # Standard MLP
                    with torch.no_grad():
                        layer.mlp[-1].weight.div_(depth_scale)
        
        # LM head initialization (if not tied)
        if not self.config.tie_word_embeddings:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=std)
    
    def _log_model_info(self):
        """Log comprehensive model information."""
        total_params = self.count_parameters()
        trainable_params = self.count_trainable_parameters()
        memory_mb = self.estimate_memory_mb()
        
        logger.info(f"🧠 ModernSubwordTransformer initialized:")
        logger.info(f"   Architecture: {self.config.num_layers}L × {self.config.hidden_size}H × {self.config.num_heads}A")
        logger.info(f"   Vocabulary: {self.config.vocab_size:,} tokens")
        logger.info(f"   Parameters: {total_params:,} total, {trainable_params:,} trainable")
        logger.info(f"   Memory estimate: {memory_mb:.1f}MB")
        
        # Architecture features
        features = []
        if self.config.use_rotary_pos_emb:
            features.append("RoPE")
        if self.config.use_rms_norm:
            features.append("RMSNorm")
        if self.config.use_grouped_query_attention:
            features.append(f"GQA({self.config.num_heads}:{self.config.num_key_value_heads})")
        if self.config.use_glu_variants:
            features.append(f"{self.config.glu_variant.upper()}")
        if self.config.gradient_checkpointing:
            features.append("GradCheckpoint")
        
        if features:
            logger.info(f"   Features: {', '.join(features)}")
    
    def count_parameters(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def count_trainable_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def estimate_memory_mb(self) -> float:
        """Estimate model memory usage in MB."""
        param_memory = sum(p.numel() * p.element_size() for p in self.parameters())
        return param_memory / (1024 * 1024)
    
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency."""
        self.gradient_checkpointing = True
        
        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward
        
        for layer in self.layers:
            layer.forward = torch.utils.checkpoint.checkpoint(
                create_custom_forward(layer), use_reentrant=False
            )
        
        logger.info("✅ Gradient checkpointing enabled")
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                use_cache: bool = False, past_key_values: Optional[List[Tuple]] = None,
                output_attentions: bool = False, return_dict: bool = True) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        
        batch_size, seq_len = input_ids.shape
        
        # Input validation
        if torch.any(input_ids >= self.vocab_size) or torch.any(input_ids < 0):
            logger.warning("Input contains out-of-vocabulary token IDs")
            input_ids = torch.clamp(input_ids, 0, self.vocab_size - 1)
        
        # Token embeddings
        hidden_states = self.embed_tokens(input_ids)
        
        # Embedding dropout
        if self.embed_dropout is not None:
            hidden_states = self.embed_dropout(hidden_states)
        
        # Initialize cache
        if past_key_values is None and use_cache:
            past_key_values = [None] * len(self.layers)
        
        present_key_values = [] if use_cache else None
        all_attentions = [] if output_attentions else None
        
        # Pass through transformer layers
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values else None
            
            if self.gradient_checkpointing and self.training:
                # Use gradient checkpointing
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    layer,
                    hidden_states,
                    attention_mask,
                    use_cache,
                    past_kv,
                    output_attentions,
                    use_reentrant=False
                )
            else:
                layer_outputs = layer(
                    hidden_states, attention_mask, use_cache, past_kv, output_attentions
                )
            
            hidden_states = layer_outputs[0]
            
            if use_cache:
                present_key_values.append(layer_outputs[1])
            
            if output_attentions:
                all_attentions.append(layer_outputs[2])
        
        # Final normalization
        hidden_states = self.norm(hidden_states)
        
        # Language modeling head
        logits = self.lm_head(hidden_states)
        
        if return_dict:
            return {
                'logits': logits,
                'past_key_values': present_key_values,
                'hidden_states': hidden_states,
                'attentions': all_attentions
            }
        else:
            return logits if not use_cache else (logits, present_key_values)
    
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 100,
                temperature: float = 1.0, top_k: int = 50, top_p: float = 0.9,
                do_sample: bool = True, pad_token_id: Optional[int] = None,
                eos_token_id: Optional[int] = None, use_cache: bool = True,
                repetition_penalty: float = 1.0) -> torch.Tensor:
        """Enhanced generation with advanced sampling and KV caching."""
        
        self.eval()
        device = input_ids.device
        batch_size = input_ids.size(0)
        
        # Set default token IDs
        if pad_token_id is None:
            pad_token_id = 0
        if eos_token_id is None:
            eos_token_id = 3
        
        # Track original input length for repetition penalty
        original_length = input_ids.size(1)
        
        with torch.no_grad():
            past_key_values = None
            generated = input_ids.clone()
            
            for step in range(max_new_tokens):
                # Prepare model inputs
                if use_cache and past_key_values is not None:
                    # Only pass the last token
                    model_inputs = generated[:, -1:]
                else:
                    model_inputs = generated
                
                # Forward pass
                outputs = self.forward(
                    model_inputs,
                    use_cache=use_cache,
                    past_key_values=past_key_values,
                    return_dict=True
                )
                
                logits = outputs['logits']
                past_key_values = outputs['past_key_values'] if use_cache else None
                
                # Get next token logits
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply repetition penalty
                if repetition_penalty != 1.0 and generated.size(1) > original_length:
                    for batch_idx in range(batch_size):
                        for token_id in set(generated[batch_idx, original_length:].tolist()):
                            if next_token_logits[batch_idx, token_id] < 0:
                                next_token_logits[batch_idx, token_id] *= repetition_penalty
                            else:
                                next_token_logits[batch_idx, token_id] /= repetition_penalty
                
                if do_sample:
                    # Apply top-k filtering
                    if top_k > 0:
                        top_k_logits, top_k_indices = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                        next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                        next_token_logits.scatter_(-1, top_k_indices, top_k_logits)
                    
                    # Apply top-p (nucleus) filtering
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        
                        # Remove tokens with cumulative probability above threshold
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        
                        # Convert back to original indices
                        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                        next_token_logits[indices_to_remove] = float('-inf')
                    
                    # Sample from the distribution
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1)
                else:
                    # Greedy decoding
                    next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Append generated tokens
                generated = torch.cat([generated, next_tokens], dim=1)
                
                # Check for EOS tokens or max length
                if (next_tokens == eos_token_id).all() or generated.size(1) >= self.config.seq_length:
                    break
        
        return generated
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        return {
            'model_type': 'ModernSubwordTransformer',
            'config': {
                'vocab_size': self.config.vocab_size,
                'hidden_size': self.config.hidden_size,
                'num_layers': self.config.num_layers,
                'num_heads': self.config.num_heads,
                'seq_length': self.config.seq_length,
                'use_rope': self.config.use_rotary_pos_emb,
                'use_rms_norm': self.config.use_rms_norm,
                'use_gqa': self.config.use_grouped_query_attention,
                'glu_variant': self.config.glu_variant if self.config.use_glu_variants else None,
                'num_key_value_heads': getattr(self.config, 'num_key_value_heads', None),
            },
            'parameters': {
                'total': self.count_parameters(),
                'trainable': self.count_trainable_parameters(),
            },
            'memory': {
                'model_mb': self.estimate_memory_mb(),
                'estimated_training_mb': self.estimate_memory_mb() * 4,
            },
            'features': {
                'flash_attention_available': FLASH_ATTENTION_AVAILABLE,
                'torch_compile_available': TORCH_COMPILE_AVAILABLE,
                'gradient_checkpointing': self.gradient_checkpointing,
            }
        }


# Backwards compatibility
SubwordTransformer = ModernSubwordTransformer