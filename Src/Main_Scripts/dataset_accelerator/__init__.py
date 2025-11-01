"""
LuminaAI Dataset Accelerator
High-performance dataset operations with automatic backend selection

Automatically uses:
- CUDA acceleration on NVIDIA GPUs
- CPU multi-threading on CPU-only systems
- Pure Python fallback if compilation failed

Usage is identical to Python implementation - completely plug-and-play!
"""

import sys
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

# Try to import compiled C++/CUDA backend
ACCELERATOR_AVAILABLE = False
CUDA_BACKEND = False

try:
    from . import _core
    ACCELERATOR_AVAILABLE = True
    CUDA_BACKEND = getattr(_core, 'cuda_available', False)
    
    logging.info("✓ Dataset accelerator loaded successfully")
    if CUDA_BACKEND:
        logging.info("  Backend: CUDA (GPU acceleration)")
    else:
        logging.info("  Backend: C++ (CPU multi-threading)")
        
except ImportError as e:
    logging.warning(f"Dataset accelerator not available: {e}")
    logging.warning("  Falling back to pure Python implementation")
    logging.warning("  To enable acceleration, run: pip install -e . --no-build-isolation")

# Pure Python fallback implementations
class PythonFastFileReader:
    """Pure Python file reader fallback"""
    
    def __init__(self, filename: str, buffer_size: int = 1024 * 1024):
        self.filename = filename
        self.buffer_size = buffer_size
    
    def read_lines(self, max_lines: int = 0) -> List[str]:
        lines = []
        with open(self.filename, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if line.strip():
                    lines.append(line.strip())
                if max_lines > 0 and i >= max_lines:
                    break
        return lines
    
    def read_lines_parallel(self, max_lines: int = 0) -> List[str]:
        # In pure Python, just use regular reading
        return self.read_lines(max_lines)


class PythonStreamingIterator:
    """Pure Python streaming iterator fallback"""
    
    def __init__(self, filename: str, seq_length: int, buffer_size: int = 10000):
        self.filename = filename
        self.seq_length = seq_length
        self.buffer_size = buffer_size
        self.file = open(filename, 'r', encoding='utf-8')
        self.token_buffer = []
        self.exhausted = False
    
    def has_next(self) -> bool:
        return not self.exhausted
    
    def next_chunk(self) -> List[int]:
        # Refill buffer if needed
        while len(self.token_buffer) < self.seq_length + 1 and not self.exhausted:
            line = self.file.readline()
            if line:
                # Simple tokenization (in practice, would use real tokenizer)
                tokens = [ord(c) for c in line.strip()]
                self.token_buffer.extend(tokens)
            else:
                self.exhausted = True
                break
        
        # Create chunk
        if len(self.token_buffer) >= self.seq_length + 1:
            chunk = self.token_buffer[:self.seq_length + 1]
            self.token_buffer = self.token_buffer[self.seq_length:]
            return chunk
        elif self.token_buffer:
            # Last chunk - pad if needed
            chunk = self.token_buffer + [0] * (self.seq_length + 1 - len(self.token_buffer))
            self.token_buffer = []
            return chunk
        else:
            return []
    
    def __del__(self):
        if hasattr(self, 'file'):
            self.file.close()


def python_fast_shuffle(arr: np.ndarray, seed: int = 42) -> np.ndarray:
    """Pure Python shuffle fallback"""
    rng = np.random.RandomState(seed)
    result = arr.copy()
    rng.shuffle(result)
    return result


def python_fast_chunk_documents(
    texts: List[str],
    seq_length: int,
    overlap: bool = True
) -> Dict[str, Any]:
    """Pure Python document chunking fallback"""
    chunks = []
    total_tokens = 0
    documents_processed = 0
    current_tokens = []
    
    for text in texts:
        if not text:
            continue
        
        # Simple tokenization (space-separated)
        tokens = [hash(word) % 50000 for word in text.split()]
        current_tokens.extend(tokens)
        documents_processed += 1
        
        # Create chunks
        while len(current_tokens) >= seq_length + 1:
            chunk = current_tokens[:seq_length + 1]
            chunks.append(chunk)
            total_tokens += seq_length + 1
            
            if overlap:
                current_tokens = current_tokens[seq_length:]
            else:
                current_tokens = current_tokens[seq_length + 1:]
    
    # Handle remaining tokens
    if len(current_tokens) >= 10:
        chunk = current_tokens + [0] * (seq_length + 1 - len(current_tokens))
        chunks.append(chunk)
        total_tokens += seq_length + 1
    
    return {
        'chunks': chunks,
        'total_tokens': total_tokens,
        'documents_processed': documents_processed
    }


def python_prepare_batch(
    chunks: List[List[int]],
    indices: List[int],
    seq_length: int
) -> Dict[str, Any]:
    """Pure Python batch preparation fallback"""
    batch_size = len(indices)
    
    input_ids = np.zeros((batch_size, seq_length), dtype=np.int64)
    labels = np.zeros((batch_size, seq_length), dtype=np.int64)
    attention_mask = np.zeros((batch_size, seq_length), dtype=np.float32)
    loss_weights = np.zeros((batch_size, seq_length), dtype=np.float32)
    
    for i, idx in enumerate(indices):
        if idx >= len(chunks):
            continue
        
        chunk = chunks[idx]
        length = min(seq_length, len(chunk) - 1)
        
        for j in range(length):
            input_ids[i, j] = chunk[j]
            labels[i, j] = chunk[j + 1]
            attention_mask[i, j] = 1.0 if chunk[j] != 0 else 0.0
            loss_weights[i, j] = 1.0 if chunk[j] != 0 else 0.0
    
    return {
        'input_ids': input_ids,
        'labels': labels,
        'attention_mask': attention_mask,
        'loss_weights': loss_weights,
        'batch_size': batch_size,
        'seq_length': seq_length
    }


# Unified interface with automatic backend selection
class FastFileReader:
    """File reader with automatic backend selection"""
    
    def __init__(self, filename: str, buffer_size: int = 1024 * 1024):
        if ACCELERATOR_AVAILABLE:
            self._reader = _core.FastFileReader(filename, buffer_size)
        else:
            self._reader = PythonFastFileReader(filename, buffer_size)
    
    def read_lines(self, max_lines: int = 0) -> List[str]:
        return self._reader.read_lines(max_lines)
    
    def read_lines_parallel(self, max_lines: int = 0) -> List[str]:
        return self._reader.read_lines_parallel(max_lines)


class StreamingIterator:
    """Streaming iterator with automatic backend selection"""
    
    def __init__(self, filename: str, seq_length: int, buffer_size: int = 10000):
        if ACCELERATOR_AVAILABLE:
            self._iterator = _core.StreamingIterator(filename, seq_length, buffer_size)
        else:
            self._iterator = PythonStreamingIterator(filename, seq_length, buffer_size)
    
    def has_next(self) -> bool:
        return self._iterator.has_next()
    
    def next_chunk(self) -> List[int]:
        return self._iterator.next_chunk()


def fast_shuffle(arr: np.ndarray, seed: int = 42) -> np.ndarray:
    """Fast shuffle with automatic backend selection"""
    if ACCELERATOR_AVAILABLE:
        return _core.fast_shuffle(arr, seed)
    else:
        return python_fast_shuffle(arr, seed)


def parallel_shuffle(arr: np.ndarray, seed: int = 42) -> np.ndarray:
    """Parallel shuffle with automatic backend selection"""
    if ACCELERATOR_AVAILABLE:
        return _core.parallel_shuffle(arr, seed)
    else:
        return python_fast_shuffle(arr, seed)


def fast_chunk_documents(
    texts: List[str],
    seq_length: int,
    overlap: bool = True
) -> Dict[str, Any]:
    """Fast document chunking with automatic backend selection"""
    if ACCELERATOR_AVAILABLE:
        return _core.fast_chunk_documents(texts, seq_length, overlap)
    else:
        return python_fast_chunk_documents(texts, seq_length, overlap)


def prepare_batch(
    chunks: List[List[int]],
    indices: List[int],
    seq_length: int
) -> Dict[str, Any]:
    """Batch preparation with automatic backend selection"""
    if ACCELERATOR_AVAILABLE:
        return _core.prepare_batch(chunks, indices, seq_length)
    else:
        return python_prepare_batch(chunks, indices, seq_length)


# Status reporting
def get_backend_info() -> Dict[str, Any]:
    """Get information about the active backend"""
    return {
        'accelerator_available': ACCELERATOR_AVAILABLE,
        'cuda_backend': CUDA_BACKEND,
        'backend': 'CUDA' if CUDA_BACKEND else ('C++' if ACCELERATOR_AVAILABLE else 'Python'),
        'version': getattr(_core, '__version__', '1.0.0') if ACCELERATOR_AVAILABLE else '1.0.0-python'
    }


def print_backend_info():
    """Print backend information"""
    info = get_backend_info()
    print("="*60)
    print("LuminaAI Dataset Accelerator")
    print("="*60)
    print(f"Backend: {info['backend']}")
    print(f"Accelerator Available: {info['accelerator_available']}")
    print(f"CUDA Support: {info['cuda_backend']}")
    print(f"Version: {info['version']}")
    print("="*60)


# Export public API
__all__ = [
    'FastFileReader',
    'StreamingIterator',
    'fast_shuffle',
    'parallel_shuffle',
    'fast_chunk_documents',
    'prepare_batch',
    'get_backend_info',
    'print_backend_info',
    'ACCELERATOR_AVAILABLE',
    'CUDA_BACKEND',
]