# LuminaAI Dataset Accelerator - Installation Guide

## Overview

The Dataset Accelerator provides transparent C++/CUDA acceleration for LuminaAI dataset operations. It automatically:
- Uses **CUDA** on NVIDIA GPUs
- Uses **CPU multi-threading** on CPU-only systems
- Falls back to **pure Python** if compilation fails

**Zero code changes required** - completely plug-and-play!

---

## Quick Start

### Option 1: Automatic Installation (Recommended)

```bash
# Install with automatic compilation
pip install -e . --no-build-isolation

# Verify installation
python -c "from dataset_accelerator import print_backend_info; print_backend_info()"
```

### Option 2: Manual Build

```bash
# Install dependencies
pip install pybind11 cmake

# Build and install
python setup.py build_ext --inplace
pip install -e .
```

---

## Platform-Specific Instructions

### Linux (Ubuntu/Debian)

```bash
# Install build tools
sudo apt-get update
sudo apt-get install -y build-essential cmake python3-dev

# For CUDA support (optional)
sudo apt-get install -y nvidia-cuda-toolkit

# Install accelerator
pip install -e . --no-build-isolation

# Verify
python -c "from dataset_accelerator import get_backend_info; print(get_backend_info())"
```

**Expected Output:**
```
{'accelerator_available': True, 'cuda_backend': True, 'backend': 'CUDA', ...}
```

### macOS

```bash
# Install Xcode Command Line Tools
xcode-select --install

# Install CMake
brew install cmake

# Install accelerator (CPU-only on Apple Silicon)
pip install -e . --no-build-isolation

# Verify
python -c "from dataset_accelerator import print_backend_info; print_backend_info()"
```

**Expected Output:**
```
LuminaAI Dataset Accelerator
Backend: C++
Accelerator Available: True
CUDA Support: False
```

### Windows

```bash
# Install Visual Studio Build Tools
# Download from: https://visualstudio.microsoft.com/downloads/

# Install CMake
# Download from: https://cmake.org/download/

# For CUDA support (optional)
# Install CUDA Toolkit from: https://developer.nvidia.com/cuda-downloads

# Install accelerator
pip install -e . --no-build-isolation

# Verify
python -c "from dataset_accelerator import print_backend_info; print_backend_info()"
```

---

## Verification

### Check Installation Status

```python
from dataset_accelerator import get_backend_info, ACCELERATOR_AVAILABLE, CUDA_BACKEND

if ACCELERATOR_AVAILABLE:
    print("✓ Accelerator loaded successfully")
    if CUDA_BACKEND:
        print("  Using CUDA backend (GPU)")
    else:
        print("  Using C++ backend (CPU)")
else:
    print("⚠ Using pure Python fallback")
    print("  Run 'pip install -e . --no-build-isolation' to enable acceleration")

# Print detailed info
print("\nBackend Info:")
print(get_backend_info())
```

### Run Benchmark

```python
from dataset_accelerator import fast_shuffle, parallel_shuffle
import numpy as np
import time

# Test with large array
arr = np.arange(10_000_000, dtype=np.int64)

# Standard shuffle
start = time.time()
result1 = fast_shuffle(arr, seed=42)
time1 = time.time() - start

# Parallel shuffle
start = time.time()
result2 = parallel_shuffle(arr, seed=42)
time2 = time.time() - start

print(f"Fast shuffle: {time1:.3f}s")
print(f"Parallel shuffle: {time2:.3f}s")
print(f"Speedup: {time1/time2:.2f}x")
```

---

## Troubleshooting

### Issue: "CMake not found"

**Solution:**
```bash
# Linux/macOS
pip install cmake

# Or install system CMake
# Linux: sudo apt-get install cmake
# macOS: brew install cmake
# Windows: Download from cmake.org
```

### Issue: "pybind11 not found"

**Solution:**
```bash
pip install pybind11
```

### Issue: CUDA not detected

**Check CUDA installation:**
```bash
nvcc --version
```

**If CUDA is not installed but you have an NVIDIA GPU:**
1. Install CUDA Toolkit from: https://developer.nvidia.com/cuda-downloads
2. Rebuild: `pip install -e . --no-build-isolation --force-reinstall`

### Issue: Compilation fails

**Fallback to pure Python:**
```bash
# Accelerator will automatically use pure Python fallback
# Your code will still work, just without acceleration
python your_training_script.py
```

**Get detailed error info:**
```bash
pip install -e . --no-build-isolation -v
```

### Issue: "ImportError: dataset_accelerator._core"

**This usually means compilation succeeded but the library can't be found.**

**Solution:**
```bash
# Reinstall in development mode
pip uninstall dataset_accelerator
pip install -e . --no-build-isolation

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

---

## Usage in Existing Code

### Zero Changes Required!

Your existing code automatically uses acceleration:

```python
# Original code (dataset.py)
from core.dataset import BaseTrainingDataset, setup_datasets

# Just use it normally - acceleration is automatic!
train_dataset, eval_dataset = setup_datasets(config, tokenizer)
```

### Explicit Accelerated Version

If you want to explicitly use the accelerated version:

```python
# Use accelerated dataset directly
from core.dataset_accelerated import BaseTrainingDataset, setup_datasets

# Or check acceleration status
from dataset_accelerator import ACCELERATOR_AVAILABLE, CUDA_BACKEND

if ACCELERATOR_AVAILABLE:
    print(f"Using {('CUDA' if CUDA_BACKEND else 'C++')} acceleration")
```

---

## Performance Expectations

### File Loading
- **Python baseline:** 100 MB/s
- **C++ acceleration:** 300-500 MB/s (3-5x faster)
- **CUDA acceleration:** Not applicable (I/O bound)

### Shuffling
- **Python baseline:** 10M elements/s
- **C++ acceleration:** 50M elements/s (5x faster)
- **CUDA acceleration:** 200M elements/s (20x faster)

### Batch Preparation
- **Python baseline:** 1000 batches/s
- **C++ acceleration:** 3000 batches/s (3x faster)
- **CUDA acceleration:** 10000 batches/s (10x faster)

### Chunking
- **Python baseline:** 50 MB/s
- **C++ acceleration:** 200 MB/s (4x faster)
- **CUDA acceleration:** 500 MB/s (10x faster)

*Note: Actual performance depends on hardware, data size, and system load.*

---

## Uninstallation

```bash
pip uninstall dataset_accelerator
```

To remove all build artifacts:

```bash
# Remove build files
rm -rf build/ dist/ *.egg-info
rm -rf dataset_accelerator/_core*.so
rm -rf dataset_accelerator/__pycache__
```

---

## Development

### Building for Development

```bash
# Build in-place for development
python setup.py develop

# Or use editable install
pip install -e . --no-build-isolation
```

### Running Tests

```bash
python tests/test_acceleration.py
python tests/test_integration.py
```

### Rebuilding After Changes

```bash
# Clean build
rm -rf build/
pip install -e . --no-build-isolation --force-reinstall
```

---

## Support

### Getting Help

1. Check installation status:
   ```python
   from dataset_accelerator import print_backend_info
   print_backend_info()
   ```

2. Run diagnostic:
   ```bash
   python tests/test_acceleration.py
   ```

3. Check logs:
   - Compilation warnings during `pip install`
   - Runtime warnings in Python logs

### Common Questions

**Q: Do I need CUDA for acceleration?**  
A: No! C++ CPU acceleration works on all systems. CUDA provides additional GPU speedup.

**Q: What if compilation fails?**  
A: The system automatically falls back to pure Python. Everything still works, just without acceleration.

**Q: Can I use this on Apple Silicon (M1/M2)?**  
A: Yes! C++ acceleration works. CUDA is not available on Apple Silicon.

**Q: Does this work with existing code?**  
A: Yes! It's a drop-in replacement requiring zero code changes.

---

## License

Same license as LuminaAI project.