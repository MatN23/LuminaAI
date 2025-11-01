# LuminaAI Dataset Accelerator - Quick Start Guide

## 🚀 5-Minute Setup (macOS)

Since you're on macOS with Apple Silicon, here's your specific setup:

### Step 1: Install Dependencies

```bash
# Install Xcode Command Line Tools (if not already installed)
xcode-select --install

# Install Python dependencies
pip install pybind11 numpy torch
```

### Step 2: Build the Accelerator

```bash
# Navigate to your project directory
cd /Users/matias/Documents/Projects/AI/LuminaAI/Src/Main_Scripts

# Build and install
pip install -e . --no-build-isolation

# If build fails, try verbose mode to see errors:
# pip install -e . --no-build-isolation -v
```

### Step 3: Verify Installation

```bash
python -c "from dataset_accelerator import print_backend_info; print_backend_info()"
```

**Expected Output:**
```
============================================================
LuminaAI Dataset Accelerator
============================================================
Backend: C++
Accelerator Available: True
CUDA Support: False
Version: 1.0.0
============================================================
```

### Step 4: Run Tests

```bash
python tests/test_acceleration.py
```

---

## ✅ Fix IntelliSense Warnings (VSCode)

The warnings you're seeing are **just IDE warnings**, not compilation errors. To fix them:

### Method 1: Auto-configure (Recommended)

```bash
# Install pybind11 first
pip install pybind11

# Get pybind11 include path
python -c "import pybind11; print(pybind11.get_include())"

# This will print something like:
# /Users/matias/Library/Python/3.11/lib/python3.11/site-packages/pybind11/include
```

Then update `.vscode/c_cpp_properties.json` with this path (I've already created a template for you).

### Method 2: Ignore Warnings

The warnings don't affect compilation. You can:

1. **Ignore them** - They're cosmetic
2. **Disable IntelliSense** for C++ temporarily:
   - Press `Cmd+Shift+P`
   - Search: "C/C++: Disable IntelliSense"

---

## 🔧 Troubleshooting

### Issue: "Command not found: pip"

```bash
# Use pip3 instead
pip3 install -e . --no-build-isolation
```

### Issue: "cmake not found"

```bash
# Install cmake via pip
pip install cmake

# Or via Homebrew
brew install cmake
```

### Issue: "Permission denied"

```bash
# Use pip with --user flag
pip install -e . --no-build-isolation --user
```

### Issue: Build fails completely

```bash
# Don't worry! The system will automatically fall back to pure Python
# Your training will still work, just without acceleration

# Check if fallback is active:
python -c "from dataset_accelerator import ACCELERATOR_AVAILABLE; print(f'Accelerator: {ACCELERATOR_AVAILABLE}')"
```

---

## 📊 Test Performance

Create a simple benchmark:

```python
# test_performance.py
from dataset_accelerator import fast_shuffle, ACCELERATOR_AVAILABLE
import numpy as np
import time

print(f"Accelerator Available: {ACCELERATOR_AVAILABLE}")

# Create large array
arr = np.arange(1_000_000, dtype=np.int64)

# Time shuffle
start = time.time()
result = fast_shuffle(arr, seed=42)
elapsed = time.time() - start

print(f"Shuffled 1M elements in {elapsed*1000:.2f}ms")
print(f"Throughput: {len(arr)/elapsed/1e6:.2f}M elements/sec")

if ACCELERATOR_AVAILABLE:
    print("✓ Using C++ acceleration!")
else:
    print("⚠ Using Python fallback")
```

Run it:
```bash
python test_performance.py
```

---

## 🎯 Using in Your Training Code

**No changes needed!** Just use your existing dataset code:

```python
# Your existing Main.py - NO CHANGES REQUIRED!
from core.dataset import BaseTrainingDataset, setup_datasets

# This automatically uses acceleration if available
train_dataset, eval_dataset = setup_datasets(config, tokenizer)

# Check if acceleration is active
if hasattr(train_dataset, 'stats'):
    stats = train_dataset.get_stats()
    print(f"Using acceleration: {stats.get('acceleration_used', False)}")
    print(f"Backend: {stats.get('backend', 'Python')}")
```

---

## 🔄 Update Existing dataset.py

If you want to use the accelerated version explicitly:

### Option 1: Replace dataset.py (Recommended)

```bash
# Backup your current dataset.py
cp dataset.py dataset_original.py

# Use accelerated version
cp dataset_accelerated.py dataset.py
```

### Option 2: Use alongside (for testing)

```python
# In your Main.py, change import:
# from core.dataset import ...
from core.dataset_accelerated import BaseTrainingDataset, setup_datasets
```

---

## 📈 Performance Expectations (macOS Apple Silicon)

| Operation | Python | C++ | Expected Speedup |
|-----------|--------|-----|------------------|
| File Reading | 100 MB/s | 300 MB/s | **3x faster** |
| Shuffling | 10M elem/s | 50M elem/s | **5x faster** |
| Chunking | 50 MB/s | 150 MB/s | **3x faster** |
| Batch Prep | 1K batch/s | 3K batch/s | **3x faster** |

*Note: CUDA not available on Apple Silicon, but C++ multi-threading provides significant speedup*

---

## ❓ FAQ

**Q: Do I need to modify my existing training code?**  
A: No! It's completely plug-and-play.

**Q: What if compilation fails?**  
A: The system automatically falls back to Python. Everything still works.

**Q: Will this work on CUDA GPUs?**  
A: Yes! On Linux/Windows with NVIDIA GPUs, it will use CUDA automatically.

**Q: Can I disable acceleration?**  
A: Yes, just don't install the accelerator package. Use original dataset.py.

**Q: How do I uninstall?**  
A: `pip uninstall dataset_accelerator`

---

## 📝 Next Steps

1. ✅ Build accelerator: `pip install -e . --no-build-isolation`
2. ✅ Run tests: `python tests/test_acceleration.py`
3. ✅ Use in training: No code changes needed!
4. ✅ Monitor performance: Check dataset stats

---

## 🆘 Still Having Issues?

1. **Check Python version**: `python --version` (need 3.8+)
2. **Check pip**: `pip --version`
3. **Try verbose install**: `pip install -e . --no-build-isolation -v`
4. **Share output** of verbose install for debugging

The IntelliSense warnings you're seeing are **cosmetic only**. They won't affect compilation or runtime. Just ignore them or follow the steps above to configure VSCode properly.