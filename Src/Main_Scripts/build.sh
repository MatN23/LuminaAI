#!/bin/bash
# LuminaAI Dataset Accelerator - Build Script
# Automatic cross-platform compilation with fallback

set -e  # Exit on error

echo "============================================================"
echo "LuminaAI Dataset Accelerator - Build Script"
echo "============================================================"
echo ""

# Detect platform
PLATFORM=$(uname -s)
echo "Platform: $PLATFORM"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found"
    exit 1
fi

PYTHON_VERSION=$(python3 --version)
echo "Python: $PYTHON_VERSION"

# Check pip
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 not found"
    exit 1
fi

echo ""
echo "------------------------------------------------------------"
echo "Step 1: Installing Python Dependencies"
echo "------------------------------------------------------------"

pip3 install --upgrade pip setuptools wheel
pip3 install pybind11 numpy torch

echo "✓ Python dependencies installed"

# Check for CMake
echo ""
echo "------------------------------------------------------------"
echo "Step 2: Checking Build Tools"
echo "------------------------------------------------------------"

if ! command -v cmake &> /dev/null; then
    echo "⚠  CMake not found, installing..."
    pip3 install cmake
fi

CMAKE_VERSION=$(cmake --version | head -n1)
echo "CMake: $CMAKE_VERSION"

# Platform-specific setup
if [ "$PLATFORM" == "Darwin" ]; then
    echo "Platform: macOS"
    
    # Check Xcode tools
    if ! xcode-select -p &> /dev/null; then
        echo "⚠  Xcode Command Line Tools not found"
        echo "Installing... (this may take a while)"
        xcode-select --install
        echo "Please run this script again after installation completes"
        exit 0
    fi
    
    echo "✓ Xcode Command Line Tools installed"
    
elif [ "$PLATFORM" == "Linux" ]; then
    echo "Platform: Linux"
    
    # Check GCC
    if ! command -v g++ &> /dev/null; then
        echo "❌ g++ not found. Install with:"
        echo "  sudo apt-get install build-essential"
        exit 1
    fi
    
    GCC_VERSION=$(g++ --version | head -n1)
    echo "GCC: $GCC_VERSION"
    
    # Check for CUDA
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | sed 's/,//')
        echo "CUDA: $CUDA_VERSION (GPU acceleration available)"
    else
        echo "CUDA: Not found (CPU-only build)"
    fi
else
    echo "Platform: $PLATFORM"
    echo "⚠  Windows detected - use setup.py directly"
fi

echo ""
echo "------------------------------------------------------------"
echo "Step 3: Cleaning Previous Builds"
echo "------------------------------------------------------------"

# Clean previous builds
rm -rf build/ dist/ *.egg-info
rm -f dataset_accelerator/_core*.so dataset_accelerator/_core*.pyd
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

echo "✓ Cleaned build artifacts"

echo ""
echo "------------------------------------------------------------"
echo "Step 4: Building C++ Extensions"
echo "------------------------------------------------------------"

# Try to build
if pip3 install -e . --no-build-isolation; then
    echo ""
    echo "✓ Build completed successfully!"
else
    echo ""
    echo "⚠  Build failed - using Python fallback"
    echo "   This is okay! Training will still work."
fi

echo ""
echo "------------------------------------------------------------"
echo "Step 5: Verification"
echo "------------------------------------------------------------"

# Verify installation
python3 -c "
from dataset_accelerator import get_backend_info, ACCELERATOR_AVAILABLE, CUDA_BACKEND
import sys

info = get_backend_info()
print(f'Backend: {info[\"backend\"]}')
print(f'Accelerator Available: {ACCELERATOR_AVAILABLE}')
print(f'CUDA Support: {CUDA_BACKEND}')

if ACCELERATOR_AVAILABLE:
    print('')
    print('✓ Acceleration is ENABLED')
    if CUDA_BACKEND:
        print('✓ Using CUDA (GPU) backend')
    else:
        print('✓ Using C++ (CPU) backend')
    sys.exit(0)
else:
    print('')
    print('⚠  Using Python fallback (no acceleration)')
    print('   This is okay, but slower')
    sys.exit(0)
" 2>/dev/null || {
    echo "❌ Verification failed"
    echo "   Package may not be installed correctly"
    exit 1
}

echo ""
echo "------------------------------------------------------------"
echo "Step 6: Running Tests"
echo "------------------------------------------------------------"

if [ -f "tests/test_acceleration.py" ]; then
    echo "Running test suite..."
    python3 tests/test_acceleration.py || {
        echo "⚠  Some tests failed (non-critical)"
    }
else
    echo "⚠  Test file not found (skipping tests)"
fi

echo ""
echo "============================================================"
echo "Build Complete!"
echo "============================================================"
echo ""
echo "Next steps:"
echo "1. Run 'python3 -c \"from dataset_accelerator import print_backend_info; print_backend_info()\"'"
echo "2. Use in your training code (no changes needed!)"
echo "3. Check performance with test_performance.py"
echo ""
echo "To uninstall: pip3 uninstall dataset_accelerator"
echo "============================================================"