#!/bin/bash
# Copyright (c) 2025 MatN23. All rights reserved.
# Licensed under the Custom License below.

set -e

echo "=================================================="
echo "Compiling Custom CUDA Kernels for Tesla T4"
echo "=================================================="

# Detect GPU architecture
if command -v nvidia-smi &> /dev/null; then
    # Get compute capability (e.g., 7.5 for T4)
    GPU_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d '.')
    echo "✅ Detected GPU compute capability: ${GPU_ARCH}"
else
    echo "⚠️  nvidia-smi not found, defaulting to sm_75 (T4)"
    GPU_ARCH=75
fi

ARCH_FLAG="-arch=sm_${GPU_ARCH}"

# Optimization flags for T4
NVCC_FLAGS="-O3 ${ARCH_FLAG} --compiler-options '-fPIC' --use_fast_math --ptxas-options=-v"

echo ""
echo "Compilation flags: ${NVCC_FLAGS}"
echo ""

# Compile fused loss kernel
echo "1️⃣  Compiling fused_loss.cu..."
nvcc ${NVCC_FLAGS} -shared fused_loss.cu -o fused_loss.so 2>&1 | grep -E "ptxas|error|warning" || true

if [ -f fused_loss.so ]; then
    echo "   ✅ fused_loss.so compiled successfully"
    ls -lh fused_loss.so
else
    echo "   ❌ fused_loss.cu compilation failed"
    exit 1
fi

echo ""

# Compile fused gradient clipping kernel
echo "2️⃣  Compiling fused_grad_clip.cu..."
nvcc ${NVCC_FLAGS} -shared fused_grad_clip.cu -o fused_grad_clip.so 2>&1 | grep -E "ptxas|error|warning" || true

if [ -f fused_grad_clip.so ]; then
    echo "   ✅ fused_grad_clip.so compiled successfully"
    ls -lh fused_grad_clip.so
else
    echo "   ❌ fused_grad_clip.cu compilation failed"
    exit 1
fi

echo ""
echo "=================================================="
echo "✅ All kernels compiled successfully!"
echo "=================================================="
echo ""
echo "Generated files:"
ls -lh *.so
echo ""
echo "Next steps:"
echo "  1. Verify kernels work: python cuda_kernels.py"
echo "  2. Update trainer.py with the new functions"
echo "  3. Run training - kernels will be used automatically"
echo ""