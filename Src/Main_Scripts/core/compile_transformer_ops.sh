#!/bin/bash
# Copyright (c) 2025 MatN23. All rights reserved.
# Licensed under the Custom License below.

# compile_transformer_ops.sh
# Compile RMSNorm, RoPE, and SwiGLU CUDA kernels

set -e

echo "=================================================="
echo "Compiling Transformer CUDA Kernels"
echo "=================================================="

# Detect GPU architecture
if command -v nvidia-smi &> /dev/null; then
    GPU_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d '.')
    echo "✅ Detected GPU compute capability: ${GPU_ARCH}"
else
    echo "⚠️  nvidia-smi not found, defaulting to sm_75 (T4)"
    GPU_ARCH=75
fi

ARCH_FLAG="-arch=sm_${GPU_ARCH}"

# Optimization flags
NVCC_FLAGS="-O3 ${ARCH_FLAG} --compiler-options '-fPIC' --use_fast_math --ptxas-options=-v"

echo ""
echo "Compilation flags: ${NVCC_FLAGS}"
echo ""

# Compile transformer ops
echo "🔨 Compiling transformer_ops.cu..."
nvcc ${NVCC_FLAGS} -shared transformer_ops.cu -o transformer_ops.so 2>&1 | grep -E "ptxas|error|warning" || true

if [ -f transformer_ops.so ]; then
    echo "   ✅ transformer_ops.so compiled successfully"
    ls -lh transformer_ops.so
else
    echo "   ❌ transformer_ops.cu compilation failed"
    exit 1
fi

echo ""
echo "=================================================="
echo "✅ Compilation complete!"
echo "=================================================="
echo ""
echo "Generated file:"
ls -lh transformer_ops.so
echo ""
echo "Next steps:"
echo "  1. Test: python transformer_ops.py"
echo "  2. Benchmark: python benchmark_transformer_ops.py"
echo "  3. Integrate into your model (see INTEGRATION_GUIDE.md)"
echo ""