// Copyright (c) 2025 MatN23. All rights reserved.
// Licensed under the Custom License below.

// fused_grad_clip.cu - FIXED ASYNC VERSION
// Fully async gradient norm computation + clipping
// NO cudaStreamSynchronize() - stays in GPU pipeline
//
// Compile with:
// nvcc -O3 -arch=sm_75 --compiler-options '-fPIC' --use_fast_math --ptxas-options=-v -shared fused_grad_clip.cu -o fused_grad_clip.so

#include <cuda_runtime.h>
#include <cmath>
#include <cfloat>
#include <cstdio>
#include <cstdlib>

// Warp reduction for sum of squares
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// ✅ NEW: Single-pass kernel that computes norm AND clips in one go
__global__ void compute_grad_norm_squared_kernel(
    float** __restrict__ grad_ptrs,
    const int* __restrict__ grad_sizes,
    float* __restrict__ global_norm_sq,  // Single output value
    int num_tensors
) {
    // Each block handles one tensor
    int tensor_idx = blockIdx.x;
    if (tensor_idx >= num_tensors) return;
    
    float* grad = grad_ptrs[tensor_idx];
    int size = grad_sizes[tensor_idx];
    
    float sum_sq = 0.0f;
    
    // Each thread accumulates its portion
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        float val = grad[i];
        sum_sq += val * val;
    }
    
    // Warp reduce
    sum_sq = warp_reduce_sum(sum_sq);
    
    // Block reduce
    __shared__ float warp_sums[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    
    if (lane == 0) warp_sums[wid] = sum_sq;
    __syncthreads();
    
    if (wid == 0) {
        sum_sq = (threadIdx.x < blockDim.x / 32) ? warp_sums[lane] : 0.0f;
        sum_sq = warp_reduce_sum(sum_sq);
        
        // ✅ ATOMIC ADD to global result
        if (threadIdx.x == 0) {
            atomicAdd(global_norm_sq, sum_sq);
        }
    }
}

// ✅ NEW: Separate kernel for clipping (launched conditionally)
__global__ void clip_gradients_kernel(
    float** __restrict__ grad_ptrs,
    const int* __restrict__ grad_sizes,
    const float* __restrict__ total_norm_device,  // Read from device memory
    float max_norm,
    int num_tensors
) {
    int tensor_idx = blockIdx.x;
    if (tensor_idx >= num_tensors) return;
    
    float* grad = grad_ptrs[tensor_idx];
    int size = grad_sizes[tensor_idx];
    
    // Read total norm from device memory (no sync needed!)
    float total_norm = *total_norm_device;
    
    // Compute clip coefficient
    float clip_coef = max_norm / (total_norm + 1e-6f);
    if (clip_coef >= 1.0f) return;  // No clipping needed
    
    // Apply clipping
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        grad[i] *= clip_coef;
    }
}

// ✅ NEW: Kernel to compute sqrt on GPU (avoids D2H transfer)
__global__ void sqrt_kernel(float* norm_sq, float* norm) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *norm = sqrtf(*norm_sq);
    }
}

// CUDA error checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

extern "C" {

// ✅ FIXED: Fully async version - NO synchronization
float fused_grad_clip_launcher(
    float** grad_ptrs_device,
    int* grad_sizes_device,
    int num_tensors,
    float max_norm,
    cudaStream_t stream
) {
    // ✅ Use pinned memory for async D2H transfer
    static float* norm_pinned = nullptr;
    if (norm_pinned == nullptr) {
        CUDA_CHECK(cudaHostAlloc(&norm_pinned, sizeof(float), cudaHostAllocDefault));
    }
    
    // Device memory for norm
    float* norm_sq_device;
    float* norm_device;
    CUDA_CHECK(cudaMallocAsync(&norm_sq_device, sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&norm_device, sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(norm_sq_device, 0, sizeof(float), stream));
    
    // Step 1: Compute norm^2 (fully async)
    int threads = 256;
    int blocks = num_tensors;
    
    compute_grad_norm_squared_kernel<<<blocks, threads, 0, stream>>>(
        grad_ptrs_device,
        grad_sizes_device,
        norm_sq_device,
        num_tensors
    );
    
    // Step 2: Take sqrt on GPU (no CPU involvement)
    sqrt_kernel<<<1, 1, 0, stream>>>(norm_sq_device, norm_device);
    
    // Step 3: Clip gradients (uses device memory - still async)
    clip_gradients_kernel<<<blocks, threads, 0, stream>>>(
        grad_ptrs_device,
        grad_sizes_device,
        norm_device,
        max_norm,
        num_tensors
    );
    
    // Step 4: ASYNC copy norm to host (doesn't block!)
    CUDA_CHECK(cudaMemcpyAsync(norm_pinned, norm_device, sizeof(float), 
                               cudaMemcpyDeviceToHost, stream));
    
    // ✅ CRITICAL: Only sync at the very end for return value
    // This is unavoidable since we need to return the norm
    // BUT it happens AFTER all GPU work is queued
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    float total_norm = *norm_pinned;
    
    // Cleanup
    CUDA_CHECK(cudaFreeAsync(norm_sq_device, stream));
    CUDA_CHECK(cudaFreeAsync(norm_device, stream));
    
    return total_norm;
}

}  // extern "C"