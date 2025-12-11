// Copyright (c) 2025 MatN23. All rights reserved.
// Licensed under the Custom License below.

// fused_loss.cu - SIMPLIFIED VERSION
// Fused Cross Entropy + Accuracy computation
//
// Compile with:
// nvcc -O3 -arch=sm_75 --compiler-options '-fPIC' --use_fast_math --ptxas-options=-v -shared fused_loss.cu -o fused_loss.so

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cfloat>
#include <cstdio>
#include <cstdlib>

// Warp reduction
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Block reduction for sum
__device__ __forceinline__ float block_reduce_sum(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    
    val = warp_reduce_sum(val);
    
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;
    if (wid == 0) val = warp_reduce_sum(val);
    
    return val;
}

// Block reduction for max
__device__ __forceinline__ float block_reduce_max(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    
    val = warp_reduce_max(val);
    
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : -FLT_MAX;
    if (wid == 0) val = warp_reduce_max(val);
    
    return val;
}

// Simplified fused kernel
__global__ void fused_cross_entropy_accuracy_kernel(
    const float* __restrict__ logits,
    const int64_t* __restrict__ labels,
    const int64_t pad_token_id,
    float* __restrict__ loss_out,
    float* __restrict__ accuracy_out,
    int64_t* __restrict__ valid_tokens_out,
    const int total_tokens,
    const int vocab_size
) {
    int token_idx = blockIdx.x;
    
    if (token_idx >= total_tokens) return;
    
    int64_t label = labels[token_idx];
    
    // Skip invalid tokens
    if (label == pad_token_id || label < 0 || label >= vocab_size) return;
    
    const float* logit_row = logits + token_idx * vocab_size;
    
    // Step 1: Find max logit for numerical stability
    float max_logit = -FLT_MAX;
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        max_logit = fmaxf(max_logit, logit_row[i]);
    }
    max_logit = block_reduce_max(max_logit);
    
    // Broadcast max to all threads
    __shared__ float s_max_logit;
    if (threadIdx.x == 0) s_max_logit = max_logit;
    __syncthreads();
    max_logit = s_max_logit;
    
    // Step 2: Compute sum of exp(logit - max)
    float sum_exp = 0.0f;
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        sum_exp += expf(logit_row[i] - max_logit);
    }
    sum_exp = block_reduce_sum(sum_exp);
    
    __shared__ float s_sum_exp;
    if (threadIdx.x == 0) s_sum_exp = sum_exp;
    __syncthreads();
    sum_exp = s_sum_exp;
    
    // Step 3: Compute loss for this token
    float label_logit = logit_row[label];
    float log_prob = (label_logit - max_logit) - logf(sum_exp + 1e-10f);
    float token_loss = -log_prob;
    
    // Sanity check
    if (isnan(token_loss) || isinf(token_loss)) {
        token_loss = 10.0f;  // Large but finite value
    }
    
    // Step 4: Find argmax for accuracy (FIXED VERSION)
    // Method: Use shared memory reduction
    __shared__ struct {
        float val;
        int idx;
    } s_data[256];
    
    // Each thread finds its local max
    float my_max = -FLT_MAX;
    int my_idx = 0;
    
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        float val = logit_row[i];
        if (val > my_max) {
            my_max = val;
            my_idx = i;
        }
    }
    
    // Write to shared memory
    s_data[threadIdx.x].val = my_max;
    s_data[threadIdx.x].idx = my_idx;
    __syncthreads();
    
    // Reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            if (s_data[threadIdx.x + stride].val > s_data[threadIdx.x].val) {
                s_data[threadIdx.x] = s_data[threadIdx.x + stride];
            }
        }
        __syncthreads();
    }
    
    // Thread 0 has the result
    int predicted = s_data[0].idx;
    int correct = (predicted == (int)label) ? 1 : 0;
    
    // Step 5: Accumulate results (only thread 0)
    if (threadIdx.x == 0) {
        atomicAdd(loss_out, token_loss);
        atomicAdd(accuracy_out, (float)correct);  // ✅ FIXED: Cast to float
        atomicAdd((unsigned long long*)valid_tokens_out, 1ULL);
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

// Host wrapper
extern "C" {

void fused_cross_entropy_accuracy_launcher(
    const float* logits,
    const int64_t* labels,
    int64_t pad_token_id,
    float* loss_out,
    float* accuracy_out,
    int64_t* valid_tokens_out,
    int total_tokens,
    int vocab_size,
    cudaStream_t stream
) {
    // Clear outputs
    CUDA_CHECK(cudaMemsetAsync(loss_out, 0, sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(accuracy_out, 0, sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(valid_tokens_out, 0, sizeof(int64_t), stream));
    
    // Launch kernel: one block per token, 256 threads per block
    int threads = 256;
    int blocks = total_tokens;
    
    fused_cross_entropy_accuracy_kernel<<<blocks, threads, 0, stream>>>(
        logits,
        labels,
        pad_token_id,
        loss_out,
        accuracy_out,
        valid_tokens_out,
        total_tokens,
        vocab_size
    );
    
    CUDA_CHECK(cudaGetLastError());
}

}  // extern "C"