// transformer_ops.cu
// Fused transformer operations: RMSNorm, RoPE, SwiGLU
// 
// Compile with:
// nvcc -O3 -arch=sm_75 --compiler-options '-fPIC' --use_fast_math --ptxas-options=-v -shared transformer_ops.cu -o transformer_ops.so

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cfloat>
#include <cstdio>

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

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

// ============================================================================
// 1. RMS NORMALIZATION
// ============================================================================

__global__ void rms_norm_kernel(
    const float* __restrict__ input,      // [batch*seq, hidden_size]
    const float* __restrict__ weight,     // [hidden_size]
    float* __restrict__ output,           // [batch*seq, hidden_size]
    const int batch_seq,
    const int hidden_size,
    const float eps
) {
    int token_idx = blockIdx.x;
    if (token_idx >= batch_seq) return;
    
    const float* x = input + token_idx * hidden_size;
    float* y = output + token_idx * hidden_size;
    
    // Step 1: Compute sum of squares
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float val = x[i];
        sum_sq += val * val;
    }
    sum_sq = block_reduce_sum(sum_sq);
    
    // Broadcast RMS to all threads
    __shared__ float s_rms;
    if (threadIdx.x == 0) {
        float mean_sq = sum_sq / hidden_size;
        s_rms = rsqrtf(mean_sq + eps);  // 1 / sqrt(mean_sq + eps)
    }
    __syncthreads();
    
    float rms = s_rms;
    
    // Step 2: Normalize and scale
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        y[i] = x[i] * rms * weight[i];
    }
}

// ============================================================================
// 2. ROTARY POSITION EMBEDDING (RoPE)
// ============================================================================

__global__ void rope_kernel(
    float* __restrict__ q,                // [batch, num_heads, seq_len, head_dim]
    float* __restrict__ k,                // [batch, num_heads, seq_len, head_dim]
    const float* __restrict__ cos,        // [max_seq_len, head_dim/2]
    const float* __restrict__ sin,        // [max_seq_len, head_dim/2]
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int head_dim,
    const int position_offset
) {
    // Each block handles one head of one sequence position
    int batch_idx = blockIdx.z;
    int head_idx = blockIdx.y;
    int pos_idx = blockIdx.x;
    
    if (batch_idx >= batch_size || head_idx >= num_heads || pos_idx >= seq_len) {
        return;
    }
    
    int offset = ((batch_idx * num_heads + head_idx) * seq_len + pos_idx) * head_dim;
    
    // RoPE is applied to pairs of dimensions
    int half_dim = head_dim / 2;
    
    for (int i = threadIdx.x; i < half_dim; i += blockDim.x) {
        int rope_idx = (position_offset + pos_idx) * half_dim + i;
        
        float cos_val = cos[rope_idx];
        float sin_val = sin[rope_idx];
        
        // Apply rotation to Q
        float q0 = q[offset + i];
        float q1 = q[offset + i + half_dim];
        q[offset + i] = q0 * cos_val - q1 * sin_val;
        q[offset + i + half_dim] = q0 * sin_val + q1 * cos_val;
        
        // Apply rotation to K
        float k0 = k[offset + i];
        float k1 = k[offset + i + half_dim];
        k[offset + i] = k0 * cos_val - k1 * sin_val;
        k[offset + i + half_dim] = k0 * sin_val + k1 * cos_val;
    }
}

// Precompute cos/sin cache
__global__ void rope_precompute_kernel(
    float* __restrict__ cos_cache,        // [max_seq_len, head_dim/2]
    float* __restrict__ sin_cache,        // [max_seq_len, head_dim/2]
    const int max_seq_len,
    const int head_dim,
    const float theta
) {
    int pos = blockIdx.x;
    int dim_idx = threadIdx.x;
    
    if (pos >= max_seq_len || dim_idx >= head_dim / 2) return;
    
    float freq = 1.0f / powf(theta, (2.0f * dim_idx) / head_dim);
    float angle = pos * freq;
    
    int idx = pos * (head_dim / 2) + dim_idx;
    cos_cache[idx] = cosf(angle);
    sin_cache[idx] = sinf(angle);
}

// ============================================================================
// 3. SwiGLU ACTIVATION
// ============================================================================

__global__ void swiglu_kernel(
    const float* __restrict__ gate,       // [batch*seq, intermediate_size]
    const float* __restrict__ up,         // [batch*seq, intermediate_size]
    float* __restrict__ output,           // [batch*seq, intermediate_size]
    const int total_tokens,
    const int intermediate_size
) {
    int token_idx = blockIdx.x;
    if (token_idx >= total_tokens) return;
    
    int offset = token_idx * intermediate_size;
    
    for (int i = threadIdx.x; i < intermediate_size; i += blockDim.x) {
        float g = gate[offset + i];
        float u = up[offset + i];
        
        // SwiGLU: gate * silu(up)
        // silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
        float silu_u = u / (1.0f + expf(-u));
        output[offset + i] = g * silu_u;
    }
}

// Fused version that combines gate projection + SwiGLU
__global__ void swiglu_fused_kernel(
    const float* __restrict__ input,      // [batch*seq, hidden_size]
    const float* __restrict__ gate_weight,// [intermediate_size, hidden_size]
    const float* __restrict__ up_weight,  // [intermediate_size, hidden_size]
    const float* __restrict__ gate_bias,  // [intermediate_size] (optional)
    const float* __restrict__ up_bias,    // [intermediate_size] (optional)
    float* __restrict__ output,           // [batch*seq, intermediate_size]
    const int total_tokens,
    const int hidden_size,
    const int intermediate_size,
    const bool use_bias
) {
    int token_idx = blockIdx.x;
    int out_idx = threadIdx.x + blockIdx.y * blockDim.x;
    
    if (token_idx >= total_tokens || out_idx >= intermediate_size) return;
    
    const float* x = input + token_idx * hidden_size;
    
    // Compute gate projection
    float gate_val = 0.0f;
    for (int i = 0; i < hidden_size; i++) {
        gate_val += gate_weight[out_idx * hidden_size + i] * x[i];
    }
    if (use_bias && gate_bias) {
        gate_val += gate_bias[out_idx];
    }
    
    // Compute up projection
    float up_val = 0.0f;
    for (int i = 0; i < hidden_size; i++) {
        up_val += up_weight[out_idx * hidden_size + i] * x[i];
    }
    if (use_bias && up_bias) {
        up_val += up_bias[out_idx];
    }
    
    // SwiGLU activation
    float silu_up = up_val / (1.0f + expf(-up_val));
    output[token_idx * intermediate_size + out_idx] = gate_val * silu_up;
}

// ============================================================================
// HOST WRAPPERS
// ============================================================================

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

// RMSNorm launcher
void rms_norm_launcher(
    const float* input,
    const float* weight,
    float* output,
    int batch_seq,
    int hidden_size,
    float eps,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = batch_seq;
    
    rms_norm_kernel<<<blocks, threads, 0, stream>>>(
        input, weight, output, batch_seq, hidden_size, eps
    );
    
    CUDA_CHECK(cudaGetLastError());
}

// RoPE precompute launcher
void rope_precompute_launcher(
    float* cos_cache,
    float* sin_cache,
    int max_seq_len,
    int head_dim,
    float theta,
    cudaStream_t stream
) {
    int blocks = max_seq_len;
    int threads = head_dim / 2;
    
    rope_precompute_kernel<<<blocks, threads, 0, stream>>>(
        cos_cache, sin_cache, max_seq_len, head_dim, theta
    );
    
    CUDA_CHECK(cudaGetLastError());
}

// RoPE apply launcher
void rope_apply_launcher(
    float* q,
    float* k,
    const float* cos,
    const float* sin,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    int position_offset,
    cudaStream_t stream
) {
    dim3 blocks(seq_len, num_heads, batch_size);
    int threads = 128;
    
    rope_kernel<<<blocks, threads, 0, stream>>>(
        q, k, cos, sin, batch_size, num_heads, seq_len, head_dim, position_offset
    );
    
    CUDA_CHECK(cudaGetLastError());
}

// SwiGLU launcher
void swiglu_launcher(
    const float* gate,
    const float* up,
    float* output,
    int total_tokens,
    int intermediate_size,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = total_tokens;
    
    swiglu_kernel<<<blocks, threads, 0, stream>>>(
        gate, up, output, total_tokens, intermediate_size
    );
    
    CUDA_CHECK(cudaGetLastError());
}

// Fused SwiGLU launcher
void swiglu_fused_launcher(
    const float* input,
    const float* gate_weight,
    const float* up_weight,
    const float* gate_bias,
    const float* up_bias,
    float* output,
    int total_tokens,
    int hidden_size,
    int intermediate_size,
    bool use_bias,
    cudaStream_t stream
) {
    int threads = 256;
    dim3 blocks(total_tokens, (intermediate_size + threads - 1) / threads);
    
    swiglu_fused_kernel<<<blocks, threads, 0, stream>>>(
        input, gate_weight, up_weight, gate_bias, up_bias,
        output, total_tokens, hidden_size, intermediate_size, use_bias
    );
    
    CUDA_CHECK(cudaGetLastError());
}

}  // extern "C"