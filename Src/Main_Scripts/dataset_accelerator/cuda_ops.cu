/*
 * LuminaAI Dataset Accelerator - CUDA Operations
 * GPU-accelerated data processing
 */

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdint>
#include <cstdio>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            return; \
        } \
    } while(0)

// Fast parallel shuffle kernel
__global__ void shuffle_kernel(int64_t* data, size_t size, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= size) return;
    
    // Initialize RNG state
    curandState state;
    curand_init(seed, idx, 0, &state);
    
    // Generate random swap partner
    size_t swap_idx = curand(&state) % size;
    
    // Atomic swap (simplified - may have race conditions)
    int64_t temp = data[idx];
    data[idx] = data[swap_idx];
    data[swap_idx] = temp;
}

// Optimized shuffle using parallel Fisher-Yates
__global__ void fisher_yates_kernel(int64_t* data, size_t size, unsigned long long seed, int iteration) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Calculate position from the end
    size_t pos = size - 1 - (idx + iteration * blockDim.x * gridDim.x);
    
    if (pos <= 0 || pos >= size) return;
    
    // Initialize RNG
    curandState state;
    curand_init(seed, idx + iteration * 1000, 0, &state);
    
    // Generate random index in range [0, pos]
    size_t swap_pos = curand(&state) % (pos + 1);
    
    // Swap
    int64_t temp = data[pos];
    data[pos] = data[swap_pos];
    data[swap_pos] = temp;
}

extern "C" void cuda_fast_shuffle(int64_t* data, size_t size) {
    if (size == 0) return;
    
    int64_t* d_data;
    size_t bytes = size * sizeof(int64_t);
    
    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_data, bytes));
    
    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_data, data, bytes, cudaMemcpyHostToDevice));
    
    // Launch kernel
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    
    unsigned long long seed = 12345ULL;
    
    // Multiple passes for better randomness
    int num_passes = (size > 1000000) ? 5 : 3;
    for (int pass = 0; pass < num_passes; ++pass) {
        fisher_yates_kernel<<<blocks, threads>>>(d_data, size, seed + pass, pass);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    // Copy back to host
    CUDA_CHECK(cudaMemcpy(data, d_data, bytes, cudaMemcpyDeviceToHost));
    
    // Free device memory
    CUDA_CHECK(cudaFree(d_data));
}

// Fast batch preparation kernel
__global__ void prepare_batch_kernel(
    const int64_t* chunks,
    const size_t* chunk_offsets,
    const size_t* indices,
    int64_t* input_ids,
    int64_t* labels,
    float* attention_mask,
    float* loss_weights,
    size_t batch_size,
    size_t seq_length
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = idx / seq_length;
    int seq_idx = idx % seq_length;
    
    if (batch_idx >= batch_size) return;
    
    // Get chunk for this batch element
    size_t chunk_idx = indices[batch_idx];
    size_t chunk_start = chunk_offsets[chunk_idx];
    size_t chunk_end = chunk_offsets[chunk_idx + 1];
    
    // Bounds check
    if (seq_idx >= (chunk_end - chunk_start - 1)) {
        // Padding
        input_ids[idx] = 0;
        labels[idx] = 0;
        attention_mask[idx] = 0.0f;
        loss_weights[idx] = 0.0f;
    } else {
        // Copy token
        int64_t token = chunks[chunk_start + seq_idx];
        input_ids[idx] = token;
        labels[idx] = chunks[chunk_start + seq_idx + 1];
        attention_mask[idx] = (token != 0) ? 1.0f : 0.0f;
        loss_weights[idx] = (token != 0) ? 1.0f : 0.0f;
    }
}

extern "C" void cuda_prepare_batch(
    const int64_t* chunks,
    const size_t* chunk_offsets,
    const size_t* indices,
    int64_t* input_ids,
    int64_t* labels,
    float* attention_mask,
    float* loss_weights,
    size_t batch_size,
    size_t seq_length,
    size_t num_chunks
) {
    // Allocate device memory
    int64_t* d_chunks;
    size_t* d_chunk_offsets;
    size_t* d_indices;
    int64_t* d_input_ids;
    int64_t* d_labels;
    float* d_attention_mask;
    float* d_loss_weights;
    
    size_t total_tokens = chunk_offsets[num_chunks];
    
    CUDA_CHECK(cudaMalloc(&d_chunks, total_tokens * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_chunk_offsets, (num_chunks + 1) * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&d_indices, batch_size * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&d_input_ids, batch_size * seq_length * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_labels, batch_size * seq_length * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_attention_mask, batch_size * seq_length * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_loss_weights, batch_size * seq_length * sizeof(float)));
    
    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_chunks, chunks, total_tokens * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_chunk_offsets, chunk_offsets, (num_chunks + 1) * sizeof(size_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_indices, indices, batch_size * sizeof(size_t), cudaMemcpyHostToDevice));
    
    // Launch kernel
    int threads = 256;
    int blocks = (batch_size * seq_length + threads - 1) / threads;
    
    prepare_batch_kernel<<<blocks, threads>>>(
        d_chunks, d_chunk_offsets, d_indices,
        d_input_ids, d_labels, d_attention_mask, d_loss_weights,
        batch_size, seq_length
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy results back
    CUDA_CHECK(cudaMemcpy(input_ids, d_input_ids, batch_size * seq_length * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(labels, d_labels, batch_size * seq_length * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(attention_mask, d_attention_mask, batch_size * seq_length * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(loss_weights, d_loss_weights, batch_size * seq_length * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Free device memory
    CUDA_CHECK(cudaFree(d_chunks));
    CUDA_CHECK(cudaFree(d_chunk_offsets));
    CUDA_CHECK(cudaFree(d_indices));
    CUDA_CHECK(cudaFree(d_input_ids));
    CUDA_CHECK(cudaFree(d_labels));
    CUDA_CHECK(cudaFree(d_attention_mask));
    CUDA_CHECK(cudaFree(d_loss_weights));
}

// Parallel tokenization kernel (simplified)
__global__ void tokenize_kernel(
    const char* text,
    size_t text_len,
    int32_t* tokens,
    size_t* token_count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= text_len) return;
    
    // Very simplified tokenization (character-level for demo)
    // In production, this would use a proper tokenizer
    if (text[idx] != ' ' && text[idx] != '\n') {
        size_t pos = atomicAdd((unsigned long long*)token_count, 1);
        tokens[pos] = static_cast<int32_t>(text[idx]);
    }
}

extern "C" void cuda_chunk_tokenize(const char* text, size_t text_len, int32_t* output, size_t* output_len) {
    char* d_text;
    int32_t* d_tokens;
    size_t* d_token_count;
    
    // Allocate
    CUDA_CHECK(cudaMalloc(&d_text, text_len));
    CUDA_CHECK(cudaMalloc(&d_tokens, text_len * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_token_count, sizeof(size_t)));
    
    // Copy input
    CUDA_CHECK(cudaMemcpy(d_text, text, text_len, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_token_count, 0, sizeof(size_t)));
    
    // Launch
    int threads = 256;
    int blocks = (text_len + threads - 1) / threads;
    tokenize_kernel<<<blocks, threads>>>(d_text, text_len, d_tokens, d_token_count);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy back
    CUDA_CHECK(cudaMemcpy(output_len, d_token_count, sizeof(size_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(output, d_tokens, (*output_len) * sizeof(int32_t), cudaMemcpyDeviceToHost));
    
    // Free
    CUDA_CHECK(cudaFree(d_text));
    CUDA_CHECK(cudaFree(d_tokens));
    CUDA_CHECK(cudaFree(d_token_count));
}