/*
 * LuminaAI Dataset Accelerator - CPU Core
 * High-performance C++ implementation with multi-threading
 */

#include <vector>
#include <string>
#include <fstream>
#include <thread>
#include <algorithm>
#include <random>
#include <cstring>
#include <cstdint>
#include <memory>
#include <mutex>

#ifdef CUDA_AVAILABLE
extern "C" {
    void cuda_fast_shuffle(int64_t* data, size_t size);
    void cuda_chunk_tokenize(const char* text, size_t text_len, int32_t* output, size_t* output_len);
}
#endif

namespace dataset_accelerator {

// Fast in-place shuffle using Fisher-Yates algorithm
void fast_shuffle(int64_t* indices, size_t size, uint64_t seed) {
    std::mt19937_64 rng(seed);
    for (size_t i = size - 1; i > 0; --i) {
        std::uniform_int_distribution<size_t> dist(0, i);
        size_t j = dist(rng);
        std::swap(indices[i], indices[j]);
    }
}

// Parallel shuffle for large arrays
void parallel_shuffle(int64_t* indices, size_t size, uint64_t seed) {
#ifdef CUDA_AVAILABLE
    // Use CUDA if available
    cuda_fast_shuffle(indices, size);
#else
    // CPU parallel shuffle
    if (size < 10000) {
        fast_shuffle(indices, size, seed);
        return;
    }
    
    unsigned int num_threads = std::thread::hardware_concurrency();
    size_t chunk_size = size / num_threads;
    
    std::vector<std::thread> threads;
    
    // Shuffle each chunk in parallel
    for (unsigned int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            size_t start = t * chunk_size;
            size_t end = (t == num_threads - 1) ? size : (t + 1) * chunk_size;
            fast_shuffle(indices + start, end - start, seed + t);
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Final global shuffle pass
    fast_shuffle(indices, size, seed + num_threads);
#endif
}

// Fast chunking for base training datasets
struct ChunkResult {
    std::vector<std::vector<int64_t>> chunks;
    size_t total_tokens;
    size_t documents_processed;
};

ChunkResult fast_chunk_documents(
    const std::vector<std::string>& texts,
    size_t seq_length,
    bool overlap = true
) {
    ChunkResult result;
    result.total_tokens = 0;
    result.documents_processed = 0;
    
    std::vector<int64_t> current_tokens;
    current_tokens.reserve(seq_length * 2);
    
    for (const auto& text : texts) {
        if (text.empty()) continue;
        
        // Simple tokenization (space-separated for demo)
        // In production, this would call the actual tokenizer
        size_t start = 0;
        for (size_t i = 0; i <= text.length(); ++i) {
            if (i == text.length() || text[i] == ' ' || text[i] == '\n') {
                if (i > start) {
                    // Convert substring to token ID (simplified)
                    current_tokens.push_back(static_cast<int64_t>(
                        std::hash<std::string>{}(text.substr(start, i - start)) % 50000
                    ));
                }
                start = i + 1;
            }
        }
        
        result.documents_processed++;
        
        // Create chunks
        while (current_tokens.size() >= seq_length + 1) {
            std::vector<int64_t> chunk(
                current_tokens.begin(), 
                current_tokens.begin() + seq_length + 1
            );
            result.chunks.push_back(std::move(chunk));
            result.total_tokens += seq_length + 1;
            
            // Sliding window with overlap
            if (overlap) {
                current_tokens.erase(
                    current_tokens.begin(), 
                    current_tokens.begin() + seq_length
                );
            } else {
                current_tokens.erase(
                    current_tokens.begin(), 
                    current_tokens.begin() + seq_length + 1
                );
            }
        }
    }
    
    // Handle remaining tokens
    if (current_tokens.size() >= 10) {
        // Pad to sequence length
        while (current_tokens.size() < seq_length + 1) {
            current_tokens.push_back(0);
        }
        result.chunks.push_back(current_tokens);
        result.total_tokens += seq_length + 1;
    }
    
    return result;
}

// Parallel file reading with buffering
class FastFileReader {
private:
    std::string filename_;
    size_t buffer_size_;
    std::mutex mutex_;
    
public:
    FastFileReader(const std::string& filename, size_t buffer_size = 1024 * 1024)
        : filename_(filename), buffer_size_(buffer_size) {}
    
    std::vector<std::string> read_lines(size_t max_lines = 0) {
        std::vector<std::string> lines;
        std::ifstream file(filename_, std::ios::binary);
        
        if (!file.is_open()) {
            return lines;
        }
        
        std::string line;
        line.reserve(1024);
        
        size_t count = 0;
        while (std::getline(file, line)) {
            if (!line.empty()) {
                lines.push_back(std::move(line));
                line.clear();
                line.reserve(1024);
                
                count++;
                if (max_lines > 0 && count >= max_lines) {
                    break;
                }
            }
        }
        
        return lines;
    }
    
    // Parallel reading for very large files
    std::vector<std::string> read_lines_parallel(size_t max_lines = 0) {
        std::vector<std::string> lines;
        
        // For files < 10MB, use simple reading
        std::ifstream file(filename_, std::ios::binary | std::ios::ate);
        size_t file_size = file.tellg();
        file.close();
        
        if (file_size < 10 * 1024 * 1024) {
            return read_lines(max_lines);
        }
        
        // For large files, use memory mapping or parallel reading
        // This is a simplified version
        unsigned int num_threads = std::thread::hardware_concurrency();
        size_t chunk_size = file_size / num_threads;
        
        std::vector<std::thread> threads;
        std::vector<std::vector<std::string>> thread_results(num_threads);
        
        for (unsigned int t = 0; t < num_threads; ++t) {
            threads.emplace_back([&, t]() {
                size_t start = t * chunk_size;
                size_t end = (t == num_threads - 1) ? file_size : (t + 1) * chunk_size;
                
                std::ifstream local_file(filename_, std::ios::binary);
                local_file.seekg(start);
                
                std::string line;
                size_t bytes_read = 0;
                
                // Skip partial first line except for first thread
                if (t > 0) {
                    std::getline(local_file, line);
                }
                
                while (bytes_read < (end - start) && std::getline(local_file, line)) {
                    if (!line.empty()) {
                        thread_results[t].push_back(line);
                    }
                    bytes_read += line.length() + 1;
                }
            });
        }
        
        for (auto& thread : threads) {
            thread.join();
        }
        
        // Merge results
        for (auto& result : thread_results) {
            lines.insert(lines.end(), result.begin(), result.end());
        }
        
        if (max_lines > 0 && lines.size() > max_lines) {
            lines.resize(max_lines);
        }
        
        return lines;
    }
};

// Batch preparation with pre-allocation
struct BatchData {
    std::vector<int64_t> input_ids;
    std::vector<int64_t> labels;
    std::vector<float> attention_mask;
    std::vector<float> loss_weights;
    size_t batch_size;
    size_t seq_length;
};

BatchData prepare_batch(
    const std::vector<std::vector<int64_t>>& chunks,
    const std::vector<size_t>& indices,
    size_t seq_length
) {
    BatchData batch;
    batch.batch_size = indices.size();
    batch.seq_length = seq_length;
    
    // Pre-allocate
    size_t total_size = batch.batch_size * seq_length;
    batch.input_ids.reserve(total_size);
    batch.labels.reserve(total_size);
    batch.attention_mask.reserve(total_size);
    batch.loss_weights.reserve(total_size);
    
    // Fill batch
    for (size_t idx : indices) {
        if (idx >= chunks.size()) continue;
        
        const auto& chunk = chunks[idx];
        
        // Input IDs and labels (shifted by 1)
        for (size_t i = 0; i < seq_length && i < chunk.size() - 1; ++i) {
            batch.input_ids.push_back(chunk[i]);
            batch.labels.push_back(chunk[i + 1]);
            
            // Attention mask (1 for valid tokens, 0 for padding)
            batch.attention_mask.push_back(chunk[i] != 0 ? 1.0f : 0.0f);
            batch.loss_weights.push_back(chunk[i] != 0 ? 1.0f : 0.0f);
        }
        
        // Pad if necessary
        size_t current_size = batch.input_ids.size() % seq_length;
        if (current_size > 0) {
            size_t pad_size = seq_length - current_size;
            for (size_t i = 0; i < pad_size; ++i) {
                batch.input_ids.push_back(0);
                batch.labels.push_back(0);
                batch.attention_mask.push_back(0.0f);
                batch.loss_weights.push_back(0.0f);
            }
        }
    }
    
    return batch;
}

// Memory-efficient streaming iterator
class StreamingIterator {
private:
    std::string filename_;
    size_t seq_length_;
    size_t buffer_size_;
    std::ifstream file_;
    std::vector<int64_t> token_buffer_;
    bool exhausted_;
    
public:
    StreamingIterator(const std::string& filename, size_t seq_length, size_t buffer_size = 10000)
        : filename_(filename), seq_length_(seq_length), buffer_size_(buffer_size), exhausted_(false) {
        file_.open(filename_, std::ios::binary);
        token_buffer_.reserve(buffer_size_);
    }
    
    bool has_next() {
        return !exhausted_ && (token_buffer_.size() >= seq_length_ + 1 || file_.is_open());
    }
    
    std::vector<int64_t> next_chunk() {
        // Refill buffer if needed
        while (token_buffer_.size() < seq_length_ + 1 && file_.is_open()) {
            std::string line;
            if (std::getline(file_, line)) {
                // Tokenize and add to buffer (simplified)
                // In production, call actual tokenizer
                if (!line.empty()) {
                    // Dummy tokenization
                    for (char c : line) {
                        token_buffer_.push_back(static_cast<int64_t>(c));
                    }
                }
            } else {
                file_.close();
                break;
            }
        }
        
        // Create chunk
        std::vector<int64_t> chunk;
        if (token_buffer_.size() >= seq_length_ + 1) {
            chunk.assign(
                token_buffer_.begin(), 
                token_buffer_.begin() + seq_length_ + 1
            );
            token_buffer_.erase(
                token_buffer_.begin(), 
                token_buffer_.begin() + seq_length_
            );
        } else if (!token_buffer_.empty()) {
            // Last chunk - pad if needed
            chunk = token_buffer_;
            while (chunk.size() < seq_length_ + 1) {
                chunk.push_back(0);
            }
            token_buffer_.clear();
            exhausted_ = true;
        } else {
            exhausted_ = true;
        }
        
        return chunk;
    }
    
    ~StreamingIterator() {
        if (file_.is_open()) {
            file_.close();
        }
    }
};

} // namespace dataset_accelerator