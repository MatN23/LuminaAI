/*
 * LuminaAI Dataset Accelerator - Python Bindings
 * PyBind11 interface for C++/CUDA acceleration
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <string>

namespace py = pybind11;

// Forward declarations from core.cpp
namespace dataset_accelerator {
    void fast_shuffle(int64_t* indices, size_t size, uint64_t seed);
    void parallel_shuffle(int64_t* indices, size_t size, uint64_t seed);
    
    struct ChunkResult {
        std::vector<std::vector<int64_t>> chunks;
        size_t total_tokens;
        size_t documents_processed;
    };
    
    ChunkResult fast_chunk_documents(
        const std::vector<std::string>& texts,
        size_t seq_length,
        bool overlap
    );
    
    class FastFileReader {
    public:
        FastFileReader(const std::string& filename, size_t buffer_size);
        std::vector<std::string> read_lines(size_t max_lines);
        std::vector<std::string> read_lines_parallel(size_t max_lines);
    };
    
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
    );
    
    class StreamingIterator {
    public:
        StreamingIterator(const std::string& filename, size_t seq_length, size_t buffer_size);
        bool has_next();
        std::vector<int64_t> next_chunk();
    };
}

// Python wrapper functions
py::array_t<int64_t> py_fast_shuffle(py::array_t<int64_t> arr, uint64_t seed) {
    py::buffer_info buf = arr.request();
    
    if (buf.ndim != 1) {
        throw std::runtime_error("Array must be 1-dimensional");
    }
    
    auto result = py::array_t<int64_t>(buf.size);
    py::buffer_info result_buf = result.request();
    
    int64_t* ptr = static_cast<int64_t*>(buf.ptr);
    int64_t* result_ptr = static_cast<int64_t*>(result_buf.ptr);
    
    // Copy data
    std::memcpy(result_ptr, ptr, buf.size * sizeof(int64_t));
    
    // Shuffle in place
    dataset_accelerator::fast_shuffle(result_ptr, buf.size, seed);
    
    return result;
}

py::array_t<int64_t> py_parallel_shuffle(py::array_t<int64_t> arr, uint64_t seed) {
    py::buffer_info buf = arr.request();
    
    if (buf.ndim != 1) {
        throw std::runtime_error("Array must be 1-dimensional");
    }
    
    auto result = py::array_t<int64_t>(buf.size);
    py::buffer_info result_buf = result.request();
    
    int64_t* ptr = static_cast<int64_t*>(buf.ptr);
    int64_t* result_ptr = static_cast<int64_t*>(result_buf.ptr);
    
    // Copy data
    std::memcpy(result_ptr, ptr, buf.size * sizeof(int64_t));
    
    // Parallel shuffle
    dataset_accelerator::parallel_shuffle(result_ptr, buf.size, seed);
    
    return result;
}

py::dict py_fast_chunk_documents(
    const std::vector<std::string>& texts,
    size_t seq_length,
    bool overlap
) {
    auto result = dataset_accelerator::fast_chunk_documents(texts, seq_length, overlap);
    
    py::dict output;
    output["chunks"] = result.chunks;
    output["total_tokens"] = result.total_tokens;
    output["documents_processed"] = result.documents_processed;
    
    return output;
}

py::dict py_prepare_batch(
    const std::vector<std::vector<int64_t>>& chunks,
    const std::vector<size_t>& indices,
    size_t seq_length
) {
    auto result = dataset_accelerator::prepare_batch(chunks, indices, seq_length);
    
    py::dict output;
    
    // Convert to numpy arrays
    auto input_ids = py::array_t<int64_t>({result.batch_size, seq_length});
    auto labels = py::array_t<int64_t>({result.batch_size, seq_length});
    auto attention_mask = py::array_t<float>({result.batch_size, seq_length});
    auto loss_weights = py::array_t<float>({result.batch_size, seq_length});
    
    auto input_ids_buf = input_ids.request();
    auto labels_buf = labels.request();
    auto attention_mask_buf = attention_mask.request();
    auto loss_weights_buf = loss_weights.request();
    
    std::memcpy(input_ids_buf.ptr, result.input_ids.data(), result.input_ids.size() * sizeof(int64_t));
    std::memcpy(labels_buf.ptr, result.labels.data(), result.labels.size() * sizeof(int64_t));
    std::memcpy(attention_mask_buf.ptr, result.attention_mask.data(), result.attention_mask.size() * sizeof(float));
    std::memcpy(loss_weights_buf.ptr, result.loss_weights.data(), result.loss_weights.size() * sizeof(float));
    
    output["input_ids"] = input_ids;
    output["labels"] = labels;
    output["attention_mask"] = attention_mask;
    output["loss_weights"] = loss_weights;
    output["batch_size"] = result.batch_size;
    output["seq_length"] = result.seq_length;
    
    return output;
}

// PyBind11 module definition
PYBIND11_MODULE(_core, m) {
    m.doc() = "LuminaAI Dataset Accelerator - High-performance C++/CUDA backend";
    
    // Version info
    m.attr("__version__") = "1.0.0";
    
#ifdef CUDA_AVAILABLE
    m.attr("cuda_available") = true;
#else
    m.attr("cuda_available") = false;
#endif
    
    // Shuffle functions
    m.def("fast_shuffle", &py_fast_shuffle,
          "Fast in-place shuffle using Fisher-Yates algorithm",
          py::arg("arr"), py::arg("seed") = 42);
    
    m.def("parallel_shuffle", &py_parallel_shuffle,
          "Parallel shuffle for large arrays (uses CUDA if available)",
          py::arg("arr"), py::arg("seed") = 42);
    
    // Chunking functions
    m.def("fast_chunk_documents", &py_fast_chunk_documents,
          "Fast document chunking for base training",
          py::arg("texts"), py::arg("seq_length"), py::arg("overlap") = true);
    
    // Batch preparation
    m.def("prepare_batch", &py_prepare_batch,
          "Prepare training batch with pre-allocation",
          py::arg("chunks"), py::arg("indices"), py::arg("seq_length"));
    
    // FastFileReader class
    py::class_<dataset_accelerator::FastFileReader>(m, "FastFileReader")
        .def(py::init<const std::string&, size_t>(),
             py::arg("filename"), py::arg("buffer_size") = 1024 * 1024)
        .def("read_lines", &dataset_accelerator::FastFileReader::read_lines,
             "Read lines from file",
             py::arg("max_lines") = 0)
        .def("read_lines_parallel", &dataset_accelerator::FastFileReader::read_lines_parallel,
             "Read lines from file using parallel I/O",
             py::arg("max_lines") = 0);
    
    // StreamingIterator class
    py::class_<dataset_accelerator::StreamingIterator>(m, "StreamingIterator")
        .def(py::init<const std::string&, size_t, size_t>(),
             py::arg("filename"), py::arg("seq_length"), py::arg("buffer_size") = 10000)
        .def("has_next", &dataset_accelerator::StreamingIterator::has_next,
             "Check if more chunks are available")
        .def("next_chunk", &dataset_accelerator::StreamingIterator::next_chunk,
             "Get next chunk from stream");
    
    // ChunkResult struct
    py::class_<dataset_accelerator::ChunkResult>(m, "ChunkResult")
        .def_readonly("chunks", &dataset_accelerator::ChunkResult::chunks)
        .def_readonly("total_tokens", &dataset_accelerator::ChunkResult::total_tokens)
        .def_readonly("documents_processed", &dataset_accelerator::ChunkResult::documents_processed);
}