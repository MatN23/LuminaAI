# Modern Transformer Training System

A state-of-the-art transformer training framework with cutting-edge architectural improvements and production-ready features.

## 🚀 Features

### **Advanced Architecture**
- **RoPE (Rotary Position Embedding)** - Superior positional encoding
- **RMSNorm** - More stable normalization than LayerNorm  
- **Grouped Query Attention (GQA)** - Efficient attention with reduced memory
- **SwiGLU/GeGLU Activations** - Modern GLU variants for better performance
- **Flash Attention** - Memory-efficient attention computation
- **Gradient Checkpointing** - Trade computation for memory

### **Modern Training Features**
- **Mixed Precision Training** - FP16/BF16 with automatic precision detection
- **Torch Compile** - JIT compilation for faster training
- **Advanced Optimizers** - AdamW with configurable parameters
- **Smart Scheduling** - Cosine with warmup, cosine restarts
- **Gradient Accumulation** - Handle large effective batch sizes
- **Dynamic Loss Scaling** - Stable FP16 training

### **Production-Ready Components**
- **Advanced BPE Tokenizer** - Fast, robust subword tokenization
- **Comprehensive Model Management** - Save/load with metadata
- **Conversation Formatting** - Built-in chat format support
- **Hardware Auto-Detection** - Optimal config selection
- **Progress Tracking** - Detailed logging and optional Wandb integration

## 🛠️ Installation

### **Requirements**
- Python 3.8+
- PyTorch 2.1.0+
- CUDA 11.8+ (for GPU training)

### **Quick Setup**
```bash
# Install dependencies
pip install -r requirements.txt

# Optional: Install Flash Attention for maximum performance
pip install flash-attn --no-build-isolation
```

### **Verify Installation**
```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

# Check if Flash Attention is available
try:
    from flash_attn import flash_attn_func
    print("Flash Attention: Available ✅")
except ImportError:
    print("Flash Attention: Not available (optional)")
```

## 📊 Data Format

The system supports multiple data formats:

### **JSONL Format (Recommended)**
```json
{"text": "Your training text here"}
{"text": "Another training example"}
```

### **Conversation Format**
```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is machine learning?"},
    {"role": "assistant", "content": "Machine learning is..."}
  ]
}
```

### **Simple Text Fields**
The system automatically detects these fields:
- `text`, `content`, `message`, `body`, `output`
- `messages` (for conversation format)

## 🔧 Configuration

All configuration is done through the hardcoded `TRAINING_CONFIG` dictionary in `Train.py`. Here are the key sections:

### **Data Configuration**
```python
"data": {
    "training_data_path": "path/to/your/data.jsonl",
    "use_conversation_format": True,
    "max_samples_train": None,  # None for all data
    "eval_split": 0.1,
    "min_text_length": 10,
    "tokenizer_train_size": 100000,
}
```

### **Model Configuration**
```python
"model": {
    "config_preset": "auto",  # "auto", "tiny", "research", "custom"
    "custom": {
        "vocab_size": 32000,
        "hidden_size": 2048,
        "num_layers": 24,
        "num_heads": 16,
        "seq_length": 4096,
        # ... more options
    }
}
```

### **Training Configuration**
```python
"training": {
    "batch_size": 4,
    "gradient_accumulation_steps": 8,
    "max_epochs": 3,
    "learning_rate": 3e-4,
    "scheduler_type": "cosine_with_warmup",
    "warmup_ratio": 0.03,
    # ... more options
}
```

## 🏃‍♂️ Quick Start

### **1. Prepare Your Data**
Place your training data in JSONL format:
```bash
# Example data structure
echo '{"text": "The quick brown fox jumps over the lazy dog."}' > train_data.jsonl
echo '{"text": "Machine learning is a subset of artificial intelligence."}' >> train_data.jsonl
```

### **2. Configure Training**
Edit the `TRAINING_CONFIG` in `Train.py`:
```python
TRAINING_CONFIG = {
    "data": {
        "training_data_path": "train_data.jsonl",
        # ... other settings
    },
    # ... other configurations
}
```

### **3. Start Training**
```bash
python Train.py
```

The system will:
- Auto-detect optimal configuration for your hardware
- Train a BPE tokenizer on your data
- Initialize the model with modern architecture
- Start training with advanced features
- Save checkpoints automatically
- Log comprehensive metrics

## 📈 Model Presets

### **Auto Detection** (Recommended)
```python
"config_preset": "auto"
```
Automatically selects optimal configuration based on available GPU memory:
- **40GB+ (A100/H100)**: Research-grade 7B model
- **20GB+ (RTX 4090)**: Medium model (~1.5B parameters)
- **10GB+ (RTX 3080)**: Small model (~350M parameters)
- **<10GB**: Tiny model for testing

### **Predefined Presets**
```python
# For debugging and testing
"config_preset": "tiny"

# For research (requires high-end GPU)
"config_preset": "research" 

# For full control
"config_preset": "custom"
```

## 💾 Model Management

The system includes sophisticated model management:

### **Automatic Saving**
- Checkpoints saved periodically during training
- Comprehensive metadata tracking
- Model versioning and integrity checks
- Human-readable model information

### **Loading Models**
```python
from model_manager import ModelManager

manager = ModelManager("models")
model, tokenizer, metadata, optimizer, scheduler = manager.load_model("model_id")
```

### **List Saved Models**
```python
models = manager.list_models()
for model in models:
    print(f"{model['name']}: {model['parameters']:,} parameters, {model['size_mb']:.1f}MB")
```

## 🔬 Advanced Features

### **Mixed Precision Training**
Automatically detected based on GPU capability:
- **Ampere+ (RTX 30xx, A100)**: BF16 (recommended)
- **Volta/Turing (V100, RTX 20xx)**: FP16
- **Older GPUs**: FP32 fallback

### **Flash Attention**
Automatically used when available for:
- 2-8x faster attention computation
- Significantly reduced memory usage
- Better scaling to long sequences

### **Torch Compile**
JIT compilation for up to 30% speed improvements:
```python
"precision": {
    "use_compile": True,
    "compile_mode": "default"  # or "max-autotune"
}
```

### **Gradient Checkpointing**
Trade computation for memory:
- Enables training larger models
- Automatically enabled for large configurations
- Minimal speed impact with major memory savings

## 📋 Hardware Requirements

### **Minimum Requirements**
- **GPU**: GTX 3070 ti 6GB or better
- **RAM**: 16GB system RAM
- **Storage**: 10GB+ free space

### **Recommended Requirements**
- **GPU**: RTX 4090/5080 or better (12GB+ VRAM)
- **RAM**: 32GB+ system RAM
- **Storage**: NVMe SSD with 50GB+ free space

### **Optimal Setup**
- **GPU**: 2x RTX 5090, A6000, A100, or H100
- **RAM**: 64GB+ system RAM
- **Storage**: High-speed NVMe SSD

## 📊 Memory Usage Examples

| Model Size | Parameters | VRAM (Training) | Batch Size | Sequence Length |
|------------|------------|-----------------|------------|-----------------|
| Tiny       | 25M        | 2-3GB          | 8          | 1024           |
| Small      | 350M       | 8-10GB         | 4          | 2048           |
| Medium     | 1.5B       | 18-22GB        | 2          | 4096           |
| Large      | 7B         | 40-48GB        | 1          | 8192           |

*Note: Memory usage varies based on sequence length, precision, and other factors.*

## 🐛 Troubleshooting

### **Out of Memory Errors**
```bash
# Reduce batch size
"batch_size": 2

# Increase gradient accumulation
"gradient_accumulation_steps": 16

# Enable gradient checkpointing
"gradient_checkpointing": True

# Use smaller model
"config_preset": "tiny"
```

### **Slow Training**
```bash
# Enable mixed precision
"use_mixed_precision": True

# Install Flash Attention
pip install flash-attn

# Enable torch compile
"use_compile": True

# Use more workers
"num_workers": 8
```

### **Tokenizer Issues**
```bash
# Increase vocab size for complex text
"vocab_size": 50000

# Adjust minimum frequency
"min_frequency": 1

# Increase training size
"tokenizer_train_size": 200000
```

## 📈 Performance Tips

### **Maximize Throughput**
1. **Use largest batch size** that fits in memory
2. **Enable mixed precision** (BF16 preferred)
3. **Install Flash Attention** for long sequences
4. **Use torch compile** for additional speedup
5. **Optimize data loading** with multiple workers

### **Memory Optimization**
1. **Enable gradient checkpointing**
2. **Use smaller sequence lengths** during initial training
3. **Progressive training** (start small, increase size)
4. **CPU offloading** for very large models

### **Training Stability**
1. **Use warmup** for learning rate
2. **Clip gradients** (default: 1.0)
3. **Monitor loss scaling** for FP16
4. **Regular checkpointing**

## 🔬 Experimental Features

### **DeepSpeed Integration** (Experimental)
```python
# Enable in requirements.txt
# pip install deepspeed

# The system detects and can use DeepSpeed automatically
```

### **Wandb Integration**
```python
"experiment": {
    "wandb_project": "your-project-name"
}
```

## 📚 Architecture Details

### **Modern Improvements Over Standard Transformers**
- **RoPE**: Better positional understanding, especially for longer sequences
- **RMSNorm**: More stable and faster than LayerNorm
- **GQA**: Reduces memory and compute while maintaining quality
- **SwiGLU**: Proven superior activation function for language models
- **Proper Initialization**: Scaled initialization for deep networks

### **Memory Optimizations**
- **Flash Attention**: O(1) memory complexity for attention
- **Gradient Checkpointing**: Trade compute for memory
- **Mixed Precision**: Halve memory usage with minimal quality loss
- **KV Caching**: Efficient generation with past key-value caching

## 🤝 Contributing

This is a research and educational project. Contributions are welcome!

### **Areas for Contribution**
- Additional model architectures
- More efficient training techniques
- Better data processing pipelines
- Enhanced monitoring and visualization
- Documentation improvements

## 📄 License

Copyright (c) 2025 Matias Nielsen. All rights reserved.

## 🙏 Acknowledgments

This project incorporates ideas and techniques from:
- **Attention Is All You Need** (Transformer architecture)
- **RoFormer** (Rotary Position Embedding)
- **PaLM** (Parallel layers and modern scaling)
- **LLaMA** (Modern architectural choices)
- **GPT-4 Technical Report** (Training best practices)

## 📞 Support

For questions and issues:
1. Check the troubleshooting section above
2. Review the comprehensive logging output
3. Experiment with different configuration presets
4. Consider the hardware requirements for your setup

---

**Happy Training! 🚀**