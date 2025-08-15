# LuminaAI: Production-Ready Conversational Transformer Training System

A comprehensive, enterprise-grade system for training conversational transformers with advanced monitoring, fault tolerance, and production-ready features. Built for reliability, scalability, and ease of use.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: Custom](https://img.shields.io/badge/License-Custom-red.svg)](LICENSE)

## 🌟 Key Features

### 🏗️ **Production Architecture**
- **Modular Design**: Clean separation of concerns for easy debugging and maintenance
- **Enterprise Monitoring**: Multi-backend logging (File, Wandb, TensorBoard)
- **Fault Tolerance**: Automatic recovery, health monitoring, and graceful error handling
- **Scalable Configuration**: From debug mode to large-scale production training

### 🚀 **Advanced Model Features**
- **Grouped Query Attention (GQA)**: Efficient attention mechanism for better performance
- **RoPE Positional Encoding**: Rotary position embeddings for better sequence modeling
- **SwiGLU Activation**: State-of-the-art activation function for improved training
- **Flash Attention Support**: Optional high-performance attention implementation
- **Mixed Precision Training**: FP16/BF16 support for faster training and lower memory usage

### 📊 **Comprehensive Monitoring**
- **Real-time Health Monitoring**: Detect training anomalies and divergence
- **Multi-backend Logging**: Wandb, TensorBoard, and structured file logging
- **Performance Metrics**: GPU utilization, memory usage, and throughput tracking
- **Automated Reporting**: HTML reports for dataset analysis and training summaries

### 🛡️ **Production Reliability**
- **Automatic Checkpointing**: Never lose training progress
- **Environment Validation**: Pre-training system checks
- **Data Validation**: Comprehensive dataset quality analysis
- **Graceful Error Recovery**: Robust error handling and recovery mechanisms

## 📁 Project Architecture

```
LuminaAI/
├── Main.py                     # Simple entry point with configuration
├── Setup.py                    # Environment setup and validation
├── config/
│   └── config_manager.py       # Advanced configuration management
├── core/
│   ├── tokenizer.py           # Production tokenizer with conversation formatting
│   ├── model.py               # Transformer architecture with modern features
│   └── dataset.py             # Robust dataset handling with validation
├── training/
│   ├── orchestrator.py        # Training coordination and management
│   ├── trainer.py             # Main training loop with fault tolerance
│   ├── training_loop.py       # Core training implementation
│   └── checkpoint.py          # Checkpoint management and recovery
├── monitoring/
│   └── logger.py              # Production logging and health monitoring
├── utils/
│   ├── data_processing.py     # Data validation and processing utilities
│   ├── environment.py         # System validation and optimization
│   └── reporting.py           # Automated report generation
└── requirements.txt           # Production dependencies
```

## 🚀 Quick Start

### 1. **Setup Environment**
```bash
# Clone the repository
git clone https://github.com/MatN23/LuminaAI.git
cd LuminaAI

# Install dependencies
pip install -r requirements.txt

# Validate and setup environment
python Setup.py
```

### 2. **First Training Run (Debug Mode)**
```bash
# Start with debug mode to validate everything works
python Main.py --config debug --test-generation
```

### 3. **Production Training**
```bash
# Medium-scale production training
python Main.py --config medium --epochs 10 --experiment-name my_production_run
```

## ⚙️ Configuration Presets

| Preset | Parameters | Memory | Use Case |
|--------|------------|---------|----------|
| `debug` | ~6M | ~2GB | Testing & debugging |
| `small` | ~50M | ~8GB | Limited resources |
| `medium` | ~400M | ~16GB | Serious training |
| `large` | ~1.2B | ~32GB+ | Production scale |

*Approximate times vary based on dataset size and hardware

## 🎯 Easy Configuration

The main configuration is done in `Main.py` - just modify these variables:

```python
# EASY CONFIGURATION SECTION
config_preset = 'medium'         # 'debug', 'small', 'medium', 'large'
train_data_path = 'data/train.jsonl'
eval_data_path = 'data/eval.jsonl'
epochs_override = 10             # Override default epochs
test_generation = True           # Test model generation after training
```

## 📊 Data Format

LuminaAI expects JSONL files with this conversation format:

```json
{
  "conversation_id": "conv_001",
  "messages": [
    {"role": "user", "content": "What is machine learning?"},
    {"role": "assistant", "content": "Machine learning is a subset of AI that enables computers to learn from data without explicit programming."}
  ],
  "metadata": {
    "source": "dataset_name",
    "quality_score": 0.95
  }
}
```

**Supported roles**: `user`, `assistant`, `system`, `prompter` (auto-converted to `user`)

## 🔧 Advanced Usage

### **Custom Training Configuration**
```bash
# Fine-tune hyperparameters
python Main.py --config medium \
  --lr 1e-4 \
  --batch-size 4 \
  --grad-accum 8 \
  --precision bf16 \
  --compile
```

### **Resume from Checkpoint**
```bash
# Automatic resume (finds latest checkpoint)
python Main.py --config medium --auto-resume

# Resume from specific checkpoint
python Main.py --config medium --resume checkpoints/model_epoch_005.pt
```

### **Data Processing Pipeline**
```bash
# Process OASST1 data
python Main.py --process-oasst raw_oasst.jsonl processed_train.jsonl

# Comprehensive data validation
python Main.py --validate-data data/train.jsonl --create-report

# Generate data quality report
python Main.py --data-report data/train.jsonl data/eval.jsonl
```

### **Environment Optimization**
```bash
# Check system capabilities
python Main.py --check-environment --estimate-time

# Optimize for your hardware
python Main.py --config medium --auto-optimize
```

## 📈 Monitoring & Observability

### **Multi-Backend Logging**
- **File Logs**: Structured logs in `logs/{experiment}/`
- **Metrics**: JSONL metrics for analysis
- **Wandb**: Real-time experiment tracking (optional)
- **TensorBoard**: Local visualization (optional)

### **Health Monitoring**
- Automatic detection of training divergence
- NaN/Inf gradient monitoring
- Memory usage tracking
- Performance bottleneck identification

### **Automated Reports**
```bash
# Generate comprehensive training report
python Main.py --generate-report experiments/my_experiment

# Dataset quality analysis
python Main.py --analyze-data data/train.jsonl
```

## 🛡️ Fault Tolerance Features

- **Automatic Checkpointing**: Save every N steps/epochs
- **Health Monitoring**: Detect and recover from training issues
- **Graceful Shutdown**: Clean interruption handling (Ctrl+C)
- **Memory Management**: Automatic garbage collection and optimization
- **Error Recovery**: Robust error handling with detailed logging

## 🔍 Debugging & Troubleshooting

### **Debug Mode**
Always start with debug mode for new setups:
```bash
python Main.py --config debug --test-generation --verbose
```

### **Common Issues & Solutions**

| Issue | Solution |
|-------|----------|
| **CUDA Out of Memory** | Reduce `batch_size`, enable `gradient_checkpointing` |
| **Training Divergence** | Lower learning rate, check data quality |
| **Slow Training** | Enable `--compile`, use mixed precision |
| **Data Loading Errors** | Run `--validate-data` first |
| **Import Errors** | Run `python Setup.py` to check dependencies |

### **Performance Optimization**
```bash
# Enable all optimizations
python Main.py --config medium \
  --precision bf16 \
  --compile \
  --gradient-checkpointing \
  --flash-attention
```

## 🎛️ Configuration Options

### **Model Architecture**
- `hidden_size`: Model dimension (512, 1024, 2048, 4096)
- `num_layers`: Transformer layers (8, 16, 24, 32)
- `num_heads`: Attention heads (8, 16, 32)
- `seq_length`: Maximum sequence length (1024, 2048, 4096)

### **Training Parameters**
- `learning_rate`: Learning rate (1e-5 to 1e-3)
- `batch_size`: Per-device batch size
- `gradient_accumulation_steps`: Effective batch size multiplier
- `precision`: fp32, fp16, bf16
- `lr_scheduler`: cosine, linear, onecycle

### **Production Settings**
- `experiment_name`: Unique experiment identifier
- `save_every_n_batches`: Checkpoint frequency
- `eval_every_n_batches`: Evaluation frequency
- `early_stopping_patience`: Early stopping threshold

## 📊 Benchmarks & Performance

### **Training Speed** (tokens/sec on various hardware)
- **RTX 4090**: ~45,000 tokens/sec (medium config, bf16)
- **A100 40GB**: ~65,000 tokens/sec (large config, bf16)
- **V100 32GB**: ~35,000 tokens/sec (medium config, fp16)
- **RTX 3080**: ~25,000 tokens/sec (small config, fp16)

### **Memory Requirements**
- **Debug**: 2GB VRAM (any modern GPU)
- **Small**: 8GB VRAM (RTX 3080, RTX 4070)
- **Medium**: 16GB VRAM (RTX 4090, A5000)
- **Large**: 32GB+ VRAM (A100, H100)

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Follow the modular architecture patterns
4. Add comprehensive error handling
5. Include tests for new functionality
6. Submit a pull request

### **Development Setup**
```bash
# Development installation
git clone https://github.com/MatN23/LuminaAI.git
cd LuminaAI
pip install -r requirements.txt
python Setup.py

# Run tests
python -m pytest tests/

# Check code quality
python Main.py --config debug --test-all
```

## 📄 License

This project is licensed under a Custom License. See [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAI** for transformer architecture insights and GPT innovations
- **Hugging Face** for tokenizer implementations and community tools
- **PyTorch Team** for the excellent deep learning framework
- **Open Assistant** for high-quality conversation datasets
- **Anthropic** for constitutional AI and safety research
- **Research Community** for advances in attention mechanisms and model architecture

---

**Built with ❤️ for the AI research community**

*LuminaAI - Illuminating the path to better conversational AI*