# LuminaAI

A production-ready machine learning framework for training large-scale conversational transformers with Mixture of Experts (MoE) architecture, intelligent data sharding, and dynamic precision optimization.

[![License](https://img.shields.io/badge/license-Custom-blue.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-brightgreen.svg)]()
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Flash Attention](https://img.shields.io/badge/flash_attention-2.0+-yellow.svg)]()

<p align="center">
  <img src="assets/logo.png" alt="LuminaAI Logo" width="200"/>
</p>

## 🚀 Key Features

### Production-Ready Training Pipeline
- **Intelligent Data Sharding**: Automatic detection and optimization for datasets of any size (MB to TB)
- **Mixture of Experts (MoE)**: Sparse expert routing with load balancing for efficient scaling
- **Dynamic Precision Management**: Automatic precision selection and multi-precision inference testing
- **Advanced Memory Management**: Gradient checkpointing, streaming datasets, and memory-efficient loading
- **Fault Tolerance**: Automatic recovery, health monitoring, and emergency checkpointing

### Scalable Architecture
- **DeepSeek-Style Transformer**: RoPE, GQA, RMSNorm, and SwiGLU activation
- **Flash Attention Support**: Optimized attention computation for long sequences  
- **Model Compilation**: PyTorch 2.0+ compilation for accelerated training
- **Multi-GPU Ready**: Distributed training capabilities with gradient accumulation

### Intelligent Dataset Handling
- **Automatic Strategy Detection**: Memory, sharded, or streaming based on dataset size
- **Memory-Efficient Loading**: Smart batching and worker allocation
- **OASST Format Support**: Native support for conversational dataset formats
- **Comprehensive Validation**: Dataset integrity checking and quality metrics

## 📊 Model Configurations

### Pre-configured Model Sizes

| Configuration | Hidden Size | Layers | Heads | Parameters | Memory (FP16) | Use Case |
|---------------|-------------|--------|-------|------------|---------------|----------|
| **Debug**     | 128         | 2      | 2     | ~0.5M      | ~0.01GB       | Development/Testing |
| **B1**        | 1024        | 12     | 16    | ~1B        | ~2GB          | Resource-constrained |
| **B7**        | 2048        | 22     | 16    | ~7B        | ~14GB         | General training |
| **B14**       | 2560        | 28     | 20    | ~14B       | ~28GB         | High-performance |
| **B50**       | 5120        | 40     | 40    | ~50B       | ~100GB        | Large-scale research |
| **B100**      | 7168        | 48     | 56    | ~100B      | ~200GB        | Cutting-edge research |
| **B200**      | 8192        | 56     | 64    | ~200B      | ~400GB        | Enterprise production |
| **B300**      | 9216        | 64     | 72    | ~300B      | ~600GB        | Advanced research |

### Specialized Configurations

| Configuration | Parameters | Optimization Focus | Key Features |
|---------------|------------|--------------------|--------------|
| **Speed Optimized (M120)** | ~120M | Maximum throughput | Real-time inference, minimal latency |
| **Memory Optimized (M70)** | ~70M | Minimal VRAM usage | Edge deployment, mobile-friendly |
| **Inference Optimized (B3)** | ~3B | Production serving | Balanced speed/quality, optimized caching |
| **Quality Focused (B6)** | ~6B | Output quality | Enhanced generation, research applications |

## 🛠 Installation

### System Requirements
- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+ (recommended)
- 16GB+ RAM (32GB+ for large models)

### Quick Install
```bash
# Clone repository
git clone https://github.com/MatN23/LuminaAI.git
cd LuminaAI

# Install dependencies
pip install -r requirements.txt

# Optional: Install Flash Attention for performance
pip install flash-attn>=2.0.0

# Optional: Install Weights & Biases for tracking
pip install wandb>=0.15.0
```

## 🚀 Quick Start

### Basic Training
```bash
python Src/Main_Scripts/main.py \
  --config b7 \
  --train-data data/conversations.jsonl \
  --epochs 10 \
  --experiment-name my_model_v1
```

### Advanced Training with Custom Settings
```bash
python Src/Main_Scripts/main.py \
  --config b14 \
  --train-data data/large_dataset.jsonl \
  --eval-data data/eval.jsonl \
  --batch-size 2 \
  --grad-accum 8 \
  --precision mixed_bf16 \
  --inference-precision auto \
  --experiment-name production_model
```

## 📁 Data Format

### Conversational Dataset (JSONL)
```json
{
  "messages": [
    {"role": "user", "content": "What is machine learning?"},
    {"role": "assistant", "content": "Machine learning is a subset of artificial intelligence..."}
  ]
}
```

### Supported Roles
- `user`, `human`, `prompter` - Human input
- `assistant`, `ai`, `bot` - AI responses  
- `system` - System instructions

## 🔧 Configuration Options

### Training Parameters
- `--config`: Model size preset (debug, b1, b7, b14, etc.)
- `--epochs`: Number of training epochs
- `--lr`: Learning rate (auto-optimized per config)
- `--batch-size`: Per-GPU batch size
- `--grad-accum`: Gradient accumulation steps
- `--precision`: Training precision (fp32, fp16, bf16, mixed_bf16, auto)
- `--inference-precision`: Inference precision (auto-detected if not specified)

### Data Processing
- `--shard-size-mb`: Override automatic shard size detection
- `--force-streaming`: Force streaming mode for any dataset size
- `--disable-sharding`: Disable sharding (for testing)
- `--max-conversations`: Limit dataset size for testing

### Monitoring & Debugging
- `--test-generation`: Enable generation testing after training
- `--test-precision`: Test multiple inference precisions
- `--check-environment`: Validate training environment
- `--estimate-time`: Estimate training duration
- `--dry-run`: Test configuration without training

## 📊 Automatic Data Sharding

LuminaAI automatically detects the optimal loading strategy based on dataset size and system resources:

### Sharding Strategies
- **Memory Strategy** (<500MB): Load entire dataset in memory
- **Sharded Strategy** (500MB-10GB): Load dataset in configurable shards
- **Streaming Strategy** (>10GB): Stream data with minimal memory footprint

### System Resource Optimization
```bash
# Automatic detection based on system specs
System Resources:
  RAM: 32.0 GB
  CPU cores: 16
  GPU memory: 80.0 GB
  
Recommended sharding config:
  Shard size: 512 MB
  Workers: 8
  Max memory usage: 22.4 GB
```

## 🎯 Precision Management

### Training Precisions
- **FP32**: Full precision (debugging, small models)
- **FP16**: Half precision (memory efficiency)  
- **BF16**: Brain float (numerical stability)
- **Mixed BF16**: Mixed precision with BF16 (recommended for large models)
- **Auto**: Automatic precision selection based on hardware

### Inference Precisions
- **Dynamic**: Runtime precision switching
- **Auto**: Automatic optimization based on model size and hardware
- **Multi-precision Testing**: Automatic benchmarking across precisions

## 🔍 Monitoring & Logging

### Real-time Metrics
```bash
Epoch 1 | Step 150 | Batch 150/2000 | Loss: 2.456789 | PPL: 11.67 | 
LR: 8.50e-05 | GradNorm: 0.8432 | Tokens/s: 1250 | GPU: 8.2GB/16.0GB | 
Strategy: sharded | Memory: 45.2%
```

### Health Monitoring
- GPU memory tracking
- System memory monitoring
- Training anomaly detection
- Automatic failure recovery
- Emergency checkpointing

### Experiment Tracking
```bash
# Weights & Biases integration
export WANDB_PROJECT="conversational-transformer"
export WANDB_ENTITY="your-username"
```

## 🛡 Data Validation & Processing

### Dataset Validation
```bash
# Comprehensive dataset analysis
python Src/Main_Scripts/main.py --validate-data data/train.jsonl --create-report

# Validation output
Data Validation Results:
  Valid conversations: 50,000
  Success rate: 98.5%
  Average tokens: 245.3
  Loading strategy: sharded
  Quality score: 9.2/10
```

### OASST Data Processing
```bash
# Convert OASST format
python Src/Main_Scripts/main.py --process-oasst input.jsonl output.jsonl --max-conversations 10000
```

## 🚀 Generation Testing

### Multi-Precision Inference Testing
```bash
python Src/Main_Scripts/main.py --test-generation --test-precision --resume checkpoints/best_model.pt

# Automatic testing across precisions
FP32: Response generated in 245ms, Memory: 2.1GB
FP16: Response generated in 180ms, Memory: 1.2GB  
BF16: Response generated in 185ms, Memory: 1.2GB
```

### Generation Parameters
- `temperature`: Randomness control (0.1-2.0)
- `top_p`: Nucleus sampling (0.1-1.0) 
- `top_k`: Vocabulary limiting (1-100)
- `max_new_tokens`: Response length limit

## 📈 Performance Optimization

### Memory Efficiency Features
- **Gradient Checkpointing**: Enabled by default for all configs
- **Flash Attention**: Automatic detection and usage for long sequences
- **Model Compilation**: PyTorch 2.0+ compilation for 20%+ speedup
- **Streaming Datasets**: Handle TB-scale datasets with constant memory

### MoE Optimization
- **Load Balancing**: Automatic expert utilization balancing
- **Capacity Factor**: Dynamic routing with overflow handling
- **Expert Pruning**: Unused expert detection and optimization

## 🔧 Environment Validation

### System Compatibility Check
```bash
python Src/Main_Scripts/main.py --check-environment

# Validation output
Environment Validation:
✓ PyTorch version: 2.1.0
✓ CUDA available: 11.8
✓ Flash Attention: Available
✓ Memory: 64GB available
✓ GPU Memory: 80GB available
⚠ Warning: Large model selected for available VRAM
```

## 📊 Training Time Estimation

```bash
python Src/Main_Scripts/main.py --estimate-time --config b7

# Estimation output
Training Time Estimates:
  Dataset size: 50,000 conversations
  Dataset strategy: sharded
  Estimated time: 12.5 hours (0.5 days)
  Total tokens: 12,250,000
  Throughput: 1,200 tokens/sec
  Memory utilization: 67%
  Note: Sharding mode will minimize memory usage
```

## 🗂 Project Structure

```
LuminaAI/
├── Src/
│   └── Main_Scripts/
│       ├── main.py                    # Main entry point with sharding support
│       ├── chat.py
        ├── core/
│       │   ├── model.py              # DeepSeek-style Transformer with MoE
│       │   ├── tokenizer.py          # GPT-4 compatible tokenization
│       │   └── dataset.py            # Sharded dataset loading
│       ├── training/
│       │   ├── trainer.py            # Advanced trainer with precision management
│       │   ├── orchestrator.py       # Training orchestration
│       │   └── checkpoint.py         # Checkpoint management
│       ├── config/
│       │   └── config_manager.py     # Configuration presets
│       ├── monitoring/
│       │   └── logger.py             # Advanced logging and monitoring
│       └── utils/
│           ├── data_processing.py    # Data utilities and validation
│           ├── environment.py        # System validation
│           └── reporting.py          # Performance reporting
├── data/
│   └── shards/                       # Automatic dataset sharding
├── checkpoints/                      # Model checkpoints
├── experiments/                      # Experiment tracking
├── logs/                            # Training logs
└── backups/                         # Emergency backups
```

## 🐛 Troubleshooting

### Memory Issues
```bash
# For OOM errors
python Src/Main_Scripts/main.py --batch-size 1 --grad-accum 16 --precision fp16

# Enable aggressive memory optimization
python Src/Main_Scripts/main.py --force-streaming --precision fp16
```

### Performance Optimization
```bash
# Enable all optimizations
python Src/Main_Scripts/main.py --precision mixed_bf16 --compile --enable-flash-attention

# Test system performance
python Src/Main_Scripts/main.py --test-sharding
```

### Data Issues
```bash
# Comprehensive data analysis
python Src/Main_Scripts/main.py --validate-data data/train.jsonl --create-report

# Clean problematic data
python Src/Main_Scripts/main.py --process-oasst dirty_data.jsonl clean_data.jsonl
```

## 🏆 Advanced Features

### Intelligent Resource Management
- Automatic GPU memory optimization
- Dynamic worker allocation based on system resources
- Smart batch size adjustment for optimal throughput
- Memory-mapped file I/O for large datasets

### Production Readiness
- Comprehensive error handling and recovery
- Health monitoring with automatic alerts
- Periodic backup creation
- Experiment reproducibility with seed management

### Research Features
- Multi-precision inference benchmarking
- Expert utilization analysis for MoE models
- Token-level loss analysis and reporting
- Advanced generation quality metrics

## 🤝 Contributing

Contributions are welcome! Please ensure:
1. Code follows the existing architecture patterns
2. New features include comprehensive testing
3. Documentation is updated for user-facing changes
4. Performance impact is considered and documented

## 📄 License

This project is licensed under a Custom License. See license headers in source files for details.

## 🆘 Support

For technical issues:
1. Check the troubleshooting section
2. Review system requirements and environment validation
3. Run diagnostic commands to identify the issue
4. Check existing GitHub issues
5. Create a detailed issue with logs, configuration, and system information

---

*LuminaAI: Pushing the boundaries of conversational AI with production-ready scaling and optimization.*