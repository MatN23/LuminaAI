# LuminaAI

A PyTorch-based framework for training large language models with Mixture of Experts (MoE) architecture support. Implements transformer models ranging from 70M to 300B parameters with automatic dataset processing and distributed training capabilities.

[![License](https://img.shields.io/badge/license-Custom-blue.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-brightgreen.svg)]()
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

## Architecture Overview

LuminaAI implements a DeepSeek-style transformer architecture with several key technical components:

- **RoPE Positional Encoding**: Rotary Position Embedding with configurable theta parameter
- **Grouped Query Attention (GQA)**: Memory-efficient attention with configurable key-value head ratios
- **RMSNorm**: Root Mean Square Layer Normalization for improved training stability
- **SwiGLU Activation**: Swish-Gated Linear Units in feed-forward layers
- **Flash Attention Integration**: Optional memory-efficient attention implementation for long sequences
- **Mixture of Experts (MoE)**: Sparse routing with configurable expert count and load balancing

The framework supports models from 70M parameters (suitable for edge deployment) to 300B parameters (research-scale), with automatic memory management and precision optimization.

## Model Configurations

### Production Configurations

| Name | Parameters | Hidden Size | Layers | Heads | KV Heads | Context | Memory (BF16) | Use Case |
|------|------------|-------------|--------|-------|----------|---------|---------------|----------|
| `debug` | 500K | 128 | 2 | 2 | 1 | 256 | ~10MB | Testing/debugging |
| `b1` | 1B | 1024 | 12 | 16 | 4 | 2048 | ~2GB | Resource-limited training |
| `b7` | 7B | 2048 | 22 | 16 | 8 | 4096 | ~14GB | General purpose training |
| `b14` | 14B | 2560 | 28 | 20 | 10 | 4096 | ~28GB | High-performance applications |
| `b50` | 50B | 5120 | 40 | 40 | 10 | 4096 | ~100GB | Large-scale research |
| `b100` | 100B | 7168 | 48 | 56 | 14 | 8192 | ~200GB | Advanced research |
| `b200` | 200B | 8192 | 56 | 64 | 16 | 8192 | ~400GB | Enterprise-scale |
| `b300` | 300B | 9216 | 64 | 72 | 18 | 8192 | ~600GB | Research frontiers |

### Specialized Configurations

| Name | Parameters | Focus | Key Features |
|------|------------|-------|--------------|
| `m120_speed` | 120M | Low latency | Optimized for inference throughput |
| `m70_memory` | 70M | Memory efficiency | Minimal VRAM requirements |
| `b3_inference` | 3B | Production serving | Balanced quality/performance |
| `b6_quality` | 6B | Output quality | Enhanced generation capabilities |

All configurations include MoE support with expert counts ranging from 4 (debug) to 144 (b300), using top-k routing with k=2 by default.

## Data Processing System

### Dataset Format

The framework processes conversational data in JSONL format with message-based structure:

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain quantum computing."},
    {"role": "assistant", "content": "Quantum computing leverages quantum mechanical phenomena..."}
  ]
}
{
  "messages": [
    {"role": "human", "content": "What is machine learning?"},
    {"role": "assistant", "content": "Machine learning is a method of data analysis..."}
  ]
}
```

**Supported Role Mappings:**
- Input roles: `user`, `human`, `prompter`
- Response roles: `assistant`, `ai`, `bot`
- System roles: `system`

### Automatic Sharding Strategy

The data loading system automatically selects processing strategy based on dataset characteristics:

| Dataset Size | Strategy | Memory Usage | Description |
|--------------|----------|--------------|-------------|
| < 500MB | Memory | Full dataset in RAM | Complete loading for small datasets |
| 500MB - 10GB | Sharded | Configurable chunks | Balanced memory/performance |
| > 10GB | Streaming | Minimal memory | Constant memory usage |

**Shard Configuration:**
- Default shard size: Automatically calculated based on available RAM
- Configurable shard size: 64MB to 2GB per shard
- Worker allocation: Based on CPU core count and memory constraints
- Overlap handling: Automatic conversation boundary detection

### Data Validation and Quality Metrics

The framework includes comprehensive data validation:

```bash
python Src/Main_Scripts/main.py --validate-data data/train.jsonl --create-report
```

**Validation Checks:**
- JSON format integrity
- Message structure validation
- Role consistency verification
- Token count distribution analysis
- Conversation length statistics
- Character encoding validation

## Training Infrastructure

### Precision Support

| Precision | Memory Usage | Stability | Performance | Recommended Use |
|-----------|--------------|-----------|-------------|-----------------|
| `fp32` | High | Excellent | Slow | Debugging, small models |
| `fp16` | 50% of fp32 | Good | Fast | Production training |
| `bf16` | 50% of fp32 | Excellent | Fast | Large models, A100+ GPUs |
| `mixed_bf16` | Optimized | Excellent | Fastest | Production (recommended) |
| `auto` | Variable | Good | Variable | Automatic hardware selection |

**Automatic Precision Selection Logic:**
- Detects GPU architecture (V100, A100, H100)
- Evaluates model size relative to available VRAM
- Selects optimal precision for stability/performance balance
- Falls back to safer precisions on numerical instability

### Memory Management

**Gradient Checkpointing:** Enabled by default across all configurations. Trades computation for memory by recomputing activations during backward pass.

**Flash Attention:** Automatically enabled for:
- Sequence lengths > 512 tokens
- Supported hardware (A100, H100)
- FP16/BF16 precision modes

**Memory Optimization Features:**
- Automatic batch size adjustment based on available VRAM
- Dynamic shard size calculation
- Gradient accumulation for large effective batch sizes
- Emergency memory recovery on OOM errors

### Fault Tolerance and Recovery

**Checkpoint Management:**
- Automatic checkpointing at configurable intervals
- Best model tracking based on evaluation metrics
- Emergency checkpoint creation on training interruption
- Checkpoint validation and compatibility checking

**Health Monitoring:**
- GPU memory utilization tracking
- Training loss anomaly detection
- Gradient norm monitoring
- System resource utilization

**Recovery Mechanisms:**
- Automatic resumption from last checkpoint
- Configurable retry attempts with backoff
- Emergency checkpoint creation on failures
- Graceful handling of hardware failures

## Installation and Requirements

### System Requirements

**Minimum Requirements:**
- Python 3.10 or higher
- PyTorch 2.0+ with CUDA support
- 16GB system RAM
- CUDA-compatible GPU with 8GB+ VRAM

**Recommended Requirements:**
- Python 3.11
- PyTorch 2.1+ with CUDA 11.8+
- 32GB+ system RAM
- A100/H100 GPU with 40GB+ VRAM for large models

### Installation Process

1. **Clone Repository:**
```bash
git clone https://github.com/MatN23/LuminaAI.git
cd LuminaAI
```

2. **Install Base Dependencies:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets tokenizers numpy pyyaml tqdm psutil
```

3. **Install Optional Performance Dependencies:**
```bash
# Flash Attention (requires compatible hardware)
pip install flash-attn>=2.0.0 --no-build-isolation

# Monitoring and logging
pip install wandb tensorboard matplotlib seaborn

# Development tools
pip install pytest black isort flake8
```

### Environment Verification

```bash
python Src/Main_Scripts/main.py --check-environment
```

This command validates:
- PyTorch installation and CUDA compatibility
- Available system and GPU memory
- Optional dependency availability
- Hardware capability assessment

## Usage Examples

### Basic Training Command

```bash
python Src/Main_Scripts/main.py \
  --config b7 \
  --train-data data/conversations.jsonl \
  --epochs 3 \
  --experiment-name my_experiment
```

### Production Training Configuration

```bash
python Src/Main_Scripts/main.py \
  --config b14 \
  --train-data data/large_dataset.jsonl \
  --eval-data data/validation.jsonl \
  --batch-size 2 \
  --grad-accum 16 \
  --precision mixed_bf16 \
  --inference-precision auto \
  --learning-rate 1e-4 \
  --warmup-ratio 0.1 \
  --save-every-n-batches 1000 \
  --eval-every-n-batches 500 \
  --experiment-name production_model_v1
```

### Resource-Constrained Training

```bash
python Src/Main_Scripts/main.py \
  --config m70_memory \
  --train-data data/small_dataset.jsonl \
  --batch-size 1 \
  --grad-accum 32 \
  --precision fp16 \
  --force-streaming \
  --max-conversations 10000 \
  --experiment-name edge_model
```

## Configuration System

### Training Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--config` | str | required | Model configuration preset |
| `--learning-rate` | float | config-dependent | Learning rate (auto-optimized per config) |
| `--batch-size` | int | config-dependent | Per-GPU batch size |
| `--grad-accum` | int | config-dependent | Gradient accumulation steps |
| `--epochs` | int | 3 | Number of training epochs |
| `--warmup-ratio` | float | 0.15 | Learning rate warmup ratio |
| `--weight-decay` | float | 0.01 | Weight decay coefficient |
| `--max-grad-norm` | float | 1.0 | Gradient clipping threshold |

### Data Processing Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--shard-size-mb` | int | auto | Override automatic shard size |
| `--num-workers` | int | auto | Data loading worker count |
| `--force-streaming` | bool | False | Force streaming mode |
| `--disable-sharding` | bool | False | Disable automatic sharding |
| `--max-conversations` | int | unlimited | Limit dataset size |

### Model and Precision Parameters

| Parameter | Type | Options | Description |
|-----------|------|---------|-------------|
| `--precision` | str | fp32, fp16, bf16, mixed_bf16, auto | Training precision |
| `--inference-precision` | str | fp32, fp16, bf16, auto, dynamic | Inference precision |
| `--compile` | bool | auto | Enable PyTorch compilation |
| `--enable-flash-attention` | bool | auto | Force Flash Attention usage |

## Monitoring and Analysis

### Training Progress Monitoring

During training, the framework provides real-time metrics:

```
Epoch 1 | Step 150 | Batch 150/2000 
Loss: 2.456789 | PPL: 11.67 | LR: 8.50e-05 
GradNorm: 0.8432 | Tokens/s: 1250 
GPU: 8.2GB/16.0GB | Memory: 45.2%
Strategy: sharded | Workers: 4
```

**Key Metrics Explained:**
- **Loss**: Cross-entropy loss (lower is better)
- **PPL**: Perplexity (exp(loss), lower indicates better language modeling)
- **LR**: Current learning rate (follows warmup/decay schedule)
- **GradNorm**: Gradient norm (used for clipping and stability monitoring)
- **Tokens/s**: Training throughput
- **GPU Memory**: Current/total GPU memory usage
- **Strategy**: Current data loading strategy

### Model Analysis Tools

**Parameter Estimation:**
```bash
python Src/Main_Scripts/main.py --estimate-parameters --config b7
```

**Memory Footprint Analysis:**
```bash
python Src/Main_Scripts/main.py --analyze-memory --config b14
```

**Training Time Estimation:**
```bash
python Src/Main_Scripts/main.py --estimate-time --config b7 --train-data data/large.jsonl
```

### Generation Testing

The framework includes comprehensive generation testing capabilities:

```bash
python Src/Main_Scripts/main.py \
  --test-generation \
  --test-precision \
  --resume checkpoints/best_model.pt \
  --temperature 0.8 \
  --top-p 0.9 \
  --max-new-tokens 256
```

**Multi-Precision Benchmarking:**
```
Testing generation across precisions:
FP32: 245ms, Memory: 2.1GB, Quality: Baseline
FP16: 180ms, Memory: 1.2GB, Quality: -0.02% vs FP32
BF16: 185ms, Memory: 1.2GB, Quality: +0.01% vs FP32
Auto: Selected BF16 based on hardware optimization
```

## Project Structure and Architecture

```
LuminaAI/
├── Src/
│   └── Main_Scripts/
│       ├── main.py                     # Entry point and CLI interface
│       ├── chat.py                     # Interactive chat interface
│       ├── core/                       # Core model and data components
│       │   ├── model.py               # DeepSeek transformer implementation
│       │   ├── tokenizer.py           # GPT-4 compatible tokenization
│       │   └── dataset.py             # Sharded dataset processing
│       ├── training/                   # Training infrastructure
│       │   ├── trainer.py             # Main training orchestration
│       │   ├── orchestrator.py        # High-level training coordination
│       │   └── checkpoint.py          # Checkpoint management system
│       ├── config/
│       │   └── config_manager.py      # Configuration presets and validation
│       ├── monitoring/
│       │   └── logger.py              # Structured logging and metrics
│       └── utils/
│           ├── data_processing.py     # Data validation and processing utilities
│           ├── environment.py         # System validation and optimization
│           └── reporting.py           # Performance analysis and reporting
├── data/
│   └── shards/                        # Automatic dataset sharding output
├── checkpoints/                       # Model checkpoints
├── experiments/                       # Experiment tracking and results
├── logs/                             # Training logs and metrics
├── backups/                          # Emergency checkpoint backups
├── requirements.txt                   # Python dependencies
└── README.md                         # This file
```

## Advanced Features

### Mixture of Experts (MoE) Implementation

The MoE system includes several advanced features:

**Expert Routing:**
- Top-k gating with configurable k (default: 2)
- Capacity factor enforcement to prevent expert overload
- Load balancing loss to encourage uniform expert utilization
- Dropout and capacity overflow handling

**Expert Architecture:**
- Each expert implements SwiGLU activation
- Independent parameter initialization per expert
- Gradient isolation between experts during training
- Sparse activation patterns for computational efficiency

**Load Balancing:**
- Auxiliary loss encourages balanced expert utilization
- Configurable load balancing weight (default: 0.01-0.03 depending on model size)
- Expert usage monitoring and reporting
- Dynamic capacity adjustment based on routing patterns

### Advanced Training Techniques

**Gradient Checkpointing:**
- Enabled by default for all configurations
- Automatic memory vs. computation trade-off
- Selective checkpointing for optimal performance
- Compatible with model compilation

**Learning Rate Scheduling:**
- Cosine annealing with warmup by default
- Linear and OneCycle schedulers available
- Automatic warmup ratio selection based on model size
- Minimum learning rate enforcement

**Numerical Stability:**
- Enhanced RMSNorm implementation with FP32 computation
- Gradient clipping with configurable thresholds
- Loss scaling for mixed precision training
- Automatic precision fallback on instability detection

## Performance Characteristics

### Throughput Benchmarks

Based on A100-80GB testing with mixed precision:

| Model Size | Batch Size | Tokens/Second | GPU Utilization | Memory Usage |
|------------|------------|---------------|-----------------|--------------|
| 1B | 8 | 3200 | 85% | 12GB |
| 7B | 4 | 1800 | 92% | 28GB |
| 14B | 2 | 950 | 95% | 45GB |
| 50B | 1 | 280 | 98% | 78GB |

*Note: Performance varies based on sequence length, hardware configuration, and precision settings.*

### Memory Efficiency

**MoE Parameter Efficiency:**
- Only top-k experts active per forward pass
- Typical efficiency: 10-20% of total parameters active
- Memory overhead: ~5% for routing infrastructure
- Sparse gradient updates reduce optimizer memory

**Optimization Strategies:**
- Automatic batch size adjustment for memory constraints
- Gradient accumulation for large effective batch sizes
- Streaming data loading for unlimited dataset sizes
- Emergency memory recovery procedures

## Troubleshooting Guide

### Common Issues and Solutions

**Out of Memory (OOM) Errors:**
```bash
# Reduce batch size and increase gradient accumulation
python Src/Main_Scripts/main.py --batch-size 1 --grad-accum 32

# Force streaming mode for large datasets
python Src/Main_Scripts/main.py --force-streaming --precision fp16

# Enable all memory optimizations
python Src/Main_Scripts/main.py --precision fp16 --disable-compile
```

**Training Instability:**
```bash
# Reduce learning rate and increase warmup
python Src/Main_Scripts/main.py --learning-rate 5e-5 --warmup-ratio 0.2

# Switch to more stable precision
python Src/Main_Scripts/main.py --precision bf16 --max-grad-norm 0.5
```

**Data Loading Issues:**
```bash
# Validate data format
python Src/Main_Scripts/main.py --validate-data data/train.jsonl

# Test with reduced workers
python Src/Main_Scripts/main.py --num-workers 1 --disable-sharding
```

**Performance Optimization:**
```bash
# Enable all optimizations
python Src/Main_Scripts/main.py --precision mixed_bf16 --compile --enable-flash-attention

# Profile data loading
python Src/Main_Scripts/main.py --test-sharding --analyze-throughput
```

### Diagnostic Commands

**Environment Validation:**
```bash
python Src/Main_Scripts/main.py --check-environment --verbose
```

**Data Analysis:**
```bash
python Src/Main_Scripts/main.py --validate-data data/train.jsonl --create-report
```

**Configuration Testing:**
```bash
python Src/Main_Scripts/main.py --dry-run --config b7 --train-data data/small.jsonl
```

**Memory Profiling:**
```bash
python Src/Main_Scripts/main.py --profile-memory --config b14
```

## Technical Limitations

### Current Limitations

1. **Single-Node Training**: Distributed training across multiple nodes is not implemented
2. **Hardware Support**: Flash Attention requires A100/H100 GPUs for optimal performance
3. **Sequence Length**: Maximum context length is model-dependent (2K-8K tokens)
4. **MoE Scaling**: Expert count limited by memory and routing efficiency
5. **Precision Support**: Some features require specific GPU architectures

### Known Issues

1. **Flash Attention Compatibility**: May fail on older GPU architectures or specific driver versions
2. **Large Dataset Loading**: Very large datasets (>100GB) may require manual sharding
3. **Mixed Precision Stability**: Some model/precision combinations may require tuning
4. **Checkpoint Compatibility**: Checkpoints may not be compatible across different PyTorch versions

## Contributing and Development

### Code Structure

The codebase follows these principles:
- Modular architecture with clear separation of concerns
- Comprehensive error handling and logging
- Extensive configuration validation
- Memory-efficient implementations
- Production-ready fault tolerance

### Development Setup

```bash
# Development installation
git clone https://github.com/MatN23/LuminaAI.git
cd LuminaAI

# Install in development mode
pip install -e .

# Install development dependencies
pip install pytest black isort flake8 mypy

# Run tests
python -m pytest tests/

# Code formatting
black Src/
isort Src/
```

## License and Citation

This project is licensed under a Custom License. See individual source files for specific license terms.

When using LuminaAI in research or production, please consider citing:

```bibtex
@software{lumina_ai,
  title={LuminaAI: A Framework for Large Language Model Training},
  author={Nielsen, Matias},
  year={2025},
  url={https://github.com/MatN23/LuminaAI}
}
```

## Support and Documentation

### Getting Help

For technical support and questions:

1. **Check Documentation**: Review this README and inline code documentation
2. **Run Diagnostics**: Use built-in diagnostic commands to identify issues
3. **Search Issues**: Check existing GitHub issues for similar problems
4. **Environment Validation**: Ensure your setup meets requirements
5. **Create Detailed Reports**: When reporting issues, include system information, configuration, and error logs

### Additional Resources

- **Configuration Reference**: See `config_manager.py` for all available options
- **Model Architecture**: Detailed implementation in `model.py`
- **Training Pipeline**: Complete training logic in `trainer.py` and `orchestrator.py`
- **Data Processing**: Comprehensive data handling in `dataset.py` and `data_processing.py`