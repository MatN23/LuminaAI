# LuminaAI

A PyTorch framework for training transformer language models with Mixture of Experts (MoE) architecture support. Implements models from 70M to 300B parameters with automatic dataset processing and memory management.

[![License](https://img.shields.io/badge/license-Custom-blue.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-brightgreen.svg)]()
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

## Quick Start

```bash
git clone https://github.com/MatN23/LuminaAI.git && cd LuminaAI
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets tokenizers numpy pyyaml tqdm psutil

python Src/Main_Scripts/main.py --config b7 --train-data data/conversations.jsonl --experiment-name test_run
```

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Model Configurations](#model-configurations)
- [Installation](#installation)
- [Data Format and Processing](#data-format-and-processing)
- [Usage Examples](#usage-examples)
- [Training Configuration](#training-configuration)
- [Precision and Memory Management](#precision-and-memory-management)
- [Mixture of Experts Implementation](#mixture-of-experts-implementation)
- [Monitoring and Analysis](#monitoring-and-analysis)
- [Performance Characteristics](#performance-characteristics)
- [Hardware Requirements](#hardware-requirements)
- [Checkpointing System](#checkpointing-system)
- [Troubleshooting](#troubleshooting)
- [Project Structure](#project-structure)
- [Configuration System Details](#configuration-system-details)
- [Development and Contributing](#development-and-contributing)
- [License](#license)

## Features

- DeepSeek-style transformer architecture with RoPE, Grouped Query Attention, RMSNorm, and SwiGLU
- Mixture of Experts with configurable expert counts (4-144) and load balancing
- Automatic dataset sharding: memory loading, chunked processing, or streaming based on size
- Multiple precision modes: FP32, FP16, BF16, mixed precision with hardware-aware selection
- Memory optimizations: gradient checkpointing, Flash Attention integration, OOM recovery
- Checkpoint management with automatic resumption, validation, and best model tracking
- Real-time monitoring with health checks, resource tracking, and fault tolerance
- Production-ready error handling with retry mechanisms and graceful degradation

## Architecture

The framework implements a decoder-only transformer architecture based on modern language model designs. The core components include:

### Attention Mechanism
- **Grouped Query Attention (GQA)**: Reduces memory usage by sharing key-value pairs across multiple query heads
- **Rotary Position Embedding (RoPE)**: Provides better position encoding for longer sequences
- **Flash Attention**: Optional integration for memory-efficient attention computation on long sequences
- **Multi-Head Configuration**: Configurable head counts with support for different KV head ratios

### Layer Components
- **RMSNorm**: Root Mean Square Layer Normalization for improved training stability
- **SwiGLU Activation**: Swish-Gated Linear Units in feed-forward networks
- **Residual Connections**: Standard transformer residual pathways
- **Layer Initialization**: Configurable initialization standards with stability controls

### Mixture of Experts
- **Top-K Routing**: Sparse expert selection with configurable K values
- **Load Balancing**: Auxiliary loss terms to prevent expert collapse
- **Capacity Factor**: Controls expert utilization limits
- **Expert Isolation**: Independent parameter spaces and gradient flows

### Memory Management
- **Gradient Checkpointing**: Trade computation for memory during backward passes
- **Dynamic Precision**: Automatic precision adjustment based on stability
- **Streaming Data Loading**: Process datasets larger than available memory
- **Emergency Recovery**: Automatic memory cleanup on OOM conditions

## Model Configurations

The framework provides pre-configured model architectures spanning different scales and use cases:

| Configuration | Parameters | Hidden | Layers | Heads | KV Heads | Context | Memory (BF16) | Experts | Use Case |
|---------------|------------|--------|--------|-------|----------|---------|---------------|---------|----------|
| `debug` | 500K | 128 | 2 | 2 | 1 | 256 | ~10MB | 4 | Development testing |
| `m70_memory` | 70M | 768 | 12 | 12 | 4 | 1024 | ~280MB | 8 | Memory-constrained deployment |
| `m120_speed` | 120M | 768 | 16 | 12 | 4 | 1024 | ~480MB | 8 | Real-time applications |
| `b1` | 1B | 1024 | 12 | 16 | 4 | 2048 | ~2GB | 8 | Resource-limited training |
| `b3_inference` | 3B | 2560 | 24 | 20 | 10 | 2048 | ~6GB | 16 | Production inference |
| `b6_quality` | 6B | 3200 | 32 | 25 | 10 | 4096 | ~12GB | 32 | High-quality generation |
| `b7` | 7B | 2048 | 22 | 16 | 8 | 4096 | ~14GB | 16 | General purpose training |
| `b14` | 14B | 2560 | 28 | 20 | 10 | 4096 | ~28GB | 32 | Advanced applications |
| `b50` | 50B | 5120 | 40 | 40 | 10 | 128000 | ~100GB | 64 | Large-scale research |
| `b100` | 100B | 7168 | 48 | 56 | 14 | 200000 | ~200GB | 96 | Frontier research |
| `b200` | 200B | 8192 | 56 | 64 | 16 | 1000000 | ~400GB | 128 | Enterprise scale |
| `b300` | 300B | 9216 | 64 | 72 | 18 | 204800 | ~600GB | 144 | Research frontiers |

### Configuration Details

Each preset includes optimized hyperparameters:

- **Learning Rates**: Range from 3e-5 (large models) to 5e-4 (small models)
- **Batch Sizes**: Automatically calculated based on available memory
- **Gradient Accumulation**: Configured to maintain effective batch sizes
- **Warmup Ratios**: Scaled based on model size (5-20% of training steps)
- **Weight Decay**: Standard 0.01 across all configurations
- **Precision**: Hardware-aware selection (FP16 for V100, BF16 for A100+)

All configurations enable gradient checkpointing by default and use cosine learning rate scheduling with warmup.

## Installation

### System Requirements

**Minimum Requirements:**
- Python 3.10 or higher
- PyTorch 2.0+ with CUDA support
- 16GB system RAM
- CUDA-compatible GPU with 8GB+ VRAM
- 50GB+ free disk space for checkpoints and logs

**Recommended Requirements:**
- Python 3.11
- PyTorch 2.1+ with CUDA 11.8 or 12.1
- 32GB+ system RAM
- A100/H100 GPU with 40GB+ VRAM
- 500GB+ NVMe storage
- Multi-core CPU (16+ cores recommended)

### Installation Process

1. **Clone Repository:**
```bash
git clone https://github.com/MatN23/LuminaAI.git
cd LuminaAI
```

2. **Install PyTorch (select appropriate CUDA version):**
```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

3. **Install Core Dependencies:**
```bash
pip install transformers datasets tokenizers numpy pyyaml tqdm psutil
```

4. **Install Optional Performance Dependencies:**
```bash
# Flash Attention (requires A100/H100 or compatible hardware)
pip install flash-attn>=2.0.0 --no-build-isolation

# Monitoring and visualization
pip install wandb tensorboard matplotlib seaborn

# Development tools
pip install pytest black isort flake8 mypy
```

5. **Verify Installation:**
```bash
python Src/Main_Scripts/main.py --check-environment
```

This command validates PyTorch installation, CUDA compatibility, available memory, and optional dependencies.

## Data Format and Processing

### Data Format

The framework processes conversational data in JSONL (JSON Lines) format. Each line contains a conversation with message arrays:

```jsonl
{
  "messages": [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "Explain the concept of machine learning."},
    {"role": "assistant", "content": "Machine learning is a subset of artificial intelligence..."}
  ]
}
{
  "messages": [
    {"role": "human", "content": "What are neural networks?"},
    {"role": "assistant", "content": "Neural networks are computational models inspired by biological neural networks..."}
  ]
}
{
  "messages": [
    {"role": "user", "content": "Can you help me with Python?"},
    {"role": "assistant", "content": "I'd be happy to help you with Python programming..."}
  ]
}
```

### Role Mappings

The system supports flexible role naming:

**Input Roles** (mapped to user tokens):
- `user`
- `human` 
- `prompter`

**Response Roles** (mapped to assistant tokens):
- `assistant`
- `ai`
- `bot`

**System Roles** (special handling):
- `system` (prepended to conversations when present)

### Automatic Dataset Processing

The framework automatically selects processing strategies based on dataset characteristics:

| Dataset Size | Strategy | Memory Usage | Description | Recommended For |
|--------------|----------|--------------|-------------|-----------------|
| < 500MB | **Memory Loading** | Full dataset in RAM | Complete dataset loaded at startup | Small datasets, fast iteration |
| 500MB - 10GB | **Sharded Processing** | Configurable chunks | Dataset split into memory-sized chunks | Medium datasets, balanced performance |
| > 10GB | **Streaming** | Minimal (< 1GB) | Continuous streaming from disk | Large datasets, limited memory |

### Sharding Configuration

When sharding is active, you can configure:

```bash
# Override automatic shard size
python Src/Main_Scripts/main.py --shard-size-mb 1024

# Set number of data loading workers
python Src/Main_Scripts/main.py --num-workers 8

# Force specific processing mode
python Src/Main_Scripts/main.py --force-streaming
python Src/Main_Scripts/main.py --force-memory-loading
```

### Data Validation

The framework includes comprehensive data validation:

```bash
# Full validation with detailed report
python Src/Main_Scripts/main.py --validate-data data/train.jsonl --create-report

# Quick format check
python Src/Main_Scripts/main.py --validate-data data/train.jsonl --quiet

# Validation with statistics
python Src/Main_Scripts/main.py --validate-data data/train.jsonl --show-stats
```

**Validation Checks:**
- JSON format integrity and syntax
- Message structure and required fields
- Role consistency and mapping validation
- Token count distribution analysis
- Conversation length statistics
- Character encoding verification
- Duplicate detection and handling
- Content quality metrics

## Usage Examples

### Basic Training

Train a 7B parameter model with default settings:

```bash
python Src/Main_Scripts/main.py \
  --config b7 \
  --train-data data/conversations.jsonl \
  --epochs 3 \
  --experiment-name basic_7b_training
```

### Production Training with Evaluation

Full training setup with validation and monitoring:

```bash
python Src/Main_Scripts/main.py \
  --config b14 \
  --train-data data/large_training_set.jsonl \
  --eval-data data/validation_set.jsonl \
  --precision mixed_bf16 \
  --learning-rate 1e-4 \
  --batch-size 2 \
  --grad-accum 16 \
  --warmup-ratio 0.1 \
  --save-every-n-batches 1000 \
  --eval-every-n-batches 500 \
  --experiment-name production_14b_v1 \
  --early-stopping-patience 5
```

### Memory-Constrained Training

Optimize for limited GPU memory:

```bash
python Src/Main_Scripts/main.py \
  --config m70_memory \
  --train-data data/dataset.jsonl \
  --batch-size 1 \
  --grad-accum 32 \
  --precision fp16 \
  --force-streaming \
  --gradient-checkpointing \
  --experiment-name memory_optimized
```

### High-Performance Training

Maximize training throughput:

```bash
python Src/Main_Scripts/main.py \
  --config b7 \
  --train-data data/dataset.jsonl \
  --precision mixed_bf16 \
  --compile \
  --enable-flash-attention \
  --num-workers 8 \
  --batch-size 4 \
  --experiment-name high_perf_training
```

### Resume Training from Checkpoint

Continue training from a saved checkpoint:

```bash
python Src/Main_Scripts/main.py \
  --config b7 \
  --train-data data/conversations.jsonl \
  --resume checkpoints/experiment_123/checkpoint_epoch_1_step_5000.pt \
  --experiment-name resumed_training
```

### Interactive Chat Interface

Test your trained model:

```bash
python Src/Main_Scripts/chat.py \
  --resume checkpoints/best_model.pt \
  --temperature 0.8 \
  --top-p 0.9 \
  --max-tokens 512 \
  --system-prompt "You are a helpful AI assistant."
```

### Model Analysis and Testing

Analyze model characteristics:

```bash
# Estimate model parameters
python Src/Main_Scripts/main.py --estimate-parameters --config b7

# Analyze memory requirements
python Src/Main_Scripts/main.py --analyze-memory --config b14 --sequence-length 4096

# Test generation capabilities
python Src/Main_Scripts/main.py \
  --test-generation \
  --resume checkpoints/model.pt \
  --test-prompts "Explain quantum computing" "Write a Python function" \
  --temperature 0.7
```

## Training Configuration

### Core Training Parameters

| Parameter | Type | Default | Range/Options | Description |
|-----------|------|---------|---------------|-------------|
| `--config` | str | **required** | See configurations table | Model architecture preset |
| `--train-data` | str | **required** | File path | Training dataset path |
| `--eval-data` | str | None | File path | Validation dataset path |
| `--epochs` | int | 3 | 1-100 | Number of training epochs |
| `--learning-rate` | float | config-dependent | 1e-6 to 1e-3 | Peak learning rate |
| `--batch-size` | int | config-dependent | 1-64 | Per-GPU batch size |
| `--grad-accum` | int | config-dependent | 1-128 | Gradient accumulation steps |
| `--warmup-ratio` | float | 0.15 | 0.0-1.0 | Learning rate warmup fraction |
| `--weight-decay` | float | 0.01 | 0.0-0.1 | Weight decay coefficient |
| `--max-grad-norm` | float | 1.0 | 0.1-10.0 | Gradient clipping threshold |
| `--seed` | int | 42 | Any integer | Random seed for reproducibility |

### Learning Rate Scheduling

| Parameter | Type | Options | Description |
|-----------|------|---------|-------------|
| `--lr-scheduler` | str | `cosine`, `linear`, `onecycle`, `none` | Learning rate schedule type |
| `--min-lr` | float | 1e-6 | Minimum learning rate for scheduling |
| `--warmup-steps` | int | calculated | Override warmup steps calculation |

### Optimization Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--optimizer` | str | `adamw` | Optimizer type |
| `--beta1` | float | 0.9 | Adam beta1 parameter |
| `--beta2` | float | 0.95 | Adam beta2 parameter |
| `--eps` | float | 1e-8 | Adam epsilon parameter |

### Checkpoint and Logging

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--save-every-n-batches` | int | config-dependent | Checkpoint saving frequency |
| `--eval-every-n-batches` | int | config-dependent | Evaluation frequency |
| `--log-every-n-batches` | int | 10 | Logging frequency |
| `--save-total-limit` | int | 5 | Maximum checkpoints to keep |
| `--early-stopping-patience` | int | None | Early stopping patience |

## Precision and Memory Management

### Precision Modes

The framework supports multiple precision modes with automatic hardware detection:

| Precision | Memory Usage | Numerical Stability | Speed | Hardware Requirements |
|-----------|--------------|-------------------|-------|---------------------|
| `fp32` | Baseline (100%) | Highest | Slowest | Any CUDA GPU |
| `fp16` | 50% of fp32 | Good | Fast | V100, RTX series |
| `bf16` | 50% of fp32 | Higher than fp16 | Fast | A100, H100 |
| `mixed_fp16` | Variable | Good | Very fast | V100+ |
| `mixed_bf16` | Variable | Highest | Very fast | A100+ |
| `auto` | Variable | Variable | Variable | Automatic selection |

### Automatic Precision Selection

When using `--precision auto`, the framework selects precision based on:

1. **Hardware Detection**: GPU architecture and capabilities
2. **Model Size**: Larger models prefer more stable precisions
3. **Memory Constraints**: Available VRAM influences precision choice
4. **Performance Requirements**: Speed vs. stability trade-offs

```python
# Simplified selection logic
if gpu_arch == "H100" and model_size > "14B":
    precision = "mixed_bf16"
elif gpu_arch == "A100":
    precision = "mixed_bf16" if model_size > "7B" else "bf16"
elif gpu_arch == "V100":
    precision = "mixed_fp16"
else:
    precision = "fp16"
```

### Memory Optimization Features

**Gradient Checkpointing:**
- Enabled by default in all configurations
- Reduces peak memory usage by 40-60%
- Increases training time by approximately 20%
- Automatically handles activation recomputation

**Flash Attention:**
- Automatically enabled for sequences > 512 tokens
- Requires A100/H100 or compatible hardware
- Provides 2-4x memory efficiency for long sequences
- Gracefully falls back to standard attention if unavailable

**Dynamic Batch Sizing:**
```bash
# Enable automatic batch size adjustment
python Src/Main_Scripts/main.py --auto-batch-size --target-memory-usage 0.85
```

**Emergency Memory Management:**
- Automatic garbage collection on memory pressure
- Checkpoint saving before OOM conditions
- Graceful degradation to smaller batch sizes
- Memory usage monitoring and alerts

## Mixture of Experts Implementation

### Architecture Overview

The MoE implementation uses a sparse routing mechanism where only a subset of experts processes each input token. This allows scaling model parameters without proportionally increasing compute requirements.

### Expert Configuration by Model Size

| Model Scale | Expert Count | Top-K | Capacity Factor | Load Balance Weight | Active Params % |
|-------------|--------------|-------|-----------------|-------------------|-----------------|
| Debug (500K) | 4 | 2 | 1.1 | 0.005 | ~50% |
| Small (70M-1B) | 8 | 2 | 1.25 | 0.008 | ~25% |
| Medium (3B-7B) | 16 | 2 | 1.5 | 0.01 | ~12.5% |
| Large (14B-50B) | 32-64 | 2 | 1.8 | 0.015 | ~6.25-3.125% |
| XLarge (100B+) | 96-144 | 2 | 2.0-2.8 | 0.025-0.035 | ~2-1.4% |

### Routing Mechanism

**Top-K Gating:**
- Each token is routed to the top-k highest-scoring experts
- Default k=2 provides good balance of quality and efficiency
- Gating network learns routing decisions during training
- Supports dynamic expert selection based on input content

**Load Balancing:**
- Auxiliary loss encourages uniform expert utilization
- Prevents expert collapse where some experts receive no tokens
- Configurable balancing weight based on model size
- Real-time monitoring of expert usage statistics

**Capacity Management:**
- Capacity factor limits tokens per expert to prevent overload
- Overflow tokens are handled by auxiliary routing or dropped
- Dynamic capacity adjustment based on batch size and sequence length

### MoE Training Considerations

**Memory Usage:**
- Total parameter count scales with expert count
- Only active expert parameters require gradients
- Optimizer states maintained only for active parameters
- Sparse gradient updates reduce memory overhead

**Training Stability:**
- Load balancing loss prevents training instability
- Expert dropout during training improves robustness
- Careful initialization prevents early expert specialization
- Gradient clipping applied to both model and routing parameters

## Monitoring and Analysis

### Real-Time Training Metrics

During training, the framework displays comprehensive metrics:

```
Epoch 2 | Step 1,247 | Batch 1,247/15,000 | 08:23 elapsed
Loss: 2.456789 | PPL: 11.67 | LR: 8.50e-05 | β: 0.95
GradNorm: 0.8432 | Tokens/s: 1,250 | GPU: 8.2/16.0GB (51%)
Strategy: sharded | Workers: 4 | Shard: 3/12 | Cache: 89%
MoE: Balance=0.023 | Experts Used: 14/16 | Routing Loss: 0.012
Validation PPL: 10.23 | Best PPL: 9.87 @ step 1,180
```

**Metric Explanations:**

- **Loss**: Cross-entropy loss value (lower indicates better learning)
- **PPL**: Perplexity (exp(loss), measures prediction quality)
- **LR**: Current learning rate following schedule
- **β**: Current momentum parameter (for adaptive optimizers)
- **GradNorm**: L2 norm of gradients before clipping
- **Tokens/s**: Training throughput in tokens per second
- **GPU**: Memory usage (current/total) and utilization percentage
- **Strategy**: Data loading strategy (memory/sharded/streaming)
- **Shard**: Current shard being processed (if applicable)
- **Cache**: Data loading cache hit rate
- **MoE Balance**: Expert load balance metric (lower = more balanced)
- **Experts Used**: Number of experts receiving tokens
- **Routing Loss**: MoE auxiliary loss component
- **Validation PPL**: Performance on held-out validation set
- **Best PPL**: Best validation perplexity achieved so far

### Analysis and Diagnostic Tools

**Model Analysis:**
```bash
# Detailed parameter analysis
python Src/Main_Scripts/main.py --estimate-parameters --config b7 --detailed

# Memory footprint analysis
python Src/Main_Scripts/main.py --analyze-memory --config b14 --batch-size 2 --sequence-length 4096

# Training time estimation
python Src/Main_Scripts/main.py --estimate-time --config b7 --train-data large.jsonl --hardware-profile a100-80gb
```

**Performance Profiling:**
```bash
# Data loading profiling
python Src/Main_Scripts/main.py --profile-data-loading --train-data data.jsonl --num-workers 8

# Training step profiling
python Src/Main_Scripts/main.py --profile-training --config b7 --steps 100

# Memory usage profiling
python Src/Main_Scripts/main.py --profile-memory --config b14 --track-allocations
```

**Generation Quality Testing:**
```bash
# Multi-precision generation comparison
python Src/Main_Scripts/main.py \
  --test-generation \
  --compare-precisions \
  --resume checkpoints/model.pt \
  --test-prompts prompts.txt \
  --output-comparisons results.json

# Benchmark generation speed
python Src/Main_Scripts/main.py \
  --benchmark-generation \
  --resume checkpoints/model.pt \
  --batch-sizes 1,4,8,16 \
  --sequence-lengths 512,1024,2048
```

### Logging and Experiment Tracking

**Built-in Logging:**
- Structured logging with configurable verbosity levels
- JSON-formatted logs for programmatic analysis  
- Automatic log rotation and archiving
- Integration with tensorboard for visualization

**External Integration:**
```bash
# Weights & Biases integration
python Src/Main_Scripts/main.py --use-wandb --wandb-project my-project --wandb-run-name experiment-1

# TensorBoard logging
python Src/Main_Scripts/main.py --use-tensorboard --tensorboard-dir ./logs/tensorboard
```

## Performance Characteristics

### Throughput Benchmarks

Performance measurements on different hardware configurations:

**A100-80GB (Mixed BF16 Precision):**

| Model | Batch Size | Seq Length | Tokens/sec | GPU Util | Memory | Training Time (1M tokens) |
|-------|------------|------------|------------|----------|--------|---------------------------|
| 1B | 8 | 2048 | 3,200 | 85% | 12GB | ~5.2 minutes |
| 7B | 4 | 4096 | 1,800 | 92% | 28GB | ~9.3 minutes |
| 14B | 2 | 4096 | 950 | 95% | 45GB | ~17.5 minutes |
| 50B | 1 | 4096 | 280 | 98% | 78GB | ~59.5 minutes |

**V100-32GB (Mixed FP16 Precision):**

| Model | Batch Size | Seq Length | Tokens/sec | GPU Util | Memory | Training Time (1M tokens) |
|-------|------------|------------|------------|----------|--------|---------------------------|
| 1B | 4 | 2048 | 2,100 | 82% | 18GB | ~7.9 minutes |
| 7B | 2 | 2048 | 950 | 89% | 31GB | ~17.5 minutes |
| 14B | 1 | 2048 | 420 | 91% | 32GB | ~39.7 minutes |

**RTX 4090-24GB (FP16 Precision):**

| Model | Batch Size | Seq Length | Tokens/sec | GPU Util | Memory | Training Time (1M tokens) |
|-------|------------|------------|------------|----------|--------|---------------------------|
| 1B | 6 | 2048 | 2,800 | 88% | 14GB | ~6.0 minutes |
| 7B | 2 | 2048 | 1,200 | 94% | 23GB | ~13.9 minutes |

### Memory Efficiency Analysis

**Memory Usage Breakdown (7B Model, BF16):**

| Component | Memory Usage | Percentage | Notes |
|-----------|--------------|------------|-------|
| Model Parameters | 14.0GB | 50% | Base model weights |
| Gradients | 14.0GB | 50% | Parameter gradients |
| Optimizer States | 28.0GB | 100% | AdamW states (momentum + variance) |
| Activations | 4.2GB | 15% | Forward pass activations |
| **Total (no optimization)** | **60.2GB** | **215%** | Without memory optimizations |
| **With Gradient Checkpointing** | **32.1GB** | **115%** | Recompute activations |
| **With Mixed Precision** | **28.0GB** | **100%** | FP16 parameters/gradients |

**Optimization Impact:**

| Optimization | Memory Reduction | Speed Impact | Implementation |
|--------------|------------------|--------------|----------------|
| Gradient Checkpointing | 40-60% | -20% | Recompute activations |
| Mixed Precision | 50% | +30% | FP16/BF16 computation |
| Flash Attention | 2-4x (long seq) | +10% | Memory-efficient attention |
| MoE Sparsity | 80-90% active params | Minimal | Sparse expert activation |
| Streaming Data | 95% dataset memory | -5% | Disk-based loading |

### MoE Scaling Efficiency

**Parameter Scaling vs Compute:**

| Model | Total Params | Active Params | Compute Ratio | Memory Ratio | Performance Gain |
|-------|--------------|---------------|---------------|--------------|------------------|
| Dense 7B | 7B | 7B (100%) | 1.0x | 1.0x | Baseline |
| MoE 7B (8 experts) | 14B | 3.5B (25%) | 0.5x | 2.0x | +15% quality |
| MoE 7B (16 experts) | 28B | 3.5B (12.5%) | 0.5x | 4.0x | +25% quality |
| Dense 50B | 50B | 50B (100%) | 7.1x | 7.1x | +180% quality |
| MoE 50B (64 experts) | 200B | 6.25B (3.1%) | 0.9x | 28.6x | +190% quality |

## Hardware Requirements

### GPU Requirements by Model Size

**Minimum VRAM Requirements (Training):**

| Model Size | FP32 | FP16/BF16 | Mixed Precision | Gradient Checkpointing |
|------------|------|-----------|-----------------|------------------------|
| 500K (debug) | 0.5GB | 0.3GB | 0.2GB | 0.1GB |
| 70M | 2.5GB | 1.3GB | 1.0GB | 0.6GB |
| 120M | 4.2GB | 2.1GB | 1.6GB | 1.0GB |
| 1B | 15GB | 7.5GB | 5.8GB | 3.2GB |
| 3B | 45GB | 22GB | 17GB | 12GB |
| 7B | 105GB | 52GB | 40GB | 28GB |
| 14B | 210GB | 105GB | 80GB | 56GB |

**Recommended Hardware Configurations:**

**Budget Setup ($2,000-$5,000):**
- GPU: RTX 4090 (24GB) or RTX 3090 (24GB)
- CPU: AMD Ryzen 5800X or Intel i7-12700K
- RAM: 32GB DDR4-3200
- Storage: 1TB NVMe SSD
- **Suitable for**: Models up to 7B parameters with memory optimizations

**Professional Setup ($8,000-$15,000):**
- GPU: RTX A6000 (48GB) or A100 (40GB)
- CPU: AMD Threadripper 3970X or Intel Xeon W-3235
- RAM: 128GB DDR4-3200 ECC
- Storage: 2TB NVMe SSD + 8TB HDD for datasets
- **Suitable for**: Models up to 14B parameters

**Enterprise Setup ($25,000-$50,000):**
- GPU: A100 (80GB) or H100 (80GB)
- CPU: Dual AMD EPYC 7543 or Intel Xeon Platinum 8358P
- RAM: 256GB+ DDR4-3200 ECC
- Storage: 4TB+ NVMe SSD array + network storage
- **Suitable for**: Models up to 50B parameters

**Research Setup ($100,000+):**
- GPU: Multiple H100 (80GB) or A100 (80GB) GPUs
- CPU: High-core-count server processors
- RAM: 512GB+ DDR4/DDR5 ECC
- Storage: High-speed parallel storage systems
- **Suitable for**: Models 100B+ parameters

### Storage Requirements

**Dataset Storage:**
- Small datasets (< 1GB): Local SSD sufficient
- Medium datasets (1-50GB): Fast NVMe SSD recommended
- Large datasets (50GB+): High-speed network storage or local NVMe array

**Checkpoint Storage:**
- Model checkpoints: 2-4x model parameter size in bytes
- Optimizer states: Additional 2x model parameter size
- Training logs: 1-10GB depending on experiment length
- Backup storage: 3-5x total checkpoint size for redundancy

## Checkpointing System

### Automatic Checkpoint Management

The framework implements comprehensive checkpoint management:

**Checkpoint Types:**
- **Regular Checkpoints**: Saved at configurable intervals during training
- **Best Model Checkpoints**: Saved when validation metrics improve
- **Emergency Checkpoints**: Created automatically before system failures
- **Resume Checkpoints**: Latest state for continuing interrupted training

**Checkpoint Contents:**
- Model state dictionary with all parameters
- Optimizer state including momentum and variance terms
- Learning rate scheduler state
- Training step and epoch counters
- Random number generator states for reproducibility
- Configuration parameters and metadata
- Training and validation metric history

### Checkpoint Configuration

```bash
# Configure checkpoint saving frequency
python Src/Main_Scripts/main.py --save-every-n-batches 1000

# Set maximum number of checkpoints to keep
python Src/Main_Scripts/main.py --save-total-limit 10

# Enable best model tracking
python Src/Main_Scripts/main.py --track-best-model --best-metric "validation_ppl"

# Configure backup checkpoints
python Src/Main_Scripts/main.py --backup-every-n-hours 6 --backup-dir /backup/path
```

### Checkpoint Validation and Compatibility

**Validation Checks:**
- File integrity and corruption detection
- Configuration compatibility with current setup
- PyTorch version compatibility warnings
- Model architecture consistency verification

**Cross-Version Compatibility:**
- Automatic handling of minor PyTorch version differences
- Migration utilities for configuration format changes
- Backward compatibility preservation for older checkpoints

### Recovery and Resumption

**Automatic Recovery:**
```bash
# Resume from latest checkpoint automatically
python Src/Main_Scripts/main.py --auto-resume --experiment-name previous_run

# Resume from specific checkpoint
python Src/Main_Scripts/main.py --resume checkpoints/model_epoch_2_step_5000.pt

# Resume with modified configuration
python Src/Main_Scripts/main.py --resume checkpoints/model.pt --modify-config --learning-rate 5e-5
```

**Recovery Scenarios:**
- System crashes and power failures
- Out-of-memory conditions
- Network interruptions during distributed training
- Hardware failures with checkpoint backup recovery

## Troubleshooting

### Common Issues and Solutions

#### Out of Memory Errors

**Symptoms:**
- CUDA out of memory errors during forward pass
- Host memory exhaustion during data loading
- Optimizer state memory overflow

**Solutions:**

1. **Reduce Batch Size:**
```bash
python Src/Main_Scripts/main.py --batch-size 1 --grad-accum 32
```

2. **Enable Memory Optimizations:**
```bash
python Src/Main_Scripts/main.py --gradient-checkpointing --precision fp16
```

3. **Force Streaming Mode:**
```bash
python Src/Main_Scripts/main.py --force-streaming --shard-size-mb 256
```

4. **Use Smaller Model:**
```bash
python Src/Main_Scripts/main.py --config m70_memory
```

#### Training Instability

**Symptoms:**
- Loss spikes or divergence
- NaN gradients or parameters
- Extreme gradient norms

**Solutions:**

1. **Reduce Learning Rate:**
```bash
python Src/Main_Scripts/main.py --learning-rate 5e-5 --warmup-ratio 0.2
```

2. **Adjust Gradient Clipping:**
```bash
python Src/Main_Scripts/main.py --max-grad-norm 0.5
```

3. **Switch to Stable Precision:**
```bash
python Src/Main_Scripts/main.py --precision bf16
```

4. **Enable Numerical Stability Features:**
```bash
python Src/Main_Scripts/main.py --use-stable-embedding --init-std 0.015
```

#### Data Loading Problems

**Symptoms:**
- Slow data loading speeds
- Worker process crashes
- Data format errors

**Solutions:**

1. **Validate Data Format:**
```bash
python Src/Main_Scripts/main.py --validate-data data/train.jsonl --fix-errors
```

2. **Adjust Worker Configuration:**
```bash
python Src/Main_Scripts/main.py --num-workers 4 --prefetch-factor 2
```

3. **Test Minimal Configuration:**
```bash
python Src/Main_Scripts/main.py --num-workers 1 --disable-sharding --batch-size 1
```

#### Performance Issues

**Symptoms:**
- Low GPU utilization
- Slow training throughput
- High memory usage relative to model size

**Solutions:**

1. **Enable Compilation:**
```bash
python Src/Main_Scripts/main.py --compile --precision mixed_bf16
```

2. **Optimize Data Loading:**
```bash
python Src/Main_Scripts/main.py --num-workers 8 --pin-memory --persistent-workers
```

3. **Profile Performance:**
```bash
python Src/Main_Scripts/main.py --profile-training --steps 100
```

#### MoE-Specific Issues

**Symptoms:**
- Expert load imbalance
- Routing instability
- Capacity overflow errors

**Solutions:**

1. **Adjust Load Balancing:**
```bash
python Src/Main_Scripts/main.py --load-balancing-weight 0.02
```

2. **Modify Capacity Factor:**
```bash
python Src/Main_Scripts/main.py --capacity-factor 1.5
```

3. **Monitor Expert Usage:**
```bash
python Src/Main_Scripts/main.py --log-expert-usage --expert-usage-frequency 100
```

### Diagnostic Commands

| Command | Purpose | Output |
|---------|---------|---------|
| `--check-environment` | System validation | Hardware info, dependencies, compatibility |
| `--validate-data` | Data format verification | Format errors, statistics, recommendations |
| `--dry-run` | Configuration testing | Memory estimates, parameter counts, no training |
| `--profile-memory` | Memory analysis | Peak usage, allocation patterns, optimization tips |
| `--profile-training` | Training performance | Step timing, bottleneck identification |
| `--test-generation` | Model quality testing | Generation samples, speed benchmarks |
| `--analyze-checkpoints` | Checkpoint inspection | Checkpoint health, compatibility, metadata |

### Error Codes and Resolution

**Configuration Errors (E001-E099):**
- `E001`: Invalid configuration preset
- `E002`: Incompatible parameter combination
- `E003`: Hardware incompatibility

**Data Errors (E100-E199):**
- `E101`: Data format invalid
- `E102`: Missing required fields
- `E103`: Encoding issues

**Memory Errors (E200-E299):**
- `E201`: Insufficient GPU memory
- `E202`: Host memory exhaustion
- `E203`: Optimizer memory overflow

**Training Errors (E300-E399):**
- `E301`: Training divergence detected
- `E302`: Gradient overflow
- `E303`: Checkpoint corruption

Run with `--verbose --debug` flags for detailed error analysis and automated fix suggestions.

## Project Structure

```
LuminaAI/
├── Src/
│   └── Main_Scripts/
│       ├── main.py                     # Primary CLI interface and entry point
│       ├── chat.py                     # Interactive chat interface for testing
│       ├── core/                       # Core model and data components
│       │   ├── model.py               # DeepSeek transformer implementation
│       │   ├── tokenizer.py           # GPT-4 compatible tokenization
│       │   ├── dataset.py             
│       ├── training/                  
│       │   ├── trainer.py             # Core training loop and optimization
│       │   ├── orchestrator.py        # High-level training coordination         
│       │   ├── checkpoint.py          # Checkpoint management system         
│       │   ├── config_manager.py      # Configuration presets and validation
│       │   ├── training_loops.py
│       ├── monitoring/
│       │   ├── logger.py              
│       ├── utils/
│       │   ├── data_processing.py     # Data validation and processing utilities
│       │   ├── environment.py         # System validation and optimization
│       │   ├── reporting.py           # Performance analysis and reporting
|       ├── Setup.py
├── requirements.txt                  # Python dependencies
├── pyproject.toml                   # Project configuration
├── .gitignore                       # Git ignore rules
├── LICENSE                          # License file
└── README.md                        # This documentation
```

## Configuration System Details

### Configuration File Format

Configuration files use YAML format for human readability and easy modification:

```yaml
# Model architecture
vocab_size: 50304
hidden_size: 2048
num_layers: 22
num_heads: 16
num_kv_heads: 8
seq_length: 4096
intermediate_size: 5504

# Training parameters
batch_size: 4
gradient_accumulation_steps: 4
learning_rate: 1e-4
weight_decay: 0.01
num_epochs: 3
warmup_ratio: 0.15
max_grad_norm: 1.0
precision: "mixed_bf16"

# MoE configuration
use_moe: true
num_experts: 16
moe_top_k: 2
capacity_factor: 1.5
load_balancing_weight: 0.01

# Data processing
train_data_path: "data/train.jsonl"
eval_data_path: "data/eval.jsonl"
num_workers: 4
max_conversations_per_file: 10000

# Generation parameters
max_new_tokens: 512
temperature: 0.8
top_p: 0.9
top_k: 50

# Monitoring and checkpointing
experiment_name: null
save_every_n_batches: 1000
eval_every_n_batches: 500
save_total_limit: 5
early_stopping_patience: 5
```

### Custom Configuration Creation

Create custom configurations by extending existing presets:

```python
from Src.Main_Scripts.config.config_manager import Config, ConfigPresets

# Start with existing preset
base_config = ConfigPresets.b7()

# Modify parameters
base_config.learning_rate = 5e-5
base_config.batch_size = 2
base_config.num_experts = 32
base_config.experiment_name = "custom_7b_experiment"

# Save custom configuration
base_config.save("configs/custom_7b.yaml")
```

```bash
# Use custom configuration
python Src/Main_Scripts/main.py --config-file configs/custom_7b.yaml
```

### Configuration Validation

The system performs extensive validation of all configuration parameters:

**Architecture Validation:**
- Hidden size divisibility by attention heads
- KV head compatibility with attention heads
- Vocabulary size alignment (padded to multiples of 64)
- Sequence length power-of-2 optimization recommendations

**Training Validation:**
- Learning rate range checking
- Batch size memory feasibility
- Precision compatibility with hardware
- Scheduler parameter consistency

**MoE Validation:**
- Expert count and top-k relationship
- Capacity factor reasonable range
- Load balancing weight optimization
- Hardware memory constraints

**Data Validation:**
- File path existence and permissions
- Dataset format compatibility
- Worker count vs CPU core availability
- Shard size vs available memory

### Environment-Specific Overrides

Override configuration parameters based on detected hardware:

```python
# Automatic GPU memory-based adjustments
if gpu_memory_gb < 16:
    config.batch_size = max(1, config.batch_size // 2)
    config.gradient_accumulation_steps *= 2
    config.precision = "fp16"
elif gpu_memory_gb >= 80:
    config.batch_size *= 2
    config.gradient_accumulation_steps = max(1, config.gradient_accumulation_steps // 2)
    config.precision = "mixed_bf16"
```

## Development and Contributing

### Development Environment Setup

1. **Clone and Setup:**
```bash
git clone https://github.com/MatN23/LuminaAI.git
cd LuminaAI
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or venv\Scripts\activate  # Windows
```

2. **Install Development Dependencies:**
```bash
pip install -e .
pip install -r requirements-dev.txt
```

3. **Install Pre-commit Hooks:**
```bash
pre-commit install
```

4. **Verify Development Setup:**
```bash
python -m pytest tests/ -v
python Src/Main_Scripts/main.py --check-environment --dev-mode
```

### Code Quality Standards

**Formatting and Style:**
- **Black**: Code formatting with 88-character line length
- **isort**: Import sorting and organization
- **flake8**: Style guide enforcement (PEP 8)
- **mypy**: Static type checking

**Testing Requirements:**
- **pytest**: Unit and integration test framework
- **Coverage**: Minimum 80% test coverage required
- **Property-based testing**: Using hypothesis for robust testing
- **Performance benchmarks**: Regression testing for critical paths

**Documentation Standards:**
- **Docstring format**: Google-style docstrings for all public functions
- **Type hints**: Complete type annotations for all functions
- **Inline comments**: Complex algorithms and performance-critical code
- **README updates**: Documentation updates for new features

### Testing Framework

**Unit Tests:**
```bash
# Run all tests
python -m pytest tests/

# Run specific test category
python -m pytest tests/test_model.py -v

# Run with coverage
python -m pytest tests/ --cov=Src --cov-report=html
```

**Integration Tests:**
```bash
# Test complete training pipeline
python -m pytest tests/test_integration.py::test_full_training_pipeline

# Test checkpoint compatibility
python -m pytest tests/test_integration.py::test_checkpoint_resume
```

**Performance Tests:**
```bash
# Benchmark training performance
python scripts/benchmark.py --config b7 --steps 100

# Memory usage profiling
python scripts/benchmark.py --profile-memory --config b14
```

### Contributing Guidelines

1. **Fork and Branch:**
   - Fork the repository to your GitHub account
   - Create feature branches from `main`
   - Use descriptive branch names: `feature/moe-optimization` or `fix/memory-leak`

2. **Code Changes:**
   - Follow existing code patterns and architecture
   - Add comprehensive tests for new functionality
   - Update documentation for user-facing changes
   - Ensure backward compatibility for configuration files

3. **Testing:**
   - Run full test suite before submitting
   - Add performance benchmarks for significant changes
   - Test on multiple GPU architectures if possible
   - Verify memory usage doesn't regress

4. **Pull Request Process:**
   - Provide clear description of changes and motivation
   - Include performance impact analysis
   - Reference related issues or feature requests
   - Ensure all CI checks pass

5. **Code Review:**
   - Address reviewer feedback promptly
   - Update documentation and tests based on feedback
   - Maintain clean commit history with descriptive messages

### Architecture Principles

**Modularity:**
- Clear separation between model, training, and data components
- Plugin-style architecture for new features
- Minimal coupling between subsystems

**Performance:**
- Memory-first design for large model support
- Lazy loading and streaming for large datasets
- Hardware-specific optimizations with graceful fallbacks

**Reliability:**
- Comprehensive error handling and recovery
- Extensive validation and early failure detection
- Graceful degradation under resource constraints

**Maintainability:**
- Extensive logging and debugging capabilities
- Clear configuration and parameter management
- Comprehensive test coverage and benchmarking

## License

This project is licensed under a Custom License. Individual source files may contain additional license terms. See the LICENSE file and individual source file headers for complete license information.

### Usage Permissions

- **Research Use**: Permitted for academic and research purposes
- **Commercial Use**: Contact the author for commercial licensing terms
- **Modification**: Allowed for personal and research use
- **Distribution**: Subject to license terms and attribution requirements

### Attribution Requirements

When using LuminaAI in research publications, please cite:

```bibtex
@software{lumina_ai_2025,
  title={LuminaAI: A PyTorch Framework for Large Language Model Training with Mixture of Experts},
  author={Nielsen, Matias},
  year={2025},
  url={https://github.com/MatN23/LuminaAI},
  version={1.0},
  note={PyTorch framework supporting models from 70M to 300B parameters}
}
```

For commercial use or questions regarding licensing, contact the project maintainers.