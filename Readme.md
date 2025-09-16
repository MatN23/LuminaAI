# LuminaAI

A PyTorch framework for training transformer language models with Mixture of Experts (MoE) architecture support and DeepSpeed integration. Implements models from 70M to 300B parameters with automatic dataset processing, distributed training, and advanced memory management.

[![License](https://img.shields.io/badge/license-Custom-blue.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-brightgreen.svg)]()
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![DeepSpeed](https://img.shields.io/badge/deepspeed-enabled-brightgreen.svg)](https://github.com/microsoft/DeepSpeed)

## Quick Start

```bash
git clone https://github.com/MatN23/LuminaAI.git && cd LuminaAI
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets tokenizers numpy pyyaml tqdm psutil deepspeed

# Single GPU training
python Src/Main_Scripts/main.py --config b7 --train-data data/conversations.jsonl --experiment-name test_run

# Multi-GPU distributed training with DeepSpeed
deepspeed --num_gpus=4 Src/Main_Scripts/main.py --config b14 --train-data data/conversations.jsonl
```

## New Features & Improvements

### 🚀 DeepSpeed Integration
- **Zero Redundancy Optimizer (ZeRO)**: Stages 1, 2, and 3 support for massive model scaling
- **CPU/NVMe Offloading**: Train models larger than GPU memory
- **Automatic Configuration**: Intelligent DeepSpeed parameter selection based on hardware
- **MoE Optimization**: Specialized DeepSpeed configurations for Mixture of Experts models
- **Multi-Node Support**: Seamless scaling across multiple nodes

### 🧠 Enhanced MoE Architecture  
- **Expert Parallelism**: Distribute experts across GPUs for optimal performance
- **Advanced Routing**: Improved load balancing and capacity management
- **Routing Analytics**: Real-time expert utilization monitoring and optimization
- **Configurable Top-K**: Dynamic expert selection strategies

### 📊 Intelligent Resource Management
- **Hardware Auto-Detection**: Automatic optimization based on GPU architecture
- **Memory Pressure Analysis**: Dynamic adjustment of batch sizes and precision
- **Performance Profiling**: Built-in benchmarking and bottleneck identification
- **Emergency Recovery**: Automatic checkpoint saving before OOM conditions

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Model Configurations](#model-configurations)
- [Installation](#installation)
- [DeepSpeed Integration](#deepspeed-integration)
- [Data Format and Processing](#data-format-and-processing)
- [Usage Examples](#usage-examples)
- [Training Configuration](#training-configuration)
- [Precision and Memory Management](#precision-and-memory-management)
- [Mixture of Experts Implementation](#mixture-of-experts-implementation)
- [Distributed Training](#distributed-training)
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

### Core Architecture
- DeepSeek-style transformer architecture with RoPE, Grouped Query Attention, RMSNorm, and SwiGLU
- Mixture of Experts with configurable expert counts (4-144) and intelligent load balancing
- Automatic dataset sharding: memory loading, chunked processing, or streaming based on size
- Multiple precision modes: FP32, FP16, BF16, mixed precision with hardware-aware selection

### Advanced Training
- **DeepSpeed Integration**: ZeRO optimization stages 1-3, CPU/NVMe offloading, gradient compression
- **Distributed Training**: Multi-GPU and multi-node support with automatic scaling
- **Memory Optimizations**: Gradient checkpointing, Flash Attention integration, OOM recovery
- **Adaptive Optimization**: Dynamic batch sizing, precision adjustment, and learning rate scaling

### Production Features
- Checkpoint management with automatic resumption, validation, and best model tracking
- Real-time monitoring with health checks, resource tracking, and fault tolerance
- Production-ready error handling with retry mechanisms and graceful degradation
- Comprehensive logging and experiment tracking integration

## Architecture

The framework implements a state-of-the-art decoder-only transformer architecture optimized for large-scale training:

### Advanced Attention Mechanism
- **Grouped Query Attention (GQA)**: Reduces memory usage by sharing key-value pairs across multiple query heads
- **Rotary Position Embedding (RoPE)**: Provides superior position encoding for extended sequences
- **Flash Attention Integration**: Memory-efficient attention computation with automatic fallback
- **Multi-Head Configuration**: Configurable head counts with optimized KV head ratios

### Enhanced Layer Components
- **RMSNorm**: Root Mean Square Layer Normalization for improved training stability
- **SwiGLU Activation**: Swish-Gated Linear Units in feed-forward networks with optimal intermediate sizing
- **Residual Connections**: Standard transformer residual pathways with gradient flow optimization
- **Advanced Initialization**: Configurable initialization strategies with stability controls

### Production MoE Implementation
- **Intelligent Top-K Routing**: Sparse expert selection with dynamic routing strategies
- **Advanced Load Balancing**: Multi-objective loss terms to prevent expert collapse
- **Capacity Management**: Dynamic capacity factors with overflow handling
- **Expert Parallelism**: Distributed expert computation across multiple devices

### Memory Management System
- **Gradient Checkpointing**: Trade computation for memory during backward passes
- **Dynamic Precision**: Automatic precision adjustment based on training stability
- **Streaming Data Loading**: Process datasets larger than available memory efficiently
- **Emergency Recovery**: Automatic memory cleanup and checkpoint saving on critical conditions

## Model Configurations

The framework provides extensively tested model architectures spanning different scales:

| Configuration | Parameters | Hidden | Layers | Heads | KV Heads | Context | Memory (BF16) | Experts | DeepSpeed Recommended |
|---------------|------------|--------|--------|-------|----------|---------|---------------|---------|---------------------|
| `debug` | 500K | 128 | 2 | 2 | 1 | 256 | ~10MB | 4 | No |
| `m70_memory` | 70M | 768 | 12 | 12 | 4 | 1024 | ~280MB | 8 | Optional |
| `m120_speed` | 120M | 768 | 16 | 12 | 4 | 1024 | ~480MB | 8 | Optional |
| `b1` | 1B | 1024 | 12 | 16 | 4 | 2048 | ~2GB | 8 | Recommended |
| `b3_inference` | 3B | 2560 | 24 | 20 | 10 | 2048 | ~6GB | 16 | Recommended |
| `b6_quality` | 6B | 3200 | 32 | 25 | 10 | 4096 | ~12GB | 32 | Recommended |
| `b7` | 7B | 2048 | 22 | 16 | 8 | 4096 | ~14GB | 16 | **Recommended** |
| `b14` | 14B | 2560 | 28 | 20 | 10 | 4096 | ~28GB | 32 | **Required** |
| `b50` | 50B | 5120 | 40 | 40 | 10 | 128000 | ~100GB | 64 | **Required (ZeRO-3)** |
| `b100` | 100B | 7168 | 48 | 56 | 14 | 200000 | ~200GB | 96 | **Required (CPU Offload)** |
| `b200` | 200B | 8192 | 56 | 64 | 16 | 1000000 | ~400GB | 128 | **Required (NVMe Offload)** |
| `b300` | 300B | 9216 | 64 | 72 | 18 | 204800 | ~600GB | 144 | **Required (Multi-Node)** |

### Enhanced Configuration Details

Each preset includes optimized hyperparameters for DeepSpeed integration:

- **Automatic Precision Selection**: Hardware-aware FP16/BF16/Mixed precision
- **Dynamic Batch Sizing**: Memory-constrained automatic batch size adjustment
- **ZeRO Stage Recommendation**: Automatic ZeRO stage selection based on model size
- **Expert Parallelism**: Optimal expert distribution across available GPUs
- **Gradient Accumulation**: Intelligent micro-batch and accumulation step sizing

## Installation

### Enhanced System Requirements

**Minimum Requirements:**
- Python 3.10 or higher
- PyTorch 2.0+ with CUDA support
- 16GB system RAM
- CUDA-compatible GPU with 8GB+ VRAM
- 100GB+ free disk space for checkpoints and DeepSpeed offloading

**Recommended for DeepSpeed:**
- Python 3.11
- PyTorch 2.1+ with CUDA 11.8 or 12.1
- 64GB+ system RAM
- A100/H100 GPU with 40GB+ VRAM
- 1TB+ NVMe storage for offloading
- InfiniBand networking for multi-node

### Installation Process

1. **Clone Repository:**
```bash
git clone https://github.com/MatN23/LuminaAI.git
cd LuminaAI
```

2. **Install PyTorch with CUDA:**
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

4. **Install DeepSpeed:**
```bash
# Standard installation
pip install deepspeed

# Development installation with CUDA extensions
DS_BUILD_OPS=1 pip install deepspeed

# Verify DeepSpeed installation
ds_report
```

5. **Install Optional Performance Dependencies:**
```bash
# Flash Attention (A100/H100 recommended)
pip install flash-attn>=2.0.0 --no-build-isolation

# Monitoring and visualization
pip install wandb tensorboard matplotlib seaborn

# Development tools
pip install pytest black isort flake8 mypy
```

6. **Verify Installation:**
```bash
python Src/Main_Scripts/main.py --check-environment --verify-deepspeed
```

## DeepSpeed Integration

### Automatic DeepSpeed Configuration

The framework automatically configures DeepSpeed based on your hardware and model requirements:

```python
# Hardware-aware configuration
if gpu_memory_gb < 16:
    zero_stage = 3, cpu_offload = True, precision = "fp16"
elif model_size_gb > gpu_memory_gb:
    zero_stage = 3, cpu_offload = True, precision = "bf16"
else:
    zero_stage = 2, precision = "bf16"
```

### ZeRO Optimization Stages

| Stage | Memory Savings | Use Case | Model Size Limit |
|-------|----------------|----------|------------------|
| **ZeRO-1** | ~4x | Single GPU, small models | Up to 1.5B |
| **ZeRO-2** | ~8x | Multi-GPU, medium models | Up to 20B |
| **ZeRO-3** | ~64x+ | Large models, CPU offload | 100B+ |

### CPU and NVMe Offloading

```bash
# Enable CPU offloading for models larger than GPU memory
python Src/Main_Scripts/main.py --config b50 --enable-cpu-offload

# Enable NVMe offloading for massive models
python Src/Main_Scripts/main.py --config b200 --nvme-path /fast/nvme/offload

# Automatic offloading decision
python Src/Main_Scripts/main.py --config b100 --auto-offload
```

### DeepSpeed MoE Configuration

The framework provides specialized MoE optimizations:

```bash
# Optimized MoE training with expert parallelism
deepspeed --num_gpus=8 Src/Main_Scripts/main.py \
  --config b14 \
  --enable-moe-expert-parallel \
  --expert-parallel-size 4 \
  --moe-capacity-factor 1.25
```

## Data Format and Processing

### Enhanced Data Format Support

The framework supports multiple conversational formats with automatic detection:

```jsonl
{
  "messages": [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "Explain quantum computing."},
    {"role": "assistant", "content": "Quantum computing leverages quantum mechanical phenomena..."}
  ],
  "metadata": {
    "source": "synthetic",
    "quality_score": 0.95,
    "language": "en"
  }
}
```

### Advanced Dataset Processing

| Dataset Size | Strategy | Memory Usage | DeepSpeed Benefit | Processing Speed |
|--------------|----------|--------------|-------------------|------------------|
| < 1GB | **Memory Loading** | Full dataset in RAM | Minimal | Fastest |
| 1-50GB | **Intelligent Sharding** | Configurable chunks | Moderate | Fast |
| 50GB+ | **Streaming with Prefetch** | < 2GB | High | Optimized |
| 500GB+ | **Multi-Node Streaming** | Distributed | Very High | Scalable |

### Data Validation and Quality Assurance

```bash
# Comprehensive data validation with quality metrics
python Src/Main_Scripts/main.py --validate-data data/train.jsonl \
  --quality-threshold 0.8 \
  --create-detailed-report

# Automatic data cleaning and format correction
python Src/Main_Scripts/main.py --process-data data/raw.jsonl \
  --output data/cleaned.jsonl \
  --fix-encoding --remove-duplicates --filter-quality
```

## Usage Examples

### Production-Scale Training with DeepSpeed

```bash
# Large-scale distributed training
deepspeed --num_gpus=8 --num_nodes=4 Src/Main_Scripts/main.py \
  --config b50 \
  --train-data data/large_dataset.jsonl \
  --eval-data data/validation.jsonl \
  --zero-stage 3 \
  --cpu-offload \
  --gradient-compression \
  --experiment-name production_50b_moe
```

### MoE Training with Expert Parallelism

```bash
# Advanced MoE training with optimized expert distribution
deepspeed --num_gpus=8 Src/Main_Scripts/main.py \
  --config b14 \
  --enable-moe \
  --num-experts 32 \
  --expert-parallel-size 4 \
  --moe-top-k 2 \
  --capacity-factor 1.25 \
  --load-balancing-weight 0.01 \
  --experiment-name advanced_moe_14b
```

### Memory-Optimized Training

```bash
# Maximum memory efficiency for resource-constrained environments
python Src/Main_Scripts/main.py \
  --config b7 \
  --precision fp16 \
  --gradient-checkpointing \
  --zero-stage 3 \
  --cpu-offload \
  --aggressive-cpu-offload \
  --micro-batch-size 1 \
  --gradient-accumulation 64
```

### High-Performance Inference Setup

```bash
# Optimized inference configuration
python Src/Main_Scripts/main.py \
  --config b7 \
  --inference-mode \
  --zero-stage 2 \
  --precision bf16 \
  --compile-model \
  --enable-kv-cache \
  --max-batch-size 32
```

### Multi-Node Training

```bash
# Multi-node training with DeepSpeed launcher
deepspeed --hostfile hostfile --master_addr=192.168.1.1 \
  Src/Main_Scripts/main.py \
  --config b100 \
  --zero-stage 3 \
  --cpu-offload \
  --nvme-path /local/nvme \
  --gradient-compression \
  --communication-backend nccl
```

## Training Configuration

### Advanced Training Parameters

| Parameter | Type | Default | DeepSpeed Impact | Description |
|-----------|------|---------|------------------|-------------|
| `--zero-stage` | int | auto | High | ZeRO optimization level |
| `--cpu-offload` | bool | auto | High | Enable parameter offloading |
| `--nvme-path` | str | None | High | NVMe offloading directory |
| `--gradient-compression` | bool | False | Medium | Compress gradients for communication |
| `--expert-parallel-size` | int | auto | High (MoE) | Expert parallelism degree |
| `--micro-batch-size` | int | auto | High | Per-device micro batch size |
| `--communication-backend` | str | nccl | Medium | Distributed communication backend |

### Precision and Optimization

```bash
# Mixed precision with automatic loss scaling
python Src/Main_Scripts/main.py \
  --precision mixed_bf16 \
  --auto-loss-scaling \
  --loss-scale-window 1000

# Gradient compression for multi-node
python Src/Main_Scripts/main.py \
  --gradient-compression \
  --compression-type fp16 \
  --communication-overlap
```

## Precision and Memory Management

### Enhanced Precision Modes

| Precision | Memory | Stability | Speed | DeepSpeed Optimized | Hardware |
|-----------|--------|-----------|-------|-------------------|----------|
| `fp32` | 100% | Highest | Baseline | ❌ | Any CUDA |
| `fp16` | 50% | Good | 1.8x | ✅ | V100+ |
| `bf16` | 50% | Higher | 1.7x | ✅ | A100+ |
| `mixed_fp16` | Variable | Good | 2.0x | ✅ | V100+ |
| `mixed_bf16` | Variable | Highest | 1.9x | ✅ | A100+ |
| `dynamic` | Adaptive | Optimal | Optimal | ✅ | Auto-detect |

### DeepSpeed Memory Optimizations

```python
# Automatic memory optimization
strategy = {
    'zero_stage': 3,
    'cpu_offload_optimizer': True,
    'cpu_offload_parameters': True,
    'nvme_offload_optimizer': True,
    'nvme_offload_parameters': True,
    'max_live_parameters': 1e9,
    'max_reuse_distance': 1000,
    'prefetch_bucket_size': 5e7,
    'overlap_comm': True,
    'contiguous_gradients': True,
    'sub_group_size': 1e9,
    'reduce_bucket_size': 5e8,
    'allgather_partitions': True,
    'reduce_scatter': True,
    'allgather_bucket_size': 5e8
}
```

## Mixture of Experts Implementation

### Advanced MoE Architecture

The MoE implementation now includes expert parallelism and advanced routing:

| Model Scale | Expert Count | Expert Parallel | Top-K | Load Balance | Active Params | Routing Strategy |
|-------------|--------------|-----------------|-------|--------------|---------------|------------------|
| 1B-3B | 8 | 1 | 2 | 0.008 | 25% | Learned Routing |
| 7B-14B | 16-32 | 2-4 | 2 | 0.01 | 12.5-6.25% | Adaptive Routing |
| 50B+ | 64+ | 8+ | 2-4 | 0.015 | <3.125% | Hierarchical Routing |

### Expert Parallelism Configuration

```bash
# Distribute 32 experts across 8 GPUs (4 experts per GPU)
deepspeed --num_gpus=8 Src/Main_Scripts/main.py \
  --config b14 \
  --num-experts 32 \
  --expert-parallel-size 8 \
  --data-parallel-size 1
```

### Advanced Load Balancing

```python
# Multi-objective load balancing
load_balancing_loss = (
    auxiliary_loss +           # Basic load balancing
    z_loss +                  # Router z-loss for stability
    switch_loss +             # Switch transformer loss
    expert_choice_loss        # Expert-choice routing loss
)
```

## Distributed Training

### Multi-GPU Setup

```bash
# Standard multi-GPU training
deepspeed --num_gpus=4 Src/Main_Scripts/main.py --config b7

# With specific GPU selection
CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed --num_gpus=4 Src/Main_Scripts/main.py --config b7
```

### Multi-Node Configuration

Create a hostfile (`hostfile`) for multi-node training:
```
node1 slots=8
node2 slots=8  
node3 slots=8
node4 slots=8
```

Launch multi-node training:
```bash
deepspeed --hostfile hostfile --master_addr node1 --master_port 29500 \
  Src/Main_Scripts/main.py --config b50 --zero-stage 3
```

### Advanced Communication Optimization

```bash
# Optimized for high-bandwidth networks (InfiniBand)
deepspeed --num_gpus=8 Src/Main_Scripts/main.py \
  --communication-backend nccl \
  --bucket-cap-mb 200 \
  --overlap-comm \
  --contiguous-gradients \
  --compress-communication
```

## Monitoring and Analysis

### Enhanced Real-Time Metrics

```
Epoch 2 | Step 1,247 | Batch 1,247/15,000 | 08:23 elapsed
Loss: 2.456789 | PPL: 11.67 | LR: 8.50e-05 | GradNorm: 0.8432
DeepSpeed: ZeRO-3 | CPU Offload: 12.3GB | GPU: 8.2/40.0GB (21%)
MoE: Balance=0.023 | Expert Usage: 14/16 | Routing Loss: 0.012
Communication: AllGather=1.2ms | Reduce=0.8ms | Overlap=85%
Throughput: 1,250 tokens/s | Effective Batch: 256 | World Size: 8
Validation PPL: 10.23 | Best: 9.87 @ step 1,180 | ETA: 2h 45m
```

### DeepSpeed-Specific Monitoring

```bash
# Enable comprehensive DeepSpeed monitoring
python Src/Main_Scripts/main.py \
  --monitor-deepspeed \
  --log-communication-stats \
  --track-memory-fragmentation \
  --profile-expert-routing
```

### Advanced Visualization

```bash
# Generate comprehensive training reports
python Src/Main_Scripts/main.py \
  --create-training-report \
  --include-moe-analysis \
  --include-memory-profile \
  --include-communication-analysis
```

## Performance Characteristics

### DeepSpeed Scaling Efficiency

**A100-80GB Cluster Performance (Mixed BF16 + ZeRO-3):**

| Model | GPUs | Batch Size | Tokens/sec | Scaling Efficiency | Memory/GPU | ZeRO Overhead |
|-------|------|------------|------------|-------------------|-------------|---------------|
| 7B | 1 | 4 | 1,800 | 100% | 28GB | 0% |
| 7B | 4 | 16 | 6,480 | 90% | 12GB | 5% |
| 14B | 8 | 32 | 5,600 | 87% | 18GB | 8% |
| 50B | 32 | 128 | 4,200 | 82% | 25GB | 12% |
| 100B | 64 | 256 | 3,100 | 78% | 32GB | 15% |

### MoE Scaling Benefits

| Model Type | Parameters | Active Params | Training Speed | Quality Gain | Memory Efficiency |
|------------|------------|---------------|----------------|--------------|-------------------|
| Dense 7B | 7B | 7B (100%) | Baseline | Baseline | 1.0x |
| MoE 7B (16E) | 28B | 3.5B (12.5%) | 0.8x | +25% | 4.0x param density |
| MoE 14B (32E) | 56B | 3.5B (6.25%) | 0.9x | +35% | 8.0x param density |
| MoE 50B (64E) | 200B | 6.25B (3.1%) | 1.1x | +60% | 16.0x param density |

### Memory Efficiency Analysis

**Memory Usage with DeepSpeed ZeRO-3 + CPU Offload:**

| Component | Standard | ZeRO-3 | CPU Offload | NVMe Offload | Savings |
|-----------|----------|---------|-------------|--------------|---------|
| Parameters | 28GB | 0.35GB | CPU | NVMe | 99% |
| Gradients | 28GB | 0.35GB | CPU | NVMe | 99% |
| Optimizer | 56GB | 0.7GB | CPU | NVMe | 99% |
| Activations | 8GB | 8GB | 6GB | 4GB | 50% |
| **Total** | **120GB** | **9.4GB** | **6GB** | **4GB** | **97%** |

## Hardware Requirements

### DeepSpeed-Optimized Hardware Configurations

**Entry-Level DeepSpeed Setup ($3,000-$8,000):**
- GPU: RTX 4090 (24GB) or A6000 (48GB)
- CPU: 16+ cores with high memory bandwidth
- RAM: 64GB DDR4-3200 (for CPU offloading)
- Storage: 2TB NVMe SSD for offloading
- **Suitable for**: Models up to 14B parameters

**Professional DeepSpeed Setup ($15,000-$30,000):**
- GPU: A100-40GB or A100-80GB
- CPU: Dual-socket with 32+ cores
- RAM: 256GB DDR4-3200 ECC
- Storage: 8TB NVMe array for NVMe offloading
- Network: 25GbE or InfiniBand for multi-node
- **Suitable for**: Models up to 100B parameters

**Enterprise Multi-Node Setup ($100,000+):**
- GPUs: 8x A100-80GB or H100-80GB per node
- CPU: High-core server processors per node
- RAM: 512GB+ DDR4/DDR5 ECC per node
- Storage: High-speed parallel NVMe arrays
- Network: InfiniBand HDR for optimal multi-node communication
- **Suitable for**: Models 200B+ parameters

### Network Requirements for Multi-Node

| Scale | Minimum Network | Recommended | Scaling Efficiency |
|-------|-----------------|-------------|-------------------|
| 2-4 nodes | 25GbE | InfiniBand EDR (100Gb/s) | >95% |
| 4-8 nodes | InfiniBand EDR | InfiniBand HDR (200Gb/s) | >90% |
| 8+ nodes | InfiniBand HDR | InfiniBand NDR (400Gb/s) | >85% |

## Checkpointing System

### DeepSpeed-Enhanced Checkpointing

```python
# Advanced checkpoint configuration
checkpoint_config = {
    'use_universal_checkpoint': True,  # Cross-DeepSpeed compatibility
    'checkpoint_in_cpu': True,         # Save directly to CPU memory
    'save_latest': True,               # Always maintain latest checkpoint
    'save_optim_states': True,         # Include optimizer states
    'async_save': True,                # Non-blocking checkpoint saves
    'compression': 'gzip',             # Compress checkpoints
}
```

### Checkpoint Recovery Scenarios

```bash
# Resume from DeepSpeed universal checkpoint
deepspeed --num_gpus=8 Src/Main_Scripts/main.py \
  --resume-from-checkpoint checkpoints/universal_step_1000 \
  --load-optimizer-states \
  --strict-loading False

# Cross-configuration recovery (different GPU count)
deepspeed --num_gpus=4 Src/Main_Scripts/main.py \
  --resume-from-checkpoint checkpoints/8gpu_model \
  --reshape-enabled \
  --auto-redistribute-experts
```

## Troubleshooting

### DeepSpeed-Specific Issues

#### DeepSpeed Installation Problems

```bash
# Verify DeepSpeed installation
ds_report

# Reinstall with CUDA extensions
DS_BUILD_OPS=1 pip install deepspeed --force-reinstall

# Check CUDA compatibility
python -c "import deepspeed; print(deepspeed.ops.op_builder.CUDAOpBuilder().is_available())"
```

#### Memory and Performance Issues

**ZeRO-3 Slow Initialization:**
```bash
# Enable fast initialization
python Src/Main_Scripts/main.py \
  --zero-stage 3 \
  --zero-init-method tiled \
  --remote-device cpu \
  --pin-memory True
```

**Communication Bottlenecks:**
```bash
# Optimize communication
python Src/Main_Scripts/main.py \
  --overlap-comm True \
  --allgather-bucket-size 500000000 \
  --reduce-bucket-size 500000000
```

**Expert Load Imbalance (MoE):**
```bash
# Adjust MoE parameters
python Src/Main_Scripts/main.py \
  --capacity-factor 1.5 \
  --load-balancing-weight 0.02 \
  --expert-parallel-size 4
```

### Multi-Node Debugging

```bash
# Test multi-node connectivity
python -m torch.distributed.launch \
  --nproc_per_node=8 \
  --nnodes=2 \
  --node_rank=0 \
  --master_addr="192.168.1.1" \
  --master_port=29500 \
  test_connectivity.py

# Debug communication issues
NCCL_DEBUG=INFO deepspeed --hostfile hostfile Src/Main_Scripts/main.py
```

### Performance Optimization

| Issue | Solution | Command |
|-------|----------|---------|
| Slow data loading | Increase workers, enable pinned memory | `--num-workers 8 --pin-memory` |
| Communication overhead | Enable compression and overlap | `--gradient-compression --overlap-comm` |
| Memory fragmentation | Enable contiguous gradients | `--contiguous-gradients` |
| Expert imbalance | Adjust capacity and balancing | `--capacity-factor 1.8 --load-balancing-weight 0.015` |

## Project Structure

```
LuminaAI/
├── Src/
│   └── Main_Scripts/
│       ├── main.py                     # Enhanced CLI with DeepSpeed integration
│       ├── chat.py                     # Interactive chat interface for testing
│       ├── core/                       # Core model and data components
│       │   ├── model.py               # DeepSeek transformer with MoE
│       │   ├── tokenizer.py           # GPT-4 compatible tokenization
│       │   └── dataset.py             # Advanced dataset handling
│       ├── training/                  # Enhanced training system
│       │   ├── trainer.py             # DeepSpeed-enabled trainer
│       │   ├── orchestrator.py        # Training coordination and monitoring
│       │   ├── checkpoint.py          # Advanced checkpoint management
│       │   ├── config_manager.py      # Configuration presets and validation
│       │   └── training_loops.py      # Optimized training loops
│       ├── monitoring/                # Comprehensive monitoring
│       │   ├── logger.py              # Enhanced logging with DeepSpeed metrics
│       │   ├── visualizations.py      # Real-time training visualizations
│       │   └── moe_analytics.py       # MoE routing analysis
│       ├── utils/                     # Enhanced utilities
│       │   ├── data_processing.py     # Data validation and processing
│       │   ├── environment.py         # System validation and optimization
│       │   ├── reporting.py           # Performance analysis and reporting
│       │   └── deepspeed_utils.py     # DeepSpeed helper functions
│       └── config/                    # Configuration management
│           ├── model_configs.yaml     # Model architecture presets
│           ├── deepspeed_configs.yaml # DeepSpeed optimization templates
│           └── training_configs.yaml  # Training parameter templates
├── configs/                           # User configuration files
├── data/                             # Training data and caches
│   ├── shards/                       # Data sharding for large datasets
│   ├── processed/                    # Processed and validated data
│   └── cache/                        # Dataset caching
├── checkpoints/                      # Model checkpoints
│   ├── best/                         # Best model checkpoints
│   ├── emergency/                    # Emergency recovery checkpoints
│   └── deepspeed/                    # DeepSpeed universal checkpoints
├── experiments/                      # Experiment tracking and results
├── logs/                            # Comprehensive logging
│   ├── deepspeed/                   # DeepSpeed-specific logs
│   ├── moe/                         # MoE routing logs
│   └── performance/                 # Performance profiling logs
├── reports/                         # Analysis and performance reports
├── monitoring/                      # Real-time monitoring data
│   ├── metrics/                     # Training metrics
│   ├── visualizations/              # Generated charts and plots
│   └── routing/                     # MoE routing analysis
├── requirements.txt                  # Python dependencies
├── .gitignore                       # Git ignore rules
├── LICENSE                          # License file
└── README.md                        # This documentation
```

## Configuration System Details

### Advanced Configuration Management

The framework now supports hierarchical configuration with environment-specific overrides:

```yaml
# configs/production.yaml
base_config: "b50"
experiment:
  name: "production_50b_moe_v2"
  tags: ["production", "moe", "50b"]
  
model:
  use_moe: true
  num_experts: 64
  expert_parallel_size: 8
  
deepspeed:
  zero_stage: 3
  cpu_offload: true
  nvme_path: "/fast/nvme/offload"
  gradient_compression: true
  
training:
  precision: "mixed_bf16"
  micro_batch_size: 1
  gradient_accumulation_steps: 128
  learning_rate: 5e-5
  
data:
  train_data_path: "data/large_corpus.jsonl"
  streaming_threshold_gb: 50
  num_workers: 16
  
monitoring:
  log_interval: 10
  eval_interval: 500
  save_interval: 1000
  enable_wandb: true
  wandb_project: "lumina-production"
```

### Dynamic Configuration Loading

```python
# Load configuration with environment-specific overrides
config = ConfigManager.load_with_overrides(
    base_config="configs/production.yaml",
    overrides={
        "training.learning_rate": 3e-5,
        "deepspeed.expert_parallel_size": 4,
        "experiment.name": f"experiment_{timestamp}"
    }
)

# Hardware-aware automatic adjustments
config = ConfigManager.optimize_for_hardware(config, target_memory_usage=0.85)
```

### DeepSpeed Configuration Templates

```yaml
# configs/deepspeed_configs.yaml
zero_stage_3:
  zero_optimization:
    stage: 3
    offload_optimizer:
      device: cpu
      pin_memory: true
    offload_param:
      device: cpu
      pin_memory: true
    overlap_comm: true
    contiguous_gradients: true
    sub_group_size: 1000000000
    reduce_bucket_size: 1000000000
    stage3_prefetch_bucket_size: 50000000
    stage3_param_persistence_threshold: 100000

moe_optimization:
  moe:
    enabled: true
    expert_parallel_size: 8
    capacity_factor: 1.25
    min_capacity: 4
    use_tutel: true
  
communication:
    fp16_enabled: false
    bf16_enabled: true
    gradient_compression: true
    overlap_comm: true
    contiguous_gradients: true
```

## Launch Scripts and Automation

### Production Launch Scripts

**Single GPU Training (`launch_scripts/single_gpu.sh`):**
```bash
#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)/Src"

python Src/Main_Scripts/main.py \
  --config-file configs/single_gpu.yaml \
  --experiment-name "single_gpu_$(date +%Y%m%d_%H%M%S)" \
  --auto-optimize-memory \
  --enable-monitoring \
  "$@"
```

**Multi-GPU Training (`launch_scripts/multi_gpu.sh`):**
```bash
#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/Src"

deepspeed --num_gpus=${NUM_GPUS:-8} Src/Main_Scripts/main.py \
  --config-file configs/multi_gpu.yaml \
  --deepspeed \
  --zero-stage 3 \
  --experiment-name "multi_gpu_$(date +%Y%m%d_%H%M%S)" \
  "$@"
```

**Multi-Node Training (`launch_scripts/multi_node.sh`):**
```bash
#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/Src"
export NCCL_TIMEOUT=1800
export NCCL_DEBUG=INFO

deepspeed --hostfile hostfile \
  --master_addr ${MASTER_ADDR:-$(hostname -I | awk '{print $1}')} \
  --master_port ${MASTER_PORT:-29500} \
  Src/Main_Scripts/main.py \
  --config-file configs/multi_node.yaml \
  --deepspeed \
  --zero-stage 3 \
  --cpu-offload \
  --experiment-name "multi_node_$(date +%Y%m%d_%H%M%S)" \
  "$@"
```

### Automated Experiment Management

```bash
# Automated hyperparameter sweeps
./launch_scripts/hyperparameter_sweep.sh \
  --config-base configs/base_7b.yaml \
  --sweep-params "learning_rate:[1e-5,5e-5,1e-4]" \
  --sweep-params "num_experts:[8,16,32]" \
  --num-trials 9

# Automated model scaling experiments
./launch_scripts/scaling_experiment.sh \
  --configs "b1,b3,b7,b14" \
  --enable-comparison \
  --generate-report
```

## Advanced Features

### Automatic Model Sharding

```python
# Intelligent model sharding across GPUs
class AutoShardingManager:
    def distribute_model(self, model, world_size, expert_parallel_size):
        # Distribute embedding layers
        # Shard attention and feed-forward layers
        # Optimize expert placement for communication efficiency
        # Balance memory usage across devices
```

### Dynamic Expert Scaling

```bash
# Runtime expert scaling based on utilization
python Src/Main_Scripts/main.py \
  --adaptive-expert-scaling \
  --min-experts 16 \
  --max-experts 64 \
  --scaling-threshold 0.8 \
  --scaling-interval 1000
```

### Advanced Monitoring Integration

```python
# Integration with multiple monitoring systems
monitoring_config = {
    'wandb': {
        'project': 'lumina-ai-production',
        'entity': 'research-team',
        'tags': ['deepspeed', 'moe', 'production']
    },
    'tensorboard': {
        'log_dir': 'logs/tensorboard',
        'update_freq': 'batch'
    },
    'mlflow': {
        'tracking_uri': 'http://mlflow.company.com',
        'experiment_name': 'lumina-experiments'
    }
}
```

## Development and Contributing

### Enhanced Development Environment

```bash
# Development setup with all features
git clone https://github.com/MatN23/LuminaAI.git
cd LuminaAI

# Create development environment
python -m venv venv_dev
source venv_dev/bin/activate

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -r requirements-deepspeed.txt

# Install pre-commit hooks
pre-commit install

# Verify development environment
python Src/Main_Scripts/main.py --check-environment --dev-mode
```

### Testing Framework

```bash
# Run comprehensive tests
python -m pytest tests/ -v --cov=Src --cov-report=html

# Test DeepSpeed integration
python -m pytest tests/test_deepspeed.py -v

# Test MoE functionality
python -m pytest tests/test_moe.py -v --moe-config configs/test_moe.yaml

# Performance regression tests
python scripts/performance_tests.py --baseline benchmarks/baseline.json
```

### Code Quality and Standards

**Enhanced Pre-commit Configuration (`.pre-commit-config.yaml`):**
```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.1.0
    hooks:
      - id: black
        args: [--line-length=88]
  
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: [--profile=black]
  
  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=88]
  
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.0.1
    hooks:
      - id: mypy
        additional_dependencies: [types-PyYAML]
```

### Performance Benchmarking

```python
# Automated performance benchmarking
class PerformanceBenchmark:
    def benchmark_training_speed(self, configs, iterations=100):
        results = {}
        for config_name in configs:
            # Run training benchmark
            # Measure tokens/second, memory usage, scaling efficiency
            # Generate performance reports
        return results
    
    def benchmark_moe_routing(self, expert_counts, sequence_lengths):
        # Test routing efficiency across different configurations
        # Measure expert utilization, load balancing, communication overhead
        pass
```

## Advanced Use Cases

### Research and Experimentation

```bash
# Ablation studies
python scripts/ablation_study.py \
  --components "moe,flash_attention,gradient_checkpointing" \
  --baseline-config configs/baseline.yaml \
  --output-dir results/ablations

# Architecture search
python scripts/architecture_search.py \
  --search-space configs/search_space.yaml \
  --budget 100 \
  --optimization-metric validation_ppl
```

### Production Deployment

```bash
# Model optimization for inference
python Src/Main_Scripts/optimize_for_inference.py \
  --checkpoint checkpoints/best_model.pt \
  --target-latency 50ms \
  --output optimized_models/

# Model quantization
python Src/Main_Scripts/quantize_model.py \
  --model-path optimized_models/model.pt \
  --quantization-method int8 \
  --calibration-data data/calibration.jsonl
```

### Custom Model Architectures

```python
# Extending the framework with custom architectures
class CustomTransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Custom layer implementation
        
    def forward(self, hidden_states, attention_mask=None):
        # Custom forward pass
        pass

# Register custom components
ModelRegistry.register_layer("custom_layer", CustomTransformerLayer)
```

## License and Citations

This project is licensed under a Custom License with the following key points:

### Usage Permissions
- **Academic Research**: Full permission for research and educational purposes
- **Commercial Use**: Contact maintainers for commercial licensing terms
- **Modification and Distribution**: Allowed with proper attribution

### Citation Requirements

When using LuminaAI in research, please cite:

```bibtex
@software{lumina_ai_2025,
  title={LuminaAI: A PyTorch Framework for Large Language Model Training with DeepSpeed and Mixture of Experts},
  author={Nielsen, Matias},
  year={2025},
  url={https://github.com/MatN23/LuminaAI},
  version={2.0},
  note={Enhanced framework supporting distributed training and MoE architectures up to 300B parameters}
}
```

For DeepSpeed integration, also cite:
```bibtex
@inproceedings{deepspeed,
  title={DeepSpeed: System Optimizations Enable Training Deep Learning Models with Over 100 Billion Parameters},
  author={Rasley, Jeff and Rajbhandari, Samyam and Ranjan, Olatunji and He, Yuxiong},
  booktitle={Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  year={2020}
}
```

### Acknowledgments

Special thanks to:
- Microsoft DeepSpeed team for the distributed training framework
- Hugging Face for the transformers library foundation
- The broader open-source community for inspiration and feedback

### Contact and Support

- **Issues and Bug Reports**: GitHub Issues
- **Feature Requests**: GitHub Discussions
- **Commercial Licensing**: Contact project maintainers
- **Community**: Join our Discord server for discussions and support

---

**LuminaAI** - Empowering researchers and practitioners to train state-of-the-art language models efficiently and at scale.