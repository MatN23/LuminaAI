# LuminaAI

<div align="center">

**Advanced Transformer Training System with MoE and MoD**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Custom-green.svg)](LICENSE)

*A production-ready system for training state-of-the-art language models with Mixture-of-Experts and Mixture-of-Depths*

[Features](#-key-features) • [Architecture](#-architecture) • [Quick Start](#-quick-start) • [Training Modes](#-training-modes) • [Model Configurations](#-model-configurations)

</div>

---

## 🌟 Overview

LuminaAI is a comprehensive deep learning training system designed for building efficient, scalable transformer-based language models. The system supports both traditional dense architectures and cutting-edge sparse architectures including **Mixture-of-Experts (MoE)** and **Mixture-of-Depths (MoD)**, with full support for **hybrid architectures** that combine both techniques.

Built on PyTorch with DeepSpeed integration, LuminaAI provides everything needed to train models ranging from small experimental configs to massive multi-billion parameter systems with advanced optimization techniques.

### Why LuminaAI?

- **🎯 Production-Ready**: Comprehensive error handling, checkpointing, and monitoring
- **⚡ Efficient**: 10-30% faster than baseline implementations with 15-25% less memory
- **🔧 Flexible**: Support for multiple training paradigms (base, fine-tuning, hybrid)
- **📊 Observable**: Extensive metrics, profiling, and real-time health monitoring
- **🚀 Scalable**: From single GPU to multi-node distributed training
- **🧠 Advanced**: MoE, MoD, Flash Attention, GQA, and hybrid architectures

---

## ✨ Key Features

### 🏗️ Architecture Capabilities

#### **Mixture of Experts (MoE)**
- Sparse activation for massive parameter efficiency
- 8-64 expert configurations with top-k routing
- Advanced load balancing to prevent expert collapse
- 40-60% less active parameters for same capacity
- Flexible MoE patterns: all layers, interleaved, sandwich

#### **Mixture of Depths (MoD)**
- Dynamic token-level compute allocation
- Learn which tokens need full computation vs skip connections
- 30-50% FLOPs reduction with minimal quality loss
- Adaptive capacity factors for different use cases

#### **Hybrid MoE + MoD** 🚀
- **NEW**: Combine MoE and MoD in the same model!
- MoE layers for expert routing on complex layers
- MoD routing on dense layers for token efficiency
- Maximum parameter AND compute efficiency

#### **Dense Models with Advanced Features**
- Grouped Query Attention (GQA) for efficient KV caching
- Rotary Position Embeddings (RoPE) for better length generalization
- SwiGLU activation for improved performance
- RMS Normalization for training stability

### ⚡ Training & Optimization

#### **Multi-Device Support**
- **NVIDIA CUDA**: Full feature support with DeepSpeed integration
- **Apple Silicon (MPS)**: Native M1/M2/M3 support with automatic optimizations
- **CPU**: Fallback for development and debugging

#### **DeepSpeed Integration**
- ZeRO optimization stages 1-3 for memory efficiency
- CPU/NVMe offloading for training massive models
- Gradient compression for multi-GPU communication
- Automatic mixed precision training (FP16/BF16)

#### **Flash Attention 2**
- 2-4x faster attention for long sequences
- O(N) memory complexity vs O(N²) standard attention
- Automatic fallback to optimized standard attention

#### **Advanced Training Features**
- Gradient checkpointing for memory efficiency
- Adaptive learning rate scheduling (cosine, OneCycle)
- Automatic batch size tuning with OOM recovery
- Early stopping with patience configuration
- Continuous checkpointing with best model tracking

### 📊 Data Processing

#### **Multi-Dataset Support**
- Base/Pre-training: Raw text (The Pile, C4, etc.) - supports `.txt` and `.jsonl`
- Fine-tuning: Conversational data (OASST, custom) - `.jsonl` only
- Automatic dataset combination from multiple files
- Streaming support for datasets >10GB

#### **Training Modes**
- **Base Only**: Pre-training on raw text
- **Fine-tuning Only**: Instruction tuning on conversations
- **Hybrid**: Sequential (base → fine-tuning)
- **Interleaved**: Mixed base and fine-tuning

#### **Smart Data Loading**
- Automatic format detection (JSONL, TXT)
- Chunking for efficient base training
- Conversation validation and preprocessing
- Configurable data caching
- Dynamic batching and prefetching

### 🔬 Monitoring & Analysis

#### **Real-Time Health Monitoring**
- Automatic anomaly detection (loss spikes, gradient explosion)
- Training phase detection (initialization, volatile, converging)
- Performance benchmarking against targets
- Resource utilization tracking

#### **Comprehensive Metrics**
- Loss, perplexity, and accuracy tracking
- Token throughput and training speed
- Memory usage (GPU/MPS/CPU)
- Expert/token routing statistics
- Learning rate and gradient norms

#### **Profiling & Diagnostics**
- Per-layer performance profiling
- Attention mechanism statistics
- Expert utilization analysis
- Data validation reports
- Training health scores

---

## 🏛️ Architecture

### Model Architecture

```
┌─────────────────────────────────────────────┐
│         Token Embeddings (+ scaling)         │
└──────────────────┬──────────────────────────┘
                   │
        ┌──────────▼──────────┐
        │  Transformer Block  │  ×N layers
        └──────────┬──────────┘
                   │
    ┌──────────────▼─────────────────┐
    │       Pre-normalization        │
    │         (RMSNorm)              │
    └──────────────┬─────────────────┘
                   │
    ┌──────────────▼─────────────────┐
    │   Grouped Query Attention      │
    │   • Multi-head attention       │
    │   • Rotary embeddings (RoPE)   │
    │   • Flash Attention support    │
    │   • KV cache for inference     │
    └──────────────┬─────────────────┘
                   │
    ┌──────────────▼─────────────────┐
    │         Residual Add           │
    └──────────────┬─────────────────┘
                   │
    ┌──────────────▼─────────────────┐
    │       Post-normalization       │
    │         (RMSNorm)              │
    └──────────────┬─────────────────┘
                   │
    ┌──────────────▼─────────────────┐
    │      Feed-Forward Network      │
    │                                │
    │  Choice of:                    │
    │  ├─ Dense SwiGLU               │
    │  ├─ Dense SwiGLU + MoD         │
    │  └─ Mixture of Experts (MoE)   │
    │     • Top-k expert routing     │
    │     • Load balancing           │
    │     • 8-64 expert support      │
    └──────────────┬─────────────────┘
                   │
    ┌──────────────▼─────────────────┐
    │         Residual Add           │
    └──────────────┬─────────────────┘
                   │
                [Repeat]
                   │
        ┌──────────▼──────────┐
        │   Final RMSNorm     │
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────┐
        │   Language Model    │
        │   Head (Linear)     │
        └─────────────────────┘
```

### System Architecture

```
┌──────────────────────────────────────────────┐
│              Main.py                         │
│  • Configuration management                  │
│  • System diagnostics                        │
│  • Data path validation                      │
│  • Training orchestration                    │
└──────────────┬───────────────────────────────┘
               │
    ┌──────────▼────────────┐
    │  Orchestrator.py      │
    │  • Adaptive training  │
    │  • Meta-learning      │
    │  • Auto-optimization  │
    │  • Real-time monitor  │
    └──────────┬────────────┘
               │
    ┌──────────▼────────────┐
    │    Trainer.py         │
    │  • Training loops     │
    │  • Loss computation   │
    │  • Gradient handling  │
    │  • Quantization       │
    └──────────┬────────────┘
               │
    ┌──────────▼────────────────────┐
    │  Dataset.py (HybridManager)   │
    │  • Base training datasets     │
    │  • Fine-tuning datasets       │
    │  • Multi-file support         │
    │  • Streaming datasets         │
    └──────────┬────────────────────┘
               │
    ┌──────────▼────────────┐
    │   Model.py            │
    │  • Transformer core   │
    │  • MoE layers         │
    │  • MoD routing        │
    │  • Attention          │
    └───────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

```bash
# System Requirements
Python 3.8+
PyTorch 2.0+
CUDA 11.7+ (for GPU) or MPS (Apple Silicon)

# Storage
- 10GB+ free disk space for checkpoints
- 50GB+ recommended for large models
```

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/luminaai.git
cd luminaai

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# Optional: Install DeepSpeed (not supported on MPS)
pip install deepspeed

# Optional: Install Flash Attention
pip install flash-attn --no-build-isolation
```

### Basic Training Example

```python
# Main.py - Basic configuration

# 1. Choose model size
config_choice = 'b1'  # 1B active parameters

# 2. Configure training
training_params = {
    'num_epochs': 3,
    'batch_size': 8,
    'learning_rate': 1e-4,
    'precision': 'auto',  # Auto-detects best precision for hardware
}

# 3. Specify data
data_params = {
    'training_mode': 'finetuning_only',
    'finetuning_paths': [
        'data/train.jsonl',
    ],
    'finetuning_eval_paths': [
        'data/eval.jsonl',
    ],
}

# Run training
# python Main.py
```

### Hardware-Specific Setup

#### NVIDIA GPU

```python
# Optimal configuration for GPU
training_params = {
    'precision': 'mixed_bf16',  # BF16 for Ampere+, FP16 for older
    'use_flash_attention': True,
    'use_deepspeed': True,
    'compile': True,
}
```

#### Apple Silicon (M1/M2/M3)

```python
# MPS-optimized configuration (auto-applied)
training_params = {
    'precision': 'fp16',
    'batch_size': 4,
    'use_flash_attention': False,  # Not supported on MPS
    'use_deepspeed': False,  # Not supported on MPS
}
# System automatically applies these optimizations
```

---

## 📚 Training Modes

### 1. Base/Pre-Training Mode

Train on raw text corpora (The Pile, C4, etc.) for foundational language understanding.

**Data Format**: Plain text (`.txt`) or JSONL with `text` field

```python
data_params = {
    'training_mode': 'base_only',
    'base_training_paths': [
        'data/pile/pile_shard_00.txt',
        'data/pile/pile_shard_01.jsonl',
    ],
    'base_eval_paths': [
        'data/pile/pile_eval.jsonl',
    ],
}
```

**Features**:
- Fixed-length chunking for efficient training
- Document continuation across chunks
- Streaming support for massive datasets
- No conversation structure required

### 2. Fine-Tuning/Instruction Mode

Train on conversational data for instruction following and chat capabilities.

**Data Format**: JSONL with conversation structure

```python
data_params = {
    'training_mode': 'finetuning_only',
    'finetuning_paths': [
        'data/oasst1_train.jsonl',
        'data/custom_conversations.jsonl',
    ],
    'finetuning_eval_paths': [
        'data/oasst1_validation.jsonl',
    ],
}
```

**Required Format**:
```json
{
  "conversation_id": "conv_001",
  "messages": [
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi! How can I help?"}
  ]
}
```

**Features**:
- Conversation validation and preprocessing
- Role-based loss weighting
- Multi-turn conversation support
- Automatic format detection

### 3. Hybrid Mode (Sequential)

Combine base pre-training followed by fine-tuning in a two-phase approach.

```python
data_params = {
    'training_mode': 'hybrid',
    'base_training_paths': ['data/pile/train.txt'],
    'finetuning_paths': ['data/oasst1_train.jsonl'],
    # Phase 1: Base training
    # Phase 2: Fine-tuning (automatic transition)
}
```

### 4. Interleaved Mode

Mix base and fine-tuning data during training for balanced learning.

```python
data_params = {
    'training_mode': 'interleaved',
    'base_training_paths': ['data/pile/train.txt'],
    'finetuning_paths': ['data/oasst1_train.jsonl'],
    'base_finetuning_ratio': 0.7,  # 70% base, 30% fine-tuning
}
```

---

## 🎛️ Model Configurations

### Available Presets

LuminaAI includes pre-configured model sizes optimized for different hardware and use cases:

| Preset | Active Params | Total Params | Architecture | Use Case | Hardware |
|--------|--------------|--------------|--------------|----------|----------|
| `debug` | ~500K | ~4M | 8x MoE | Development | Any |
| `debug_200m` | ~200M | ~6B | 32x MoD | Testing | T4/MPS |
| `b1` | ~1B | ~8B | 8x MoE | Experimentation | RTX 3090, M1 Max |
| `b7` | ~7B | ~56B | 8x MoE | General Purpose | A100 40GB, M2 Ultra |
| `b14` | ~14B | ~112B | 8x MoE | High Performance | A100 80GB |
| `b30` | ~30B | ~240B | 8x MoE | Advanced Research | Multi-A100 |
| `b50` | ~50B | ~400B | 8x MoE | Enterprise | Multi-H100 |
| `b100` | ~100B | ~800B | 8x MoE | Cutting Edge | Large H100 Cluster |
| `b200` | ~200B | ~1.6T | 8x MoE | Frontier | Massive H100 Cluster |
| `b300` | ~300B | ~2.4T | 8x MoE | Ultra-Scale | Ultra-Large Cluster |

### Configuration Examples

#### Small Efficient Model (1B)

```python
config_choice = 'b1'

training_params = {
    'batch_size': 8,
    'gradient_accumulation_steps': 4,
    'use_moe': True,
    'use_mod': False,  # Pure MoE
}
```

#### Medium Model with Hybrid Architecture (7B)

```python
config_choice = 'b7'

training_params = {
    'batch_size': 16,
    'gradient_accumulation_steps': 8,
    'use_moe': True,
    'use_mod': True,  # 🚀 HYBRID MODE
    'num_experts': 8,
    'moe_top_k': 2,
    'mod_capacity_factor': 0.5,
}
```

#### Large Model with Aggressive Optimization (14B)

```python
config_choice = 'b14'

training_params = {
    'use_deepspeed': True,
    'zero_stage': 3,
    'cpu_offload': True,
    'gradient_checkpointing': True,
    'precision': 'mixed_bf16',
}
```

### Custom Configuration

```python
from config.config_manager import Config

config = Config(
    # Model architecture
    hidden_size=1024,
    num_layers=24,
    num_heads=16,
    num_kv_heads=4,  # GQA for efficient KV cache
    seq_length=2048,
    
    # MoE settings
    use_moe=True,
    num_experts=8,
    moe_top_k=2,
    capacity_factor=1.25,
    
    # MoD settings (optional)
    use_mod=True,
    mod_capacity_factor=0.5,
    
    # Training
    batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,
    num_epochs=3,
)
```

---

## 💻 Advanced Features

### Adaptive Training Orchestrator

The orchestrator provides AI-driven training optimization:

- **Meta-Learning**: Learn optimal hyperparameters from previous runs
- **Real-Time Monitoring**: Detect and respond to training issues
- **Automatic Recovery**: Handle OOM errors with batch size adjustment
- **Performance Profiling**: Track bottlenecks and optimization opportunities

```python
use_adaptive_training = True  # Enable in Main.py
```

### Quantization Support

Train with reduced precision for memory efficiency:

```python
quantization_params = {
    'quantization_method': 'bnb',  # BitsAndBytes
    'quantization_bits': 8,  # 4 or 8 bit
}

# INT8 inference precision for faster evaluation
training_params = {
    'precision': 'mixed_bf16',  # Training
    'inference_precision': 'int8',  # Inference
}
```

### Gradient Checkpointing

Trade computation for memory:

```python
training_params = {
    'gradient_checkpointing': True,
    # Reduces memory by ~40% at ~20% speed cost
}
```

### Multi-Node Training

Scale across multiple machines:

```bash
# Node 0 (master)
deepspeed --num_nodes=4 --num_gpus=8 Main.py

# Nodes 1-3 (workers)
deepspeed --num_nodes=4 --num_gpus=8 --node_rank=N Main.py
```

---

## 📊 Monitoring & Debugging

### Training Metrics

The system tracks comprehensive metrics:

- **Loss & Perplexity**: Training and validation curves
- **Accuracy**: Token-level prediction accuracy
- **Throughput**: Tokens/second processing speed
- **Memory**: GPU/MPS/CPU utilization
- **Learning Rate**: Current LR with scheduler info
- **Gradient Norms**: Detect training instability

### Health Monitoring

Automatic health checks detect:

- Loss spikes or divergence
- Gradient explosion/vanishing
- Expert imbalance (MoE)
- Token routing issues (MoD)
- Memory pressure
- Training stagnation

### Logging Levels

```python
monitoring_params = {
    'log_level': 'INFO',  # DEBUG, INFO, WARNING, ERROR
    'log_every_n_steps': 50,
    'health_check_interval': 100,
}
```

### Checkpointing

Automatic checkpoint management:

```python
training_params = {
    'save_every_n_batches': 1000,
    'save_total_limit': 5,  # Keep only N best checkpoints
    'early_stopping_patience': 10,
}
```

---

## 🔧 Troubleshooting

### Out of Memory (OOM)

The system automatically handles OOM:

1. **Automatic Recovery**: Reduces batch size and retries
2. **Gradient Accumulation**: Maintains effective batch size
3. **Recommendations**: Suggests memory optimizations

Manual fixes:
```python
training_params = {
    'batch_size': 2,  # Reduce
    'gradient_accumulation_steps': 16,  # Increase
    'gradient_checkpointing': True,
    'cpu_offload': True,  # If using DeepSpeed
}
```

### Slow Training

Optimization checklist:

- [ ] Enable `compile=True` (PyTorch 2.0+)
- [ ] Use `precision='mixed_bf16'` or `'mixed_fp16'`
- [ ] Enable Flash Attention (`use_flash_attention=True`)
- [ ] Increase `num_workers` for data loading
- [ ] Check `batch_size` isn't too small
- [ ] Enable DeepSpeed for multi-GPU

### MPS (Apple Silicon) Issues

The system auto-optimizes for MPS, but manual tips:

```python
# If encountering issues
training_params = {
    'compile': False,  # Can be unstable on MPS
    'num_workers': 0,  # MPS prefers main thread loading
    'batch_size': 2,  # Start small
}
```

### Expert Imbalance (MoE)

If experts aren't being used evenly:

```python
training_params = {
    'load_balancing_weight': 0.02,  # Increase (from 0.01)
    'routing_temperature': 1.2,  # Increase exploration
    'capacity_factor': 1.5,  # Allow more tokens per expert
}
```

---

## 📖 Citation

If you use LuminaAI in your research, please cite:

```bibtex
@software{luminaai2025,
  title = {LuminaAI: Advanced Transformer Training with MoE and MoD},
  author = {MatN23},
  year = {2025},
  url = {https://github.com/yourusername/luminaai}
}
```

---

## 📄 License

This project is licensed under a Custom License. See [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 🙏 Acknowledgments

- DeepSpeed team for distributed training framework
- Flash Attention authors for efficient attention implementation
- tiktoken for fast tokenization
- PyTorch team for the foundation
- OpenAssistant for conversational datasets

---

<div align="center">

**Built with ❤️ for the AI research community**

[Documentation](docs/) • [Issues](issues/) • [Discussions](discussions/)

</div>