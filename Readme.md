# LuminaAI: Advanced Transformer Training Framework

[![License](https://img.shields.io/badge/license-Custom-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0%2B-red.svg)](https://pytorch.org)
[![DeepSpeed](https://img.shields.io/badge/deepspeed-0.12%2B-green.svg)](https://github.com/microsoft/DeepSpeed)

**LuminaAI** is a production-grade framework for training large language models with advanced features including DeepSeek-style architectures, Mixture of Experts (MoE), DeepSpeed integration, and comprehensive monitoring. Built for researchers and practitioners who need state-of-the-art training capabilities with enterprise-level reliability.

## Table of Contents

- [Features](#features)
- [Architecture Overview](#architecture-overview)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Configuration](#configuration)
- [Training Models](#training-models)
- [Monitoring and Analytics](#monitoring-and-analytics)
- [Advanced Features](#advanced-features)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Performance Optimization](#performance-optimization)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Features

### Core Capabilities

- **🔥 DeepSeek-Style Transformers**: Advanced transformer architecture with optimized attention mechanisms
- **🧠 Mixture of Experts (MoE)**: Scalable MoE implementation with intelligent routing
- **⚡ DeepSpeed Integration**: Full DeepSpeed support with ZeRO optimization stages
- **🚀 Production Ready**: Enterprise-grade reliability with comprehensive error handling
- **📊 Advanced Monitoring**: Real-time metrics, health monitoring, and performance tracking
- **🎯 Adaptive Training**: Intelligent training orchestration with automated optimization
- **💾 Smart Checkpointing**: Universal checkpoint format with automatic recovery
- **🔧 Flexible Configuration**: YAML-based configuration with validation and presets

### Model Architectures

- **DeepSeek Transformers**: Optimized decoder-only architecture with RMSNorm and SwiGLU
- **Grouped Query Attention (GQA)**: Efficient attention with configurable KV heads
- **Rotary Position Embedding (RoPE)**: Advanced positional encoding for long sequences
- **Flash Attention**: Memory-efficient attention computation
- **MoE Integration**: Top-k routing with load balancing and capacity management

### Training Features

- **Multi-GPU Training**: Distributed training across multiple GPUs and nodes
- **Mixed Precision**: FP16, BF16, and dynamic precision support
- **Gradient Checkpointing**: Memory optimization for large models
- **CPU/NVMe Offloading**: Handle models larger than GPU memory
- **Automatic Mixed Precision**: Optimized training with loss scaling
- **Streaming Datasets**: Handle arbitrarily large datasets efficiently

## Architecture Overview

LuminaAI follows a modular architecture designed for scalability and maintainability:

```
LuminaAI/
├── Src/
│   └── Main_Scripts/
│       ├── main.py                 # Enhanced CLI with DeepSpeed integration
│       ├── chat.py                 # Interactive chat interface for testing
│       ├── core/                   # Core model and data components
│       │   ├── model.py           # DeepSeek transformer with MoE
│       │   ├── tokenizer.py       # GPT-4 compatible tokenization
│       │   └── dataset.py         # Advanced dataset handling
│       ├── training/              # Enhanced training system
│       │   ├── trainer.py         # DeepSpeed-enabled trainer
│       │   ├── orchestrator.py    # Training coordination and monitoring
│       │   ├── checkpoint.py      # Advanced checkpoint management
│       │   ├── config_manager.py  # Configuration presets and validation
│       │   └── training_loops.py  # Optimized training loops
│       ├── monitoring/            # Comprehensive monitoring
│       │   ├── logger.py          # Enhanced logging with DeepSpeed metrics
│       │   ├── visualizations.py  # Real-time training visualizations
│       │   └── moe_analytics.py   # MoE routing analysis
│       ├── utils/                 # Enhanced utilities
│       │   ├── data_processing.py # Data validation and processing
│       │   ├── environment.py     # System validation and optimization
│       │   ├── reporting.py       # Performance analysis and reporting
│       │   └── deepspeed_utils.py # DeepSpeed helper functions
│       └── config/               # Configuration management
│           ├── model_configs.yaml    # Model architecture presets
│           ├── deepspeed_configs.yaml # DeepSpeed optimization templates
│           └── training_configs.yaml  # Training parameter templates
├── configs/                      # User configuration files
├── data/                        # Training data and caches
│   ├── shards/                  # Data sharding for large datasets
│   ├── processed/               # Processed and validated data
│   └── cache/                   # Dataset caching
├── checkpoints/                 # Model checkpoints
│   ├── best/                    # Best model checkpoints
│   ├── emergency/               # Emergency recovery checkpoints
│   └── deepspeed/               # DeepSpeed universal checkpoints
├── experiments/                 # Experiment tracking and results
├── logs/                       # Comprehensive logging
│   ├── deepspeed/              # DeepSpeed-specific logs
│   ├── moe/                    # MoE routing logs
│   └── performance/            # Performance profiling logs
├── reports/                    # Analysis and performance reports
├── monitoring/                 # Real-time monitoring data
│   └── metrics/                # Training metrics
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
├── LICENSE                     # License file
└── README.md                   # This documentation
```

### Core Components

#### Model Core (`core/`)

- **`model.py`**: DeepSeek-style transformer implementation with MoE support
  - Optimized attention mechanisms with GQA and Flash Attention
  - SwiGLU feed-forward networks
  - RMSNorm for improved stability
  - Mixture of Experts with intelligent routing

- **`tokenizer.py`**: Advanced tokenization system
  - GPT-4 compatible tokenizer with extended vocabulary
  - Conversation-aware tokenization
  - Streaming tokenization for large datasets
  - Multi-threading support

- **`dataset.py`**: Sophisticated data handling
  - Memory-mapped datasets for efficiency
  - Streaming datasets for arbitrarily large data
  - Automatic sharding and load balancing
  - Data validation and preprocessing

#### Training System (`training/`)

- **`trainer.py`**: Production-grade trainer with DeepSpeed integration
  - Multi-GPU distributed training
  - Automatic mixed precision
  - Gradient accumulation and clipping
  - Memory optimization strategies

- **`orchestrator.py`**: Training coordination and monitoring
  - Experiment management
  - Resource allocation
  - Performance monitoring
  - Fault tolerance and recovery

- **`checkpoint.py`**: Advanced checkpoint management
  - Universal checkpoint format
  - Automatic backup and recovery
  - Checkpoint compression
  - Cross-platform compatibility

#### Monitoring (`monitoring/`)

- **`logger.py`**: Comprehensive training health monitoring system
  - Real-time performance metrics collection
  - Automated anomaly detection and alerts
  - Health scoring and diagnostics
  - Training stability analysis
  - Resource usage tracking with recommendations

## Quick Start

### Basic Training

```bash
# Clone the repository
git clone https://github.com/your-org/luminaai.git
cd luminaai

# Install dependencies
pip install -r requirements.txt

# Train a model with default settings
python Src/Main_Scripts/main.py

# Train with specific configuration
python Src/Main_Scripts/main.py --config configs/my_model.yaml

# Interactive chat with trained model
python Src/Main_Scripts/chat.py --checkpoint checkpoints/best_model.pt
```

### Multi-GPU Training

```bash
# Single node, multiple GPUs
deepspeed --num_gpus=4 Src/Main_Scripts/main.py --deepspeed

# Multi-node training
deepspeed --num_gpus=4 --num_nodes=2 --master_addr=10.0.0.1 Src/Main_Scripts/main.py --deepspeed
```

## Installation

### Requirements

- Python 3.9 or higher
- PyTorch 2.0 or higher
- CUDA 11.8 or higher (for GPU training)
- 16GB+ RAM (32GB+ recommended)
- 100GB+ storage space

### Standard Installation

```bash
# Clone repository
git clone https://github.com/your-org/luminaai.git
cd luminaai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Development Installation

```bash
# Install with development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

### Docker Installation

```bash
# Build Docker image
docker build -t luminaai .

# Run training in container
docker run --gpus all -v $(pwd)/data:/app/data luminaai
```

## Configuration

LuminaAI uses a hierarchical configuration system with Python dataclasses and built-in presets.

### Configuration Presets

The framework includes carefully tuned presets for different model sizes:

#### Model Size Presets

```python
from config.config_manager import ConfigPresets

# Debug configuration (minimal for testing)
config = ConfigPresets.debug()

# Small model (1B active parameters, 8B total with MoE)
config = ConfigPresets.b1()

# Medium model (7B active parameters, 56B total - Mixtral-style)
config = ConfigPresets.b7()

# Large model (14B active parameters, 112B total)
config = ConfigPresets.b14()

# Extra large models (50B+, 100B+, 200B+, 300B+ active parameters)
config = ConfigPresets.b50()
config = ConfigPresets.b100()
config = ConfigPresets.b200()
config = ConfigPresets.b300()
```

### Available Model Configurations

| Model | Active Params | Total Params (MoE) | Hidden Size | Layers | Heads | Sequence Length | Use Case |
|-------|---------------|---------------------|-------------|---------|-------|-----------------|----------|
| **debug** | ~500K | ~4M (8x500K) | 128 | 2 | 2 | 256 | Development/Testing |
| **b1** | ~1B | ~8B (8x1B) | 1536 | 16 | 12 | 2048 | Resource-constrained environments |
| **b7** | ~7B | ~56B (8x7B) | 4096 | 32 | 32 | 4096 | General purpose (Mixtral-style) |
| **b14** | ~14B | ~112B (8x14B) | 5120 | 40 | 40 | 4096 | High-performance applications |
| **b50** | ~50B | ~400B (8x50B) | 8192 | 64 | 64 | 128000 | Large-scale research |
| **b100** | ~100B | ~800B (8x100B) | 12288 | 80 | 96 | 200000 | Enterprise applications |
| **b200** | ~200B | ~1.6T (8x200B) | 16384 | 100 | 128 | 1000000 | Frontier research |
| **b300** | ~300B | ~2.4T (8x300B) | 20480 | 120 | 160 | 204800 | State-of-the-art models |

### Model Architecture Details

All models follow the **8x Mixture of Experts pattern** with consistent design principles:

#### Core Architecture Features
- **Attention**: Grouped Query Attention (GQA) with configurable KV heads
- **Activation**: SwiGLU activation function for improved performance  
- **Normalization**: RMSNorm for training stability
- **Position Encoding**: Rotary Position Embedding (RoPE) for long sequences
- **MoE Routing**: Top-1 routing (configurable to top-2) with load balancing

#### Scaling Pattern
- **Parameter Efficiency**: 8x total parameters, but only 1x active per forward pass
- **Expert Distribution**: 8 experts consistently across all model sizes
- **Memory Efficiency**: ~12.5% parameter efficiency (1/8 active at any time)
- **Training Speed**: Significantly faster than dense models of equivalent quality

#### Hardware Requirements by Model Size

| Model | Min GPU Memory | Recommended Hardware | Training Time (est.) |
|-------|----------------|---------------------|---------------------|
| **debug** | 2GB | Any modern GPU | Minutes |
| **b1** | 8GB | RTX 3090/4090 | Hours |
| **b7** | 40GB | A100-40GB/80GB | 1-3 days |
| **b14** | 80GB | Multiple A100-80GB | 3-7 days |
| **b50** | 200GB+ | Multi-node H100 clusters | 1-2 weeks |
| **b100** | 400GB+ | Large H100 clusters | 2-4 weeks |
| **b200** | 800GB+ | Massive clusters + NVMe | 1-2 months |
| **b300** | 1.2TB+ | Supercomputer-scale | 2-3 months |

#### Preset Information and Comparison

```python
# Get information about all available presets
preset_info = ConfigPresets.get_preset_info()
for preset_name, info in preset_info.items():
    print(f"{preset_name}: {info['description']}")
    print(f"  Active params: {info['active_params']}")
    print(f"  Hardware: {info['hardware']}")
    print(f"  Memory usage: {info['memory_usage']}")

# Compare presets side by side
comparison = ConfigPresets.compare_presets()
for preset, stats in comparison.items():
    print(f"{preset}: {stats['parameter_efficiency']} efficiency")
```

### Advanced Configuration Options

#### Model Architecture

```python
config = Config(
    # Core architecture
    hidden_size=4096,
    num_layers=32,
    num_heads=32,
    num_kv_heads=8,
    seq_length=4096,
    
    # MoE settings
    use_moe=True,
    num_experts=8,
    moe_top_k=2,
    capacity_factor=1.25,
    load_balancing_weight=0.01,
    
    # Training parameters
    batch_size=16,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,
    weight_decay=0.01,
    
    # Optimization
    precision="mixed_bf16",
    gradient_checkpointing=True,
    use_flash_attention=True,
    
    # DeepSpeed
    use_deepspeed=True,
    zero_stage=3,
    cpu_offload=True
)
```

#### DeepSpeed Configuration

```python
deepspeed_config = {
    "train_batch_size": 128,
    "gradient_accumulation_steps": 8,
    "fp16": {"enabled": True},
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {"device": "cpu"},
        "offload_param": {"device": "cpu"},
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": 200000000,
        "stage3_prefetch_bucket_size": 50000000
    }
}
```

## Training Models

### Basic Training Workflow

```python
from config.config_manager import ConfigPresets
from training.orchestrator import AdaptiveTrainingOrchestrator
from core.tokenizer import ConversationTokenizer
from core.model import DeepSeekTransformer

# Load configuration
config = ConfigPresets.b7()
config.train_data_path = "data/train.jsonl"
config.eval_data_path = "data/eval.jsonl"

# Initialize orchestrator
orchestrator = AdaptiveTrainingOrchestrator(config)

# Run training
orchestrator.initialize_training()
orchestrator.run_adaptive_training()
```

### Data Preparation

#### Converting Data to JSONL Format

```python
from utils.data_processing import convert_to_conversation_format

# Convert from various formats
convert_to_conversation_format(
    input_file="data/raw_conversations.json",
    output_file="data/train.jsonl",
    format_type="oasst"  # or "alpaca", "sharegpt"
)
```

#### Data Validation

```python
from utils.data_processing import validate_data_comprehensive

# Validate training data
stats = validate_data_comprehensive(
    data_path="data/train.jsonl",
    tokenizer=tokenizer,
    sample_size=1000
)

print(f"Total conversations: {stats['total_conversations']}")
print(f"Average tokens per conversation: {stats['avg_tokens']}")
print(f"Data quality score: {stats['quality_score']:.2f}")
```

### Advanced Training Features

#### Curriculum Learning

```python
# Progressive sequence length training
config.curriculum_learning = True
config.min_seq_length = 1024
config.max_seq_length = 4096
config.curriculum_steps = 1000
```

#### Dynamic Batch Sizing

```python
# Automatic batch size optimization
config.dynamic_batching = True
config.target_memory_usage = 0.9
config.min_batch_size = 1
config.max_batch_size = 32
```

#### Multi-Task Training

```python
# Train on multiple datasets simultaneously
config.multi_task = True
config.task_weights = {
    "conversation": 1.0,
    "instruction": 0.5,
    "code": 0.3
}
```

## Monitoring and Analytics

### Real-Time Monitoring

LuminaAI provides comprehensive monitoring capabilities:

#### Training Metrics Dashboard

```python
from monitoring.logger import TrainingHealthMonitor

# Initialize health monitor
health_monitor = TrainingHealthMonitor(log_dir="logs/health")

# Log training step
health_monitor.log_step({
    'global_step': step,
    'loss': loss.item(),
    'learning_rate': lr,
    'grad_norm': grad_norm,
    'tokens_per_sec': throughput,
    'memory_usage': memory_stats
})

# Get health summary
health_summary = health_monitor.get_health_summary()
print(f"Training health: {health_summary['health_status']}")
```

#### MoE Routing Analysis

```python
from training.trainer import MoEOptimizationManager

# Analyze expert utilization (basic metrics only)
moe_optimizer = MoEOptimizationManager(config)
routing_diagnostics = moe_optimizer.get_routing_diagnostics()

print(f"Expert balance score: {routing_diagnostics['expert_balance_score']}")
print(f"Load balance trend: {routing_diagnostics['load_balance_trend']}")

# Get recommendations for routing improvement
if routing_diagnostics['recommendations']:
    print("Routing recommendations:")
    for rec in routing_diagnostics['recommendations']:
        print(f"  - {rec}")
```

#### Performance Profiling

```python
from monitoring.logger import TrainingHealthMonitor

# Monitor training health and performance
health_monitor = TrainingHealthMonitor(log_dir="logs/health")

# Log training metrics
health_monitor.log_step({
    'global_step': step,
    'loss': loss.item(),
    'learning_rate': lr,
    'grad_norm': grad_norm,
    'tokens_per_sec': throughput,
    'memory_usage': memory_stats
})

# Get comprehensive health diagnostics
diagnostics = health_monitor.get_training_diagnostics()
print(f"Training stability: {diagnostics['training_stability']['status']}")
print(f"Performance efficiency: {diagnostics['performance_efficiency']['efficiency_status']}")

# Save detailed health report
health_monitor.save_health_report("reports/health_report.json")
```

### Logging and Monitoring

#### Structured Logging

```python
import logging
from monitoring.logger import TrainingHealthMonitor

# Configure comprehensive health monitoring
health_monitor = TrainingHealthMonitor(log_dir="logs/training")

# Log training metrics with health analysis
health_monitor.log_step({
    'global_step': step,
    'loss': loss.item(),
    'learning_rate': lr,
    'grad_norm': grad_norm,
    'tokens_per_sec': throughput,
    'memory_usage': memory_stats
})

# Get health summary with recommendations
health_summary = health_monitor.get_health_summary()
print(f"Health status: {health_summary['health_status']}")
print(f"Health score: {health_summary['overall_health_score']:.2f}")

# Check for training alerts
recent_alerts = health_monitor.metrics_collector.get_recent_alerts(minutes=10)
for alert in recent_alerts:
    print(f"Alert [{alert.severity}]: {alert.message}")
    if alert.recommendation:
        print(f"  Recommendation: {alert.recommendation}")
```

## Advanced Features

### Mixture of Experts (MoE)

#### MoE Configuration

```python
# Configure MoE model
config.use_moe = True
config.num_experts = 8
config.moe_top_k = 2
config.capacity_factor = 1.25
config.load_balancing_weight = 0.01

# Expert parallel configuration
config.expert_parallel_size = 2  # Distribute experts across 2 GPUs
```

#### MoE Optimization

```python
from training.trainer import MoEOptimizationManager

# Initialize MoE optimizer
moe_optimizer = MoEOptimizationManager(config)

# Create optimized DeepSpeed configuration
deepspeed_config = moe_optimizer.create_deepspeed_moe_config(base_config)

# Monitor routing balance
routing_diagnostics = moe_optimizer.get_routing_diagnostics()
if routing_diagnostics['expert_balance_score'] < 0.7:
    print("Warning: Poor expert load balancing detected")
```

### DeepSpeed Integration

#### ZeRO Optimization

```python
# ZeRO Stage 1: Optimizer state sharding
config.zero_stage = 1

# ZeRO Stage 2: Gradient sharding
config.zero_stage = 2

# ZeRO Stage 3: Parameter sharding
config.zero_stage = 3
config.stage3_prefetch_bucket_size = 50000000
config.stage3_param_persistence_threshold = 100000
```

#### CPU/NVMe Offloading

```python
# CPU offloading for optimizer states
config.cpu_offload_optimizer = True

# CPU offloading for parameters
config.cpu_offload_parameters = True

# NVMe offloading for very large models
config.nvme_offload_optimizer = True
config.nvme_offload_parameters = True
config.nvme_path = "/tmp/deepspeed_nvme"
```

#### Communication Optimization

```python
# Overlap communication and computation
config.overlap_comm = True
config.contiguous_gradients = True

# Optimize bucket sizes
config.allgather_bucket_size = 200000000
config.reduce_bucket_size = 200000000

# Use efficient communication backend
config.communication_backend = "nccl"
```

### Checkpoint Management

#### Universal Checkpoints

```python
from training.checkpoint import CheckpointManager

# Initialize checkpoint manager
checkpoint_manager = CheckpointManager(config)

# Save checkpoint with metadata
checkpoint_path = checkpoint_manager.save_checkpoint(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    global_step=step,
    current_epoch=epoch,
    metrics=training_metrics,
    is_best=True
)

# Load checkpoint with automatic compatibility checking
epoch = checkpoint_manager.load_checkpoint(
    checkpoint_path="checkpoints/best_model.pt",
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    strict=False
)
```

#### Automatic Backup and Recovery

```python
# Configure automatic backup
config.backup_every_n_hours = 6
config.max_backup_count = 5
config.enable_emergency_save = True

# Emergency checkpoint on interruption
checkpoint_manager.emergency_save(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    global_step=step,
    current_epoch=epoch,
    metrics=metrics
)
```

### Adaptive Training

#### Hyperparameter Optimization

```python
from training.orchestrator import AdaptiveHyperparameterOptimizer

# Initialize adaptive optimizer
adaptive_optimizer = AdaptiveHyperparameterOptimizer()

# Check if learning rate should be adjusted
lr_adjustment = adaptive_optimizer.should_adjust_learning_rate(current_metrics)
if lr_adjustment:
    new_lr = current_lr * lr_adjustment['factor']
    trainer.adjust_learning_rate(new_lr)
    print(f"Learning rate adjusted: {current_lr} -> {new_lr}")
```

#### Dynamic Architecture Optimization

```python
from training.orchestrator import ArchitectureEvolution

# Initialize architecture evolution
arch_evolution = ArchitectureEvolution()

# Check if experts should be added/removed
expert_suggestion = arch_evolution.should_add_expert(
    expert_utilization=expert_stats,
    performance_metrics=training_metrics
)

if expert_suggestion:
    print(f"Suggestion: {expert_suggestion['action']}")
    print(f"Reasoning: {expert_suggestion['reasoning']}")
```

## API Reference

### Core Classes

#### DeepSeekTransformer

```python
class DeepSeekTransformer(nn.Module):
    def __init__(self, config: DeepSeekConfig):
        """Initialize DeepSeek transformer with MoE support."""
        
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through the transformer."""
        
    def get_memory_footprint(self) -> Dict[str, float]:
        """Get detailed memory usage analysis."""
        
    def estimate_flops(self, seq_len: int, batch_size: int = 1) -> Dict[str, int]:
        """Estimate FLOPs for forward pass."""
```

#### ConversationTokenizer

```python
class ConversationTokenizer:
    def __init__(self, model_name: str = "gpt-4", 
                 max_context_length: int = 8192):
        """Initialize conversation-aware tokenizer."""
        
    def encode_conversation(self, conversation: Dict[str, Any]) -> List[int]:
        """Encode conversation to token sequence."""
        
    def decode(self, token_ids: List[int]) -> str:
        """Decode token sequence to text."""
        
    def get_vocab_size(self) -> int:
        """Get total vocabulary size."""
```

#### EnhancedConversationTrainer

```python
class EnhancedConversationTrainer:
    def __init__(self, model, tokenizer, config, logger):
        """Initialize trainer with DeepSpeed support."""
        
    def train(self, train_dataset, eval_dataset=None):
        """Main training loop."""
        
    def evaluate(self, eval_dataset, max_batches: int = 100) -> Dict[str, float]:
        """Evaluate model on validation set."""
        
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory usage statistics."""
```

### Configuration System

#### Config Class

```python
@dataclass
class Config:
    # Model architecture
    vocab_size: int = 50304
    hidden_size: int = 512
    num_layers: int = 8
    num_heads: int = 8
    num_kv_heads: int = 4
    
    # MoE parameters
    use_moe: bool = True
    num_experts: int = 8
    moe_top_k: int = 1
    capacity_factor: float = 1.25
    
    # Training parameters
    batch_size: int = 2
    learning_rate: float = 1e-4
    num_epochs: int = 3
    
    # DeepSpeed parameters
    use_deepspeed: bool = True
    zero_stage: int = 0  # Auto-select
    cpu_offload: bool = False
    
    def validate(self):
        """Validate configuration parameters."""
        
    def save(self, path: str):
        """Save configuration to YAML file."""
        
    @classmethod
    def load(cls, path: str) -> 'Config':
        """Load configuration from YAML file."""
```

### Utility Functions

#### Data Processing

```python
def process_oasst_data(input_file: str, output_file: str) -> int:
    """Process OASST dataset into conversation format."""
    
def validate_data_comprehensive(data_path: str, tokenizer, 
                               sample_size: int = 1000) -> Dict[str, Any]:
    """Comprehensive data validation with statistics."""
    
def create_data_summary_report(data_path: str, output_path: str):
    """Generate comprehensive data analysis report."""
```

#### Environment Optimization

```python
def validate_environment() -> List[str]:
    """Validate training environment and return issues."""
    
def optimize_cuda_settings():
    """Optimize CUDA settings for training."""
    
def estimate_training_time(config: Config, dataset_size: int) -> float:
    """Estimate training time in hours."""
```

## Examples

### Training a Small Conversation Model

```python
#!/usr/bin/env python3
"""Example: Train a small conversational model."""

from config.config_manager import ConfigPresets
from training.orchestrator import AdaptiveTrainingOrchestrator
from utils.data_processing import process_oasst_data

def main():
    # Process training data
    process_oasst_data(
        input_file="data/raw/oasst1_conversations.json",
        output_file="data/processed/train.jsonl"
    )
    
    # Load configuration
    config = ConfigPresets.debug()  # Small model for testing
    config.train_data_path = "data/processed/train.jsonl"
    config.num_epochs = 5
    config.experiment_name = "small_conversation_model"
    
    # Initialize training
    orchestrator = AdaptiveTrainingOrchestrator(config)
    orchestrator.initialize_training()
    
    # Run training
    orchestrator.run_adaptive_training()
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()
```

### Multi-GPU MoE Training

```python
#!/usr/bin/env python3
"""Example: Multi-GPU MoE model training."""

import os
import torch.distributed as dist
from config.config_manager import ConfigPresets

def setup_distributed():
    """Setup distributed training."""
    if 'RANK' in os.environ:
        dist.init_process_group(backend='nccl')

def main():
    setup_distributed()
    
    # Load large MoE configuration
    config = ConfigPresets.b7()  # 7B active parameter model
    config.use_moe = True
    config.num_experts = 8
    config.moe_top_k = 2
    
    # DeepSpeed configuration
    config.use_deepspeed = True
    config.zero_stage = 3
    config.cpu_offload = True
    
    # Training settings
    config.batch_size = 4  # Micro batch size
    config.gradient_accumulation_steps = 16
    config.num_epochs = 3
    config.experiment_name = "mixtral_style_moe"
    
    # Data paths
    config.train_data_path = "data/large_conversations.jsonl"
    config.eval_data_path = "data/eval_conversations.jsonl"
    
    # Initialize and run training
    from training.orchestrator import AdaptiveTrainingOrchestrator
    orchestrator = AdaptiveTrainingOrchestrator(config)
    orchestrator.run_adaptive_training()

if __name__ == "__main__":
    main()
```

### Custom Training Loop

```python
#!/usr/bin/env python3
"""Example: Custom training loop with monitoring."""

import torch
from core.model import DeepSeekTransformer, DeepSeekConfig
from core.tokenizer import ConversationTokenizer
from core.dataset import ConversationDataset
from training.trainer import EnhancedConversationTrainer
from monitoring.logger import TrainingHealthMonitor

def custom_training():
    # Model configuration
    model_config = DeepSeekConfig(
        vocab_size=50304,
        hidden_size=1024,
        num_layers=24,
        num_heads=16,
        num_kv_heads=4,
        seq_length=2048,
        use_moe=True,
        num_experts=8,
        moe_top_k=2
    )
    
    # Initialize components
    model = DeepSeekTransformer(model_config)
    tokenizer = ConversationTokenizer(model_name="gpt-4")
    
    # Load dataset
    train_dataset = ConversationDataset(
        data_path="data/train.jsonl",
        tokenizer=tokenizer,
        config=model_config
    )
    
    # Initialize health monitor
    health_monitor = TrainingHealthMonitor(log_dir="logs/custom_training")
    
    # Create trainer
    from config.config_manager import Config
    training_config = Config(
        batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        num_epochs=3,
        use_deepspeed=True,
        zero_stage=2
    )
    
    trainer = EnhancedConversationTrainer(
        model=model,
        tokenizer=tokenizer,
        config=training_config,
        logger=health_monitor
    )
    
    # Custom training loop
    trainer.train(train_dataset)
    
    # Generate final report
    health_monitor.save_health_report("reports/training_health.json")
    
    print("Custom training completed!")

if __name__ == "__main__":
    custom_training()
```

### Interactive Chat Interface

LuminaAI includes a sophisticated chat interface (`chat.py`) for testing and interacting with trained models:

#### Features

- **Automatic Checkpoint Detection**: Finds and loads the best available checkpoint automatically
- **Multi-Shard Support**: Handles ZeRO checkpoint shards and merges them seamlessly  
- **Conversation Modes**: Multiple generation strategies for different use cases
- **Real-time Statistics**: Token counts, response timing, and session analytics
- **Command System**: Built-in commands for configuration and management

#### Usage

```bash
# Auto-detect and load best checkpoint
python chat.py

# Load specific checkpoint file
python chat.py --checkpoint model.pt

# Load ZeRO checkpoint directory 
python chat.py --checkpoint ./checkpoints/deepspeed_epoch_5/

# Start with creative mode and show timing
python chat.py --mode creative --show-timing

# Set system prompt and conversation history limit
python chat.py --system-prompt "You are a helpful AI assistant" --max-history 20
```

#### Conversation Modes

The chat interface supports multiple generation modes optimized for different scenarios:

| Mode | Temperature | Top-p | Top-k | Best For |
|------|-------------|-------|-------|----------|
| **standard** | 0.8 | 0.9 | 50 | General conversation |
| **creative** | 1.0 | 0.95 | 100 | Creative writing, brainstorming |
| **analytical** | 0.3 | 0.7 | 20 | Technical analysis, reasoning |
| **precise** | 0.1 | 0.5 | 10 | Factual responses, coding |

#### Interactive Commands

```bash
# Available commands during chat
/help          # Show command help
/quit, /exit   # Exit chat session
/clear         # Clear conversation history  
/stats         # Show session statistics
/mode <mode>   # Change conversation mode
/system <text> # Set system prompt
/save [name]   # Save conversation to file
```

#### Advanced Features

**Automatic Configuration Inference**: 
- Detects model architecture from checkpoint
- Handles missing configuration gracefully
- Supports both single files and ZeRO checkpoint directories

**Robust Error Handling**:
- Recovers from invalid token generation
- Handles out-of-vocabulary tokens gracefully  
- Provides helpful error messages and fallbacks

**Session Analytics**:
```python
# Statistics tracked during chat session
{
    'messages_sent': 15,
    'tokens_generated': 2847,
    'avg_response_time': 1.23,
    'session_duration': '00:15:42',
    'current_mode': 'standard'
}
```

**Conversation Persistence**:
```bash
# Conversations saved in JSON format with metadata
{
    "name": "chat_20250120_143022", 
    "conversation": [...],
    "mode": "standard",
    "system_prompt": "You are a helpful assistant"
}
```
        
        # Add user message to history
        conversation_history.append({
            'role': 'user',
            'content': user_input
        })
        
        # Encode conversation
        tokens = tokenizer.encode_conversation({
            'messages': conversation_history
        })
        
        # Generate response (simplified for example)
        with torch.no_grad():
            input_ids = torch.tensor([tokens], dtype=torch.long)
            output = model(input_ids)
            # Response generation logic would go here
            
        # Add assistant response to history
        response = "This is a placeholder response."
        conversation_history.append({
            'role': 'assistant', 
            'content': response
        })
        
        print(f"Assistant: {response}")

if __name__ == "__main__":
    chat_interface()
```

## Performance Optimization

### Memory Optimization

LuminaAI implements several strategies to handle large models efficiently:

#### ZeRO Optimization Stages

- **ZeRO-1**: Optimizer state partitioning (8x memory reduction for optimizer states)
- **ZeRO-2**: Gradient partitioning (additional gradient memory reduction)  
- **ZeRO-3**: Parameter partitioning (enables training models larger than single GPU memory)

#### CPU and NVMe Offloading

```python
# Configure offloading for very large models
config.cpu_offload_optimizer = True      # Offload optimizer states to CPU
config.cpu_offload_parameters = True     # Offload parameters to CPU
config.nvme_offload_optimizer = True     # Offload to NVMe for maximum capacity
config.nvme_path = "/tmp/deepspeed_nvme" # NVMe mount point
```

#### Gradient Checkpointing

```python
# Reduce activation memory usage
config.gradient_checkpointing = True
config.partition_activations = True  # Advanced activation partitioning
```

### Compute Optimization

#### Mixed Precision Training

```python
# Automatic precision selection based on hardware
config.precision = "auto"  # Selects optimal precision for your GPU
config.auto_tune_precision = True
config.dynamic_precision = False
```

#### Model Compilation

```python
# PyTorch 2.0 compilation for faster training
config.compile = True
```

### MoE-Specific Optimizations

#### Expert Parallelism

```python
# Distribute experts across GPUs for optimal communication
config.expert_parallel_size = 2  # Use 2 GPUs for expert parallelism
config.overlap_alltoall = True    # Overlap communication with computation
```

#### Capacity and Load Balancing

```python
# Optimize token routing for better expert utilization
config.capacity_factor = 1.25       # Buffer for token distribution
config.load_balancing_weight = 0.01 # Encourage balanced expert usage
config.router_jitter_noise = 0.01   # Prevent hot expert concentration
```

## Troubleshooting

### Common Issues

#### DeepSpeed Initialization Failures

```bash
# Check DeepSpeed installation
pip install deepspeed

# Verify CUDA compatibility
python -c "import deepspeed; print(deepspeed.__version__)"

# Run with verbose logging
DEEPSPEED_LOG_LEVEL=INFO python main.py --deepspeed
```

#### Memory Issues

```python
# Reduce batch size and enable offloading
config.batch_size = 1
config.gradient_accumulation_steps = 32
config.cpu_offload = True
config.gradient_checkpointing = True
```

#### MoE Training Instability

```python
# Increase capacity factor and adjust load balancing
config.capacity_factor = 2.0
config.load_balancing_weight = 0.02
config.moe_top_k = 1  # Use top-1 routing for stability
```

#### Data Loading Problems

```python
# Enable streaming for large datasets
config.streaming_threshold_gb = 5.0
config.num_workers = 0  # Disable multiprocessing if causing issues
```

### Performance Troubleshooting

#### Low Throughput

1. **Check GPU utilization**: Use `nvidia-smi` to monitor GPU usage
2. **Optimize batch size**: Use the largest batch size that fits in memory
3. **Enable compilation**: Set `config.compile = True` 
4. **Check data loading**: Ensure data loading isn't the bottleneck

#### High Memory Usage

1. **Enable gradient checkpointing**: `config.gradient_checkpointing = True`
2. **Use CPU offloading**: `config.cpu_offload = True`
3. **Reduce sequence length**: Lower `config.seq_length`
4. **Use smaller precision**: Switch to `fp16` or `bf16`

### Debugging Tools

#### Training Health Monitoring

```python
# Get comprehensive training diagnostics
health_monitor = TrainingHealthMonitor()
diagnostics = health_monitor.get_training_diagnostics()

print(f"Training stability: {diagnostics['training_stability']}")
print(f"Performance efficiency: {diagnostics['performance_efficiency']}")
print(f"Resource utilization: {diagnostics['resource_utilization']}")

# Get specific recommendations
recommendations = diagnostics['recommendations']
for rec in recommendations:
    print(f"Recommendation: {rec}")
```

#### DeepSpeed Debugging

```python
# Enable detailed DeepSpeed logging
import deepspeed
deepspeed.init_distributed()

# Check DeepSpeed configuration
trainer.debug_training_setup()  # Built-in debugging method
```

## Contributing

We welcome contributions to LuminaAI! Here's how you can help:

### Development Setup

```bash
# Clone the repository
git clone https://github.com/your-org/luminaai.git
cd luminaai

# Create development environment
python -m venv dev_env
source dev_env/bin/activate

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

### Code Style

- Follow PEP 8 style guidelines
- Use type hints for all function signatures
- Include comprehensive docstrings
- Add unit tests for new functionality

### Pull Request Process

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and add tests
4. Ensure all tests pass: `python -m pytest tests/`
5. Update documentation if needed
6. Submit a pull request

### Reporting Issues

When reporting bugs, please include:

- Python and PyTorch versions
- Hardware specifications (GPU model, memory)
- Complete error traceback
- Minimal reproduction case
- Configuration used

## License

LuminaAI is licensed under a custom license. See the [LICENSE](LICENSE) file for full details.

## TL;DR

**LuminaAI** is a production-ready framework for training large language models with advanced features:

- **🚀 Quick Start**: `python main.py` to begin training with sensible defaults
- **🧠 Smart Architecture**: DeepSeek transformers with MoE support (8x efficiency gains)
- **⚡ Enterprise Scale**: DeepSpeed integration for multi-GPU/multi-node training  
- **📊 Built-in Monitoring**: Real-time health monitoring with automatic issue detection
- **🔧 Zero Config**: Intelligent presets from 1B to 300B+ parameter models
- **💾 Fault Tolerant**: Automatic checkpointing with universal format compatibility

**Perfect for**: Researchers, ML engineers, and organizations training conversational AI models who need production reliability without configuration complexity.

---

## Summary

LuminaAI represents a comprehensive solution for modern large language model training, combining cutting-edge techniques with production-grade reliability. Built from the ground up with DeepSpeed integration and Mixture of Experts support, it enables training of models from 1B to 300B+ active parameters on everything from single GPUs to massive clusters.

The framework's adaptive training orchestrator continuously monitors training health, automatically detects issues, and provides actionable recommendations - reducing the expertise barrier for training large models while maintaining the flexibility experts need. With features like universal checkpointing, automatic precision optimization, and intelligent resource management, LuminaAI bridges the gap between research experimentation and production deployment.

Whether you're training a small conversational model for specific tasks or scaling to frontier-class models, LuminaAI provides the tools, monitoring, and optimizations needed to succeed. The framework's modular architecture ensures extensibility while its comprehensive configuration system offers both simplicity for beginners and deep customization for advanced users.

**LuminaAI: Making large-scale language model training accessible, reliable, and efficient.**