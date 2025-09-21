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
- [Command Line Interface](#command-line-interface)
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

- **DeepSeek-Style Transformers**: Advanced transformer architecture with optimized attention mechanisms
- **Mixture of Experts (MoE)**: Scalable MoE implementation with intelligent routing
- **DeepSpeed Integration**: Full DeepSpeed support with ZeRO optimization stages
- **Production Ready**: Enterprise-grade reliability with comprehensive error handling
- **Advanced Monitoring**: Real-time metrics, health monitoring, and performance tracking
- **Adaptive Training**: Intelligent training orchestration with automated optimization
- **Smart Checkpointing**: Universal checkpoint format with automatic recovery
- **Flexible Configuration**: Python-based configuration with validation and presets
- **Enhanced CLI**: Comprehensive command-line interface with over 40 commands
- **Interactive Chat**: Advanced chat interface for model testing and evaluation

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
├── Main.py                       # Enhanced main script with comprehensive CLI
├── Readme.md                    # This documentation
├── config/
│   ├── config_manager.py        # Configuration classes and presets
│   └── __init__.py
├── core/                        # Core model and data components
│   ├── model.py                 # DeepSeek transformer with MoE
│   ├── tokenizer.py             # GPT-4 compatible tokenization
│   ├── dataset.py               # Advanced dataset handling
│   └── __init__.py
├── training/                    # Enhanced training system
│   ├── trainer.py               # DeepSpeed-enabled trainer
│   ├── orchestrator.py          # Training coordination and monitoring
│   ├── checkpoint.py            # Advanced checkpoint management
│   └── __init__.py
├── monitoring/                  # Comprehensive monitoring
│   ├── logger.py                # Enhanced logging with health metrics
│   ├── visualizations.py        # Real-time training visualizations
│   ├── moe_analytics.py         # MoE routing analysis
│   └── __init__.py
├── utils/                       # Enhanced utilities
│   ├── data_processing.py       # Data validation and processing
│   ├── environment.py           # System validation and optimization
│   ├── reporting.py             # Performance analysis and reporting
│   └── __init__.py
├── data/                        # Training data and caches
│   ├── shards/                  # Data sharding for large datasets
│   ├── processed/               # Processed and validated data
│   └── cache/                   # Dataset caching
├── checkpoints/                 # Model checkpoints
│   ├── best/                    # Best model checkpoints
│   ├── emergency/               # Emergency recovery checkpoints
│   └── deepspeed/               # DeepSpeed universal checkpoints
├── experiments/                 # Experiment tracking and results
├── logs/                        # Comprehensive logging
│   ├── deepspeed/               # DeepSpeed-specific logs
│   ├── moe/                     # MoE routing logs
│   └── performance/             # Performance profiling logs
├── reports/                     # Analysis and performance reports
├── monitoring/                  # Real-time monitoring data
│   └── metrics/                 # Training metrics
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
├── LICENSE                      # License file
└── README.md                    # This documentation
```

## Quick Start

### Basic Training

```bash
# Clone the repository
git clone https://github.com/your-org/luminaai.git
cd luminaai

# Install dependencies
pip install -r requirements.txt

# Train a model with default settings (debug configuration)
python Main.py

# Train with specific preset
python Main.py --config b7  # 7B active parameter model

# Interactive chat with trained model
python Main.py --analyze-model --checkpoint best
```

### Multi-GPU Training

```bash
# Single node, multiple GPUs
deepspeed --num_gpus=4 Main.py

# Multi-node training
deepspeed --num_gpus=4 --num_nodes=2 --master_addr=10.0.0.1 Main.py
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

### Hardcoded Configuration Override

For quick experimentation, you can modify the hardcoded configuration section in `Main.py`:

```python
# Base model configuration - select from ConfigPresets
config_choice = 'debug'  # Options: 'debug', 'b1', 'b7', 'b14', 'b50', 'b100', 'b200', 'b300'

# Override specific parameters
override_params = {
    'use_moe': True,
    'num_epochs': 3,
    'learning_rate': 1e-4,
    'batch_size': 1,
    'gradient_accumulation_steps': 16,
    'train_data_path': 'oasst1_data/oasst1_train.jsonl',
    'eval_data_path': 'oasst1_data/oasst1_train.jsonl',
}

# DeepSpeed and optimization settings - MANUAL OVERRIDES
manual_deepspeed_overrides = {
    'use_deepspeed': True,
    'cpu_offload': True,
    'cpu_offload_optimizer': True,
    'zero_stage': 2,
    'nvme_path': "Deepspeed/Tmp",
}
```

## Command Line Interface

LuminaAI provides a comprehensive CLI with over 40 commands organized into functional groups:

### Training Workflow

```bash
# Resume training from checkpoints
python Main.py --resume                    # Resume from latest checkpoint
python Main.py --resume best               # Resume from best checkpoint
python Main.py --resume path/to/checkpoint # Resume from specific checkpoint

# Continue with new experiment name
python Main.py --resume --continue-as new_experiment_name

# Test configuration without training
python Main.py --dry-run

# Force restart ignoring existing checkpoints
python Main.py --force-restart
```

### Model Analysis

```bash
# Analyze model architecture and parameters
python Main.py --analyze-model
python Main.py --analyze-model --checkpoint best

# Show detailed model statistics
python Main.py --model-stats

# Analyze memory requirements
python Main.py --memory-analysis
```

### Data Management

```bash
# Validate training data quality
python Main.py --validate-data

# Process raw data files
python Main.py --process-data input.jsonl output.jsonl

# Generate data summary report
python Main.py --data-report

# Split data into shards for distributed training
python Main.py --create-shards 8
```

### System Diagnostics

```bash
# Validate training environment and dependencies
python Main.py --check-environment

# Benchmark training performance
python Main.py --benchmark-performance

# Profile memory usage during training
python Main.py --profile-memory

# Debug training setup and configuration
python Main.py --debug-setup

# Test DeepSpeed configuration
python Main.py --test-deepspeed
```

### Model Export

```bash
# Export trained model for inference
python Main.py --export-model --checkpoint best --format pytorch
python Main.py --export-model --format huggingface --output-dir exported_models
```

### Checkpoint Management

```bash
# List all available checkpoints
python Main.py --list-checkpoints

# Create backup of current training state
python Main.py --create-backup

# Clean up old checkpoints (keep best and latest)
python Main.py --clean-checkpoints

# Merge multiple checkpoints (experimental)
python Main.py --merge-checkpoints checkpoint1.pt checkpoint2.pt
```

### Experiment Management

```bash
# List all experiments
python Main.py --list-experiments

# Archive an experiment
python Main.py --archive-experiment experiment_name

# Compare multiple experiments
python Main.py --compare-experiments exp1 exp2 exp3
```

### Monitoring and Logging

```bash
# Enable TensorBoard logging
python Main.py --tensorboard

# Enable Weights & Biases logging
python Main.py --wandb

# Set logging level
python Main.py --log-level DEBUG

# Increase verbosity
python Main.py --verbose -vv
```

### Advanced Features

```bash
# Enable automatic hyperparameter tuning
python Main.py --auto-tune

# Set distributed training backend
python Main.py --distributed-backend nccl

# Set mixed precision mode
python Main.py --mixed-precision bf16

# Use torch.compile for model optimization
python Main.py --compile-model
```

## Training Models

### Basic Training Workflow

```python
from config.config_manager import ConfigPresets
from training.orchestrator import AdaptiveTrainingOrchestrator

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
from utils.data_processing import process_oasst_data

# Convert OASST data to conversation format
process_oasst_data(
    input_file="data/raw_conversations.json",
    output_file="data/train.jsonl"
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

## Monitoring and Analytics

### Real-Time Health Monitoring

LuminaAI includes a sophisticated health monitoring system that tracks training progress and automatically detects issues:

```python
from monitoring.logger import TrainingHealthMonitor

# Initialize health monitor
health_monitor = TrainingHealthMonitor(log_dir="logs/health")

# Log training step with automatic health analysis
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

# Check for automated alerts and recommendations
recent_alerts = health_monitor.metrics_collector.get_recent_alerts(minutes=10)
for alert in recent_alerts:
    print(f"Alert [{alert.severity}]: {alert.message}")
    if alert.recommendation:
        print(f"  Recommendation: {alert.recommendation}")
```

### MoE Routing Analysis

```python
from training.trainer import MoEOptimizationManager

# Analyze expert utilization
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

### Interactive Chat Interface

LuminaAI includes a sophisticated chat interface for testing and interacting with trained models:

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

### Using the CLI for Complete Workflow

```bash
#!/bin/bash
# Complete training workflow using CLI commands

# Step 1: Validate environment
echo "Checking environment..."
python Main.py --check-environment

# Step 2: Process and validate data
echo "Processing training data..."
python Main.py --process-data raw_data.json processed_data.jsonl
python Main.py --validate-data

# Step 3: Debug training setup
echo "Testing training setup..."
python Main.py --debug-setup

# Step 4: Run training with monitoring
echo "Starting training..."
python Main.py --tensorboard --log-level INFO

# Step 5: Analyze results
echo "Analyzing trained model..."
python Main.py --analyze-model --checkpoint best
python Main.py --export-model --checkpoint best --format pytorch

# Step 6: Test with chat interface
echo "Testing model..."
python chat.py --checkpoint best --mode standard

echo "Training workflow completed!"
```

## Performance Optimization

### Memory Optimization

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

### Compute Optimization

#### Mixed Precision Training

```python
# Automatic precision selection based on hardware
config.precision = "auto"  # Selects optimal precision for your GPU
config.auto_tune_precision = True
config.dynamic_precision = False
```

### MoE-Specific Optimizations

#### Expert Parallelism

```python
# Distribute experts across GPUs for optimal communication
config.expert_parallel_size = 2  # Use 2 GPUs for expert parallelism
config.overlap_alltoall = True    # Overlap communication with computation
```

## Troubleshooting

### Common Issues

#### DeepSpeed Initialization Failures

```bash
# Check DeepSpeed installation
pip install deepspeed

# Verify CUDA compatibility
python -c "import deepspeed; print(deepspeed.__version__)"

# Run diagnostic tests
python Main.py --test-deepspeed
```

#### Memory Issues

```bash
# Use built-in memory profiling
python Main.py --profile-memory

# Analyze memory requirements
python Main.py --memory-analysis

# Enable aggressive memory optimization
python Main.py --cpu-offload --zero-stage 3 --gradient-checkpointing
```

#### Data Loading Problems

```bash
# Validate data format and quality
python Main.py --validate-data

# Check data processing
python Main.py --data-report
```

### Debugging Tools

#### Training Health Monitoring

```python
# Get comprehensive training diagnostics
health_monitor = TrainingHealthMonitor()
diagnostics = health_monitor.get_training_diagnostics()

print(f"Training stability: {diagnostics['training_stability']}")
print(f"Performance efficiency: {diagnostics['performance_efficiency']}")
print(f"Resource utilization: {diagnostics['resource_utilization']}")
```

#### Built-in Debugging Commands

```bash
# Complete system diagnostics
python Main.py --debug-setup

# Test all components
python Main.py --check-environment --test-deepspeed --benchmark-performance

# Monitor training in real-time
python Main.py --verbose --log-level DEBUG
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

## License

LuminaAI is licensed under a custom license. See the [LICENSE](LICENSE) file for full details.

## TL;DR

**LuminaAI** is a production-ready framework for training large language models with advanced features:

- **Quick Start**: `python Main.py` to begin training with sensible defaults
- **Smart Architecture**: DeepSeek transformers with MoE support (8x efficiency gains)
- **Enterprise Scale**: DeepSpeed integration for multi-GPU/multi-node training  
- **Built-in Monitoring**: Real-time health monitoring with automatic issue detection
- **Zero Config**: Intelligent presets from 1B to 300B+ parameter models
- **Fault Tolerant**: Automatic checkpointing with universal format compatibility
- **Comprehensive CLI**: 40+ commands for every aspect of training and analysis
- **Interactive Testing**: Advanced chat interface for model evaluation

**Perfect for**: Researchers, ML engineers, and organizations training conversational AI models who need production reliability without configuration complexity.

**Key Commands**:
```bash
python Main.py                    # Start training with defaults
python Main.py --analyze-model    # Analyze model architecture
python Main.py --debug-setup      # Test system configuration
python Main.py --list-experiments # Manage experiments
python chat.py                    # Interactive model testing
```

---

**LuminaAI: Making large-scale language model training accessible, reliable, and efficient.**