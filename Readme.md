# LuminaAI

A machine learning framework for training conversational transformers with Mixture of Experts architecture.

[![License](https://img.shields.io/badge/license-Custom-blue.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-brightgreen.svg)]()
[![PyTorch](https://img.shields.io/badge/pytorch-1.13+-ee4c2c.svg)](https://pytorch.org/)
[![WandB](https://img.shields.io/badge/weights_&_biases-integrated-orange.svg)](https://wandb.ai)

<p align="center">
  <img src="assets/logo.png" alt="LuminaAI Logo" width="200"/>
</p>

## Features

### Core Capabilities
- Training pipeline with error handling
- Support for conversational datasets (OASST format)
- Mixed precision training (FP16/BF16)
- Mixture of Experts (MoE) architecture
- AdamW optimizer
- Flash attention support
- PyTorch 2.0+ model compilation

### Training
- Gradient accumulation
- Learning rate scheduling (Cosine, OneCycle, Linear)
- Early stopping with configurable patience
- Automatic and manual checkpointing
- Training resumption from checkpoints

### Monitoring
- Training metrics logging
- Health checks and anomaly detection
- Automatic failure recovery
- GPU memory monitoring
- Periodic backups

### Data Processing
- GPT-4 compatible tokenization
- Dataset validation
- Multi-threaded batch processing
- OpenAssistant dataset format support

## Requirements

```bash
# Core dependencies
torch>=1.13.0
tiktoken>=0.5.0
numpy>=1.21.0
psutil>=5.8.0

# Optional dependencies
flash-attn>=2.0.0  # For flash attention support
wandb>=0.15.0      # For experiment tracking
```

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/MatN23/LuminaAI.git
cd LuminaAI
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

## Usage

### Basic Training

```bash
python Src/Main_Scripts/main.py \
  --config medium \
  --train-data data/train.jsonl \
  --eval-data data/eval.jsonl \
  --epochs 10 \
  --lr 1e-4 \
  --batch-size 4 \
  --experiment-name my_experiment
```

### Configuration Presets

Available configurations:
- **`debug`**: Small model for testing
- **`small`**: Lightweight model 
- **`medium`**: Balanced configuration
- **`large`**: High-capacity model

### Data Format

Training data in JSONL format:

```json
{
  "messages": [
    {"role": "user", "content": "Hello, how are you?"},
    {"role": "assistant", "content": "I'm doing well, thank you for asking!"}
  ]
}
```

Supported roles: `user`, `assistant`, `system`, `prompter`, `human`, `ai`, `bot`

## Model Architecture

### Components
- **Mixture of Experts (MoE)**: Sparse expert routing
- **Rotary Position Embedding (RoPE)**: Positional encoding
- **SwiGLU Activation**: Activation function
- **RMSNorm**: Layer normalization
- **Weight Tying**: Shared embedding and output weights

### Optimization
- **AdamW Optimizer**: Adaptive learning rate with weight decay
- **Gradient Checkpointing**: Memory optimization
- **Mixed Precision**: FP16/BF16 training

### Model Configurations

| Config | Hidden Size | Layers | Heads | Experts | Active Experts | Seq Length | Parameters | Memory (FP16) | Precision |
|--------|-------------|--------|-------|---------|----------------|------------|------------|---------------|-----------|
| Debug  | 128         | 2      | 2     | 4       | 2              | 256        | ~0.5M      | ~0.01GB       | FP32      |
| Small  | 768         | 6      | 12    | 8       | 2              | 2048       | ~1B        | ~2GB          | FP16      |
| Medium | 2048        | 22     | 16    | 16      | 2              | 4096       | ~7B        | ~14GB         | Mixed BF16|
| Large  | 4096        | 32     | 32    | 32      | 2              | 4096       | ~30B       | ~60GB         | Mixed BF16|
| XLarge | 8192        | 48     | 64    | 64      | 2              | 8192       | ~200B      | ~400GB        | Mixed BF16|

### Specialized Configurations

| Config | Hidden Size | Layers | Heads | Experts | Active Experts | Seq Length | Parameters | Use Case |
|--------|-------------|--------|-------|---------|----------------|------------|------------|----------|
| Speed Optimized | 512 | 12 | 8 | 4 | 2 | 1024 | ~60M | Real-time applications |
| Memory Optimized | 512 | 8 | 8 | 4 | 2 | 1024 | ~35M | Memory-constrained environments |
| Inference Optimized | 1280 | 16 | 20 | 8 | 2 | 2048 | ~350M | Production inference |
| Quality Focused | 3200 | 26 | 25 | 16 | 2 | 4096 | ~3B | High-quality generation |
| Production | 1536 | 18 | 24 | 12 | 2 | 2048 | ~500M | Production deployments |
| Experimental | 2560 | 24 | 20 | 24 | 2 | 3072 | ~1.5B | Research and experimentation |

*Parameter counts include all model weights. Memory estimates are for model weights only (FP16) and exclude activations, optimizer states, and gradients.*

## Configuration

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--config` | Configuration preset | `medium` |
| `--train-data` | Training data path | Required |
| `--eval-data` | Evaluation data path | Optional |
| `--epochs` | Number of training epochs | `10` |
| `--lr` | Learning rate | `1e-4` |
| `--batch-size` | Batch size per GPU | `2` |
| `--grad-accum` | Gradient accumulation steps | `4` |
| `--precision` | Training precision | `fp16` |
| `--experiment-name` | Experiment identifier | `experiment` |

### Custom Configuration

YAML configuration file example:

```yaml
# config/custom.yaml
model_name: "custom_model"
hidden_size: 1024
num_layers: 24
num_heads: 16
num_experts: 16
num_active_experts: 2
seq_length: 4096
vocab_size: 50304

learning_rate: 1e-4
batch_size: 4
gradient_accumulation_steps: 4
max_grad_norm: 1.0
weight_decay: 0.01

precision: "fp16"
compile: true
gradient_checkpointing: true

lr_scheduler: "cosine"
warmup_ratio: 0.1
min_lr: 1e-6

early_stopping_patience: 5
eval_every_n_batches: 500
save_every_n_batches: 1000
health_check_interval: 100
```

## Monitoring

### Metrics Tracked
- Training loss and perplexity
- Learning rate and gradient norms
- GPU and system memory usage
- Training throughput
- Expert utilization (for MoE)

### Log Output
```
Epoch 1 | Step 150 | Batch 150/2000 | Loss: 2.456789 | PPL: 11.67 | 
LR: 8.50e-05 | GradNorm: 0.8432 | Tokens/s: 1250 | GPU: 8.2GB/16.0GB
```

### Experiment Tracking
Supports Weights & Biases integration:

```bash
export WANDB_PROJECT="conversational-transformer"
export WANDB_ENTITY="your-username"
```

## Utilities

### Data Processing

Process OASST dataset:
```bash
python Src/Main_Scripts/main.py --process-oasst input.jsonl output.jsonl --max-conversations 10000
```

Validate dataset:
```bash
python Src/Main_Scripts/main.py --validate-data data/train.jsonl
```

Create data report:
```bash
python Src/Main_Scripts/main.py --validate-data data/train.jsonl --create-report
```

### Environment

Check training environment:
```bash
python Src/Main_Scripts/main.py --check-environment
```

Estimate training time:
```bash
python Src/Main_Scripts/main.py --estimate-time --config medium
```

## Text Generation

Built-in text generation:

```bash
python Src/Main_Scripts/main.py --test-generation --resume checkpoints/best_model.pt
```

### Generation Parameters
- **Temperature**: Controls randomness (0.1-2.0)
- **Top-k**: Vocabulary limit for sampling (1-100)
- **Top-p**: Nucleus sampling threshold (0.1-1.0)
- **Max Length**: Maximum tokens to generate (1-2048)

## Error Handling

### Recovery Features
- Checkpoint resumption from latest save
- Skip batches with NaN/Inf gradients
- Automatic GPU memory cleanup
- Periodic training state backups

### Validation
- Dataset validation before training
- Corrupted sample handling
- Emergency checkpoint saving
- Training anomaly detection

## Project Structure

```
LuminaAI/
├── Src/
│   └── Main_Scripts/
│       ├── main.py        # Main entry point
│       ├── core/
│       │   ├── model.py           # Transformer with MoE architecture
│       │   ├── tokenizer.py       # Tokenization
│       │   └── dataset.py         # Dataset handling
│       ├── training/
│       │   ├── trainer.py         # Training logic
│       │   ├── orchestrator.py    # Training orchestration
│       │   └── checkpoint.py      # Checkpoint management
│       ├── config/
│       │   └── config_manager.py  # Configuration management
│       ├── monitoring/
│       │   └── logger.py          # Logging and monitoring
│       └── utils/
│           ├── data_processing.py # Data utilities
│           ├── environment.py     # Environment checks
│           └── reporting.py       # Report generation
├── data/                  # Training data
├── checkpoints/           # Model checkpoints
├── experiments/           # Experiment outputs
├── logs/                  # Training logs
└── backups/               # Emergency backups
```

## Troubleshooting

### Common Issues

**CUDA Out of Memory**:
```bash
python Src/Main_Scripts/main.py --batch-size 1 --grad-accum 8
```

**Slow Training**:
```bash
python Src/Main_Scripts/main.py --precision fp16 --compile
```

**Data Loading Errors**:
```bash
python Src/Main_Scripts/main.py --validate-data data/train.jsonl --create-report
```

### Optimization Options

1. Use mixed precision: `--precision fp16`
2. Install flash attention: `pip install flash-attn`
3. Enable model compilation: `--compile` (PyTorch 2.0+)
4. Enable gradient checkpointing for large models
5. Adjust batch size for optimal memory usage

## Performance

### Memory Requirements (A100 80GB)

| Model Size | Parameters | Training Precision | VRAM (weights only) |
| ---------- | ---------- | ------------------ | ------------------- |
| Small      | 500M       | FP16               | ~1GB                |
| Medium     | 4B         | BF16               | ~8GB                |
| Large      | 15B        | BF16               | ~30GB               |

*Training requires additional memory for activations, optimizer states, and gradients (typically 3-4x model size).*

## License

This project is licensed under a Custom License. See license headers in source files for details.

## Support

For issues:
1. Check the troubleshooting section
2. Review existing GitHub issues
3. Create a new issue with logs, configuration, and system information