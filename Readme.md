# LuminaAI

A production-ready machine learning framework for training conversational transformers with comprehensive monitoring, fault tolerance, and optimization features.

## 🚀 Features

### Core Capabilities
- **Production-Grade Training**: Robust training pipeline with comprehensive error handling
- **Conversational AI**: Specialized for chat/conversation datasets (OASST format support)
- **Mixed Precision Training**: FP16/BF16 support with automatic mixed precision
- **Advanced Architecture**: Grouped Query Attention (GQA), RoPE, SwiGLU, RMSNorm
- **Flash Attention**: Optional flash attention for improved memory efficiency
- **Model Compilation**: PyTorch 2.0+ compilation support for faster training

### Training Features
- **Gradient Accumulation**: Efficient training with large effective batch sizes
- **Learning Rate Scheduling**: Cosine, OneCycle, and Linear schedulers
- **Early Stopping**: Configurable patience-based early stopping
- **Checkpointing**: Automatic and manual checkpoint management
- **Resume Training**: Seamless training resumption from checkpoints

### Monitoring & Reliability
- **Comprehensive Logging**: Detailed training metrics and system monitoring
- **Health Monitoring**: Real-time training health checks and anomaly detection
- **Fault Tolerance**: Automatic recovery from failures with retry logic
- **Memory Management**: GPU memory optimization and monitoring
- **Backup System**: Automatic periodic backups of training state

### Data Processing
- **Enhanced Tokenization**: GPT-4 compatible tokenization with special tokens
- **Data Validation**: Comprehensive dataset validation and quality checks
- **Batch Processing**: Multi-threaded data processing with error handling
- **OASST Support**: Native support for OpenAssistant dataset format

## 📋 Requirements

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

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone https://github.com/MatN23/LuminaAI.git
cd LuminaAI
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

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

The framework includes several pre-configured setups:

- **`debug`**: Small model for testing (fast iteration)
- **`small`**: Lightweight model for experimentation
- **`medium`**: Balanced model for most use cases
- **`large`**: High-capacity model for production

### Data Format

Training data should be in JSONL format with OpenAssistant-style conversations:

```json
{
  "messages": [
    {"role": "user", "content": "Hello, how are you?"},
    {"role": "assistant", "content": "I'm doing well, thank you for asking!"}
  ]
}
```

Supported roles: `user`, `assistant`, `system`, `prompter`, `human`, `ai`, `bot`

## 📊 Model Architecture

### Transformer Architecture
- **Grouped Query Attention (GQA)**: Efficient attention mechanism
- **Rotary Position Embedding (RoPE)**: Advanced positional encoding
- **SwiGLU Activation**: State-of-the-art activation function
- **RMSNorm**: Improved layer normalization
- **Weight Tying**: Shared embedding and output projection weights

### Default Model Sizes

| Config | Hidden Size | Layers | Heads | KV Heads | Seq Length | Parameters | Memory (FP16) | Precision |
|--------|-------------|--------|-------|----------|------------|------------|---------------|-----------|
| Debug  | 128         | 2      | 2     | 1        | 256        | ~0.5M      | ~0.01GB       | FP32      |
| Small  | 768         | 6      | 12    | 4        | 2048       | ~125M      | ~0.25GB       | FP16      |
| Medium | 2048        | 22     | 16    | 8        | 4096       | ~1B        | ~2GB          | Mixed BF16|
| Large  | 4096        | 32     | 32    | 8        | 4096       | ~7B        | ~14GB         | Mixed BF16|

### Additional Specialized Presets

| Config | Hidden Size | Layers | Heads | KV Heads | Seq Length | Parameters | Use Case |
|--------|-------------|--------|-------|----------|------------|------------|----------|
| Speed Optimized | 512 | 12 | 8 | 4 | 1024 | ~60M | Real-time applications |
| Memory Optimized | 512 | 8 | 8 | 4 | 1024 | ~35M | Memory-constrained environments |
| Inference Optimized | 1280 | 16 | 20 | 5 | 2048 | ~350M | Production inference |
| Quality Focused | 3200 | 26 | 25 | 5 | 4096 | ~3B | High-quality generation |
| Production | 1536 | 18 | 24 | 8 | 2048 | ~500M | Production deployments |
| Experimental | 2560 | 24 | 20 | 10 | 3072 | ~1.5B | Research and experimentation |

**Note**: Parameter counts include embedding layers, attention weights, MLP weights, and layer norm parameters. Memory estimates are for model weights only (FP16 precision) and don't include activations, optimizer states, or gradients which can increase memory usage by 3-4x during training.

## ⚙️ Configuration

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

### Advanced Configuration

Create a custom YAML configuration file:

```yaml
# config/custom.yaml
model_name: "custom_model"
hidden_size: 1024
num_layers: 24
num_heads: 16
num_kv_heads: 4
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

# Scheduler options
lr_scheduler: "cosine"
warmup_ratio: 0.1
min_lr: 1e-6

# Training behavior
early_stopping_patience: 5
eval_every_n_batches: 500
save_every_n_batches: 1000
health_check_interval: 100
```

## 📈 Monitoring & Logging

### Built-in Monitoring
- **Training Metrics**: Loss, perplexity, learning rate, gradient norms
- **System Metrics**: GPU memory usage, system memory, throughput
- **Health Checks**: Gradient anomaly detection, loss stability monitoring
- **Checkpointing**: Automatic saving of best models and training state

### Log Output Example
```
Epoch 1 | Step    150 | Batch  150/2000 | Loss: 2.456789 | PPL: 11.67 | 
LR: 8.50e-05 | GradNorm: 0.8432 | Tokens/s: 1250 | GPU: 8.2GB/16.0GB | Health: OK
```

### Experiment Tracking
Integration with Weights & Biases (wandb) for advanced experiment tracking:

```python
# Enable wandb logging
export WANDB_PROJECT="conversational-transformer"
export WANDB_ENTITY="your-username"
```

## 🔧 Utilities

### Data Processing

**Process OASST Dataset**:
```bash
python Src/Main_Scripts/main.py --process-oasst input.jsonl output.jsonl --max-conversations 10000
```

**Validate Dataset**:
```bash
python Src/Main_Scripts/main.py --validate-data data/train.jsonl
```

**Create Data Report**:
```bash
python Src/Main_Scripts/main.py --validate-data data/train.jsonl --create-report
```

### Environment Checks

**Validate Training Environment**:
```bash
python Src/Main_Scripts/main.py --check-environment
```

**Estimate Training Time**:
```bash
python Src/Main_Scripts/main.py --estimate-time --config medium
```

## 🎯 Generation & Inference

The framework includes built-in text generation capabilities:

```python
# After training, test generation
python Src/Main_Scripts/main.py --test-generation --resume checkpoints/best_model.pt
```

### Generation Parameters
- **Temperature**: Controls randomness (0.1-2.0)
- **Top-k**: Limits vocabulary for sampling (1-100)
- **Top-p**: Nucleus sampling threshold (0.1-1.0)
- **Max Length**: Maximum tokens to generate (1-2048)

## 🛡️ Fault Tolerance

### Automatic Recovery
- **Checkpoint Resumption**: Automatically resumes from latest checkpoint
- **Gradient Error Handling**: Skips batches with NaN/Inf gradients
- **Memory Management**: Automatic GPU memory cleanup
- **Training State Backup**: Periodic backups to prevent data loss

### Error Handling
- **Data Validation**: Comprehensive dataset validation before training
- **Graceful Degradation**: Continues training even with some corrupted samples
- **Emergency Checkpoints**: Saves state before crashes
- **Health Monitoring**: Detects and reports training anomalies

## 📁 Project Structure

```
LuminaAI/
├── Src/
│   └── Main_Scripts/
│       ├── main.py        # Main entry point
│       ├── core/
│       │   ├── model.py           # Transformer model architecture
│       │   ├── tokenizer.py       # Enhanced tokenization
│       │   └── dataset.py         # Dataset handling
│       ├── training/
│       │   ├── trainer.py         # Main training logic
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

## 🔍 Troubleshooting

### Common Issues

**CUDA Out of Memory**:
```bash
# Reduce batch size or enable gradient checkpointing
python Src/Main_Scripts/main.py --batch-size 1 --grad-accum 8
```

**Slow Training**:
```bash
# Enable model compilation and mixed precision
python Src/Main_Scripts/main.py --precision fp16 --compile
```

**Data Loading Errors**:
```bash
# Validate and fix dataset
python Src/Main_Scripts/main.py --validate-data data/train.jsonl --create-report
```

### Performance Optimization

1. **Use Mixed Precision**: `--precision fp16`
2. **Enable Flash Attention**: Install `flash-attn`
3. **Model Compilation**: `--compile` (PyTorch 2.0+)
4. **Gradient Checkpointing**: For large models
5. **Optimal Batch Size**: Balance memory and throughput

## 📊 Benchmarks

### Training Throughput (A100 80GB)

| Model Size | Params   | Training Precision | Inference Precision | Approx VRAM (weights only, single GPU) |
| ---------- | -------- | ------------------ | ------------------- | -------------------------------------- |
| Small      | 100M–1B  | FP16 / BF16        | FP16 / INT8         | 0.2–2 GB                               |
| Medium     | 1B–7B    | FP16 / BF16        | FP16 / INT8         | 2–14 GB                                |
| Large      | 7B–20B   | BF16               | FP16 / INT8         | 14–40 GB                               |
| Huge       | 20B–70B  | BF16               | FP16 / INT8         | 40–140 GB                              |
| Massive    | 70B–175B | BF16               | FP16 / INT8         | 140–350+ GB                            |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Submit a pull request

## 📄 License

This project is licensed under a Custom License. See the license headers in source files for details.

## 🙏 Acknowledgments

- OpenAI for the transformer architecture innovations
- Meta AI for RoPE and other architectural improvements
- HuggingFace for tokenization and model architecture inspiration
- The open-source ML community for continuous innovation

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review existing GitHub issues
3. Create a new issue with detailed information
4. Include logs, configuration, and system information

---

**Happy Training! 🚀**