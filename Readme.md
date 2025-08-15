# Production-Ready Conversational Transformer Training System

A modular, production-ready system for training conversational transformers with comprehensive monitoring, fault tolerance, and debugging capabilities.

## 🚀 Features

- **Modular Architecture**: Split into logical modules for better debugging and maintenance
- **Production Monitoring**: Comprehensive logging, metrics tracking, and health monitoring
- **Fault Tolerance**: Automatic recovery, checkpointing, and graceful error handling
- **Flexible Configuration**: Multiple presets and easy customization
- **Enhanced Stability**: Numerical stability improvements and gradient monitoring
- **Multi-Backend Logging**: Support for Wandb, TensorBoard, and structured file logging

## 📁 Project Structure

```
├── Main.py                    # Entry point
├── config/
│   └── config_manager.py      # Configuration management
├── core/
│   ├── tokenizer.py          # Tokenizer implementation
│   ├── model.py              # Model architecture
│   └── dataset.py            # Dataset handling
├── training/
│   ├── orchestrator.py       # Training orchestration
│   ├── trainer.py            # Main trainer class
│   ├── training_loop.py      # Training loop implementation
│   └── checkpoint.py         # Checkpoint management
├── monitoring/
│   └── logger.py             # Production logging
├── utils/
│   ├── data_processing.py    # Data utilities
│   ├── environment.py        # Environment validation
│   └── reporting.py          # Report generation
├── requirements.txt          # Dependencies
├── setup.py                  # Setup script
└── README.md                 # This file
```

## 🛠️ Installation

1. **Clone and setup**:
```bash
git clone https://github.com/MatN23/LuminaAI.git
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run setup**:
```bash
python setup.py
```

This will create the directory structure, validate dependencies, and create sample data.

## 🚀 Quick Start

### 1. Debug Run (Recommended for first time)
```bash
python Main.py --config debug --test-generation
```

### 2. Small Model Training
```bash
python Main.py --config small --train-data oasst1_data/oasst1_train.jsonl --eval-data data/eval.jsonl
```

### 3. Production Training
```bash
python Main.py --config medium --epochs 10 --experiment-name my_experiment
```

## 📊 Configuration Presets

| Preset | Use Case | Parameters | Memory |
|--------|----------|------------|---------|
| `debug` | Testing & debugging | 6M params | ~2GB |
| `small` | Limited resources | 50M params | ~8GB |
| `medium` | Serious training | 400M params | ~16GB |
| `large` | High-end training | 1.2B params | ~32GB |

## 🔧 Advanced Usage

### Custom Configuration
```bash
python Main.py --config small --lr 1e-4 --batch-size 8 --epochs 5
```

### Resume Training
```bash
python Main.py --config medium --resume checkpoints/model_epoch_005.pt
```

### Data Processing
```bash
# Process OASST data
python Main.py --process-oasst raw_data.jsonl processed_data.jsonl

# Validate data
python Main.py --validate-data data/train.jsonl --create-report
```

### Environment Check
```bash
python Main.py --check-environment --estimate-time
```

## 📈 Monitoring

The system provides comprehensive monitoring through multiple backends:

- **File Logging**: Structured logs in `logs/`
- **Metrics**: JSONL metrics in `logs/{experiment}/metrics.jsonl`
- **Wandb**: Real-time experiment tracking (optional)
- **TensorBoard**: Local visualization (optional)

## 🛡️ Fault Tolerance

- **Automatic Checkpointing**: Regular model saves
- **Health Monitoring**: Detects training issues
- **Graceful Recovery**: Resume from last checkpoint
- **Signal Handling**: Clean shutdown on interruption

## 🔍 Debugging Features

- **Modular Design**: Easy to isolate and debug individual components
- **Comprehensive Logging**: Detailed logs for every operation
- **Error Handling**: Graceful error recovery with detailed messages
- **Memory Monitoring**: Track GPU and system memory usage
- **Generation Testing**: Built-in response generation testing

## 📝 Data Format

The system expects JSONL files with this format:

```json
{
  "conversation_id": "conv_001",
  "messages": [
    {"role": "user", "content": "Hello, how are you?"},
    {"role": "assistant", "content": "I'm doing well, thank you!"}
  ],
  "metadata": {
    "source": "dataset_name"
  }
}
```

## 🚨 Troubleshooting

### Common Issues

1. **Import Errors**: Run `python setup.py` to check dependencies
2. **CUDA Out of Memory**: Reduce `batch_size` or use `gradient_checkpointing`
3. **Data Loading Errors**: Validate data with `--validate-data`
4. **Training Instability**: Check logs for NaN gradients or losses

### Debug Mode
For debugging, always start with:
```bash
python Main.py --config debug --test-generation
```

### Memory Issues
```bash
# Check system resources
python Main.py --check-environment

# Use smaller batch size
python Main.py --config small --batch-size 2 --grad-accum 8
```

## 📊 Performance Tips

1. **Use Mixed Precision**: `--precision fp16` or `bf16`
2. **Enable Compilation**: `--compile` (PyTorch 2.0+)
3. **Optimize Batch Size**: Find the largest that fits in memory
4. **Use Flash Attention**: Install `flash-attn` for longer sequences

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with proper error handling
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI for GPT architecture insights
- Hugging Face for tokenizer implementations
- PyTorch team for the excellent framework
- Open Assistant for conversation datasets