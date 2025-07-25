# 🤖 LuminaAI: A Character-Level Transformer Chatbot

A powerful character-level Transformer implementation for training conversational AI models from scratch. This project provides a complete pipeline for training, fine-tuning, and chatting with your own character-level language models.

## ✨ Features

- **Character-Level Generation**: Train models that understand text at the character level for fine-grained control
- **Fixed Transformer Architecture**: Proper encoder-based transformer with causal masking (no more decoder confusion!)
- **Flexible Training Pipeline**: Train from scratch or fine-tune existing models
- **Interactive Chat Interface**: Real-time conversation with comprehensive error handling
- **Advanced Sampling Methods**: Support for top-k, nucleus (top-p), greedy, and temperature sampling
- **Robust Memory Management**: Automatic cleanup and OOM protection
- **Easy Dataset Management**: Built-in support for OASST1 dataset and custom data formats
- **GPU Acceleration**: Full support for CUDA, MPS (Apple Silicon), and CPU training
- **Vocabulary Expansion**: Seamlessly handle new characters during fine-tuning
- **Comprehensive Logging**: Detailed logging for debugging and monitoring

## 🚀 Quick Start

### 1. Clone and Setup
```bash
git clone <your-repo-url>
cd character-transformer-chatbot
pip install -r requirements.txt
```

### 2. Download Dataset
```bash
python Dataset_download.py
```
This will download the OpenAssistant (OASST1) dataset to `oasst1_data/`.

### 3. Train Your Model
```bash
python Train.py
```

### 4. Start Chatting
```bash
python ChatAI.py
```

## 📁 Project Structure

```
├── Dataset_download.py    # Downloads OASST1 dataset with validation
├── Train.py              # Main training script (Fixed Transformer)
├── fine_tune.py          # Fine-tuning script with vocab extension
├── ChatAI.py             # Interactive chat interface
├── oasst1_data/          # Dataset directory (created by download script)
├── Model.pth             # Trained model (created by training)
├── FineTuned_Model.pth   # Fine-tuned model (created by fine-tuning)
├── requirements.txt      # Python dependencies
├── LICENSE               # Custom commercial license
├── SECURITY.md          # Security policy
└── README.md            # This file
```

## 🔧 Training Configuration

### Default Hyperparameters
```python
config = {
    'hidden_size': 512,      # Model dimension  
    'seq_length': 512,       # Sequence length for training
    'batch_size': 16,        # Batch size (adjust for GPU memory)
    'num_layers': 6,         # Number of transformer layers
    'nhead': 8,              # Number of attention heads
    'learning_rate': 3e-4,   # Learning rate with warmup
    'epochs': 100,           # Training epochs
    'dropout': 0.1,          # Dropout rate
    'warmup_ratio': 0.1,     # 10% of steps for warmup
    'weight_decay': 0.01     # AdamW weight decay
}
```

### Training Features
- **Warmup + Cosine LR Scheduling**: Stable training with gradual learning rate decay
- **Gradient Clipping**: Prevents exploding gradients (max_norm=1.0)
- **AdamW Optimizer**: Better weight decay handling than standard Adam
- **Automatic Checkpointing**: Saves best model and regular checkpoints
- **Real-time Monitoring**: Loss, accuracy, and sample generation during training
- **Memory Management**: Automatic cleanup and OOM protection
- **Label Smoothing**: Improved training stability

## 💬 Chat Interface

### Basic Usage
```bash
python ChatAI.py
```

### Interactive Commands
- `exit` or `quit` - Exit the chat
- `clear` - Clear conversation history  
- `temp 0.8` - Set temperature (0.1-2.0)
- `topk 10` - Set top-k sampling value (1-50)
- `nucleus` - Switch to nucleus (top-p) sampling
- `topk_mode` - Switch to top-k sampling
- `greedy` - Switch to greedy sampling
- `help` - Show help message

### Example Session
```
🤖 Character-level Transformer AI is ready!

💬 CHAT COMMANDS:
==============================================================
  'exit' or 'quit'     - Exit the chat
  'temp X'             - Set temperature (0.1-2.0, e.g., 'temp 0.8')
  'topk X'             - Set top-k value (1-50, e.g., 'topk 10')
  'nucleus'            - Switch to nucleus sampling
==============================================================

🧑 You: Hello, how are you?
🤖 AI: I'm doing well, thank you for asking! How can I help you today?

🧑 You: temp 1.2
🌡️ Temperature set to 1.2

🧑 You: Tell me a joke
🤖 AI: Why don't scientists trust atoms? Because they make up everything!
```

## 🎯 Fine-Tuning

Fine-tune your pre-trained model on custom data:

### 1. Prepare Your Data
The fine-tuning script supports JSONL format (like OASST) or plain text:

**JSONL Format (recommended):**
```json
{"text": "Your custom conversation", "role": "prompter", "lang": "en"}
{"text": "AI response here", "role": "assistant", "lang": "en"}
```

**Plain Text Format:**
```
<|user|>Your custom question
<|bot|>Your custom response
<|user|>Another question  
<|bot|>Another response
```

### 2. Run Fine-Tuning
```bash
python fine_tune.py
```

### Fine-tuning Features
- **Vocabulary Expansion**: Automatically handles new characters
- **Intelligent Weight Initialization**: Preserves pre-trained weights, properly initializes new parameters
- **Conservative Hyperparameters**: Lower learning rate (1e-4) for stability
- **Comprehensive Error Handling**: Validates data and handles malformed inputs
- **Memory Efficient**: Proper cleanup and OOM protection

## 🔄 Model Architecture

### Fixed Architecture (No More Decoder Confusion!)
```
Input Text → Character Embedding → Positional Encoding → 
TransformerEncoder Layers (with causal masking) → Layer Norm → 
Linear Output → Softmax → Next Character
```

### Key Components
- **Character Embedding**: Maps characters to dense vectors (scaled by √d_model)
- **Positional Encoding**: Sinusoidal position embeddings for sequence understanding
- **Multi-Head Self-Attention**: Captures long-range dependencies with causal masking
- **Feed-Forward Networks**: Point-wise transformations with GELU activation
- **Layer Normalization**: Pre-norm architecture for training stability
- **Dropout Regularization**: Prevents overfitting

## 📊 Sampling Methods

### Temperature Sampling
Controls randomness in generation:
- **Low (0.1-0.7)**: More focused, deterministic
- **Medium (0.7-1.0)**: Balanced creativity  
- **High (1.0-2.0)**: More random, creative

### Top-K Sampling
Considers only the top K most likely tokens:
```python
# Example: top_k=5 considers only 5 most likely characters
🧑 You: topk 5
🔢 Top-k set to 5
```

### Nucleus (Top-P) Sampling
Dynamically adjusts the candidate pool based on cumulative probability:
```python
# Example: top_p=0.9 uses tokens that make up 90% probability mass
🧑 You: nucleus
🎯 Switched to nucleus sampling (p=0.9)
```

### Greedy Sampling
Always picks the most likely token (deterministic):
```python
🧑 You: greedy
🎯 Switched to greedy sampling
```

## 🛠️ Advanced Usage

### Custom Dataset Format
The training script supports multiple data formats:

**Chat Format (OASST style):**
```json
{"text": "Question here", "role": "prompter", "lang": "en", "deleted": false}
{"text": "Response here", "role": "assistant", "lang": "en", "deleted": false}
```

**Plain Text:**
```
Any plain text content for general language modeling
```

### Training from Scratch
Modify hyperparameters in `Train.py`:
```python
config = {
    'hidden_size': 1024,    # Larger model
    'num_layers': 12,       # Deeper network  
    'batch_size': 32,       # Bigger batches
    'epochs': 200,          # Longer training
    'learning_rate': 1e-3   # Higher learning rate
}
```

### Memory Optimization
For limited GPU memory:
```python
config = {
    'batch_size': 8,        # Smaller batches
    'seq_length': 256,      # Shorter sequences
    'hidden_size': 256,     # Smaller model
    'num_layers': 4         # Fewer layers
}
```

## 📈 Monitoring Training

### Training Output
```
2025-01-XX XX:XX:XX - INFO - Starting training with configuration:
2025-01-XX XX:XX:XX - INFO -   hidden_size: 512
2025-01-XX XX:XX:XX - INFO -   batch_size: 16
2025-01-XX XX:XX:XX - INFO - Model parameters: 8,420,352
2025-01-XX XX:XX:XX - INFO - Training steps: 12,500 (warmup: 1,250)

Epoch  10 | Loss: 2.1543 | Accuracy: 45.67% | LR: 2.40e-04 | Time: 125.3s | Elapsed: 1250.5s
Epoch  20 | Loss: 1.8932 | Accuracy: 52.34% | LR: 2.85e-04 | Time: 118.7s | Elapsed: 2367.2s
Epoch  50 | Loss: 1.2543 | Accuracy: 67.89% | LR: 1.50e-04 | Time: 115.2s | Elapsed: 5743.1s

--- Sample Generation ---
<|user|>Hello<|bot|>Hello! I'm doing well, thank you for asking. How can I help you today?
--- End Sample ---
```

### Key Metrics
- **Loss**: Cross-entropy loss with label smoothing (lower is better)
- **Accuracy**: Character-level prediction accuracy  
- **Learning Rate**: Current learning rate (with warmup + cosine scheduling)
- **Sample Quality**: Generated text samples every 10 epochs
- **Memory Usage**: Automatic cleanup prevents OOM errors

## 🔍 Troubleshooting

### Common Issues

**CUDA Out of Memory:**
```python
# In Train.py, reduce batch_size and seq_length
config = {
    'batch_size': 8,      # Reduced from 16
    'seq_length': 256,    # Reduced from 512
}
```

**Model Not Learning (Loss Not Decreasing):**
```python
# Try these adjustments:
config = {
    'learning_rate': 1e-3,    # Higher learning rate
    'epochs': 200,            # Longer training
    'warmup_ratio': 0.05,     # Shorter warmup
}
```

**Poor Generation Quality:**
```python
# In ChatAI.py, adjust sampling parameters:
🧑 You: temp 0.8          # Lower temperature for more focus
🧑 You: topk 5            # Lower top-k for less randomness
🧑 You: nucleus           # Try nucleus sampling
```

**"Model file not found" Error:**
```bash
# Make sure you've trained a model first:
python Train.py

# Check if Model.pth exists:
ls -la *.pth
```

**Vocabulary Issues During Fine-tuning:**
- The fine-tuning script automatically handles new characters
- Check that your data encoding matches (UTF-8 recommended)
- Ensure your JSONL file isn't corrupted

**Memory Leaks During Training:**
- The fixed scripts include automatic memory cleanup
- Monitor GPU memory with `nvidia-smi` (for CUDA)
- Reduce batch size if memory usage grows over time

### Performance Tips

1. **GPU Memory**: 
   - Start with smaller batch_size (8-16) and increase if memory allows
   - Use gradient accumulation for effective larger batches
   
2. **Training Speed**: 
   - Increase batch_size for faster training (if memory allows)
   - Use mixed precision training for newer GPUs
   
3. **Model Quality**: 
   - Larger hidden_size (512-1024) generally improves quality
   - More layers (6-12) can help but increases training time
   
4. **Convergence**: 
   - Use warmup for stable training (10% of total steps)
   - Monitor loss curves - should decrease smoothly

## 🐛 What Was Fixed

### Critical Fixes Applied:
1. **❌ Incorrect TransformerDecoder Usage** → **✅ Proper TransformerEncoder with causal masking**
2. **❌ Memory leaks in training loops** → **✅ Automatic cleanup and garbage collection**
3. **❌ Thread configuration too aggressive** → **✅ Optimized thread settings**
4. **❌ Vocabulary extension bugs** → **✅ Proper weight initialization for new tokens**
5. **❌ Poor error handling** → **✅ Comprehensive exception handling**
6. **❌ Missing input validation** → **✅ Robust data validation and fallbacks**
7. **❌ Nucleus sampling edge cases** → **✅ Improved sampling with fallbacks**
8. **❌ No OOM protection** → **✅ Memory monitoring and cleanup**

### Improvements Added:
- **Logging**: Comprehensive logging throughout all scripts
- **Validation**: Input validation and data integrity checks
- **Documentation**: Detailed inline comments and help messages
- **Error Recovery**: Graceful handling of edge cases
- **Memory Management**: Automatic cleanup and OOM prevention
- **User Experience**: Better CLI interface with help and status messages

## 📚 Technical Details

### Model Specifications
- **Architecture**: Multi-layer Transformer encoder with causal attention
- **Input**: Character-level tokenization
- **Output**: Next-character prediction with causal masking
- **Loss Function**: Cross-entropy loss with label smoothing (0.1)
- **Optimizer**: AdamW with weight decay and beta parameters (0.9, 0.95)
- **Regularization**: Dropout + gradient clipping + weight decay
- **Attention**: Multi-head self-attention with causal masks
- **Activation**: GELU activation functions
- **Normalization**: Pre-norm architecture with LayerNorm

### File Formats
- **Model Files**: PyTorch `.pth` format with complete state and metadata
- **Data Files**: UTF-8 encoded JSONL or text files
- **Checkpoints**: Include model state, optimizer state, scheduler state, and training metadata

### Hardware Requirements
- **Minimum**: CPU with 8GB RAM
- **Recommended**: GPU with 8GB+ VRAM (CUDA or Apple Silicon)
- **Optimal**: GPU with 16GB+ VRAM for larger models

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with proper tests
4. Commit your changes (`git commit -m 'Add amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add comprehensive error handling
- Include proper logging
- Test on multiple devices (CPU/GPU)
- Update documentation for new features

## 📄 License

This project is licensed under a **Custom Commercial Use License**.

### 📋 License Summary
- ✅ **Free for**: Personal, educational, and research use
- ❌ **Restricted**: Commercial use without permission
- 💼 **Commercial licensing**: Available - contact for details

### 🔍 Key Points
- **Non-commercial use** is completely free
- **Commercial use** requires a separate license agreement
- **Educational and research** use is explicitly allowed
- All intellectual property rights remain with Matias Nielsen

### 📞 Commercial Licensing
For commercial use inquiries, please contact:
- **Email**: Matiasnhmb@gmail.com
- **Copyright Holder**: Matias Nielsen

See the full [LICENSE](LICENSE) file for complete terms and conditions.

## 🙏 Acknowledgments

- OpenAssistant team for the OASST1 dataset
- PyTorch team for the excellent deep learning framework
- The broader open-source AI community
- Hugging Face for the datasets library

## 📞 Support

If you encounter issues or have questions:

1. **Check the troubleshooting section** above
2. **Look through existing issues** in the repository
3. **Create a new issue** with detailed information:
   - Your system configuration (OS, GPU, Python version)
   - Complete error messages
   - Steps to reproduce the problem
   - Expected vs actual behavior

### Getting Help
- 🐛 **Bug reports**: Use GitHub issues with the "bug" label
- 💡 **Feature requests**: Use GitHub issues with the "enhancement" label
- ❓ **Questions**: Use GitHub discussions or issues with the "question" label
- 📧 **Security issues**: Email directly to matiasnhmb@gmail.com

## 🔮 Future Roadmap

### Planned Features
- [ ] **Subword Tokenization**: BPE/SentencePiece support for better efficiency
- [ ] **Beam Search**: Multiple candidate generation
- [ ] **Model Quantization**: INT8/FP16 support for deployment
- [ ] **Web Interface**: Gradio/Streamlit GUI
- [ ] **API Server**: REST API for model serving
- [ ] **Multi-turn Memory**: Better conversation context management
- [ ] **ONNX Export**: Cross-platform deployment support
- [ ] **Distributed Training**: Multi-GPU support

### Performance Improvements
- [ ] **Mixed Precision Training**: Faster training on modern GPUs
- [ ] **Gradient Accumulation**: Simulate larger batch sizes
- [ ] **Dynamic Batching**: Efficient inference batching
- [ ] **KV-Cache**: Faster generation with key-value caching

---

**Happy Training! 🚀**

*Built with ❤️ by the LuminaAI team*