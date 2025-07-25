# 🤖 LuminaAI: A charecter-level Chatbot

A powerful character-level Transformer implementation for training conversational AI models from scratch. This project provides a complete pipeline for training, fine-tuning, and chatting with your own character-level language models.

## ✨ Features

- **Character-Level Generation**: Train models that understand text at the character level for fine-grained control
- **Flexible Training Pipeline**: Train from scratch or fine-tune existing models
- **Interactive Chat Interface**: Real-time conversation with your trained models
- **Advanced Sampling Methods**: Support for top-k, nucleus (top-p), and temperature sampling
- **Robust Architecture**: Transformer decoder with causal masking, dropout, and proper regularization
- **Easy Dataset Management**: Built-in support for OASST1 dataset and custom data formats
- **GPU Acceleration**: Full support for CUDA, MPS (Apple Silicon), and CPU training
- **Vocabulary Expansion**: Seamlessly handle new characters during fine-tuning

## 🚀 Quick Start

### 1. Clone and Setup
```bash
git clone <your-repo-url>
cd character-lstm-chatbot
pip install torch numpy
```

### 2. Download Dataset
```bash
python Dataset_Download.py
```
This will download the OpenAssistant (OASST1) dataset to `oasst1_data/`.

### 3. Train Your Model
```bash
python improved_transformer.py  # Main training script
```

### 4. Start Chatting
```bash
python improved_chat_script.py
```

## 📁 Project Structure

```
├── Dataset_Download.py          # Downloads OASST1 dataset
├── improved_transformer.py      # Main training script (Transformer)
├── improved_chat_script.py      # Interactive chat interface (for LSTM)
├── improved_finetune_script.py  # Fine-tuning script (for LSTM)
├── oasst1_data/                 # Dataset directory (created by Dataset_Download)
├── best_model.pth               # Trained Transformer model
└── README.md                    # This file
```

## 🔧 Training Configuration

### Default Hyperparameters
```python
hidden_size = 512      # LSTM hidden dimension
seq_length = 512       # Sequence length for training
batch_size = 16        # Batch size (adjust based on GPU memory)
num_layers = 6         # Number of LSTM layers
learning_rate = 3e-4   # Learning rate with warmup
epochs = 100           # Training epochs
dropout = 0.1          # Dropout rate for regularization
```

### Training Features
- **Warmup + Cosine LR Scheduling**: Stable training with gradual learning rate decay
- **Gradient Clipping**: Prevents exploding gradients (max_norm=1.0)
- **AdamW Optimizer**: Better weight decay handling than standard Adam
- **Automatic Checkpointing**: Saves best model and regular checkpoints
- **Real-time Monitoring**: Loss, accuracy, and sample generation during training

## 💬 Chat Interface

### Basic Usage
```bash
python improved_chat_script.py
```

### Interactive Commands
- `exit` or `quit` - Exit the chat
- `clear` - Clear conversation history
- `temp 0.8` - Set temperature (0.1-2.0)
- `topk 10` - Set top-k sampling value
- `nucleus` - Switch to nucleus (top-p) sampling
- `topk_mode` - Switch back to top-k sampling

### Example Session
```
🤖 Chat with the AI!
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
Create `fine_tune_data.txt` with your custom conversation data:
```
### User: Your custom question
### Bot: Your custom response
### User: Another question
### Bot: Another response
```

### 2. Run Fine-Tuning
```bash
python improved_finetune_script.py
```

### Features
- **Vocabulary Expansion**: Automatically handles new characters
- **Intelligent Initialization**: Preserves pre-trained weights, initializes new parameters
- **Conservative Hyperparameters**: Lower learning rate and shorter training for stability
- **Multiple Save Points**: Best model, checkpoints, and final model

## 🔄 Model Architecture

```
Input Text → Character Embedding → Positional Encoding → 
Transformer Decoder Layers (with causal masking) → Layer Norm → 
Linear Output → Softmax → Next Character
```

### Key Components
- **Character Embedding**: Maps characters to dense vectors (scaled by √d_model)
- **Positional Encoding**: Sinusoidal position embeddings for sequence understanding
- **Multi-Head Self-Attention**: Captures long-range dependencies with causal masking
- **Feed-Forward Networks**: Point-wise transformations in each layer
- **Layer Normalization**: Stabilizes training and improves convergence
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
sample_text(model, prompt, top_k=5)
```

### Nucleus (Top-P) Sampling
Dynamically adjusts the candidate pool:
```python
# Example: top_p=0.9 uses tokens that make up 90% probability mass
sample_text(model, prompt, sampling_method="nucleus", top_p=0.9)
```

## 🛠️ Advanced Usage

### Custom Dataset Format
The training script supports multiple data formats:

**Chat Format:**
```
<|user|>Question here<|bot|>Response here
<|user|>Another question<|bot|>Another response
```

**Plain Text:**
```
Any plain text content for general language modeling
```

### Training from Scratch
```python
# Modify improved_transformer.py hyperparameters
hidden_size = 1024     # Larger model
num_layers = 8         # Deeper network
batch_size = 32        # Bigger batches
epochs = 200           # Longer training
```

### Memory Optimization
For limited GPU memory:
```python
batch_size = 8         # Smaller batches
seq_length = 256       # Shorter sequences
hidden_size = 256      # Smaller model
```

## 📈 Monitoring Training

### Training Output
```
Epoch  50 | Loss: 1.2543 | Accuracy: 67.89% | LR: 1.50e-04 | Time: 45.2s | ETA: 203.1s

--- Sample Generation ---
<|user|>Hello<|bot|>Hello! I'm doing well, thank you for asking. How can I help you today?
--- End Sample ---
```

### Key Metrics
- **Loss**: Cross-entropy loss (lower is better)
- **Accuracy**: Character-level prediction accuracy
- **Learning Rate**: Current learning rate (with scheduling)
- **Sample Quality**: Generated text samples during training

## 🔍 Troubleshooting

### Common Issues

**CUDA Out of Memory:**
```python
# Reduce batch_size and seq_length
batch_size = 8
seq_length = 256
```

**Model Not Learning:**
```python
# Try higher learning rate or longer training
learning_rate = 1e-3
epochs = 200
```

**Poor Generation Quality:**
```python
# Try different sampling parameters
temperature = 0.8
top_k = 10
# Or switch to nucleus sampling
sampling_method = "nucleus"
top_p = 0.9
```

**Vocabulary Issues During Fine-tuning:**
- The fine-tuning script automatically handles new characters
- Check that your data encoding matches (UTF-8 recommended)

### Performance Tips

1. **GPU Memory**: Reduce `batch_size` and `seq_length` if you run out of memory
2. **Training Speed**: Increase `batch_size` for faster training (if memory allows)
3. **Model Quality**: Larger `hidden_size` and more `num_layers` generally improve quality
4. **Convergence**: Use learning rate warmup for stable training

## 📚 Technical Details

### Model Specifications
- **Architecture**: Multi-layer Transformer decoder with causal attention
- **Input**: Character-level tokenization
- **Output**: Next-character prediction with causal masking
- **Loss Function**: Cross-entropy loss
- **Optimizer**: AdamW with weight decay
- **Regularization**: Dropout + gradient clipping
- **Attention**: Multi-head self-attention with causal masks

### File Formats
- **Model Files**: PyTorch `.pth` format with complete state
- **Data Files**: UTF-8 encoded text files
- **Checkpoints**: Include model state, optimizer state, and metadata

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

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

See the full LICENSE file for complete terms and conditions.

## 🙏 Acknowledgments

- OpenAssistant team for the OASST1 dataset
- PyTorch team for the excellent deep learning framework
- The broader open-source AI community

## 📞 Support

If you encounter issues or have questions:
1. Check the troubleshooting section above
2. Look through existing issues in the repository
3. Create a new issue with detailed information about your problem

---

**Happy Training! 🚀**
