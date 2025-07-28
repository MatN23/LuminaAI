# 🤖 LuminaAI: A Word-Level Transformer Chatbot with Desktop App

A powerful word-level Transformer implementation for training conversational AI models with a sleek desktop interface. This project provides a complete pipeline for training, fine-tuning, and chatting with your own word-level language models through an elegant desktop application.

## ✨ Features

- **Word-Level Generation**: Train models that understand text at the word level for better semantic understanding
- **Desktop Application**: Beautiful, user-friendly desktop interface built with modern UI components
- **Fixed Transformer Architecture**: Proper encoder-based transformer with causal masking
- **Flexible Training Pipeline**: Train from scratch or fine-tune existing models
- **Interactive Chat Interface**: Real-time conversation with comprehensive error handling
- **Advanced Sampling Methods**: Support for top-k, nucleus (top-p), greedy, and temperature sampling
- **Robust Memory Management**: Automatic cleanup and OOM protection
- **Easy Dataset Management**: Built-in support for OASST1 dataset and custom data formats
- **GPU Acceleration**: Full support for CUDA, MPS (Apple Silicon), and CPU training
- **Vocabulary Expansion**: Seamlessly handle new words during fine-tuning
- **Comprehensive Logging**: Detailed logging for debugging and monitoring
- **Professional Desktop UI**: Modern interface with customizable themes and settings

## 🚀 Quick Start

### 1. Clone and Setup
```bash
git clone https://github.com/MatN23/LuminaAI.git
cd lumina-ai-chatbot
pip install -r requirements.txt (There may be dependencies that are required not there)
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

### 4. Launch Desktop Application
```bash
python BuildApp.py
```

### 5. Alternative: Command Line Chat
```bash
python ChatAI.py
```

## 📁 Project Structure

```
├── Dataset_download.py    # Downloads OASST1 dataset with validation
├── Train.py              # Main training script (Word-level Transformer)
├── fine_tune.py          # Fine-tuning script with vocab extension
├── ChatAI.py             # Command-line chat interface
├── BuildApp.py           # Desktop application launcher
├── lumina_desktop.py     # Desktop app customization and UI components
├── oasst1_data/          # Dataset directory (created by download script)
├── Model.pth             # Trained model (created by training)
├── FineTuned_Model.pth   # Fine-tuned model (created by fine-tuning)
├── requirements.txt      # Python dependencies
├── LICENSE               # Custom commercial license
├── SECURITY.md          # Security policy
└── README.md            # This file
```

## 🖥️ Desktop Application

### Features
- **Modern UI**: Clean, intuitive interface with professional styling
- **Real-time Chat**: Instant responses with typing indicators
- **Theme Customization**: Multiple themes and color schemes
- **Settings Panel**: Easy configuration of model parameters
- **Chat History**: Persistent conversation history
- **Export Options**: Save conversations and model outputs
- **System Tray**: Minimize to system tray for easy access
- **Performance Monitoring**: Real-time model performance metrics

### Desktop App Usage
```bash
python BuildApp.py
```

### Customization
The desktop interface can be customized through `lumina_desktop.py`:
- **Themes**: Modify color schemes and UI elements
- **Layout**: Adjust window layouts and component positioning
- **Features**: Add custom buttons, panels, and functionality
- **Styling**: CSS-like styling for professional appearance

### Desktop Controls
- **Model Settings**: Adjust temperature, top-k, sampling methods
- **Chat Management**: Clear history, export conversations
- **Theme Selector**: Switch between light/dark themes
- **Performance Panel**: Monitor GPU usage, inference speed
- **Advanced Options**: Fine-tuning parameters, model switching

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
    'weight_decay': 0.01,    # AdamW weight decay
    'vocab_size': 50000      # Word vocabulary size
}
```

### Training Features
- **Word-Level Tokenization**: Better semantic understanding than character level
- **Warmup + Cosine LR Scheduling**: Stable training with gradual learning rate decay
- **Gradient Clipping**: Prevents exploding gradients (max_norm=1.0)
- **AdamW Optimizer**: Better weight decay handling than standard Adam
- **Automatic Checkpointing**: Saves best model and regular checkpoints
- **Real-time Monitoring**: Loss, accuracy, and sample generation during training
- **Memory Management**: Automatic cleanup and OOM protection
- **Label Smoothing**: Improved training stability

## 💬 Chat Interfaces

### Desktop Application
Launch the modern desktop interface:
```bash
python BuildApp.py
```

**Desktop Features:**
- Beautiful, responsive UI with modern design
- Real-time parameter adjustment with sliders
- Visual feedback and animations
- Chat history with search functionality
- Export and import conversation logs
- System integration and notifications

### Command Line Interface
```bash
python ChatAI.py
```

**CLI Commands:**
- `exit` or `quit` - Exit the chat
- `clear` - Clear conversation history  
- `temp 0.8` - Set temperature (0.1-2.0)
- `topk 10` - Set top-k sampling value (1-50)
- `nucleus` - Switch to nucleus (top-p) sampling
- `topk_mode` - Switch to top-k sampling
- `greedy` - Switch to greedy sampling
- `help` - Show help message

### Example Desktop Session
The desktop app provides:
- **Visual Parameter Controls**: Sliders for temperature, top-k values
- **Sampling Method Buttons**: Easy switching between sampling strategies
- **Real-time Feedback**: Visual indicators for model state and performance
- **Professional Styling**: Clean, modern interface that's easy on the eyes
- **Responsive Design**: Adapts to different screen sizes and resolutions

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
- **Word Vocabulary Expansion**: Automatically handles new words and terms
- **Intelligent Weight Initialization**: Preserves pre-trained weights, properly initializes new parameters
- **Conservative Hyperparameters**: Lower learning rate (1e-4) for stability
- **Comprehensive Error Handling**: Validates data and handles malformed inputs
- **Memory Efficient**: Proper cleanup and OOM protection

## 🔄 Model Architecture

### Word-Level Architecture
```
Input Text → Word Tokenization → Word Embedding → Positional Encoding → 
TransformerEncoder Layers (with causal masking) → Layer Norm → 
Linear Output → Softmax → Next Word
```

### Key Components
- **Word Tokenization**: Maps text to word-level tokens for better semantic understanding
- **Word Embedding**: Maps words to dense vectors (scaled by √d_model)
- **Positional Encoding**: Sinusoidal position embeddings for sequence understanding
- **Multi-Head Self-Attention**: Captures long-range dependencies with causal masking
- **Feed-Forward Networks**: Point-wise transformations with GELU activation
- **Layer Normalization**: Pre-norm architecture for training stability
- **Dropout Regularization**: Prevents overfitting

### Advantages of Word-Level Processing
- **Better Semantics**: Words carry more semantic meaning than characters
- **Faster Training**: Shorter sequences lead to faster training
- **Improved Quality**: Better understanding of language structure
- **Efficient Memory**: Reduced sequence lengths for same content
- **Natural Boundaries**: Respects natural language word boundaries

## 📊 Sampling Methods

### Temperature Sampling
Controls randomness in generation:
- **Low (0.1-0.7)**: More focused, deterministic
- **Medium (0.7-1.0)**: Balanced creativity  
- **High (1.0-2.0)**: More random, creative

### Top-K Sampling
Considers only the top K most likely words:
```python
# Desktop: Use slider to adjust top-k value (1-50)
# CLI: topk 5
```

### Nucleus (Top-P) Sampling
Dynamically adjusts the candidate pool based on cumulative probability:
```python
# Desktop: Toggle nucleus sampling button
# CLI: nucleus
```

### Greedy Sampling
Always picks the most likely word (deterministic):
```python
# Desktop: Select greedy mode button
# CLI: greedy
```

## 🖥️ Desktop Application Details

### UI Components (lumina_desktop.py)
- **Chat Panel**: Main conversation area with message bubbles
- **Control Panel**: Parameter adjustment sliders and buttons
- **Settings Dialog**: Advanced configuration options
- **Theme Manager**: Light/dark theme switching
- **Status Bar**: Real-time model status and performance
- **Menu System**: File operations, help, and settings

### Customization Options
```python
# In lumina_desktop.py, customize:
- Color schemes and themes
- Window layouts and sizing
- Custom buttons and controls
- Font styles and sizes
- Animation effects
- Notification settings
```

### Desktop Features
- **Responsive Design**: Adapts to different screen resolutions
- **Keyboard Shortcuts**: Quick access to common functions
- **Auto-save**: Automatic saving of chat history and settings
- **Multi-monitor Support**: Works across multiple displays
- **System Integration**: Native OS integration and notifications
- **Performance Optimized**: Smooth 60fps UI with efficient rendering

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
    'learning_rate': 1e-3,  # Higher learning rate
    'vocab_size': 100000    # Larger vocabulary
}
```

### Memory Optimization
For limited GPU memory:
```python
config = {
    'batch_size': 8,        # Smaller batches
    'seq_length': 256,      # Shorter sequences
    'hidden_size': 256,     # Smaller model
    'num_layers': 4,        # Fewer layers
    'vocab_size': 25000     # Smaller vocabulary
}
```

## 📈 Monitoring Training

### Training Output
```
2025-01-XX XX:XX:XX - INFO - Starting word-level training with configuration:
2025-01-XX XX:XX:XX - INFO -   hidden_size: 512
2025-01-XX XX:XX:XX - INFO -   vocab_size: 50000
2025-01-XX XX:XX:XX - INFO -   batch_size: 16
2025-01-XX XX:XX:XX - INFO - Model parameters: 12,845,312
2025-01-XX XX:XX:XX - INFO - Training steps: 8,750 (warmup: 875)

Epoch  10 | Loss: 1.8432 | Accuracy: 52.34% | LR: 2.40e-04 | Time: 95.3s | Elapsed: 953.5s
Epoch  20 | Loss: 1.5621 | Accuracy: 61.45% | LR: 2.85e-04 | Time: 87.2s | Elapsed: 1825.7s
Epoch  50 | Loss: 0.9876 | Accuracy: 78.92% | LR: 1.50e-04 | Time: 89.8s | Elapsed: 4476.3s

--- Sample Generation ---
<|user|>Hello<|bot|>Hello! I'm doing well, thank you for asking. How can I help you today?
--- End Sample ---
```

### Key Metrics
- **Loss**: Cross-entropy loss with label smoothing (lower is better)
- **Accuracy**: Word-level prediction accuracy  
- **Learning Rate**: Current learning rate (with warmup + cosine scheduling)
- **Sample Quality**: Generated text samples every 10 epochs
- **Memory Usage**: Automatic cleanup prevents OOM errors
- **Vocabulary Coverage**: Percentage of dataset vocabulary learned

## 🔍 Troubleshooting

### Common Issues

**Desktop App Won't Launch:**
```bash
# Check dependencies
pip install -r requirements.txt

# Verify BuildApp.py exists
ls -la BuildApp.py

# Check Python version (3.8+ recommended)
python --version
```

**CUDA Out of Memory:**
```python
# In Train.py, reduce batch_size and seq_length
config = {
    'batch_size': 8,      # Reduced from 16
    'seq_length': 256,    # Reduced from 512
    'vocab_size': 25000   # Smaller vocabulary
}
```

**Model Not Learning (Loss Not Decreasing):**
```python
# Try these adjustments:
config = {
    'learning_rate': 1e-3,    # Higher learning rate
    'epochs': 200,            # Longer training
    'warmup_ratio': 0.05,     # Shorter warmup
    'vocab_size': 30000       # Adjust vocabulary size
}
```

**Poor Generation Quality:**
- **Desktop**: Use sliders to adjust temperature (0.7-0.9) and top-k (10-20)
- **CLI**: `temp 0.8` and `topk 15` for balanced results
- Try nucleus sampling for more diverse outputs

**Desktop UI Issues:**
```python
# In lumina_desktop.py, check:
- Screen resolution compatibility
- Theme settings
- Font availability
- Graphics driver updates
```

**Vocabulary Issues During Fine-tuning:**
- The fine-tuning script automatically handles new words
- Check that your data encoding matches (UTF-8 recommended)
- Ensure your JSONL file isn't corrupted
- Monitor vocabulary size growth during training

## 🐛 What Was Fixed & Improved

### Major Updates:
1. **✅ Word-Level Tokenization**: Upgraded from character-level to word-level processing
2. **✅ Desktop Application**: Added beautiful desktop interface with BuildApp.py
3. **✅ UI Customization**: Added lumina_desktop.py for interface customization
4. **✅ Better Semantics**: Word-level understanding for improved responses
5. **✅ Professional Interface**: Modern, responsive desktop application
6. **✅ Enhanced User Experience**: Visual controls and real-time feedback

### Previous Fixes Maintained:
1. **✅ Proper TransformerEncoder with causal masking**
2. **✅ Automatic cleanup and garbage collection**
3. **✅ Optimized thread settings**
4. **✅ Robust weight initialization for new tokens**
5. **✅ Comprehensive exception handling**
6. **✅ Robust data validation and fallbacks**
7. **✅ Improved sampling with fallbacks**
8. **✅ Memory monitoring and cleanup**

## 📚 Technical Details

### Model Specifications
- **Architecture**: Multi-layer Transformer encoder with causal attention
- **Input**: Word-level tokenization with configurable vocabulary size
- **Output**: Next-word prediction with causal masking
- **Vocabulary**: Configurable word vocabulary (default: 50,000 words)
- **Loss Function**: Cross-entropy loss with label smoothing (0.1)
- **Optimizer**: AdamW with weight decay and beta parameters (0.9, 0.95)
- **Regularization**: Dropout + gradient clipping + weight decay
- **Attention**: Multi-head self-attention with causal masks
- **Activation**: GELU activation functions
- **Normalization**: Pre-norm architecture with LayerNorm

### Desktop Application Stack
- **UI Framework**: Modern Python GUI framework
- **Styling**: CSS-like styling system
- **Threading**: Non-blocking UI with background model inference
- **State Management**: Reactive state management for real-time updates
- **Performance**: Optimized rendering for smooth 60fps experience

### File Formats
- **Model Files**: PyTorch `.pth` format with complete state and metadata
- **Data Files**: UTF-8 encoded JSONL or text files
- **Checkpoints**: Include model state, optimizer state, scheduler state, and training metadata
- **UI Config**: JSON configuration files for desktop app settings

### Hardware Requirements
- **Minimum**: CPU with 8GB RAM, integrated graphics
- **Recommended**: GPU with 8GB+ VRAM (CUDA or Apple Silicon)
- **Optimal**: GPU with 16GB+ VRAM for larger models and smooth desktop experience

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with proper tests
4. Test both CLI and desktop interfaces
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Test both command-line and desktop interfaces
- Add comprehensive error handling
- Include proper logging
- Test on multiple devices (CPU/GPU)
- Update documentation for new features
- Ensure desktop UI remains responsive

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
- Desktop UI framework contributors

## 📞 Support

If you encounter issues or have questions:

1. **Check the troubleshooting section** above
2. **Look through existing issues** in the repository
3. **Create a new issue** with detailed information:
   - Your system configuration (OS, GPU, Python version)
   - Complete error messages (CLI and/or desktop)
   - Steps to reproduce the problem
   - Expected vs actual behavior
   - Screenshots for desktop UI issues

### Getting Help
- 🐛 **Bug reports**: Use GitHub issues with the "bug" label
- 💡 **Feature requests**: Use GitHub issues with the "enhancement" label
- ❓ **Questions**: Use GitHub discussions or issues with the "question" label
- 🖥️ **Desktop UI issues**: Include screenshots and system info
- 📧 **Security issues**: Email directly to matiasnhmb@gmail.com

## 🔮 Future Roadmap

### Planned Features
- [ ] **Mobile App**: Extend desktop app to mobile platforms
- [ ] **Cloud Sync**: Synchronize conversations across devices
- [ ] **Plugin System**: Extensible plugin architecture
- [ ] **Voice Integration**: Speech-to-text and text-to-speech
- [ ] **Multi-language UI**: Localization for different languages
- [ ] **Advanced Themes**: More customization options
- [ ] **Model Marketplace**: Easy downloading of pre-trained models
- [ ] **Collaborative Features**: Share conversations and models

### Performance Improvements
- [ ] **Mixed Precision Training**: Faster training on modern GPUs
- [ ] **Gradient Accumulation**: Simulate larger batch sizes
- [ ] **Dynamic Batching**: Efficient inference batching
- [ ] **KV-Cache**: Faster generation with key-value caching
- [ ] **Model Quantization**: INT8/FP16 support for deployment
- [ ] **Beam Search**: Multiple candidate generation

### Desktop Enhancements
- [ ] **Advanced Analytics**: Detailed conversation analytics
- [ ] **Custom Widgets**: User-created UI components
- [ ] **Automation**: Scheduled tasks and automated responses
- [ ] **Integration APIs**: Connect with external services
- [ ] **Advanced Theming**: Visual theme editor
- [ ] **Performance Dashboard**: Real-time system monitoring

---

**Happy Training & Chatting! 🚀✨**