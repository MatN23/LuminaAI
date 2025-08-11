# 🌌 LuminaAI Desktop
### Neural Transformer Chat Interface with Glassmorphic UI

<div align="center">

![LuminaAI Banner](assets/banner.png)

**🧠 Advanced Neural Architecture • 🎨 Glassmorphic Interface • ⚡ Real-time Intelligence**

[![License](https://img.shields.io/badge/license-Custom-ff6b6b.svg?style=for-the-badge)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-4dabf7.svg?style=for-the-badge&logo=python)](https://python.org)
[![Electron](https://img.shields.io/badge/electron-28.0+-69db7c.svg?style=for-the-badge&logo=electron)](https://electronjs.org)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-ee5a52.svg?style=for-the-badge&logo=pytorch)](https://pytorch.org)
[![Neural Network](https://img.shields.io/badge/AI-Neural_Transformers-9775fa.svg?style=for-the-badge)]()

*Experience the future of human-AI interaction through a revolutionary desktop interface*

[🚀 Quick Start](#-quick-start) • [🎯 Features](#-key-features) • [🛠️ Installation](#️-installation) • [🧠 Training](#-neural-training) • [📚 Documentation](#-documentation)

</div>

---

## 🎯 Key Features

### 🎨 **Revolutionary Glassmorphic Interface**
- **Advanced Glassmorphism**: Multi-layer translucent effects with dynamic blur and refraction
- **Neural Particle System**: Real-time animated neural network visualization with 10,000+ particles
- **Fluid Animations**: 120fps GSAP-powered micro-interactions and morphing transitions
- **Adaptive Theming**: Dynamic color schemes that respond to conversation context
- **Immersive UX**: Cinematic loading sequences and context-aware visual feedback

### 🧠 **State-of-the-Art Neural Engine**
- **Subword Tokenization**: Advanced BPE (Byte-Pair Encoding) for superior language understanding
- **Multi-Modal Sampling**: Temperature-controlled Top-K, Nucleus (Top-P), and adaptive sampling strategies
- **Streaming Intelligence**: Real-time token generation with typing indicators and thinking animations  
- **Memory Architecture**: Hierarchical conversation context with attention-based history management
- **Hardware Acceleration**: Optimized inference pipelines for CUDA, Metal Performance Shaders (MPS), and multi-core CPU

### 🚀 **Native Desktop Excellence**
- **Deep System Integration**: Platform-native menus, dialogs, and keyboard shortcuts
- **Universal Compatibility**: Seamless operation across Windows 10+, macOS 10.14+, and Linux distributions
- **Performance Monitoring**: Real-time inference metrics, memory usage, and GPU utilization
- **Smart Notifications**: Contextual alerts and conversation status updates
- **File System Integration**: Drag-and-drop model loading and conversation exports

---

## 📊 Technical Specifications

<div align="center">

| Component | Specification | Details |
|-----------|---------------|---------|
| **Neural Architecture** | Transformer-based | Custom-trained on conversational datasets |
| **Tokenization** | Subword BPE | Vocabulary size: 32,000+ tokens |
| **Inference Engine** | PyTorch 2.0+ | Optimized with TorchScript compilation |
| **Frontend** | Electron 28+ | Chromium-based with native APIs |
| **Backend** | Flask + SocketIO | Real-time WebSocket communication |
| **Animation Engine** | GSAP 3.12+ | Hardware-accelerated transitions |
| **Minimum RAM** | 16GB | 32GB recommended for large models |
| **Storage** | 30GB+ | SSD recommended for optimal performance |

</div>

---

## 🛠️ Installation

### ⚡ Automated Setup (Recommended)

```bash
# Clone the revolutionary AI interface
git clone https://github.com/MatN23/LuminaAI.git
cd lumina-ai-desktop

# Linux/macOS - One-command installation
chmod +x install.sh && ./install.sh

# Windows - PowerShell execution
./install.bat
```

### 🔧 Manual Installation

```bash
# 1. Install Node.js dependencies
npm install --production

# 2. Create Python virtual environment (recommended)
python -m venv lumina_env
source lumina_env/bin/activate  # Linux/macOS
# or
lumina_env\Scripts\activate     # Windows

# 3. Install Python dependencies with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install flask flask-socketio flask-cors numpy transformers

# 4. Install additional dependencies
pip install -r requirements.txt
```

### 🐳 Docker Deployment (Advanced)

```bash
# Build the container
docker build -t lumina-ai .

# Run with GPU support
docker run --gpus all -p 5001:5001 -p 3000:3000 lumina-ai
```

---

## 🧠 Neural Training

### 📚 Dataset Preparation

LuminaAI uses the high-quality OASST1 (OpenAssistant) conversational dataset for training:

```bash
# Download and prepare training data
wget https://huggingface.co/datasets/OpenAssistant/oasst1/resolve/main/oasst1_train.jsonl
mkdir -p oasst1_data
mv oasst1_train.jsonl oasst1_data/
```

### 🚀 Training Your Custom Model

```bash
# Start neural training with advanced configuration
python Train.py \
    --data oasst1_data/oasst1_train.jsonl \
    --model_size medium \
    --batch_size 32 \
    --learning_rate 3e-4 \
    --max_epochs 10 \
    --gradient_accumulation 4 \
    --tokenizer_type subword \
    --vocab_size 32000 \
    --save_every 1000 \
    --validate_every 500 \
    --mixed_precision \
    --compile_model
```

### ⚙️ Training Configuration Options

| Parameter | Description | Default | Advanced Options |
|-----------|-------------|---------|------------------|
| `--data` | Training dataset path | Required | Supports JSONL, CSV, Parquet |
| `--model_size` | Architecture scale | `medium` | `small`, `medium`, `large`, `xl` |
| `--tokenizer_type` | Tokenization method | `subword` | `word`, `char`, `sentencepiece` |
| `--vocab_size` | Vocabulary size | `32000` | `16000`, `32000`, `64000` |
| `--mixed_precision` | Use FP16 training | `False` | Significantly faster training |
| `--compile_model` | PyTorch 2.0 compilation | `False` | Up to 2x performance boost |

### 📈 Training Monitoring

Monitor your training progress with built-in visualization:

```bash
# View training metrics in real-time
python -m tensorboard.main --logdir=./logs --port=6006
```

---

## 🚀 Quick Start

### 1. Launch LuminaAI

```bash
# Start the neural interface
npm start

# Development mode with debugging
npm run dev
```

### 2. Load Your Trained Model

1. **Prepare Model Files**: Ensure you have your trained `.pth` file and `tokenizer.pkl`
2. **Open LuminaAI**: Launch the application
3. **Load Neural Model**: Click "Load Model" or press `Ctrl/Cmd + O`
4. **Select Files**: Choose your model file (tokenizer will be auto-detected)
5. **Begin Conversation**: Start interacting with your custom AI!

### 3. Advanced Configuration

Access the settings panel to fine-tune generation parameters:

- **Temperature Control**: 0.1 (deterministic) → 2.0 (creative chaos)
- **Top-K Sampling**: Limit token selection to top K candidates
- **Nucleus Sampling**: Dynamic vocabulary based on cumulative probability
- **Response Length**: 25-500 tokens with smart truncation
- **Memory Depth**: Conversation context window size

---

## ⌨️ Keyboard Shortcuts & Controls

<div align="center">

| Shortcut | Action | Description |
|----------|--------|-------------|
| `Ctrl/Cmd + O` | Load Model | Open neural model selection dialog |
| `Ctrl/Cmd + R` | Reload Interface | Refresh the glassmorphic interface |
| `Ctrl/Cmd + K` | Focus Input | Quick focus to message input field |
| `Ctrl/Cmd + Enter` | Send Message | Submit message to neural engine |
| `Ctrl/Cmd + Shift + C` | Clear Chat | Reset conversation history |
| `Ctrl/Cmd + I` | Model Info | Display neural architecture details |
| `Ctrl/Cmd + ,` | Settings | Open configuration panel |
| `F11` | Fullscreen | Toggle immersive fullscreen mode |
| `Ctrl/Cmd + D` | Developer Tools | Open Chromium DevTools |
| `Escape` | Close Modal | Dismiss any open dialog |

</div>

---

## 🔬 Advanced Features

### 🧪 **Neural Architecture Insights**
- **Attention Visualization**: See what your model focuses on during generation
- **Token Probability Analysis**: Real-time probability distributions for each token
- **Layer Activation Maps**: Deep neural network introspection tools
- **Performance Profiling**: Detailed inference timing and memory analysis

### 🎛️ **Advanced Sampling Techniques**
- **Contrastive Search**: Balanced between quality and diversity
- **Typical Sampling**: Information-theoretic token selection
- **Repetition Penalties**: Dynamic suppression of repetitive content
- **Custom Stopping Criteria**: Configurable end-of-sequence detection

### 📊 **Model Management**
- **Multi-Model Support**: Switch between different trained models instantly
- **Model Comparison Mode**: A/B test different model versions
- **Checkpoint Loading**: Resume from any training checkpoint
- **Model Quantization**: INT8/FP16 optimization for faster inference

---

## 🚨 Troubleshooting & Performance

### ⚡ **Optimization Tips**

```bash
# Enable maximum performance mode
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=true
```

### 🔧 **Common Solutions**

**🚫 Model Loading Issues**
```bash
# Verify model structure
python -c "import torch; print(torch.load('model.pth').keys())"

# Check tokenizer compatibility
python -c "import pickle; print(len(pickle.load(open('tokenizer.pkl', 'rb'))))"
```

**⚡ Performance Optimization**
- **GPU Memory**: Reduce batch size if CUDA out-of-memory errors occur
- **CPU Performance**: Enable multi-threading with `torch.set_num_threads()`
- **M1/M2 Macs**: Ensure MPS backend is available with `torch.backends.mps.is_available()`

**🔌 Backend Connection Issues**
```bash
# Check port availability
netstat -an | grep 5001

# Restart backend service
pkill -f "python.*lumina_desktop.py" && python lumina_desktop.py
```

---

## 🏗️ Development & Building

### 🛠️ Development Environment

```bash
# Start development server with hot reload
npm run dev

# Enable advanced debugging
DEBUG=lumina:* npm run dev

# Run with specific Electron version
npm run dev -- --electron-version=28.0.0
```

### 📦 Production Building

```bash
# Build for current platform
npm run build

# Cross-platform compilation
npm run dist:win     # Windows
npm run dist:mac     # macOS
npm run dist:linux   # Linux

# Create portable package
npm run pack
```

### 🧪 Testing & Quality Assurance

```bash
# Run comprehensive test suite
npm test

# Neural engine unit tests
python -m pytest tests/neural/

# End-to-end interface testing
npm run test:e2e

# Performance benchmarking
npm run benchmark
```

---

## 📚 Documentation

### 🔗 **Essential Links**
- **[🎯 API Reference](docs/API.md)** - Complete backend API documentation
- **[🧠 Model Architecture](docs/ARCHITECTURE.md)** - Neural network design details
- **[⚙️ Configuration Guide](docs/CONFIG.md)** - Advanced setup and tuning
- **[🚀 Deployment Guide](docs/DEPLOYMENT.md)** - Production deployment strategies
- **[🔬 Research Papers](docs/RESEARCH.md)** - Academic foundations and citations

### 🤝 **Community & Support**
- **[💬 Discord Server](https://discord.gg/lumina-ai)** - Real-time community support
- **[📖 GitHub Discussions](https://github.com/MatN23/LuminaAI/discussions)** - Feature requests and ideas
- **[🐛 Issue Tracker](https://github.com/MatN23/LuminaAI/issues)** - Bug reports and fixes
- **[📝 Wiki](https://github.com/MatN23/LuminaAI/wiki)** - Comprehensive guides and tutorials

---

## 🌟 Acknowledgments & Credits

<div align="center">

**Built on the shoulders of giants**

🧠 **[PyTorch](https://pytorch.org/)** - Deep learning infrastructure  
⚡ **[Electron](https://electronjs.org/)** - Cross-platform desktop framework  
🎨 **[GSAP](https://greensock.com/)** - Premium animation library  
🔄 **[Socket.IO](https://socket.io/)** - Real-time communication  
📚 **[OpenAssistant](https://open-assistant.io/)** - High-quality conversational datasets  
🤗 **[Hugging Face](https://huggingface.co/)** - Transformers and tokenization libraries  

**Special thanks to the open-source AI community for advancing the boundaries of human-machine interaction**

</div>

---

<div align="center">

## 🚀 Ready to Experience the Future?

**[⬇️ Download LuminaAI](https://github.com/MatN23/LuminaAI/releases/latest)**

*🌌 Advanced Neural Intelligence • 🎨 Glassmorphic Beauty • ⚡ Desktop Performance*

**Created with 💜 and ☕ by [Matias Nielsen](https://github.com/MatN23)**

[![Star this project](https://img.shields.io/github/stars/MatN23/LuminaAI?style=social)](https://github.com/MatN23/LuminaAI)
[![Follow the creator](https://img.shields.io/github/followers/MatN23?style=social)](https://github.com/MatN23)

</div>