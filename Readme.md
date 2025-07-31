# 🌟 LuminaAI Desktop - Neural Chat Interface

<div align="center">

![LuminaAI Logo](assets/logo.png)

**Advanced Neural Transformer Desktop Application**

*Ultra-modern glassmorphism interface with real-time AI conversation*

[![License](https://img.shields.io/badge/license-Custom-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)
[![Electron](https://img.shields.io/badge/electron-28.0+-green.svg)](https://electronjs.org)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org)

[🚀 Quick Start](#-installation) • [📖 Documentation](#-usage) • [🤝 Community](#-community-forks)

</div>

## ✨ Features

### 🎨 **Ultra-Modern Interface**
- **Glassmorphism Design**: Stunning translucent UI with blur effects
- **Animated Particles**: Dynamic background with neural network visualization  
- **Real-time Animations**: Smooth GSAP-powered transitions and effects
- **Responsive Layout**: Optimized for all screen sizes
- **Dark Theme**: Eye-friendly dark interface with vibrant accents

### 🧠 **Advanced Neural Engine**
- **Model Training**: Complete training pipeline with `train.py`
- **Fine-tuning Support**: Advanced fine-tuning with `fine_tune.py`
- **Real-time Chat**: Interactive chat interface via `ChatAI.py`
- **Desktop Integration**: Native desktop app with `lumina_desktop.py`
- **App Building**: Custom app generation with `buildapp.py`
- **Multiple Sampling Methods**: Top-K, Nucleus (Top-P), and Greedy sampling
- **Memory Management**: Conversation context and history tracking
- **Device Optimization**: CUDA, MPS (Apple Silicon), and CPU support

### 🚀 **Desktop Integration**
- **Native Menus**: Full desktop menu integration
- **File Dialogs**: Native model loading dialogs  
- **Keyboard Shortcuts**: Complete hotkey support
- **Cross-Platform**: Windows, macOS, and Linux support

## 📋 Requirements

### System Requirements
- **Operating System**: Windows 10+, macOS 10.14+, or Linux
- **RAM**: 16GB minimum, 32GB recommended
- **VRAM**: 8GB minimum, 16GB recommended  
- **Storage**: 2GB free space
- **Network**: Internet connection for initial setup

### Software Dependencies
- **Node.js** 16.0+ ([Download](https://nodejs.org/))
- **Python** 3.8+ ([Download](https://python.org/))
- **PyTorch** 2.0+ (automatically installed)

### Optional (for GPU acceleration)
- **CUDA** 11.8+ for NVIDIA GPUs
- **MPS** support for Apple Silicon Macs

## 🛠️ Installation

### Quick Install (Recommended)

```bash
# Clone the repository
git clone https://github.com/MatN23/LuminaAI.git
cd lumina-ai-desktop

# One-command setup
make setup

# Start development
make dev
```

### Manual Installation

```bash
# Install Node.js dependencies  
npm install

# Install Python dependencies
pip install -r requirements.txt

# Or using the development setup
make install-dev
```

### 🐳 Docker Installation

```bash
# Development environment
make docker-dev

# Or production setup
make docker-up
```

## 🚀 Usage

### Core Scripts

Your LuminaAI project includes 5 main Python scripts:

#### 🤖 **AI Training & Fine-tuning**
```bash
# Train a new model
python train.py

# Fine-tune existing model  
python fine_tune.py
```

#### 💬 **Chat Interface**
```bash
# Start chat interface
python ChatAI.py

# Or start full desktop server
python lumina_desktop.py
```

#### 🏗️ **App Building**
```bash
# Build desktop application
python buildapp.py
```

### Starting the Application

```bash
# Start full development environment
make dev

# Start only Python backend
make dev-backend

# Start only Electron frontend  
make dev-frontend

# Using npm directly
npm start
```

### ⌨️ Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl/Cmd + O` | Load neural model |
| `Ctrl/Cmd + K` | Focus message input |
| `Ctrl/Cmd + Enter` | Send message |
| `Ctrl/Cmd + Shift + C` | Clear conversation |
| `Ctrl/Cmd + I` | Show model info |
| `Escape` | Close modal |

## ⚙️ Configuration

### Generation Settings

- **Temperature** (0.1 - 2.0): Controls randomness
  - Low (0.3): More focused, deterministic
  - Medium (0.8): Balanced creativity  
  - High (1.2): More creative, unpredictable

- **Sampling Methods**:
  - **Top-K**: Select from top K most likely tokens
  - **Nucleus (Top-P)**: Dynamic vocabulary based on probability mass
  - **Greedy**: Always select most likely token

- **Max Length**: Maximum response length (25-500 tokens)

### Model Requirements

Your PyTorch model should include:
- `model_state_dict`: The trained model weights
- `config`: Model configuration parameters  
- Corresponding `tokenizer.pkl` file

### 🚀 Submit Your Fork

Have you created something amazing with LuminaAI? We'd love to feature it!

1. **Fork Guidelines**: Ensure your fork follows our [Contributing Guidelines](CONTRIBUTING.md)
2. **Documentation**: Include a comprehensive README
3. **License Compatibility**: Use compatible open-source licenses
4. **Submit PR**: Add your fork to this section via pull request

**Submission Template:**
```markdown
| **[Fork Name](https://github.com/username/fork)** | [@username](https://github.com/username) | Brief description | ![Stars](https://img.shields.io/github/stars/username/fork.svg) |
```

## 🏗️ Project Structure

```
lumina-ai-desktop/
├── 📄 train.py                         # Model training script
├── 📄 fine_tune.py                     # Model fine-tuning script  
├── 📄 ChatAI.py                        # Chat interface backend
├── 📄 lumina_desktop.py                # Main desktop server
├── 📄 buildapp.py                      # Application builder
├── 📁 .github/                         # GitHub automation
│   ├── workflows/                      # CI/CD pipelines
│   └── ISSUE_TEMPLATE/                 # Issue templates
├── 📁 models/                          # Model storage
├── 📁 data/                            # Training data
├── 📁 assets/                          # Static assets
├── 📄 requirements.txt                 # Python dependencies
├── 📄 package.json                     # Node.js configuration
├── 📄 docker-compose.yml               # Docker setup
├── 📄 Makefile                         # Development commands
└── 📄 README.md                        # This file
```

## 🔧 Development

### Development Commands

```bash
# Setup project for first time
make setup

# Start development environment  
make dev

# Run tests
make test

# Build application
make build

# Clean build artifacts
make clean

# Docker development
make docker-dev

# View all commands
make help
```

### Testing

```bash
# Test all Python scripts
make test-python

# Run full test suite
make test

# Test individual scripts
python -c "import train; print('✅ train.py')"
python -c "import fine_tune; print('✅ fine_tune.py')"  
python -c "import ChatAI; print('✅ ChatAI.py')"
python -c "import lumina_desktop; print('✅ lumina_desktop.py')"
python -c "import buildapp; print('✅ buildapp.py')"
```

### Building for Distribution

```bash
# Build for current platform
make build

# Build for all platforms
make build-all

# Build specific platforms
npm run build:win     # Windows
npm run build:mac     # macOS  
npm run build:linux   # Linux
```

## 🚀 Automation

### GitHub Actions

- **Automatic Testing**: Tests all 5 Python scripts on every push
- **Multi-Platform Builds**: Builds Windows, macOS, and Linux versions
- **Automatic Releases**: Creates releases when you push version tags
- **Dependency Updates**: Keeps packages updated via Dependabot

### Releasing

```bash
# Create and push a version tag
git tag v1.0.0
git push --tags

# GitHub Actions will automatically:
# 1. Run all tests
# 2. Build for all platforms  
# 3. Create GitHub release
# 4. Upload installers (.exe, .dmg, .AppImage)
```

## 🤝 Contributing

We welcome contributions from the community!

### 🐛 Reporting Issues

1. **Check existing issues** before creating new ones
2. **Use issue templates** for bugs and feature requests  
3. **Include system information** and error logs
4. **Specify which script** was being used

### 💻 Code Contributions

1. **Fork** the repository
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Test your changes** (`make test`)
4. **Follow code standards** (`make lint`)
5. **Submit a pull request**

### 🔧 Development Setup

```bash
# Fork and clone the repo
git clone https://github.com/YOUR_USERNAME/LuminaAI.git
cd lumina-ai-desktop

# Set up development environment
make setup

# Start developing
make dev
```

## 🆘 Support

### Common Issues

**Model won't load**
- Ensure your `.pth` file includes model configuration
- Check that `tokenizer.pkl` exists in the same directory
- Verify PyTorch is properly installed

**Backend connection failed**  
- Check if Python dependencies are installed (`pip install -r requirements.txt`)
- Ensure no other application is using port 5001
- Try running `python lumina_desktop.py` directly

**Build issues**
- Run `make clean` to clear build artifacts
- Ensure Node.js and Python are properly installed
- Check that all dependencies are installed (`make install`)

### Getting Help

- **Issues**: Report bugs on [GitHub Issues](https://github.com/MatN23/LuminaAI/issues)
- **Discussions**: Use [GitHub Discussions](https://github.com/MatN23/LuminaAI/discussions) for questions

## 📄 License

This project is licensed under a Custom License. See the [LICENSE](LICENSE) file for details.

## 🌟 Acknowledgments

### 🛠️ Built With

- **[Electron](https://electronjs.org/)**: Cross-platform desktop framework
- **[PyTorch](https://pytorch.org/)**: Machine learning framework
- **[Flask](https://flask.palletsprojects.com/)**: Python web framework
- **[Socket.IO](https://socket.io/)**: Real-time communication
- **[GSAP](https://greensock.com/gsap/)**: Advanced animations

---

<div align="center">

### 🧠 Advanced Neural Intelligence • 🎨 Beautiful Interface • 🚀 Desktop Power

**Made with ❤️ by [Matias Nielsen](https://github.com/MatN23) and the LuminaAI Community**

[⭐ Star us on GitHub](https://github.com/MatN23/LuminaAI)

*"The future of AI interaction is here"*

</div>