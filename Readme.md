# 🌟 LuminaAI Desktop - Neural Chat Interface

<div align="center">

![LuminaAI Logo](assets/logo.png)

**Advanced Neural Transformer Desktop Application**

*Ultra-modern glassmorphism interface with real-time AI conversation*

[![License](https://img.shields.io/badge/license-Custom-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)
[![Electron](https://img.shields.io/badge/electron-28.0+-green.svg)](https://electronjs.org)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org)

</div>

## ✨ Features

### 🎨 **Ultra-Modern Interface**
- **Glassmorphism Design**: Stunning translucent UI with blur effects
- **Animated Particles**: Dynamic background with neural network visualization
- **Real-time Animations**: Smooth GSAP-powered transitions and effects
- **Responsive Layout**: Optimized for all screen sizes
- **Dark Theme**: Eye-friendly dark interface with vibrant accents

### 🧠 **Advanced Neural Engine**
- **Word-Level Tokenization**: Sophisticated text processing
- **Multiple Sampling Methods**: Top-K, Nucleus (Top-P), and Greedy sampling
- **Real-time Generation**: Live typing indicators and streaming responses
- **Memory Management**: Conversation context and history tracking
- **Device Optimization**: CUDA, MPS (Apple Silicon), and CPU support

### 🚀 **Desktop Integration**
- **Native Menus**: Full desktop menu integration
- **File Dialogs**: Native model loading dialogs
- **Keyboard Shortcuts**: Complete hotkey support
- **System Notifications**: Toast notifications and status updates
- **Cross-Platform**: Windows, macOS, and Linux support

## 📋 Requirements

### System Requirements
- **Operating System**: Windows 10+, macOS 10.14+, or Linux
- **RAM**: 16GB minimum, 32GB recommended
- **Storage**: 5GB free space
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
# Clone or download the project
git clone https://github.com/MatN23/LuminaAI.git
cd lumina-ai-desktop

# Run installation script
chmod +x install.sh
./install.sh

# Or on Windows
install.bat


### Manual Installation

```bash
# Install Node.js dependencies
npm install

# Install Python dependencies
pip install torch numpy flask flask-socketio flask-cors

# Or using requirements.txt
pip install -r requirements.txt
```

## 🚀 Usage

### Starting the Application

```bash
# Start LuminaAI Desktop
npm start

# Or for development mode
npm run dev
```

### Loading a Model

1. **Prepare your PyTorch model**: Ensure you have a `.pth` file and corresponding `tokenizer.pkl`
2. **Launch LuminaAI**: Start the application
3. **Load Model**: Click "Load Model" or use `Ctrl/Cmd + O`
4. **Select File**: Choose your model file
5. **Start Chatting**: Begin your neural conversation!

### Keyboard Shortcuts

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

## 🏗️ Project Structure

```
lumina-ai-desktop/
├── main.js                 # Electron main process
├── lumina_desktop.py      # Python backend server
├── package.json           # Node.js dependencies
├── requirements.txt       # Python dependencies
├── renderer/              # Frontend files
│   └── index.html        # Main UI
├── assets/               # Icons and images
├── models/              # Model storage
├── logs/               # Application logs
└── dist/              # Built applications
```

## 🔧 Development

### Running in Development Mode

```bash
# Start with development tools
npm run dev

# This enables:
# - DevTools access
# - Hot reload
# - Debug logging
# - Development menu options
```

### Building for Distribution

```bash
# Build for current platform
npm run build

# Build for all platforms
npm run dist

# Package without building installer
npm run pack
```

### Customization

The interface can be customized by modifying:
- `renderer/index.html`: UI structure and styling
- `main.js`: Electron configuration and menus
- `lumina_desktop.py`: Backend API and AI logic

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under a Custom License. See the [LICENSE](LICENSE) file for details.

## 🆘 Support

### Common Issues

**Model won't load**
- Ensure your `.pth` file includes model configuration
- Check that `tokenizer.pkl` exists in the same directory
- Verify PyTorch is properly installed

**Backend connection failed**
- Check if Python dependencies are installed
- Ensure no other application is using port 5001
- Verify firewall settings

**Performance issues**
- Close unnecessary applications
- Use GPU acceleration if available
- Reduce max response length
- Lower temperature for faster generation

### Getting Help

- **Issues**: Report bugs on GitHub Issues
- **Discussions**: Join community discussions
- **Documentation**: Check the wiki for detailed guides

## 🌟 Acknowledgments

- Built with [Electron](https://electronjs.org/) for cross-platform desktop support
- Powered by [PyTorch](https://pytorch.org/) for neural inference
- UI animations by [GSAP](https://greensock.com/gsap/)
- Real-time communication via [Socket.IO](https://socket.io/)

---

<div align="center">

**🧠 Advanced Neural Intelligence • 🎨 Beautiful Interface • 🚀 Desktop Power**

*Created with ❤️ by Matias Nielsen*

</div>
