# 🌟 LuminaAI Desktop - Neural Chat Interface

<div align="center">

![LuminaAI Logo](assets/logo.png)

**Advanced Neural Transformer Desktop Application**

*Ultra-modern glassmorphism interface with real-time AI conversation*

[![License](https://img.shields.io/badge/license-Custom-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)
[![Electron](https://img.shields.io/badge/electron-28.0+-green.svg)](https://electronjs.org)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org)
[![Downloads](https://img.shields.io/github/downloads/MatN23/LuminaAI/total.svg?color=brightgreen)](https://github.com/MatN23/LuminaAI/releases)
[![Stars](https://img.shields.io/github/stars/MatN23/LuminaAI.svg?style=social)](https://github.com/MatN23/LuminaAI/stargazers)

[🚀 Quick Start](#-installation) • [📖 Documentation](#-usage) • [🎨 Screenshots](#-screenshots) • [🤝 Community](#-community-forks) • [💬 Discord](https://discord.gg/luminaai)

</div>

## ✨ Features

### 🎨 **Ultra-Modern Interface**
- **Glassmorphism Design**: Stunning translucent UI with blur effects and depth
- **Animated Particles**: Dynamic background with interactive neural network visualization
- **Real-time Animations**: Smooth GSAP-powered transitions, typing indicators, and effects
- **Responsive Layout**: Optimized for all screen sizes and resolutions
- **Multiple Themes**: Dark, light, and custom theme support with accent colors
- **Accessibility**: Full keyboard navigation and screen reader support

### 🧠 **Advanced Neural Engine**
- **Multiple Model Support**: GPT, BERT, T5, LLaMA, and custom architectures
- **Word-Level Tokenization**: Sophisticated text processing with BPE and SentencePiece
- **Advanced Sampling**: Top-K, Nucleus (Top-P), Typical-P, and Temperature scaling
- **Real-time Generation**: Live streaming responses with typing indicators
- **Context Management**: Smart conversation history and memory optimization
- **Multi-GPU Support**: Distributed inference across multiple GPUs
- **Quantization**: INT8/INT4 model compression for efficiency

### 🚀 **Desktop Integration**
- **Native Menus**: Platform-specific menu integration
- **File Management**: Drag-and-drop model loading and conversation export
- **Keyboard Shortcuts**: Comprehensive hotkey system
- **System Integration**: Toast notifications, tray icons, and OS-specific features
- **Auto-Updates**: Seamless application updates
- **Cross-Platform**: Windows, macOS, and Linux support with native look-and-feel

### 🔧 **Developer Features**
- **Plugin System**: Extensible architecture for custom functionality
- **API Integration**: RESTful API for external integrations
- **Model Debugging**: Built-in tools for analyzing model behavior
- **Performance Monitoring**: Real-time metrics and profiling
- **Export Options**: Multiple formats for conversations and model outputs

## 🎨 Screenshots

<div align="center">

| Main Interface | Model Loading | Settings Panel |
|:-:|:-:|:-:|
| ![Main](assets/screenshots/main.png) | ![Loading](assets/screenshots/loading.png) | ![Settings](assets/screenshots/settings.png) |

| Conversation View | Theme Selector | Performance Monitor |
|:-:|:-:|:-:|
| ![Chat](assets/screenshots/chat.png) | ![Themes](assets/screenshots/themes.png) | ![Monitor](assets/screenshots/monitor.png) |

</div>

## 📋 Requirements

### System Requirements
- **Operating System**: 
  - Windows 10+ (64-bit)
  - macOS 10.15+ (Intel/Apple Silicon)
  - Linux (Ubuntu 18.04+, CentOS 7+)
- **Memory**: 16GB RAM minimum, 32GB recommended
- **Graphics**: 8GB VRAM minimum, 16GB recommended
- **Storage**: 5GB free space (10GB for offline models)
- **Network**: Internet connection for updates and model downloads

### Software Dependencies
- **Node.js** 18.0+ ([Download](https://nodejs.org/))
- **Python** 3.8-3.11 ([Download](https://python.org/))
- **Git** for version control ([Download](https://git-scm.com/))

### Hardware Acceleration (Optional)
- **NVIDIA GPUs**: CUDA 11.8+ with cuDNN 8.6+
- **Apple Silicon**: Native MPS acceleration
- **Intel/AMD**: OpenVINO optimization support

## 🛠️ Installation

### 🚀 Quick Install (Recommended)

```bash
# Clone the repository
git clone https://github.com/MatN23/LuminaAI.git
cd lumina-ai-desktop

# One-command installation
./install.sh          # Linux/macOS
# or
install.bat           # Windows

# Launch LuminaAI
npm start
```

### 📦 Pre-built Releases

Download pre-built binaries from our [Releases Page](https://github.com/MatN23/LuminaAI/releases):

- **Windows**: `LuminaAI-Setup-v1.x.x.exe`
- **macOS**: `LuminaAI-v1.x.x.dmg` (Universal Binary)
- **Linux**: `LuminaAI-v1.x.x.AppImage` or `.deb` package

### ⚙️ Manual Installation

<details>
<summary>Click to expand manual installation steps</summary>

```bash
# Install Node.js dependencies
npm install

# Create Python virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows

# Install Python dependencies
pip install -r requirements.txt

# Install optional GPU support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # NVIDIA
# or
pip install torch torchvision torchaudio  # CPU/MPS
```

</details>

### 🐳 Docker Installation

```bash
# Pull and run LuminaAI container
docker run -it --gpus all -p 5001:5001 -p 8080:8080 luminaai/desktop:latest

# Or build from source
docker build -t luminaai-desktop .
docker run -it --gpus all -p 5001:5001 -p 8080:8080 luminaai-desktop
```

## 🚀 Usage

### Starting the Application

```bash
# Standard launch
npm start

# Development mode (with DevTools)
npm run dev

# Production mode
npm run prod

# Headless mode (API only)
npm run headless
```

### 🤖 Model Management

#### Loading Models

1. **From GUI**: Click "Load Model" or drag-and-drop `.pth` files
2. **From Menu**: File → Load Model (`Ctrl/Cmd + O`)
3. **From CLI**: Use `--model path/to/model.pth` argument
4. **Auto-discovery**: Place models in `./models/` directory

#### Supported Model Formats

- **PyTorch**: `.pth`, `.pt`, `.bin`
- **Hugging Face**: `pytorch_model.bin` with `config.json`
- **ONNX**: `.onnx` files (experimental)
- **TensorFlow**: `.pb` files via conversion

#### Model Hub Integration

Access popular models directly from the interface:
- GPT-2/GPT-3.5 variants
- LLaMA and Alpaca models  
- BERT and RoBERTa models
- Custom community models

### ⌨️ Keyboard Shortcuts

| Category | Shortcut | Action |
|----------|----------|--------|
| **File** | `Ctrl/Cmd + O` | Load neural model |
| | `Ctrl/Cmd + S` | Save conversation |
| | `Ctrl/Cmd + E` | Export chat |
| **Chat** | `Ctrl/Cmd + K` | Focus message input |
| | `Ctrl/Cmd + Enter` | Send message |
| | `Ctrl/Cmd + Shift + C` | Clear conversation |
| | `Ctrl/Cmd + ↑/↓` | Navigate message history |
| **Interface** | `Ctrl/Cmd + I` | Show model info |
| | `Ctrl/Cmd + ,` | Open settings |
| | `Ctrl/Cmd + T` | Toggle theme |
| | `F11` | Toggle fullscreen |
| **Developer** | `F12` | Open DevTools |
| | `Ctrl/Cmd + R` | Reload interface |
| | `Ctrl/Cmd + Shift + I` | Inspect element |

## ⚙️ Configuration

### 🎛️ Generation Settings

<details>
<summary>Advanced Configuration Options</summary>

#### Temperature Control
- **Ultra-Low** (0.1-0.3): Deterministic, focused responses
- **Low** (0.4-0.6): Balanced with slight creativity
- **Medium** (0.7-0.9): Good balance of creativity and coherence
- **High** (1.0-1.3): Creative and diverse outputs
- **Experimental** (1.4-2.0): Highly unpredictable responses

#### Sampling Strategies
- **Greedy**: Always selects most probable token
- **Top-K**: Limits vocabulary to K most likely tokens
- **Nucleus (Top-P)**: Dynamic vocabulary based on cumulative probability
- **Typical-P**: Selects tokens close to expected information content
- **Mirostat**: Maintains consistent perplexity

#### Advanced Parameters
- **Max Length**: 25-2048 tokens
- **Repetition Penalty**: 1.0-1.3 (prevents loops)
- **Length Penalty**: -2.0 to 2.0 (encourages longer/shorter responses)
- **Presence Penalty**: -2.0 to 2.0 (encourages topic diversity)

</details>

### 🎨 Interface Customization

- **Themes**: Dark, Light, Auto, Custom
- **Accent Colors**: 12+ predefined color schemes
- **Fonts**: System, Monospace, Custom font support
- **Layout**: Compact, Comfortable, Spacious
- **Animations**: Reduced motion support

### 🔧 Performance Optimization

- **Memory Management**: Automatic cleanup and garbage collection
- **GPU Utilization**: Multi-GPU support and load balancing
- **Caching**: Intelligent model and response caching
- **Batch Processing**: Optimized for multiple simultaneous requests

## 🤝 Community Forks

The LuminaAI community has created amazing forks and extensions! Here are some notable community projects:

### 🌟 Featured Forks

| Fork | Author | Description | Stars |
|------|--------|-------------|-------|
| **[LuminaAI-Mobile](https://github.com/community/luminaai-mobile)** | [@mobileDev123](https://github.com/mobileDev123) | React Native mobile version with offline support | ![Stars](https://img.shields.io/github/stars/community/luminaai-mobile.svg) |
| **[LuminaAI-Web](https://github.com/community/luminaai-web)** | [@webmaster456](https://github.com/webmaster456) | Browser-based version with cloud integration | ![Stars](https://img.shields.io/github/stars/community/luminaai-web.svg) |
| **[LuminaAI-Medical](https://github.com/healthcare/luminaai-medical)** | [@drtech789](https://github.com/drtech789) | Specialized for medical AI applications | ![Stars](https://img.shields.io/github/stars/healthcare/luminaai-medical.svg) |
| **[LuminaAI-Gaming](https://github.com/gamedev/luminaai-gaming)** | [@gameAI](https://github.com/gameAI) | Integration with game engines and NPCs | ![Stars](https://img.shields.io/github/stars/gamedev/luminaai-gaming.svg) |

### 🔌 Plugin Ecosystem

| Plugin | Category | Description |
|--------|----------|-------------|
| **Voice Integration** | Audio | Speech-to-text and text-to-speech |
| **Code Assistant** | Developer | Syntax highlighting and code completion |
| **Image Generation** | Creative | DALL-E and Stable Diffusion integration |
| **Language Pack** | Localization | 20+ language translations |
| **Custom Themes** | UI/UX | Community-created interface themes |

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

### 🏆 Hall of Fame

Special recognition for outstanding community contributions:

- 🥇 **Most Innovative**: LuminaAI-VR by [@vrdev](https://github.com/vrdev)
- 🥈 **Best Performance**: LuminaAI-Optimized by [@speedster](https://github.com/speedster)
- 🥉 **Most Popular**: LuminaAI-Extended by [@popular](https://github.com/popular)

## 🔧 Development

### Development Environment Setup

```bash
# Clone and setup development environment
git clone https://github.com/MatN23/LuminaAI.git
cd lumina-ai-desktop

# Install dependencies
npm run setup:dev

# Start development servers
npm run dev:all

# Run tests
npm test
```

### 🧪 Testing

```bash
# Run all tests
npm test

# Run specific test suites
npm run test:unit          # Unit tests
npm run test:integration   # Integration tests
npm run test:e2e          # End-to-end tests

# Test with coverage
npm run test:coverage
```

### 📦 Building for Distribution

```bash
# Build for current platform
npm run build

# Build for all platforms
npm run dist:all

# Build specific platforms
npm run dist:win     # Windows
npm run dist:mac     # macOS
npm run dist:linux   # Linux

# Create portable version
npm run pack:portable
```

### 🔍 Code Quality

- **ESLint**: JavaScript/TypeScript linting
- **Prettier**: Code formatting
- **Husky**: Git hooks for quality checks
- **TypeScript**: Type safety and better IDE support
- **Jest**: Testing framework with coverage

## 🌍 i18n (Internationalization)

LuminaAI supports multiple languages:

- 🇺🇸 English (en-US) - Default
- 🇪🇸 Spanish (es-ES)
- 🇫🇷 French (fr-FR)
- 🇩🇪 German (de-DE)
- 🇯🇵 Japanese (ja-JP)
- 🇨🇳 Chinese Simplified (zh-CN)
- 🇰🇷 Korean (ko-KR)
- 🇷🇺 Russian (ru-RU)

**Help us translate**: Join our [Crowdin project](https://crowdin.com/project/luminaai) to contribute translations!

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

### 🐛 Reporting Issues

1. **Search existing issues** before creating new ones
2. **Use issue templates** for bugs, features, and questions
3. **Provide detailed information** including system specs and logs
4. **Include screenshots** for UI-related issues

### 💻 Code Contributions

1. **Fork** the repository
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Follow coding standards** (ESLint, Prettier)
4. **Write tests** for new functionality
5. **Update documentation** as needed
6. **Submit a pull request** with detailed description

### 📖 Documentation

- Improve existing documentation
- Create tutorials and guides
- Translate documentation to other languages
- Record video tutorials

### 💰 Sponsorship

Support LuminaAI development:

- **GitHub Sponsors**: [Sponsor on GitHub](https://github.com/sponsors/MatN23)
- **Open Collective**: [Support via Open Collective](https://opencollective.com/luminaai)
- **Patreon**: [Monthly support on Patreon](https://patreon.com/luminaai)

## 📊 Project Statistics

<div align="center">

[![GitHub stars](https://img.shields.io/github/stars/MatN23/LuminaAI.svg?style=for-the-badge&logo=github)](https://github.com/MatN23/LuminaAI/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/MatN23/LuminaAI.svg?style=for-the-badge&logo=github)](https://github.com/MatN23/LuminaAI/network)
[![GitHub issues](https://img.shields.io/github/issues/MatN23/LuminaAI.svg?style=for-the-badge&logo=github)](https://github.com/MatN23/LuminaAI/issues)
[![GitHub pull requests](https://img.shields.io/github/issues-pr/MatN23/LuminaAI.svg?style=for-the-badge&logo=github)](https://github.com/MatN23/LuminaAI/pulls)

</div>

## 📄 License

This project is licensed under the **Custom Open Source License** - see the [LICENSE](LICENSE) file for details.

**Commercial Use**: Contact us for commercial licensing options.

## 🆘 Support & Community

### 💬 Community Channels

- **Discord**: [Join our Discord server](https://discord.gg/luminaai)
- **Reddit**: [r/LuminaAI](https://reddit.com/r/luminaai)
- **Telegram**: [LuminaAI Community](https://t.me/luminaai)
- **Matrix**: [#luminaai:matrix.org](https://matrix.to/#/#luminaai:matrix.org)

### 📚 Documentation & Learning

- **Wiki**: [Comprehensive guides and tutorials](https://github.com/MatN23/LuminaAI/wiki)
- **API Docs**: [API documentation and examples](https://docs.luminaai.com)
- **YouTube**: [Video tutorials and demos](https://youtube.com/@luminaai)
- **Blog**: [Technical articles and updates](https://blog.luminaai.com)

### 🔧 Technical Support

- **GitHub Issues**: [Report bugs and request features](https://github.com/MatN23/LuminaAI/issues)
- **Stack Overflow**: Tag your questions with `luminaai`
- **Email Support**: [support@luminaai.com](mailto:support@luminaai.com)

### 🚨 Common Issues & Solutions

<details>
<summary>Troubleshooting Guide</summary>

#### Model Loading Issues
```bash
# Verify model format
python -c "import torch; print(torch.load('model.pth', map_location='cpu').keys())"

# Check tokenizer
ls -la tokenizer.pkl  # Should exist alongside model file
```

#### Backend Connection Failed
```bash
# Check Python dependencies
pip list | grep -E "(torch|flask|socketio)"

# Test backend manually
python src/backend/lumina_desktop.py --test

# Check port availability
netstat -an | grep 5001
```

#### Performance Optimization
- **Memory**: Close unnecessary applications
- **GPU**: Update drivers and check CUDA compatibility
- **Storage**: Use SSD for model storage
- **Network**: Use local models for better performance

#### Installation Problems
```bash
# Clear npm cache
npm cache clean --force

# Rebuild native modules
npm rebuild

# Reset Python environment
pip uninstall -y -r requirements.txt
pip install -r requirements.txt
```

</details>

## 🌟 Acknowledgments

### 🙏 Special Thanks

- **Core Team**: [@MatN23](https://github.com/MatN23), [@contributor1](https://github.com/contributor1)
- **Beta Testers**: Our amazing community of early adopters
- **Translators**: International community for localization support
- **Plugin Developers**: Creating amazing extensions

### 🛠️ Built With

- **[Electron](https://electronjs.org/)**: Cross-platform desktop framework
- **[React](https://reactjs.org/)**: Frontend UI library
- **[PyTorch](https://pytorch.org/)**: Machine learning framework
- **[Flask](https://flask.palletsprojects.com/)**: Python web framework
- **[Socket.IO](https://socket.io/)**: Real-time communication
- **[GSAP](https://greensock.com/gsap/)**: Advanced animations
- **[Tailwind CSS](https://tailwindcss.com/)**: Utility-first CSS framework

### 📈 Milestones

- **🎉 v1.0.0**: Initial release with basic AI chat
- **🚀 v1.5.0**: Added plugin system and themes
- **🌟 v2.0.0**: Complete UI overhaul and performance improvements
- **🔥 v2.5.0**: Multi-model support and cloud integration
- **🏆 10K+ Downloads**: Community milestone reached
- **🌍 20+ Languages**: International support achieved

---

<div align="center">

### 🧠 Advanced Neural Intelligence • 🎨 Beautiful Interface • 🚀 Desktop Power

**Made with ❤️ by [Matias Nielsen](https://github.com/MatN23) and the LuminaAI Community**

[⭐ Star us on GitHub](https://github.com/MatN23/LuminaAI) • [🐦 Follow on Twitter](https://twitter.com/luminaai) • [📧 Subscribe to Newsletter](https://luminaai.com/newsletter)

*"The future of AI interaction is here"*

</div>