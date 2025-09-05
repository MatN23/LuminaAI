# LuminaAI Desktop - Neural Chat Interface

A modern Electron desktop application for conversational AI using PyTorch transformer models. Features a sleek neural-themed interface with real-time model inference capabilities.

## Features

- **Modern UI**: Glass-morphism design with neural network animations
- **PyTorch Integration**: Load and run your own transformer models (.pt, .pth files)
- **Real-time Chat**: WebSocket-based communication for responsive conversations
- **Advanced Sampling**: Multiple text generation methods (Top-K, Nucleus, Greedy)
- **Model Analysis**: Detailed model architecture inspection
- **Cross-platform**: Works on Windows, macOS, and Linux

## Prerequisites

### Required Software

1. **Node.js** (v16.0.0 or higher)
   - Download from [nodejs.org](https://nodejs.org/)

2. **Python** (3.7 or higher)
   - Download from [python.org](https://python.org/)
   - Make sure Python is added to your PATH

3. **Git** (optional, for cloning)
   - Download from [git-scm.com](https://git-scm.com/)

## Installation

### 1. Clone or Download

```bash
git clone https://github.com/your-username/luminaai-desktop.git
cd luminaai-desktop
```

### 2. Install Node.js Dependencies

```bash
npm install
```

### 3. Install Python Dependencies

The app will automatically prompt to install Python packages when you first run it, or install manually:

```bash
pip install -r python_requirements.txt
```

Required Python packages:
- `torch` - PyTorch neural network framework
- `numpy` - Numerical computing
- `flask` - Web framework for backend API
- `flask-socketio` - WebSocket support
- `flask-cors` - Cross-origin resource sharing

### 4. Prepare Your Model Files

Place your PyTorch model files (.pt, .pth) in an accessible directory. The app supports:
- Standard PyTorch state dictionaries
- Checkpoint files with metadata
- Models with embedded configuration

## Running the Application

### Development Mode

```bash
npm run dev
```

### Production Mode

```bash
npm start
```

### Using the Startup Script

For automatic dependency management:

```bash
node start.js
```

## Project Structure

```
luminaai-desktop/
├── main.js              # Electron main process
├── index.html           # UI interface (your existing file)
├── chat_server.py       # Python Flask backend
├── Chat.py              # Original chat script (your existing file)
├── start.js             # Startup script with dependency management
├── package.json         # Node.js dependencies and build config
├── python_requirements.txt # Python dependencies
├── config/              # Configuration modules (from your existing code)
├── core/                # Core model and tokenizer modules (from your existing code)
├── checkpoint.py        # Checkpoint management (from your existing code)
└── assets/              # App icons and resources
```

## Usage

### Loading a Model

1. **Via Menu**: File → Load Model... or File → Load Checkpoint...
2. **Via UI**: Click the "Load Model" button in the sidebar
3. **Auto-detection**: The app will search for available checkpoints automatically

### Supported Model Formats

- `.pt` / `.pth` - Standard PyTorch model files
- `.bin` - Binary model files
- `.safetensors` - Safe tensor format
- Checkpoint directories with `best_checkpoint.pt` or similar

### Generation Settings

Configure text generation in the sidebar:

- **Temperature**: Controls randomness (0.1 = focused, 2.0 = creative)
- **Sampling Method**: Top-K, Nucleus (Top-P), or Greedy
- **Top-K**: Limits vocabulary to top K tokens
- **Top-P**: Nucleus sampling threshold
- **Max Length**: Maximum response length in tokens

### Keyboard Shortcuts

- `Ctrl/Cmd + O`: Load model
- `Ctrl/Cmd + K`: Clear chat history
- `Ctrl/Cmd + N`: New chat
- `Ctrl/Cmd + I`: Show model info
- `Ctrl/Cmd + Enter`: Send message (in input field)

## Building for Distribution

### All Platforms

```bash
npm run build
```

### Platform-Specific

```bash
npm run build-win    # Windows
npm run build-mac    # macOS  
npm run build-linux  # Linux
```

Built applications will be in the `dist/` directory.

## Configuration

### Backend Server

The Python backend runs on `localhost:5001` by default. You can modify this in:

- `main.js` (Electron startup)
- `chat_server.py` (Flask server)
- `index.html` (WebSocket connection)

### Model Configuration

Your existing `config/config_manager.py` provides model configuration presets:

- `debug` - Small model for testing
- `small` - Lightweight model
- `medium` - Balanced performance (default)
- `large` - High-capacity model

## Troubleshooting

### Python Backend Issues

1. **"Python not found"**
   - Ensure Python is installed and in PATH
   - Try `python3` if `python` doesn't work

2. **"PyTorch not available"**
   - Install PyTorch: `pip install torch`
   - For GPU support: Follow [PyTorch installation guide](https://pytorch.org/)

3. **"Failed to start backend"**
   - Check if port 5001 is already in use
   - Ensure all Python dependencies are installed

### Model Loading Issues

1. **"Model file not found"**
   - Verify the file path is correct
   - Ensure you have read permissions

2. **"Config mismatches detected"**
   - The model was trained with different settings
   - The app will auto-adjust when possible

3. **"Out of memory"**
   - Use a smaller model
   - Reduce sequence length in settings
   - Close other GPU-intensive applications

### UI Issues

1. **"Connection failed"**
   - Wait for backend to fully start
   - Check if firewall is blocking localhost connections

2. **Blank screen**
   - Try refreshing with `Ctrl/Cmd + R`
   - Check developer console for errors (`F12`)

## Development

### Adding New Features

1. **Frontend**: Modify `index.html` and the embedded JavaScript
2. **Backend**: Extend `chat_server.py` with new API endpoints
3. **Electron**: Update `main.js` for new desktop features

### API Endpoints

The backend provides these REST endpoints:

- `GET /api/system/status` - System status and capabilities
- `POST /api/model/load` - Load a model from file path
- `GET /api/model/info` - Get loaded model information
- `POST /api/chat` - Send chat message and get response
- `POST /api/chat/clear` - Clear conversation history
- `GET /api/health` - Health check

### WebSocket Events

Real-time communication via Socket.IO:

- `connect` / `disconnect` - Connection status
- `chat_message` - Send message for generation
- `typing_start` / `typing_stop` - Typing indicators
- `message_generated` - Receive generated response
- `generation_error` - Generation error occurred

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Credits

- Built with [Electron](https://www.electronjs.org/)
- UI powered by [PyTorch](https://pytorch.org/)
- Backend using [Flask](https://flask.palletsprojects.com/)
- WebSocket communication via [Socket.IO](https://socket.io/)

## Support

If you encounter issues:

1. Check the troubleshooting section above
2. Look at existing issues on GitHub
3. Create a new issue with:
   - Your operating system
   - Node.js and Python versions
   - Error messages
   - Steps to reproduce

---

**Note**: This application requires your own trained PyTorch transformer models. It does not include pre-trained models and is designed to work with models you have trained using the provided training infrastructure.

## Advanced Configuration

### Custom Model Configurations

You can create custom model configurations by modifying `config/config_manager.py`:

```python
class CustomConfig(Config):
    def __init__(self):
        super().__init__()
        self.hidden_size = 1024
        self.num_layers = 24
        self.num_heads = 16
        # ... other parameters
```

### Backend Customization

Modify `chat_server.py` to add custom endpoints or change model loading behavior:

```python
@app.route('/api/custom/endpoint', methods=['POST'])
def custom_endpoint():
    # Your custom logic here
    return jsonify({'success': True})
```

### UI Theming

The interface uses CSS custom properties for easy theming. Modify the `:root` section in `index.html`:

```css
:root {
    --accent-primary: #your-color;
    --bg-primary: your-gradient;
    /* ... other theme variables */
}
```

## Performance Optimization

### For Large Models

1. **GPU Memory**: Monitor GPU memory usage
2. **Sequence Length**: Reduce `seq_length` for memory-constrained systems
3. **Batch Size**: Keep batch size at 1 for inference
4. **Precision**: Use FP16 if your GPU supports it

### For Better Responsiveness

1. **Async Generation**: The app uses background threads for model inference
2. **WebSocket**: Real-time communication reduces latency
3. **Caching**: Tokenizer caching improves performance

## Security Considerations

### Network Security

- The backend only binds to localhost by default
- CORS is configured for Electron app origins only
- No external network access required for core functionality

### File System Access

- The app only accesses files you explicitly select
- Model files are loaded read-only
- Conversation exports are saved to user-selected locations

## Deployment

### Standalone Executable

The built application includes:
- Electron runtime
- Node.js dependencies
- Python backend script
- Your model files (if included in build)

### System Requirements

**Minimum:**
- 4GB RAM
- 2GB free disk space
- OpenGL 2.0 compatible graphics

**Recommended:**
- 8GB+ RAM
- NVIDIA GPU with 4GB+ VRAM (for large models)
- SSD storage for better model loading times

## Extending the Application

### Adding Model Types

To support additional model architectures:

1. Extend the `TransformerModel` class in `core/model.py`
2. Update the checkpoint loading logic in `chat_server.py`
3. Add configuration presets in `config/config_manager.py`

### Custom Tokenizers

To use different tokenization methods:

1. Extend `ConversationTokenizer` in `core/tokenizer.py`
2. Update the tokenizer initialization in `chat_server.py`
3. Ensure compatibility with your model's vocabulary

### Additional UI Features

The modular JavaScript architecture allows easy extension:

```javascript
// Add new sidebar sections
const newSection = document.createElement('div');
newSection.className = 'section';
// ... configure section

// Add new API endpoints
fetch('/api/your/endpoint', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(data)
});
```

## FAQ

**Q: Can I use models from Hugging Face?**
A: The app is designed for models trained with the included infrastructure. Hugging Face models would require adapter code.

**Q: What's the maximum model size supported?**
A: Limited by your system's RAM/VRAM. Successfully tested with models up to 7B parameters on 24GB GPU.

**Q: Can I run this on a CPU-only system?**
A: Yes, but generation will be significantly slower. Recommended for smaller models only.

**Q: How do I backup my conversations?**
A: Use File → Export Chat or the conversation save/load features in the chat interface.

**Q: Can I customize the neural network visualization?**
A: Yes, modify the particle and neural node generation functions in the JavaScript section of `index.html`.

## Changelog

### Version 1.0.0
- Initial release
- PyTorch model support
- Modern Electron interface
- WebSocket real-time communication
- Multiple sampling methods
- Cross-platform compatibility

---

**Happy chatting with your neural models!** 🤖