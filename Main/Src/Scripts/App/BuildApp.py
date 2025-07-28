#!/usr/bin/env python3
"""
LuminaAI Desktop Application Setup
Creates all necessary files for the desktop app
"""

import os
import json
from pathlib import Path

def create_directory_structure():
    """Create the directory structure for the desktop app."""
    directories = [
        'renderer',
        'assets',
        'models',
        'logs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")

def create_package_json():
    """Create package.json for Electron app."""
    package_json = {
        "name": "lumina-ai-desktop",
        "version": "1.0.0",
        "description": "LuminaAI Neural Desktop Interface - Advanced AI Chat Application",
        "main": "main.js",
        "scripts": {
            "start": "electron .",
            "dev": "electron . --dev",
            "build": "electron-builder",
            "dist": "electron-builder --publish=never",
            "pack": "electron-builder --dir",
            "postinstall": "electron-builder install-app-deps"
        },
        "keywords": ["ai", "neural", "desktop", "electron", "chat", "transformer"],
        "author": {
            "name": "Matias Nielsen",
            "email": "contact@example.com"
        },
        "license": "Custom",
        "devDependencies": {
            "electron": "^28.0.0",
            "electron-builder": "^24.6.4"
        },
        "build": {
            "appId": "com.lumina.ai.desktop",
            "productName": "LuminaAI Desktop",
            "directories": {
                "output": "dist"
            },
            "files": [
                "**/*",
                "!**/node_modules/*/{CHANGELOG.md,README.md,README,readme.md,readme}",
                "!**/node_modules/*/{test,__tests__,tests,powered-test,example,examples}",
                "!**/node_modules/*.d.ts",
                "!**/node_modules/.bin",
                "!**/*.{iml,o,hprof,orig,pyc,pyo,rbc,swp,csproj,sln,xproj}",
                "!.editorconfig",
                "!**/._*",
                "!**/{.DS_Store,.git,.hg,.svn,CVS,RCS,SCCS,.gitignore,.gitattributes}",
                "!**/{__pycache__,thumbs.db,.flowconfig,.idea,.vs,.nyc_output}",
                "!**/{appveyor.yml,.travis.yml,circle.yml}",
                "!**/{npm-debug.log,yarn.lock,.yarn-integrity,.yarn-metadata.json}"
            ],
            "mac": {
                "category": "public.app-category.productivity",
                "icon": "assets/icon.icns",
                "target": [
                    {
                        "target": "dmg",
                        "arch": ["x64", "arm64"]
                    }
                ]
            },
            "win": {
                "icon": "assets/icon.ico",
                "target": [
                    {
                        "target": "nsis",
                        "arch": ["x64", "ia32"]
                    }
                ]
            },
            "linux": {
                "icon": "assets/icon.png",
                "target": [
                    {
                        "target": "AppImage",
                        "arch": ["x64"]
                    }
                ]
            }
        }
    }
    
    with open('package.json', 'w') as f:
        json.dump(package_json, f, indent=2)
    print("✅ Created package.json")

def create_main_js():
    """Create main.js for Electron main process."""
    main_js = '''const { app, BrowserWindow, Menu, dialog, shell, ipcMain } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const fs = require('fs');

let mainWindow;
let pythonProcess;

function createWindow() {
    // Create the browser window
    mainWindow = new BrowserWindow({
        width: 1400,
        height: 900,
        minWidth: 1200,
        minHeight: 800,
        webPreferences: {
            nodeIntegration: true,
            contextIsolation: false,
            enableRemoteModule: true
        },
        titleBarStyle: process.platform === 'darwin' ? 'hiddenInset' : 'default',
        vibrancy: process.platform === 'darwin' ? 'ultra-dark' : undefined,
        backgroundColor: '#0a0a0b',
        show: false,
        icon: getIconPath(),
        frame: true,
        transparent: false
    });

    // Load the app
    mainWindow.loadFile('renderer/index.html');

    // Show window when ready
    mainWindow.once('ready-to-show', () => {
        mainWindow.show();
        
        // Focus window and bring to front
        if (process.platform === 'darwin') {
            app.focus();
        }
        mainWindow.focus();
    });

    // Handle window closed
    mainWindow.on('closed', () => {
        mainWindow = null;
        if (pythonProcess) {
            pythonProcess.kill();
        }
    });

    // Create menu
    createMenu();

    // Open DevTools in development
    if (process.env.NODE_ENV === 'development') {
        mainWindow.webContents.openDevTools();
    }
}

function getIconPath() {
    const iconPath = path.join(__dirname, 'assets');
    if (process.platform === 'win32') {
        return path.join(iconPath, 'icon.ico');
    } else if (process.platform === 'darwin') {
        return path.join(iconPath, 'icon.icns');
    } else {
        return path.join(iconPath, 'icon.png');
    }
}

function createMenu() {
    const isMac = process.platform === 'darwin';
    
    const template = [
        ...(isMac ? [{
            label: app.getName(),
            submenu: [
                { role: 'about' },
                { type: 'separator' },
                { role: 'services' },
                { type: 'separator' },
                { role: 'hide' },
                { role: 'hideothers' },
                { role: 'unhide' },
                { type: 'separator' },
                { role: 'quit' }
            ]
        }] : []),
        {
            label: 'File',
            submenu: [
                {
                    label: 'Load Neural Model...',
                    accelerator: 'CmdOrCtrl+O',
                    click: async () => {
                        const result = await dialog.showOpenDialog(mainWindow, {
                            title: 'Select Neural Model',
                            filters: [
                                { name: 'PyTorch Models', extensions: ['pth', 'pt'] },
                                { name: 'All Files', extensions: ['*'] }
                            ],
                            properties: ['openFile']
                        });
                        
                        if (!result.canceled && result.filePaths.length > 0) {
                            mainWindow.webContents.send('load-model', result.filePaths[0]);
                        }
                    }
                },
                { type: 'separator' },
                {
                    label: 'Open Models Folder',
                    click: () => {
                        const modelsPath = path.join(__dirname, 'models');
                        if (!fs.existsSync(modelsPath)) {
                            fs.mkdirSync(modelsPath, { recursive: true });
                        }
                        shell.openPath(modelsPath);
                    }
                },
                { type: 'separator' },
                ...(isMac ? [] : [{ role: 'quit' }])
            ]
        },
        {
            label: 'Edit',
            submenu: [
                { role: 'undo' },
                { role: 'redo' },
                { type: 'separator' },
                { role: 'cut' },
                { role: 'copy' },
                { role: 'paste' },
                { role: 'selectall' }
            ]
        },
        {
            label: 'View',
            submenu: [
                { role: 'reload' },
                { role: 'forceReload' },
                { role: 'toggleDevTools' },
                { type: 'separator' },
                { role: 'resetZoom' },
                { role: 'zoomIn' },
                { role: 'zoomOut' },
                { type: 'separator' },
                { role: 'togglefullscreen' }
            ]
        },
        {
            label: 'Neural',
            submenu: [
                {
                    label: 'Clear Memory',
                    accelerator: 'CmdOrCtrl+Shift+C',
                    click: () => {
                        mainWindow.webContents.send('clear-chat');
                    }
                },
                {
                    label: 'Model Information',
                    accelerator: 'CmdOrCtrl+I',
                    click: () => {
                        mainWindow.webContents.send('show-model-info');
                    }
                },
                { type: 'separator' },
                {
                    label: 'Temperature: Low (0.3)',
                    click: () => {
                        mainWindow.webContents.send('set-temperature', 0.3);
                    }
                },
                {
                    label: 'Temperature: Medium (0.8)',
                    click: () => {
                        mainWindow.webContents.send('set-temperature', 0.8);
                    }
                },
                {
                    label: 'Temperature: High (1.2)',
                    click: () => {
                        mainWindow.webContents.send('set-temperature', 1.2);
                    }
                }
            ]
        },
        {
            label: 'Window',
            submenu: [
                { role: 'minimize' },
                { role: 'zoom' },
                ...(isMac ? [
                    { type: 'separator' },
                    { role: 'front' }
                ] : [
                    { role: 'close' }
                ])
            ]
        },
        {
            label: 'Help',
            submenu: [
                {
                    label: 'About LuminaAI',
                    click: () => {
                        dialog.showMessageBox(mainWindow, {
                            type: 'info',
                            title: 'About LuminaAI',
                            message: 'LuminaAI Neural Desktop Interface',
                            detail: 'Advanced neural transformer interface for desktop\\nVersion 1.0.0\\n\\nBuilt with Electron, Python, and PyTorch\\nCreated by Matias Nielsen',
                            buttons: ['OK']
                        });
                    }
                },
                {
                    label: 'Keyboard Shortcuts',
                    click: () => {
                        dialog.showMessageBox(mainWindow, {
                            type: 'info',
                            title: 'Keyboard Shortcuts',
                            message: 'LuminaAI Shortcuts',
                            detail: 'Ctrl/Cmd + O: Load Model\\nCtrl/Cmd + K: Focus Input\\nCtrl/Cmd + Shift + C: Clear Memory\\nCtrl/Cmd + I: Model Info\\nCtrl/Cmd + Enter: Send Message\\nEscape: Close Modal',
                            buttons: ['OK']
                        });
                    }
                },
                { type: 'separator' },
                {
                    label: 'System Requirements',
                    click: () => {
                        dialog.showMessageBox(mainWindow, {
                            type: 'info',
                            title: 'System Requirements',
                            message: 'LuminaAI System Requirements',
                            detail: '• Python 3.8 or higher\\n• PyTorch (CPU or GPU)\\n• 4GB RAM minimum\\n• CUDA support (optional)\\n• Apple Silicon support (MPS)',
                            buttons: ['OK']
                        });
                    }
                },
                {
                    label: 'Learn More',
                    click: () => {
                        shell.openExternal('https://github.com');
                    }
                }
            ]
        }
    ];

    const menu = Menu.buildFromTemplate(template);
    Menu.setApplicationMenu(menu);
}

function startPythonBackend() {
    const pythonCmd = process.platform === 'win32' ? 'python' : 'python3';
    const scriptPath = path.join(__dirname, 'lumina_desktop.py');
    
    console.log(`Starting Python backend: ${pythonCmd} ${scriptPath}`);
    
    pythonProcess = spawn(pythonCmd, [scriptPath], {
        stdio: ['pipe', 'pipe', 'pipe'],
        cwd: __dirname
    });

    pythonProcess.stdout.on('data', (data) => {
        console.log(`Python stdout: ${data}`);
    });

    pythonProcess.stderr.on('data', (data) => {
        console.error(`Python stderr: ${data}`);
    });

    pythonProcess.on('close', (code) => {
        console.log(`Python process exited with code ${code}`);
        if (code !== 0) {
            dialog.showErrorBox('Backend Error', 
                'Python backend crashed. Please check if Python and required packages are installed.\\n\\nRequired: pip install torch numpy flask flask-socketio flask-cors');
        }
    });

    pythonProcess.on('error', (error) => {
        console.error(`Failed to start Python process: ${error}`);
        dialog.showErrorBox('Startup Error', 
            `Failed to start Python backend: ${error.message}\\n\\nPlease ensure Python is installed and available in PATH.`);
    });
}

// App event handlers
app.whenReady().then(() => {
    createWindow();
    startPythonBackend();

    app.on('activate', () => {
        if (BrowserWindow.getAllWindows().length === 0) {
            createWindow();
        }
    });
});

app.on('window-all-closed', () => {
    if (pythonProcess) {
        pythonProcess.kill('SIGTERM');
    }
    
    if (process.platform !== 'darwin') {
        app.quit();
    }
});

app.on('before-quit', () => {
    if (pythonProcess) {
        pythonProcess.kill('SIGTERM');
    }
});

app.on('will-quit', () => {
    if (pythonProcess) {
        pythonProcess.kill('SIGTERM');
    }
});

// Handle protocol for deep linking (optional)
app.setAsDefaultProtocolClient('lumina-ai');

// IPC handlers
ipcMain.handle('get-app-version', () => {
    return app.getVersion();
});

ipcMain.handle('get-app-path', () => {
    return app.getAppPath();
});

// Security: Prevent new window creation
app.on('web-contents-created', (event, contents) => {
    contents.on('new-window', (event, navigationUrl) => {
        event.preventDefault();
        shell.openExternal(navigationUrl);
    });
});
'''
    
    with open('main.js', 'w') as f:
        f.write(main_js)
    print("✅ Created main.js")

def create_renderer_files():
    """Move the HTML file to renderer directory."""
    renderer_dir = Path('renderer')
    
    # The HTML content is already created in the frontend artifact
    # We'll create a simple instruction file instead
    readme_content = '''# LuminaAI Desktop Renderer

Place the frontend HTML file (index.html) in this directory.

The HTML file should contain:
- Ultra-modern glassmorphism UI
- Real-time neural chat interface  
- Advanced settings panel
- Animated background effects
- Socket.IO integration
- GSAP animations
- Responsive design

File structure:
- renderer/
  - index.html (main UI file)
  - assets/ (optional additional assets)
'''
    
    with open(renderer_dir / 'README.md', 'w') as f:
        f.write(readme_content)
    print("✅ Created renderer/README.md")

def create_installation_script():
    """Create installation script for dependencies."""
    install_script = '''#!/bin/bash
# LuminaAI Desktop Installation Script

echo "🚀 Installing LuminaAI Desktop Dependencies..."
echo "================================================"

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js first:"
    echo "   https://nodejs.org/"
    exit 1
fi

# Check if Python is installed
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "❌ Python is not installed. Please install Python 3.8+ first:"
    echo "   https://python.org/"
    exit 1
fi

echo "✅ Node.js version: $(node --version)"
echo "✅ Python version: $(python3 --version 2>/dev/null || python --version)"

echo ""
echo "📦 Installing Node.js dependencies..."
npm install

echo ""
echo "🐍 Installing Python dependencies..."
if command -v python3 &> /dev/null; then
    python3 -m pip install torch numpy flask flask-socketio flask-cors
else
    python -m pip install torch numpy flask flask-socketio flask-cors
fi

echo ""
echo "🎉 Installation complete!"
echo ""
echo "To start LuminaAI Desktop:"
echo "  npm start"
echo ""
echo "For development mode:"
echo "  npm run dev"
echo ""
echo "To build distributables:"
echo "  npm run build"
'''
    
    with open('install.sh', 'w') as f:
        f.write(install_script)
    
    # Make executable on Unix systems
    import stat
    try:
        os.chmod('install.sh', stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH)
    except:
        pass
    
    print("✅ Created install.sh")
    
    # Create Windows batch file
    install_bat = '''@echo off
REM LuminaAI Desktop Installation Script for Windows

echo 🚀 Installing LuminaAI Desktop Dependencies...
echo ================================================

REM Check if Node.js is installed
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Node.js is not installed. Please install Node.js first:
    echo    https://nodejs.org/
    pause
    exit /b 1
)

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is not installed. Please install Python 3.8+ first:
    echo    https://python.org/
    pause
    exit /b 1
)

echo ✅ Node.js version:
node --version
echo ✅ Python version:
python --version

echo.
echo 📦 Installing Node.js dependencies...
npm install

echo.
echo 🐍 Installing Python dependencies...
python -m pip install torch numpy flask flask-socketio flask-cors

echo.
echo 🎉 Installation complete!
echo.
echo To start LuminaAI Desktop:
echo   npm start
echo.
echo For development mode:
echo   npm run dev
echo.
echo To build distributables:
echo   npm run build
echo.
pause
'''
    
    with open('install.bat', 'w') as f:
        f.write(install_bat)
    
    print("✅ Created install.bat")

def create_requirements_file():
    """Create Python requirements.txt."""
    requirements = '''# LuminaAI Desktop Python Dependencies
torch>=2.0.0
numpy>=1.21.0
flask>=2.3.0
flask-socketio>=5.3.0
flask-cors>=4.0.0
'''
    
    with open('requirements.txt', 'w') as f:
        f.write(requirements)
    print("✅ Created requirements.txt")

def create_gitignore():
    """Create .gitignore file."""
    gitignore = '''# Dependencies
node_modules/
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.venv/

# Build outputs
dist/
build/
*.egg-info/

# Logs
logs/
*.log
npm-debug.log*
yarn-debug.log*
yarn-error.log*

# Runtime data
pids/
*.pid
*.seed
*.pid.lock

# Coverage directory used by tools like istanbul
coverage/

# Dependency directories
jspm_packages/

# Optional npm cache directory
.npm

# Optional REPL history
.node_repl_history

# Output of 'npm pack'
*.tgz

# Yarn Integrity file
.yarn-integrity

# dotenv environment variables file
.env

# IDE files
.vscode/
.idea/
*.swp
*.swo
*~

# OS generated files
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Model files (too large for git)
*.pth
*.pt
models/
'''
    
    with open('.gitignore', 'w') as f:
        f.write(gitignore)
    print("✅ Created .gitignore")

def create_readme():
    """Create comprehensive README.md."""
    readme = '''# 🌟 LuminaAI Desktop - Neural Chat Interface

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
- **RAM**: 4GB minimum, 8GB recommended
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
# Clone or download the project
git clone <repository-url>
cd lumina-ai-desktop

# Run installation script
chmod +x install.sh
./install.sh

# Or on Windows
install.bat
```

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
'''
    
    with open('README.md', 'w') as f:
        f.write(readme)
    print("✅ Created README.md")

def create_startup_script():
    """Create a simple startup script."""
    startup_py = '''#!/usr/bin/env python3
"""
LuminaAI Desktop Startup Script
Quick launcher for the desktop application
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    print("🌟 LuminaAI Desktop Launcher")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not Path('main.js').exists():
        print("❌ Error: main.js not found. Please run from the LuminaAI directory.")
        return 1
    
    # Check if node_modules exists
    if not Path('node_modules').exists():
        print("📦 Installing dependencies...")
        try:
            subprocess.run(['npm', 'install'], check=True)
        except subprocess.CalledProcessError:
            print("❌ Failed to install Node.js dependencies")
            return 1
    
    # Start the application
    print("🚀 Launching LuminaAI Desktop...")
    try:
        subprocess.run(['npm', 'start'], check=True)
    except subprocess.CalledProcessError:
        print("❌ Failed to start application")
        return 1
    except KeyboardInterrupt:
        print("\\n👋 LuminaAI Desktop closed by user")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
'''
    
    with open('startup.py', 'w') as f:
        f.write(startup_py)
    print("✅ Created startup.py")

def create_all_files():
    """Create all necessary files for the desktop app."""
    print("🚀 Creating LuminaAI Desktop Application Files...")
    print("=" * 50)
    
    create_directory_structure()
    create_package_json()
    create_main_js()
    create_renderer_files()
    create_installation_script()
    create_requirements_file()
    create_gitignore()
    create_readme()
    create_startup_script()
    
    print("\n" + "=" * 50)
    print("✅ LuminaAI Desktop setup complete!")
    print("\nNext steps:")
    print("1. Save the frontend HTML to 'renderer/index.html'")
    print("2. Save the backend Python script as 'lumina_desktop.py'")
    print("3. Run './install.sh' (or 'install.bat' on Windows)")
    print("4. Start with 'npm start'")
    print("\n🌟 Enjoy your neural desktop interface!")

if __name__ == "__main__":
    create_all_files()