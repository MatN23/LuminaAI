# ==================================================
# setup.py - PyInstaller build script
# ==================================================

from pathlib import Path
import os
import sys

def create_build_files():
    """Create all necessary build files for packaging the desktop app."""
    
    # 1. Create the main desktop app file (save as lumina_desktop.py)
    desktop_app_code = '''#!/usr/bin/env python3
"""
LuminaAI Desktop GUI Application
Save this file as: lumina_desktop.py
"""
# The main GUI code goes here - copy from the artifact above
'''
    
    # 2. Create PyInstaller spec file for detailed configuration
    spec_file_content = '''# -*- mode: python ; coding: utf-8 -*-

import sys
from pathlib import Path

block_cipher = None

# Define the main script
main_script = 'lumina_desktop.py'

# Analysis
a = Analysis(
    [main_script],
    pathex=['.'],
    binaries=[],
    datas=[
        # Include any data files your app needs
        ('*.pth', '.'),  # Include model files if present
        ('requirements.txt', '.'),
        ('README.md', '.'),
    ],
    hiddenimports=[
        'torch',
        'torch.nn',
        'torch.nn.functional',
        'numpy',
        'tkinter',
        'tkinter.ttk',
        'tkinter.scrolledtext',
        'tkinter.messagebox',
        'tkinter.filedialog',
        'queue',
        'threading',
        'logging',
        'pathlib',
        'json',
        'math',
        're',
        'gc',
        'time',
        'os',
        'sys'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib',
        'scipy',
        'pandas',
        'PIL',
        'cv2'
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# Configure executable
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='LuminaAI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # Set to True for debugging
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='icon.ico' if Path('icon.ico').exists() else None,
)

# Create distribution
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='LuminaAI'
)

# macOS App Bundle
if sys.platform == 'darwin':
    app = BUNDLE(
        coll,
        name='LuminaAI.app',
        icon='icon.icns' if Path('icon.icns').exists() else None,
        bundle_identifier='com.matias.luminaai',
        info_plist={
            'CFBundleDisplayName': 'LuminaAI',
            'CFBundleGetInfoString': 'LuminaAI - Character-Level Transformer Chatbot',
            'CFBundleVersion': '1.0.0',
            'CFBundleShortVersionString': '1.0.0',
            'NSHighResolutionCapable': True,
            'NSRequiresAquaSystemAppearance': False,
        },
    )
'''
    
    # 3. Create build scripts
    build_windows_script = '''@echo off
echo Building LuminaAI for Windows...

REM Check if PyInstaller is installed
python -c "import PyInstaller" 2>nul
if errorlevel 1 (
    echo Installing PyInstaller...
    pip install pyinstaller
)

REM Clean previous builds
if exist "build" rmdir /s /q "build"
if exist "dist" rmdir /s /q "dist"
if exist "*.spec" del "*.spec"

REM Build the executable
echo Building executable...
pyinstaller --name="LuminaAI" ^
    --windowed ^
    --onedir ^
    --clean ^
    --noconfirm ^
    --add-data="*.pth;." ^
    --add-data="requirements.txt;." ^
    --hidden-import=torch ^
    --hidden-import=torch.nn ^
    --hidden-import=torch.nn.functional ^
    --hidden-import=numpy ^
    --exclude-module=matplotlib ^
    --exclude-module=scipy ^
    --exclude-module=pandas ^
    lumina_desktop.py

if exist "dist\\LuminaAI" (
    echo.
    echo ✅ Build completed successfully!
    echo 📁 Your executable is in: dist\\LuminaAI\\
    echo 🚀 Run: dist\\LuminaAI\\LuminaAI.exe
    echo.
    echo 💡 Copy your Model.pth file to the dist\\LuminaAI\\ folder
    pause
) else (
    echo ❌ Build failed!
    pause
)
'''
    
    build_macos_script = '''#!/bin/bash
echo "Building LuminaAI for macOS..."

# Check if PyInstaller is installed
if ! python3 -c "import PyInstaller" 2>/dev/null; then
    echo "Installing PyInstaller..."
    pip3 install pyinstaller
fi

# Clean previous builds
rm -rf build dist *.spec

# Build the app
echo "Building macOS app..."
pyinstaller --name="LuminaAI" \\
    --windowed \\
    --onedir \\
    --clean \\
    --noconfirm \\
    --add-data="*.pth:." \\
    --add-data="requirements.txt:." \\
    --hidden-import=torch \\
    --hidden-import=torch.nn \\
    --hidden-import=torch.nn.functional \\
    --hidden-import=numpy \\
    --exclude-module=matplotlib \\
    --exclude-module=scipy \\
    --exclude-module=pandas \\
    --osx-bundle-identifier=com.matias.luminaai \\
    lumina_desktop.py

if [ -d "dist/LuminaAI.app" ]; then
    echo
    echo "✅ Build completed successfully!"
    echo "📁 Your app is in: dist/"
    echo "🚀 Run: open dist/LuminaAI.app"
    echo
    echo "💡 Copy your Model.pth file to the same folder as the app"
    
    # Make the app executable
    chmod +x "dist/LuminaAI.app/Contents/MacOS/LuminaAI"
    
    # Optional: Create a DMG file
    read -p "Create DMG installer? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Creating DMG..."
        hdiutil create -volname "LuminaAI" -srcfolder "dist/LuminaAI.app" -ov -format UDZO "LuminaAI.dmg"
        echo "✅ DMG created: LuminaAI.dmg"
    fi
else
    echo "❌ Build failed!"
fi
'''
    
    # 4. Create a requirements file specifically for the desktop app
    desktop_requirements = '''# LuminaAI Desktop App Requirements
torch>=2.0.0
numpy>=1.21.0
pyinstaller>=5.0
datasets>=2.12.0

# Optional: For better icons and UI
# pillow>=9.0.0

# Development
# auto-py-to-exe  # GUI for PyInstaller
'''
    
    # 5. Create installation instructions
    installation_instructions = '''# 🚀 LuminaAI Desktop App - Build Instructions

## 📋 Prerequisites

1. **Python 3.8+** installed on your system
2. **Your trained model file** (Model.pth)
3. **All dependencies** from requirements.txt

## 🛠️ Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
pip install pyinstaller
```

### 2. Prepare Files
Make sure you have these files in your project folder:
- `lumina_desktop.py` (the GUI application)
- `Model.pth` (your trained model)
- `requirements.txt`
- Build scripts (see below)

## 🏗️ Building the Desktop App

### For Windows (.exe)

1. **Save the build script** as `build_windows.bat`
2. **Run the build script**:
   ```cmd
   build_windows.bat
   ```
3. **Your executable** will be in `dist/LuminaAI/LuminaAI.exe`

### For macOS (.app)

1. **Save the build script** as `build_macos.sh`
2. **Make it executable**:
   ```bash
   chmod +x build_macos.sh
   ```
3. **Run the build script**:
   ```bash
   ./build_macos.sh
   ```
4. **Your app** will be in `dist/LuminaAI.app`

## 📦 Distribution

### Windows
- Copy the entire `dist/LuminaAI/` folder
- Include your `Model.pth` file in the same folder
- Users run `LuminaAI.exe`

### macOS
- Copy `LuminaAI.app` 
- Include your `Model.pth` file in the same folder
- Users double-click the app to run

## 🐛 Troubleshooting

### Common Issues:

1. **"Module not found" errors**:
   ```bash
   pip install --upgrade torch numpy
   ```

2. **Large file size**:
   - Use `--onefile` instead of `--onedir` for smaller distribution
   - Exclude unnecessary modules with `--exclude-module`

3. **Model not found**:
   - Ensure `Model.pth` is in the same folder as the executable
   - Or use the "Load Model" button in the app

4. **Performance issues**:
   - The first run might be slower due to PyTorch initialization
   - Subsequent runs will be faster

### Advanced Options:

**Custom Icon**: Add `icon.ico` (Windows) or `icon.icns` (macOS) to your project folder

**Smaller executable**: Use the `--onefile` option:
```bash
pyinstaller --onefile --windowed lumina_desktop.py
```

**Debug mode**: Remove `--windowed` to see console output for debugging

## 🎯 Features of the Desktop App

- ✅ **GUI Interface**: Easy-to-use chat interface
- ✅ **Model Loading**: Browse and load any .pth model file
- ✅ **Settings Control**: Adjust temperature, sampling methods, etc.
- ✅ **Cross-Platform**: Works on Windows, macOS, and Linux
- ✅ **GPU Support**: Automatic CUDA/MPS/CPU detection
- ✅ **Thread-Safe**: Non-blocking UI during AI generation
- ✅ **Memory Management**: Automatic cleanup and optimization

## 📊 System Requirements

**Minimum**:
- Python 3.8+
- 4GB RAM
- 1GB free disk space

**Recommended**:
- Python 3.9+
- 8GB+ RAM
- GPU with 4GB+ VRAM (CUDA or Apple Silicon)
- 2GB+ free disk space

## 🤝 Support

If you encounter issues:
1. Check the console output (run without `--windowed`)
2. Verify all dependencies are installed
3. Ensure your model file is compatible
4. Check file permissions

Happy chatting with your LuminaAI desktop app! 🤖✨
'''
    
    return {
        'spec_file': spec_file_content,
        'build_windows': build_windows_script,
        'build_macos': build_macos_script,
        'desktop_requirements': desktop_requirements,
        'instructions': installation_instructions
    }

if __name__ == "__main__":
    print("🚀 Creating LuminaAI Desktop App build files...")
    
    files = create_build_files()
    
    # Create the files
    Path("lumina_desktop.spec").write_text(files['spec_file'])
    Path("build_windows.bat").write_text(files['build_windows'])
    Path("build_macos.sh").write_text(files['build_macos'])
    Path("desktop_requirements.txt").write_text(files['desktop_requirements'])
    Path("BUILD_INSTRUCTIONS.md").write_text(files['instructions'])
    
    # Make macOS script executable
    import stat
    if Path("build_macos.sh").exists():
        Path("build_macos.sh").chmod(stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH)
    
    print("✅ Build files created successfully!")
    print("\n📁 Files created:")
    print("  - lumina_desktop.spec (PyInstaller configuration)")
    print("  - build_windows.bat (Windows build script)")
    print("  - build_macos.sh (macOS build script)")
    print("  - desktop_requirements.txt (Dependencies)")
    print("  - BUILD_INSTRUCTIONS.md (Detailed instructions)")
    
    print("\n🏗️ Next steps:")
    print("1. Save the GUI code as 'lumina_desktop.py'")
    print("2. Install dependencies: pip install -r desktop_requirements.txt")
    print("3. Run the appropriate build script for your platform")
    print("4. Your desktop app will be in the 'dist/' folder")
    
    print("\n🎯 Quick build commands:")
    print("Windows: build_windows.bat")
    print("macOS: ./build_macos.sh")