# Copyright (c) 2025 Matias Nielsen. All rights reserved.
# Licensed under the Custom License below.

from pathlib import Path
import os
import sys

def create_build_files():
    """Create all necessary build files for packaging the desktop app."""
    
    # 1. Create the main desktop app file (save as lumina_desktop.py)
    desktop_app_code = '''#!/usr/bin/env python3
"""
LuminaAI Desktop GUI Application
Cross-platform desktop interface for the character-level transformer chatbot.
App opens without requiring a model file - model loading is optional.

Save this file as: lumina_desktop.py
"""
# The main GUI code goes here - copy from the enhanced artifact above
# This version opens without requiring Model.pth and gracefully handles missing PyTorch
'''
    
    # 2. Create PyInstaller spec file for detailed configuration
    spec_file_content = '''# -*- mode: python ; coding: utf-8 -*-

import sys
from pathlib import Path
import glob

block_cipher = None

# Define the main script
main_script = 'lumina_desktop.py'

# Dynamically build data files list (only include files that exist)
data_files = []

# Add model files if they exist
pth_files = glob.glob('*.pth')
if pth_files:
    print(f"Including {len(pth_files)} .pth model files")
    for pth_file in pth_files:
        data_files.append((pth_file, '.'))
else:
    print("No .pth files found - building without models")

# Add optional files if they exist
optional_files = ['requirements.txt', 'desktop_requirements.txt', 'README.md', 'BUILD_INSTRUCTIONS.md']
for file_name in optional_files:
    if Path(file_name).exists():
        data_files.append((file_name, '.'))
        print(f"Including {file_name}")

# Analysis
a = Analysis(
    [main_script],
    pathex=['.'],
    binaries=[],
    datas=data_files,
    hiddenimports=[
        # Core GUI modules (always needed)
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
        'sys',
        # AI modules (optional - app handles gracefully if missing)
        'torch',
        'torch.nn',
        'torch.nn.functional',
        'numpy',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude heavy unused packages to reduce size
        'matplotlib',
        'scipy',
        'pandas',
        'PIL',
        'cv2',
        'jupyter',
        'IPython',
        'notebook',
        'sklearn',
        'seaborn',
        'plotly',
        'bokeh',
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
            'CFBundleDocumentTypes': [
                {
                    'CFBundleTypeExtensions': ['pth'],
                    'CFBundleTypeName': 'PyTorch Model',
                    'CFBundleTypeRole': 'Viewer',
                    'LSHandlerRank': 'Owner'
                }
            ],
        },
    )
'''
    
    # 3. Create build scripts
    build_windows_script = '''@echo off
echo 🚀 Building LuminaAI for Windows...
echo    App can run without model files!

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

REM Build add-data arguments based on what files exist
set ADD_DATA_ARGS=

REM Check for model files
dir "*.pth" >nul 2>&1
if %errorlevel% == 0 (
    echo 📁 Found .pth model files - including in build
    set ADD_DATA_ARGS=%ADD_DATA_ARGS% --add-data="*.pth;."
) else (
    echo 📁 No .pth files found - building without models ^(recommended^)
)

REM Check for requirements files
if exist "requirements.txt" (
    set ADD_DATA_ARGS=%ADD_DATA_ARGS% --add-data="requirements.txt;."
)

if exist "desktop_requirements.txt" (
    set ADD_DATA_ARGS=%ADD_DATA_ARGS% --add-data="desktop_requirements.txt;."
)

if exist "BUILD_INSTRUCTIONS.md" (
    set ADD_DATA_ARGS=%ADD_DATA_ARGS% --add-data="BUILD_INSTRUCTIONS.md;."
)

REM Build the executable
echo Building executable...
pyinstaller --name="LuminaAI" ^
    --windowed ^
    --onedir ^
    --clean ^
    --noconfirm ^
    %ADD_DATA_ARGS% ^
    --hidden-import=tkinter ^
    --hidden-import=tkinter.ttk ^
    --hidden-import=tkinter.scrolledtext ^
    --hidden-import=tkinter.messagebox ^
    --hidden-import=tkinter.filedialog ^
    --hidden-import=torch ^
    --hidden-import=torch.nn ^
    --hidden-import=torch.nn.functional ^
    --hidden-import=numpy ^
    --exclude-module=matplotlib ^
    --exclude-module=scipy ^
    --exclude-module=pandas ^
    --exclude-module=jupyter ^
    --exclude-module=IPython ^
    --exclude-module=sklearn ^
    lumina_desktop.py

if exist "dist\\LuminaAI" (
    echo.
    echo ✅ Build completed successfully!
    echo 📁 Your executable is in: dist\\LuminaAI\\
    echo 🚀 Run: dist\\LuminaAI\\LuminaAI.exe
    echo.
    echo 🎯 App Features:
    echo    • Opens without requiring model files
    echo    • Load models using the "Load Model" button
    echo    • Works with or without PyTorch installed
    echo    • Full GUI functionality always available
    echo.
    echo 💡 Optional: Copy Model.pth to dist\\LuminaAI\\ for auto-loading
    echo.
    pause
) else (
    echo ❌ Build failed!
    echo Check error messages above
    pause
)
'''
    
    build_macos_script = '''#!/bin/bash
echo "🚀 Building LuminaAI for macOS..."
echo "   App can run without model files!"

# Check if PyInstaller is installed
if ! python3 -c "import PyInstaller" 2>/dev/null; then
    echo "Installing PyInstaller..."
    pip3 install pyinstaller
fi

# Clean previous builds
rm -rf build dist *.spec

# Check for optional files and build add-data arguments
ADD_DATA_ARGS=""

# Add model files if they exist
if ls *.pth 1> /dev/null 2>&1; then
    echo "📁 Found .pth model files - including in build"
    ADD_DATA_ARGS="$ADD_DATA_ARGS --add-data=*.pth:."
else
    echo "📁 No .pth files found - building without models (recommended)"
fi

# Add requirements.txt if it exists
if [ -f "requirements.txt" ]; then
    ADD_DATA_ARGS="$ADD_DATA_ARGS --add-data=requirements.txt:."
fi

# Add desktop requirements if it exists
if [ -f "desktop_requirements.txt" ]; then
    ADD_DATA_ARGS="$ADD_DATA_ARGS --add-data=desktop_requirements.txt:."
fi

# Add build instructions if they exist
if [ -f "BUILD_INSTRUCTIONS.md" ]; then
    ADD_DATA_ARGS="$ADD_DATA_ARGS --add-data=BUILD_INSTRUCTIONS.md:."
fi

# Build the app
echo "Building macOS app..."
pyinstaller --name="LuminaAI" \\
    --windowed \\
    --onedir \\
    --clean \\
    --noconfirm \\
    $ADD_DATA_ARGS \\
    --hidden-import=tkinter \\
    --hidden-import=tkinter.ttk \\
    --hidden-import=tkinter.scrolledtext \\
    --hidden-import=tkinter.messagebox \\
    --hidden-import=tkinter.filedialog \\
    --hidden-import=torch \\
    --hidden-import=torch.nn \\
    --hidden-import=torch.nn.functional \\
    --hidden-import=numpy \\
    --exclude-module=matplotlib \\
    --exclude-module=scipy \\
    --exclude-module=pandas \\
    --exclude-module=jupyter \\
    --exclude-module=IPython \\
    --exclude-module=sklearn \\
    --osx-bundle-identifier=com.matias.luminaai \\
    lumina_desktop.py

if [ -d "dist/LuminaAI.app" ]; then
    echo
    echo "✅ Build completed successfully!"
    echo "📁 Your app is in: dist/"
    echo "🚀 Run: open dist/LuminaAI.app"
    echo
    echo "🎯 App Features:"
    echo "   • Opens without requiring model files"
    echo "   • Load models using the 'Load Model' button"
    echo "   • Works with or without PyTorch installed"
    echo "   • Full GUI functionality always available"
    echo
    echo "💡 Optional: Copy Model.pth to same folder for auto-loading"
    
    # Make the app executable
    chmod +x "dist/LuminaAI.app/Contents/MacOS/LuminaAI"
    
    # Optional: Create a DMG file
    echo
    read -p "Create DMG installer? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Creating DMG..."
        hdiutil create -volname "LuminaAI" -srcfolder "dist/LuminaAI.app" -ov -format UDZO "LuminaAI.dmg"
        echo "✅ DMG created: LuminaAI.dmg"
        echo "📦 Ready for distribution!"
    fi
else
    echo "❌ Build failed!"
    echo "Check error messages above"
fi
'''
    
    # 4. Create a requirements file specifically for the desktop app
    desktop_requirements = '''# LuminaAI Desktop App Requirements
# Core GUI dependencies (always required)
# tkinter is built into Python, no need to install

# AI dependencies (optional - app will run without these)
torch>=2.0.0
numpy>=1.21.0

# Build dependencies
pyinstaller>=5.0

# Optional: Dataset handling (if you want to use Train.py)
datasets>=2.12.0

# Optional: Better icons and UI
# pillow>=9.0.0

# Development tools (optional)
# auto-py-to-exe  # GUI for PyInstaller
'''
    
    # 5. Create installation instructions
    installation_instructions = '''# 🚀 LuminaAI Desktop App - Build Instructions
## Updated for App That Opens Without Model Files

## 📋 Prerequisites

1. **Python 3.8+** installed on your system
2. **Basic GUI libraries** (tkinter - usually built into Python)
3. **Optional**: PyTorch and model files for AI functionality

## 🆕 What's New

- ✅ **App opens without requiring model files**
- ✅ **Works without PyTorch installed** (shows helpful warnings)
- ✅ **GUI fully functional always**
- ✅ **Load models on-demand using "Load Model" button**
- ✅ **Better error handling and user guidance**

## 🛠️ Setup

### 1. Install Dependencies
```bash
# Minimum (GUI only)
pip install pyinstaller

# Full functionality (AI enabled)
pip install -r desktop_requirements.txt
```

### 2. Prepare Files
Make sure you have these files in your project folder:
- `lumina_desktop.py` (the enhanced GUI application)
- `desktop_requirements.txt` (dependencies)
- Build scripts (created by this script)
- **Optional**: `Model.pth` (your trained model - not required)

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
- **Optional**: Include Model.pth file for auto-loading
- Users run `LuminaAI.exe` (works immediately)
- Users can load models using the GUI

### macOS
- Copy `LuminaAI.app` or distribute the DMG
- **Optional**: Include Model.pth file for auto-loading
- Users double-click to run (works immediately)
- Users can load models using the GUI

## 🎯 App Behavior

### Without Model File:
- ✅ **App opens successfully**
- ✅ **GUI fully functional**
- ✅ **Settings can be adjusted**
- ⚠️ **Chat shows "Load model first" message**
- 🔘 **Send button disabled until model loaded**

### With Model File:
- ✅ **Auto-loads Model.pth if present**
- ✅ **Manual loading via "Load Model" button**
- ✅ **Full chat functionality**
- ✅ **All features enabled**

### Without PyTorch:
- ✅ **App still opens and works**
- ⚠️ **Shows PyTorch installation instructions**
- 📝 **Provides clear guidance to user**

## 🐛 Troubleshooting

### Common Issues:

1. **"tkinter not found"**:
   - Reinstall Python with tkinter support
   - On Linux: `sudo apt-get install python3-tk`

2. **App opens but no AI functionality**:
   - This is normal! Install PyTorch: `pip install torch numpy`
   - Or distribute PyTorch with your app

3. **Large file size**:
   - Use `--onefile` instead of `--onedir`
   - PyTorch is large - this is expected
   - Consider distributing without PyTorch

4. **Model not found**:
   - This is fine! Use "Load Model" button
   - Or place Model.pth in app folder

### Size Optimization:

**Smaller executable** (GUI only):
```bash
pyinstaller --onefile --windowed --exclude-module=torch --exclude-module=numpy lumina_desktop.py
```

**Full functionality** (larger):
```bash
pyinstaller --onefile --windowed lumina_desktop.py
```

## 🎯 Features of the Enhanced Desktop App

- ✅ **No Dependencies Required**: Opens without model files
- ✅ **Graceful Degradation**: Works with missing PyTorch
- ✅ **GUI Interface**: Always functional chat interface
- ✅ **Dynamic Model Loading**: Browse and load any .pth file
- ✅ **Smart Status Messages**: Clear user guidance
- ✅ **Settings Control**: Adjust temperature, sampling, etc.
- ✅ **Cross-Platform**: Windows, macOS, Linux
- ✅ **GPU Support**: Automatic CUDA/MPS/CPU detection
- ✅ **Thread-Safe**: Non-blocking UI during AI generation
- ✅ **Memory Management**: Automatic cleanup
- ✅ **Error Recovery**: Handles missing dependencies gracefully

## 📊 System Requirements

**Minimum (GUI only)**:
- Python 3.8+
- 512MB RAM
- 100MB free disk space
- No GPU required

**Recommended (Full AI)**:
- Python 3.9+
- 8GB+ RAM
- GPU with 4GB+ VRAM (CUDA or Apple Silicon)
- 2GB+ free disk space
- PyTorch installed

## 🚀 Distribution Strategies

### Strategy 1: Lightweight Distribution
- Build without PyTorch
- Users install PyTorch themselves
- Smaller download size
- Users get latest PyTorch version

### Strategy 2: Full Distribution
- Include PyTorch in build
- Larger download size
- Works immediately for users
- No additional installation needed

### Strategy 3: Hybrid Distribution
- Provide both versions
- Let users choose based on needs
- Include clear instructions

## 🤝 Support

If you encounter issues:
1. Check if app opens (it should always open)
2. Check status messages in the app
3. Install PyTorch if needed: `pip install torch numpy`
4. Use "Load Model" button to load .pth files
5. Check console output (remove `--windowed` for debugging)

## 🎉 Success!

Your LuminaAI desktop app now:
- ✅ Opens without requiring model files
- ✅ Guides users through setup process  
- ✅ Works with or without PyTorch
- ✅ Provides professional user experience
- ✅ Handles errors gracefully
- ✅ Is ready for distribution!

Happy chatting with your enhanced LuminaAI desktop app! 🤖✨
'''
    
    return {
        'desktop_app_code': desktop_app_code,
        'spec_file': spec_file_content,
        'build_windows': build_windows_script,
        'build_macos': build_macos_script,
        'desktop_requirements': desktop_requirements,
        'instructions': installation_instructions
    }

if __name__ == "__main__":
    print("🚀 Creating Enhanced LuminaAI Desktop App build files...")
    print("   Updated for app that opens without model files!")
    
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
    
    print("✅ Enhanced build files created successfully!")
    print("\n📁 Files created:")
    print("  - lumina_desktop.spec (Enhanced PyInstaller configuration)")
    print("  - build_windows.bat (Windows build script)")
    print("  - build_macos.sh (macOS build script)")
    print("  - desktop_requirements.txt (Updated dependencies)")
    print("  - BUILD_INSTRUCTIONS.md (Complete instructions)")
    
    print("\n🆕 Enhancements:")
    print("  ✅ App opens without requiring model files")
    print("  ✅ Works without PyTorch (shows helpful messages)")
    print("  ✅ Better error handling and user guidance")
    print("  ✅ Optimized build configuration")
    print("  ✅ Multiple distribution strategies")
    
    print("\n🏗️ Next steps:")
    print("1. Save the enhanced GUI code as 'lumina_desktop.py'")
    print("2. Install dependencies: pip install -r desktop_requirements.txt")
    print("3. Run the appropriate build script for your platform")
    print("4. Your desktop app will be in the 'dist/' folder")
    print("5. App will open immediately - no model required!")
    
    print("\n🎯 Quick build commands:")
    print("Windows: build_windows.bat")
    print("macOS: ./build_macos.sh")
    
    print("\n💡 Distribution options:")
    print("• Include PyTorch: Full functionality, larger size")
    print("• Exclude PyTorch: Smaller size, users install separately")
    print("• Hybrid: Provide both versions")