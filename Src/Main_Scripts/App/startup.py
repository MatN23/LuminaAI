#!/usr/bin/env python3
"""
Enhanced LuminaAI Desktop Startup Script
Automatically handles both Python backend and Electron frontend
"""

import subprocess
import sys
import os
import time
import threading
import signal
import json
from pathlib import Path
import requests
import webbrowser

def check_python_dependencies():
    """Check if required Python packages are installed."""
    required_packages = [
        'flask', 'flask_cors', 'flask_socketio', 
        'torch', 'numpy', 'pickle'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing Python packages: {', '.join(missing_packages)}")
        print("📦 Install with: pip install " + " ".join(missing_packages))
        return False
    return True

def check_node_dependencies():
    """Check if Node.js and Electron are available."""
    try:
        # Check if npm is available
        subprocess.run(['npm', '--version'], check=True, capture_output=True)
        
        # Check if package.json exists
        if not Path('package.json').exists():
            create_package_json()
        
        # Install dependencies if node_modules doesn't exist
        if not Path('node_modules').exists():
            print("📦 Installing Node.js dependencies...")
            subprocess.run(['npm', 'install'], check=True)
        
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Node.js/npm not found. Please install Node.js")
        return False

def create_package_json():
    """Create package.json if it doesn't exist."""
    package_json = {
        "name": "lumina-ai-desktop",
        "version": "1.0.0",
        "description": "LuminaAI Neural Desktop Interface",
        "main": "main.js",
        "scripts": {
            "start": "electron .",
            "dev": "electron . --dev"
        },
        "devDependencies": {
            "electron": "^28.0.0"
        }
    }
    
    with open('package.json', 'w') as f:
        json.dump(package_json, f, indent=2)
    print("✅ Created package.json")

def wait_for_backend(port=5001, timeout=30):
    """Wait for the backend server to be ready."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f'http://localhost:{port}/api/health', timeout=2)
            if response.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)
    return False

def start_backend():
    """Start the Python backend server."""
    print("🧠 Starting LuminaAI Backend...")
    try:
        # Import and run the backend
        backend_process = subprocess.Popen([
            sys.executable, 'lumina_desktop.py'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        return backend_process
    except Exception as e:
        print(f"❌ Failed to start backend: {e}")
        return None

def start_electron():
    """Start the Electron frontend."""
    print("🖥️  Starting Electron Frontend...")
    try:
        # First try electron directly
        try:
            electron_process = subprocess.Popen(['electron', '.'])
            return electron_process
        except FileNotFoundError:
            # Try npx electron
            try:
                electron_process = subprocess.Popen(['npx', 'electron', '.'])
                return electron_process
            except FileNotFoundError:
                # Try npm start
                electron_process = subprocess.Popen(['npm', 'start'])
                return electron_process
    except Exception as e:
        print(f"❌ Failed to start Electron: {e}")
        return None

def open_web_fallback():
    """Open web interface as fallback."""
    print("🌐 Opening web interface as fallback...")
    time.sleep(2)  # Wait a bit for backend to start
    webbrowser.open('http://localhost:5001')

def main():
    print("🌟 LuminaAI Desktop Launcher")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path('lumina_desktop.py').exists():
        print("❌ Error: lumina_desktop.py not found. Please run from the LuminaAI directory.")
        return 1
    
    # Check Python dependencies
    if not check_python_dependencies():
        return 1
    
    # Check Node.js dependencies (optional for Electron)
    electron_available = check_node_dependencies()
    
    # Start backend server
    backend_process = start_backend()
    if not backend_process:
        print("❌ Failed to start backend server")
        return 1
    
    print("⏳ Waiting for backend to initialize...")
    if not wait_for_backend():
        print("❌ Backend failed to start properly")
        backend_process.terminate()
        return 1
    
    print("✅ Backend server is ready!")
    
    # Start Electron frontend or fallback to web
    electron_process = None
    if electron_available:
        electron_process = start_electron()
        if electron_process:
            print("✅ Electron frontend started!")
        else:
            print("⚠️  Electron failed, opening web interface...")
            open_web_fallback()
    else:
        print("⚠️  Electron not available, opening web interface...")
        open_web_fallback()
    
    print("\n🚀 LuminaAI Desktop is running!")
    print("🔗 Backend: http://localhost:5001")
    if electron_process:
        print("🖥️  Frontend: Electron App")
    else:
        print("🌐 Frontend: Web Browser")
    print("\nPress Ctrl+C to stop...")
    
    def signal_handler(sig, frame):
        print("\n🔌 Shutting down LuminaAI Desktop...")
        if backend_process:
            backend_process.terminate()
        if electron_process:
            electron_process.terminate()
        print("👋 Goodbye!")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Wait for processes
        if electron_process:
            electron_process.wait()
        else:
            # Keep the script running if only web interface
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        signal_handler(None, None)
    finally:
        if backend_process:
            backend_process.terminate()
        if electron_process:
            electron_process.terminate()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())