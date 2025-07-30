#!/usr/bin/env python3
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
        print("\n👋 LuminaAI Desktop closed by user")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
