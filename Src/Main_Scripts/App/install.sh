#!/bin/bash
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
