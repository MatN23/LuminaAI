@echo off
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
