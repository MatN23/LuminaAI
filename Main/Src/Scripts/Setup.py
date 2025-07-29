# Copyright (c) 2025 Matias Nielsen. All rights reserved.
# Licensed under the Custom License below.
"""
Setup and Run Script for OASST1 Word Transformer Training
Copyright (c) 2025 Matias Nielsen. All rights reserved.
"""

import os
import sys
import subprocess
import importlib.util
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required!")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def check_and_install_packages():
    """Check and install required packages."""
    required_packages = [
        ("torch", "torch>=2.0.0"),
        ("datasets", "datasets>=2.14.0"),
        ("huggingface_hub", "huggingface_hub"),
        ("numpy", "numpy"),
        ("tqdm", "tqdm"),
    ]
    
    missing_packages = []
    
    print("🔍 Checking required packages...")
    
    for package_name, pip_name in required_packages:
        try:
            importlib.import_module(package_name)
            print(f"✅ {package_name} is installed")
        except ImportError:
            print(f"❌ {package_name} is missing")
            missing_packages.append(pip_name)
    
    if missing_packages:
        print(f"\n📦 Installing missing packages: {', '.join(missing_packages)}")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "--upgrade"
            ] + missing_packages)
            print("✅ All packages installed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install packages: {e}")
            return False
    
    return True

def check_required_files():
    """Check if all required files are present."""
    required_files = [
        "model_manager.py",
        "word_transformer.py",
        "Train.py",
        "Dataset_download.py"
    ]
    
    missing_files = []
    
    print("📁 Checking required files...")
    for file_name in required_files:
        if Path(file_name).exists():
            print(f"✅ {file_name}")
        else:
            print(f"❌ {file_name} is missing")
            missing_files.append(file_name)
    
    if missing_files:
        print(f"\n❌ Missing files: {', '.join(missing_files)}")
        print("Please ensure all required files are in the current directory.")
        return False
    
    return True

def check_torch_device():
    """Check available PyTorch devices."""
    try:
        import torch
        
        print("🖥️  Checking available devices...")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("✅ Apple Silicon (MPS) available")
            return "mps"
        else:
            print("⚠️  Using CPU (training will be slower)")
            return "cpu"
    
    except ImportError:
        print("❌ PyTorch not available")
        return None

def download_dataset():
    """Download the OASST1 dataset."""
    dataset_dir = Path("oasst1_data")
    train_file = dataset_dir / "oasst1_train.jsonl"
    val_file = dataset_dir / "oasst1_validation.jsonl"
    
    if train_file.exists() and val_file.exists():
        print("✅ Dataset files already exist")
        return True
    
    print("📥 Downloading OASST1 dataset...")
    try:
        result = subprocess.run([sys.executable, "Dataset_download.py"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Dataset downloaded successfully!")
            return True
        else:
            print(f"❌ Dataset download failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        return False

def estimate_training_time(device_type):
    """Estimate training time based on device."""
    estimates = {
        "cuda": "2-6 hours (depending on GPU)",
        "mps": "4-8 hours (Apple Silicon)",
        "cpu": "12-24 hours (very slow, not recommended)"
    }
    
    return estimates.get(device_type, "Unknown")

def run_training():
    """Run the training script."""
    print("🚀 Starting training...")
    print("Note: You can interrupt training with Ctrl+C")
    print("The best model will be saved automatically.")
    print("-" * 50)
    
    try:
        subprocess.run([sys.executable, "Train.py"])
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"❌ Training failed: {e}")

def main():
    """Main setup and run function."""
    print("=" * 60)
    print("🤖 OASST1 Word Transformer Training Setup")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Check and install packages
    if not check_and_install_packages():
        return 1
    
    # Check required files
    if not check_required_files():
        return 1
    
    # Check PyTorch device
    device_type = check_torch_device()
    if device_type is None:
        return 1
    
    # Show training estimate
    time_estimate = estimate_training_time(device_type)
    print(f"⏱️  Estimated training time: {time_estimate}")
    
    # Download dataset
    if not download_dataset():
        return 1
    
    print("\n" + "=" * 60)
    print("✅ Setup completed successfully!")
    print("=" * 60)
    
    # Ask user if they want to start training
    try:
        response = input("\n🚀 Start training now? (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            run_training()
        else:
            print("👍 Setup complete. Run 'python Train.py' when ready to train.")
    except KeyboardInterrupt:
        print("\n👍 Setup complete. Run 'python Train.py' when ready to train.")
    
    return 0

if __name__ == "__main__":
    exit(main())