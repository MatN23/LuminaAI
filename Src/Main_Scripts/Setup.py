# Copyright (c) 2025 Matias Nielsen. All rights reserved.
# Licensed under the Custom License below.

import os
import sys
import logging
from pathlib import Path


def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        'torch',
        'numpy', 
        'tiktoken',
        'yaml',
        'psutil'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} is missing")
    
    if missing_packages:
        print("\nMissing packages detected. Install them with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True


def create_sample_data():
    """Create sample training data for testing."""
    from utils.data_processing import create_sample_data
    
    # Create sample training data
    train_path = "data/train.jsonl"
    eval_path = "data/eval.jsonl"
    
    if not Path(train_path).exists():
        create_sample_data(train_path, num_conversations=1000)
        print(f"✓ Created sample training data: {train_path}")
    
    if not Path(eval_path).exists():
        create_sample_data(eval_path, num_conversations=100)
        print(f"✓ Created sample evaluation data: {eval_path}")


def validate_setup():
    """Validate the complete setup."""
    print("\nValidating setup...")
    
    # Check if we can import our modules
    try:
        from config.config_manager import Config
        from core.tokenizer import ConversationTokenizer
        from core.model import TransformerModel
        print("✓ All modules import successfully")
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    
    # Check if we can create a basic config
    try:
        config = Config()
        print("✓ Configuration system works")
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        return False
    
    return True


def main():
    """Main setup function."""
    print("Setting up Production Conversational Transformer Training System...")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        print("\nPlease install missing dependencies and run setup again.")
        sys.exit(1)
    
    # Create sample data
    try:
        create_sample_data()
    except Exception as e:
        print(f"Warning: Could not create sample data: {e}")
    
    # Validate setup
    if validate_setup():
        print("\n" + "=" * 60)
        print("✓ Setup completed successfully!")
        print("\nYou can now run training with:")
        print("python Main.py --config debug --test-generation")
    else:
        print("\n" + "=" * 60)
        print("✗ Setup validation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()