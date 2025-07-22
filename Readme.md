Overview

This project implements a character-level Transformer model trained on conversational datasets (e.g., OpenAssistant). Built with PyTorch, it supports CUDA GPUs and Apple Silicon MPS devices.

Features

Character-level Transformer architecture
Dataset loading and preprocessing from JSONL files
Training with configurable hyperparameters
Supports CUDA and MPS acceleration
Saves model and vocab mappings for reuse
Requirements

Python 3.10+
PyTorch 2.0+
CUDA-enabled GPU (recommended) or Apple Silicon MPS
Dataset files in JSONL format

Download and prepare the dataset:

python Dataset_download.py
Usage

Train the model:

python train_transformer.py
The script will automatically use CUDA if available, then MPS, or fallback to CPU.

Configuration

Adjust hyperparameters in train_transformer.py:

hidden_size = 256
num_layers = 4
nhead = 8
batch_size = 64
seq_length = 128
learning_rate = 5e-4
epochs = 500
License and Commercial Use

This project is licensed under the MIT License for personal, educational, and research purposes.

Commercial use requires a separate license and royalty agreement.

Please contact Matias Nielsen at matiasnhmb@gmail.com for commercial licensing inquiries.

Contact:

For questions or support, email: matiasnhmb@gmail.com
