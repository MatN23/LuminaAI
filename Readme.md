🧠 LuminaAI — Character-Level Transformer

A lightweight, character-level Transformer model implemented in PyTorch, designed to train on custom conversational datasets (e.g. OpenAssistant). Built for experimentation, learning, and development of simple AI chatbots and NLP models from the ground up.

🚀 Features

🔡 Character-Level Modeling – Fine-grained language understanding without tokenizers.
⚙️ Transformer Encoder – Multi-head attention, positional encodings, feed-forward layers.
🧪 Training Loop – Supports gradient clipping, batching, custom sequence lengths.
💻 Hardware Support – Runs on CPU, CUDA (NVIDIA GPUs), and MPS (Apple Silicon).
🧾 Custom Dataset Support – Load datasets in JSONL or plain text for dialogue training.
🧠 Open-Source Learning Tool – Easy to read, modify, and experiment with.
📂 Project Structure

main/
│
├── src/                  # Core model code
│   ├── model.py          # Transformer model definition
│   ├── dataset.py        # Data loading and preprocessing
│   └── train.py          # Training loop
│      
├── LICENSE
├── README.md
└── requirements.txt
⚡ Quick Start

🧠 How It Works

This model uses a simplified Transformer encoder that learns from characters rather than full words or tokens. It predicts the next character based on a sequence of previous ones, making it highly flexible and easy to train on smaller datasets.

You can modify:

hidden_size
num_layers
seq_len
batch_size
learning_rate
All in train.py or via CLI arguments.

🧪 Example Dataset Format

Supports JSONL conversational format:

{ "user": "Hello!", "bot": "Hi there!" }
Or simple text format:

<|user|> How are you?
<|bot|> I'm fine, thank you.
📜 License

This project is licensed under a custom commercial-restricted license:

✅ Free for personal and non-commercial use
🚫 Commercial use requires a paid license and written permission
⚖️ Legal protection under U.S. (California) law
See LICENSE for full terms.

🔒 Security

If you discover a vulnerability, please report it privately. See SECURITY.md for guidelines.

🤝 Contributing

Contributions are welcome!

Fork the repo
Create a new branch
Submit a PR with a clear explanation of changes
📬 Contact

For questions, licensing, or commercial inquiries:
📧 matiasnhmb@gmail.com
