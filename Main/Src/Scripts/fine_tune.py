import os
import time
import math
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

torch.set_num_threads(2)
torch.set_num_interop_threads(1)

# --- Device selection ---
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
    print("Using device: MPS (Apple Silicon)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using device: CUDA (NVIDIA GPU)")
else:
    device = torch.device("cpu")
    print("Using device: CPU")

# --- Dataset class ---
class CharDataset(Dataset):
    def __init__(self, text, seq_length, char_to_ix):
        self.seq_length = seq_length
        self.data = [char_to_ix[c] for c in text]

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        return (
            torch.tensor(self.data[idx:idx + self.seq_length], dtype=torch.long),
            torch.tensor(self.data[idx + 1:idx + self.seq_length + 1], dtype=torch.long)
        )

# --- Load and process dataset with role tokens and English filter ---
def load_text_data(path):
    """
    Load text from JSONL, filtering only English messages,
    skipping deleted ones, and adding role special tokens:
    <|user|> for "prompter", <|assistant|> for "assistant".
    Returns a concatenated string with newline separation.
    """
    role_tokens = {
        "prompter": "<|user|>",
        "assistant": "<|assistant|>"
    }

    texts = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if record.get("deleted", False):
                continue  # Skip deleted messages
            if record.get("lang") != "en":
                continue  # Skip non-English messages

            text = record.get("text", "").strip()
            if not text:
                continue

            role = record.get("role", "").lower()
            token = role_tokens.get(role, "")

            if token:
                texts.append(f"{token} {text}")
            else:
                texts.append(text)

    return "\n".join(texts) + "\n"

# --- Positional Encoding ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# --- Transformer Model ---
class CharTransformer(nn.Module):
    def __init__(self, vocab_size, hidden_size, seq_length, num_layers, nhead, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_enc = PositionalEncoding(hidden_size, seq_length, dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.fc_out = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc_out.bias.data.zero_()
        self.fc_out.weight.data.uniform_(-initrange, initrange)

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        return mask.masked_fill(mask == 1, float('-inf'))

    def forward(self, x):
        seq_len = x.size(1)
        mask = self.generate_square_subsequent_mask(seq_len).to(x.device)
        x = self.embedding(x) * math.sqrt(self.hidden_size)
        x = self.pos_enc(x)
        x = self.decoder(x, x, tgt_mask=mask)
        x = self.layer_norm(x)
        x = self.dropout(x)
        return self.fc_out(x)

# --- Load pre-trained model for fine-tuning ---
def load_pretrained_model(model_path, device):
    print(f"Loading pre-trained model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

    model_config = {
        'vocab_size': checkpoint['vocab_size'],
        'hidden_size': checkpoint['hidden_size'],
        'seq_length': checkpoint['seq_length'],
        'num_layers': checkpoint['num_layers'],
        'nhead': checkpoint['nhead']
    }

    char_to_ix = checkpoint['char_to_ix']
    ix_to_char = checkpoint['ix_to_char']

    return model_config, char_to_ix, ix_to_char, checkpoint

# --- Adapt vocab ---
def adapt_vocabulary(pretrained_char_to_ix, pretrained_ix_to_char, new_text):
    new_chars = set(new_text)
    pretrained_chars = set(pretrained_char_to_ix.keys())
    new_vocab_chars = new_chars - pretrained_chars

    if new_vocab_chars:
        print(f"Found {len(new_vocab_chars)} new chars: {sorted(new_vocab_chars)}")
        extended_char_to_ix = pretrained_char_to_ix.copy()
        extended_ix_to_char = pretrained_ix_to_char.copy()
        next_idx = len(pretrained_char_to_ix)
        for c in sorted(new_vocab_chars):
            extended_char_to_ix[c] = next_idx
            extended_ix_to_char[next_idx] = c
            next_idx += 1
        return extended_char_to_ix, extended_ix_to_char, len(new_vocab_chars)
    else:
        print("No new chars found. Using original vocab.")
        return pretrained_char_to_ix, pretrained_ix_to_char, 0

# --- Extend model vocab layers ---
def extend_model_vocabulary(model, old_vocab_size, new_vocab_size):
    if new_vocab_size > old_vocab_size:
        print(f"Extending model vocab from {old_vocab_size} to {new_vocab_size}")
        old_emb = model.embedding.weight.data
        new_emb = nn.Embedding(new_vocab_size, model.hidden_size).to(device)
        new_emb.weight.data[:old_vocab_size] = old_emb
        nn.init.uniform_(new_emb.weight.data[old_vocab_size:], -0.1, 0.1)
        model.embedding = new_emb

        old_fc_w = model.fc_out.weight.data
        old_fc_b = model.fc_out.bias.data
        new_fc = nn.Linear(model.hidden_size, new_vocab_size).to(device)
        new_fc.weight.data[:old_vocab_size] = old_fc_w
        new_fc.bias.data[:old_vocab_size] = old_fc_b
        nn.init.uniform_(new_fc.weight.data[old_vocab_size:], -0.1, 0.1)
        nn.init.zeros_(new_fc.bias.data[new_vocab_size - (new_vocab_size - old_vocab_size):])
        model.fc_out = new_fc

# --- Learning rate scheduler ---
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']

    def step(self, step):
        if step < self.warmup_steps:
            lr = self.base_lr * step / self.warmup_steps
        else:
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

# --- Training ---
def main():
    pretrained_model_path = "Model.pth"  # your pretrained model path
    data_path = os.path.join("oasst1_data", "oasst1_train.jsonl")  # path to train JSONL
    output_model_path = "FineTuned_Model.pth"

    learning_rate = 1e-4
    epochs = 50
    batch_size = 16
    dropout = 0.1
    warmup_steps = 500
    weight_decay = 0.01

    print("Loading fine-tuning dataset...")
    text = load_text_data(data_path)
    print(f"Dataset loaded. Length: {len(text):,} chars")

    model_config, pretrained_char_to_ix, pretrained_ix_to_char, checkpoint = load_pretrained_model(pretrained_model_path, device)
    char_to_ix, ix_to_char, new_vocab_count = adapt_vocabulary(pretrained_char_to_ix, pretrained_ix_to_char, text)

    model = CharTransformer(
        vocab_size=model_config['vocab_size'],
        hidden_size=model_config['hidden_size'],
        seq_length=model_config['seq_length'],
        num_layers=model_config['num_layers'],
        nhead=model_config['nhead'],
        dropout=dropout
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])

    if new_vocab_count > 0:
        extend_model_vocabulary(model, model_config['vocab_size'], len(char_to_ix))

    vocab_size = len(char_to_ix)
    seq_length = model_config['seq_length']

    print(f"Fine-tuning model:")
    print(f"  Original vocab size: {model_config['vocab_size']}")
    print(f"  New vocab size: {vocab_size}")
    print(f"  Architecture: hidden size {model_config['hidden_size']}, layers {model_config['num_layers']}, heads {model_config['nhead']}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    dataset = CharDataset(text, seq_length, char_to_ix)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    print(f"Dataset sequences: {len(dataset):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    total_steps = len(dataloader) * epochs
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps)

    step = 0
    best_loss = float('inf')
    global_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_start = time.time()
        total_loss = 0
        total_correct = 0
        total_chars = 0

        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.reshape(-1, vocab_size), targets.reshape(-1))
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            current_lr = scheduler.step(step)
            step += 1

            total_loss += loss.item() * inputs.numel()
            preds = outputs.argmax(dim=2)
            total_correct += (preds == targets).sum().item()
            total_chars += targets.numel()

        avg_loss = total_loss / total_chars
        accuracy = 100 * total_correct / total_chars
        epoch_time = time.time() - epoch_start
        elapsed = time.time() - global_start

        print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}% | LR: {current_lr:.2e} | "
              f"Time: {epoch_time:.1f}s | Elapsed: {elapsed:.1f}s")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "char_to_ix": char_to_ix,
                "ix_to_char": ix_to_char,
                "hidden_size": model_config['hidden_size'],
                "num_layers": model_config['num_layers'],
                "nhead": model_config['nhead'],
                "vocab_size": vocab_size,
                "seq_length": seq_length,
                "epoch": epoch,
                "loss": avg_loss
            }, output_model_path)

    print(f"Fine-tuning complete. Model saved to '{output_model_path}'")

if __name__ == "__main__":
    main()