import time
import math
import json
import torch
import torch.nn as nn
import torch.optim as optim
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

# --- Load and process dataset ---
def load_oasst_jsonl(path):
    text = ""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            role = record.get("role", "")
            content = record.get("text", "").strip()
            if role == "user":
                text += "<|user|>" + content + "\n"
            elif role == "assistant":
                text += "<|bot|>" + content + "\n"
    return text

# --- Positional Encoding ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

# --- Transformer Model ---
class CharTransformer(nn.Module):
    def __init__(self, vocab_size, hidden_size, seq_length, num_layers, nhead):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_enc = PositionalEncoding(hidden_size, seq_length)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc_out = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_enc(x)
        x = x.transpose(0, 1)
        x = self.encoder(x)
        x = x.transpose(0, 1)
        return self.fc_out(x)

# --- Hyperparameters ---
hidden_size = 512
seq_length = 256
batch_size = 128
num_layers = 8
nhead = 8
learning_rate = 5e-4
epochs = 500

# --- Load dataset and encode ---
data_path = "oasst1_data/oasst1_train.jsonl"
print("Loading dataset...")
text = load_oasst_jsonl(data_path)
print("Dataset loaded.")

chars = sorted(set(text))
vocab_size = len(chars)
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}
data_ix = [char_to_ix[c] for c in text]

# --- Batch generator ---
def get_batches(data, batch_size, seq_length):
    total_length = len(data)
    num_batches = total_length // (batch_size * seq_length)
    data = data[:num_batches * batch_size * seq_length]
    data = torch.tensor(data, dtype=torch.long)
    data = data.view(batch_size, -1)

    for i in range(0, data.size(1) - seq_length, seq_length):
        inputs = data[:, i:i+seq_length]
        targets = data[:, i+1:i+seq_length+1]
        yield inputs, targets

# --- Init model ---
model = CharTransformer(vocab_size, hidden_size, seq_length, num_layers, nhead).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# --- Training loop with timing ---
print("Starting training...")
global_start_time = time.time()

for epoch in range(1, epochs + 1):
    epoch_start = time.time()

    model.train()
    total_loss = 0
    total_correct = 0
    total_chars = 0

    for inputs, targets in get_batches(data_ix, batch_size, seq_length):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.reshape(-1, vocab_size), targets.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

        total_loss += loss.item() * inputs.numel()
        preds = outputs.argmax(dim=2)
        total_correct += (preds == targets).sum().item()
        total_chars += targets.numel()

    epoch_time = time.time() - epoch_start
    elapsed_time = time.time() - global_start_time
    avg_epoch_time = elapsed_time / epoch
    est_total = avg_epoch_time * epochs
    est_remaining = est_total - elapsed_time
    avg_loss = total_loss / total_chars
    acc = 100 * total_correct / total_chars

    print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Accuracy: {acc:.2f}% | "
          f"Epoch Time: {epoch_time:.1f}s | Elapsed: {elapsed_time:.1f}s | Est Remaining: {est_remaining:.1f}s")

# --- Save model ---
torch.save({
    "model_state_dict": model.state_dict(),
    "char_to_ix": char_to_ix,
    "ix_to_char": ix_to_char,
    "hidden_size": hidden_size,
    "num_layers": num_layers,
    "vocab_size": vocab_size
}, "model.pth")

print("✅ Training complete. Model saved to 'model.pth'")
