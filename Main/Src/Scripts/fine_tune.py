import time
import torch
import torch.nn as nn
import torch.optim as optim

# --- Device selection with fallback ---
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
    print("Using device: MPS (Apple Silicon GPU)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using device: CUDA (NVIDIA GPU)")
else:
    device = torch.device("cpu")
    print("Using device: CPU")

# --- Load and process dataset ---
def load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    text = ""
    for line in lines:
        line = line.strip()
        if line.startswith("### User:"):
            text += "<|user|>" + line[len("### User:"):].strip() + "\n"
        elif line.startswith("### Bot:"):
            text += "<|bot|>" + line[len("### Bot:"):].strip() + "\n"
        else:
            text += line + "\n"
    return text

# --- Hyperparameters (use same as original or modify) ---
hidden_size = 1024
seq_length = 500
batch_size = 256
learning_rate = 1e-4  # Usually lower for fine-tuning
epochs = 100  # Fewer epochs for fine-tuning
num_layers = 12

# --- Model class (same as original) ---
class CharLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers=3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        x = self.embed(x)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size):
        return (
            torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device),
            torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device),
        )

# --- Load your saved model checkpoint ---
checkpoint = torch.load("model.pth", map_location=device)

vocab_size = checkpoint["vocab_size"]
hidden_size = checkpoint["hidden_size"]
num_layers = checkpoint["num_layers"]
char_to_ix = checkpoint["char_to_ix"]
ix_to_char = checkpoint["ix_to_char"]

model = CharLSTM(vocab_size, hidden_size, num_layers).to(device)
model.load_state_dict(checkpoint["model_state_dict"])

# Optimize with IPEX if CPU and available

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# --- Load new fine-tuning data ---
data_path = "fine_tune_data.txt"  # Change this path to your fine-tune dataset
text = load_dataset(data_path)

# Make sure vocab matches original — if not, you may need vocab update or error handling
# Here we assume same vocab, so convert chars to indices with char_to_ix
data_ix = [char_to_ix.get(c, 0) for c in text]  # Use 0 index for unknown chars if any

# --- Prepare batches ---
def get_batches(data, batch_size, seq_length):
    total_length = len(data)
    num_batches = total_length // (batch_size * seq_length)
    data = data[:num_batches * batch_size * seq_length]
    data = torch.tensor(data, dtype=torch.long).to(device)
    data = data.view(batch_size, -1)

    for i in range(0, data.size(1) - seq_length, seq_length):
        inputs = data[:, i:i+seq_length]
        targets = data[:, i+1:i+seq_length+1]
        yield inputs, targets

# --- Fine-tuning loop ---
start_time = time.time()

for epoch in range(1, epochs + 1):
    model.train()
    hidden = model.init_hidden(batch_size)
    total_loss = 0
    total_correct = 0
    total_chars = 0

    for inputs, targets in get_batches(data_ix, batch_size, seq_length):
        optimizer.zero_grad()
        output, hidden = model(inputs, hidden)
        hidden = (hidden[0].detach(), hidden[1].detach())

        loss = criterion(output.reshape(-1, vocab_size), targets.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

        total_loss += loss.item() * inputs.numel()
        preds = output.argmax(dim=2)
        total_correct += (preds == targets).sum().item()
        total_chars += targets.numel()

    elapsed = time.time() - start_time
    epochs_done = epoch
    est_total = elapsed / epochs_done * epochs
    est_remaining = est_total - elapsed

    if epoch % 5 == 0 or epoch == 1:
        avg_loss = total_loss / total_chars
        accuracy = 100 * total_correct / total_chars
        print(f"Fine-tune Epoch {epoch:4d} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}% | "
              f"Elapsed: {elapsed:.1f}s | Est Remaining: {est_remaining:.1f}s")

# --- Save fine-tuned model ---
torch.save({
    "model_state_dict": model.state_dict(),
    "char_to_ix": char_to_ix,
    "ix_to_char": ix_to_char,
    "hidden_size": hidden_size,
    "num_layers": num_layers,
    "vocab_size": vocab_size
}, "char_lstm_model_finetuned.pth")
print("Fine-tuned model saved to 'FineTunedModel.pth'")
