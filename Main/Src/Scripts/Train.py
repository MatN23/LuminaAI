import time
import math
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from pathlib import Path
import argparse

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

# --- Load and process dataset ---
def load_text_data(path):
    """
    Load text from a JSONL file with structure similar to your snippet.
    Only keeps English, non-deleted messages.
    Prepends special tokens based on 'role' to help model distinguish speaker turns.
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
                continue  # Skip deleted entries
            if record.get("lang") != "en":
                continue  # Skip non-English

            text = record.get("text") or record.get("content") or ""
            if not text and "message" in record:
                text = record["message"].get("text", "")

            text = text.strip()
            if not text:
                continue

            role = record.get("role", "").lower()
            token = role_tokens.get(role, "")  # Default empty if role unknown

            if token:
                texts.append(f"{token} {text}")
            else:
                # If role missing or unknown, just append text
                texts.append(text)

    return "\n".join(texts) + "\n"

# --- Improved Positional Encoding ---
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

# --- Improved Transformer Model ---
class CharTransformer(nn.Module):
    def __init__(self, vocab_size, hidden_size, seq_length, num_layers, nhead, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_enc = PositionalEncoding(hidden_size, seq_length, dropout)
        
        # Use decoder layers for causal language modeling
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
        
        # Initialize weights
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc_out.bias.data.zero_()
        self.fc_out.weight.data.uniform_(-initrange, initrange)

    def generate_square_subsequent_mask(self, sz):
        """Generate causal mask for autoregressive generation"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        return mask.masked_fill(mask == 1, float('-inf'))

    def forward(self, x):
        seq_len = x.size(1)
        mask = self.generate_square_subsequent_mask(seq_len).to(x.device)
        
        # Embedding and positional encoding
        x = self.embedding(x) * math.sqrt(self.hidden_size)  # Scale embeddings
        x = self.pos_enc(x)
        
        # Transformer decoder (for causal modeling)
        x = self.decoder(x, x, tgt_mask=mask)
        x = self.layer_norm(x)
        x = self.dropout(x)
        
        return self.fc_out(x)

# --- Training utilities ---
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_checkpoint(model, optimizer, epoch, loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)

def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']

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

# --- Generate text function ---
def generate_text(model, char_to_ix, ix_to_char, prompt="<|user|>", max_length=200, temperature=0.8):
    model.eval()
    with torch.no_grad():
        # Encode prompt
        input_ids = torch.tensor([char_to_ix.get(c, 0) for c in prompt], dtype=torch.long).unsqueeze(0).to(device)
        generated = input_ids.clone()
        
        for _ in range(max_length):
            # Only use last seq_length tokens to avoid memory issues
            input_seq = generated[:, -512:] if generated.size(1) > 512 else generated
            
            outputs = model(input_seq)
            next_token_logits = outputs[0, -1, :] / temperature
            
            # Apply softmax and sample
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
            
            # Stop if we generate end token or reach max length
            if next_token.item() == char_to_ix.get('\n', 0):
                break
                
        # Decode generated text
        generated_text = ''.join([ix_to_char[idx.item()] for idx in generated[0]])
        return generated_text

# --- Main training function ---
def main():
    # --- Hyperparameters ---
    hidden_size = 512
    seq_length = 512
    batch_size = 16  # Reduced for better gradient updates
    num_layers = 6   # Reduced for faster training
    nhead = 8
    learning_rate = 3e-4  # Slightly lower learning rate
    epochs = 100
    dropout = 0.1
    warmup_steps = 1000
    
    # --- Load dataset and encode ---
    data_path = "../oasst1_data/oasst1_train.jsonl"
    print("Loading dataset...")
    text = load_text_data(data_path)
    print(f"Dataset loaded. Text length: {len(text):,} characters")

    # Create vocabulary
    chars = sorted(set(text))
    vocab_size = len(chars)
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    ix_to_char = {i: ch for i, ch in enumerate(chars)}
    print(f"Vocabulary size: {vocab_size}")

    # Create dataset and dataloader
    dataset = CharDataset(text, seq_length, char_to_ix)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    print(f"Dataset size: {len(dataset):,} sequences")

    # --- Initialize model ---
    model = CharTransformer(vocab_size, hidden_size, seq_length, num_layers, nhead, dropout).to(device)
    print(f"Model parameters: {count_parameters(model):,}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Learning rate scheduler
    total_steps = len(dataloader) * epochs
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps)

    # --- Training loop ---
    print("Starting training...")
    global_start_time = time.time()
    step = 0
    best_loss = float('inf')

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        total_loss = 0
        total_correct = 0
        total_chars = 0

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.reshape(-1, vocab_size), targets.reshape(-1))
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Update learning rate
            current_lr = scheduler.step(step)
            step += 1

            # Statistics
            total_loss += loss.item() * inputs.numel()
            preds = outputs.argmax(dim=2)
            total_correct += (preds == targets).sum().item()
            total_chars += targets.numel()

        # Epoch statistics
        epoch_time = time.time() - epoch_start
        elapsed_time = time.time() - global_start_time
        avg_loss = total_loss / total_chars
        acc = 100 * total_correct / total_chars
        
        print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Accuracy: {acc:.2f}% | "
              f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s | Elapsed: {elapsed_time:.1f}s")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "char_to_ix": char_to_ix,
                "ix_to_char": ix_to_char,
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "nhead": nhead,
                "vocab_size": vocab_size,
                "seq_length": seq_length,
                "epoch": epoch,
                "loss": avg_loss
            }, "Model.pth")

        # Generate sample text every 10 epochs
        if epoch % 10 == 0:
            print("\n--- Sample Generation ---")
            # Use first few characters of your text as prompt
            prompt = text[:20] if len(text) > 20 else text[:5]
            sample = generate_text(model, char_to_ix, ix_to_char, prompt, max_length=200)
            print(sample[:300] + "..." if len(sample) > 300 else sample)
            print("--- End Sample ---\n")

        # Save checkpoint every 25 epochs
        if epoch % 25 == 0:
            save_checkpoint(model, optimizer, epoch, avg_loss, f"checkpoint_epoch_{epoch}.pth")

    print("✅ Training complete. Model saved to 'Model.pth'")

if __name__ == "__main__":
    main()