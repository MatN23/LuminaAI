import torch
import torch.nn as nn
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class CharLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers=2):
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

    def init_hidden(self, batch_size=1):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device))

# Load model and vocab
checkpoint = torch.load("char_lstm_model.pth", map_location=device)
char_to_ix = checkpoint["char_to_ix"]
ix_to_char = checkpoint["ix_to_char"]
hidden_size = checkpoint["hidden_size"]
num_layers = checkpoint["num_layers"]
vocab_size = len(char_to_ix)

model = CharLSTM(vocab_size, hidden_size, num_layers).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

def top_k_sampling(probs, k=5):
    top_k_idx = np.argpartition(probs, -k)[-k:]
    top_k_probs = probs[top_k_idx]
    top_k_probs = top_k_probs / top_k_probs.sum()
    chosen_idx = np.random.choice(top_k_idx, p=top_k_probs)
    return chosen_idx

def sample(model, start_str, max_length=300, temperature=1.0, top_k=5):
    input_ix = [char_to_ix.get(ch, char_to_ix.get(" ", 0)) for ch in start_str]
    input_tensor = torch.tensor(input_ix, dtype=torch.long).unsqueeze(0).to(device)

    hidden = model.init_hidden(batch_size=1)

    # Warm up hidden state with prompt
    for i in range(input_tensor.size(1)):
        _, hidden = model(input_tensor[:, i].unsqueeze(1), hidden)

    output_str = ""
    input_char = input_tensor[0, -1].unsqueeze(0).unsqueeze(0)

    for _ in range(max_length):
        out, hidden = model(input_char, hidden)
        out_dist = out[0, -1] / temperature
        probs = torch.softmax(out_dist, dim=0).cpu().detach().numpy()

        next_ix = top_k_sampling(probs, k=top_k)
        next_char = ix_to_char[next_ix]
        output_str += next_char

        if next_char == "\n":
            break

        input_char = torch.tensor([[next_ix]], dtype=torch.long).to(device)

    return output_str.strip()

print("Chat with the AI. Type 'exit' or 'quit' to stop.")

conversation_history = ""

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    # Append user input with tokens
    conversation_history += f"<|user|> {user_input} <|bot|> "
    response = sample(model, conversation_history, max_length=300, temperature=0.7, top_k=8)
    print("AI:", response)

    # Append model's response to history to keep context
    conversation_history += response + "\n"
