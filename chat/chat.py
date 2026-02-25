import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import sys

# Config:
device = 'cpu'
n_embd = 384
n_head = 6
block_size = 256
model_path = 'tinystoriesgru.pt'

# Vocab:
chars = ['\n', ' ', '!', '"', '#', '$', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '>', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '|', '\xad', '´', 'é', 'ñ', '\u200a', '\u200b', '–', '—', '‘', '’', '“', '”', '…', ' 🎓']
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Architecture:
class tinystoriesgru(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head):
        super().__init__()
        self.n_embd = n_embd
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd) 
        self.dropout = nn.Dropout(0.0)
        self.attn = nn.MultiheadAttention(n_embd, n_head, batch_first=True)
        self.gru_cell = nn.GRUCell(n_embd * 3, n_embd)
        self.priority_gate = nn.Linear(n_embd, 1)
        self.memory_writer = nn.Linear(n_embd, n_embd)
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
    def generate_step(self, current_idx, history_idx, hidden, memory):
# current_idx: The single token we just generated [1, 1], history_idx: The last ~256 tokens [1, T] (For Attention):
        T = history_idx.shape[1]        
# Crop history if it exceeds block_size:
        if T > block_size:
            history_idx = history_idx[:, -block_size:]
            T = block_size
        tok_emb = self.token_emb(history_idx)
        pos_emb = self.pos_emb(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        attn_mask = torch.triu(
            torch.ones(T, T, device=device), diagonal=1).bool()
        attn_out, _ = self.attn(x, x, x, attn_mask=attn_mask)
        current_context = attn_out[:, -1, :] # [1, n_embd]
        current_token_emb = self.token_emb(current_idx).squeeze(1) # [1, n_embd]
        combined_input = torch.cat([current_token_emb, current_context, memory], dim=1)
        hidden = self.gru_cell(combined_input, hidden)
        p = torch.sigmoid(self.priority_gate(hidden))
        candidate = torch.tanh(self.memory_writer(hidden))
        memory = (1 - p) * memory + p * candidate
        out = self.ln_f(hidden)
        logits = self.head(out)
        
        return logits, hidden, memory

model = tinystoriesgru(vocab_size, n_embd, n_head).to(device)

if not os.path.exists(model_path):
    files = [f for f in os.listdir('.') if f.startswith('tinystories_step_')]
    if len(files) > 0:
        files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        model_path = files[-1]
        print(f"Latest checkpoint: {model_path}")
    else:
        print("No checkpoints found.")
        sys.exit()

try:
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
except Exception as e:
    print(f"Error loading: {e}")
    sys.exit()

# Loop play:
while True:
    user_input = input("\n> ")
    if user_input.lower() == 'exit': break
# Parse Temperature: (You can specify temperature using:):
    if ':' in user_input:
        prompt_str, temp_str = user_input.split(':')
        prompt = prompt_str.strip()
        try: temperature = float(temp_str)
        except: temperature = 0.6
    else:
        prompt = user_input
        temperature = 0.6
# The tokensier is character level and suffers when the prompt ends with the word without a space. So it is added by default:
    if len(prompt) > 0 and not prompt.endswith(' '): prompt += ' '

    try:
        idx = torch.tensor([encode(prompt)], dtype=torch.long).to(device)
    except:
        print("Unknown characters.")
        continue

    print(f"{prompt}", end='', flush=True)
    hidden = torch.zeros(1, n_embd, device=device)
    memory = torch.zeros(1, n_embd, device=device)
# Buffer:
    history = idx 

    with torch.no_grad():
        for t in range(idx.shape[1] - 1):
# We feed the history up to point t:
            curr = idx[:, t].unsqueeze(1)
            hist = idx[:, :t+1] 
            _, hidden, memory = model.generate_step(curr, hist, hidden, memory)
        curr = idx[:, -1].unsqueeze(1)
        hist = idx
        logits, hidden, memory = model.generate_step(curr, hist, hidden, memory)

# First token:
    probs = F.softmax(logits / temperature, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    
# Loop generation:
    for _ in range(500):
        char = itos[next_token.item()]
        print(char, end='', flush=True)
        if char == '<': break 
        
# Updates history:
        history = torch.cat([history, next_token], dim=1)
        if history.shape[1] > block_size:
            history = history[:, -block_size:]

        with torch.no_grad():
            logits, hidden, memory = model.generate_step(next_token, history, hidden, memory)
            
            probs = F.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)


    print("\n")
