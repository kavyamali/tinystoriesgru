import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import sys

# Config:
batch_size = 128
block_size = 256
max_iters = 50000000
learning_rate = 8e-4
eval_interval = 200
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embd = 124
dropout = 0.1
print(f"Device: {device.upper()}")

# Datasheet:
try:
    with open('TinyStories-valid.txt', 'r', encoding='utf-8') as f: text = f.read()
except FileNotFoundError:
    print("Error: Tinystories-valid.txt not found.")
    sys.exit()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(50)
        for k in range(50):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def run_diagnostics(model):
    print("\nDiagnostics")
    with torch.no_grad():
# Hidden GRU weights:
        W_hh = model.gru_cell.weight_hh
        W_ir, W_iz, W_in = W_hh.chunk(3, 0)
        for name, W in [("Reset", W_ir), ("Update", W_iz), ("New", W_in)]:
            sr = torch.linalg.eigvals(W).abs().max().item()
            print(f"GRU {name} Spectral Radius: {sr:.4f}")

# The memory logic:
        W_mem = model.memory_writer.weight
        sr_mem = torch.linalg.eigvals(W_mem).abs().max().item()
        print(f"Memory Writer Spectral Radius: {sr_mem:.4f}")

# Architecture:
class tinystoriesgru(nn.Module):
    def __init__(self, vocab_size, n_embd):
        super().__init__()
        self.n_embd = n_embd
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.dropout = nn.Dropout(dropout)

# Input is Token + Memory (n_embd * 2):
        self.gru_cell = nn.GRUCell(n_embd * 2, n_embd)

# Memory Gating logic:
        self.priority_gate = nn.Linear(n_embd, 1)
        self.memory_writer = nn.Linear(n_embd, n_embd)

        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_emb(idx)
        pos_emb = self.pos_emb(torch.arange(T, device=device))
        x = self.dropout(tok_emb + pos_emb)

        hidden = torch.zeros(B, self.n_embd, device=device)
        memory = torch.zeros(B, self.n_embd, device=device)
        outputs = []

        for t in range(T):
            current_token = x[:, t, :]
# Cat token with current persistent memory:
            combined_input = torch.cat([current_token, memory], dim=1)

            hidden = self.gru_cell(combined_input, hidden)

            p = torch.sigmoid(self.priority_gate(hidden))
            candidate = torch.tanh(self.memory_writer(hidden))

# Update memory stream:
            memory = (1 - p) * memory + p * candidate
            outputs.append(hidden)

        rnn_out = torch.stack(outputs, dim=1)
        rnn_out = self.ln_f(rnn_out)
        logits = self.head(rnn_out)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

model = tinystoriesgru(vocab_size, n_embd).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
params = sum(p.numel() for p in model.parameters())/1e6
print(f"Model Size: {params:.2f} M Params")

# Training Loop:
for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train {losses['train']:.4f}, val {losses['val']:.4f}")
        run_diagnostics(model)
        torch.save(model.state_dict(), f'tinystories_mem_gru_{iter}.pt')

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()

# Clip grads to prevent immediate spikes:
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

# This is used to contract the radius and make it stable:
    with torch.no_grad():
        W_hh = model.gru_cell.weight_hh
        W_gates = W_hh.chunk(3, 0)
        for W in W_gates:
            sr = torch.linalg.eigvals(W).abs().max()
            if sr > 0.95:
                W.data *= (0.95 / sr)

# Leash the Memory Writer (The second feedback loop):
        W_mem = model.memory_writer.weight
        sr_mem = torch.linalg.eigvals(W_mem).abs().max()
        if sr_mem > 0.95:
            W_mem.data *= (0.95 / sr_mem)

    if iter % 10 == 0:
        print(f"Step {iter}: Loss {loss.item():.4f}")
