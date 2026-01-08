import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import sys

# Config:
batch_size = 128
block_size = 256       
max_iters = 5000
learning_rate = 8e-4
eval_interval = 200
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embd = 384
n_head = 6
dropout = 0.1
print(f"Device: {device.upper()}")
# This loads my training datasheet(downloaded from huggingface):
try:
    with open('Tinystories-valid.txt', 'r', encoding='utf-8') as f: text = f.read()
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

# Architecture:
class tinystoriesgru(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head):
        super().__init__()
        self.n_embd = n_embd
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd) 
        self.dropout = nn.Dropout(dropout)
# This is the attention head used. I've used nn.MultiheadAttention,but it could also be exlplained as follows:
#        x = tok_emb + pos_emb
# Temporal Logic (The GRU Layer). This runs fast on cuDNN. v_seq is the "Syntax Stream":
#        v_seq, h_new = self.gru(x, hidden) 
        
# The attention gates. We create a causal mask so they can't see the future:
#        attn_mask = (self.tril[:T, :T] == 0).to(device)
        
        # Gate 1: Focuses on pattern A
#        gate1_out, _ = self.attn_gate_1(v_seq, v_seq, v_seq, attn_mask=attn_mask)
        
# Fuse the gates into the main stream. We treat them as "residual corrections" to the GRU's thought:
#        refined_stream = v_seq + gate1_out
        self.attn = nn.MultiheadAttention(n_embd, n_head, batch_first=True, dropout=dropout)
# n_embd is set to 3 for the input+residual memory+ all context we get from the attention:
        self.gru_cell = nn.GRUCell(n_embd * 3, n_embd)
# The memory_writer saves the data:
        self.priority_gate = nn.Linear(n_embd, 1)
        self.memory_writer = nn.Linear(n_embd, n_embd)
# Output:
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)
# Creates mask buffer:
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, idx, targets=None):
        B, T = idx.shape
# Embedding:
        tok_emb = self.token_emb(idx) 
        pos_emb = self.pos_emb(torch.arange(T, device=device))
        x = self.dropout(tok_emb + pos_emb)
# This is the attention mask, ran on 'device'(the same as input):
        attn_mask = (self.tril[:T, :T] == 0).to(device)
# Used for older pytorch version, as I have a 970 (isCasual=true is removed):
        attn_output, _ = self.attn(x, x, x, attn_mask=attn_mask)
# Runs reccurence.
        hidden = torch.zeros(B, self.n_embd, device=device)
        memory = torch.zeros(B, self.n_embd, device=device)
        outputs = []
        
        for t in range(T):
# Input:
            current_token = tok_emb[:, t, :]    
            current_context = attn_output[:, t, :] 
# Input+ memory:
            combined_input = torch.cat([current_token, current_context, memory], dim=1)
            hidden = self.gru_cell(combined_input, hidden)
            p = torch.sigmoid(self.priority_gate(hidden))
            candidate = torch.tanh(self.memory_writer(hidden))
# This is the memory logic(similar to what was used for optimising LSTM): 
            memory = (1 - p) * memory + p * candidate
            outputs.append(hidden)
            
# Stacks the torch and decodes:
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

model = tinystoriesgru(vocab_size, n_embd, n_head).to(device)
# Using the AdamW optimiser:
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
params = sum(p.numel() for p in model.parameters())/1e6
print(f"Model Size: {params:.2f} M Params")

# Loops the training:
for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train {losses['train']:.4f}, val {losses['val']:.4f}")
        torch.save(model.state_dict(), f'tinystories_gru_{iter}.pt')

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
# Iteration interval is set to 10:    
    if iter % 10 == 0:
        print(f"Step {iter}: Loss {loss.item():.4f}")