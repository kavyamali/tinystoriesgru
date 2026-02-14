import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import sys

# Config:
device = 'cpu'
n_embd = 96
n_head = 6
block_size = 256
model_path = 'tinystoriesgru-0.2M-int8.pt'

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

    def generate_step(self, current_idx, history_idx, hidden, memory, anchors=None):
        T = history_idx.shape[1]        
        if T > block_size:
            history_idx = history_idx[:, -block_size:]
            T = block_size
        tok_emb = self.token_emb(history_idx)
        pos_emb = self.pos_emb(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        if anchors is not None and anchors.shape[1] > 0:
            N_anchors = anchors.shape[1]
            fake_pos_idx = torch.tensor([T-1] * N_anchors, device=device)
            anchor_pos = self.pos_emb(fake_pos_idx)
            anchors_with_pos = anchors + anchor_pos.unsqueeze(0)
            x = torch.cat([x, anchors_with_pos], dim=1)
        
# Query search based attention(to fill the memory channel the model would accept post training. This is optional and can be removed, bringing complexity to O(T2d):
        q = x[:, -1:, :]

# Keys/Values = full prefix (already causal because prefix only):
        k = x         
        v = x      
        attn_out, _ = self.attn(q, k, v)

# Extract context vector:
        current_context = attn_out[:, 0, :]  # (B, d)
        current_token_emb = self.token_emb(current_idx).squeeze(1) 
        combined_input = torch.cat([current_token_emb, current_context, memory], dim=1)
        hidden = self.gru_cell(combined_input, hidden)
        p = torch.sigmoid(self.priority_gate(hidden))
        candidate = torch.tanh(self.memory_writer(hidden))
        memory = (1 - p) * memory + p * candidate
        out = self.ln_f(hidden)
        logits = self.head(out)
        return logits, hidden, memory

# Character names (Includes some common nouns from the datasheet):
names = ["Amy","Anna","Ben","Benny","Betsy","Billy","Blue","Bob","Buddy",
         "Buzz","Clara","Daisy","Emily","Emma","Freddy","Grace","Jack","Jill",
         "Jimmy","Joe","Julie","Leo","Lily","Lilly","Lucy","Max","Mia","Mike",
         "Molly","Mittens","Nick","Pip","Rex","Roxy","Sam","Spot","Sue","Tim",
         "Tom","Tommy","Tweety","aunt","baby","bear","bird","boy","bunny","cat",
         "chicken","child","churchkeeper","cow","dad","deer","dog","doll","dragon",
         "farmer","fish","fox","goose","grandma","grandpa","horse","hippo","kitten",
         "knight","king","lady","man","manor","men","monster","mouse","people","pony",
         "prince","princess","puppy","queen","rabbit","seal","sheep","squirrel","teacher","teddy","tiger","toad","toy","uncle","woman"]

# Vocab:
chars = ['\n', ' ', '!', '"', '#', '$', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '>', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '|', '\xad', '´', 'é', 'ñ', '\u200a', '\u200b', '–', '—', '‘', '’', '“', '”', '…', ' 🎓']
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

model = tinystoriesgru(vocab_size, n_embd, n_head).to(device)

if not os.path.exists(model_path):
    files = [f for f in os.listdir('.') if f.startswith('tinystories_step_')]
    if len(files) > 0:
        files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        model_path = files[-1]
    else:
        print("No checkpoints found.")
        sys.exit()

try:
# Load the file dictionary(this is specifically used for the quantised model. The dequantisation is weight only based, not recurrent integer):
    checkpoint = torch.load(model_path, map_location=device)
    dequantized_state_dict = {}
    for k, v in checkpoint.items():
        if isinstance(v, tuple):
            q_weight, scale = v
            dequantized_state_dict[k] = q_weight.float() * scale
        else:
            dequantized_state_dict[k] = v
    
    model.load_state_dict(dequantized_state_dict)
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
        except: temperature = 0.5
    else:
        prompt = user_input
        temperature = 0.5
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
    history = idx 
    character_vault = {} 
    generated_text_buffer = prompt

    with torch.no_grad():
        for t in range(idx.shape[1] - 1):
        # We feed the history up to point t:
            curr = idx[:, t].unsqueeze(1)
            hist = idx[:, :t+1] 
            _, hidden, memory = model.generate_step(curr, hist, hidden, memory, anchors=None)
        
        curr = idx[:, -1].unsqueeze(1)
        hist = idx
        logits, hidden, memory = model.generate_step(curr, hist, hidden, memory, anchors=None)
# First token:
    probs = F.softmax(logits / temperature, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
# Loop generation:    
    for _ in range(500):
        char = itos[next_token.item()]
        print(char, end='', flush=True)
        if char == '<': break 
        
        generated_text_buffer += char
        for name in names:
            if generated_text_buffer.endswith(name):
                if name not in character_vault:
                    name_idxs = torch.tensor([encode(name)], dtype=torch.long).to(device)
                    with torch.no_grad():
                        name_vecs = model.token_emb(name_idxs) 
                        stored_concept = torch.mean(name_vecs, dim=1, keepdim=True)
                        character_vault[name] = stored_concept

        anchor_tensor = None
        if len(character_vault) > 0:
            anchor_tensor = torch.cat(list(character_vault.values()), dim=1)

        history = torch.cat([history, next_token], dim=1)
        if history.shape[1] > block_size:
            history = history[:, -block_size:]

        with torch.no_grad():
            logits, hidden, memory = model.generate_step(next_token, history, hidden, memory, anchors=anchor_tensor)
            probs = F.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

    print("\n")