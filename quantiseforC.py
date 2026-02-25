import torch
import numpy as np

def export_full_strength(model_path, out_path):
    state_dict = torch.load(model_path, map_location='cpu')
    f = open(out_path, "wb")
    
    keys =[
        'token_emb.weight', 'pos_emb.weight', 'gru_cell.weight_ih', 'gru_cell.weight_hh', 'gru_cell.bias_ih', 'gru_cell.bias_hh','priority_gate.weight', 'priority_gate.bias', 'memory_writer.weight', 'memory_writer.bias', 'ln_f.weight', 'ln_f.bias', 'head.weight', 'head.bias'
    ]

# Use 8192 scaling: Allows weights up to +/- 4.0 without clipping:
    SCALE = 8192.0
    for k in keys:
        t = state_dict[k].float().numpy()
        
# Will warn you if any weights are still getting clipped:
        clipped = np.sum((t * SCALE > 32767) | (t * SCALE < -32768))
        if clipped > 0:
            print(f"{clipped} weights clipped in {k}")
            
        q_t = np.clip(np.round(t * SCALE), -32768, 32767).astype(np.int16)
        f.write(q_t.tobytes())
    
    f.close()

# Enter your checkpoint in place of tinystoriespuregru for input:
export_full_strength('tinystoriespuregru.pt', 'model_q15.bin')
