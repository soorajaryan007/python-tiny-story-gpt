import torch
import pickle
from model import StoryGPT

# -------------------------
# DEVICE + PARAMS
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

embed_dim = 256
num_heads = 4
ff_dim = 512
num_layers = 3
max_len = 256

# -------------------------
# LOAD VOCAB
# -------------------------
with open("vocab.pkl", "rb") as f:
    stoi, itos = pickle.load(f)

vocab_size = len(stoi)

def encode(text):
    return [stoi[w] for w in text.split()]

def decode(ids):
    words = []
    for i in ids:
        w = itos[i]
        if w not in ("<SOS>", "<PAD>"):
            words.append(w)
    return " ".join(words)

# -------------------------
# INITIALIZE MODEL
# -------------------------
model = StoryGPT(vocab_size, embed_dim, num_heads, ff_dim,
                 num_layers, max_len, device).to(device)

model.load_state_dict(torch.load("story_model.pth", map_location=device))
model.eval()

# -------------------------
# GENERATION FUNCTION
# -------------------------
def generate_story(prompt, max_new=80):
    src = torch.tensor([encode(prompt)]).to(device)
    tgt = torch.tensor([[stoi["<SOS>"]]], device=device)

    for _ in range(max_new):
        out = model(src, tgt)
        next_id = out[0, -1].argmax().item()

        if next_id == stoi["<EOS>"]:
            break

        tgt = torch.cat([tgt,
                         torch.tensor([[next_id]], device=device)], dim=1)

    return decode(tgt[0].tolist())

# -------------------------
# TEST
# -------------------------
print("\nGenerated Story:\n")
print(generate_story("ghost story"))
