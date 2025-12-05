import torch
import torch.nn as nn
import pickle
from model import StoryGPT

# -------------------------
# DEVICE + Hyperparams
# -------------------------
torch.set_float32_matmul_precision("high")

device = "cuda" if torch.cuda.is_available() else "cpu"
embed_dim = 256
num_heads = 4
ff_dim = 512
num_layers = 3
max_len = 256
batch_size = 8
epochs = 50
lr = 3e-4

print("Training on:", device)

# -------------------------
# Load Data
# -------------------------
raw = open("stories.txt").read().strip().split("\n")

# -------------------------
# Build Vocabulary
# -------------------------
def build_vocab(lines):
    words = set()
    for line in lines:
        if "<IN>" not in line: continue
        joined = line.replace("<IN>", "").replace("<OUT>", "")
        for w in joined.split():
            words.add(w)

    words = ["<PAD>", "<SOS>", "<EOS>"] + sorted(list(words))
    stoi = {w:i for i,w in enumerate(words)}
    itos = {i:w for i,w in enumerate(words)}
    return stoi, itos

stoi, itos = build_vocab(raw)
vocab_size = len(stoi)

with open("vocab.pkl", "wb") as f:
    pickle.dump((stoi, itos), f)

def encode(text):
    return [stoi[w] for w in text.split()]

def pad(x, L):
    return x + [stoi["<PAD>"]] * (L - len(x))

# -------------------------
# Prepare Dataset
# -------------------------
inputs, targets = [], []

for line in raw:
    if "<IN>" not in line or "<OUT>" not in line:
        continue

    inp = line.split("<IN>")[1].split("<OUT>")[0].strip()
    out = line.split("<OUT>")[1].strip()

    inp_ids = encode(inp)
    out_ids = [stoi["<SOS>"]] + encode(out) + [stoi["<EOS>"]]

    inputs.append(inp_ids)
    targets.append(out_ids)

max_in = max(len(x) for x in inputs)
max_out = max(len(x) for x in targets)

inputs = torch.tensor([pad(x, max_in) for x in inputs])
targets = torch.tensor([pad(x, max_out) for x in targets])

# -------------------------
# Model
# -------------------------
model = StoryGPT(vocab_size, embed_dim, num_heads, ff_dim,
                 num_layers, max_len, device).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss()

# -------------------------
# Training Loop
# -------------------------
for epoch in range(epochs):
    total_loss = 0

    for i in range(0, len(inputs), batch_size):
        src = inputs[i:i+batch_size].to(device)
        tgt = targets[i:i+batch_size].to(device)

        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        optimizer.zero_grad()
        logits = model(src, tgt_in)

        loss = loss_fn(logits.reshape(-1, vocab_size),
                       tgt_out.reshape(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch} Loss: {total_loss:.2f}")

# -------------------------
# SAVE MODEL
# -------------------------
torch.save(model.state_dict(), "story_model.pth")
print("\n✔ Model saved as story_model.pth")
print("✔ Vocabulary saved as vocab.pkl")
