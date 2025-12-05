import torch
import torch.nn as nn

class StoryGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_layers, max_len, device):
        super().__init__()

        self.device = device
        self.tok_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_len, embed_dim)

        layer = nn.TransformerDecoderLayer(
            embed_dim, num_heads, ff_dim, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers)
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, src, tgt):
        B, S = src.shape
        B, T = tgt.shape

        src_pos = torch.arange(S, device=self.device)
        tgt_pos = torch.arange(T, device=self.device)

        src_emb = self.tok_emb(src) + self.pos_emb(src_pos)
        tgt_emb = self.tok_emb(tgt) + self.pos_emb(tgt_pos)

        out = self.decoder(tgt_emb, src_emb)
        return self.fc(out)
