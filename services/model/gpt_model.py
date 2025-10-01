import torch.nn as nn
import torch

from services.model.transformer.transformer import Transformer
from services.model.utils.layernorm import LayerNorm


class GPT_backbone(nn.Module):
    def __init__(self, conf):
        super().__init__()
        # Init embeddings
        self.tok_emb = nn.Embedding(conf.vocab_size, conf.emb_dim)  # [50257 x 720]
        self.pos_emb = nn.Embedding(conf.context_length, conf.emb_dim)  # [720 x 128]
        self.dropout = nn.Dropout(conf.drop_rate)

        # Create transformer blocks
        self.trf_blocks = nn.Sequential(
            *[Transformer(conf) for _ in range(conf.n_layers)]
        )

        # Normalization
        self.final_norm = LayerNorm(conf.emb_dim)

        # outout layer
        self.out_head = nn.Linear(conf.emb_dim, conf.vocab_size, bias=False)

    def forward(
        self,
        in_idx,
    ):
        batch_size, seq_lenght = in_idx.shape  # [8 x 128] -> [8 x 128 x 720]
        # Create weight embeddings
        # Embedding is a kind of high dimensional dictionary for each word
        tok_emb = self.tok_emb(in_idx)  # [batch_size x seq_lenght x emb_dim]
        # Posititonal embedding
        # Positional embedding is necessary to include information about place in the senteces
        # Sometimes you have repetated words in a sentence, with this pos embedding won't be assigned
        # same value to the same word
        pos_emb = self.pos_emb(torch.arange(seq_lenght))  # [seq_lenght x embx_dim]

        # Create input
        X = tok_emb + pos_emb
        X = self.dropout(X)
        X = self.trf_blocks(X)
        X = self.final_norm(X)
        logits = self.out_head(X)

        return logits
