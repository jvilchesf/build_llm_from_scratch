import torch.nn as nn

from services.model.transformer.multihead_attention import MultiHeadAttention
from services.model.utils.layernorm import LayerNorm
from services.model.utils.feedforward import FeedForward


class Transformer(nn.Module):
    def __init__(self, conf):
        """
        Class created to implement the all transformer block that consist on:
        1. Layer Norm 1
        2. Mask multihead attention  (mmha)
        3. Dropout
        4. Layer Norm 2
        5. Feed Forward
            5.1 Linear Layer
            5.2 Gelu
            5.3 Linear Layer
        6. Dropout
        """
        super().__init__()
        # Init var
        self.norm1 = LayerNorm(conf.emb_dim)
        self.att = MultiHeadAttention(
            conf.emb_dim,
            conf.emb_dim,
            conf.context_length,
            conf.drop_rate,
            conf.n_heads,
            conf.qkv_bias,
        )
        self.dropout = nn.Dropout(conf.drop_rate)
        self.norm2 = LayerNorm(conf.emb_dim)

        self.ff = FeedForward(conf)

    def forward(
        self,
        X,
    ):
        """
        Run the input through the transformer block steps
        """

        # Shortcut is the sum of the input to avoid gradient vanishing
        shortcut = X
        X = self.norm1(X)
        X = self.att(X)
        X = self.dropout(X)
        X = X + shortcut

        shortcut = X
        X = self.norm2(X)
        X = self.ff(X)
        X = X + shortcut
        return X
