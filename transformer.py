import torch.nn as nn
import torch

from multihead_attention import MultiHeadAttention
from layernorm import LayerNorm
from feedforward import FeedForward


class Transformer(nn.Module):
    def __init__(self,
                 conf
                 ):
        '''
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
        '''
        super().__init__()
        # Init var
        self.layer_norm_1 = LayerNorm(conf.emb_dim)
        self.mmha = MultiHeadAttention(conf.emb_dim, conf.emb_dim,
                                       conf.context_lenght, conf.dropout_rate,
                                       conf.num_heads, conf.qkv_bias)
        self.dropout = nn.Dropout(conf.dropout_rate)
        self.layer_norm_2 = LayerNorm(conf.emb_dim)

        self.ff = FeedForward(conf)

    def forward(self,
                X: torch.tensor,
                ):
        '''
        Run the input through the transformer block steps
        '''

        # Shortcut is the sum of the input to avoid gradient vanishing
        shortcut = X
        X = self.layer_norm_1(X)
        X = self.mmha(X)
        X = self.dropout(X)
        X = X + shortcut

        shortcut = X
        X = self.layer_norm_2(X)
        X = self.ff(X)
        X = X + shortcut
        return X
