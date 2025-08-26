from torch import nn
import torch
from gelu import GELU


class FeedForward(nn.Module):
    def __init__(self,
                 conf: dict
                 ):

        super().__init__()
        self.layers = nn.Sequential(
                nn.Linear(conf.emb_dim, 4 * conf.emb_dim),
                GELU(),
                nn.Linear(4 * conf.emb_dim, conf.emb_dim)
        )

    def forward(self,
                X: torch.tensor,
                ):

        X = self.layers(X)
        return X
