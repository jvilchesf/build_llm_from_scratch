from torch import nn
import torch


class LayerNorm(nn.Module):
    def __init__(self,
                 emb_dim: int):
        super().__init__()

        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self,
                X: torch.tensor,
                ):

        mean = X.mean(dim=-1, keepdim=True)
        var = X.var(dim=-1, keepdim=True, unbiased=False)

        norm_x = (X - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
