from torch import nn
from services.model.utils.gelu import GELU


class FeedForward(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(conf.emb_dim, 4 * conf.emb_dim),
            GELU(),
            nn.Linear(4 * conf.emb_dim, conf.emb_dim),
        )

    def forward(
        self,
        X,
    ):
        X = self.layers(X)
        return X
