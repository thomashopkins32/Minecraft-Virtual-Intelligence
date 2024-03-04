import torch.nn as nn


class LinearReasoner(nn.Module):
    """
    Feed-forward reasoner (critic) module.

    This module estimates future reward for a given input.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.l1 = nn.Linear(embed_dim, 1)

    def forward(self, x):
        return self.l1(x)
