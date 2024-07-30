import torch
import torch.nn as nn


class LinearCritic(nn.Module):
    """
    Feed-forward critic module.

    This module estimates future reward for a given input.
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.l1 = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.l1(x)
