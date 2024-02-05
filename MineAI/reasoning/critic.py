import torch.nn as nn


class LinearReasoner(nn.Module):
    '''
    Feed-forward reasoner (critic) module.

    This module estimates future reward for a given input.
    '''
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass