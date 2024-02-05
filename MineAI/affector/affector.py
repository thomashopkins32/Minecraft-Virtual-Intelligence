import torch.nn as nn


class LinearAffector(nn.Module):
    '''
    Feed-forward affector (action) module.

    This module produces actions for the environment given some input using linear layers.
    '''
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        pass