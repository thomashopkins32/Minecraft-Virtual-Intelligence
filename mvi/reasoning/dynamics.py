import torch
import torch.nn as nn


class InverseDynamics(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1 : torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Inverse dynamics module forward pass. This module takes as input the
        feature representation of the current state and the next state and
        tries to predict the action vector that it took to get there.

        Parameters
        ----------
        x1 : torch.Tensor
            Feature representation of the current state of the environment
        x2 : torch.Tensor
            Feature representation of the next state of the environment

        Returns
        -------
        torch.Tensor
            Action taken in the environment to get from x1 to x2
        """
        raise NotImplementedError()


class ForwardDynamics(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        Forward dynamics module forward pass. This module takes as input the
        feature representation of the current state and an action and tries to
        predict the feature representation of the next state.
        
        Parameters
        ----------
        x : torch.Tensor
            Feature representation of the current state of the environment
        a : torch.Tensor
            Action taken in the environment
        
        Returns
        -------
        torch.Tensor
            Feature representation of the next state of the environment
        """
        raise NotImplementedError()
