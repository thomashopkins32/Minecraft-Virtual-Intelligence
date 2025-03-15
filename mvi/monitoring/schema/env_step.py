from dataclasses import dataclass

import torch

from .event import Event


@dataclass
class EnvStep(Event):
    """
    After a single action has been taken in the environment.

    Attributes
    ----------
    reward: float
        The reward given by the environment.
    """
    observation: torch.Tensor
    action: torch.Tensor
    next_observation: torch.Tensor
    reward: float

