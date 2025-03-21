from dataclasses import dataclass
from datetime import datetime

import torch


@dataclass
class Event:
    """
    Base class for an event. Contains attributes common to all events.

    Attributes
    ----------
    timestamp: datetime
        The time that the event occurred.
    """

    timestamp: datetime


@dataclass
class Start(Event):
    """
    The start of the simulation.
    """

    pass


@dataclass
class Stop(Event):
    """
    The end of the simulation.
    """

    pass


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


@dataclass
class EnvReset(Event):
    """
    After the environment has been reset.
    """

    observation: torch.Tensor


@dataclass
class ModuleForwardStart(Event):
    """
    The start of a `nn.Module.forward` call.
    """

    name: str
    inputs: torch.Tensor


@dataclass
class ModuleForwardEnd(Event):
    """
    The end of a `nn.Module.forward` call.
    """

    name: str
    outputs: torch.Tensor
