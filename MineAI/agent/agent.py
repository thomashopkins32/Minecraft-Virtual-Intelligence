from typing import Tuple

import torch
import torch.nn as nn
from torchvision.transforms.functional import crop # type: ignore
import gymnasium

from MineAI.perception.visual import VisualPerception
from MineAI.affector.affector import LinearAffector
from MineAI.reasoning.critic import LinearReasoner


class AgentV1(nn.Module):
    """
    Overarching module for the MineAI agent.

    Version 1 of this agent will stand in place and observe the environment. It will be allowed to move its head only to look around.
    So, it will have a visual perception module to start with and a simple learning algorithm to guide where its looking.
    This is to test the following:
    - Can the agent learn to focus its attention on new information?
    - How fast is the visual perception module? Does it need to be faster?
    """

    def __init__(self, action_space: gymnasium.MultiDiscrete):
        super().__init__()
        self.roi_shape = (32, 32)
        self.vision = VisualPerception(out_channels=32, roi_shape=self.roi_shape)
        self.affector = LinearAffector(32 + 32, action_space)
        self.reasoner = LinearReasoner(32 + 32)
        self.next_roi_coords = None

    def forward(self, x_obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.next_roi_coords is not None:
            x_roi = crop(x_obs, self.next_roi_coords[0], self.next_roi_coords[1], self.roi_shape[0], self.roi_shape[1])
        else:
            x_roi = None
        x = self.vision(x_obs, x_roi)
        actions = self.affector(x)
        self.next_roi_coords = actions[-1]
        value = self.reasoner(x)

        return actions[:-1], value
