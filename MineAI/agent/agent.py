from typing import Union

import torch
import torch.nn as nn

from perception.visual import VisualPerception


class AgentV1(nn.Module):
    '''
    Overarching module for the MineAI agent.

    Version 1 of this agent will stand in place and observe the environment. It will be allowed to move its head only to look around.
    So, it will have a visual perception module to start with and a simple learning algorithm to guide where its looking.
    This is to test the following:
    - Can the agent learn to focus its attention on new information?
    - How fast is the visual perception module? Does it need to be faster?
    '''
    def __init__(self):
        super().__init__()
        self.vision = VisualPerception(out_channels=32, roi_shape=(32, 32))
        # Outputs are the 25 discrete pitch deltas and 25 discrete yaw deltas for the camera
        self.pitch_action = nn.Linear(32 + 32, 25)
        self.yaw_action = nn.Linear(32 + 32, 25)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x_obs: torch.Tensor) -> Union[torch.Tensor, torch.Tensor]:
        x = self.vision(x_obs)
        pitch = self.softmax(self.pitch_action(x))
        yaw = self.softmax(self.yaw_action(x))
        return pitch, yaw
