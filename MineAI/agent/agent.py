import torch
import torch.nn as nn

from perception.visual import VisualPerception


class Agent(nn.Module):
    '''
    Overarching module for the MineAI agent.

    Version 1 of this agent will stand in place and observe the environment. It will be allowed to move its head only to look around.
    So, it will have a visual perception module to start with and a simple learning algorithm to guide where its looking.
    This is to test the following:
    - Can the agent learn to focus its attention on new information?
    - How fast is the visual perception module? Does it need to be faster?
    '''
    def __init__(self):
        pass

    def forward(self, x_obs):
        pass
