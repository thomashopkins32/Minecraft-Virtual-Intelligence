import torch
from torch.utils.tensorboard import SummaryWriter
from gymnasium import spaces

from mvi.agent.agent import AgentV1


writer = SummaryWriter()

action_space = spaces.MultiDiscrete([3, 3, 4, 25, 25, 8, 244, 36])
agent = AgentV1(action_space=action_space)

x = torch.randn((1, 3, 160, 256))
x_roi = torch.randn((1, 3, 20, 20))

writer.add_graph(agent, input_to_model=(x, x_roi), verbose=True, use_strict_trace=False)