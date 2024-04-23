import torch
from gymnasium import spaces

from mvi.agent.agent import AgentV1


#try using torch.autograd.detect_anomaly() or torch.autograd.gradcheck()
action_space = spaces.MultiDiscrete([3, 3, 4, 25, 25, 8, 244, 36])
agent = AgentV1(action_space=action_space)
