from typing import Tuple

import torch
import torch.nn as nn
from gymnasium.spaces import MultiDiscrete

from mvi.perception.visual import VisualPerception
from mvi.affector.affector import LinearAffector
from mvi.reasoning.critic import LinearReasoner
from mvi.memory.trajectory import PPOTrajectory
from mvi.learning.ppo import PPO
from mvi.config import AgentConfig


class AgentV1:
    """
    Version 1 of this agent will stand in place and observe the environment. It will be allowed to move its head only to look around.
    So, it will have a visual perception module to start with and a simple learning algorithm to guide where its looking.
    This is to test the following:
    - Can the agent learn to focus its attention on new information?
    - How fast is the visual perception module? Does it need to be faster?
    """

    def __init__(self, config: AgentConfig, action_space: MultiDiscrete):
        self.vision = VisualPerception(out_channels=32)
        self.affector = LinearAffector(32 + 32, action_space)
        self.reasoner = LinearReasoner(32 + 32)
        self.memory = PPOTrajectory(config.max_buffer_size, config.discount_factor, config.gae_discount_factor)
        self.ppo = PPO(self.affector, self.reasoner, config.ppo_config)

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        pass

    def forward(self, x_obs: torch.Tensor, x_roi: torch.Tensor) -> Tuple[
        Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
        torch.Tensor,
    ]:
        x = self.vision(x_obs, x_roi)
        actions = self.affector(x)
        value = self.reasoner(x)
        return actions, value
