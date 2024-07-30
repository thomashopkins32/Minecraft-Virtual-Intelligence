from typing import Tuple

import torch
import torch.nn as nn
from torchvision.transforms.functional import crop, center_crop  # type: ignore
from gymnasium.spaces import MultiDiscrete

from mvi.perception.visual import VisualPerception
from mvi.affector.affector import LinearAffector
from mvi.reasoning.critic import LinearCritic
from mvi.reasoning.dynamics import InverseDynamics, ForwardDynamics
from mvi.memory.trajectory import TrajectoryBuffer
from mvi.learning.ppo import PPO
from mvi.learning.icm import ICM
from mvi.utils import sample_action


class AgentV1:
    """
    Version 1 of this agent will stand in place and observe the environment. It will be allowed to move its head only to look around.
    So, it will have a visual perception module to start with and a simple learning algorithm to guide where its looking.
    This is to test the following:
    - Can the agent learn to focus its attention on new information?
    - How fast is the visual perception module? Does it need to be faster?
    """

    def __init__(self, agent_config: AgentConfig, action_space: MultiDiscrete):
        self.config = agent_config

        self.vision = VisualPerception(out_channels=32)
        self.affector = LinearAffector(32 + 32, action_space)
        self.critic = LinearCritic(32 + 32)
        self.inverse_dynamics = InverseDynamics(32 + 32, action_space)
        self.forward_dynamics = ForwardDynamics(32 + 32, action_space)
        self.trajectory_buffer = TrajectoryBuffer(
            self.config.max_buffer_size,
            self.config.discount_factor,
            self.config.gae_discount_factor
        )
        self.ppo = PPO(self.vision, self.affector, self.critic, self.config.ppo_config)
        self.icm = ICM()

        self.roi_action = None

    def action(self, obs: torch.Tensor) -> torch.Tensor:
        """Get the action given the observation"""

        if self.roi_action is None:
            roi_obs = center_crop(obs, self.config.roi_shape)
        else:
            roi_obs = crop(
                obs,
                self.roi_action[0],
                self.roi_action[1],
                self.config.roi_shape[0],
                self.config.roi_shape[1],
            )
        
        with torch.no_grad():
            visual_features = self.vision(obs, roi_obs)
            actions = self.affector(visual_features)
            value = self.critic(visual_features)
        action, logp_action = sample_action(actions)
        self.trajectory_buffer.store_observation(obs)
        self.trajectory_buffer.store_region_of_interest(roi_obs)
        self.trajectory_buffer.store_value(value)
        self.trajectory_buffer.store_action(action, logp_action.sum())

        self.roi_action = action[-2:]
        
        # Once the trajectory buffer is full, we can start learning
        if len(self.trajectory_buffer) == self.config.max_buffer_size:
            self.ppo.update(self.trajectory_buffer)
            self.icm.update(self.trajectory_buffer)

        return action[:-2]

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
        # TODO: Is this still needed?
        x = self.vision(x_obs, x_roi)
        actions = self.affector(x)
        value = self.reasoner(x)
        return actions, value
