from typing import Union

import torch
from torchvision.transforms.functional import center_crop, crop  # type: ignore
from gymnasium.spaces import MultiDiscrete

from mvi.perception.visual import VisualPerception
from mvi.affector.affector import LinearAffector
from mvi.reasoning.critic import LinearCritic
from mvi.reasoning.dynamics import InverseDynamics, ForwardDynamics
from mvi.memory.trajectory import TrajectoryBuffer
from mvi.learning.icm import ICM
from mvi.learning.ppo import PPO
from mvi.config import AgentConfig
from mvi.utils import sample_action
from mvi.monitoring import AgentMonitor


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
        self.critic = LinearCritic(32 + 32)
        self.memory = TrajectoryBuffer(config.max_buffer_size)
        self.inverse_dynamics = InverseDynamics(32 + 32, action_space)
        self.forward_dynamics = ForwardDynamics(32 + 32, action_space.shape[0] + 2)
        self.ppo = PPO(self.affector, self.critic, config.ppo)
        self.icm = ICM(self.forward_dynamics, self.inverse_dynamics, config.icm)
        self.config = config
        self.monitor = AgentMonitor(config.monitor)

        # region of interest initialization
        self.roi_action: Union[torch.Tensor, None] = None
        self.prev_visual_features: torch.Tensor = torch.zeros(
            (1, 64), dtype=torch.float
        )

        # Apply monitoring to all module forward passes
        self.vision.forward = self.monitor.monitor_module("vision")(self.vision.forward)
        self.affector.forward = self.monitor.monitor_module("affector")(self.affector.forward)
        self.critic.forward = self.monitor.monitor_module("critic")(self.critic.forward)
        self.inverse_dynamics.forward = self.monitor.monitor_module("inverse_dynamics")(self.inverse_dynamics.forward)
        self.forward_dynamics.forward = self.monitor.monitor_module("forward_dynamics")(self.forward_dynamics.forward)

    def _transform_observation(self, obs: torch.Tensor) -> torch.Tensor:
        if self.roi_action is None:
            roi_obs = center_crop(obs, self.config.roi_shape)
        else:
            roi_obs = crop(
                obs,
                self.roi_action[:, 0],
                self.roi_action[:, 1],
                self.config.roi_shape[0],
                self.config.roi_shape[1],
            )
        return roi_obs

    def act(self, obs: torch.Tensor, reward: float = 0.0) -> torch.Tensor:
        roi_obs = self._transform_observation(obs)
        with torch.no_grad():
            visual_features = self.vision(obs, roi_obs)
            actions = self.affector(visual_features)
            value = self.critic(visual_features)
        action, logp_action = sample_action(actions)
        self.roi_action = action[:, -2:].round().long()

        # Get the intrinsic reward associated with the previous observation
        with torch.no_grad():
            intrinsic_reward = self.icm.intrinsic_reward(
                self.prev_visual_features, action, visual_features
            )

        self.memory.store(
            visual_features, action, reward, intrinsic_reward, value, logp_action
        )

        # Once the trajectory buffer is full, we can start learning
        if len(self.memory) == self.config.max_buffer_size:
            self.ppo.update(self.memory)
            self.icm.update(self.memory)

        self.prev_visual_features = visual_features

        # Monitor parameters after updates
        if len(self.memory) == self.config.max_buffer_size:
            self.monitor.monitor_parameters("vision", self.vision)
            self.monitor.monitor_parameters("affector", self.affector)
            self.monitor.monitor_parameters("critic", self.critic)
            self.monitor.monitor_parameters("inverse_dynamics", self.inverse_dynamics)
            self.monitor.monitor_parameters("forward_dynamics", self.forward_dynamics)

        # Log metrics
        self.monitor.log({
            "reward": reward,
            "intrinsic_reward": intrinsic_reward,
            "value": value,
            "action_dist": actions.mean(),  # or any other action distribution metric
            "roi_position": self.roi_action.float().mean(),
        })
        self.monitor.increment_step()

        return action[:, :-2].long()

    def close(self):
        """Add a method to properly close the monitor"""
        self.monitor.close()
