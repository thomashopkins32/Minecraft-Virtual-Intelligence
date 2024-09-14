from typing import Union

import torch
from torchvision.transforms.functional import center_crop, crop  # type: ignore
from gymnasium.spaces import MultiDiscrete

from mvi.perception.visual import VisualPerception
from mvi.affector.affector import LinearAffector
from mvi.reasoning.critic import LinearReasoner
from mvi.memory.trajectory import TrajectoryBuffer
from mvi.learning.ppo import PPO
from mvi.config import AgentConfig
from mvi.utils import sample_action


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
        self.memory = TrajectoryBuffer(config.max_buffer_size)
        self.inverse_dynamics = InverseDynamics(32 + 32, action_space)
        self.forward_dynamics = ForwardDynamics(32 + 32, action_space)
        self.ppo = PPO(self.affector, self.reasoner, config.ppo)
        self.config = config

        # region of interest initialization
        self.roi_action: Union[torch.Tensor, None] = None

    def _transform_observation(self, obs: torch.Tensor) -> torch.Tensor:
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
        return roi_obs

    def act(self, obs: torch.Tensor, reward: float = 0.0) -> torch.Tensor:

        roi_obs = self._transform_observation(obs)
        with torch.no_grad():
            visual_features = self.vision(obs, roi_obs)
            actions = self.affector(visual_features)
            value = self.reasoner(visual_features)
        action, logp_action = sample_action(actions)
        self.roi_action = action[-2:]

        self.memory.store(visual_features, action, reward, value, logp_action)

        # Once the trajectory buffer is full, we can start learning
        if len(self.memory) == self.config.max_buffer_size:
            self.ppo.update(self.memory)

        return action[:-2]
