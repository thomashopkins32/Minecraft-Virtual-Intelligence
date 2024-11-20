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
from mvi.learning.td import TemporalDifferenceActorCritic
from mvi.config import AgentConfig
from mvi.utils import sample_action, joint_logp_action


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
        self.td = TemporalDifferenceActorCritic(self.affector, self.critic, config.td)
        self.icm = ICM(self.forward_dynamics, self.inverse_dynamics, config.icm)
        self.config = config

        # region of interest initialization
        self.roi_action: Union[torch.Tensor, None] = None
        self.prev_action_logp: Union[torch.Tensor, None] = None
        self.prev_value: Union[torch.Tensor, None] = None
        self.prev_visual_features: Union[torch.Tensor, None] = None


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
        visual_features = self.vision(obs, roi_obs)

        # TODO: Separate into separate functions for online and offline learning
        if self.prev_visual_features is not None and self.prev_action_logp is not None and self.prev_value is not None:
            td_loss = self.td.loss(self.prev_value, self.prev_action_logp, reward, visual_features.clone().detach())
            # TODO: Implement ICM loss for online learning
            icm_loss = self.icm.loss(self.prev_visual_features, , visual_features.clone().detach())
            loss = td_loss + icm_loss
            loss.backward()
            # TODO: Construct end-to-end online optimizer for agent
            self.optimizer.step()

        actions = self.affector(visual_features)
        value = self.critic(visual_features)
        action = sample_action(actions)
        logp_action = joint_logp_action(actions, action)
        self.roi_action = action[:, -2:].round().long()

        # Get the intrinsic reward associated with the previous observation
        # TODO: Possible off by one error with intrinsic rewards??
        if self.prev_visual_features is not None:
            intrinsic_reward = self.icm.intrinsic_reward(
                self.prev_visual_features, action, visual_features
            )
        else:
            intrinsic_reward = 0.0

        self.memory.store(
            visual_features, action, reward, intrinsic_reward, value, logp_action
        )

        # Once the trajectory buffer is full, we can start learning offline
        if len(self.memory) == self.config.max_buffer_size:
            self.ppo.update(self.memory)
            self.icm.update(self.memory)

        self.prev_visual_features = visual_features
        self.prev_action_logp = logp_action
        self.prev_value = value

        # Return the action suitable for the environment
        return action[:, :-2].long()
