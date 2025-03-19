import torch
from torchvision.transforms.functional import center_crop, crop  # type: ignore
from gymnasium.spaces import MultiDiscrete

from ..perception.visual import VisualPerception
from ..affector.affector import LinearAffector
from ..reasoning.critic import LinearCritic
from ..reasoning.dynamics import InverseDynamics, ForwardDynamics
from ..memory.trajectory import TrajectoryBuffer
from ..learning.icm import ICM
from ..learning.ppo import PPO
from ..config import AgentConfig
from ..utils import sample_action


class AgentV1:
    """
    Version 1 of this agent will stand in place and observe the environment. It will be allowed to move its head only to look around.
    So, it will have a visual perception module to start with and a simple learning algorithm to guide where its looking.
    This is to test the following:
    - Can the agent learn to focus its attention on new information?
    - How fast is the visual perception module? Does it need to be faster?
    """

    def __init__(self, config: AgentConfig, action_space: MultiDiscrete) -> None:
        self.vision = VisualPerception(out_channels=32)
        self.affector = LinearAffector(32 + 32, action_space)
        self.critic = LinearCritic(32 + 32)
        self.memory = TrajectoryBuffer(config.max_buffer_size)
        self.inverse_dynamics = InverseDynamics(32 + 32, action_space)
        self.forward_dynamics = ForwardDynamics(32 + 32, action_space.shape[0] + 2)
        self.ppo = PPO(self.affector, self.critic, config.ppo)
        self.icm = ICM(self.forward_dynamics, self.inverse_dynamics, config.icm)
        self.config = config

        # region of interest initialization
        self.roi_action: torch.Tensor | None = None
        self.prev_visual_features: torch.Tensor = torch.zeros(
            (1, 64), dtype=torch.float
        )

    def _transform_observation(self, obs: torch.Tensor) -> torch.Tensor:
        if self.roi_action is None:
            roi_obs = center_crop(obs, list(self.config.roi_shape))
        else:
            roi_obs = crop(
                obs,
                int(self.roi_action[:, 0]),
                int(self.roi_action[:, 1]),
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

        return action[:, :-2].long()
