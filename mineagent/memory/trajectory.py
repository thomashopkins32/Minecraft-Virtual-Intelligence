from collections import deque
from typing import Any

import torch


def _squeeze_leading_batch(t: torch.Tensor) -> torch.Tensor:
    """Drop a leading singleton batch dim so PPO/ICM stack to ``(T, ...)``.

    ``AgentV1.act`` runs with batch size 1, so tensors arrive as ``(1, D)``.
    Learning code (and the unit tests) expect per-step vectors of rank 1.
    """
    t = t.detach()
    if t.dim() >= 1 and t.shape[0] == 1:
        t = t.squeeze(0)
    return t


def _as_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().reshape(-1)[0].item())
    return float(value)


class TrajectoryBuffer:
    """
    Used to store a single trajectory.

    A single item in a trajectory contains the following information:
    - The visual features computed from the observation of the environment
    - The action taken given the features computed
    - The reward given by the environment for the PREVIOUS action
    - The value assigned to the features computed
    - The log probability of sampling the action taken

    Note: a single item is not complete enough information to learn from.
    At least two consecutive items in the trajectory are necessary for learning.
    This is because the next observation and reward are not available until the next step is stored.
    """

    def __init__(self, max_buffer_size: int):
        self.max_buffer_size = max_buffer_size

        self.features_buffer: deque[torch.Tensor] = deque([], maxlen=max_buffer_size)
        self.actions_buffer: deque[torch.Tensor] = deque([], maxlen=max_buffer_size)
        self.rewards_buffer: deque[float] = deque([], maxlen=max_buffer_size)
        self.intrinsic_rewards_buffer: deque[float] = deque([], maxlen=max_buffer_size)
        self.values_buffer: deque[float] = deque([], maxlen=max_buffer_size)
        self.log_probs_buffer: deque[torch.Tensor] = deque([], maxlen=max_buffer_size)
        self.focus_buffer: deque[torch.Tensor] = deque([], maxlen=max_buffer_size)
        self.focus_logp_buffer: deque[torch.Tensor] = deque([], maxlen=max_buffer_size)

    def __len__(self):
        return len(self.features_buffer)

    def store(
        self,
        visual_features: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        intrinsic_reward: float,
        value: float,
        log_prob: torch.Tensor,
        focus: torch.Tensor | None = None,
        focus_logp: torch.Tensor | None = None,
    ) -> None:
        """
        Append a single time-step to the trajectory.

        Parameters
        ----------
        visual_features : torch.Tensor
            Features computed by visual perception from the observation of the environment
        action : torch.Tensor
            Environment action tensor (keys, mouse, buttons, scroll -- no focus)
        reward : float
            Reward value from the environment for the previous action
        intrinsic_reward : float
            Reward value from the Intrinsic Curiosity Module (ICM)
        value : float
            Value assigned to the observation by the agent
        log_prob : torch.Tensor
            Log probability of selecting each environment sub-action
        focus : torch.Tensor | None
            Focus/ROI coordinates (2-dim), stored separately from env action
        focus_logp : torch.Tensor | None
            Log probability of the focus coordinates
        """
        self.features_buffer.append(_squeeze_leading_batch(visual_features))
        self.actions_buffer.append(_squeeze_leading_batch(action))
        self.rewards_buffer.append(_as_float(reward))
        self.intrinsic_rewards_buffer.append(_as_float(intrinsic_reward))
        self.values_buffer.append(_as_float(value))
        self.log_probs_buffer.append(_squeeze_leading_batch(log_prob))
        if focus is not None:
            self.focus_buffer.append(_squeeze_leading_batch(focus))
        if focus_logp is not None:
            self.focus_logp_buffer.append(_squeeze_leading_batch(focus_logp))
