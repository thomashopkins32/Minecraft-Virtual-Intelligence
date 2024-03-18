from typing import Tuple
import logging

import torch

from mvi.utils import discount_cumsum, statistics


class PPOTrajectory:
    """
    Inspired by https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/ppo.py
    Used to store a single trajectory.
    """

    def __init__(
        self,
        max_buffer_size: int,
        discount_factor: float = 0.99,
        gae_discount_factor: float = 0.95,
    ):
        self.max_buffer_size = max_buffer_size
        self.discount_factor = discount_factor
        self.gae_discount_factor = gae_discount_factor
        self.observations: list[Tuple[torch.Tensor, torch.Tensor]] = []
        self.actions: list[torch.Tensor] = []
        self.rewards: list[float] = []
        self.values: list[float] = []
        self.log_probs: list[torch.Tensor] = []

    def store(
        self,
        observation: Tuple[torch.Tensor, torch.Tensor],
        action: torch.Tensor,
        reward: float,
        value: float,
        log_prob: torch.Tensor,
    ) -> None:
        """
        Append a single time-step to the trajectory.

        Parameters
        ----------
        observation : Tuple[torch.Tensor, torch.Tensor]
            Raw observation from the environment along with the region of interest.
        action : torch.Tensor
            Action tensor for the MineDojo environment + the region of interest (x,y) coordinates
        reward : float
            Raw reward value from the environment.
        value : float
            Value assigned to the observation by the agent.
        log_prob : torch.Tensor
            Probability of selecting each action.
        """
        if len(self.observations) == self.max_buffer_size:
            logging.warn(
                f"Cannot store additional time-steps in an already full trajectory. Current size: {len(self.observations)}. Max size: {self.max_buffer_size}"
            )
            return
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)

    def get(self, last_value: float) -> dict[str, torch.Tensor]:
        """
        Computes the advantages and reward-to-go then returns the data from the trajectory.

        Parameters
        ----------
        last_value : Int
            Value assigned to the last observation in the trajectory.
        """
        size = len(self.observations)
        if size < self.max_buffer_size:
            logging.warn(
                f"Computing information on a potentially unfinished trajectory. Current size: {size}. Max size: {self.max_buffer_size}"
            )
        # Separate the observations into separate tensors
        env_observations = torch.stack([obs[0] for obs in self.observations])
        roi_observations = torch.stack([obs[1] for obs in self.observations])

        self.rewards.append(last_value)
        self.values.append(last_value)
        rewards = torch.tensor(self.rewards)
        values = torch.tensor(self.values)

        deltas = rewards[:-1] + self.discount_factor * values[1:] - values[:-1]
        advantages = torch.tensor(
            discount_cumsum(
                deltas.numpy(), self.discount_factor * self.gae_discount_factor
            ).copy()
        )
        print(advantages)
        returns = torch.tensor(
            discount_cumsum(rewards.numpy(), self.discount_factor)[:-1].copy()
        ).squeeze()

        # Normalize advantages
        adv_mean, adv_std = statistics(advantages)
        advantages = (advantages - adv_mean) / adv_std
        print(advantages)

        return {
            "env_observations": env_observations,
            "roi_observations": roi_observations,
            "actions": torch.stack(self.actions),
            "returns": returns,
            "advantages": advantages,
            "log_probs": torch.stack(self.log_probs),
        }
