from dataclasses import dataclass
from typing import Tuple, Union, Generator
import logging

import numpy as np
import torch

from mvi.utils import discount_cumsum, statistics


@dataclass
class TrajectorySample:
    env_observations: torch.Tensor
    roi_observations: torch.Tensor
    actions: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    log_probabilities: torch.Tensor


class TrajectoryBuffer:
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
        self.observations_buffer: list[Tuple[torch.Tensor, torch.Tensor]] = []
        self.actions_buffer: list[torch.Tensor] = []
        self.rewards_buffer: list[float] = []
        self.values_buffer: list[float] = []
        self.log_probs_buffer: list[torch.Tensor] = []

    def __len__(self):
        return len(self.observations_buffer)

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
        if len(self.observations_buffer) == self.max_buffer_size:
            logging.warn(
                f"Cannot store additional time-steps in an already full trajectory. Current size: {len(self.observations_buffer)}. Max size: {self.max_buffer_size}"
            )
            return
        self.observations_buffer.append(observation)
        self.actions_buffer.append(action)
        self.rewards_buffer.append(reward)
        self.values_buffer.append(value)
        self.log_probs_buffer.append(log_prob)

    def finalize_trajectory(self, last_value: float) -> None:
        """
        Computes the advantages and reward-to-go then returns the data from the trajectory.

        Parameters
        ----------
        last_value : Int
            Value assigned to the last observation in the trajectory.

        """
        size = len(self.observations_buffer)
        if size < self.max_buffer_size:
            logging.warn(
                f"Computing information on a potentially unfinished trajectory. Current size: {size}. Max size: {self.max_buffer_size}"
            )
        # Separate the observations into separate tensors
        self.env_observations = torch.stack(
            [obs[0] for obs in self.observations_buffer]
        )
        self.roi_observations = torch.stack(
            [obs[1] for obs in self.observations_buffer]
        )

        self.rewards_buffer.append(last_value)
        self.values_buffer.append(last_value)
        self.rewards = torch.tensor(self.rewards_buffer)
        self.values = torch.tensor(self.values_buffer)

        deltas = (
            self.rewards[:-1]
            + self.discount_factor * self.values[1:]
            - self.values[:-1]
        )
        advantages = torch.tensor(
            discount_cumsum(
                deltas.numpy(), self.discount_factor * self.gae_discount_factor
            ).copy()
        )
        self.returns = torch.tensor(
            discount_cumsum(self.rewards.numpy(), self.discount_factor)[:-1].copy()
        ).squeeze()

        # Normalize advantages TODO: We may have to do this by the batch instead of overall
        adv_mean, adv_std = statistics(advantages)
        self.norm_advantages = (advantages - adv_mean) / adv_std

        self.actions = torch.stack(self.actions_buffer)
        self.log_probabilities = torch.stack(self.log_probs_buffer)

    def _get_sample(self, indices: np.array) -> TrajectorySample:
        return TrajectorySample(
            env_observations=self.env_observations[indices],
            roi_observations=self.roi_observations[indices],
            actions=self.actions[indices],
            returns=self.returns[indices],
            advantages=self.norm_advantages[indices],
            log_probabilities=self.log_probabilities[indices],
        )

    def get(
        self, shuffle: bool = False, batch_size: Union[int, None] = None
    ) -> Generator[TrajectorySample, None, None]:
        """
        Computes the advantages and reward-to-go then returns the data from the trajectory.

        Parameters
        ----------
        shuffle : bool, optional
            Randomize the order of the trajectory buffer
        batch_size : int, optional
            Yield the data in batches instead of all at once

        Yields
        ------
        PPOSample
            Subset of information about the trajectory, possibly shuffled
        """
        size = len(self.observations_buffer)
        if size < self.max_buffer_size:
            logging.warn(
                f"Retrieving information on a potentially unfinished trajectory. Current size: {size}. Max size: {self.max_buffer_size}"
            )

        if shuffle:
            indices = np.random.permutation(size)
        else:
            indices = np.arange(size)

        if batch_size is None:
            batch_size = size

        start_idx = 0
        while start_idx < size:
            yield self._get_sample(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size
