import torch
import torch.nn as nn
import torch.optim as optim

from mvi.config import TDConfig


class TemporalDifferenceActorCritic:
    """One-step actor-critic learning algorithm"""

    def __init__(self, actor: nn.Module, critic: nn.Module, config: TDConfig):
        """
        Parameters
        ----------
        actor : nn.Module
            Neural network for the actor
        critic : nn.Module
            Neural network for the critic
        config : TDConfig
            Configuration for the Temporal Difference Actor Critic algorithm
        """
        # Actor & critic
        self.actor = actor
        self.critic = critic

        # Learning hyperpaameters
        self.discount_factor = config.discount_factor

    def _compute_actor_loss(
        self, action_logp: torch.Tensor, delta: torch.Tensor
    ) -> torch.Tensor:
        """Simple policy gradient loss"""
        loss = -delta * action_logp
        return loss

    def _compute_critic_loss(self, delta: torch.Tensor) -> torch.Tensor:
        """Squared error between the predicted value and estimated future return"""
        loss = 0.5 * delta**2
        return loss

    def _compute_delta(
        self, value: torch.Tensor, reward: float, next_features: torch.Tensor
    ) -> torch.Tensor:
        """Computes the TD error using a differentiable value prediction and estimated future return"""
        with torch.no_grad():
            next_value = self.critic(next_features)
        delta = reward + self.discount_factor * next_value - value
        return delta

    def loss(
        self,
        value: torch.Tensor,
        action_logp: torch.Tensor,
        reward: float,
        next_features: torch.Tensor,
    ) -> torch.Tensor:
        """Updates the actor and critic models given the a trajectory"""
        # TODO: Figure out if we need to compute discount ** time_step here
        delta = self._compute_delta(value, reward, next_features)
        actor_loss = self._compute_actor_loss(action_logp, delta)
        critic_loss = self._compute_critic_loss(delta)

        return actor_loss + critic_loss
