from typing import Dict, Any, Tuple
from itertools import chain

import torch
import torch.nn as nn
import torch.optim as optim

from mvi.agent.agent import AgentV1
from mvi.utils import joint_logp_action
from mvi.config import PPOConfig
from mvi.memory.trajectory import PPOTrajectory, PPOSample


class PPO:
    """Inspired by https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/ppo.py"""

    def __init__(self, actor: nn.Module, critic: nn.Module, config: PPOConfig):
        """
        Parameters
        ----------
        actor : nn.Module
            Neural network for the actor
        critic : nn.Module
            Neural network for the critic
        config : PPOConfig
            Configuration for the PPO algorithm
        """
        # Actor & critic
        self.actor = actor
        self.critic = critic

        # Training duration
        self.train_actor_iters = config.train_actor_iters
        self.train_critic_iters = config.train_critic_iters

        # Learning hyperparameters
        self.clip_ratio = config.clip_ratio
        self.target_kl = config.target_kl
        self.actor_optim = optim.Adam(
            chain(self.agent.vision.parameters(), self.agent.affector.parameters()),
            lr=config.actor_lr,
        )
        self.critic_optim = optim.Adam(
            chain(self.agent.vision.parameters(), self.agent.reasoner.parameters()),
            lr=config.critic_lr,
        )

    def _compute_actor_loss(self, data: PPOSample) -> Tuple[torch.Tensor, torch.Tensor]:
        env_obs, roi_obs, act, adv, logp_old = (
            data.env_observations,
            data.roi_observations,
            data.actions,
            data.advantages,
            data.log_probabilities,
        )

        action_dist, _ = self.agent(env_obs, roi_obs)
        logp = joint_logp_action(action_dist, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
        loss = -(torch.min(ratio * adv, clip_adv)).mean()

        kl = (logp_old - logp).mean()

        return loss, kl

    def _update_actor(self, data: PPOTrajectory) -> None:
        self.agent.train()
        buffer_size = len(data)
        for sample in data.get(
            shuffle=True, batch_size=buffer_size // self.train_actor_iters
        ):
            self.actor_optim.zero_grad()
            loss, kl = self._compute_actor_loss(sample)
            if kl > 1.5 * self.target_kl:
                # early stopping
                break
            loss.backward()
            self.actor_optim.step()
        self.agent.eval()

    def _compute_critic_loss(self, data: PPOSample) -> torch.Tensor:
        env_obs, roi_obs, ret = (
            data.env_observations,
            data.roi_observations,
            data.returns,
        )
        _, v = self.agent(env_obs, roi_obs)
        return ((v - ret) ** 2).mean()

    def _update_critic(self, data: PPOTrajectory) -> None:
        self.agent.train()
        buffer_size = len(data)
        for sample in data.get(
            shuffle=True, batch_size=buffer_size // self.train_critic_iters
        ):
            self.critic_optim.zero_grad()
            loss = self._compute_critic_loss(sample)
            loss.backward()
            self.critic_optim.step()
        self.agent.eval()

    def update(self, data: PPOTrajectory) -> None:
        """Updates the actor and critic models given the a dataset of trajectories"""
        self._update_actor(data)
        self._update_critic(data)
