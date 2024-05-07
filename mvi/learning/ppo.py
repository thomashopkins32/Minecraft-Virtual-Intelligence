from typing import Dict, Any, Tuple
from itertools import chain

import torch
import torch.optim as optim

from mvi.agent.agent import AgentV1
from mvi.utils import joint_logp_action


class PPO:
    """Inspired by https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/ppo.py"""

    def __init__(
        self,
        agent: AgentV1, # TODO: decide if this should be the full agent or split into actor & critic
        clip_ratio: float = 0.2,
        target_kl: float = 0.01,
        actor_lr: float = 3e-4,
        critic_lr: float = 1e-3,
        train_actor_iters: int = 80,
        train_critic_iters: int = 80,
        save_freq: int = 10,
    ):
        """
        Parameters
        ----------
        agent : AgentV1
            Agent that contains the actor and critic modules
        clip_ratio : float, optional
            Maximum allowed divergence of the new policy from the old policy in the objective function (aka epsilon)
        target_kl : float, optional
            Target KL divergence for policy updates; used in model selection (early stopping)
        actor_lr : float, optional
            Learning rate for the actor module
        critic_lr : float, optional
            Learning rate for the critic module
        train_actor_iters : int, optional
            Number of iterations to train the actor per epoch
        train_critic_iters : int, optional
            Number of iterations to train the critic per epoch
        save_freq : int, optional
            Rate in terms of number of epochs that the actor and critic models are saved to disk
        """
        # Environment & Agent
        self.agent = agent

        # Training duration
        self.train_actor_iters = train_actor_iters
        self.train_critic_iters = train_critic_iters

        # Learning hyperparameters
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl

        # Optimizers
        self.actor_optim = optim.Adam(
            chain(self.agent.vision.parameters(), self.agent.affector.parameters()),
            lr=actor_lr,
        )
        self.critic_optim = optim.Adam(
            chain(self.agent.vision.parameters(), self.agent.reasoner.parameters()),
            lr=critic_lr,
        )

    def _compute_actor_loss(
        self, data: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        env_obs, roi_obs, act, adv, logp_old = (
            data["env_observations"],
            data["roi_observations"],
            data["actions"],
            data["advantages"],
            data["log_probs"],
        )

        action_dist, _ = self.agent(env_obs, roi_obs)
        logp = joint_logp_action(action_dist, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
        loss = -(torch.min(ratio * adv, clip_adv)).mean()

        kl = (logp_old - logp).mean()

        return loss, kl

    def _update_actor(self, data: Dict[str, Any]) -> None:
        self.agent.train()
        for _ in range(self.train_actor_iters):
            self.actor_optim.zero_grad()
            loss, kl = self._compute_actor_loss(data)
            if kl > 1.5 * self.target_kl:
                # early stopping
                break
            loss.backward()
            self.actor_optim.step()
        self.agent.eval()

    def _compute_critic_loss(self, data: Dict[str, Any]) -> torch.Tensor:
        env_obs, roi_obs, ret = (
            data["env_observations"],
            data["roi_observations"],
            data["returns"],
        )
        _, v = self.agent(env_obs, roi_obs)
        return ((v - ret) ** 2).mean()

    def _update_critic(self, data: Dict[str, Any]) -> None:
        self.agent.train()
        for _ in range(self.train_critic_iters):
            self.critic_optim.zero_grad()
            loss = self._compute_critic_loss(data)
            loss.backward()
            self.critic_optim.step()
        self.agent.eval()

    def train(self, data: Dict[str, Any]) -> None:
        self._update_actor(data)
        self._update_critic(data)
