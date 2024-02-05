from typing import Dict, Any, Str, Float, UInt

import torch
import torch.optim as optim
import gymnasium

from MineAI.memory.trajectory import PPOTrajectory


class PPO:
    """Inspired by https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/ppo.py"""

    def __init__(
        self,
        env: gymnasium.Env,
        actor: torch.nn.Module,
        critic: torch.nn.Module,
        epochs: UInt = 50,
        steps_per_epoch: UInt = 4000,
        discount_factor: Float = 0.99,
        gae_discount_factor: Float = 0.97,
        clip_ratio: Float = 0.2,
        target_kl: Float = 0.01,
        actor_lr: Float = 3e-4,
        critic_lr: Float = 1e-3,
        train_actor_iters: UInt = 80,
        train_critic_iters: UInt = 80,
        save_freq: UInt = 10,
    ):
        """
        Parameters
        ----------
        env : gymnasium.Env
            Environment for the agent to interact with; already initialized
        actor : torch.nn.Module
            Neural network to use as the policy; already initialized
        critic : torch.nn.Module
            Neural network to use as the value function; already initialized
        epochs : int, optional
            Number of policy updates to perform after sampling experience
        steps_per_epoch : int, optional
            Number of steps of interaction with the environment per epoch
        discount_factor : float, optional
            Used to weight preference for long-term reward (aka gamma)
        gae_discount_factor : float, optional
            Used to weight preference for long-term advantage (aka lambda)
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
        self.env = env
        self.actor = actor
        self.critic = critic

        # Training duration
        self.epochs = (epochs,)
        self.steps_per_epoch = steps_per_epoch
        self.train_actor_iters = train_actor_iters
        self.train_critic_iters = train_critic_iters
        self.save_freq = save_freq

        # Learning hyperparameters
        self.discount_factor = discount_factor
        self.gae_discount_factor = gae_discount_factor
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

    def _compute_actor_loss(self, data: Dict[Str, Any]) -> Float:
        obs, act, adv, logp_old = (
            data["observations"],
            data["actions"],
            data["advantages"],
            data["log_probs"],
        )

        _, logp = self.actor(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
        loss = -(torch.min(ratio * adv, clip_adv)).mean()

        kl = (logp_old - logp).mean()

        return loss, kl

    def _update_actor(self, data: Dict[Str, Any], optimizer: optim.Optimizer) -> None:
        self.actor.train()
        for _ in range(self.train_actor_iters):
            optimizer.zero_grad()
            loss, kl = self._compute_actor_loss(data)
            if kl > 1.5 * self.target_kl:
                # early stopping
                break
            loss.backward()
            optimizer.step()
        self.actor.eval()

    def _compute_critic_loss(self, data: Dict[Str, Any]) -> Float:
        obs, ret = data["observations"], data["returns"]
        return ((self.critic(obs) - ret) ** 2).mean()

    def _update_critic(self, data: Dict[Str, Any], optimizer: optim.Optimizer) -> None:
        self.critic.train()
        for _ in range(self.train_critic_iters):
            optimizer.zero_grad()
            loss = self._compute_critic_loss(data)
            loss.backward()
            optimizer.step()
        self.critic.eval()

    def run(self):
        """Runs the proximal policy optimization algorithm"""

        actor_optim = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        critic_optim = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        for e in range(self.epochs):
            trajectory_buffer = PPOTrajectory(
                max_buffer_size=self.steps_per_epoch,
                discount_factor=self.dicsount_factor,
                gae_discount_factor=self.gae_discount_factor,
            )
            obs = self.env.reset().as_tensor(dtype=torch.float)
            t_return = 0.0
            for t in range(self.steps_per_epoch):
                a, logp = self.actor(obs)
                v = self.critic(obs)

                next_obs, reward, _, info = self.env.step(a)
                t_return += reward

                trajectory_buffer.store(obs, a, reward, v, logp)
                obs = next_obs.as_tensor(dtype=torch.float)
            data = trajectory_buffer.get()
            self._update_actor(data, actor_optim)
            self._update_critic(data, critic_optim)
