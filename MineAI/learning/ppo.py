from typing import Dict, Any, Tuple

import torch
import torch.optim as optim
from torchvision.transforms.functional import center_crop, crop # type: ignore
import gymnasium

from MineAI.memory.trajectory import PPOTrajectory
from MineAI.agent.agent import AgentV1


class PPO:
    """Inspired by https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/ppo.py"""

    def __init__(
        self,
        env: gymnasium.Env,
        agent: AgentV1,
        roi_shape: Tuple[int, int] = (32, 32),
        epochs: int = 50,
        steps_per_epoch: int = 4000,
        discount_factor: float = 0.99,
        gae_discount_factor: float = 0.97,
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
        env : gymnasium.Env
            Environment for the agent to interact with; already initialized
        agent : AgentV1
            Neural network to use as the policy and value function; already initialized
        roi_shape : Tuple[int, int], optional
            Shape of the region of interest the agent has and will operate on
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
        self.agent = agent
        self.roi_shape = roi_shape

        # Training duration
        self.epochs = epochs
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

    def _compute_actor_loss(self, data: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        obs, act, adv, logp_old = (
            data["observations"],
            data["actions"],
            data["advantages"],
            data["log_probs"],
        )

        (_, logp), _ = self.agent(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
        loss = -(torch.min(ratio * adv, clip_adv)).mean()

        kl = (logp_old - logp).mean()

        return loss, kl

    def _update_actor(self, data: Dict[str, Any], optimizer: optim.Optimizer) -> None:
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

    def _compute_critic_loss(self, data: Dict[str, Any]) -> torch.Tensor:
        obs, ret = data["observations"], data["returns"]
        _, v = self.agent(obs[0], obs[1])
        return ((v - ret) ** 2).mean()

    def _update_critic(self, data: Dict[str, Any], optimizer: optim.Optimizer) -> None:
        self.agent.train()
        for _ in range(self.train_critic_iters):
            optimizer.zero_grad()
            loss = self._compute_critic_loss(data)
            loss.backward()
            optimizer.step()
        self.agent.eval()

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
            roi_obs = center_crop(obs, self.roi_shape)
            t_return = 0.0
            for t in range(self.steps_per_epoch):
                a, v = self.agent(obs, roi_obs)
                env_action = a[1]
                logp_env_action = a[2]
                roi_action = torch.round(a[3])
                next_obs, reward, _, _ = self.env.step(env_action)
                t_return += reward

                trajectory_buffer.store((obs, roi_obs), (env_action, roi_action), reward, v, logp_env_action)
                obs = next_obs.as_tensor(dtype=torch.float)
                roi_obs = crop(next_obs, roi_action[0], roi_action[1], self.roi_shape[0], self.roi_shape[1])
            data = trajectory_buffer.get()
            self._update_actor(data, actor_optim)
            self._update_critic(data, critic_optim)
