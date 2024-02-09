from typing import Dict, Any, Tuple

import torch
import torch.optim as optim
from torchvision.transforms.functional import center_crop, crop  # type: ignore
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

    def _compute_actor_loss(
        self, data: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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

        actor_optim = optim.Adam(
            self.agent.vision.parameters() + self.agent.affector.parameters(),
            lr=self.actor_lr,
        )
        critic_optim = optim.Adam(
            self.agent.vision.parameters() + self.agent.reasoner.parameters(),
            lr=self.critic_lr,
        )

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
                action, logp_action = sample_action(a)
                env_action = action[:-2]  # Don't include the region of interest
                roi_action = action[-2:]
                next_obs, reward, _, _ = self.env.step(env_action)
                t_return += reward

                trajectory_buffer.store((obs, roi_obs), action, reward, v, logp_action)
                obs = next_obs.as_tensor(dtype=torch.float)
                roi_obs = crop(
                    next_obs,
                    roi_action[0],
                    roi_action[1],
                    self.roi_shape[0],
                    self.roi_shape[1],
                )
            _, last_v = self.agent(obs, roi_obs)
            data = trajectory_buffer.get(last_v)
            self._update_actor(data, actor_optim)
            self._update_critic(data, critic_optim)

    def sample_action(
        self,
        x: Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Samples actions from the various distributions and combines them into an action tensor.
        Outputs the action tensor and a logp tensor showing the log probability of taking that action.

        Parameters
        ----------
        x : Tuple
            List of distributions to sample from

        Returns
        -------
        torch.Tensor
            Action tensor representing the items sampled from the various distributions
        torch.Tensor
            Log probabilities of sampling the corresponding action
        """
        action = torch.zeros((10,))
        long_action = torch.multinomial(x[0], num_samples=1)
        long_logp = x[0][long_action].log()
        lat_action = torch.multinomial(x[1], num_samples=1)
        lat_logp = x[1][lat_action].log()
        vert_action = torch.multinomial(x[2], num_samples=1)
        vert_logp = x[2][vert_action].log()
        pitch_action = torch.multinomial(x[3], num_samples=1)
        pitch_logp = x[3][pitch_action].log()
        yaw_action = torch.multinomial(x[4], num_samples=1)
        yaw_logp = x[4][yaw_action].log()
        func_action = torch.multinomial(x[5], num_samples=1)
        func_logp = x[5][func_action].log()
        craft_action = torch.multinomial(x[6], num_samples=1)
        craft_logp = x[6][craft_action].log()
        inventory_action = torch.multinomial(x[7], num_samples=1)
        inventory_logp = x[7][inventory_action].log()

        # TODO: Sample ROI action from mean, std stored in x[8], x[9]
        raise NotImplementedError()