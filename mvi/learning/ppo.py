from typing import Dict, Any, Tuple
from itertools import chain

import torch
import torch.optim as optim

from mvi.agent.agent import AgentV1
from mvi.utils import sample_multinomial, sample_guassian


class PPO:
    """Inspired by https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/ppo.py"""

    def __init__(
        self,
        agent: AgentV1,
        clip_ratio: float = 0.2,
        target_kl: float = 0.01,
        actor_lr: float = 3e-4,
        critic_lr: float = 1e-3,
        train_actor_iters: int = 80,
        train_critic_iters: int = 80,
    ):
        """
        Parameters
        ----------
        agent : AgentV1
            Minecraft agent model used as the actor and critic
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
        """
        # Environment & Agent
        self.agent = agent

        # Training duration
        self.train_actor_iters = train_actor_iters
        self.train_critic_iters = train_critic_iters

        # Learning hyperparameters
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl
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
        logp = self._joint_logp_action(action_dist, act)
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

    def _joint_logp_action(
        self,
        action_dists: Tuple[
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
        actions_taken: torch.Tensor,
    ) -> torch.Tensor:
        """
        Outputs the log probability of a sample as if the sample was taken
        from the distribution already.

        Parameters
        ----------
        action_dists : Tuple
            List of distributions to sample from
        actions_taken : torch.Tensor
            Samples produced already

        Returns
        -------
        torch.Tensor
            Log probabilities of sampling the corresponding action. To get the joint log-probability of the
            action, you can `.sum()` this tensor.
        """
        long_actions_taken = actions_taken.long()
        joint_logp = (  # longitudinal movement
            action_dists[0].gather(1, long_actions_taken[:, 0].unsqueeze(-1)).squeeze()
        )
        # Avoid += here as it is an in-place operation (which is bad for autograd)
        joint_logp = joint_logp + (  # lateral movement
            action_dists[1].gather(1, long_actions_taken[:, 1].unsqueeze(-1)).squeeze()
        )
        joint_logp = joint_logp + (  # vertical movement
            action_dists[2].gather(1, long_actions_taken[:, 2].unsqueeze(-1)).squeeze()
        )
        joint_logp = joint_logp + (  # pitch movement
            action_dists[3].gather(1, long_actions_taken[:, 3].unsqueeze(-1)).squeeze()
        )
        joint_logp = joint_logp + (  # yaw movement
            action_dists[4].gather(1, long_actions_taken[:, 4].unsqueeze(-1)).squeeze()
        )
        joint_logp = joint_logp + (  # functional actions
            action_dists[5].gather(1, long_actions_taken[:, 5].unsqueeze(-1)).squeeze()
        )
        joint_logp = joint_logp + (  # crafting actions
            action_dists[6].gather(1, long_actions_taken[:, 6].unsqueeze(-1)).squeeze()
        )
        joint_logp = joint_logp + (  # inventory actions
            action_dists[7].gather(1, long_actions_taken[:, 7].unsqueeze(-1)).squeeze()
        )
        # Focus actions
        x_roi_dist = torch.distributions.Normal(
            action_dists[8][:, 0], action_dists[9][:, 0]
        )
        joint_logp = joint_logp + x_roi_dist.log_prob(actions_taken[:, 8])
        y_roi_dist = torch.distributions.Normal(
            action_dists[8][:, 1], action_dists[9][:, 1]
        )
        joint_logp = joint_logp + y_roi_dist.log_prob(actions_taken[:, 9])

        return joint_logp

    def _sample_action(
        self,
        action_dists: Tuple[
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
        action_dists : Tuple
            List of distributions to sample from

        Returns
        -------
        torch.Tensor
            Action tensor representing the items sampled from the various distributions
        torch.Tensor
            Log probabilities of sampling the corresponding action. To get the joint log-probability of the
            action, you can `.sum()` this tensor.
        """
        # Initialize action and log buffer
        action = torch.zeros((10,), dtype=torch.int)
        logp_action = torch.zeros((10,), dtype=torch.float)

        action[0], logp_action[0] = sample_multinomial(action_dists[0][0])
        action[1], logp_action[1] = sample_multinomial(action_dists[1][0])
        action[2], logp_action[2] = sample_multinomial(action_dists[2][0])
        action[3], logp_action[3] = sample_multinomial(action_dists[3][0])
        action[4], logp_action[4] = sample_multinomial(action_dists[4][0])
        action[5], logp_action[5] = 0, 0  # sample_multinomial(action_dists[5][0])
        action[6], logp_action[6] = 0, 0  # sample_multinomial(action_dists[6][0])
        action[7], logp_action[7] = 0, 0  # sample_multinomial(action_dists[7][0])
        action[8], logp_action[8] = sample_guassian(
            action_dists[8][0, 0], action_dists[9][0, 0]
        )
        action[9], logp_action[9] = sample_guassian(
            action_dists[8][0, 1], action_dists[9][0, 1]
        )

        return action, logp_action

    def _save_models(self):
        """Saves model checkpoints to the current working directory"""
        # TODO 
        raise NotImplementedError

    def update(self, data: Dict[str, Any]) -> None:
        """Updates the actor and critic models given the a dataset of trajectories"""
        self._update_actor(data)
        self._update_critic(data)
        self._save_models()
