import torch

from mvi.memory.trajectory import TrajectoryBuffer, TrajectorySample


class ICM:
    """

    """
    def __init__(self):
        raise NotImplementedError()

    def intrinsic_reward(self, embedding: torch.Tensor, action: torch.Tensor, next_embedding: torch.Tensor) -> float:
        """
        Computes the intrinsic reward of the agent. This is used as a proxy for curiosity based on how well
        the agent is able to predict the `next_embedding` given the `embedding` and `action` vectors.
        The idea is that if the prediction error is low, then we have probably seen similar observations many times before
        and we should explore elsewhere.

        Parameters
        ----------
        embedding : torch.Tensor
            Features extracted from the current observation
            Expected shape (1, self.embed_dim)
        action : torch.Tensor
            Action taken in the environment to go from the current observation -> next observation
            Expected shape (1, self.action_dim)
        next_embedding : torch.Tensor
            Features extracted from the next observation
            Expected shape (1, self.embed_dim)

        Returns
        -------
        float
            Weighted mean-squared error between the prediction from the forward dynamics model
            and the `next_embedding`
        """
        raise NotImplementedError()

    def _update_inverse_dynamics(self, data: TrajectoryBuffer) -> None:
        raise NotImplementedError()

    def _update_forward_dynamics(self, data: TrajectoryBuffer) -> None:
        raise NotImplementedError()

    def update(self, data: TrajectoryBuffer) -> None:
        raise NotImplementedError()