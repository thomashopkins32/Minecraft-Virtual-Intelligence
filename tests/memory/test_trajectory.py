import pytest
import numpy as np
import torch

from mvi.memory.trajectory import TrajectoryBuffer


@pytest.fixture()
def trajectory_buffer() -> TrajectoryBuffer:
    return TrajectoryBuffer(10, discount_factor=0.5, gae_discount_factor=0.5)


def test_ppo_trajectory(trajectory_buffer: TrajectoryBuffer):
    obs = (
        torch.randn((3, 32, 32), dtype=torch.float),
        torch.randn((3, 5, 5), dtype=torch.float),
    )
    action = torch.zeros((10,))
    reward = 1.0
    value = 10.0
    log_prob = torch.zeros((10,))
    for _ in range(10):
        trajectory_buffer.store(obs, action, reward, value, log_prob)

    trajectory_buffer.finalize_trajectory(last_value=2.5)

    result = list(trajectory_buffer.get(shuffle=False, batch_size=None))[0]
    env_obs = result.env_observations
    roi_obs = result.roi_observations
    actions = result.actions
    returns = result.returns
    advantages = result.advantages
    log_probs = result.log_probabilities

    assert type(env_obs) == torch.Tensor
    assert type(roi_obs) == torch.Tensor
    assert type(actions) == torch.Tensor
    assert type(returns) == torch.Tensor
    assert type(advantages) == torch.Tensor
    assert type(log_probs) == torch.Tensor

    assert env_obs.shape == (10, 3, 32, 32)
    assert roi_obs.shape == (10, 3, 5, 5)
    assert actions.shape == (10, 10)
    assert returns.shape == (10,)
    assert advantages.shape == (10,)
    assert log_probs.shape == (10, 10)

    expected_advantages = torch.tensor(
        [
            0.4243,
            0.4242,
            0.4241,
            0.4235,
            0.4212,
            0.4118,
            0.3745,
            0.2254,
            -0.3712,
            -2.7577,
        ]
    )
    assert np.allclose(advantages.numpy(), expected_advantages.numpy(), rtol=1e-3)
