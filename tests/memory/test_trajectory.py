import pytest
import numpy as np
import torch

from MineAI.memory.trajectory import PPOTrajectory


@pytest.fixture()
def ppo_trajectory():
    return PPOTrajectory(10, discount_factor=0.5, gae_discount_factor=0.5)


def test_ppo_trajectory(ppo_trajectory):
    obs = (torch.randn((1, 3, 32, 32), dtype=torch.float), torch.randn((1, 3, 5, 5), dtype=torch.float))
    action = torch.zeros((10,))
    reward = 1.0
    value = 10.0
    log_prob = 0.1
    for _ in range(10):
        ppo_trajectory.store(obs, action, reward, value, log_prob)
    
    result = ppo_trajectory.get(last_value=2.5)
    env_obs = result["env_observations"]
    roi_obs = result["roi_observations"]
    actions = result["actions"]
    returns = result["returns"]
    advantages = result["advantages"]
    log_probs = result["log_probs"]

    assert type(env_obs) == torch.Tensor
    assert type(roi_obs) == torch.Tensor
    assert type(actions) == torch.Tensor
    assert type(returns) == torch.Tensor
    assert type(advantages) == torch.Tensor
    assert type(log_probs) == torch.Tensor

    assert env_obs.shape == (10, 3, 32, 32)
    assert roi_obs.shape == (10, 3, 5, 5)
    assert actions.shape == (10, 10)
    assert returns.shape == (10, 1)
    assert advantages.shape == (10, 1)
    assert log_probs.shape == (10, 1)

    expected_advantages = torch.tensor([
        -5.3333282470703125, -5.333312988125, -5.333251953125, -5.3330078125, -5.328125, -5.3125, -5.25, -5.0, -4.0
    ])
    assert np.array_equal(advantages, expected_advantages)





