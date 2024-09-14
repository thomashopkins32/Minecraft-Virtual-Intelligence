import pytest
import torch

from mvi.memory.trajectory import TrajectoryBuffer


MAX_BUFFER_SIZE = 10


@pytest.fixture()
def trajectory():
    return TrajectoryBuffer(max_buffer_size=MAX_BUFFER_SIZE)


def test_trajectory(trajectory: TrajectoryBuffer):
    obs = torch.randn((64,), dtype=torch.float)
    action = torch.zeros((10,))
    reward = 1.0
    value = 10.0
    log_prob = torch.zeros((10,))
    for _ in range(MAX_BUFFER_SIZE):
        trajectory.store(obs, action, reward, value, log_prob)
