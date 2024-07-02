import pytest
import torch
from mvi.agent.agent import AgentV1
from mvi.learning.ppo import PPO
from mvi.config import PPOConfig
from mvi.memory.trajectory import PPOTrajectory

from tests.helper import MockEnv


@pytest.fixture
def ppo_module():
    env = MockEnv()
    agent = AgentV1(env.action_space)
    return PPO(agent, PPOConfig(train_actor_iters=2, train_critic_iters=2))


def test_ppo_update(ppo_module: PPO) -> None:
    torch.autograd.anomaly_mode.set_detect_anomaly(True)

    buffer_size = 3
    trajectory = PPOTrajectory(max_buffer_size=buffer_size)
    for _ in range(buffer_size):
        trajectory.store(
            (
                torch.zeros((3, 160, 256), dtype=torch.float),
                torch.zeros((3, 20, 20), dtype=torch.float),
            ),
            torch.zeros((10,), dtype=torch.long),
            0.0,
            0.0,
            torch.ones((1,), dtype=torch.float),
        )
    trajectory.finalize_trajectory(0.0)
    ppo_module.update(trajectory)
