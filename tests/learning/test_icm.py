import pytest
from mvi.agent.agent import AgentV1
import torch
from mvi.learning.icm import ICM
from mvi.config import AgentConfig, PPOConfig, ICMConfig
from mvi.memory.trajectory import TrajectoryBuffer

from tests.helper import ACTION_SPACE


@pytest.fixture
def icm_module() -> ICM:
    agent = AgentV1(
        AgentConfig(
            ppo=PPOConfig(),
            icm=ICMConfig(
                train_forward_dynamics_iters=2, train_inverse_dynamics_iters=2
            ),
        ),
        ACTION_SPACE,
    )
    return agent.icm


def test_icm_update(icm_module: ICM) -> None:
    torch.autograd.anomaly_mode.set_detect_anomaly(True)

    buffer_size = 6
    trajectory = TrajectoryBuffer(max_buffer_size=buffer_size)
    for _ in range(buffer_size):
        trajectory.store(
            torch.zeros((64,), dtype=torch.float),
            torch.zeros((10,), dtype=torch.long),
            0.0,
            0.0,
            0.0,
            torch.ones((1,), dtype=torch.float),
        )
    icm_module.update(trajectory)
