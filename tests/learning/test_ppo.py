import pytest
import torch
from mvi.agent.agent import AgentV1
from mvi.learning.ppo import PPO
from mvi.config import PPOConfig

from tests.helper import MockEnv


@pytest.fixture
def ppo_module():
    env = MockEnv()
    agent = AgentV1(env.action_space)
    return PPO(
        agent,
        PPOConfig(train_actor_iters=2, train_critic_iters=2)
    )


def test_ppo_update(ppo_module: PPO) -> None:
    torch.autograd.anomaly_mode.set_detect_anomaly(True)

    bs = 3
    data = {
        "env_observations": torch.zeros((bs, 3, 160, 256), dtype=torch.float),
        "roi_observations": torch.zeros((bs, 3, 20, 20), dtype=torch.float),
        "actions": torch.zeros((bs, 10), dtype=torch.long),
        "returns": torch.zeros((bs, 1), dtype=torch.float),
        "advantages": torch.zeros((bs, 1), dtype=torch.float),
        "log_probs": torch.zeros((bs, 1), dtype=torch.float),
    }

    ppo_module.update(data)
