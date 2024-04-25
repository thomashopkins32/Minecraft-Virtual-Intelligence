import pytest
import torch
from mvi.agent.agent import AgentV1
from mvi.learning.ppo import PPO

from tests.helper import MockEnv


@pytest.fixture
def ppo_module():
    env = MockEnv()
    agent = AgentV1(env.action_space)
    return PPO(
        env,
        agent,
        epochs=2,
        steps_per_epoch=3,
        train_actor_iters=2,
        train_critic_iters=2,
        save_freq=100,
    )


def test_ppo_run(ppo_module):
    torch.autograd.anomaly_mode.set_detect_anomaly(True)
    ppo_module.run()
