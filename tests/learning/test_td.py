import pytest
import torch

from mvi.agent.agent import AgentV1
from mvi.learning.td import TemporalDifferenceActorCritic
from mvi.config import AgentConfig, PPOConfig, ICMConfig, TDConfig

from tests.helper import ACTION_SPACE

@pytest.fixture
def td_module() -> TemporalDifferenceActorCritic:
    agent = AgentV1(
        AgentConfig(
            ppo=PPOConfig(train_actor_iters=2, train_critic_iters=2),
            icm=ICMConfig(),
            td=TDConfig(),
        ),
        ACTION_SPACE,
    )
    return TemporalDifferenceActorCritic(agent.affector, agent.critic, agent.config.td)


def test_loss(td_module: TemporalDifferenceActorCritic) -> None:
    torch.autograd.anomaly_mode.set_detect_anomaly(True)

    current_state_value = torch.tensor([998.0,], dtype=torch.float)
    next_state_features = torch.zeros((64,), dtype=torch.float)
    logp_action = torch.tensor([0.99,], dtype=torch.float)
    reward = 1000.0
    time_step = 10

    loss = td_module.loss(current_state_value, logp_action, reward, next_state_features, time_step)

    # TODO: Calculate what the loss should be and assert its accurate