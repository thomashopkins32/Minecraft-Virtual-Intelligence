import pytest
import torch

from mvi.agent.agent import AgentV1
from mvi.config import AgentConfig, PPOConfig
from tests.perception.test_visual import VISUAL_EXPECTED_PARAMS
from tests.affector.test_affector import LINEAR_AFFECTOR_EXPECTED_PARAMS
from tests.reasoning.test_critic import LINEAR_REASONER_EXPECTED_PARAMS
from tests.helper import ACTION_SPACE


# Visual perception + 2 action linear layers
AGENT_V1_EXPECTED_PARAMS = (
    VISUAL_EXPECTED_PARAMS
    + LINEAR_AFFECTOR_EXPECTED_PARAMS
    + LINEAR_REASONER_EXPECTED_PARAMS
)


@pytest.fixture
def agent_v1_module():
    return AgentV1(AgentConfig(ppo=PPOConfig()), ACTION_SPACE)


def test_agent_v1_act(agent_v1_module: AgentV1):
    input_tensor = torch.randn((1, 3, 160, 256))

    action = agent_v1_module.act(input_tensor)

    assert len(action) == 8


def test_agent_v1_params(agent_v1_module: AgentV1):
    modules = [
        agent_v1_module.vision,
        agent_v1_module.affector,
        agent_v1_module.reasoner,
    ]
    num_params = sum(sum(p.numel() for p in m.parameters()) for m in modules)
    assert num_params == AGENT_V1_EXPECTED_PARAMS
