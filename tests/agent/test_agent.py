import pytest
import torch

from MineAI.agent.agent import AgentV1
from tests.perception.test_visual import VISUAL_EXPECTED_PARAMS


# Visual perception + 2 action linear layers
AGENT_V1_EXPECTED_PARAMS = VISUAL_EXPECTED_PARAMS + 2 * (64 * 25 + 25)


@pytest.fixture
def agent_v1_module():
    return AgentV1()


def test_agent_v1_forward(agent_v1_module):
    input_tensor = torch.randn((32, 3, 160, 256))

    out1, out2 = agent_v1_module(input_tensor)

    assert out1.shape[1] == 25
    assert out2.shape[1] == 25


def test_agent_v1_params(agent_v1_module):
    num_params = sum(p.numel() for p in agent_v1_module.parameters())
    assert num_params == AGENT_V1_EXPECTED_PARAMS