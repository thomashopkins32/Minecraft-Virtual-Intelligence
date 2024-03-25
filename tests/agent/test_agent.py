import pytest
import torch
from torchvision.transforms.functional import center_crop  # type: ignore

from mvi.agent.agent import AgentV1
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
    return AgentV1(ACTION_SPACE)


def test_agent_v1_forward(agent_v1_module):
    input_tensor = torch.randn((32, 3, 160, 256))
    roi_tensor = center_crop(input_tensor, output_size=(32, 32))

    action_dist, value = agent_v1_module(input_tensor, roi_tensor)

    assert len(action_dist) == 10
    assert value.shape == (32, 1)


def test_agent_v1_params(agent_v1_module):
    num_params = sum(p.numel() for p in agent_v1_module.parameters())
    assert num_params == AGENT_V1_EXPECTED_PARAMS
