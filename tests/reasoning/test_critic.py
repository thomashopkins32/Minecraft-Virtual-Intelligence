import pytest
import torch

from MineAI.reasoning.critic import LinearReasoner


EMBED_DIM = 64
LINEAR_REASONER_EXPECTED_PARAMS = EMBED_DIM + 1


@pytest.fixture
def linear_affector_module():
    return LinearReasoner(embed_dim=EMBED_DIM)


def test_linear_affector_forward(linear_affector_module):
    input_tensor = torch.randn((32, EMBED_DIM))
    out = linear_affector_module(input_tensor)
    assert out.shape == (32, 1)


def test_linear_affector_params(linear_affector_module):
    num_params = sum(p.numel() for p in linear_affector_module.parameters())
    assert num_params == LINEAR_REASONER_EXPECTED_PARAMS
