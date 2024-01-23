import pytest
import torch
from torchvision.transforms.functional import crop

from MineAI.perception.visual import (
    VisualPerception,
    FoveatedPerception,
    PeripheralPerception,
)

@pytest.fixture
def visual_perception_module():
    return VisualPerception()

def test_visual_perception_forward(visual_perception_module):
    input_tensor = torch.randn((32, 3, 160, 256))

    output = visual_perception_module(input_tensor)

    cropped_input = crop(input_tensor, 0, 0, 32, 32)
    output2 = visual_perception_module(input_tensor, cropped_input)
    

@pytest.fixture
def foveated_perception_module():
    return FoveatedPerception(3, 32)


def test_foveated_perception_forward(foveated_perception_module):
    input_tensor = torch.randn((32, 3, 32, 32))
    output = foveated_perception_module.forward(input_tensor)
    assert output.shape[1] == 32 * 6 * 6  # Check the number of output channels


def test_foveated_perception_module_parameters(foveated_perception_module):
    num_params = sum(p.numel() for p in foveated_perception_module.parameters())
    expected_params = (3 * 3 * 3 * 16) + 16 + (3 * 3 * 16 * 32) + 32
    assert num_params == expected_params


@pytest.fixture
def peripheral_perception_module():
    return PeripheralPerception(1, 32)


def test_peripheral_perception_forward(peripheral_perception_module):
    input_tensor = torch.randn((32, 1, 160, 256))
    output = peripheral_perception_module.forward(input_tensor)
    assert output.shape[1] == 32 * 3 * 8


def test_peripheral_perception_module_parameters(peripheral_perception_module):
    num_params = sum(p.numel() for p in peripheral_perception_module.parameters())
    expected_params = (24 * 24 * 1 * 16) + 16 + (10 * 10 * 16 * 32) + 32
    assert num_params == expected_params

