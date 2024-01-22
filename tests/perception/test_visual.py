import pytest
import torch

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


@pytest.fixture
def foveated_perception_module():
    return FoveatedPerception(3, 32)


def test_foveated_perception_forward(foveated_perception_module):
    # Input tensor with shape (BS, 3, 160, 256)
    input_tensor = torch.randn((32, 3, 160, 256))

    # Forward pass
    output = foveated_perception_module.forward(input_tensor)

    # Check if the output has the expected shape
    assert output.shape[1] == 16  # Check the number of output channels
    assert (
        output.shape[2] == input_tensor.shape[2] // 4
    )  # Check the downsampling by MaxPool2d
    assert output.shape[3] == input_tensor.shape[3] // 4


def test_foveated_perception_module_parameters(foveated_perception_module):
    # Check the number of parameters in the module
    num_params = sum(p.numel() for p in foveated_perception_module.parameters())

    # Expected number of parameters can be calculated based on the convolution layers
    expected_params = 3 * (
        3 * 8 * 3 * 3 + 8 * 16 * 3 * 3
    )  # 3 conv layers with specified parameters

    assert num_params == expected_params


@pytest.fixture
def peripheral_percetion_module():
    return PeripheralPerception(1, 32)


def test_peripheral_perception_forward(peripheral_perception_module):
    # Input tensor with shape (BS, 3, 160, 256)
    input_tensor = torch.randn((32, 3, 160, 256))

    # Forward pass
    output = peripheral_perception_module.forward(input_tensor)

    # Check if the output has the expected shape
    assert output.shape[1] == 16  # Check the number of output channels
    assert (
        output.shape[2] == input_tensor.shape[2] // 4
    )  # Check the downsampling by MaxPool2d
    assert output.shape[3] == input_tensor.shape[3] // 4

