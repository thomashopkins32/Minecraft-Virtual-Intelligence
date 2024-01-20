import pytest
import torch

from MineAI.perception.visual import (
    VisualPerception,
    FoveatedPerception,
    PeripheralPerception,
    VisualAttention,
)


@pytest.fixture
def foveated_perception_model():
    return FoveatedPerception()


def test_foveated_perception_forward(foveated_perception_model):
    # Input tensor with shape (BS, 3, 160, 256)
    input_tensor = torch.randn((32, 3, 160, 256))

    # Forward pass
    output = foveated_perception_model.forward(input_tensor)

    # Check if the output has the expected shape
    assert output.shape[1] == 16  # Check the number of output channels
    assert (
        output.shape[2] == input_tensor.shape[2] // 4
    )  # Check the downsampling by MaxPool2d
    assert output.shape[3] == input_tensor.shape[3] // 4


def test_foveated_perception_module_parameters(foveated_perception_model):
    # Check the number of parameters in the module
    num_params = sum(p.numel() for p in foveated_perception_model.parameters())

    # Expected number of parameters can be calculated based on the convolution layers
    expected_params = 3 * (
        3 * 8 * 3 * 3 + 8 * 16 * 3 * 3
    )  # 3 conv layers with specified parameters

    assert num_params == expected_params
