import torch
import torch.nn as nn
from torchvision.transforms.functional import center_crop

from utils import compute_kernel_size, compute_stride


class VisualPerception(nn.Module):
    """
    Visual perception module for the agent. This should handle image related data.

    Image data will be streamed to this module. It should be available to process additional `forward` calls as soon as
    the previous `forward` call ends. If the image processing is too slow, we may miss frames streaming from the environment.

    There are planned to be two sub-modules of the visual perception currently:
    - Peripheral Visual Perception
        - Large convolutional filters that will pass over an entire image
    - Foveated Visual Perception
        - Small convolutional filter that will pass over a very small region of interest within the image
        - The region of interest shall be determined by the most recent output of the actor module
    """

    def __init__(self, output_shape=(32, 8, 8), roi_shape=(16, 16)):
        super().__init__()
        # Set region of interest height and width
        self.roi_shape = roi_shape

        # Set up sub-modules
        self.foveated_perception = FoveatedPerception(output_shape)
        self.peripheral_perception = PeripheralPerception(output_shape)
        self.combiner = VisualAttention(output_shape[0])

    def forward(self, x_img: torch.Tensor, x_roi: torch.Tensor = None) -> torch.Tensor:
        """
        Process visual information from the environment.

        Parameters
        ----------
        x_img : torch.Tensor
            Image coming from the environment (BS, 3, 160, 256)
        x_roi : torch.Tensor, optional
            Region of interest foveated perception will operate on

        Returns
        -------
        torch.Tensor
            Visual features
        """
        if x_roi is None:
            x_roi = center_crop(x_img, self.roi_shape)
        else:
            assert x_roi.shape == (
                x_img.shape[0],
                x_img.shape[1],
                self.roi_shape[0],
                self.roi_shape[1],
            )

        fov_x = self.foveated_perception(x_roi)
        per_x = self.peripheral_perception(x_img)

        combined = self.combiner(fov_x, per_x)

        return combined


class FoveatedPerception(nn.Module):
    """
    Foveated perception module for the agent.
    This should handle image related data directly from the environment.

    This module focuses on computation of fine-grained visual features.
    It does so by using small convolutions with low stride.
    """

    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.conv1 = nn.Conv2d(
            input_shape[0], output_shape[0] // 2, kernel_size=3, stride=1, padding=1
        )
        self.conv2 = nn.Conv2d(
            output_shape[0] // 2, output_shape[0], kernel_size=3, stride=1, padding=1
        )
        stride = compute_stride(
            (input_shape[1], input_shape[2]),
            (output_shape[1] // 2, output_shape[2] // 2),
            2,
        )
        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=stride)
        stride = compute_stride(
            (output_shape[1] // 2, output_shape[2] // 2),
            (output_shape[1], output_shape[2]),
            2,
        )
        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=stride)
        self.gelu = nn.GELU()

    def forward(self, x_img: torch.Tensor) -> torch.Tensor:
        """
        Computation done by the foveated perception module.

        Parameters
        ----------
        x_img : torch.Tensor
            RBG tensor array (BS, 3, H, W)

        Returns
        -------
        torch.Tensor
            Set of visual features (BS, -1, nH, nW)
        """
        x = self.gelu(self.mp1(self.conv1(x_img)))
        x = self.gelu(self.mp2(self.conv2(x)))
        return x


class PeripheralPerception(nn.Module):
    """
    Peripheral perception module for the agent.
    This should handle image related data directly from the environment.

    This module focuses on computation of coarse-grained visual features.
    It does so by using large convolutions with large stride.
    """

    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.conv1 = nn.Conv2d(input_shape[0], output_shape[0] // 2, 20, 4)
        self.conv2 = nn.Conv2d(output_shape[0] // 2, output_shape[0], 10, 2)
        self.max_pool = nn.MaxPool2d(2, 2)
        self.gelu = nn.GELU()

    def forward(self, x_img: torch.Tensor) -> torch.Tensor:
        """
        Computation done by the peripheral perception module.

        Parameters
        ----------
        x_img : torch.Tensor
            Grayscale image of the environment (BS, 1, 160, 256)

        Returns
        -------
        torch.Tensor
            Set of visual features (BS, -1, nH, nW)
        """
        x = self.gelu(self.max_pool(self.conv1(x_img)))
        x = self.gelu(self.max_pool(self.conv2(x)))
        return x


class VisualAttention(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.conv = nn.Conv2d(in_features, 1, 1)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, fov_x: torch.Tensor, per_x: torch.Tensor) -> torch.Tensor:
        """
        Computation for combining the outputs of the other perception modules.
        Performs attention over the concatenated result.

        Parameters
        ----------
        fov_x : torch.Tensor
            Output of the foveated perception module
        per_x : torch.Tensor
            Output of the peripheral perception module

        Returns
        -------
        torch.Tensor
            Fused feature representation
        """
        combined_features = torch.cat((fov_x, per_x), dim=1)
        attention_features = self.conv(combined_features).view(
            combined_features.size(0), 1, -1
        )
        attention_weights = self.softmax(attention_features)
        flat_combined_features = combined_features.view(combined_features.size(0), -1)
        x = (flat_combined_features * attention_weights).sum(dim=2)
        return x
