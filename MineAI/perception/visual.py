import torch
import torch.nn as nn
from torchvision.transforms.functional import center_crop, rgb_to_grayscale


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

    def __init__(self, out_channels=32, roi_shape=(32, 32)):
        super().__init__()
        # Set region of interest height and width
        self.roi_shape = roi_shape

        # Set up sub-modules
        self.foveated_perception = FoveatedPerception(3, out_channels)
        self.peripheral_perception = PeripheralPerception(1, out_channels)

        # Combiner
        self.attention = nn.MultiheadAttention(128, 3, batch_first=True)

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
        gray_x = rgb_to_grayscale(x_img)

        fov_x = self.foveated_perception(x_roi)
        per_x = self.peripheral_perception(gray_x)

        combined = torch.cat((fov_x, per_x), dim=1)
        combined = combined.view(combined.size(0), 1, -1)
        out = self.attention(combined, combined, combined)

        return out


class FoveatedPerception(nn.Module):
    """
    Foveated perception module for the agent.
    This should handle image related data directly from the environment.

    This module focuses on computation of fine-grained visual features.
    It does so by using small convolutions with low stride.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels // 2, kernel_size=3, stride=1, padding=1
        )
        self.conv2 = nn.Conv2d(
            out_channels // 2, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.gelu = nn.GELU()
        self.flatten = nn.Flatten()

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
        return self.flatten(x)


class PeripheralPerception(nn.Module):
    """
    Peripheral perception module for the agent.
    This should handle image related data directly from the environment.

    This module focuses on computation of coarse-grained visual features.
    It does so by using large convolutions with large stride.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, 40, 1)
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels, 20, 1)
        self.mp1 = nn.MaxPool2d(5, 5)
        self.mp2 = nn.MaxPool2d(5, 5)
        self.gelu = nn.GELU()
        self.flatten = nn.Flatten()

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
        x = self.gelu(self.mp1(self.conv1(x_img)))
        x = self.gelu(self.mp2(self.conv2(x)))
        return self.flatten(x)

