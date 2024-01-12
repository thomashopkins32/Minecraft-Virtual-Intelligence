import torch
import torch.nn as nn
from torchvision.transforms.functional import center_crop

class VisualPerception(nn.Module):
    '''
    Visual perception module for the agent. This should handle image related data.

    Image data will be streamed to this module. It should be available to process additional `forward` calls as soon as
    the previous `forward` call ends. If the image processing is too slow, we may miss frames streaming from the environment.

    There are planned to be two sub-modules of the visual perception currently:
    - Peripheral Visual Perception
        - Large convolutional filters that will pass over an entire image
    - Foveated Visual Perception
        - Small convolutional filter that will pass over a very small region of interest within the image
        - The region of interest shall be determined by the most recent output of the actor module
    '''
    def __init__(self, roi_width=10, roi_height=10):
        super().__init__()
        # Set region of interest height and width
        self.roi_width = roi_width
        self.roi_height = roi_height

        # Set up sub-modules
        self.foveated_perception = FoveatedPerception()
        self.peripheral_perception = PeripheralPerception()

        # TODO: How do we combine the visual features from the submodules?


    def forward(self, x_img, x_roi=None):
        '''
        Process visual information from the environment.

        Parameters
        ----------
        x_img : nn.Tensor
            Image coming from the environment (shape TBD)
        x_roi : nn.Tensor, optional
            Region of interest proposed by the actor module
        '''
        if x_roi is None:
            x_roi = center_crop(x_img, (self.roi_width, self.roi_height))
        else:
            assert x_roi.shape == (x_img.shape[0], x_img.shape[1], self.roi_width, self.roi_height)

        fov_x = self.foveated_perception(x_roi)
        per_x = self.peripheral_perception(x_img)

        return torch.tensor()


class FoveatedPerception(nn.Module):
    '''
    Foveated perception module for the agent.
    This should handle image related data directly from the environment.

    This module focuses on computation of fine-grained visual features.
    It does so by using small convolutions with low stride.
    '''
    def __init__(self):
        super().__init__()

    def forward(self, x_img):
        pass


class PeripheralPerception(nn.Module):
    '''
    Peripheral perception module for the agent.
    This should handle image related data directly from the environment.

    This module focuses on computation of coarse-grained visual features.
    It does so by using large convolutions with large stride.
    '''
    def __init__(self):
        super().__init__()



    def forward(self, x_img):
        pass
