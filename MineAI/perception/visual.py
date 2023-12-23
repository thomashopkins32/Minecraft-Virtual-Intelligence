

class VisualPerception:
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
    def __init__(self):
        pass

    def forward(self, x_img, x_roi=None):
        '''
        Process visual information from the environment.

        Parameters
        ----------
        x_img : Tensor
            Image coming from the environment (shape TBD)
        x_roi : Tensor, optional
            Region of interest proposed by the actor module
        '''
        pass