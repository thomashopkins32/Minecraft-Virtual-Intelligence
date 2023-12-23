from tinygrad import Tensor, nn


class PerceptionModule:
    '''
    Perception module for the agent. This should handle all of the perceptual input that an agent may receive.
    This includes (but may not be limited to):
    - Visual perception
    - Auditory perception
    - Temporal perception (memory)

    Observations from the environment should be streamed into this module. The agent will process the new data
    when it is ready to process it. This means that it can end up missing potentially significant input frames.

    I expect this module to be broken up in the future to account for the fact that some perception processes
    may finish before others.
    '''
    def __init__(self):
        pass

    def forward(self, x_img, x_audio, x_mem):
        '''
        Forward pass over the network. This method determines how we 
        '''
