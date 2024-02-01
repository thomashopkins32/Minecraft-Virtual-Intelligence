from abc import ABC, abstractmethod


class Algorithm(ABC):
    ''' Base class for learning algorithms '''
    # TODO: Come up with an interface after implementing some algorithms
    #       may not be possible to do generally
    @abstractmethod
    def update_policy(self, model):
        pass

    @abstractmethod
    def reward(self, model):
        pass

    @abstractmethod
    def save_trajectory(self, env):
        pass
