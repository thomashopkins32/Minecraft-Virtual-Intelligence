from typing import List, Tuple, Dict, Int, Float, Str, Any
import logging

import torch


class Trajectory:
    '''
    Inspired by https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/ppo.py
    Used to store a single trajectory.
    '''
    def __init__(self, max_buffer_size: Int, discount_factor: Float = 0.99, gae_discount_factor: Float = 0.95):
        self.max_buffer_size = max_buffer_size
        self.discount_factor = discount_factor
        self.gae_discount_factor = gae_discount_factor
        self.observations = []
        self.actions = []
        self.advantages = []
        self.rewards = []
        self.returns = []
        self.values = []
        self.log_probs = []

    def store(self, observation: torch.Tensor, action: Dict[int, int], reward: Int, value: Int, log_prob: Dict[Int, Float]) -> None:
        '''
        Append a single time-step to the trajectory.

        Parameters
        ----------
        observation : torch.Tensor
            Raw observation from the environment.
        action : Dict[int, int]
            Action dictionary for the MineDojo environment. There are different combinations of actions that can be made.
        reward : Int
            Raw reward value from the environment.
        value : Int
            Value assigned to the observation by the agent.
        log_prob : Float
            Probability of selecting each action.
        '''
        if len(self.observations) == self.max_buffer_size:
            logging.warn(f"Cannot store additional time-steps in an already full trajectory. Current size: {len(self.observations)}. Max size: {self.max_buffer_size}")
            return
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)

    def get(self, last_value: Int) -> Dict[Str, Any]:
        '''
        Computes the advantages and reward-to-go then returns the data from the trajectory.

        Parameters
        ----------
        last_value : Int
            Value assigned to the last observation in the trajectory.
        '''
        size = len(self.observations)
        if size < self.max_buffer_size:
            logging.warn(f"Computing information on a potentially unfinished trajectory. Current size: {size}. Max size: {self.max_buffer_size}")
        rewards = torch.cat((torch.stack(self.rewards), last_value))
        values = torch.cat((torch.stack(self.values), last_value))

        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]




        return dict()




