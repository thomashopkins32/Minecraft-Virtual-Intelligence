from typing import Tuple, Any, Union

import torch
import torch.nn as nn


# (x, y) coorindate where the agent will focus its attention next
FOCUS_NUM_ACTIONS = 2


class LinearAffector(nn.Module):
    """
    Feed-forward affector (action) module.

    This module produces actions for the environment given some input using linear layers.
    """

    def __init__(self, embed_dim: int, action_space: Any):
        super().__init__()

        # Movement
        self.longitudinal_action = nn.Linear(embed_dim, action_space.nvec[0])
        self.lateral_action = nn.Linear(embed_dim, action_space.nvec[1])
        self.vertical_action = nn.Linear(embed_dim, action_space.nvec[2])
        self.pitch_action = nn.Linear(embed_dim, action_space.nvec[3])
        self.yaw_action = nn.Linear(embed_dim, action_space.nvec[4])

        """
        # Manipulation
        self.functional_action = nn.Linear(embed_dim, action_space.nvec[5])
        self.craft_action = nn.Linear(embed_dim, action_space.nvec[6])
        self.inventory_action = nn.Linear(embed_dim, action_space.nvec[7])
        """

        # Internal
        ## The percentage of the image width to crop from
        self.focus_action = nn.Linear(embed_dim, FOCUS_NUM_ACTIONS)

        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.action_space = action_space

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        long_dist = self.softmax(self.longitudinal_action(x))
        lat_dist = self.softmax(self.lateral_action(x))
        vert_dist = self.softmax(self.vertical_action(x))
        pitch_dist = self.softmax(self.pitch_action(x))
        yaw_dist = self.softmax(self.yaw_action(x))

        """
        func_dist = self.softmax(self.functional_action(x))
        craft_dist = self.softmax(self.craft_action(x))
        inventory_dist = self.softmax(self.inventory_action(x))
        """

        # If this doesn't work try (A * torch.ones((x.size(0), 1, 1)))
        action = torch.zeros((10,)).unsqueeze(0).repeat(x.size(0), 0)
        logp_action = torch.zeros_like(action)
        action[0], logp_action[0] = self.get_action_and_logp(long_dist)
        action[1], logp_action[1] = self.get_action_and_logp(lat_dist)
        action[2], logp_action[2] = self.get_action_and_logp(vert_dist)
        action[3], logp_action[3] = self.get_action_and_logp(pitch_dist)
        action[4], logp_action[4] = self.get_action_and_logp(yaw_dist)
        '''
        no_op[5] = torch.multinomial(func_dist, num_samples=1, replacement=False)
        no_op[6] = torch.multinomial(craft_dist, num_samples=1, replacement=False)
        no_op[7] = torch.multinomial(inventory_dist, num_samples=1, replacement=False)
        '''
        action[5], logp_action[5] = 0, 1.0
        action[6], logp_action[6] = 0, 1.0
        action[7], logp_action[7] = 0, 1.0

        # TODO: This should be a gaussian distribution
        # so we need mean, std for x and y separately.
        # A total of 2 layers could work, one for the means of both
        # and one for the stds of both.
        # This way,
        focus_coords = self.sigmoid(self.focus_action(x))

        return action, logp_action

    def get_action_and_logp(self, dist):
        # TODO: Move to utils and rename to `sample_with_logp`
        a = torch.multinomial(dist, num_samples=1, replacement=False)
        return a, torch.log(dist[a])
