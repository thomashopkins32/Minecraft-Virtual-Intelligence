from typing import Tuple, Any

import torch
import torch.nn as nn


# (x, y) coorindate where the agent will focus its attention next
FOCUS_NUM_ACTIONS = 2


class LinearAffector(nn.Module):
    """
    Feed-forward affector (action) module.

    This module produces distributions over actions for the environment given some input using linear layers.
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
        ## distribution for which we can sample regions of interest
        self.focus_x_mean_std = nn.Linear(embed_dim, FOCUS_NUM_ACTIONS)
        self.focus_y_mean_std = nn.Linear(embed_dim, FOCUS_NUM_ACTIONS)

        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.action_space = action_space

    def forward(self, x: torch.Tensor) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        long_dist = self.softmax(self.longitudinal_action(x))
        lat_dist = self.softmax(self.lateral_action(x))
        vert_dist = self.softmax(self.vertical_action(x))
        pitch_dist = self.softmax(self.pitch_action(x))
        yaw_dist = self.softmax(self.yaw_action(x))
        func_dist = torch.zeros((x.size(0), self.action_space.nvec[5]))
        craft_dist = torch.zeros((x.size(0), self.action_space.nvec[6]))
        inventory_dist = torch.zeros((x.size(0), self.action_space.nvec[7]))
        """ TODO: Add these options back in
        func_dist = self.softmax(self.functional_action(x))
        craft_dist = self.softmax(self.craft_action(x))
        inventory_dist = self.softmax(self.inventory_action(x))
        """
        roi_means = self.focus_x_mean_std(x)
        roi_stds = self.focus_y_mean_std(x)

        return (
            long_dist,
            lat_dist,
            vert_dist,
            pitch_dist,
            yaw_dist,
            func_dist,
            craft_dist,
            inventory_dist,
            roi_means,
            roi_stds,
        )
