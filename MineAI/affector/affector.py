from typing import Int, Tuple

import torch
import torch.nn as nn
import gymnasium


# TODO: Refactor to use action_space
# MineDojo action spaces
## 0: noop, 1: forward, 2: back
LONGITUDINAL_NUM_ACTIONS = 3
## 0: noop, 1: move left, 2: move right
LATERAL_NUM_ACTIONS = 3
## 0: noop, 1: jump, 2: sneak, 3:sprint
JUMP_SNEAK_SPRINT_ACTIONS = 4
## 0: -180 degree, 24: 180 degree
DELTA_PITCH_ACTIONS = 25
## 0: -180 degree, 24: 180 degree
DELTA_YAW_ACTIONS = 25
## 0: noop, 1: use, 2: drop, 3: attack, 4: craft, 5: equip, 6: place, 7: destroy
FUNCTIONAL_ACTIONS = 8
## All possible items to be crafted
CRAFT_ACTIONS = 244  # To remove? Not sure this is necessary
## Inventory slot indices
INVENTORY_ACTIONS = 36

# Internal action spaces
## (x, y) coorindate where the agent will focus its attention next
FOCUS_NUM_ACTIONS = 2


class LinearAffector(nn.Module):
    """
    Feed-forward affector (action) module.

    This module produces actions for the environment given some input using linear layers.
    """

    def __init__(self, embed_dim: Int, action_space: gymnasium.spaces.MultiDiscrete):
        super().__init__()

        # Movement
        self.longitudinal_action = nn.Linear(embed_dim, LONGITUDINAL_NUM_ACTIONS)
        self.lateral_action = nn.Linear(embed_dim, LATERAL_NUM_ACTIONS)
        self.vertical_action = nn.Linear(embed_dim, JUMP_SNEAK_SPRINT_ACTIONS)
        self.pitch_action = nn.Linear(embed_dim, DELTA_PITCH_ACTIONS)
        self.yaw_action = nn.Linear(embed_dim, DELTA_YAW_ACTIONS)

        """
        # Manipulation
        self.functional_action = nn.Linear(embed_dim, FUNCTIONAL_ACTIONS)
        self.craft_action = nn.Linear(embed_dim, CRAFT_ACTIONS)
        self.inventory_action = nn.Linear(embed_dim, INVENTORY_ACTIONS)
        """

        # Internal
        self.focus_action = nn.Linear(embed_dim, FOCUS_NUM_ACTIONS)

        self.softmax = nn.Softmax(dim=1)
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
    ]:
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
        func_dist = torch.zeros(x.size(0), FUNCTIONAL_ACTIONS)
        craft_dist = torch.zeros(x.size(0), CRAFT_ACTIONS)
        inventory_dist = torch.zeros(x.size(0), INVENTORY_ACTIONS)

        focus_coords = self.focus_action(x)

        return (
            long_dist,
            lat_dist,
            vert_dist,
            pitch_dist,
            yaw_dist,
            func_dist,
            craft_dist,
            inventory_dist,
            focus_coords,
        )
