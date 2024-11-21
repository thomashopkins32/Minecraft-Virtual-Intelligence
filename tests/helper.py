import os

import numpy as np
import gymnasium
from gymnasium import spaces


PROJECT_ROOT = os.path.abspath(os.path.join(__file__, "..", ".."))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config_templates", "config.yaml")
ACTION_SPACE = spaces.MultiDiscrete([3, 3, 4, 25, 25, 8, 244, 36])


class MockEnv(gymnasium.Env):
    def __init__(self):
        super().__init__()

        self.observation_space = spaces.Dict(
            {
                "rgb": spaces.Box(0, 255, shape=(160, 256), dtype=int),
            }
        )

        self.action_space = ACTION_SPACE

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        return {"rgb": np.zeros((3, 160, 256), dtype=int)}

    def step(self, action):
        return {"rgb": np.zeros((3, 160, 256), dtype=int)}, 0, False, None

    def render(self):
        pass

    def close(self):
        pass
