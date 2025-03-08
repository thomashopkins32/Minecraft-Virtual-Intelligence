from typing import cast

import torch
import minedojo
from gymnasium.spaces import MultiDiscrete
from mvi.agent.agent import AgentV1
from mvi.config import get_config


def run() -> None:
    """
    Entry-point for the project.

    Runs the Minecraft simulation with the virtual intelligence in it.
    """
    # Setup
    config = get_config()
    engine_config = config.engine
    env = minedojo.make(task_id="open-ended", image_size=engine_config.image_size)
    action_space = cast(MultiDiscrete, env.action_space)
    agent = AgentV1(config.agent, action_space)

    obs = env.reset()["rgb"].copy()  # type: ignore[no-untyped-call]
    obs = torch.tensor(obs, dtype=torch.float).unsqueeze(0)
    total_return = 0.0
    for s in range(engine_config.max_steps):
        action = agent.act(obs).squeeze(0)
        next_obs, reward, _, _ = env.step(action)
        total_return += reward
        obs = torch.tensor(next_obs["rgb"].copy(), dtype=torch.float).unsqueeze(0)


if __name__ == "__main__":
    run()
