import torch
import minedojo  # type: ignore

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
    agent = AgentV1(config.agent, env.action_space)

    obs = torch.tensor(env.reset()["rgb"].copy(), dtype=torch.float).unsqueeze(0)
    total_return = 0.0
    for s in range(engine_config.max_steps):
        action = agent.act(obs)
        next_obs, reward, _, _ = env.step(action)
        total_return += reward
        obs = torch.tensor(next_obs["rgb"].copy(), dtype=torch.float).unsqueeze(0)


if __name__ == "__main__":
    run()
