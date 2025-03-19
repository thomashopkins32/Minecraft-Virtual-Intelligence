from typing import cast
from datetime import datetime

import torch
import minedojo
from gymnasium.spaces import MultiDiscrete

from .agent.agent import AgentV1
from .config import get_config
from .monitoring.event_bus import event_bus
from .monitoring.event import Event, EnvReset, EnvStep


def run() -> None:
    """
    Entry-point for the project.

    Runs the Minecraft simulation with the virtual intelligence in it.
    """
    config = get_config()
    engine_config = config.engine
    env = minedojo.make(task_id="open-ended", image_size=engine_config.image_size)
    action_space = cast(MultiDiscrete, env.action_space)
    agent = AgentV1(config.agent, action_space)

    obs = env.reset()["rgb"].copy()  # type: ignore[no-untyped-call]
    event_bus.publish(EnvReset(timestamp=datetime.now(), observation=obs))
    obs = torch.tensor(obs, dtype=torch.float).unsqueeze(0)
    total_return = 0.0
    for s in range(engine_config.max_steps):
        action = agent.act(obs).squeeze(0)
        next_obs, reward, _, _ = env.step(action)
        total_return += reward
        obs = torch.tensor(next_obs["rgb"].copy(), dtype=torch.float).unsqueeze(0)
        event_bus.publish(EnvStep(timestamp=datetime.now(), observation=obs, action=action, reward=reward, next_observation=next_obs))


@event_bus.subscribe(Event)
def log_timestamp(event: Event) -> None:
    print(f"{type(event)}: {event.timestamp}")


if __name__ == "__main__":
    run()
