import minedojo  # type: ignore

from mvi.agent.agent import AgentV1
from mvi.learning.ppo import PPO
from mvi.config import get_config


def run() -> None:
    """
    Entry-point for the project.

    Runs the Minecraft simulation with the virtual intelligence in it.
    """
    config = get_config()
    env = minedojo.make(task_id="open-ended", image_size=config["Engine"]["image_size"])
    agent = AgentV1(env.action_space)
    ppo = PPO(env, agent, **config["PPO"])
    ppo.run()


if __name__ == "__main__":
    run()
