import sys

import minedojo  # type: ignore

from mvi.agent.agent import AgentV1
from mvi.learning.ppo import PPO
from mvi.config import parse_config

if __name__ == "__main__":

    config = parse_config(sys.argv[1])

    env = minedojo.make(task_id="open-ended", image_size=(160, 256))
    agent = AgentV1(env.action_space)
    ppo = PPO(env, agent, **config["PPO"])
    ppo.run()
