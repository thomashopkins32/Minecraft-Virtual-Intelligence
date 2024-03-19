import minedojo  # type: ignore

from mvi.agent.agent import AgentV1
from mvi.learning.ppo import PPO

if __name__ == "__main__":
    env = minedojo.make(task_id="open-ended", image_size=(160, 256))
    agent = AgentV1(env.action_space)
    ppo = PPO(env, agent)
    ppo.run()
