import minedojo  # type: ignore

import mvi.agent.agent as agent
import mvi.learning.ppo as ppo

if __name__ == "__main__":
    env = minedojo.make(task_id="open-ended", image_size=(160, 256))
    agent = agent.AgentV1(env.action_space)
    ppo = ppo.PPO(env, agent)
    ppo.run()
