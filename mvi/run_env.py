import torch
import minedojo  # type: ignore

from mvi.agent.agent import AgentV1

env = minedojo.make(task_id="open-ended", image_size=(160, 256))
agent = AgentV1(env.action_space)

obs = env.reset()
done = False
agent.eval()
while not done:
    t_obs = torch.tensor(obs["rgb"].copy(), dtype=torch.float).unsqueeze(0)
    full_action = env.action_space.no_op()
    with torch.no_grad():
        action_dist, _ = agent(t_obs)
    pitch = action_dist[3].multinomial(num_samples=1, replacement=False)
    yaw = action_dist[4].multinomial(num_samples=1, replacement=False)
    full_action[3] = pitch
    full_action[4] = yaw
    obs, reward, done, info = env.step(full_action)
    env.render()
env.close()
