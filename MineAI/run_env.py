import torch
import minedojo

from agent.agent import AgentV1

env = minedojo.make(task_id="open-ended", image_size=(160, 256))
agent = AgentV1()

obs = env.reset()
done = False
agent.eval()
while not done:
    t_obs = torch.tensor(obs["rgb"].copy(), dtype=torch.float).unsqueeze(0)
    full_action = env.action_space.no_op()
    with torch.no_grad():
        pitch_dist, yaw_dist = agent(t_obs)
    pitch = pitch_dist.multinomial(num_samples=1, replacement=False)
    yaw = yaw_dist.multinomial(num_samples=1, replacement=False)
    full_action[3] = pitch
    full_action[4] = yaw
    obs, reward, done, info = env.step(full_action)
    env.render()
env.close()