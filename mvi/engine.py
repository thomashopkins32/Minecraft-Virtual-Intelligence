import torch
from torchvision.transforms.functional import crop, center_crop  # type: ignore
import minedojo  # type: ignore

from mvi.agent.agent import AgentV1
from mvi.learning.ppo import PPO
from mvi.config import get_config
from mvi.memory.trajectory import PPOTrajectory
from mvi.utils import sample_action


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

    """
    # Setup
    config = get_config()
    engine_config = config.engine
    env = minedojo.make(task_id="open-ended", image_size=engine_config.image_size)
    agent = AgentV1(env.action_space)
    ppo = PPO(agent, config.ppo)

    # Environment Loop
    obs = torch.tensor(env.reset()["rgb"].copy(), dtype=torch.float).unsqueeze(0)
    roi_obs = center_crop(obs, engine_config.roi_shape)
    for s in range(engine_config.max_steps):
        trajectory_buffer = PPOTrajectory(
            max_buffer_size=engine_config.max_buffer_size,
            discount_factor=engine_config.discount_factor,
            gae_discount_factor=engine_config.gae_discount_factor,
        )
        t_return = 0.0
        for t in range(engine_config.max_buffer_size):
            with torch.no_grad():
                a, v = agent(obs, roi_obs)
            action, logp_action = sample_action(a)
            env_action = action[:-2].numpy()  # Don't include the region of interest
            roi_action = action[-2:]
            next_obs, reward, _, _ = env.step(env_action)
            t_return += reward

            trajectory_buffer.store(
                (obs.squeeze(), roi_obs.squeeze()),
                action,
                reward,
                v,
                # Sum the log probability to get the joint probability of selecting the full action
                logp_action.sum(),
            )
            obs = torch.tensor(next_obs["rgb"].copy(), dtype=torch.float).unsqueeze(0)
            roi_obs = crop(
                obs,
                roi_action[0],
                roi_action[1],
                engine_config.roi_shape[0],
                engine_config.roi_shape[1],
            )
        _, last_v = agent(obs, roi_obs)
        trajectory_buffer.finalize_trajectory(last_v)

        # Update models
        ppo.update(trajectory_buffer)
        """


if __name__ == "__main__":
    run()
