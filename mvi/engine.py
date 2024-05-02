import torch
from torchvision.transforms.functional import center_crop

from mvi.memory.trajectory import TrajectoryBuffer


def run():
    """Runs the Minecraft environment with the virtual intelligence in it"""

    for e in range(epochs):
        trajectory_buffer = TrajectoryBuffer(
            max_buffer_size=steps_per_epoch,
            discount_factor=discount_factor,
            gae_discount_factor=gae_discount_factor,
        )
        obs = torch.tensor(
            env.reset()["rgb"].copy(), dtype=torch.float
        ).unsqueeze(0)
        roi_obs = center_crop(obs, roi_shape)
        t_return = 0.0
        for t in range(steps_per_epoch):
            with torch.no_grad():
                a, v = agent(obs, roi_obs)
            action, logp_action = _sample_action(a)
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
            obs = torch.tensor(next_obs["rgb"].copy(), dtype=torch.float).unsqueeze(
                0
            )
            roi_obs = crop(
                obs,
                roi_action[0],
                roi_action[1],
                roi_shape[0],
                roi_shape[1],
            )
        _, last_v = agent(obs, roi_obs)
        data = trajectory_buffer.get(last_v)
        _update_actor(data, actor_optim)
        _update_critic(data, critic_optim)
