import torch
from torchvision.transforms.functional import center_crop

from mvi.memory.trajectory import TrajectoryBuffer


def run(
    roi_shape: Tuple[int, int] = (32, 32),
    epochs: int = 50,
    steps_per_epoch: int = 4000,
    discount_factor: float = 0.99,
    gae_discount_factor: float = 0.97,
    save_freq: int = 10,
    **ppo_kwargs: Dict[str, Any],
):
    """
    Runs the Minecraft environment with the virtual intelligence in it

    Parameters
    ----------
    roi_shape : Tuple[int, int]
        Image shape used by foveated perception
    epochs : int, optional
        Number of policy updates to perform after sampling experience
    steps_per_epoch : int, optional
        Number of steps of interaction with the environment per epoch
    discount_factor : float, optional
        Used to weight preference for long-term reward (aka gamma)
    gae_discount_factor : float, optional
        Used to weight preference for long-term advantage (aka lambda)
    clip_ratio : float, optional
        Maximum allowed divergence of the new policy from the old policy in the objective function (aka epsilon)
    save_freq : int, optional
        Rate in terms of number of epochs that the actor and critic models are saved to disk
    ppo_kwargs : Dict[str, Any]
        Parameters passed through to the PPO learning algorithm
    """

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
