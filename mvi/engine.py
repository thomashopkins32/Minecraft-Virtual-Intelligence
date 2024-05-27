import minedojo  # type: ignore

from mvi.agent.agent import AgentV1
from mvi.learning.ppo import PPO
from mvi.config import get_config


def run() -> None:
    """
    Entry-point for the project.

    Runs the Minecraft simulation with the virtual intelligence in it.
    """

    # Setup
    config = get_config()
    engine_config = config["Engine"]
    env = minedojo.make(task_id="open-ended", image_size=config["Engine"]["image_size"])
    agent = AgentV1(env.action_space)
    ppo = PPO(env, agent, **config["PPO"])

    # Environment Loop
    for s in range(engine_config["max_steps"]):
        trajectory_buffer = PPOTrajectory(
            max_buffer_size=self.steps_per_epoch,
            discount_factor=self.discount_factor,
            gae_discount_factor=self.gae_discount_factor,
        )
        obs = torch.tensor(
            self.env.reset()["rgb"].copy(), dtype=torch.float
        ).unsqueeze(0)
        roi_obs = center_crop(obs, self.roi_shape)
        t_return = 0.0
        for t in range(self.steps_per_epoch):
            with torch.no_grad():
                a, v = self.agent(obs, roi_obs)
            action, logp_action = self._sample_action(a)
            env_action = action[:-2].numpy()  # Don't include the region of interest
            roi_action = action[-2:]
            next_obs, reward, _, _ = self.env.step(env_action)
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
                self.roi_shape[0],
                self.roi_shape[1],
            )
        _, last_v = self.agent(obs, roi_obs)
        data = trajectory_buffer.get(last_v)
        self._update_actor(data, actor_optim)
        self._update_critic(data, critic_optim)

    # Update models


if __name__ == "__main__":
    run()
