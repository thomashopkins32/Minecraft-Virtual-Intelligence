import os

from mvi.learning.ppo import PPO
from mvi.engine import run
from mvi.config import parse_config


def test_engine_config():
    project_root = os.path.abspath(os.path.join(__file__, "..", ".."))
    config = parse_config(os.path.join(project_root, "templates", "config.yaml"))
    run(**config["Engine"], **config["PPO"])

def test_ppo_config():
    project_root = os.path.abspath(os.path.join(__file__, "..", ".."))
    config = parse_config(os.path.join(project_root, "templates", "config.yaml"))
    ppo = PPO(None, None, **config["PPO"])

    assert ppo.clip_ratio == config["PPO"]["clip_ratio"]
    assert ppo.target_kl == config["PPO"]["target_kl"]
    assert ppo.actor_optim.param_groups[0]["lr"] == config["PPO"]["actor_lr"]
    assert ppo.critic_optim.param_groups[0]["lr"] == config["PPO"]["critic_lr"]
    assert ppo.train_actor_iters == config["PPO"]["train_actor_iters"]
    assert ppo.train_critic_iters == config["PPO"]["train_critic_iters"]
