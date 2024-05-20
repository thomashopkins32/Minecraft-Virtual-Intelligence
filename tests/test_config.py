import os

import pytest

from mvi.learning.ppo import PPO
from mvi.config import parse_config, update_config


def test_ppo_config():
    project_root = os.path.abspath(os.path.join(__file__, "..", ".."))
    config = parse_config(os.path.join(project_root, "templates", "config.yaml"))
    ppo = PPO(None, None, **config["PPO"])

    assert ppo.roi_shape == config["PPO"]["roi_shape"]
    assert ppo.epochs == config["PPO"]["epochs"]
    assert ppo.steps_per_epoch == config["PPO"]["steps_per_epoch"]
    assert ppo.discount_factor == config["PPO"]["discount_factor"]
    assert ppo.gae_discount_factor == config["PPO"]["gae_discount_factor"]
    assert ppo.clip_ratio == config["PPO"]["clip_ratio"]
    assert ppo.target_kl == config["PPO"]["target_kl"]
    assert ppo.actor_lr == config["PPO"]["actor_lr"]
    assert ppo.critic_lr == config["PPO"]["critic_lr"]
    assert ppo.train_actor_iters == config["PPO"]["train_actor_iters"]
    assert ppo.train_critic_iters == config["PPO"]["train_critic_iters"]
    assert ppo.save_freq == config["PPO"]["save_freq"]


def test_parse_config():
    project_root = os.path.abspath(os.path.join(__file__, "..", ".."))
    parse_config(os.path.join(project_root, "templates", "config.yaml"))


def test_update_config():
    example_config = {
        "Level0": {
            "Level1-1": "Test1",
            "Level1-2": {
                "Level2": 2,
            },
        },
    }

    # Other - empty list
    config = update_config(example_config, [])
    assert config["Level0"]["Level1-1"] == "Test1"
    assert config["Level0"]["Level1-2"]["Level2"] == 2 

    # Valid update - replace values
    to_update = [ 
        "Level0.Level1-1=Test2",
        "Level0.Level1-2.Level2=7"
    ]
    config = update_config(example_config, to_update)
    assert config["Level0"]["Level1-1"] == "Test2"
    assert config["Level0"]["Level1-2"]["Level2"] == 7

    # Invalid update - type mismatch
    to_update = [
        "Level0=10"
    ]
    with pytest.raises(ValueError) as _:
        config = update_config(example_config, to_update)

    # Invalid update - key not found
    to_update = [
        "Test1"
    ]
    with pytest.raises(ValueError) as _:
        config = update_config(example_config, to_update)




    
