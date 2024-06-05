import os

import pytest
import yaml

from mvi.learning.ppo import PPO
from mvi.config import parse_config, update_config, PPOConfig, EngineConfig, Config


def test_ppo_config():
    """Test consistency between configuration and PPO algorithm"""
    project_root = os.path.abspath(os.path.join(__file__, "..", ".."))
    template_path = os.path.join(project_root, "templates", "config.yaml")

    # Base template (expected values)
    config_dict = yaml.safe_load(template_path)

    # Parsing
    config = parse_config(template_path)
    ppo = PPO(None, None, config.ppo)

    # Comparison
    assert ppo.clip_ratio == config_dict["ppo"]["clip_ratio"]
    assert ppo.target_kl == config_dict["ppo"]["target_kl"]
    assert ppo.actor_optim.param_groups[-1]["lr"] == config_dict["ppo"]["actor_lr"]
    assert ppo.critic_optim.param_groups[-1]["lr"] == config_dict["ppo"]["critic_lr"]
    assert ppo.train_actor_iters == config_dict["ppo"]["train_actor_iters"]
    assert ppo.train_critic_iters == config_dict["ppo"]["train_critic_iters"]


def test_parse_config():
    project_root = os.path.abspath(os.path.join(__file__, "..", ".."))
    template_path = os.path.join(project_root, "templates", "config.yaml")

    # Base template (expected values)
    config_dict = yaml.safe_load(template_path)

    # Parsing
    config = parse_config(template_path)

    # Comparison
    assert config.engine.discount_factor == config_dict["engine"]["discount_factor"]
    assert (
        config.engine.gae_discount_factor
        == config_dict["engine"]["gae_discount_factor"]
    )
    assert config.engine.image_size == config_dict["engine"]["image_size"]
    assert config.engine.max_buffer_size == config_dict["engine"]["max_buffer_size"]
    assert config.engine.max_steps == config_dict["engine"]["max_steps"]
    assert config.engine.roi_shape == config_dict["engine"]["roi_shape"]
    assert config.ppo.actor_lr == config_dict["ppo"]["actor_lr"]
    assert config.ppo.critic_lr == config_dict["ppo"]["critic_lr"]
    assert config.ppo.critic_lr == config_dict["ppo"]["critic_lr"]
    assert config.ppo.clip_ratio == config_dict["ppo"]["clip_ratio"]
    assert config.ppo.target_kl == config_dict["ppo"]["target_kl"]
    assert config.ppo.train_actor_iters == config_dict["ppo"]["train_actor_iters"]
    assert config.ppo.train_critic_iters == config_dict["ppo"]["train_critic_iters"]


def test_update_config():
    example_config = Config(engine=EngineConfig(), ppo=PPOConfig())

    # Other - empty list
    config = update_config(example_config, [])
    # TODO: Write comparison== for Config

    # Valid update - replace values
    to_update = ["engine.image_size=[200,200]", "ppo.clip_ratio=3.0"]
    config = update_config(example_config, to_update)
    assert config.engine.image_size == (200, 200)
    assert config.ppo.clip_ratio == 3.0

    # Invalid update - type mismatch
    to_update = ["engine.image_size=10"]
    with pytest.raises(ValueError) as _:
        config = update_config(example_config, to_update)

    # Invalid update - key not found
    to_update = ["Test1"]
    with pytest.raises(ValueError) as _:
        config = update_config(example_config, to_update)
