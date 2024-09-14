import os
from dataclasses import asdict
from copy import deepcopy

import pytest
import yaml

from mvi.agent.agent import AgentV1
from mvi.config import (
    parse_config,
    update_config,
    PPOConfig,
    EngineConfig,
    AgentConfig,
    Config,
)
from tests.helper import ACTION_SPACE


def test_ppo_config():
    """Test consistency between configuration and PPO algorithm"""
    project_root = os.path.abspath(os.path.join(__file__, "..", ".."))
    template_path = os.path.join(project_root, "templates", "config.yaml")

    # Base template (expected values)
    with open(template_path, "r") as fp:
        config_dict = yaml.load(fp, yaml.Loader)

    # Parsing
    config = parse_config(template_path)
    agent = AgentV1(config.agent, ACTION_SPACE)

    # Comparison
    assert agent.ppo.clip_ratio == config_dict["agent"]["ppo"]["clip_ratio"]
    assert agent.ppo.target_kl == config_dict["agent"]["ppo"]["target_kl"]
    assert (
        agent.ppo.actor_optim.param_groups[-1]["lr"]
        == config_dict["agent"]["ppo"]["actor_lr"]
    )
    assert (
        agent.ppo.critic_optim.param_groups[-1]["lr"]
        == config_dict["agent"]["ppo"]["critic_lr"]
    )
    assert (
        agent.ppo.train_actor_iters == config_dict["agent"]["ppo"]["train_actor_iters"]
    )
    assert (
        agent.ppo.train_critic_iters
        == config_dict["agent"]["ppo"]["train_critic_iters"]
    )


def test_parse_config():
    project_root = os.path.abspath(os.path.join(__file__, "..", ".."))
    template_path = os.path.join(project_root, "templates", "config.yaml")

    # Base template (expected values)
    with open(template_path, "r") as fp:
        config_dict = yaml.load(fp, yaml.Loader)

    # Parsing
    config = parse_config(template_path)

    # Comparison
    assert (
        config.agent.ppo.discount_factor
        == config_dict["agent"]["ppo"]["discount_factor"]
    )
    assert (
        config.agent.ppo.gae_discount_factor
        == config_dict["agent"]["ppo"]["gae_discount_factor"]
    )
    assert config.engine.image_size == config_dict["engine"]["image_size"]
    assert config.engine.max_steps == config_dict["engine"]["max_steps"]
    assert config.agent.roi_shape == config_dict["agent"]["roi_shape"]
    assert config.agent.max_buffer_size == config_dict["agent"]["max_buffer_size"]
    assert config.agent.ppo.actor_lr == config_dict["agent"]["ppo"]["actor_lr"]
    assert config.agent.ppo.critic_lr == config_dict["agent"]["ppo"]["critic_lr"]
    assert config.agent.ppo.critic_lr == config_dict["agent"]["ppo"]["critic_lr"]
    assert config.agent.ppo.clip_ratio == config_dict["agent"]["ppo"]["clip_ratio"]
    assert config.agent.ppo.target_kl == config_dict["agent"]["ppo"]["target_kl"]
    assert (
        config.agent.ppo.train_actor_iters
        == config_dict["agent"]["ppo"]["train_actor_iters"]
    )
    assert (
        config.agent.ppo.train_critic_iters
        == config_dict["agent"]["ppo"]["train_critic_iters"]
    )


def test_update_config():
    config = Config(engine=EngineConfig(), agent=AgentConfig(ppo=PPOConfig()))

    # Other - empty list
    before_change = deepcopy(config)
    update_config(config, [])

    assert asdict(before_change) == asdict(config)

    # Valid update - replace values
    to_update = ["engine.image_size=[200,200]", "agent.ppo.clip_ratio=3.0"]
    update_config(config, to_update)
    assert config.engine.image_size == (200, 200)
    assert config.agent.ppo.clip_ratio == 3.0
    assert asdict(before_change) != asdict(config)

    # Invalid update - type mismatch
    to_update = ["engine.image_size=10"]
    with pytest.raises(ValueError) as _:
        update_config(config, to_update)

    # Invalid update - key not found
    to_update = ["Test1"]
    with pytest.raises(ValueError) as _:
        update_config(config, to_update)
