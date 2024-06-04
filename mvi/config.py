from dataclasses import dataclass, is_dataclass
import argparse
from typing import Any, Tuple

import yaml
from dacite import from_dict


@dataclass
class EngineConfig:
    """
    Configuration definition for the Engine running the Minecraft experience loop

    Attributes
    ----------
    image_size : Tuple[int, int], optional
        Height and width for Minecraft rendered images
    max_steps : int, optional
        Total number of environment steps before program termination
    max_buffer_size : int, optional
        Trajectory buffer capacity prior to model updates
    roi_shape : Tuple[int, int], optional
        Height and width of region of interest for visual perception
    discount_factor : float, optional
        Discount factor for calculating rewards
    gae_discount_factor : float, optional
        Discount factor for Generalized Advantage Estimation
    """

    image_size: Tuple[int, int] = (160, 256)
    max_steps: int = 10_000
    max_buffer_size: int = 50
    roi_shape: Tuple[int, int] = (32, 32)
    discount_factor: float = 0.99
    gae_discount_factor: float = 0.97


@dataclass
class PPOConfig:
    """
    Configuration definition for the PPO learning algorithm

    Attributes
    ----------
    clip_ratio : float, optional
        Maximum allowed divergence of the new policy from the old policy in the objective function (aka epsilon)
    target_kl : float, optional
        Target KL divergence for policy updates; used in model selection (early stopping)
    actor_lr : float, optional
        Learning rate for the actor module
    critic_lr : float, optional
        Learning rate for the critic module
    train_actor_iters : int, optional
        Number of iterations to train the actor per epoch
    train_critic_iters : int, optional
        Number of iterations to train the critic per epoch
    """

    clip_ratio: float = 0.2
    target_kl: float = 0.01
    actor_lr: float = 3.0e-4
    critic_lr: float = 1.0e-3
    train_actor_iters: int = 80
    train_critic_iters: int = 80


@dataclass
class Config:
    """
    Configuration definitions for the full program

    Attributes
    ----------
    engine : EngineConfig
        Configuration for the engine
    ppo : PPOConfig
        Configuration for the PPO learning algorithm
    """

    engine: EngineConfig
    ppo: PPOConfig


def get_config() -> Config:
    arguments = parse_arguments()
    config = parse_config(arguments.file)
    return config


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Specify arguments for running Minecraft Virtual Intelligence"
    )
    parser.add_argument(
        "-f",
        "--file",
        help="Path to YAML configuration file",
        required=False,
        default=None,
    )
    parser.add_argument(
        "key_value_pairs",
        nargs="+",
        help="Key-value pairs to override in the configuration (nested configs can be accessed via '.')",
    )
    return parser.parse_args()


def parse_config(yaml_path: str) -> Config:
    """
    Parses the configuration file used by the engine.

    Parameters
    ----------
    yaml_path : str
        Path to the yaml configuration file to use

    Returns
    -------
    Config
        The current configuration
    """
    if yaml_path is None:
        config = Config()
    else:
        with open(yaml_path, "r") as fp:
            config_dict = yaml.safe_load(fp)
        config = from_dict(data_class=Config, data=config_dict)

    return config


def parse_value(value: str) -> Any:
    """Parses the value as if it was being loaded in a YAML file"""
    return yaml.load(value, Loader=yaml.SafeLoader)


def set_value(instance: Any, keys: list[str], value: Any) -> None:

    for key in keys[:-1]:
        instance = getattr(instance, key)
        if is_dataclass(instance):
            continue
        else:
            raise ValueError(
                f"Expected attribute '{key}' to be a dataclass instance but got '{type(key)}'"
            )

    attr = keys[-1]
    value = parse_value(value)
    old_value = getattr(instance, attr)
    if type(value) != type(old_value):
        raise ValueError(
            f"Expected attribute to be '{type(old_value)}' but got '{type(value)}'"
        )

    setattr(instance, attr, value)


def update_config(config: Config, key_value_pairs: list[str]) -> Config:
    """Updates the configuration using the command-line argumments"""

    for pair in key_value_pairs:
        path, value = pair.split("=", 1)
        keys = path.split(".")
        set_value(config, keys, value)

    return config
