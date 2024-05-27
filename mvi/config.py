from dataclasses import dataclass
import logging
import argparse
from typing import Dict, Any, Tuple

import yaml


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


def get_config() -> Dict[str, Any]:
    # TODO: Refactor to use new dataclasses
    arguments = parse_arguments()
    config = parse_config(arguments.file)
    return update_config(config, arguments.key_value_pairs)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Specify arguments for running Minecraft Virtual Intelligence"
    )
    parser.add_argument(
        "-f", "--file", help="Path to YAML configuration file", required=True
    )
    parser.add_argument(
        "key_value_pairs",
        nargs="+",
        help="Key-value pairs to override in the configuration (nested configs can be accessed via '.')",
    )
    return parser.parse_args()


def parse_config(yaml_path: str) -> Dict[str, Any]:
    """
    Parses the configuration file used by the engine.

    Parameters
    ----------
    yaml_path : str
        Path to the yaml configuration file to use

    Returns
    -------
    Dict[str, Any]
        The configuration as a dictionary
    """
    # TODO: Refactor to use new dataclasses
    with open(yaml_path, "r") as fp:
        config = yaml.safe_load(fp)

    return config


def parse_value(value: Any) -> Any:
    """Parses the value as if it was being loaded in a YAML file"""
    return yaml.load(value, Loader=yaml.SafeLoader)


def update_config(config: Dict[str, Any], key_value_pairs: list[str]) -> Dict[str, Any]:
    """ "Updates the configuration using the command-line argumments"""
    # TODO: Refactor to use new dataclasses

    for pair in key_value_pairs:
        key, value = pair.split("=")
        value = parse_value(value)
        keys = key.split(".")
        current_config = config
        for depth, k in enumerate(keys[:-1]):
            if not isinstance(current_config, dict) and k not in current_config:
                raise ValueError(
                    f"Could not find Key '{k}' in '{key}' in configuration at depth {depth}"
                )
            current_config = current_config[k]
        last_key = keys[-1]
        if last_key not in current_config:
            raise ValueError(
                f"Could not find Key '{k}' in '{key}' in configuration at depth {depth}"
            )
        if type(current_config[last_key]) != type(value):
            raise ValueError(
                f"Cannot replace '{key}' with type {type(current_config[last_key])} with type {type(value)}"
            )
        current_config[last_key] = value
        logging.warning(f"Updating '{key}' to be '{value}'")

    return config
