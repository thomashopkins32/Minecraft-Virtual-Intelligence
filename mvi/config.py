from collections.abc import Iterable
from dataclasses import dataclass, is_dataclass
import argparse
from typing import Any

import yaml
from dacite import from_dict


@dataclass
class EngineConfig:
    """
    Configuration definition for the Engine running the Minecraft experience loop

    Attributes
    ----------
    image_size : tuple[int, int], optional
        Height and width for Minecraft rendered images
    max_steps : int, optional
        Total number of environment steps before program termination
    """

    image_size: tuple[int, int] = (160, 256)
    max_steps: int = 10_000


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
    discount_factor : float, optional
        Discount factor for calculating rewards
    gae_discount_factor : float, optional
        Discount factor for Generalized Advantage Estimation
    """

    clip_ratio: float = 0.2
    target_kl: float = 0.01
    actor_lr: float = 3.0e-4
    critic_lr: float = 1.0e-3
    train_actor_iters: int = 80
    train_critic_iters: int = 80
    discount_factor: float = 0.99
    gae_discount_factor: float = 0.97


@dataclass
class TDConfig:
    """Configuration definition for the Temporal Difference Actor Critic learning algorithm"""

    discount_factor: float = 0.99
    actor_lr: float = 1.0e-3
    critic_lr: float = 1.0e-3


@dataclass
class ICMConfig:
    """
    Configuration definition for the ICM learning algorithm

    Attributes
    ----------
    scaling_factor : float, optional
        Used to scale the influence of curiosity on the reward signal
        (must be > 0)
    inverse_dynamics_lr : float, optional
        Learning rate for the inverse dynamics module
    forward_dynamics_lr : float, optional
        Learning rate for the forward dynamics module
    """

    scaling_factor: float = 1.0
    train_inverse_dynamics_iters: int = 80
    train_forward_dynamics_iters: int = 80
    inverse_dynamics_lr: float = 1.0e-3
    forward_dynamics_lr: float = 1.0e-3


@dataclass
class AgentConfig:
    """
    Configuration definitions for the agent

    Attributes
    ----------
    ppo : PPOConfig
        Configuration for the PPO learning algorithm
    max_buffer_size : int, optional
        Trajectory buffer capacity prior to model updates
    roi_shape : tuple[int, int], optional
        Height and width of region of interest for visual perception
    """

    ppo: PPOConfig
    icm: ICMConfig
    max_buffer_size: int = 50
    roi_shape: tuple[int, int] = (32, 32)


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
    agent: AgentConfig


def get_config() -> Config:
    arguments = parse_arguments()
    if arguments.file is not None:
        config = parse_config(arguments.file)
    else:
        config = Config(
            engine=EngineConfig(), agent=AgentConfig(ppo=PPOConfig(), icm=ICMConfig())
        )
    if arguments.key_value_pairs is not None:
        update_config(config, arguments.key_value_pairs)
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
        "-kvp",
        "--key_value_pairs",
        nargs="*",
        help="Key-value pairs to override in the configuration (nested configs can be accessed via '.')",
        required=False,
        default=None,
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
    with open(yaml_path, "r") as fp:
        config_dict = yaml.load(fp, yaml.Loader)
    config = from_dict(data_class=Config, data=config_dict)

    return config


def parse_value(value: str) -> Any:
    """Parses the value as if it was being loaded in a YAML file"""
    return yaml.load(value, Loader=yaml.SafeLoader)


def _set_value(instance: Any, keys: list[str], value: Any) -> None:

    for key in keys[:-1]:
        instance = getattr(instance, key)
        if not is_dataclass(instance):
            raise ValueError(
                f"Expected attribute '{key}' to be a dataclass instance but got '{type(key)}'"
            )

    attr = keys[-1]
    value = parse_value(value)
    old_value = getattr(instance, attr)

    # Type mismatch based on raw types or being iterables
    if (
        (
            not isinstance(value, Iterable)
            and not isinstance(old_value, Iterable)
            and type(value) != type(old_value)
        )
        or (isinstance(value, Iterable) and not isinstance(old_value, Iterable))
        or (not isinstance(value, Iterable) and isinstance(old_value, Iterable))
    ):
        raise ValueError(
            f"Expected attribute to be '{type(old_value)}' but got '{type(value)}'"
        )

    # Need to handle special cases of tuples
    if isinstance(old_value, tuple):
        setattr(instance, attr, tuple(value))
    else:
        setattr(instance, attr, value)


def update_config(config: Config, key_value_pairs: list[str]) -> None:
    """Updates the configuration using the command-line argumments"""

    for pair in key_value_pairs:
        path, value = pair.split("=", 1)
        keys = path.split(".")
        _set_value(config, keys, value)
