import logging
import argparse
from typing import Dict, Any

import yaml


def get_config() -> Dict[str, Any]:
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
    with open(yaml_path, "r") as fp:
        config = yaml.safe_load(fp)

    return config


def parse_value(value: Any) -> Any:
    """Parses the value as if it was being loaded in a YAML file"""
    return yaml.load(value, Loader=yaml.SafeLoader)


def update_config(config: Dict[str, Any], key_value_pairs: list[str]) -> Dict[str, Any]:
    """ "Updates the configuration using the command-line argumments"""

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
