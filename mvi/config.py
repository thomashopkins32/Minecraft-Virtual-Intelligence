from typing import Dict, Any

import yaml


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
