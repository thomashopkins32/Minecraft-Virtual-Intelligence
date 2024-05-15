import sys
import argparse

import minedojo  # type: ignore

from mvi.agent.agent import AgentV1
from mvi.learning.ppo import PPO
from mvi.config import parse_config


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        dscirption="Specify arguments for running Minecraft Virtual Intelligence"
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


def update_config(config, arguments):
    """"Updates the configuration using the command-line argumments"""
    # TODO
    return config


def run() -> None:
    """
    Entry-point for the project.

    Runs the Minecraft simulation with the virtual intelligence in it.
    """
    arguments = parse_arguments()
    config = parse_config(arguments.file)
    update_config(config, arguments)

    env = minedojo.make(task_id="open-ended", image_size=(160, 256))
    agent = AgentV1(env.action_space)
    ppo = PPO(env, agent, **config["PPO"])
    ppo.run()


if __name__ == "__main__":
    run()
