import shutil

import numpy as np
import pytest

from mineagent.client.protocol import GLFW, KEY_TO_INDEX, NUM_KEYS, MSG_TYPE_ACTION
from mineagent.env import (
    MinecraftEnv,
    minecraft_launch_argv,
    minecraft_launch_env,
    validate_minecraft_launch,
)


def _action(
    *,
    keys: np.ndarray | None = None,
    mouse_dx: float = 0.0,
    mouse_dy: float = 0.0,
    mouse_buttons: np.ndarray | None = None,
    scroll_delta: float = 0.0,
) -> dict:
    return {
        "keys": np.zeros(NUM_KEYS, dtype=np.int8) if keys is None else keys,
        "mouse_dx": mouse_dx,
        "mouse_dy": mouse_dy,
        "mouse_buttons": (
            np.zeros(3, dtype=np.int8) if mouse_buttons is None else mouse_buttons
        ),
        "scroll_delta": scroll_delta,
    }


def test_action_to_message_press_hold_release():
    env = MinecraftEnv()
    keys = np.zeros(NUM_KEYS, dtype=np.int8)
    keys[KEY_TO_INDEX[GLFW.KEY_W]] = 1

    press = env._action_to_message(_action(keys=keys))
    assert press.msg_type == MSG_TYPE_ACTION
    assert press.key_press == [GLFW.KEY_W]
    assert press.key_release == []

    hold = env._action_to_message(_action(keys=keys.copy()))
    assert hold.key_press == []
    assert hold.key_release == []

    release = env._action_to_message(_action())
    assert release.key_press == []
    assert release.key_release == [GLFW.KEY_W]


def test_action_to_message_mouse_and_buttons():
    env = MinecraftEnv()
    msg = env._action_to_message(
        _action(
            mouse_dx=1.5,
            mouse_buttons=np.array([1, 0, 0], dtype=np.int8),
            scroll_delta=-1.0,
        )
    )
    assert msg.has_mouse is True
    assert msg.mouse_dx == 1.5
    assert msg.has_buttons is True
    assert msg.button_press == 0b001
    assert msg.has_scroll is True
    assert msg.scroll == -1.0

    hold_click = env._action_to_message(
        _action(mouse_buttons=np.array([1, 0, 0], dtype=np.int8))
    )
    assert hold_click.has_buttons is False
    assert hold_click.has_mouse is False
    assert hold_click.has_scroll is False


def test_minecraft_launch_argv_headed():
    assert minecraft_launch_argv(headless=False) == ["gradle", "runClient"]


def test_minecraft_launch_argv_headless():
    assert minecraft_launch_argv(headless=True) == [
        "xvfb-run",
        "-a",
        "-s",
        "-screen 0 1920x1080x24",
        "gradle",
        "runClient",
    ]


def test_minecraft_launch_env_software_gl_off(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("LIBGL_ALWAYS_SOFTWARE", raising=False)
    env = minecraft_launch_env(software_gl=False)
    assert "LIBGL_ALWAYS_SOFTWARE" not in env


def test_minecraft_launch_env_software_gl_on(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("LIBGL_ALWAYS_SOFTWARE", raising=False)
    env = minecraft_launch_env(software_gl=True)
    assert env["LIBGL_ALWAYS_SOFTWARE"] == "1"


def test_validate_headless_requires_xvfb_run(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match="xvfb-run on PATH"):
        validate_minecraft_launch(headless=True)


def test_validate_headed_requires_display(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    with pytest.raises(RuntimeError, match="No DISPLAY or WAYLAND_DISPLAY"):
        validate_minecraft_launch(headless=False)


def test_validate_headed_with_display(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    validate_minecraft_launch(headless=False)
