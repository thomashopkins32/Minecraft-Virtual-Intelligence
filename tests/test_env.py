import numpy as np

from mineagent.client.protocol import GLFW, KEY_TO_INDEX, NUM_KEYS, MSG_TYPE_ACTION
from mineagent.env import MinecraftEnv


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
