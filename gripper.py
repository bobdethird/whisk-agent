from __future__ import annotations

from typing import Literal

from sim_env import SimEnv
from so101_mujoco_utils import move_to_pose


OPEN_GRIPPER = 50.0
CLOSED_GRIPPER = -5.0
GripperCommand = Literal["open", "close"]


def gripper(env: SimEnv, command: GripperCommand, duration: float = 0.5) -> dict[str, float]:
    if env.viewer is None:
        raise RuntimeError("gripper() requires an active MuJoCo viewer on env.viewer.")
    if command not in {"open", "close"}:
        raise ValueError("gripper command must be either 'open' or 'close'.")

    target_position = dict(env.current_position)
    target_position["gripper"] = OPEN_GRIPPER if command == "open" else CLOSED_GRIPPER
    print(f"gripper: {command}")
    move_to_pose(env.model, env.data, env.viewer, target_position, duration=duration)
    env.current_position = target_position
    return target_position
