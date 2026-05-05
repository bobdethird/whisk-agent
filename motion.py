from __future__ import annotations

from dataclasses import dataclass

import mujoco  # type: ignore[import-not-found]
import numpy as np  # type: ignore[import-not-found]

from sim_env import HORIZONTAL_WRIST_ROLL_DEGREES, SimEnv
from so101_kinematics import pose_from_position_rotation, rotation_error_rad
from so101_mujoco_utils import move_to_pose


DEFAULT_POSITION_WEIGHT = 1.0
DEFAULT_ORIENTATION_WEIGHT = 0.01
DEFAULT_MAX_ITERATIONS = 100


@dataclass(frozen=True)
class IKPlan:
    target_pose: np.ndarray
    target_position: dict[str, float]
    position_error: float
    orientation_error: float


def _target_xyz(xyz: np.ndarray | tuple[float, float, float] | list[float]) -> np.ndarray:
    target = np.asarray(xyz, dtype=float)
    if target.shape != (3,):
        raise ValueError(f"Expected xyz to contain exactly 3 values, got shape {target.shape}.")
    return target


def solve_ik(env: SimEnv, xyz: np.ndarray | tuple[float, float, float] | list[float], gripper_position: float | None = None) -> IKPlan:
    target = _target_xyz(xyz)
    current_position = dict(env.current_position)
    current_position["wrist_roll"] = HORIZONTAL_WRIST_ROLL_DEGREES
    current_pose = env.kinematics.forward_kinematics(current_position, frame="mujoco")
    target_pose = pose_from_position_rotation(target, current_pose[:3, :3])
    target_position = env.kinematics.inverse_kinematics(
        current_position,
        target_pose,
        position_weight=DEFAULT_POSITION_WEIGHT,
        orientation_weight=DEFAULT_ORIENTATION_WEIGHT,
        gripper=current_position["gripper"] if gripper_position is None else float(gripper_position),
        max_iterations=DEFAULT_MAX_ITERATIONS,
    )
    target_position["wrist_roll"] = HORIZONTAL_WRIST_ROLL_DEGREES
    solved_pose = env.kinematics.forward_kinematics(target_position, frame="mujoco")
    return IKPlan(
        target_pose=target_pose,
        target_position=target_position,
        position_error=float(np.linalg.norm(target_pose[:3, 3] - solved_pose[:3, 3])),
        orientation_error=float(rotation_error_rad(target_pose[:3, :3], solved_pose[:3, :3])),
    )


def show_target(env: SimEnv, target_pose: np.ndarray) -> None:
    if env.viewer is None:
        return

    mujoco.mjv_initGeom(
        env.viewer.user_scn.geoms[0],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.012, 0.0, 0.0],
        pos=target_pose[:3, 3],
        mat=np.eye(3).flatten(),
        rgba=[0.0, 1.0, 0.0, 0.45],
    )
    env.viewer.user_scn.ngeom = 1
    env.viewer.sync()


def move(
    env: SimEnv,
    xyz: np.ndarray | tuple[float, float, float] | list[float],
    gripper_position: float | None = None,
    duration: float = 2.0,
    show_marker: bool = True,
) -> IKPlan:
    if env.viewer is None:
        raise RuntimeError("move() requires an active MuJoCo viewer on env.viewer.")

    plan = solve_ik(env, xyz, gripper_position=gripper_position)
    target = plan.target_pose[:3, 3]
    print(
        "move: "
        f"x={target[0]:.4f} y={target[1]:.4f} z={target[2]:.4f} m, "
        f"IK error={plan.position_error:.6f} m"
    )
    if show_marker:
        show_target(env, plan.target_pose)
    move_to_pose(env.model, env.data, env.viewer, plan.target_position, duration=duration)
    env.current_position = dict(plan.target_position)
    return plan
