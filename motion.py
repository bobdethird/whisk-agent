from __future__ import annotations

from dataclasses import dataclass

import mujoco  # type: ignore[import-not-found]
import numpy as np  # type: ignore[import-not-found]
from scipy.optimize import least_squares  # type: ignore[import-untyped]

from sim_env import HORIZONTAL_WRIST_ROLL_DEGREES, SimEnv
from so101_kinematics import (
    claw_target_pose_to_gripperframe_pose,
    gripperframe_pose_to_claw_target_pose,
    pose_from_position_rotation,
    rotation_error_rad,
)
from so101_mujoco_utils import move_to_pose


DEFAULT_POSITION_WEIGHT = 1.0
DEFAULT_ORIENTATION_WEIGHT = 0.01
DEFAULT_MAX_ITERATIONS = 100
FIXED_WRIST_POSITION_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex")
REFINEMENT_MAX_EVALUATIONS = 200


@dataclass(frozen=True)
class IKPlan:
    target_pose: np.ndarray
    gripperframe_pose: np.ndarray
    target_position: dict[str, float]
    position_error: float
    orientation_error: float


def _target_xyz(xyz: np.ndarray | tuple[float, float, float] | list[float]) -> np.ndarray:
    target = np.asarray(xyz, dtype=float)
    if target.shape != (3,):
        raise ValueError(f"Expected xyz to contain exactly 3 values, got shape {target.shape}.")
    return target


def _joint_bounds_degrees(env: SimEnv, joint_names: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    lower: list[float] = []
    upper: list[float] = []
    for joint_name in joint_names:
        joint_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"Could not find MuJoCo joint named {joint_name!r}.")
        lower.append(float(np.rad2deg(env.model.jnt_range[joint_id, 0])))
        upper.append(float(np.rad2deg(env.model.jnt_range[joint_id, 1])))
    return np.array(lower, dtype=float), np.array(upper, dtype=float)


def _claw_target_pose_for_position(env: SimEnv, position: dict[str, float]) -> np.ndarray:
    gripperframe_pose = env.kinematics.forward_kinematics(position, frame="mujoco")
    return gripperframe_pose_to_claw_target_pose(gripperframe_pose)


def _refine_fixed_wrist_position(
    env: SimEnv,
    target_pose: np.ndarray,
    seed_position: dict[str, float],
) -> dict[str, float]:
    lower_bounds, upper_bounds = _joint_bounds_degrees(env, FIXED_WRIST_POSITION_JOINTS)
    candidate = dict(seed_position)
    candidate["wrist_roll"] = HORIZONTAL_WRIST_ROLL_DEGREES

    def residual(joint_values: np.ndarray) -> np.ndarray:
        trial = dict(candidate)
        for joint_name, value in zip(FIXED_WRIST_POSITION_JOINTS, joint_values):
            trial[joint_name] = float(value)
        return _claw_target_pose_for_position(env, trial)[:3, 3] - target_pose[:3, 3]

    initial_values = np.array([candidate[joint] for joint in FIXED_WRIST_POSITION_JOINTS], dtype=float)
    result = least_squares(
        residual,
        initial_values,
        bounds=(lower_bounds, upper_bounds),
        max_nfev=REFINEMENT_MAX_EVALUATIONS,
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
    )

    refined = dict(candidate)
    for joint_name, value in zip(FIXED_WRIST_POSITION_JOINTS, result.x):
        refined[joint_name] = float(value)

    current_error = np.linalg.norm(residual(initial_values))
    refined_error = np.linalg.norm(residual(result.x))
    return refined if refined_error <= current_error else candidate


def solve_ik(env: SimEnv, xyz: np.ndarray | tuple[float, float, float] | list[float], gripper_position: float | None = None) -> IKPlan:
    target = _target_xyz(xyz)
    current_position = dict(env.current_position)
    current_position["wrist_roll"] = HORIZONTAL_WRIST_ROLL_DEGREES
    current_pose = env.kinematics.forward_kinematics(current_position, frame="mujoco")
    target_pose = pose_from_position_rotation(target, current_pose[:3, :3])
    gripperframe_pose = claw_target_pose_to_gripperframe_pose(target_pose)
    target_position = env.kinematics.inverse_kinematics(
        current_position,
        gripperframe_pose,
        position_weight=DEFAULT_POSITION_WEIGHT,
        orientation_weight=DEFAULT_ORIENTATION_WEIGHT,
        gripper=current_position["gripper"] if gripper_position is None else float(gripper_position),
        max_iterations=DEFAULT_MAX_ITERATIONS,
    )
    target_position = _refine_fixed_wrist_position(env, target_pose, target_position)
    solved_gripperframe_pose = env.kinematics.forward_kinematics(target_position, frame="mujoco")
    solved_target_pose = gripperframe_pose_to_claw_target_pose(solved_gripperframe_pose)
    return IKPlan(
        target_pose=target_pose,
        gripperframe_pose=gripperframe_pose,
        target_position=target_position,
        position_error=float(np.linalg.norm(target_pose[:3, 3] - solved_target_pose[:3, 3])),
        orientation_error=float(rotation_error_rad(target_pose[:3, :3], solved_target_pose[:3, :3])),
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
