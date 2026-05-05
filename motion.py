from __future__ import annotations

from dataclasses import dataclass

import mujoco  # type: ignore[import-not-found]
import numpy as np  # type: ignore[import-not-found]
from scipy.optimize import least_squares  # type: ignore[import-untyped]

from sim_env import HORIZONTAL_WRIST_ROLL_DEGREES, SimEnv
from so101_kinematics import (
    FIXED_JAW_TOOL_POINT,
    MUJOCO_SITE_NAME,
    ToolPointName,
    claw_target_pose_to_gripperframe_pose,
    gripperframe_pose_to_claw_target_pose,
    gripperframe_pose_to_tool_target_pose,
    pose_from_position_rotation,
    rotation_error_rad,
    tool_target_pose_to_gripperframe_pose,
)
from so101_mujoco_utils import JOINT_ORDER, convert_to_list, move_to_pose


DEFAULT_POSITION_WEIGHT = 1.0
DEFAULT_ORIENTATION_WEIGHT = 0.01
# A high weight pulls the IK into different local minima for some targets;
# match the default position-mode weight and let the explicit target rotation
# act as a soft hint. Position is what matters for landing the gripper on a
# small bar.
EXPLICIT_ORIENTATION_WEIGHT = DEFAULT_ORIENTATION_WEIGHT
DEFAULT_MAX_ITERATIONS = 100
FIXED_WRIST_POSITION_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex")
REFINEMENT_MAX_EVALUATIONS = 200
TOOL_DIAGNOSTIC_GEOMS = {
    "fixed_jaw_tip": "fixed_jaw_sph_tip1",
    "moving_jaw_tip": "moving_jaw_sph_tip1",
}


@dataclass(frozen=True)
class IKPlan:
    target_pose: np.ndarray
    gripperframe_pose: np.ndarray
    target_position: dict[str, float]
    position_error: float
    orientation_error: float
    tool_point: ToolPointName


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


def _tool_target_pose_for_position(
    env: SimEnv,
    position: dict[str, float],
    tool_point: ToolPointName,
) -> np.ndarray:
    gripperframe_pose = env.kinematics.forward_kinematics(position, frame="mujoco")
    return gripperframe_pose_to_tool_target_pose(gripperframe_pose, tool_point)


def _refine_fixed_wrist_position(
    env: SimEnv,
    target_pose: np.ndarray,
    seed_position: dict[str, float],
    tool_point: ToolPointName,
) -> dict[str, float]:
    lower_bounds, upper_bounds = _joint_bounds_degrees(env, FIXED_WRIST_POSITION_JOINTS)
    candidate = dict(seed_position)
    candidate["wrist_roll"] = HORIZONTAL_WRIST_ROLL_DEGREES

    def residual(joint_values: np.ndarray) -> np.ndarray:
        trial = dict(candidate)
        for joint_name, value in zip(FIXED_WRIST_POSITION_JOINTS, joint_values):
            trial[joint_name] = float(value)
        return _tool_target_pose_for_position(env, trial, tool_point)[:3, 3] - target_pose[:3, 3]

    initial_values = np.array([candidate[joint] for joint in FIXED_WRIST_POSITION_JOINTS], dtype=float)
    initial_values = np.clip(initial_values, lower_bounds, upper_bounds)
    for joint_name, value in zip(FIXED_WRIST_POSITION_JOINTS, initial_values):
        candidate[joint_name] = float(value)

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


def solve_ik(
    env: SimEnv,
    xyz: np.ndarray | tuple[float, float, float] | list[float],
    gripper_position: float | None = None,
    *,
    rotation: np.ndarray | None = None,
    tool_point: ToolPointName = FIXED_JAW_TOOL_POINT,
) -> IKPlan:
    target = _target_xyz(xyz)
    current_position = dict(env.current_position)

    if rotation is None:
        current_position["wrist_roll"] = HORIZONTAL_WRIST_ROLL_DEGREES
        current_pose = env.kinematics.forward_kinematics(current_position, frame="mujoco")
        target_pose = pose_from_position_rotation(target, current_pose[:3, :3])
        gripperframe_pose = tool_target_pose_to_gripperframe_pose(target_pose, tool_point)
        orientation_weight = DEFAULT_ORIENTATION_WEIGHT
    else:
        target_rotation = np.asarray(rotation, dtype=float)
        if target_rotation.shape != (3, 3):
            raise ValueError(f"rotation must be a 3x3 matrix; got {target_rotation.shape}.")
        target_pose = pose_from_position_rotation(target, target_rotation)
        gripperframe_pose = claw_target_pose_to_gripperframe_pose(target_pose)
        orientation_weight = EXPLICIT_ORIENTATION_WEIGHT

    target_position = env.kinematics.inverse_kinematics(
        current_position,
        gripperframe_pose,
        position_weight=DEFAULT_POSITION_WEIGHT,
        orientation_weight=orientation_weight,
        gripper=current_position["gripper"] if gripper_position is None else float(gripper_position),
        max_iterations=DEFAULT_MAX_ITERATIONS,
    )
    if rotation is None:
        target_position = _refine_fixed_wrist_position(env, target_pose, target_position, tool_point)

    solved_gripperframe_pose = env.kinematics.forward_kinematics(target_position, frame="mujoco")
    if rotation is None:
        solved_target_pose = gripperframe_pose_to_tool_target_pose(solved_gripperframe_pose, tool_point)
    else:
        solved_target_pose = gripperframe_pose_to_claw_target_pose(solved_gripperframe_pose)

    return IKPlan(
        target_pose=target_pose,
        gripperframe_pose=gripperframe_pose,
        target_position=target_position,
        position_error=float(np.linalg.norm(target_pose[:3, 3] - solved_target_pose[:3, 3])),
        orientation_error=float(rotation_error_rad(target_pose[:3, :3], solved_target_pose[:3, :3])),
        tool_point=tool_point,
    )


def solved_tool_point_deltas(env: SimEnv, plan: IKPlan) -> dict[str, np.ndarray]:
    """Return target-minus-tool deltas for a solved plan without mutating env.data."""
    data = mujoco.MjData(env.model)
    data.qpos[:] = env.data.qpos
    data.qpos[: len(JOINT_ORDER)] = convert_to_list(plan.target_position)
    mujoco.mj_forward(env.model, data)

    target = plan.target_pose[:3, 3]
    deltas: dict[str, np.ndarray] = {}
    site_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, MUJOCO_SITE_NAME)
    if site_id >= 0:
        deltas[MUJOCO_SITE_NAME] = target - data.site_xpos[site_id].copy()

    for label, geom_name in TOOL_DIAGNOSTIC_GEOMS.items():
        geom_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if geom_id >= 0:
            deltas[label] = target - data.geom_xpos[geom_id].copy()
    return deltas


def print_tool_point_diagnostics(env: SimEnv, plan: IKPlan) -> None:
    deltas = solved_tool_point_deltas(env, plan)
    formatted = ", ".join(f"{label}=({_format_delta(delta)}) m" for label, delta in deltas.items())
    print(f"tool deltas target-minus-tool: {formatted}")


def _format_delta(delta: np.ndarray) -> str:
    return f"{delta[0]:+.4f}, {delta[1]:+.4f}, {delta[2]:+.4f}"


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
    tool_point: ToolPointName = FIXED_JAW_TOOL_POINT,
    show_tool_diagnostics: bool = False,
) -> IKPlan:
    if env.viewer is None:
        raise RuntimeError("move() requires an active MuJoCo viewer on env.viewer.")

    plan = solve_ik(env, xyz, gripper_position=gripper_position, tool_point=tool_point)
    target = plan.target_pose[:3, 3]
    print(
        "move: "
        f"x={target[0]:.4f} y={target[1]:.4f} z={target[2]:.4f} m, "
        f"IK error={plan.position_error:.6f} m, tool={tool_point}"
    )
    if show_tool_diagnostics:
        print_tool_point_diagnostics(env, plan)
    if show_marker:
        show_target(env, plan.target_pose)
    move_to_pose(env.model, env.data, env.viewer, plan.target_position, duration=duration)
    env.current_position = dict(plan.target_position)
    return plan
