"""Shared SO-101 AprilTag motion planning and execution helpers.

This module keeps perception separate from motion. Camera scripts should
produce a latched ``T_base_tag``; this module turns that into offset-correct
``gripper_frame_link`` waypoints, solves IK iteratively, and optionally streams
those waypoints to the follower.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable, Sequence

import numpy as np


DEFAULT_MOVE_DURATION_S = 2.0
DEFAULT_MOVE_RATE_HZ = 30.0
DEFAULT_ORIENTATION_WEIGHT = 0.2
DEFAULT_SAFE_Z_M = 0.18
DEFAULT_POSITION_PRIORITY_MM = 3.0
IK_MAX_ITERS = 50
IK_CONVERGENCE_TOL_MM = 0.5


@dataclass(frozen=True)
class MotionConfig:
    fk_joint_names: Sequence[str]
    gripper_obs_key: str
    workspace_min: np.ndarray
    workspace_max: np.ndarray
    position_weight: float
    orientation_weight: float
    max_residual_mm: float
    move_duration_s: float = DEFAULT_MOVE_DURATION_S
    move_rate_hz: float = DEFAULT_MOVE_RATE_HZ
    fallback_residual_mm: float = DEFAULT_POSITION_PRIORITY_MM
    fallback_orientation_weight: float = 0.0
    safe_z_m: float = DEFAULT_SAFE_Z_M
    best_effort: bool = True
    dry_run: bool = False
    label: str = "move"


@dataclass(frozen=True)
class IKSolution:
    joints: np.ndarray
    residual_mm: float
    iters_used: int
    orientation_weight: float
    axis_error_deg: float | None
    target_pose: np.ndarray
    solved_pose: np.ndarray


def format_xyz_mm(xyz: np.ndarray) -> str:
    mm = np.asarray(xyz, dtype=np.float64).reshape(3) * 1000.0
    return f"({mm[0]:+7.1f}, {mm[1]:+7.1f}, {mm[2]:+7.1f}) mm"


def invert_transform(T: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def unit_vector(v: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(v))
    if norm < 1e-9:
        return np.asarray(fallback, dtype=np.float64).reshape(3)
    return v / norm


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    a_u = unit_vector(a, np.array([0.0, 0.0, 1.0]))
    b_u = unit_vector(b, np.array([0.0, 0.0, 1.0]))
    dot = float(np.clip(np.dot(a_u, b_u), -1.0, 1.0))
    return float(np.degrees(np.arccos(dot)))


def in_workspace(
    xyz: np.ndarray,
    workspace_min: np.ndarray,
    workspace_max: np.ndarray,
) -> bool:
    xyz = np.asarray(xyz, dtype=np.float64).reshape(3)
    return bool(np.all(xyz >= workspace_min) and np.all(xyz <= workspace_max))


def clamp_xyz_to_workspace(
    xyz: np.ndarray,
    workspace_min: np.ndarray,
    workspace_max: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Clamp an xyz point into the workspace box and return clipped distance."""
    xyz = np.asarray(xyz, dtype=np.float64).reshape(3)
    clipped = np.clip(xyz, workspace_min, workspace_max)
    delta_mm = float(np.linalg.norm(clipped - xyz) * 1000.0)
    return clipped, delta_mm


def build_tag_aligned_pose(
    T_base_flange_current: np.ndarray,
    T_base_tag: np.ndarray,
    hover_z_m: float,
    tag_z_sign: float,
    tool_offset_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return desired ``T_base_gripper_frame`` and the target tag-Z direction.

    ``tool_offset_m`` is the vector from ``gripper_frame_link`` origin to the
    desired contact/aim point, expressed in ``gripper_frame_link`` coordinates.
    A nonzero offset makes IK target the frame pose whose tool point, not frame
    origin, lands at the tag-centered hover/contact point.
    """
    current_R = T_base_flange_current[:3, :3]
    target_tag_z = unit_vector(tag_z_sign * T_base_tag[:3, 2], current_R[:, 2])

    x_hint = current_R[:, 0]
    x_axis = x_hint - np.dot(x_hint, target_tag_z) * target_tag_z
    if np.linalg.norm(x_axis) < 1e-6:
        y_hint = current_R[:, 1]
        x_axis = np.cross(y_hint, target_tag_z)
    x_axis = unit_vector(x_axis, current_R[:, 0])
    y_axis = unit_vector(np.cross(target_tag_z, x_axis), current_R[:, 1])
    x_axis = unit_vector(np.cross(y_axis, target_tag_z), current_R[:, 0])

    T_base_contact = np.eye(4, dtype=np.float64)
    T_base_contact[:3, :3] = np.column_stack([x_axis, y_axis, target_tag_z])
    T_base_contact[:3, 3] = T_base_tag[:3, 3].copy()
    T_base_contact[2, 3] += hover_z_m

    T_gripper_contact = np.eye(4, dtype=np.float64)
    T_gripper_contact[:3, 3] = np.asarray(tool_offset_m, dtype=np.float64).reshape(3)
    T_base_gripper = T_base_contact @ invert_transform(T_gripper_contact)
    return T_base_gripper, target_tag_z


def transform_point(T: np.ndarray, point: np.ndarray) -> np.ndarray:
    """Transform a 3D point by a homogeneous transform."""
    point = np.asarray(point, dtype=np.float64).reshape(3)
    return T[:3, :3] @ point + T[:3, 3]


def make_z_ceiling_waypoints(
    T_base_flange_current: np.ndarray,
    T_base_target: np.ndarray,
    safe_z_m: float,
) -> list[tuple[str, np.ndarray]]:
    """Build lift, translate, descend waypoints in this repo's z-up frame."""
    current = np.asarray(T_base_flange_current, dtype=np.float64)
    target = np.asarray(T_base_target, dtype=np.float64)
    ceiling_z = float(max(safe_z_m, current[2, 3], target[2, 3]))

    lift = target.copy()
    lift[:3, 3] = current[:3, 3].copy()
    lift[2, 3] = ceiling_z

    translate = target.copy()
    translate[2, 3] = ceiling_z

    return [
        ("lift", lift),
        ("translate", translate),
        ("descend", target),
    ]


def converged_inverse_kinematics(
    kinematics,
    current_joint_pos: np.ndarray,
    desired_ee_pose: np.ndarray,
    fk_joint_count: int,
    position_weight: float,
    orientation_weight: float,
    target_axis: np.ndarray | None = None,
    max_iters: int = IK_MAX_ITERS,
    tol_mm: float = IK_CONVERGENCE_TOL_MM,
) -> IKSolution:
    """Iterate placo's single-step IK until the position residual converges."""
    q = np.asarray(current_joint_pos, dtype=np.float64).copy()
    best_q = q[:fk_joint_count].copy()
    best_res_mm = float("inf")
    best_T = np.asarray(
        kinematics.forward_kinematics(best_q), dtype=np.float64
    )
    iters_used = 0
    target_xyz = desired_ee_pose[:3, 3]

    for i in range(max_iters):
        q = np.asarray(
            kinematics.inverse_kinematics(
                current_joint_pos=q,
                desired_ee_pose=desired_ee_pose,
                position_weight=position_weight,
                orientation_weight=orientation_weight,
            ),
            dtype=np.float64,
        )
        if q.shape[0] < fk_joint_count or not np.all(np.isfinite(q[:fk_joint_count])):
            break

        T = np.asarray(
            kinematics.forward_kinematics(q[:fk_joint_count]), dtype=np.float64
        )
        res_mm = float(np.linalg.norm(T[:3, 3] - target_xyz) * 1000.0)
        iters_used = i + 1
        if res_mm < best_res_mm:
            best_res_mm = res_mm
            best_q = q[:fk_joint_count].copy()
            best_T = T
        if res_mm < tol_mm:
            break

    axis_error_deg = None
    if target_axis is not None:
        axis_error_deg = angle_deg(best_T[:3, 2], target_axis)

    return IKSolution(
        joints=best_q,
        residual_mm=best_res_mm,
        iters_used=iters_used,
        orientation_weight=orientation_weight,
        axis_error_deg=axis_error_deg,
        target_pose=desired_ee_pose.copy(),
        solved_pose=best_T,
    )


def solve_waypoint_with_fallback(
    kinematics,
    current_joint_pos: np.ndarray,
    desired_ee_pose: np.ndarray,
    config: MotionConfig,
    target_axis: np.ndarray | None,
) -> IKSolution:
    n = len(config.fk_joint_names)
    primary = converged_inverse_kinematics(
        kinematics=kinematics,
        current_joint_pos=current_joint_pos,
        desired_ee_pose=desired_ee_pose,
        fk_joint_count=n,
        position_weight=config.position_weight,
        orientation_weight=config.orientation_weight,
        target_axis=target_axis,
    )
    if primary.residual_mm <= config.fallback_residual_mm:
        return primary
    if config.fallback_orientation_weight == config.orientation_weight:
        return primary

    fallback = converged_inverse_kinematics(
        kinematics=kinematics,
        current_joint_pos=current_joint_pos,
        desired_ee_pose=desired_ee_pose,
        fk_joint_count=n,
        position_weight=config.position_weight,
        orientation_weight=config.fallback_orientation_weight,
        target_axis=target_axis,
    )
    if fallback.residual_mm < primary.residual_mm:
        return fallback
    return primary


def smooth_send_action(
    robot,
    current_joints_deg: np.ndarray,
    target_joints_deg: np.ndarray,
    gripper: float,
    config: MotionConfig,
    keepalive: Callable[[], None] | None = None,
) -> None:
    """Stream a smoothstep-interpolated joint-space segment."""
    if config.move_duration_s <= 0.0 or config.move_rate_hz <= 0.0:
        action = {
            f"{name}.pos": float(val)
            for name, val in zip(config.fk_joint_names, target_joints_deg)
        }
        action[config.gripper_obs_key] = gripper
        robot.send_action(action)
        return

    num_steps = max(2, int(round(config.move_duration_s * config.move_rate_hz)))
    dt = config.move_duration_s / num_steps
    delta = target_joints_deg - current_joints_deg
    start = time.monotonic()
    for step in range(1, num_steps + 1):
        t = step / num_steps
        alpha = 3.0 * t * t - 2.0 * t * t * t
        q = current_joints_deg + alpha * delta
        action = {
            f"{name}.pos": float(val)
            for name, val in zip(config.fk_joint_names, q)
        }
        action[config.gripper_obs_key] = gripper
        robot.send_action(action)
        if keepalive is not None:
            keepalive()
        next_t = start + step * dt
        slack = next_t - time.monotonic()
        if slack > 0.0:
            time.sleep(slack)


def solve_and_execute_tag_waypoints(
    *,
    kinematics,
    robot,
    current_joints_deg: np.ndarray,
    gripper: float,
    T_base_tag: np.ndarray,
    hover_z_m: float,
    tag_z_sign: float,
    tool_offset_m: np.ndarray,
    config: MotionConfig,
    keepalive: Callable[[], None] | None = None,
) -> bool:
    """Plan, solve, and optionally execute a z-ceiling trajectory to a tag."""
    label = config.label
    current_joints_deg = np.asarray(current_joints_deg, dtype=np.float64)
    T_current = np.asarray(
        kinematics.forward_kinematics(current_joints_deg), dtype=np.float64
    )
    target, target_axis = build_tag_aligned_pose(
        T_base_flange_current=T_current,
        T_base_tag=T_base_tag,
        hover_z_m=hover_z_m,
        tag_z_sign=tag_z_sign,
        tool_offset_m=tool_offset_m,
    )
    unclamped_target_xyz = target[:3, 3].copy()
    clamped_target_xyz, target_clip_mm = clamp_xyz_to_workspace(
        unclamped_target_xyz,
        config.workspace_min,
        config.workspace_max,
    )
    target[:3, 3] = clamped_target_xyz
    print(f"[{label}] current flange @ {format_xyz_mm(T_current[:3, 3])}")
    print(f"[{label}] raw tag center @ {format_xyz_mm(T_base_tag[:3, 3])}")
    print(
        f"[{label}] motion settings: hover={hover_z_m * 1000.0:+.1f} mm, "
        f"safe_z={config.safe_z_m * 1000.0:+.1f} mm, "
        f"orientation_weight={config.orientation_weight:.3f}, "
        f"fallback_orientation_weight={config.fallback_orientation_weight:.3f}, "
        f"position_priority={config.fallback_residual_mm:.1f} mm, "
        f"max_residual={config.max_residual_mm:.1f} mm, "
        f"best_effort={config.best_effort}"
    )
    print(
        f"[{label}] workspace clamp: min={format_xyz_mm(config.workspace_min)} "
        f"max={format_xyz_mm(config.workspace_max)}"
    )
    print(
        f"[{label}] tool offset gripper->contact = "
        f"{format_xyz_mm(np.asarray(tool_offset_m, dtype=np.float64))}"
    )
    if target_clip_mm > 0.0:
        print(
            f"[{label}] final IK frame target {format_xyz_mm(unclamped_target_xyz)} "
            f"is outside workspace; clamped to nearest reachable box point "
            f"{format_xyz_mm(target[:3, 3])} (clip {target_clip_mm:.1f} mm)"
        )
    print(f"[{label}] final IK frame target @ {format_xyz_mm(target[:3, 3])}")

    waypoints = make_z_ceiling_waypoints(T_current, target, config.safe_z_m)
    q_start = current_joints_deg.copy()
    solutions: list[tuple[str, IKSolution]] = []

    for name, desired in waypoints:
        xyz = desired[:3, 3]
        if not in_workspace(xyz, config.workspace_min, config.workspace_max):
            clamped_xyz, clip_mm = clamp_xyz_to_workspace(
                xyz,
                config.workspace_min,
                config.workspace_max,
            )
            print(
                f"[{label}] waypoint {name!r} {format_xyz_mm(xyz)} outside "
                f"workspace; clamped to {format_xyz_mm(clamped_xyz)} "
                f"(clip {clip_mm:.1f} mm)"
            )
            desired = desired.copy()
            desired[:3, 3] = clamped_xyz
            xyz = desired[:3, 3]

        sol = solve_waypoint_with_fallback(
            kinematics=kinematics,
            current_joint_pos=q_start,
            desired_ee_pose=desired,
            config=config,
            target_axis=target_axis,
        )
        axis_msg = ""
        if sol.axis_error_deg is not None:
            axis_msg = f"; +Z/tag-Z angle {sol.axis_error_deg:.1f} deg"
        fallback_msg = ""
        if sol.orientation_weight != config.orientation_weight:
            fallback_msg = (
                f" (fallback orientation_weight={sol.orientation_weight:.3f})"
            )
        print(
            f"[{label}] waypoint {name}: target {format_xyz_mm(xyz)} -> "
            f"residual {sol.residual_mm:.1f} mm in {sol.iters_used} iter(s)"
            f"{axis_msg}{fallback_msg}"
        )
        if sol.residual_mm > config.max_residual_mm:
            radius_mm = float(np.linalg.norm(xyz[:2]) * 1000.0)
            hint = ""
            if radius_mm > 350.0:
                hint = (
                    f" hint: horizontal radius {radius_mm:.1f} mm is near "
                    "the SO-101 reach limit; move the tag closer or lower "
                    "--safe-z-mm/--hover-z-mm."
                )
            if not config.best_effort or not np.isfinite(sol.residual_mm):
                print(
                    f"[{label}] residual {sol.residual_mm:.1f} mm > "
                    f"{config.max_residual_mm:.1f} mm; refusing to move.{hint}"
                )
                return False
            print(
                f"[{label}] residual {sol.residual_mm:.1f} mm > "
                f"{config.max_residual_mm:.1f} mm; continuing with closest "
                f"IK solution because best_effort=True.{hint}"
            )

        solutions.append((name, sol))
        q_start = sol.joints.copy()

    print(f"[{label}] solved {len(solutions)} waypoint(s); joint deltas:")
    prev = current_joints_deg
    for name, sol in solutions:
        print(f"[{label}]   {name}:")
        for i, joint_name in enumerate(config.fk_joint_names):
            cur = float(prev[i])
            tgt = float(sol.joints[i])
            print(
                f"    {joint_name:>14}: {cur:+7.2f} -> {tgt:+7.2f}  "
                f"(delta {tgt - cur:+6.2f})"
            )
        prev = sol.joints

    final_solution = solutions[-1][1]
    tool_offset_m = np.asarray(tool_offset_m, dtype=np.float64).reshape(3)
    solved_contact_xyz = transform_point(final_solution.solved_pose, tool_offset_m)
    target_contact_xyz = transform_point(target, tool_offset_m)
    raw_tag_hover_xyz = T_base_tag[:3, 3].copy()
    raw_tag_hover_xyz[2] += hover_z_m
    print(f"[{label}] final solved IK frame @ {format_xyz_mm(final_solution.solved_pose[:3, 3])}")
    print(
        f"[{label}] final solved configured contact @ "
        f"{format_xyz_mm(solved_contact_xyz)}"
    )
    print(
        f"[{label}] configured contact target @ {format_xyz_mm(target_contact_xyz)} "
        f"(error {np.linalg.norm(solved_contact_xyz - target_contact_xyz) * 1000.0:.1f} mm)"
    )
    print(
        f"[{label}] raw tag-center hover point @ {format_xyz_mm(raw_tag_hover_xyz)} "
        f"(before workspace clamp/tool offset)"
    )

    if config.dry_run:
        print(f"[{label}] --dry-run: skipping send_action")
        return True

    q_send = current_joints_deg.copy()
    for name, sol in solutions:
        print(
            f"[{label}] streaming waypoint {name}: "
            f"{config.move_duration_s:.2f} s @ {config.move_rate_hz:.1f} Hz"
        )
        smooth_send_action(
            robot=robot,
            current_joints_deg=q_send,
            target_joints_deg=sol.joints,
            gripper=gripper,
            config=config,
            keepalive=keepalive,
        )
        q_send = sol.joints.copy()

    print(f"[{label}] waypoint trajectory complete")
    return True
