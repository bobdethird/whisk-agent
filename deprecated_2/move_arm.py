"""Move the SO101 arm's claw tip to an absolute world-space (x, y, z) point.

Uses lerobot's placo-based inverse kinematics, then smoothly interpolates the
joint positions from the current configuration to the IK solution.

World space here is the URDF's `base_link` frame:
    +X forward, +Y left, +Z up, all in meters.

The target point (x, y, z) is the position of the *furthest point of the claw
gripper* (the very tip of the jaws), not the URDF's `gripper_frame_link`.
The constant `TIP_OFFSET` defines the offset between them.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import time
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from lerobot.model.kinematics import RobotKinematics
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig


URDF_PATH = Path(__file__).parent / "SO101" / "so101_new_calib.urdf"
PORT = "/dev/tty.usbmodem5AE60557941"
ROBOT_ID = "follower-1"

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]

DOWN_ORIENTATION = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ]
)

TIP_OFFSET = np.array([0.0, 0.0, 0.001])

GRIPPER_LOCAL_Z = np.array([0.0, 0.0, 1.0])
WORLD_DOWN = np.array([0.0, 0.0, -1.0])

WRIST_ROLL_CENTERING_WEIGHT = 1e-3
IK_ITERS = 80
IK_CONVERGENCE_TOL_MM = 0.5
MAX_IK_RESIDUAL_MM = 15.0
JOINT_LIMIT_MARGIN_DEG = 1.0

# Largest single-joint move (deg) we'll command from the current physical pose.
# Larger jumps usually mean the QP picked a different IK branch from a
# fall-back seed; commanding them produces the wild swings the user reports.
MAX_JOINT_STEP_DEG = 90.0

# Vertical mode pairs a hard position task with a soft "point straight down"
# axis-align task. AXISALIGN_VERTICAL_WEIGHT << position weight (1.0) so
# position dominates if both can't be satisfied; AXIS_ALIGNMENT_TOL_DEG is
# how far the actual gripper Z is allowed to drift from world-down before
# we flag the solution as a bad vertical pose.
AXISALIGN_VERTICAL_WEIGHT = 0.05
AXIS_ALIGNMENT_TOL_DEG = 15.0

# Coarse base-frame guardrail. This is intentionally a little larger than the
# useful tabletop workspace; FK residual validation below is the exact check.
WORKSPACE_MIN = np.array([-0.40, -0.45, 0.00], dtype=float)
WORKSPACE_MAX = np.array([0.48, 0.45, 0.55], dtype=float)

# Extra seeds make the solve deterministic when the current pose is a poor
# local starting point. These are IK seeds only; the robot is never commanded
# to them unless the final FK residual validates.
FALLBACK_IK_SEEDS_DEG = (
    ("zero", np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)),
    ("elbow_up", np.array([0.0, -35.0, 70.0, -35.0, 0.0], dtype=float)),
    ("elbow_down", np.array([0.0, 35.0, -70.0, 35.0, 0.0], dtype=float)),
)


class MoveArmError(RuntimeError):
    """Raised before motion when IK/reachability validation fails."""


@dataclass(frozen=True)
class IKSolution:
    seed_name: str
    joints_deg: np.ndarray
    residual_mm: float
    iters_used: int
    target_frame_pos: np.ndarray
    solved_frame_pose: np.ndarray
    solved_tip: np.ndarray
    joint_limit_violations: tuple[str, ...]
    # Largest |solution - current| over all arm joints, in degrees, along with
    # the joint name that hit it. Used both to score candidates and to reject
    # solutions that would slam the motors through a different IK branch.
    max_joint_step_deg: float
    max_joint_step_joint: str
    # Angle (deg) between the gripper's local +Z axis and world-down. Only
    # populated when vertical=True and orientation is None; otherwise None.
    axis_alignment_deg: float | None

    @property
    def valid(self) -> bool:
        if not (
            np.all(np.isfinite(self.joints_deg))
            and np.isfinite(self.residual_mm)
        ):
            return False
        if self.joint_limit_violations:
            return False
        if (
            np.isfinite(self.max_joint_step_deg)
            and self.max_joint_step_deg > MAX_JOINT_STEP_DEG
        ):
            return False
        if (
            self.axis_alignment_deg is not None
            and np.isfinite(self.axis_alignment_deg)
            and self.axis_alignment_deg > AXIS_ALIGNMENT_TOL_DEG
        ):
            return False
        return True


def _build_pose(x: float, y: float, z: float, rotation: np.ndarray) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = [x, y, z]
    return pose


def _format_xyz_mm(xyz: np.ndarray) -> str:
    mm = np.asarray(xyz, dtype=float).reshape(3) * 1000.0
    return f"({mm[0]:+7.1f}, {mm[1]:+7.1f}, {mm[2]:+7.1f}) mm"


def _read_arm_and_gripper_deg(robot: SO101Follower) -> tuple[np.ndarray, float]:
    obs = robot.get_observation()
    arm_q = np.array([float(obs[f"{m}.pos"]) for m in ARM_JOINTS], dtype=float)
    gripper = float(obs.get("gripper.pos", 0.0))
    return arm_q, gripper


def _load_joint_limits_deg() -> dict[str, tuple[float, float]]:
    root = ET.parse(URDF_PATH).getroot()
    limits: dict[str, tuple[float, float]] = {}
    for joint in root.findall("joint"):
        name = joint.attrib.get("name")
        if name not in ARM_JOINTS:
            continue
        limit = joint.find("limit")
        if limit is None:
            continue
        limits[name] = (
            float(np.degrees(float(limit.attrib["lower"]))),
            float(np.degrees(float(limit.attrib["upper"]))),
        )
    missing = [name for name in ARM_JOINTS if name not in limits]
    if missing:
        raise MoveArmError(f"URDF is missing joint limits for: {', '.join(missing)}")
    return limits


def _joint_limit_violations(
    q: np.ndarray,
    limits: dict[str, tuple[float, float]],
    margin_deg: float = JOINT_LIMIT_MARGIN_DEG,
) -> tuple[str, ...]:
    violations: list[str] = []
    for i, name in enumerate(ARM_JOINTS):
        lo, hi = limits[name]
        val = float(q[i])
        if not np.isfinite(val):
            violations.append(f"{name}=non-finite")
        elif val < lo - margin_deg or val > hi + margin_deg:
            violations.append(f"{name}={val:+.1f} deg outside [{lo:+.1f}, {hi:+.1f}]")
    return tuple(violations)


def _tip_from_frame_pose(frame_pose: np.ndarray) -> np.ndarray:
    return frame_pose[:3, 3] + frame_pose[:3, :3] @ TIP_OFFSET


def _axis_alignment_deg(frame_pose: np.ndarray) -> float:
    """Angle (deg) between the gripper's local +Z (FK rotation column 2) and world-down."""
    gripper_z_world = np.asarray(frame_pose[:3, :3], dtype=float)[:, 2]
    cos_angle = float(np.dot(gripper_z_world, WORLD_DOWN))
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return float(np.degrees(np.arccos(cos_angle)))


def _max_joint_step(q: np.ndarray, current_q_arm: np.ndarray) -> tuple[float, str]:
    """Largest |q - current| over the arm joints, in degrees, with the joint name."""
    q = np.asarray(q, dtype=float)
    current = np.asarray(current_q_arm, dtype=float)
    if not (np.all(np.isfinite(q)) and np.all(np.isfinite(current))):
        return float("inf"), ARM_JOINTS[0]
    diff = np.abs(q - current)
    idx = int(np.argmax(diff))
    return float(diff[idx]), ARM_JOINTS[idx]


def _build_ik_solution(
    *,
    seed_name: str,
    best_q: np.ndarray,
    best_residual_mm: float,
    iters_used: int,
    best_target_frame_pos: np.ndarray,
    best_pose: np.ndarray,
    joint_limits: dict[str, tuple[float, float]],
    current_q_arm: np.ndarray,
    track_axis_alignment: bool,
) -> IKSolution:
    """Bundle the per-iteration best results into an `IKSolution`.

    `track_axis_alignment` should be True only for the vertical+free-orientation
    branch, where the soft axis-align task is the only thing constraining
    the gripper's rotation. In that mode we record how far the gripper drifted
    from world-down so the validator can flag tilted grasps.
    """
    max_step_deg, max_step_joint = _max_joint_step(best_q, current_q_arm)
    axis_alignment_deg = _axis_alignment_deg(best_pose) if track_axis_alignment else None
    return IKSolution(
        seed_name=seed_name,
        joints_deg=best_q,
        residual_mm=best_residual_mm,
        iters_used=iters_used,
        target_frame_pos=best_target_frame_pos,
        solved_frame_pose=best_pose,
        solved_tip=_tip_from_frame_pose(best_pose),
        joint_limit_violations=_joint_limit_violations(best_q, joint_limits),
        max_joint_step_deg=max_step_deg,
        max_joint_step_joint=max_step_joint,
        axis_alignment_deg=axis_alignment_deg,
    )


def _assert_target_in_workspace(target_tip: np.ndarray) -> None:
    if not np.all(np.isfinite(target_tip)):
        raise MoveArmError(f"target contains non-finite values: {target_tip}")
    if np.all(target_tip >= WORKSPACE_MIN) and np.all(target_tip <= WORKSPACE_MAX):
        return
    clipped = np.clip(target_tip, WORKSPACE_MIN, WORKSPACE_MAX)
    dist_mm = float(np.linalg.norm(clipped - target_tip) * 1000.0)
    raise MoveArmError(
        "target tip is outside the coarse SO101 workspace: "
        f"target={_format_xyz_mm(target_tip)}, "
        f"nearest={_format_xyz_mm(clipped)}, outside_by={dist_mm:.1f} mm"
    )


def _coerce_arm_joints(q: np.ndarray) -> np.ndarray | None:
    q = np.asarray(q, dtype=float).ravel()
    if q.shape[0] < len(ARM_JOINTS):
        return None
    return q[: len(ARM_JOINTS)].copy()


def _inverse_kinematics_step(
    kinematics: RobotKinematics,
    q: np.ndarray,
    target_pose: np.ndarray,
    position_weight: float,
    orientation_weight: float,
) -> np.ndarray | None:
    try:
        return _coerce_arm_joints(kinematics.inverse_kinematics(
            q,
            target_pose,
            position_weight=position_weight,
            orientation_weight=orientation_weight,
        ))
    except RuntimeError:
        return None


def _solve_ik_from_seed(
    kinematics: RobotKinematics,
    axisalign_task,
    seed_name: str,
    seed_q: np.ndarray,
    target_tip: np.ndarray,
    orientation: np.ndarray | None,
    vertical: bool,
    joint_limits: dict[str, tuple[float, float]],
    current_q_arm: np.ndarray,
    iters: int = IK_ITERS,
) -> IKSolution:
    q = seed_q.copy()
    best_q = q.copy()
    best_residual_mm = float("inf")
    best_target_frame_pos = np.full(3, np.nan, dtype=float)
    best_pose = np.eye(4, dtype=float)
    iters_used = 0
    track_axis_alignment = vertical and orientation is None

    if vertical and orientation is None:
        axisalign_task.configure("vertical_axis", "soft", AXISALIGN_VERTICAL_WEIGHT)
        gripper_frame_pos = target_tip + np.array([0.0, 0.0, TIP_OFFSET[2]])
        target_pose = _build_pose(*gripper_frame_pos, np.eye(3))
        for _ in range(iters):
            next_q = _inverse_kinematics_step(kinematics, q, target_pose, 1.0, 0.0)
            if next_q is None or not np.all(np.isfinite(next_q)):
                break
            q = next_q
            pose = np.asarray(
                kinematics.forward_kinematics(q[: len(ARM_JOINTS)]),
                dtype=float,
            )
            residual_mm = float(np.linalg.norm(_tip_from_frame_pose(pose) - target_tip) * 1000.0)
            iters_used += 1
            if residual_mm < best_residual_mm:
                best_residual_mm = residual_mm
                best_q = q.copy()
                best_target_frame_pos = gripper_frame_pos.copy()
                best_pose = pose
            if residual_mm <= IK_CONVERGENCE_TOL_MM:
                break
        return _build_ik_solution(
            seed_name=seed_name,
            best_q=best_q,
            best_residual_mm=best_residual_mm,
            iters_used=iters_used,
            best_target_frame_pos=best_target_frame_pos,
            best_pose=best_pose,
            joint_limits=joint_limits,
            current_q_arm=current_q_arm,
            track_axis_alignment=track_axis_alignment,
        )

    axisalign_task.configure("vertical_axis", "soft", 0.0)

    if orientation is None:
        rotation = np.asarray(
            kinematics.forward_kinematics(q[: len(ARM_JOINTS)]),
            dtype=float,
        )[:3, :3]
        for _ in range(iters):
            gripper_frame_pos = target_tip - rotation @ TIP_OFFSET
            target_pose = _build_pose(*gripper_frame_pos, rotation)
            next_q = _inverse_kinematics_step(kinematics, q, target_pose, 1.0, 0.0)
            if next_q is None or not np.all(np.isfinite(next_q)):
                break
            q = next_q
            rotation = np.asarray(
                kinematics.forward_kinematics(q[: len(ARM_JOINTS)]),
                dtype=float,
            )[:3, :3]
            pose = np.asarray(
                kinematics.forward_kinematics(q[: len(ARM_JOINTS)]),
                dtype=float,
            )
            residual_mm = float(np.linalg.norm(_tip_from_frame_pose(pose) - target_tip) * 1000.0)
            iters_used += 1
            if residual_mm < best_residual_mm:
                best_residual_mm = residual_mm
                best_q = q.copy()
                best_target_frame_pos = gripper_frame_pos.copy()
                best_pose = pose
            if residual_mm <= IK_CONVERGENCE_TOL_MM:
                break
        return _build_ik_solution(
            seed_name=seed_name,
            best_q=best_q,
            best_residual_mm=best_residual_mm,
            iters_used=iters_used,
            best_target_frame_pos=best_target_frame_pos,
            best_pose=best_pose,
            joint_limits=joint_limits,
            current_q_arm=current_q_arm,
            track_axis_alignment=track_axis_alignment,
        )

    rotation = np.asarray(orientation, dtype=float)
    for _ in range(iters):
        gripper_frame_pos = target_tip - rotation @ TIP_OFFSET
        target_pose = _build_pose(*gripper_frame_pos, rotation)
        next_q = _inverse_kinematics_step(kinematics, q, target_pose, 1.0, 1.0)
        if next_q is None or not np.all(np.isfinite(next_q)):
            break
        q = next_q
        pose = np.asarray(
            kinematics.forward_kinematics(q[: len(ARM_JOINTS)]),
            dtype=float,
        )
        residual_mm = float(np.linalg.norm(_tip_from_frame_pose(pose) - target_tip) * 1000.0)
        iters_used += 1
        if residual_mm < best_residual_mm:
            best_residual_mm = residual_mm
            best_q = q.copy()
            best_target_frame_pos = gripper_frame_pos.copy()
            best_pose = pose
        if residual_mm <= IK_CONVERGENCE_TOL_MM:
            break
    return _build_ik_solution(
        seed_name=seed_name,
        best_q=best_q,
        best_residual_mm=best_residual_mm,
        iters_used=iters_used,
        best_target_frame_pos=best_target_frame_pos,
        best_pose=best_pose,
        joint_limits=joint_limits,
        current_q_arm=current_q_arm,
        track_axis_alignment=track_axis_alignment,
    )


def _solve_ik_for_tip(
    kinematics: RobotKinematics,
    axisalign_task,
    current_q: np.ndarray,
    target_tip: np.ndarray,
    orientation: np.ndarray | None,
    vertical: bool,
    joint_limits: dict[str, tuple[float, float]],
    iters: int = IK_ITERS,
) -> tuple[IKSolution, list[IKSolution]]:
    seed_specs = [("current", current_q.copy())]
    seed_specs.extend((name, seed.copy()) for name, seed in FALLBACK_IK_SEEDS_DEG)

    attempts = [
        _solve_ik_from_seed(
            kinematics=kinematics,
            axisalign_task=axisalign_task,
            seed_name=name,
            seed_q=seed,
            target_tip=target_tip,
            orientation=orientation,
            vertical=vertical,
            joint_limits=joint_limits,
            current_q_arm=current_q,
            iters=iters,
        )
        for name, seed in seed_specs
    ]
    candidates = [sol for sol in attempts if sol.valid]
    if not candidates:
        return attempts[0], attempts

    def score(sol: IKSolution) -> tuple[float, float]:
        travel = float(np.linalg.norm(sol.joints_deg - current_q))
        return (sol.residual_mm, travel)

    return min(candidates, key=score), attempts


def _format_joints(q: np.ndarray) -> str:
    return ", ".join(
        f"{name}={float(q[i]):+.1f}"
        for i, name in enumerate(ARM_JOINTS)
    )


def _describe_validation_failures(sol: IKSolution, max_residual_mm: float) -> list[str]:
    """Human-readable reasons this IK solution would fail post-IK validation.

    Returns an empty list when the solution is acceptable. The order matches
    the order of the underlying checks: finite -> joint limits -> FK residual
    -> max single-joint step -> (vertical only) axis alignment.
    """
    failures: list[str] = []
    if not np.all(np.isfinite(sol.joints_deg)):
        failures.append("solution contains non-finite joint angles")
    if not np.isfinite(sol.residual_mm):
        failures.append("FK residual is non-finite")
    if sol.joint_limit_violations:
        failures.append(
            "joint limits violated: " + "; ".join(sol.joint_limit_violations)
        )
    if np.isfinite(sol.residual_mm) and sol.residual_mm > max_residual_mm:
        failures.append(
            f"FK residual {sol.residual_mm:.2f} mm > {max_residual_mm:.1f} mm threshold"
        )
    if (
        np.isfinite(sol.max_joint_step_deg)
        and sol.max_joint_step_deg > MAX_JOINT_STEP_DEG
    ):
        failures.append(
            f"max joint step {sol.max_joint_step_deg:.1f} deg on "
            f"'{sol.max_joint_step_joint}' > {MAX_JOINT_STEP_DEG:.1f} deg "
            f"(IK picked a different branch from this seed; commanding it "
            f"would slam the motor)"
        )
    if (
        sol.axis_alignment_deg is not None
        and np.isfinite(sol.axis_alignment_deg)
        and sol.axis_alignment_deg > AXIS_ALIGNMENT_TOL_DEG
    ):
        failures.append(
            f"axis alignment {sol.axis_alignment_deg:.1f} deg from world-down "
            f"> {AXIS_ALIGNMENT_TOL_DEG:.1f} deg (vertical mode could not "
            f"satisfy 'point straight down' at this target)"
        )
    return failures


def _format_attempt_line(sol: IKSolution, max_residual_mm: float) -> str:
    """One-line summary of an IK attempt, including all post-IK metrics."""
    axis = (
        f" axis={sol.axis_alignment_deg:5.1f}deg"
        if sol.axis_alignment_deg is not None
        else ""
    )
    step = (
        f" step={sol.max_joint_step_deg:5.1f}deg@{sol.max_joint_step_joint}"
        if np.isfinite(sol.max_joint_step_deg)
        else " step=  inf"
    )
    failures = _describe_validation_failures(sol, max_residual_mm)
    status = "OK" if not failures else "INVALID"
    return (
        f"[move_arm] IK seed {sol.seed_name:<10} {status:<7} "
        f"residual={sol.residual_mm:7.2f}mm iters={sol.iters_used:2d}"
        f"{step}{axis} solved_tip={_format_xyz_mm(sol.solved_tip)}"
    )


def _format_ik_diagnostics(
    target_tip: np.ndarray,
    current_q: np.ndarray,
    solution: IKSolution,
    attempts: list[IKSolution],
    *,
    vertical: bool,
    orientation: np.ndarray | None,
    max_residual_mm: float,
) -> str:
    """Multi-line, copy-pasteable diagnostic block.

    Used both as the normal log output (printed before the move executes) and
    as the body of the `MoveArmError` raised on validation failure -- the same
    text either way, so a failed run yields a single block the user can paste
    as a repro.
    """
    lines: list[str] = []
    lines.append(f"[move_arm] target tip       {_format_xyz_mm(target_tip)}")
    lines.append(
        f"[move_arm] mode             "
        f"vertical={vertical}, orientation={'set' if orientation is not None else 'free'}"
    )
    lines.append(f"[move_arm] current joints   {_format_joints(current_q)}")
    for sol in attempts:
        lines.append(_format_attempt_line(sol, max_residual_mm))
    selected_failures = _describe_validation_failures(solution, max_residual_mm)
    selected_status = "OK" if not selected_failures else "INVALID"
    lines.append(
        f"[move_arm] selected seed   {solution.seed_name!r} "
        f"[{selected_status}]; target joints {_format_joints(solution.joints_deg)}"
    )
    if selected_failures:
        lines.append("[move_arm] selected_failures:")
        for f in selected_failures:
            lines.append(f"[move_arm]   - {f}")
    return "\n".join(lines)


def calibrate_arm() -> None:
    """Run the SO101 follower's calibration routine and exit.

    Use this when the motors and the saved calibration file have drifted apart
    (e.g. the URDF zero pose no longer matches all-joints-at-zero, or the
    homing-offset prompt keeps appearing on connect). Follows lerobot's
    interactive flow: pose the arm in the middle of its range of motion, then
    sweep each joint through its full range.
    """
    robot = SO101Follower(
        SO101FollowerConfig(port=PORT, id=ROBOT_ID, disable_torque_on_disconnect=False)
    )
    robot.connect(calibrate=False)
    try:
        robot.calibrate()
    finally:
        robot.disconnect()


def move_arm(
    x: float,
    y: float,
    z: float,
    orientation: np.ndarray | None = None,
    vertical: bool = False,
    duration: float = 2.0,
    hz: float = 50.0,
    max_residual_mm: float = MAX_IK_RESIDUAL_MM,
    dry_run: bool = False,
) -> None:
    target_tip = np.array([x, y, z], dtype=float)
    _assert_target_in_workspace(target_tip)
    if duration < 0.0:
        raise MoveArmError(f"duration must be non-negative, got {duration}")
    if hz <= 0.0:
        raise MoveArmError(f"hz must be positive, got {hz}")
    joint_limits = _load_joint_limits_deg()

    robot = SO101Follower(
        SO101FollowerConfig(port=PORT, id=ROBOT_ID, disable_torque_on_disconnect=False)
    )
    kinematics = RobotKinematics(
        urdf_path=str(URDF_PATH),
        target_frame_name="gripper_frame_link",
        joint_names=ARM_JOINTS,
    )

    axisalign_task = kinematics.solver.add_axisalign_task(
        "gripper_frame_link", GRIPPER_LOCAL_Z, WORLD_DOWN
    )
    axisalign_task.configure("vertical_axis", "soft", 0.0)

    wrist_roll_centering = kinematics.solver.add_joints_task()
    wrist_roll_centering.set_joint("wrist_roll", 0.0)
    wrist_roll_centering.configure(
        "wrist_roll_centering", "soft", WRIST_ROLL_CENTERING_WEIGHT
    )

    robot.connect()
    try:
        current_q, gripper_pos = _read_arm_and_gripper_deg(robot)
        current_violations = _joint_limit_violations(current_q, joint_limits)
        if current_violations:
            raise MoveArmError(
                "current joint readings are outside URDF limits; calibration "
                "or the saved motor offsets are likely stale: "
                + "; ".join(current_violations)
            )

        target_solution, attempts = _solve_ik_for_tip(
            kinematics=kinematics,
            axisalign_task=axisalign_task,
            current_q=current_q,
            target_tip=target_tip,
            orientation=orientation,
            vertical=vertical,
            joint_limits=joint_limits,
        )
        diagnostics = _format_ik_diagnostics(
            target_tip,
            current_q,
            target_solution,
            attempts,
            vertical=vertical,
            orientation=orientation,
            max_residual_mm=max_residual_mm,
        )
        print(diagnostics)

        selected_failures = _describe_validation_failures(target_solution, max_residual_mm)
        if selected_failures:
            raise MoveArmError(
                "IK solution failed post-IK validation; refusing to move.\n"
                + diagnostics
            )
        if dry_run:
            print("[move_arm] --dry-run: validation passed; skipping send_action")
            return

        if duration == 0.0:
            action = {
                f"{m}.pos": float(target_solution.joints_deg[i])
                for i, m in enumerate(ARM_JOINTS)
            }
            action["gripper.pos"] = gripper_pos
            robot.send_action(action)
            return

        start = time.perf_counter()
        dt = 1.0 / hz
        while True:
            elapsed = time.perf_counter() - start
            alpha = min(elapsed / duration, 1.0)
            q = (1.0 - alpha) * current_q + alpha * target_solution.joints_deg

            action = {f"{m}.pos": float(q[i]) for i, m in enumerate(ARM_JOINTS)}
            action["gripper.pos"] = gripper_pos
            robot.send_action(action)

            if alpha >= 1.0:
                break
            time.sleep(dt)
    finally:
        robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Move the SO101 gripper to (x, y, z) in the base frame."
    )
    parser.add_argument("x", type=float, nargs="?", help="X position in meters (forward)")
    parser.add_argument("y", type=float, nargs="?", help="Y position in meters (left)")
    parser.add_argument("z", type=float, nargs="?", help="Z position in meters (up)")
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Run the SO101 follower calibration routine and exit (x/y/z are ignored)",
    )
    parser.add_argument(
        "--vertical",
        action="store_true",
        help="Force the gripper to point straight down at the target",
    )
    parser.add_argument(
        "--max-residual-mm",
        type=float,
        default=MAX_IK_RESIDUAL_MM,
        help=(
            "Abort if FK of the selected IK solution misses the requested claw "
            f"tip by more than this many millimetres (default {MAX_IK_RESIDUAL_MM:.0f})"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run IK/reachability validation and print diagnostics without moving",
    )
    parser.add_argument("--duration", type=float, default=2.0, help="Interpolation time (s)")
    parser.add_argument("--hz", type=float, default=50.0, help="Control loop rate (Hz)")
    args = parser.parse_args()

    if args.calibrate:
        calibrate_arm()
    else:
        if args.x is None or args.y is None or args.z is None:
            parser.error("x, y, and z are required unless --calibrate is given")
        move_arm(
            args.x,
            args.y,
            args.z,
            vertical=args.vertical,
            duration=args.duration,
            hz=args.hz,
            max_residual_mm=args.max_residual_mm,
            dry_run=args.dry_run,
        )
