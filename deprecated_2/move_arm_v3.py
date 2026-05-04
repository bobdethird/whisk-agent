"""Move the SO101 gripper tip to a base-frame XYZ using LeRobot IK.

V3 is intentionally thin:
    - no cameras
    - no user-facing orientation controls
    - no custom multi-seed solver
    - minimal validation and an optional wrist-roll hold

It uses LeRobot's `RobotKinematics.inverse_kinematics` directly with
`orientation_weight=0.0`, so the IK objective is position-only. The target
provided by the caller is the gripper tip position in `base_link`; LeRobot IK
targets `gripper_frame_link`, so this module only compensates the small fixed
tip offset before calling IK.

By default, the CLI runs in simulation mode and prints the planned joint path.
Use `--execute` to command the physical robot.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import numpy as np

from lerobot.model.kinematics import RobotKinematics
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig


URDF_PATH = Path(__file__).parent / "SO101" / "so101_new_calib.urdf"
PORT = "/dev/tty.usbmodem5AE60557941"
ROBOT_ID = "follower-1"

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
WRIST_ROLL_INDEX = ARM_JOINTS.index("wrist_roll")
TARGET_FRAME = "gripper_frame_link"
SKELETON_LINKS = [
    "base_link",
    "shoulder_link",
    "upper_arm_link",
    "lower_arm_link",
    "wrist_link",
    "gripper_link",
    "gripper_frame_link",
]

# In this SO-101 URDF, gripper_frame_link is already the TCP between the jaws.
# Older scripts carried a 1 mm offset, but that makes the printed "tip" differ
# from the LeRobot IK target frame and obscures frame/calibration errors.
TIP_OFFSET = np.zeros(3, dtype=float)
MIN_ALLOWED_Z_M = 0.0

DEFAULT_SIM_JOINTS_DEG = np.zeros(len(ARM_JOINTS), dtype=float)
DEFAULT_GRIPPER_POS = 0.0

POSITION_WEIGHT = 1.0
ORIENTATION_WEIGHT = 0.0
IK_MAX_ITERS_PER_WAYPOINT = 20
IK_CONVERGENCE_TOL_MM = 1.0
MAX_FINAL_RESIDUAL_MM = 15.0


@dataclass(frozen=True)
class MoveArmV3Plan:
    """A planned or simulated V3 move."""

    target_tip: np.ndarray
    start_tip: np.ndarray
    final_tip: np.ndarray
    start_joints_deg: np.ndarray
    final_joints_deg: np.ndarray
    joint_waypoints_deg: list[np.ndarray]
    final_residual_mm: float
    max_step_residual_mm: float
    duration_s: float
    hz: float
    max_final_residual_mm: float

    @property
    def valid(self) -> bool:
        return (
            np.isfinite(self.final_residual_mm)
            and self.final_residual_mm <= self.max_final_residual_mm
        )


def _format_xyz_mm(xyz: np.ndarray) -> str:
    mm = np.asarray(xyz, dtype=float).reshape(3) * 1000.0
    return f"({mm[0]:+7.1f}, {mm[1]:+7.1f}, {mm[2]:+7.1f}) mm"


def _format_joints(q: np.ndarray) -> str:
    return ", ".join(f"{name}={float(q[i]):+.1f}" for i, name in enumerate(ARM_JOINTS))


def _build_kinematics() -> RobotKinematics:
    return RobotKinematics(
        urdf_path=str(URDF_PATH),
        target_frame_name=TARGET_FRAME,
        joint_names=ARM_JOINTS,
    )


def _read_arm_and_gripper_deg(robot: SO101Follower) -> tuple[np.ndarray, float]:
    obs = robot.get_observation()
    arm_q = np.array([float(obs[f"{m}.pos"]) for m in ARM_JOINTS], dtype=float)
    gripper = float(obs.get("gripper.pos", DEFAULT_GRIPPER_POS))
    return arm_q, gripper


def read_current_arm_joints_deg() -> np.ndarray:
    """Read current arm joints without commanding the robot."""

    robot = SO101Follower(
        SO101FollowerConfig(port=PORT, id=ROBOT_ID, disable_torque_on_disconnect=False)
    )
    robot.connect()
    try:
        arm_q, _ = _read_arm_and_gripper_deg(robot)
        return arm_q
    finally:
        robot.disconnect()


def _tip_from_frame_pose(frame_pose: np.ndarray) -> np.ndarray:
    return frame_pose[:3, 3] + frame_pose[:3, :3] @ TIP_OFFSET


def _pose_with_position_and_rotation(position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    pose = np.eye(4, dtype=float)
    pose[:3, :3] = rotation
    pose[:3, 3] = np.asarray(position, dtype=float).reshape(3)
    return pose


def _coerce_arm_joints(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float).reshape(-1)
    if q.shape[0] < len(ARM_JOINTS):
        raise RuntimeError(
            f"IK returned {q.shape[0]} joint(s), expected at least {len(ARM_JOINTS)}"
        )
    q_arm = q[: len(ARM_JOINTS)].copy()
    if not np.all(np.isfinite(q_arm)):
        raise RuntimeError(f"IK returned non-finite joints: {q_arm}")
    return q_arm


def _solve_waypoint_position_only(
    *,
    kinematics: RobotKinematics,
    seed_q: np.ndarray,
    waypoint_tip: np.ndarray,
    reference_rotation: np.ndarray,
    wrist_roll_hold_deg: float | None = None,
    max_iters: int = IK_MAX_ITERS_PER_WAYPOINT,
    tol_mm: float = IK_CONVERGENCE_TOL_MM,
) -> tuple[np.ndarray, float]:
    """Run LeRobot IK repeatedly for one position-only Cartesian waypoint."""

    tip_offset_world = reference_rotation @ TIP_OFFSET
    waypoint_frame_pos = waypoint_tip - tip_offset_world
    desired_pose = _pose_with_position_and_rotation(
        waypoint_frame_pos,
        reference_rotation,
    )

    q = np.asarray(seed_q, dtype=float).copy()
    best_q = q.copy()
    best_residual_mm = float("inf")

    for _ in range(max_iters):
        q = _coerce_arm_joints(
            kinematics.inverse_kinematics(
                q,
                desired_pose,
                position_weight=POSITION_WEIGHT,
                orientation_weight=ORIENTATION_WEIGHT,
            )
        )
        if wrist_roll_hold_deg is not None:
            # With position-only IK, wrist_roll is effectively nullspace for
            # the tip point. Holding it removes the visible wrist spin without
            # changing the target tip position.
            q[WRIST_ROLL_INDEX] = float(wrist_roll_hold_deg)
        solved_pose = np.asarray(kinematics.forward_kinematics(q), dtype=float)
        solved_tip = _tip_from_frame_pose(solved_pose)
        residual_mm = float(np.linalg.norm(solved_tip - waypoint_tip) * 1000.0)
        if residual_mm < best_residual_mm:
            best_residual_mm = residual_mm
            best_q = q.copy()
        if residual_mm <= tol_mm:
            break

    return best_q, best_residual_mm


def plan_tip_move(
    *,
    kinematics: RobotKinematics,
    target_tip: np.ndarray,
    current_joints_deg: np.ndarray,
    duration: float = 2.0,
    hz: float = 50.0,
    max_final_residual_mm: float = MAX_FINAL_RESIDUAL_MM,
    hold_wrist_roll: bool = True,
) -> MoveArmV3Plan:
    """Plan a position-only IK path from current joints to target tip XYZ.

    The generated Cartesian waypoints are a straight line in tip space. Each
    waypoint is solved by one LeRobot IK call seeded from the previous waypoint.
    This mirrors replay-style open-loop IK, while keeping this script small.
    """

    if duration < 0.0:
        raise ValueError(f"duration must be non-negative, got {duration}")
    if hz <= 0.0:
        raise ValueError(f"hz must be positive, got {hz}")

    target_tip = np.asarray(target_tip, dtype=float).reshape(3)
    if not np.all(np.isfinite(target_tip)):
        raise ValueError(f"target tip contains non-finite values: {target_tip}")

    q_seed = np.asarray(current_joints_deg, dtype=float).reshape(-1)
    if q_seed.shape[0] != len(ARM_JOINTS):
        raise ValueError(
            f"current_joints_deg must have {len(ARM_JOINTS)} values, got {q_seed.shape[0]}"
        )
    if not np.all(np.isfinite(q_seed)):
        raise ValueError(f"current_joints_deg contains non-finite values: {q_seed}")

    start_pose = np.asarray(kinematics.forward_kinematics(q_seed), dtype=float)
    start_tip = _tip_from_frame_pose(start_pose)

    # We need a rotation to build a homogeneous target pose, but orientation is
    # not part of the IK objective. Reusing the start rotation keeps tip-offset
    # compensation stable without constraining orientation.
    reference_rotation = start_pose[:3, :3].copy()

    num_steps = max(1, int(round(duration * hz)))
    if duration == 0.0:
        num_steps = 1

    q = q_seed.copy()
    wrist_roll_hold_deg = float(q_seed[WRIST_ROLL_INDEX]) if hold_wrist_roll else None
    joint_waypoints: list[np.ndarray] = []
    residuals_mm: list[float] = []

    for step in range(1, num_steps + 1):
        alpha = step / num_steps
        waypoint_tip = start_tip + alpha * (target_tip - start_tip)
        q, residual_mm = _solve_waypoint_position_only(
            kinematics=kinematics,
            seed_q=q,
            waypoint_tip=waypoint_tip,
            reference_rotation=reference_rotation,
            wrist_roll_hold_deg=wrist_roll_hold_deg,
        )
        residuals_mm.append(residual_mm)
        joint_waypoints.append(q.copy())

    final_pose = np.asarray(kinematics.forward_kinematics(q), dtype=float)
    final_tip = _tip_from_frame_pose(final_pose)
    final_residual_mm = float(np.linalg.norm(final_tip - target_tip) * 1000.0)
    max_step_residual_mm = max(residuals_mm) if residuals_mm else final_residual_mm

    return MoveArmV3Plan(
        target_tip=target_tip,
        start_tip=start_tip,
        final_tip=final_tip,
        start_joints_deg=q_seed,
        final_joints_deg=q,
        joint_waypoints_deg=joint_waypoints,
        final_residual_mm=final_residual_mm,
        max_step_residual_mm=max_step_residual_mm,
        duration_s=duration,
        hz=hz,
        max_final_residual_mm=max_final_residual_mm,
    )


def print_plan(plan: MoveArmV3Plan, *, label: str = "move_arm_v3") -> None:
    status = "OK" if plan.valid else "INVALID"
    print(f"[{label}] start tip        {_format_xyz_mm(plan.start_tip)}")
    print(f"[{label}] target tip       {_format_xyz_mm(plan.target_tip)}")
    print(f"[{label}] final tip        {_format_xyz_mm(plan.final_tip)}")
    print(f"[{label}] start joints     {_format_joints(plan.start_joints_deg)}")
    print(f"[{label}] final joints     {_format_joints(plan.final_joints_deg)}")
    print(
        f"[{label}] steps={len(plan.joint_waypoints_deg)} "
        f"duration={plan.duration_s:.2f}s hz={plan.hz:.1f} "
        f"final_residual={plan.final_residual_mm:.2f}mm "
        f"max_step_residual={plan.max_step_residual_mm:.2f}mm "
        f"status={status}"
    )
    if not plan.valid:
        print(
            f"[{label}] invalid plan: final residual "
            f"{plan.final_residual_mm:.2f}mm > "
            f"{plan.max_final_residual_mm:.1f}mm. Do not execute this target "
            "from this start pose."
        )
    if plan.start_tip[2] < MIN_ALLOWED_Z_M:
        print(
            f"[{label}] warning: FK start TCP z={plan.start_tip[2] * 1000.0:.1f}mm "
            "is below base_link z=0. Check robot calibration/homing or confirm "
            "that base_link z=0 is not your table/workspace floor."
        )


def _link_positions(kinematics: RobotKinematics, q_deg: np.ndarray) -> np.ndarray | None:
    """Return base-frame positions for the simple arm skeleton, if available."""

    positions: list[np.ndarray] = []
    try:
        kinematics.forward_kinematics(q_deg)
        for link in SKELETON_LINKS:
            T = np.asarray(kinematics.robot.get_T_world_frame(link), dtype=float)
            positions.append(T[:3, 3].copy())
    except Exception:
        return None
    return np.vstack(positions)


def print_fk_chain(
    kinematics: RobotKinematics,
    q_deg: np.ndarray,
    *,
    label: str = "move_arm_v3",
) -> None:
    """Print base-frame link positions for FK debugging."""

    print(f"[{label}] FK link chain for joints: {_format_joints(q_deg)}")
    try:
        kinematics.forward_kinematics(q_deg)
        for link in SKELETON_LINKS:
            T = np.asarray(kinematics.robot.get_T_world_frame(link), dtype=float)
            print(f"[{label}]   {link:<20} {_format_xyz_mm(T[:3, 3])}")
    except Exception as e:
        print(f"[{label}] FK link-chain debug unavailable: {e}")


def _tip_path_from_waypoints(
    kinematics: RobotKinematics,
    plan: MoveArmV3Plan,
) -> np.ndarray:
    points = [plan.start_tip]
    for q in plan.joint_waypoints_deg:
        pose = np.asarray(kinematics.forward_kinematics(q), dtype=float)
        points.append(_tip_from_frame_pose(pose))
    return np.vstack(points)


def _set_axes_equal(ax: Any, points: np.ndarray) -> None:
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    center = (mins + maxs) / 2.0
    radius = max(float(np.max(maxs - mins)) / 2.0, 0.05)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(max(0.0, center[2] - radius), center[2] + radius)


def visualize_plan(
    plan: MoveArmV3Plan,
    *,
    kinematics: RobotKinematics,
    show: bool = True,
    save_path: str | Path | None = None,
) -> None:
    """Render a simple Matplotlib 3D visualization of the simulated IK plan."""

    import matplotlib.pyplot as plt

    tip_path = _tip_path_from_waypoints(kinematics, plan)
    requested_path = np.vstack([plan.start_tip, plan.target_tip])
    start_skeleton = _link_positions(kinematics, plan.start_joints_deg)
    final_skeleton = _link_positions(kinematics, plan.final_joints_deg)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    solved_color = "#1f77b4" if plan.valid else "#d95f02"
    solved_label = "solved FK tip path" if plan.valid else "failed FK tip path"

    ax.plot(
        requested_path[:, 0],
        requested_path[:, 1],
        requested_path[:, 2],
        color="#999999",
        linestyle="--",
        linewidth=1.5,
        label="requested straight tip path",
    )
    ax.plot(
        tip_path[:, 0],
        tip_path[:, 1],
        tip_path[:, 2],
        color=solved_color,
        linewidth=2.5,
        label=solved_label,
    )
    ax.scatter(*plan.start_tip, color="#2ca02c", s=45, label="start tip")
    ax.scatter(*plan.target_tip, color="#d62728", s=70, marker="x", label="target tip")
    ax.scatter(*plan.final_tip, color="#ff7f0e", s=45, label="final solved tip")

    skeleton_points = []
    if start_skeleton is not None:
        skeleton_points.append(start_skeleton)
        ax.plot(
            start_skeleton[:, 0],
            start_skeleton[:, 1],
            start_skeleton[:, 2],
            color="#8c8c8c",
            linewidth=2.0,
            marker="o",
            markersize=3,
            alpha=0.65,
            label="start arm",
        )
    if final_skeleton is not None:
        skeleton_points.append(final_skeleton)
        ax.plot(
            final_skeleton[:, 0],
            final_skeleton[:, 1],
            final_skeleton[:, 2],
            color="#111111",
            linewidth=2.5,
            marker="o",
            markersize=4,
            label="final arm",
        )

    all_points = [tip_path, requested_path]
    all_points.extend(skeleton_points)
    _set_axes_equal(ax, np.vstack(all_points))

    ax.set_title(
        f"MoveArmV3 IK simulation {'OK' if plan.valid else 'FAILED'} "
        f"(final residual {plan.final_residual_mm:.2f} mm)"
    )
    ax.set_xlabel("X forward (m)")
    ax.set_ylabel("Y left (m)")
    ax.set_zlabel("Z up (m)")
    ax.legend(loc="upper left")
    ax.view_init(elev=24, azim=-58)
    fig.tight_layout()

    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=180)
        print(f"[move_arm_v3] saved visualization to {out}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def _send_joint_waypoints(
    robot: SO101Follower,
    plan: MoveArmV3Plan,
    *,
    gripper_pos: float,
) -> None:
    if not plan.joint_waypoints_deg:
        return

    dt = 1.0 / plan.hz
    for q in plan.joint_waypoints_deg:
        t0 = time.perf_counter()
        action = {f"{m}.pos": float(q[i]) for i, m in enumerate(ARM_JOINTS)}
        action["gripper.pos"] = float(gripper_pos)
        robot.send_action(action)
        sleep_s = dt - (time.perf_counter() - t0)
        if sleep_s > 0.0:
            time.sleep(sleep_s)


def _send_cartesian_closed_loop(
    robot: SO101Follower,
    kinematics: RobotKinematics,
    plan: MoveArmV3Plan,
    *,
    gripper_pos: float,
    hold_wrist_roll: bool,
) -> None:
    """Execute the planned Cartesian schedule, seeding IK from live joints."""

    if not plan.joint_waypoints_deg:
        return

    start_pose = np.asarray(kinematics.forward_kinematics(plan.start_joints_deg), dtype=float)
    reference_rotation = start_pose[:3, :3].copy()
    wrist_roll_hold_deg = (
        float(plan.start_joints_deg[WRIST_ROLL_INDEX]) if hold_wrist_roll else None
    )

    dt = 1.0 / plan.hz
    num_steps = len(plan.joint_waypoints_deg)
    for step in range(1, num_steps + 1):
        loop_start = time.perf_counter()
        current_q, _ = _read_arm_and_gripper_deg(robot)
        alpha = step / num_steps
        waypoint_tip = plan.start_tip + alpha * (plan.target_tip - plan.start_tip)
        q_cmd, _ = _solve_waypoint_position_only(
            kinematics=kinematics,
            seed_q=current_q,
            waypoint_tip=waypoint_tip,
            reference_rotation=reference_rotation,
            wrist_roll_hold_deg=wrist_roll_hold_deg,
        )
        action = {f"{m}.pos": float(q_cmd[i]) for i, m in enumerate(ARM_JOINTS)}
        action["gripper.pos"] = float(gripper_pos)
        robot.send_action(action)
        sleep_s = dt - (time.perf_counter() - loop_start)
        if sleep_s > 0.0:
            time.sleep(sleep_s)


def move_arm(
    x: float,
    y: float,
    z: float,
    *,
    duration: float = 2.0,
    hz: float = 50.0,
    simulate: bool = False,
    current_joints_deg: np.ndarray | None = None,
    simulate_from_robot: bool = False,
    gripper_pos: float | None = None,
    visualize: bool = False,
    save_plot: str | Path | None = None,
    max_final_residual_mm: float = MAX_FINAL_RESIDUAL_MM,
    hold_wrist_roll: bool = True,
    closed_loop_execution: bool = True,
    debug_fk_chain: bool = False,
) -> MoveArmV3Plan:
    """Move or simulate moving the gripper tip to base-frame `(x, y, z)`.

    In simulation mode no robot connection is opened. If `current_joints_deg`
    is omitted, the simulation starts from all-zero arm joints.
    """

    target_tip = np.array([x, y, z], dtype=float)
    kinematics = _build_kinematics()

    if simulate:
        if current_joints_deg is not None:
            q0 = np.asarray(current_joints_deg, dtype=float)
        elif simulate_from_robot:
            q0 = read_current_arm_joints_deg()
            print(f"[move_arm_v3] simulation start joints read from robot: {_format_joints(q0)}")
        else:
            q0 = DEFAULT_SIM_JOINTS_DEG.copy()
        plan = plan_tip_move(
            kinematics=kinematics,
            target_tip=target_tip,
            current_joints_deg=q0,
            duration=duration,
            hz=hz,
            max_final_residual_mm=max_final_residual_mm,
            hold_wrist_roll=hold_wrist_roll,
        )
        print_plan(plan)
        if debug_fk_chain:
            print_fk_chain(kinematics, q0)
        if visualize or save_plot is not None:
            visualize_plan(
                plan,
                kinematics=kinematics,
                show=visualize,
                save_path=save_plot,
            )
        print("[move_arm_v3] simulation only; no robot commands sent")
        return plan

    robot = SO101Follower(
        SO101FollowerConfig(port=PORT, id=ROBOT_ID, disable_torque_on_disconnect=False)
    )
    robot.connect()
    try:
        q0, observed_gripper = _read_arm_and_gripper_deg(robot)
        plan = plan_tip_move(
            kinematics=kinematics,
            target_tip=target_tip,
            current_joints_deg=q0,
            duration=duration,
            hz=hz,
            max_final_residual_mm=max_final_residual_mm,
            hold_wrist_roll=hold_wrist_roll,
        )
        print_plan(plan)
        if debug_fk_chain:
            print_fk_chain(kinematics, q0)
        if visualize or save_plot is not None:
            visualize_plan(
                plan,
                kinematics=kinematics,
                show=visualize,
                save_path=save_plot,
            )
        if not plan.valid:
            raise RuntimeError(
                "MoveArmV3 IK plan is invalid; refusing to command robot. "
                f"final residual {plan.final_residual_mm:.2f}mm > "
                f"{plan.max_final_residual_mm:.1f}mm"
            )
        held_gripper = observed_gripper if gripper_pos is None else gripper_pos
        if closed_loop_execution:
            print("[move_arm_v3] executing closed-loop: IK seeded from live joints each tick")
            _send_cartesian_closed_loop(
                robot,
                kinematics,
                plan,
                gripper_pos=held_gripper,
                hold_wrist_roll=hold_wrist_roll,
            )
        else:
            print("[move_arm_v3] executing open-loop preplanned joint waypoints")
            _send_joint_waypoints(
                robot,
                plan,
                gripper_pos=held_gripper,
            )
        return plan
    finally:
        robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Move or simulate moving the SO101 gripper tip to base-frame XYZ "
            "with thin LeRobot position-only IK."
        )
    )
    parser.add_argument("x", type=float, help="target X in base_link, meters")
    parser.add_argument("y", type=float, help="target Y in base_link, meters")
    parser.add_argument("z", type=float, help="target Z in base_link, meters")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="send commands to the physical robot; default is simulation only",
    )
    parser.add_argument("--duration", type=float, default=2.0, help="move duration in seconds")
    parser.add_argument("--hz", type=float, default=50.0, help="control/planning rate in Hz")
    parser.add_argument(
        "--max-final-residual-mm",
        type=float,
        default=MAX_FINAL_RESIDUAL_MM,
        help=(
            "mark/refuse plans whose final FK tip misses the target by more "
            f"than this many mm (default {MAX_FINAL_RESIDUAL_MM:.1f})"
        ),
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="show a Matplotlib 3D plot of the simulated IK plan",
    )
    parser.add_argument(
        "--save-plot",
        type=Path,
        help="save the 3D visualization PNG to this path",
    )
    parser.add_argument(
        "--current-joints",
        nargs=len(ARM_JOINTS),
        type=float,
        metavar="DEG",
        help=(
            "simulation start joints in degrees: "
            + ", ".join(ARM_JOINTS)
            + "; defaults to all zeros"
        ),
    )
    parser.add_argument(
        "--sim-from-robot",
        action="store_true",
        help="in simulation mode, read current arm joints from the robot without commanding motion",
    )
    parser.add_argument(
        "--allow-wrist-roll",
        action="store_true",
        help="do not hold wrist_roll at its starting value during position-only IK",
    )
    parser.add_argument(
        "--open-loop-execution",
        action="store_true",
        help="execute preplanned joint waypoints instead of closed-loop live-observation IK",
    )
    parser.add_argument(
        "--debug-fk-chain",
        action="store_true",
        help="print base-frame FK positions for each arm link before moving",
    )
    args = parser.parse_args()

    move_arm(
        args.x,
        args.y,
        args.z,
        duration=args.duration,
        hz=args.hz,
        simulate=not args.execute,
        current_joints_deg=(
            None if args.current_joints is None else np.array(args.current_joints, dtype=float)
        ),
        simulate_from_robot=args.sim_from_robot,
        visualize=args.visualize,
        save_plot=args.save_plot,
        max_final_residual_mm=args.max_final_residual_mm,
        hold_wrist_roll=not args.allow_wrist_roll,
        closed_loop_execution=not args.open_loop_execution,
        debug_fk_chain=args.debug_fk_chain,
    )
