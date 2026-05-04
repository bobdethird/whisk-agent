"""Move the SO101 gripper tip to a base-frame XYZ with a LeRobot EE pipeline.

V4 keeps the user-facing behavior from V3 but routes Cartesian waypoints
through a LeRobot `RobotProcessorPipeline`, matching the upstream
`so100_to_so100_EE` examples:

    EE action dict -> bounds/safety -> IK processor -> joint action dict

The IK processor here is intentionally local and tiny. LeRobot's stock
`InverseKinematicsEEToJoints` constrains orientation through the default
`RobotKinematics.inverse_kinematics` weights. For our current use case, the
target is position-only XYZ in `base_link`, so this variant calls the same
LeRobot IK with `orientation_weight=0.0` and holds `wrist_roll` by default.

By default, the CLI runs in simulation mode and prints the planned joint path.
Use `--execute` to command the physical robot.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any

import numpy as np

from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import (
    RobotAction,
    RobotActionProcessorStep,
    RobotObservation,
    RobotProcessorPipeline,
    TransitionKey,
)
from lerobot.processor.converters import (
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.robots.so_follower.robot_kinematic_processor import EEBoundsAndSafety
from lerobot.utils.rotation import Rotation


URDF_PATH = Path(__file__).parent / "SO101" / "so101_new_calib.urdf"
PORT = "/dev/tty.usbmodem5AE60557941"
ROBOT_ID = "follower-1"

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
MOTOR_NAMES = ARM_JOINTS + ["gripper"]
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
TIP_OFFSET = np.zeros(3, dtype=float)
MIN_ALLOWED_Z_M = 0.0

DEFAULT_SIM_JOINTS_DEG = np.zeros(len(ARM_JOINTS), dtype=float)
DEFAULT_GRIPPER_POS = 0.0

POSITION_WEIGHT = 1.0
ORIENTATION_WEIGHT = 0.0
IK_MAX_ITERS_PER_WAYPOINT = 20
IK_CONVERGENCE_TOL_MM = 1.0
MAX_FINAL_RESIDUAL_MM = 15.0

EE_WORKSPACE_BOUNDS = {"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]}
DEFAULT_MAX_EE_STEP_M = 0.10


@dataclass(frozen=True)
class MoveArmV4Plan:
    """A planned or simulated V4 move."""

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


@dataclass
class PositionOnlyIKEEToJoints(RobotActionProcessorStep):
    """LeRobot-style EE action -> joint action step with position-only IK."""

    kinematics: RobotKinematics
    motor_names: list[str]
    initial_guess_current_joints: bool = True
    hold_wrist_roll: bool = True
    wrist_roll_hold_deg: float | None = None
    position_weight: float = POSITION_WEIGHT
    orientation_weight: float = ORIENTATION_WEIGHT
    q_curr: np.ndarray | None = field(default=None, init=False, repr=False)

    def action(self, action: RobotAction) -> RobotAction:
        x = action.pop("ee.x")
        y = action.pop("ee.y")
        z = action.pop("ee.z")
        wx = action.pop("ee.wx")
        wy = action.pop("ee.wy")
        wz = action.pop("ee.wz")
        gripper_pos = action.pop("ee.gripper_pos")

        if None in (x, y, z, wx, wy, wz, gripper_pos):
            raise ValueError(
                "Missing required EE action fields: ee.x/y/z/wx/wy/wz/gripper_pos"
            )

        observation = self.transition.get(TransitionKey.OBSERVATION)
        if observation is None:
            raise ValueError("Observation is required for LeRobot IK processing")

        q_raw = np.array(
            [float(observation[f"{name}.pos"]) for name in self.motor_names],
            dtype=float,
        )

        if self.initial_guess_current_joints:
            self.q_curr = q_raw.copy()
        elif self.q_curr is None:
            self.q_curr = q_raw.copy()

        if self.hold_wrist_roll and self.wrist_roll_hold_deg is None:
            self.wrist_roll_hold_deg = float(q_raw[WRIST_ROLL_INDEX])

        desired_pose = np.eye(4, dtype=float)
        desired_pose[:3, :3] = Rotation.from_rotvec([wx, wy, wz]).as_matrix()
        desired_pose[:3, 3] = [float(x), float(y), float(z)]

        q_target = np.asarray(
            self.kinematics.inverse_kinematics(
                self.q_curr,
                desired_pose,
                position_weight=self.position_weight,
                orientation_weight=self.orientation_weight,
            ),
            dtype=float,
        ).reshape(-1)

        if q_target.shape[0] < len(self.motor_names):
            raise RuntimeError(
                f"IK returned {q_target.shape[0]} joint(s), "
                f"expected at least {len(self.motor_names)}"
            )
        if not np.all(np.isfinite(q_target[: len(self.motor_names)])):
            raise RuntimeError(f"IK returned non-finite joints: {q_target}")

        if self.hold_wrist_roll and self.wrist_roll_hold_deg is not None:
            q_target[WRIST_ROLL_INDEX] = float(self.wrist_roll_hold_deg)

        self.q_curr = q_target.copy()
        for i, name in enumerate(self.motor_names):
            if name == "gripper":
                action["gripper.pos"] = float(gripper_pos)
            else:
                action[f"{name}.pos"] = float(q_target[i])
        return action

    def transform_features(self, features: dict[Any, Any]) -> dict[Any, Any]:
        return features

    def reset(self) -> None:
        self.q_curr = None


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


def _build_ee_to_joints_pipeline(
    *,
    kinematics: RobotKinematics,
    initial_guess_current_joints: bool,
    hold_wrist_roll: bool,
    wrist_roll_hold_deg: float | None,
    max_ee_step_m: float,
) -> RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction]:
    return RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[
            EEBoundsAndSafety(
                end_effector_bounds=EE_WORKSPACE_BOUNDS,
                max_ee_step_m=max_ee_step_m,
            ),
            PositionOnlyIKEEToJoints(
                kinematics=kinematics,
                motor_names=MOTOR_NAMES,
                initial_guess_current_joints=initial_guess_current_joints,
                hold_wrist_roll=hold_wrist_roll,
                wrist_roll_hold_deg=wrist_roll_hold_deg,
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
        name="MoveArmV4EEToJoints",
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


def _observation_from_joints(q_arm: np.ndarray, gripper_pos: float) -> RobotObservation:
    q = np.asarray(q_arm, dtype=float).reshape(len(ARM_JOINTS))
    obs = {f"{name}.pos": float(q[i]) for i, name in enumerate(ARM_JOINTS)}
    obs["gripper.pos"] = float(gripper_pos)
    return obs


def _ee_action_from_tip(
    tip_xyz: np.ndarray,
    reference_rotation: np.ndarray,
    gripper_pos: float,
) -> RobotAction:
    tip_offset_world = reference_rotation @ TIP_OFFSET
    frame_xyz = np.asarray(tip_xyz, dtype=float).reshape(3) - tip_offset_world
    rotvec = Rotation.from_matrix(reference_rotation).as_rotvec()
    return {
        "ee.x": float(frame_xyz[0]),
        "ee.y": float(frame_xyz[1]),
        "ee.z": float(frame_xyz[2]),
        "ee.wx": float(rotvec[0]),
        "ee.wy": float(rotvec[1]),
        "ee.wz": float(rotvec[2]),
        "ee.gripper_pos": float(gripper_pos),
    }


def _arm_joints_from_action(action: RobotAction) -> np.ndarray:
    return np.array([float(action[f"{name}.pos"]) for name in ARM_JOINTS], dtype=float)


def _solve_waypoint_with_pipeline(
    *,
    pipeline: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    kinematics: RobotKinematics,
    seed_q: np.ndarray,
    waypoint_tip: np.ndarray,
    reference_rotation: np.ndarray,
    gripper_pos: float,
    max_iters: int = IK_MAX_ITERS_PER_WAYPOINT,
    tol_mm: float = IK_CONVERGENCE_TOL_MM,
) -> tuple[np.ndarray, float]:
    """Run the LeRobot-style EE pipeline for one Cartesian waypoint."""

    q = np.asarray(seed_q, dtype=float).copy()
    best_q = q.copy()
    best_residual_mm = float("inf")

    for _ in range(max_iters):
        ee_action = _ee_action_from_tip(waypoint_tip, reference_rotation, gripper_pos)
        obs = _observation_from_joints(q, gripper_pos)
        joint_action = pipeline((ee_action, obs))
        q = _arm_joints_from_action(joint_action)

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
    gripper_pos: float = DEFAULT_GRIPPER_POS,
) -> MoveArmV4Plan:
    """Plan a position-only LeRobot EE-pipeline path to target tip XYZ."""

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
    reference_rotation = start_pose[:3, :3].copy()

    num_steps = max(1, int(round(duration * hz)))
    if duration == 0.0:
        num_steps = 1

    path_len_m = float(np.linalg.norm(target_tip - start_tip))
    max_ee_step_m = max(DEFAULT_MAX_EE_STEP_M, path_len_m / num_steps * 1.25)
    pipeline = _build_ee_to_joints_pipeline(
        kinematics=kinematics,
        initial_guess_current_joints=False,
        hold_wrist_roll=hold_wrist_roll,
        wrist_roll_hold_deg=float(q_seed[WRIST_ROLL_INDEX]) if hold_wrist_roll else None,
        max_ee_step_m=max_ee_step_m,
    )

    q = q_seed.copy()
    joint_waypoints: list[np.ndarray] = []
    residuals_mm: list[float] = []

    for step in range(1, num_steps + 1):
        alpha = step / num_steps
        waypoint_tip = start_tip + alpha * (target_tip - start_tip)
        q, residual_mm = _solve_waypoint_with_pipeline(
            pipeline=pipeline,
            kinematics=kinematics,
            seed_q=q,
            waypoint_tip=waypoint_tip,
            reference_rotation=reference_rotation,
            gripper_pos=gripper_pos,
        )
        residuals_mm.append(residual_mm)
        joint_waypoints.append(q.copy())

    final_pose = np.asarray(kinematics.forward_kinematics(q), dtype=float)
    final_tip = _tip_from_frame_pose(final_pose)
    final_residual_mm = float(np.linalg.norm(final_tip - target_tip) * 1000.0)
    max_step_residual_mm = max(residuals_mm) if residuals_mm else final_residual_mm

    return MoveArmV4Plan(
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


def print_plan(plan: MoveArmV4Plan, *, label: str = "move_arm_v4") -> None:
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
    label: str = "move_arm_v4",
) -> None:
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
    plan: MoveArmV4Plan,
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
    plan: MoveArmV4Plan,
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
        f"MoveArmV4 EE-pipeline IK {'OK' if plan.valid else 'FAILED'} "
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
        print(f"[move_arm_v4] saved visualization to {out}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def _send_joint_waypoints(
    robot: SO101Follower,
    plan: MoveArmV4Plan,
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
    plan: MoveArmV4Plan,
    *,
    gripper_pos: float,
    hold_wrist_roll: bool,
) -> None:
    """Execute the planned Cartesian schedule, seeding IK from live joints."""

    if not plan.joint_waypoints_deg:
        return

    start_pose = np.asarray(kinematics.forward_kinematics(plan.start_joints_deg), dtype=float)
    reference_rotation = start_pose[:3, :3].copy()
    path_len_m = float(np.linalg.norm(plan.target_tip - plan.start_tip))
    max_ee_step_m = max(
        DEFAULT_MAX_EE_STEP_M,
        path_len_m / len(plan.joint_waypoints_deg) * 1.25,
    )
    pipeline = _build_ee_to_joints_pipeline(
        kinematics=kinematics,
        initial_guess_current_joints=True,
        hold_wrist_roll=hold_wrist_roll,
        wrist_roll_hold_deg=(
            float(plan.start_joints_deg[WRIST_ROLL_INDEX]) if hold_wrist_roll else None
        ),
        max_ee_step_m=max_ee_step_m,
    )

    dt = 1.0 / plan.hz
    num_steps = len(plan.joint_waypoints_deg)
    for step in range(1, num_steps + 1):
        loop_start = time.perf_counter()
        current_q, _ = _read_arm_and_gripper_deg(robot)
        alpha = step / num_steps
        waypoint_tip = plan.start_tip + alpha * (plan.target_tip - plan.start_tip)
        ee_action = _ee_action_from_tip(waypoint_tip, reference_rotation, gripper_pos)
        obs = _observation_from_joints(current_q, gripper_pos)
        joint_action = pipeline((ee_action, obs))
        robot.send_action(joint_action)
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
) -> MoveArmV4Plan:
    """Move or simulate moving the gripper tip to base-frame `(x, y, z)`."""

    target_tip = np.array([x, y, z], dtype=float)
    kinematics = _build_kinematics()

    if simulate:
        if current_joints_deg is not None:
            q0 = np.asarray(current_joints_deg, dtype=float)
        elif simulate_from_robot:
            q0 = read_current_arm_joints_deg()
            print(f"[move_arm_v4] simulation start joints read from robot: {_format_joints(q0)}")
        else:
            q0 = DEFAULT_SIM_JOINTS_DEG.copy()
        held_gripper = DEFAULT_GRIPPER_POS if gripper_pos is None else float(gripper_pos)
        plan = plan_tip_move(
            kinematics=kinematics,
            target_tip=target_tip,
            current_joints_deg=q0,
            duration=duration,
            hz=hz,
            max_final_residual_mm=max_final_residual_mm,
            hold_wrist_roll=hold_wrist_roll,
            gripper_pos=held_gripper,
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
        print("[move_arm_v4] simulation only; no robot commands sent")
        return plan

    robot = SO101Follower(
        SO101FollowerConfig(port=PORT, id=ROBOT_ID, disable_torque_on_disconnect=False)
    )
    robot.connect()
    try:
        q0, observed_gripper = _read_arm_and_gripper_deg(robot)
        held_gripper = observed_gripper if gripper_pos is None else float(gripper_pos)
        plan = plan_tip_move(
            kinematics=kinematics,
            target_tip=target_tip,
            current_joints_deg=q0,
            duration=duration,
            hz=hz,
            max_final_residual_mm=max_final_residual_mm,
            hold_wrist_roll=hold_wrist_roll,
            gripper_pos=held_gripper,
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
                "MoveArmV4 IK plan is invalid; refusing to command robot. "
                f"final residual {plan.final_residual_mm:.2f}mm > "
                f"{plan.max_final_residual_mm:.1f}mm"
            )
        if closed_loop_execution:
            print("[move_arm_v4] executing closed-loop EE pipeline from live joints")
            _send_cartesian_closed_loop(
                robot,
                kinematics,
                plan,
                gripper_pos=held_gripper,
                hold_wrist_roll=hold_wrist_roll,
            )
        else:
            print("[move_arm_v4] executing open-loop preplanned joint waypoints")
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
            "with a LeRobot-style EE pipeline and position-only IK."
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
