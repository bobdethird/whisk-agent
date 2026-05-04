"""Move the SO101 arm's claw tip to (x, y, z) using LeRobot's IK pipeline.

This is the v2 of move_arm.py. Where v1 calls
`RobotKinematics.inverse_kinematics` directly inside a hand-rolled solver, v2
builds the canonical LeRobot end-effector action pipeline:

    (RobotAction, RobotObservation)
        -> EEBoundsAndSafety           (workspace clip + per-step jump guard)
        -> InverseKinematicsEEToJoints (placo IK, seeded from current joints)
        -> RobotAction                 (joint-space goal positions)

Per control tick we:
1.  Read the latest observation from the robot.
2.  Compute the next interpolated EE pose (linear in position, SLERP in
    orientation) along the path from the captured starting pose to the target.
3.  Push `(action, observation)` through the pipeline. The pipeline clips the
    target to the workspace, aborts on unsafe Cartesian jumps, and runs IK
    against the current measured joints (closed-loop).
4.  Send the resulting joint goals to the arm with `robot.send_action`.

This mirrors the data flow used by `lerobot_teleoperate.py` and
`make_default_robot_action_processor` in `lerobot.processor.factory`, which
both stream `(action, obs)` tuples through a `RobotProcessorPipeline` whose
`to_transition`/`to_output` are the standard tuple converters.

Compared with v1, v2:
- Performs IK every tick against the live observation (closed-loop) instead
  of solving once and joint-space-interpolating between current and target.
- Uses LeRobot's `EEBoundsAndSafety` as the workspace + jump guard instead of
  a bespoke `_assert_target_in_workspace` + `MAX_JOINT_STEP_DEG` check.
- Keeps the gripper at its observed position; pass an explicit `gripper_pos`
  argument or `--gripper-pos` to drive the jaw.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import RobotProcessorPipeline
from lerobot.processor.converters import (
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.robots.so_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    InverseKinematicsEEToJoints,
)
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.rotation import Rotation

# ---------------------------------------------------------------------------
# Robot / URDF setup -- mirrors move_arm.py.
# ---------------------------------------------------------------------------
URDF_PATH = Path(__file__).parent / "SO101" / "so101_new_calib.urdf"
PORT = "/dev/tty.usbmodem5AE60557941"
ROBOT_ID = "follower-1"

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
# `InverseKinematicsEEToJoints` writes one `<name>.pos` per entry in this
# list (using the IK solution for arm joints, and `ee.gripper_pos` for
# "gripper"). Order must match the URDF arm chain followed by the gripper.
MOTOR_NAMES = ARM_JOINTS + ["gripper"]
TARGET_FRAME = "gripper_frame_link"

# 180 deg rotation about world +X. Aligns the gripper's local +Z with world -Z,
# i.e. "gripper points straight down". Same orientation as v1's DOWN_ORIENTATION.
DOWN_ROTVEC = np.array([np.pi, 0.0, 0.0], dtype=float)

# Tip is 1 mm in front of `gripper_frame_link` along its local +Z. Same as v1.
# Use this to convert between the user-visible "claw tip" position and the
# `gripper_frame_link` position the IK solves for.
TIP_OFFSET = np.array([0.0, 0.0, 0.001], dtype=float)

# Workspace clip applied by `EEBoundsAndSafety`. These are bounds on
# `gripper_frame_link` in the base frame -- intentionally generous; the URDF
# joint limits and IK convergence are the precise reachability check.
EE_BOUNDS_MIN = np.array([-0.40, -0.45, 0.00], dtype=float)
EE_BOUNDS_MAX = np.array([0.48, 0.45, 0.55], dtype=float)

# Hard cap on |Δposition| between two successive pipeline calls.
# `EEBoundsAndSafety` raises ValueError when a single step exceeds this.
# We size num_steps below to keep per-step travel well under this value.
MAX_EE_STEP_M = 0.05

# Soft pull on `wrist_roll` toward 0 added to the placo solver. The default
# SO101 IK leaves `wrist_roll` underdetermined when the gripper points along
# its own +Z (e.g. straight down): the rotation about that axis doesn't
# affect the EE pose, so placo will happily settle on any angle, including
# >90 deg from the current pose. This soft task pulls the IK back toward a
# neutral wrist roll without overriding position/orientation. Same trick as
# v1's `move_arm.py`.
WRIST_ROLL_CENTERING_WEIGHT = 1e-3


def _configure_kinematics(kinematics: RobotKinematics) -> None:
    """Add SO101-specific soft tasks to the placo solver.

    Mutates `kinematics.solver` in place; safe to call once per kinematics
    instance (placo will error on duplicate task names).
    """
    wrist_roll_task = kinematics.solver.add_joints_task()
    wrist_roll_task.set_joint("wrist_roll", 0.0)
    wrist_roll_task.configure(
        "wrist_roll_centering", "soft", WRIST_ROLL_CENTERING_WEIGHT
    )


def _build_pipeline(
    kinematics: RobotKinematics,
) -> RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction]:
    """Build the canonical LeRobot EE -> joints action pipeline.

    The shape (`tuple[RobotAction, RobotObservation] -> RobotAction`) and the
    `to_transition`/`to_output` converters match
    `lerobot.processor.factory.make_default_robot_action_processor` and the
    teleop loop in `lerobot.scripts.lerobot_teleoperate`.
    """
    return RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[
            EEBoundsAndSafety(
                end_effector_bounds={
                    "min": EE_BOUNDS_MIN.tolist(),
                    "max": EE_BOUNDS_MAX.tolist(),
                },
                max_ee_step_m=MAX_EE_STEP_M,
            ),
            InverseKinematicsEEToJoints(
                kinematics=kinematics,
                motor_names=MOTOR_NAMES,
                # Seed every IK call with the latest measured joints. This is
                # the recommended setting for closed-loop control; it avoids
                # drift when the arm is perturbed and keeps the IK on the
                # correct branch as long as we don't command discontinuous
                # poses.
                initial_guess_current_joints=True,
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )


def _slerp_quat(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two unit quaternions.

    Both inputs and the output use lerobot.utils.rotation's [x, y, z, w]
    convention. `t` is a scalar in [0, 1].
    """
    q0 = np.asarray(q0, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    dot = float(np.dot(q0, q1))
    # Antipodal quaternions represent the same rotation; flip one to take
    # the shortest great-circle path.
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        # Very small angle: lerp + renormalize is numerically stable.
        out = q0 + t * (q1 - q0)
        return out / np.linalg.norm(out)
    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta_0 = np.sin(theta_0)
    s0 = np.sin(theta_0 * (1.0 - t)) / sin_theta_0
    s1 = np.sin(theta_0 * t) / sin_theta_0
    return s0 * q0 + s1 * q1


def _interpolate_pose(
    start_pos: np.ndarray,
    end_pos: np.ndarray,
    start_R: np.ndarray,
    end_R: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Linear interpolation in position + SLERP in orientation.

    Returns `(position, rotvec)` ready to drop into an EE action dict.
    """
    pos = start_pos + alpha * (end_pos - start_pos)
    q0 = Rotation.from_matrix(start_R).as_quat()
    q1 = Rotation.from_matrix(end_R).as_quat()
    q_interp = _slerp_quat(q0, q1, alpha)
    rotvec = Rotation.from_quat(q_interp).as_rotvec()
    return pos, rotvec


def _read_joints_and_obs(
    robot: SO101Follower,
) -> tuple[np.ndarray, float, RobotObservation]:
    obs = robot.get_observation()
    arm_q = np.array([float(obs[f"{m}.pos"]) for m in ARM_JOINTS], dtype=float)
    gripper_pos = float(obs.get("gripper.pos", 0.0))
    return arm_q, gripper_pos, obs


def _format_xyz_mm(xyz: np.ndarray) -> str:
    mm = np.asarray(xyz, dtype=float).reshape(3) * 1000.0
    return f"({mm[0]:+7.1f}, {mm[1]:+7.1f}, {mm[2]:+7.1f}) mm"


def _format_rotvec_deg(rotvec: np.ndarray) -> str:
    rv = np.asarray(rotvec, dtype=float).reshape(3)
    angle_deg = float(np.degrees(np.linalg.norm(rv)))
    return f"axis={rv}, angle={angle_deg:+.1f} deg"


def move_arm(
    x: float,
    y: float,
    z: float,
    target_rotvec: np.ndarray | None = None,
    gripper_pos: float | None = None,
    duration: float = 2.0,
    hz: float = 50.0,
    dry_run: bool = False,
) -> None:
    """Stream a Cartesian trajectory to the SO101 via the LeRobot IK pipeline.

    Args:
        x, y, z: target *claw tip* position in the base frame, meters.
        target_rotvec: target gripper rotation as an axis-angle vector
            (radians). Defaults to "gripper points straight down".
        gripper_pos: gripper jaw position to hold during the move (RANGE_0_100
            on the SO101's gripper). If None, the current gripper reading is
            held.
        duration: target trajectory duration in seconds. Combined with `hz`
            this sets the minimum number of control ticks; we may override
            with more ticks to keep per-step Cartesian travel under
            `MAX_EE_STEP_M`.
        hz: control loop rate in Hz.
        dry_run: print the trajectory plan without commanding the arm.
    """
    target_tip = np.array([x, y, z], dtype=float)
    rotvec = (
        DOWN_ROTVEC.copy()
        if target_rotvec is None
        else np.asarray(target_rotvec, dtype=float).reshape(3)
    )
    target_R = Rotation.from_rotvec(rotvec).as_matrix()
    # Convert "tip" target into "gripper_frame_link" target. The IK targets
    # gripper_frame_link, but the user-facing API talks in tip coordinates
    # for parity with v1.
    target_pos = target_tip - target_R @ TIP_OFFSET

    # Validate the target *before* connecting. EEBoundsAndSafety would also
    # silently clip; we'd rather refuse and let the caller fix the input.
    if (target_pos < EE_BOUNDS_MIN).any() or (target_pos > EE_BOUNDS_MAX).any():
        clipped = np.clip(target_pos, EE_BOUNDS_MIN, EE_BOUNDS_MAX)
        miss_mm = float(np.linalg.norm(clipped - target_pos) * 1000.0)
        raise ValueError(
            "target gripper pose is outside workspace bounds: "
            f"target={_format_xyz_mm(target_pos)}, "
            f"nearest={_format_xyz_mm(clipped)}, outside_by={miss_mm:.1f} mm"
        )
    if duration < 0.0:
        raise ValueError(f"duration must be non-negative, got {duration}")
    if hz <= 0.0:
        raise ValueError(f"hz must be positive, got {hz}")

    robot = SO101Follower(
        SO101FollowerConfig(port=PORT, id=ROBOT_ID, disable_torque_on_disconnect=False)
    )
    kinematics = RobotKinematics(
        urdf_path=str(URDF_PATH),
        target_frame_name=TARGET_FRAME,
        joint_names=ARM_JOINTS,
    )
    _configure_kinematics(kinematics)
    pipeline = _build_pipeline(kinematics)

    robot.connect()
    try:
        arm_q0, current_gripper, obs = _read_joints_and_obs(robot)
        held_gripper = float(current_gripper if gripper_pos is None else gripper_pos)

        # Capture starting EE pose. After this the arm must hold still until
        # we start streaming targets, otherwise the planned trajectory and
        # the actual joints diverge.
        T_start = np.asarray(kinematics.forward_kinematics(arm_q0), dtype=float)
        start_pos = T_start[:3, 3].copy()
        start_R = T_start[:3, :3].copy()
        start_tip = start_pos + start_R @ TIP_OFFSET

        cart_dist = float(np.linalg.norm(target_pos - start_pos))
        # 0.5 * MAX_EE_STEP_M gives a 2x safety margin against the
        # `EEBoundsAndSafety` jump check.
        min_steps_for_safety = max(1, int(np.ceil(cart_dist / max(0.5 * MAX_EE_STEP_M, 1e-6))))
        steps_for_duration = max(1, int(np.ceil(duration * hz)))
        num_steps = max(min_steps_for_safety, steps_for_duration)

        per_step_mm = cart_dist * 1000.0 / num_steps
        print(f"[move_arm_v2] start tip          {_format_xyz_mm(start_tip)}")
        print(f"[move_arm_v2] target tip         {_format_xyz_mm(target_tip)}")
        print(f"[move_arm_v2] start frame_link   {_format_xyz_mm(start_pos)}")
        print(f"[move_arm_v2] target frame_link  {_format_xyz_mm(target_pos)}")
        print(f"[move_arm_v2] target rotvec      {_format_rotvec_deg(rotvec)}")
        print(f"[move_arm_v2] cart distance      {cart_dist*1000:.1f} mm")
        print(
            f"[move_arm_v2] {num_steps} steps @ {hz:.1f} Hz "
            f"(~{num_steps / hz:.2f}s, ~{per_step_mm:.1f} mm/step, "
            f"max_ee_step_m={MAX_EE_STEP_M*1000:.0f} mm)"
        )

        if dry_run:
            print("[move_arm_v2] --dry-run: skipping send_action")
            return

        dt = 1.0 / hz
        for i in range(num_steps):
            loop_start = time.perf_counter()

            alpha = (i + 1) / num_steps
            interp_pos, interp_rotvec = _interpolate_pose(
                start_pos, target_pos, start_R, target_R, alpha
            )
            action: RobotAction = {
                "ee.x": float(interp_pos[0]),
                "ee.y": float(interp_pos[1]),
                "ee.z": float(interp_pos[2]),
                "ee.wx": float(interp_rotvec[0]),
                "ee.wy": float(interp_rotvec[1]),
                "ee.wz": float(interp_rotvec[2]),
                "ee.gripper_pos": held_gripper,
            }

            try:
                joint_action = pipeline((action, obs))
            except ValueError as e:
                # EEBoundsAndSafety raises on Cartesian jumps; surface a
                # diagnostic that explains which step failed and how far we
                # had moved by then.
                raise RuntimeError(
                    f"[move_arm_v2] pipeline rejected step {i+1}/{num_steps} "
                    f"at alpha={alpha:.3f} ({_format_xyz_mm(interp_pos)}): {e}"
                ) from e

            robot.send_action(joint_action)

            # Refresh observation so the next IK call seeds from the latest
            # measured joints (closed-loop).
            obs = robot.get_observation()

            precise_sleep(max(dt - (time.perf_counter() - loop_start), 0.0))
    finally:
        robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Move the SO101 gripper tip to (x, y, z) in the base frame using "
            "LeRobot's InverseKinematicsEEToJoints + EEBoundsAndSafety pipeline."
        )
    )
    parser.add_argument("x", type=float, help="X position in meters (forward)")
    parser.add_argument("y", type=float, help="Y position in meters (left)")
    parser.add_argument("z", type=float, help="Z position in meters (up)")
    parser.add_argument(
        "--rotvec",
        type=float,
        nargs=3,
        metavar=("WX", "WY", "WZ"),
        default=None,
        help=(
            "Target gripper orientation as a rotation vector (axis-angle, "
            "radians). Defaults to [pi, 0, 0] which makes the gripper point "
            "straight down."
        ),
    )
    parser.add_argument(
        "--gripper-pos",
        type=float,
        default=None,
        help=(
            "Gripper position to hold during the move (RANGE_0_100). "
            "Default: hold the current reading."
        ),
    )
    parser.add_argument("--duration", type=float, default=2.0, help="Trajectory duration (s)")
    parser.add_argument("--hz", type=float, default=50.0, help="Control loop rate (Hz)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned trajectory and exit without commanding the arm",
    )
    args = parser.parse_args()

    rotvec = np.array(args.rotvec, dtype=float) if args.rotvec is not None else None
    move_arm(
        args.x,
        args.y,
        args.z,
        target_rotvec=rotvec,
        gripper_pos=args.gripper_pos,
        duration=args.duration,
        hz=args.hz,
        dry_run=args.dry_run,
    )
