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
import time
from pathlib import Path

import numpy as np

from lerobot.model.kinematics import RobotKinematics
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig


URDF_PATH = Path(__file__).parent / "SO101" / "so101_new_calib.urdf"
PORT = "/dev/tty.usbmodem5AE60557941"
ROBOT_ID = "follower-1"

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
MOTOR_NAMES = ARM_JOINTS + ["gripper"]

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


def _build_pose(x: float, y: float, z: float, rotation: np.ndarray) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = [x, y, z]
    return pose


def _read_joints_deg(robot: SO101Follower) -> np.ndarray:
    obs = robot.get_observation()
    return np.array([float(obs[f"{m}.pos"]) for m in MOTOR_NAMES])


def _solve_ik_for_tip(
    kinematics: RobotKinematics,
    axisalign_task,
    current_q: np.ndarray,
    target_tip: np.ndarray,
    orientation: np.ndarray | None,
    vertical: bool,
    iters: int = IK_ITERS,
) -> np.ndarray:
    q = current_q.copy()

    if vertical and orientation is None:
        axisalign_task.configure("vertical_axis", "soft", 1.0)
        gripper_frame_pos = target_tip + np.array([0.0, 0.0, TIP_OFFSET[2]])
        target_pose = _build_pose(*gripper_frame_pos, np.eye(3))
        for _ in range(iters):
            q = kinematics.inverse_kinematics(
                q, target_pose, position_weight=1.0, orientation_weight=0.0
            )
        return q

    axisalign_task.configure("vertical_axis", "soft", 0.0)

    if orientation is None:
        rotation = kinematics.forward_kinematics(q[: len(ARM_JOINTS)])[:3, :3]
        for _ in range(iters):
            gripper_frame_pos = target_tip - rotation @ TIP_OFFSET
            target_pose = _build_pose(*gripper_frame_pos, rotation)
            q = kinematics.inverse_kinematics(
                q, target_pose, position_weight=1.0, orientation_weight=0.0
            )
            rotation = kinematics.forward_kinematics(q[: len(ARM_JOINTS)])[:3, :3]
        return q

    rotation = np.asarray(orientation, dtype=float)
    for _ in range(iters):
        gripper_frame_pos = target_tip - rotation @ TIP_OFFSET
        target_pose = _build_pose(*gripper_frame_pos, rotation)
        q = kinematics.inverse_kinematics(
            q, target_pose, position_weight=1.0, orientation_weight=1.0
        )
    return q


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
) -> None:
    target_tip = np.array([x, y, z], dtype=float)

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
        current_q = _read_joints_deg(robot)
        target_q = _solve_ik_for_tip(
            kinematics, axisalign_task, current_q, target_tip, orientation, vertical
        )

        start = time.perf_counter()
        dt = 1.0 / hz
        while True:
            elapsed = time.perf_counter() - start
            alpha = min(elapsed / duration, 1.0)
            q = (1.0 - alpha) * current_q + alpha * target_q

            robot.send_action({f"{m}.pos": float(q[i]) for i, m in enumerate(MOTOR_NAMES)})

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
        )
