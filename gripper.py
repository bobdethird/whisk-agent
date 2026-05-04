#!/usr/bin/env python3
import argparse
import time

import numpy as np
from scipy.spatial.transform import Rotation

from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import RobotProcessorPipeline
from lerobot.processor.converters import (
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.robots.so_follower.robot_kinematic_processor import InverseKinematicsEEToJoints
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.robot_utils import precise_sleep

MOTORS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
IK_JOINTS = MOTORS[:-1]
EE_FRAME = "gripper_frame_link"

PORT = "/dev/tty.usbmodem5AE60557941"
ROBOT_ID = "follower-1"
USE_DEGREES = False
URDF_PATH = "./SO101/so101_new_calib.urdf"

# Gripper position is reported in 0-100 (percent open) on the SO101 follower.
GRIPPER_OPEN: float = 100.0
GRIPPER_CLOSED: float = 0.0


def obs_to_q(obs: RobotObservation) -> np.ndarray:
    return np.array([float(obs[f"{name}.pos"]) for name in MOTORS], dtype=float)


def ee_action_from_pose(T: np.ndarray, gripper_pos: float) -> RobotAction:
    xyz = T[:3, 3]
    rotvec = Rotation.from_matrix(T[:3, :3]).as_rotvec()
    return {
        "ee.x": float(xyz[0]),
        "ee.y": float(xyz[1]),
        "ee.z": float(xyz[2]),
        "ee.wx": float(rotvec[0]),
        "ee.wy": float(rotvec[1]),
        "ee.wz": float(rotvec[2]),
        "ee.gripper_pos": float(gripper_pos),
    }


def _set_gripper(
    target: float,
    *,
    speed: float,
    fps: float,
    max_relative_target: float,
) -> None:
    target = float(np.clip(target, 0.0, 100.0))

    robot = SO101Follower(
        SO101FollowerConfig(
            port=PORT,
            id=ROBOT_ID,
            use_degrees=USE_DEGREES,
            max_relative_target=max_relative_target,
            disable_torque_on_disconnect=False,
        )
    )

    kinematics = RobotKinematics(
        urdf_path=URDF_PATH,
        target_frame_name=EE_FRAME,
        joint_names=IK_JOINTS,
    )

    ee_to_joints = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[
            InverseKinematicsEEToJoints(
                kinematics=kinematics,
                motor_names=MOTORS,
                initial_guess_current_joints=False,
            )
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    robot.connect()
    try:
        start_obs = robot.get_observation()
        q0 = obs_to_q(start_obs)
        T0 = kinematics.forward_kinematics(q0)

        g0 = float(start_obs["gripper.pos"])

        delta = abs(target - g0)
        duration = 0.0 if delta < 1e-9 else delta / max(speed, 1e-9)
        n_steps = max(1, int(duration * fps))
        period = 1.0 / fps

        for i in range(1, n_steps + 1):
            t0 = time.perf_counter()
            alpha = i / n_steps

            g_step = (1.0 - alpha) * g0 + alpha * target

            obs = robot.get_observation()
            ee_action = ee_action_from_pose(T0, g_step)
            joint_action = ee_to_joints((ee_action, obs))
            robot.send_action(joint_action)

            precise_sleep(max(period - (time.perf_counter() - t0), 0.0))
    finally:
        robot.disconnect()


def open_gripper(
    *,
    speed: float = 50.0,
    fps: float = 50.0,
    max_relative_target: float = 10.0,
) -> None:
    _set_gripper(
        GRIPPER_OPEN,
        speed=speed,
        fps=fps,
        max_relative_target=max_relative_target,
    )


def close_gripper(
    *,
    speed: float = 50.0,
    fps: float = 50.0,
    max_relative_target: float = 10.0,
) -> None:
    _set_gripper(
        GRIPPER_CLOSED,
        speed=speed,
        fps=fps,
        max_relative_target=max_relative_target,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["open", "close"])
    p.add_argument(
        "--speed",
        type=float,
        default=50.0,
        help="Gripper speed in 0-100 units/sec (move time = |target - current| / speed)",
    )
    p.add_argument("--fps", type=float, default=50.0)
    p.add_argument("--max-relative-target", type=float, default=10.0)
    args = p.parse_args()

    fn = open_gripper if args.action == "open" else close_gripper
    fn(
        speed=args.speed,
        fps=args.fps,
        max_relative_target=args.max_relative_target,
    )


if __name__ == "__main__":
    main()
