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


def move(
    x: float,
    y: float,
    z: float,
    *,
    gripper: float | None = None,
    speed: float = 0.15,
    fps: float = 50.0,
    max_relative_target: float = 10.0,
) -> None:
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

        T_goal = T0.copy()
        T_goal[:3, 3] = np.array([x, y, z], dtype=float)

        gripper_pos = gripper
        if gripper_pos is None:
            gripper_pos = float(start_obs["gripper.pos"])

        dist = float(np.linalg.norm(T_goal[:3, 3] - T0[:3, 3]))
        duration = 0.0 if dist < 1e-9 else dist / max(speed, 1e-9)
        n_steps = max(1, int(duration * fps))
        period = 1.0 / fps

        for i in range(1, n_steps + 1):
            t0 = time.perf_counter()
            alpha = i / n_steps

            T_step = T0.copy()
            T_step[:3, 3] = (1.0 - alpha) * T0[:3, 3] + alpha * T_goal[:3, 3]

            obs = robot.get_observation()
            ee_action = ee_action_from_pose(T_step, gripper_pos)
            joint_action = ee_to_joints((ee_action, obs))
            robot.send_action(joint_action)

            precise_sleep(max(period - (time.perf_counter() - t0), 0.0))
    finally:
        robot.disconnect()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--x", type=float, required=True)
    p.add_argument("--y", type=float, required=True)
    p.add_argument("--z", type=float, required=True)
    p.add_argument("--gripper", type=float, default=None, help="0-100; default keeps current")
    p.add_argument(
        "--speed",
        type=float,
        default=0.15,
        help="Cartesian translation speed in m/s (move time = distance / speed)",
    )
    p.add_argument("--fps", type=float, default=50.0)
    p.add_argument("--max-relative-target", type=float, default=10.0)
    args = p.parse_args()

    move(
        x=args.x,
        y=args.y,
        z=args.z,
        gripper=args.gripper,
        speed=args.speed,
        fps=args.fps,
        max_relative_target=args.max_relative_target,
    )


if __name__ == "__main__":
    main()
