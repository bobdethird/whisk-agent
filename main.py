from pathlib import Path

import mujoco  # type: ignore[import-not-found]
import mujoco.viewer  # type: ignore[import-not-found]
import numpy as np  # type: ignore[import-not-found]

from so101_kinematics import (
    SO101Kinematics,
    pose_from_position_rotation,
    pose_from_target_relative_gripper_angle,
    rotation_error_rad,
)
from so101_mujoco_utils import hold_position, move_to_pose, set_initial_pose


MODEL_PATH = Path(__file__).parent / "simulation_code" / "model" / "scene.xml"
DEFAULT_POSITION_WEIGHT = 1.0
DEFAULT_ORIENTATION_WEIGHT = 0.01
TARGETED_POSITION_WEIGHT = 1.0
TARGETED_ORIENTATION_WEIGHT = 0.01
TARGETED_MAX_ITERATIONS = 100
POSITION_WARNING_THRESHOLD_M = 0.01
ORIENTATION_WARNING_THRESHOLD_DEG = 5.0

STARTING_POSITION = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -45.0,
    "elbow_flex": 90.0,
    "wrist_flex": -45.0,
    "wrist_roll": 0.0,
    "gripper": 50.0,
}


def show_target(viewer, target_pose: np.ndarray):
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[0],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.015, 0.0, 0.0],
        pos=target_pose[:3, 3],
        mat=np.eye(3).flatten(),
        rgba=[0.0, 1.0, 0.0, 0.45],
    )
    viewer.user_scn.ngeom = 1
    viewer.sync()


def move(x: float, y: float, z: float, gripper_angle_degrees: float | None = None):
    """Move to a world-space end-effector position in meters.

    When provided, gripper_angle_degrees is measured from horizontal in the
    target-relative vertical plane. Positive angles point up; negative angles
    point down.
    """
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    kinematics = SO101Kinematics()
    starting_ee_pose = kinematics.forward_kinematics(STARTING_POSITION, frame="mujoco")
    target_world_position = np.array([x, y, z], dtype=float)
    position_weight = DEFAULT_POSITION_WEIGHT
    orientation_weight = DEFAULT_ORIENTATION_WEIGHT
    max_iterations = 25
    if gripper_angle_degrees is None:
        target_ee_pose = pose_from_position_rotation(target_world_position, starting_ee_pose[:3, :3])
    else:
        target_ee_pose = pose_from_target_relative_gripper_angle(
            target_world_position,
            gripper_angle_degrees,
        )
        position_weight = TARGETED_POSITION_WEIGHT
        orientation_weight = TARGETED_ORIENTATION_WEIGHT
        max_iterations = TARGETED_MAX_ITERATIONS

    target_position = kinematics.inverse_kinematics(
        STARTING_POSITION,
        target_ee_pose,
        position_weight=position_weight,
        orientation_weight=orientation_weight,
        gripper=STARTING_POSITION["gripper"],
        max_iterations=max_iterations,
    )
    solved_pose = kinematics.forward_kinematics(target_position, frame="mujoco")
    position_error = np.linalg.norm(target_ee_pose[:3, 3] - solved_pose[:3, 3])
    orientation_error = rotation_error_rad(target_ee_pose[:3, :3], solved_pose[:3, :3])

    print(f"Target world coordinates: x={x:.3f} m, y={y:.3f} m, z={z:.3f} m")
    print(f"Target world position: {target_ee_pose[:3, 3]}")
    print(f"IK position error: {position_error:.6f} m")
    print(f"IK orientation error: {np.rad2deg(orientation_error):.3f} deg")
    if gripper_angle_degrees is not None:
        print(f"Target gripper angle: {gripper_angle_degrees:.1f} deg from horizontal")
    if (
        position_error > POSITION_WARNING_THRESHOLD_M
        or np.rad2deg(orientation_error) > ORIENTATION_WARNING_THRESHOLD_DEG
    ):
        print("Warning: target pose may be near or outside the arm's reachable workspace.")

    set_initial_pose(model, data, STARTING_POSITION)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        show_target(viewer, target_ee_pose)
        hold_position(model, data, viewer, duration=1.0)
        move_to_pose(model, data, viewer, target_position, duration=3.0)
        hold_position(model, data, viewer, duration=3.0)


if __name__ == "__main__":
    move(0.20, 0.00, 0.40, gripper_angle_degrees=90.0)
