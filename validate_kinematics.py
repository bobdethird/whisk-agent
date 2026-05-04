from pathlib import Path

import mujoco  # type: ignore[import-not-found]
import numpy as np  # type: ignore[import-not-found]

from so101_kinematics import MUJOCO_SITE_NAME, SO101Kinematics, rotation_error_rad
from so101_mujoco_utils import set_initial_pose


MODEL_PATH = Path(__file__).parent / "simulation_code" / "model" / "scene.xml"
POSITION_TOLERANCE_M = 1e-5
ROTATION_TOLERANCE_RAD = 1e-4

TEST_POSES = {
    "zero": {
        "shoulder_pan": 0.0,
        "shoulder_lift": 0.0,
        "elbow_flex": 0.0,
        "wrist_flex": 0.0,
        "wrist_roll": 0.0,
        "gripper": 50.0,
    },
    "assignment_pick": {
        "shoulder_pan": -45.0,
        "shoulder_lift": 45.0,
        "elbow_flex": -45.0,
        "wrist_flex": 90.0,
        "wrist_roll": 0.0,
        "gripper": 50.0,
    },
    "assignment_place": {
        "shoulder_pan": 45.0,
        "shoulder_lift": 45.0,
        "elbow_flex": -45.0,
        "wrist_flex": 90.0,
        "wrist_roll": 0.0,
        "gripper": 50.0,
    },
    "mixed": {
        "shoulder_pan": 30.0,
        "shoulder_lift": -30.0,
        "elbow_flex": 45.0,
        "wrist_flex": -20.0,
        "wrist_roll": 25.0,
        "gripper": 20.0,
    },
}


def get_mujoco_site_pose(model: mujoco.MjModel, data: mujoco.MjData, position_dict: dict[str, float]) -> np.ndarray:
    set_initial_pose(model, data, position_dict)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, MUJOCO_SITE_NAME)
    if site_id < 0:
        raise ValueError(f"MuJoCo site not found: {MUJOCO_SITE_NAME}")

    pose = np.eye(4)
    pose[:3, 3] = data.site_xpos[site_id]
    pose[:3, :3] = data.site_xmat[site_id].reshape(3, 3)
    return pose


def main():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    kinematics = SO101Kinematics()

    max_position_error = 0.0
    max_rotation_error = 0.0

    print(f"Using kinematics backend: {kinematics.backend_name}")
    for name, position_dict in TEST_POSES.items():
        mujoco_pose = get_mujoco_site_pose(model, data, position_dict)
        kinematics_pose = kinematics.forward_kinematics(position_dict, frame="mujoco")

        position_error = float(np.linalg.norm(mujoco_pose[:3, 3] - kinematics_pose[:3, 3]))
        rotation_error = rotation_error_rad(mujoco_pose[:3, :3], kinematics_pose[:3, :3])
        max_position_error = max(max_position_error, position_error)
        max_rotation_error = max(max_rotation_error, rotation_error)

        print(
            f"{name:16s} position_error={position_error:.6e} m "
            f"rotation_error={rotation_error:.6e} rad"
        )

    print(
        f"max_position_error={max_position_error:.6e} m "
        f"max_rotation_error={max_rotation_error:.6e} rad"
    )

    if max_position_error > POSITION_TOLERANCE_M or max_rotation_error > ROTATION_TOLERANCE_RAD:
        raise SystemExit(
            "FK validation failed: LeRobot/Placo and MuJoCo gripperframe poses do not agree."
        )


if __name__ == "__main__":
    main()
