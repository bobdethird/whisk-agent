import sys
from pathlib import Path

import mujoco  # type: ignore[import-not-found]
import mujoco.viewer  # type: ignore[import-not-found]
import numpy as np  # type: ignore[import-not-found]

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from sim_env import STARTING_POSITION
from so101_kinematics import (
    FIXED_JAW_TOOL_POINT,
    SO101Kinematics,
    gripperframe_pose_to_tool_target_pose,
    tool_target_pose_to_gripperframe_pose,
    translated_pose,
)
from so101_mujoco_utils import hold_position, move_to_pose, set_initial_pose


MODEL_PATH = ROOT_DIR / "simulation_code" / "model" / "scene.xml"

TARGET_FORWARD_RANGE_M = 0.20
TARGET_LATERAL_RANGE_M = 0.20
TARGET_MIN_LATERAL_M = 0.04
TARGET_Z_RANGE_M = (0.02, 0.2)
IK_POSITION_TOLERANCE_M = 2e-3
MAX_TARGET_ATTEMPTS = 60


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


def sample_target_offset(rng: np.random.Generator) -> np.ndarray:
    lateral_offset = rng.uniform(TARGET_MIN_LATERAL_M, TARGET_LATERAL_RANGE_M)
    if rng.random() < 0.5:
        lateral_offset *= -1.0

    return np.array(
        [
            rng.uniform(-TARGET_FORWARD_RANGE_M, TARGET_FORWARD_RANGE_M),
            lateral_offset,
            rng.uniform(*TARGET_Z_RANGE_M),
        ]
    )


def solve_random_target(
    kinematics: SO101Kinematics,
    starting_tool_pose: np.ndarray,
) -> tuple[np.ndarray, dict[str, float], np.ndarray, float]:
    rng = np.random.default_rng()
    best_error = float("inf")
    best_result = None

    for _ in range(MAX_TARGET_ATTEMPTS):
        target_offset = sample_target_offset(rng)
        target_tool_pose = translated_pose(starting_tool_pose, target_offset)
        target_ee_pose = tool_target_pose_to_gripperframe_pose(target_tool_pose, FIXED_JAW_TOOL_POINT)
        target_position = kinematics.inverse_kinematics(
            STARTING_POSITION,
            target_ee_pose,
            position_weight=1.0,
            orientation_weight=0.01,
            gripper=STARTING_POSITION["gripper"],
        )
        solved_pose = kinematics.forward_kinematics(target_position, frame="mujoco")
        solved_tool_pose = gripperframe_pose_to_tool_target_pose(solved_pose, FIXED_JAW_TOOL_POINT)
        position_error = float(np.linalg.norm(target_tool_pose[:3, 3] - solved_tool_pose[:3, 3]))

        if position_error < best_error:
            best_error = position_error
            best_result = (target_tool_pose, target_position, target_offset, position_error)

        if position_error <= IK_POSITION_TOLERANCE_M:
            return target_tool_pose, target_position, target_offset, position_error

    if best_result is None:
        raise RuntimeError("Unable to generate a Cartesian IK target.")
    return best_result


def main():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    kinematics = SO101Kinematics()
    starting_gripperframe_pose = kinematics.forward_kinematics(STARTING_POSITION, frame="mujoco")
    starting_tool_pose = gripperframe_pose_to_tool_target_pose(starting_gripperframe_pose, FIXED_JAW_TOOL_POINT)
    target_tool_pose, target_position, target_offset, position_error = solve_random_target(
        kinematics,
        starting_tool_pose,
    )

    print(f"Using kinematics backend: {kinematics.backend_name}")
    print(
        "Random target offset: "
        f"forward/back x={target_offset[0]:.3f} m, "
        f"left/right y={target_offset[1]:.3f} m, "
        f"up z={target_offset[2]:.3f} m"
    )
    print(f"IK tool point: {FIXED_JAW_TOOL_POINT}")
    print(f"IK position error: {position_error:.6f} m")
    print("Target joint position:")
    for joint, value in target_position.items():
        print(f"  {joint}: {value:.3f}")

    set_initial_pose(model, data, STARTING_POSITION)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        show_target(viewer, target_tool_pose)
        hold_position(model, data, viewer, duration=1.0)
        move_to_pose(model, data, viewer, target_position, duration=3.0)
        hold_position(model, data, viewer, duration=3.0)
        move_to_pose(model, data, viewer, STARTING_POSITION, duration=3.0)
        hold_position(model, data, viewer, duration=1.0)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        if "mjpython" in str(exc):
            raise SystemExit(
                "MuJoCo viewer on macOS requires mjpython. Run:\n"
                "  conda activate whisk-agent\n"
                "  cd /Users/cadenli/Documents/launchpad/whisk/agent-1\n"
                "  mjpython mujoco_sim/run_cartesian_ik_demo.py"
            ) from exc
        raise
