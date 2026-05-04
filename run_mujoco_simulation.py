from pathlib import Path

import mujoco
import mujoco.viewer

from so101_mujoco_utils import hold_position, move_to_pose, set_initial_pose


MODEL_PATH = Path(__file__).parent / "SO101" / "scene.xml"

# On macOS, run this viewer with:
#   conda activate whisk-agent
#   mjpython run_mujoco_simulation.py

STARTING_POSITION = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -45.0,
    "elbow_flex": 90.0,
    "wrist_flex": -45.0,
    "wrist_roll": 0.0,
    "gripper": 0.0,
}

DESIRED_POSITION = {
    "shoulder_pan": 0.0,
    "shoulder_lift": 0.0,
    "elbow_flex": 0.0,
    "wrist_flex": 0.0,
    "wrist_roll": 0.0,
    "gripper": 0.0,
}


def main():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    set_initial_pose(model, data, STARTING_POSITION)

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            move_to_pose(model, data, viewer, DESIRED_POSITION, duration=2.0)
            hold_position(model, data, viewer, duration=2.0)
            move_to_pose(model, data, viewer, STARTING_POSITION, duration=2.0)
            hold_position(model, data, viewer, duration=2.0)
    except RuntimeError as exc:
        if "mjpython" in str(exc):
            raise SystemExit(
                "MuJoCo viewer on macOS requires mjpython. Run:\n"
                "  conda activate whisk-agent\n"
                "  mjpython run_mujoco_simulation.py"
            ) from exc
        raise


if __name__ == "__main__":
    main()
