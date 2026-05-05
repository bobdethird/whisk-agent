import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from so101_mujoco_utils import send_position_command, set_initial_pose


MODEL_PATH = ROOT_DIR / "simulation_code" / "model" / "scene.xml"
HORIZONTAL_WRIST_ROLL_DEGREES = -90.0

STARTING_POSITION = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -45.0,
    "elbow_flex": 90.0,
    "wrist_flex": -45.0,
    "wrist_roll": HORIZONTAL_WRIST_ROLL_DEGREES,
    "gripper": 0.0,
}


def main():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    set_initial_pose(model, data, STARTING_POSITION)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start = time.time()
        while viewer.is_running() and time.time() - start < 30:
            step_start = time.time()

            send_position_command(data, STARTING_POSITION)
            mujoco.mj_step(model, data)
            viewer.sync()

            sleep_time = model.opt.timestep - (time.time() - step_start)
            if sleep_time > 0:
                time.sleep(sleep_time)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        if "mjpython" in str(exc):
            raise SystemExit(
                "MuJoCo viewer on macOS requires mjpython. Run:\n"
                "  conda activate whisk-agent\n"
                "  cd /Users/cadenli/Documents/launchpad/whisk/agent-1\n"
                "  mjpython mujoco_sim/run_mujoco_simulation_startingpose.py"
            ) from exc
        raise
