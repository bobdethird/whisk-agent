import time
from pathlib import Path

import mujoco
import mujoco.viewer

MODEL_PATH = Path(__file__).parent / "SO101" / "scene.xml"

# On macOS, run this viewer with: conda activate whisk-agent && mjpython temp.py
m = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
d = mujoco.MjData(m)

try:
    with mujoco.viewer.launch_passive(m, d) as viewer:
        # scene.xml includes the robot, floor, lighting, and position actuators.
        start = time.time()
        while viewer.is_running() and time.time() - start < 30:
            step_start = time.time()

            mujoco.mj_step(m, d)

            with viewer.lock():
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

            viewer.sync()

            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
except RuntimeError as exc:
    if "mjpython" in str(exc):
        raise SystemExit(
            "MuJoCo viewer on macOS requires mjpython. Run:\n"
            "  conda activate whisk-agent\n"
            "  mjpython temp.py"
        ) from exc
    raise
