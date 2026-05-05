"""Launch the MuJoCo passive viewer for a scene.

Usage:
  mjpython simulation_code/view_scene.py [path/to/scene.xml]

Defaults to simulation_code/model/scene.xml if no path is given.

Why this exists: `mjpython -m mujoco.viewer --mjcf=...` crashes with
"RuntimeError: Caught an unknown exception!" on conda Python under macOS
because `viewer.launch_from_path` constructs the GUI on the Python (secondary)
thread instead of dispatching to the AppKit main thread. `launch_passive`
takes the dispatched path and works correctly.
"""

import os
import sys
import time

import mujoco
import mujoco.viewer


def main() -> None:
    default = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "model", "scene.xml"
    )
    path = sys.argv[1] if len(sys.argv) > 1 else default

    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()
            remaining = model.opt.timestep - (time.time() - step_start)
            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()
