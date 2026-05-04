# SO-101 MuJoCo Simulation

This repo contains the MuJoCo setup for the ECE 4560 SO-101 assignment. It loads the SO-101 model from `simulation_code/model/scene.xml` and runs a simple motion sequence:

1. Start from a hardware-like pose.
2. Move all joints to zero over 2 seconds.
3. Hold for 2 seconds.
4. Return to the starting pose over 2 seconds.

## Requirements

- macOS, Linux, or Windows with a working graphical desktop
- Conda or Miniforge
- Python environment named `whisk-agent`

On macOS, use `mjpython` for scripts that open the MuJoCo viewer. MuJoCo installs this command with the Python package.

## Fresh Setup

From the repo root:

```bash
cd /path/to/agent-1
```

Create the conda environment if it does not already exist:

```bash
conda create -n whisk-agent python=3.14 -y
```

Activate it:

```bash
conda activate whisk-agent
```

Install MuJoCo:

```bash
python -m pip install --upgrade pip
python -m pip install mujoco
```

Verify MuJoCo imports:

```bash
python -c "import mujoco; print(mujoco.__version__)"
```

## Model Files

The expected model layout is:

```text
simulation_code/
└── model/
    ├── scene.xml
    ├── so101_new_calib.xml
    ├── so101_new_calib.urdf
    └── assets/
        └── *.stl
```

If `simulation_code/model/scene.xml` is missing, download the SO-101 model files from:

https://github.com/TheRobotStudio/SO-ARM100/tree/main/Simulation/SO101

Place the contents of that `SO101` folder inside `simulation_code/model/`.

## Run The Viewer

To open the SO-101 model directly:

```bash
conda activate whisk-agent
cd /path/to/agent-1
mjpython -m mujoco.viewer --mjcf=simulation_code/model/scene.xml
```

If you are not on macOS, this may also work:

```bash
python -m mujoco.viewer --mjcf=simulation_code/model/scene.xml
```

## Run The Starting Pose Script

This holds the robot at the starting pose for 30 seconds:

```bash
conda activate whisk-agent
cd /path/to/agent-1
mjpython run_mujoco_simulation_startingpose.py
```

## Run The Motion Script

This runs the assignment motion:

```bash
conda activate whisk-agent
cd /path/to/agent-1
mjpython run_mujoco_simulation.py
```

The script uses this starting pose:

```python
STARTING_POSITION = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -45.0,
    "elbow_flex": 90.0,
    "wrist_flex": -45.0,
    "wrist_roll": 0.0,
    "gripper": 0.0,
}
```

## Troubleshooting

If `mujoco` cannot be imported, make sure the `whisk-agent` environment is active:

```bash
conda activate whisk-agent
python -m pip show mujoco
```

If the viewer fails on macOS, use `mjpython` instead of `python`:

```bash
mjpython run_mujoco_simulation.py
```

If MuJoCo cannot find meshes, confirm that `scene.xml`, `so101_new_calib.xml`, and the `assets/` folder are all inside `simulation_code/model/`.
