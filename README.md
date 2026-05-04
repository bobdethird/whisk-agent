# SO-101 MuJoCo Simulation

This repo contains a MuJoCo setup for viewing and simulating the SO-101 robot arm.

## Model Files

The MuJoCo model lives in `SO101/`:

- `SO101/scene.xml`: the main MuJoCo scene. Use this for simulation.
- `SO101/so101_new_calib.xml`: the robot model included by `scene.xml`.
- `SO101/assets/`: mesh files used by the model.
- `SO101/so101_new_calib.urdf`: URDF version of the robot. It is useful for inspection, but it does not define MuJoCo position actuators.

Use `scene.xml` for actuator-based motion. The URDF loads in MuJoCo, but it has `nu = 0`, meaning there are no controls available through `data.ctrl`.

## Environment

Use the existing conda environment:

```bash
conda activate whisk-agent
```

MuJoCo is already installed in this environment. To verify:

```bash
python -c "import mujoco; print(mujoco.__version__)"
```

If MuJoCo is missing, install it inside the environment:

```bash
python -m pip install mujoco
```

## macOS Viewer Requirement

On macOS, MuJoCo's passive viewer must be run with `mjpython`, not regular `python`.

Use:

```bash
mjpython temp.py
```

instead of:

```bash
python temp.py
```

If you use `python`, you may see:

```text
RuntimeError: `launch_passive` requires that the Python script be run under `mjpython` on macOS
```

## Run The Viewer

`temp.py` opens the SO-101 model in the MuJoCo viewer for 30 seconds.

```bash
conda activate whisk-agent
mjpython temp.py
```

## Run The Motion Demo

`run_mujoco_simulation.py` uses the MuJoCo position actuators to:

1. Start from a nonzero arm pose.
2. Move to all-zero joint positions over 2 seconds.
3. Hold that pose for 2 seconds.
4. Return to the starting pose over 2 seconds.
5. Hold the starting pose.

Run it with:

```bash
conda activate whisk-agent
mjpython run_mujoco_simulation.py
```

## Helper Functions

`so101_mujoco_utils.py` contains the shared simulation helpers:

- `convert_to_dictionary(qpos)`: converts MuJoCo radians to a robot-style joint dictionary.
- `convert_to_list(position_dict)`: converts the joint dictionary back to MuJoCo units.
- `set_initial_pose(model, data, position_dict)`: sets the initial joint positions.
- `send_position_command(data, position_dict)`: writes position targets to `data.ctrl`.
- `move_to_pose(model, data, viewer, desired_position, duration)`: interpolates to a target pose.
- `hold_position(model, data, viewer, duration)`: holds the current pose.

## Troubleshooting

If the robot loads but does not respond to commands, make sure the script is loading:

```python
SO101/scene.xml
```

not:

```python
SO101/so101_new_calib.urdf
```

The XML scene has six position actuators. The URDF has joints but no MuJoCo actuators.
