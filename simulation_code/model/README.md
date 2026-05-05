# SO101 Robot - URDF and MuJoCo Description

This repository contains the URDF and MuJoCo (MJCF) files for the SO101 robot.

## Overview

- `so101.xml` is the active MuJoCo robot model and is vendored from the Google DeepMind MuJoCo Menagerie SO-101 package.
- `robotstudio_so101/` keeps the upstream Menagerie package, including license, README, sample scenes, camera mount assets, and gripper collision meshes.
- `so101_new_calib.xml` and `so101_new_calib.urdf` are retained as the original TheRobotStudio/LeRobot-compatible source models used by the kinematics adapter.
- Base collision meshes were removed from the original generated model due to problematic collision behavior during simulation and planning.

## Calibration Methods

The generated MuJoCo file `scene.xml` includes `so101.xml` by default.

The original source models provide two calibrated SO101 robot files:

- **New Calibration (Default)**: Each joint's virtual zero is set to the **middle** of its joint range. Use -> `so101_new_calib.xml`. 
- **Old Calibration**: Each joint's virtual zero is set to the configuration where the robot is **fully extended horizontally**. Use -> `so101_old_calib.xml`.

To switch calibration methods, update `ROBOT_MODEL_PATH` in `mujoco_sim/apriltag_world_config.py` and regenerate `scene.xml`.

## Motor Parameters

Motor properties for the STS3215 motors used in the robot are adapted from the [Open Duck Mini project](https://github.com/apirrone/Open_Duck_Mini).

## Gripper Note

In LeRobot, the gripper is represented as a **linear joint**, where:

* `0` = fully closed
* `100` = fully open

This mapping is **not yet reflected** in the current URDF and MuJoCo files. 

---

Feel free to open an issue or contribute improvements!
