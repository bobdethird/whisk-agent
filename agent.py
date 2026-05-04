from __future__ import annotations

import mujoco  # type: ignore[import-not-found]
import numpy as np  # type: ignore[import-not-found]

from gripper import OPEN_GRIPPER, gripper
from motion import move
from pose_estimation import detect_apriltags, require_named_id
from sim_env import SimEnv
from so101_mujoco_utils import hold_position


def tag_site_name(tag_id: int) -> str:
    return f"tag36h11_{tag_id:05d}_site"


def true_tag_position(env: SimEnv, tag_id: int) -> np.ndarray:
    site_id = require_named_id(env.model, mujoco.mjtObj.mjOBJ_SITE, tag_site_name(tag_id))
    return env.data.site_xpos[site_id].copy()


def show_tag_points(env: SimEnv, true_position: np.ndarray, estimated_position: np.ndarray) -> None:
    if env.viewer is None:
        return

    mujoco.mjv_initGeom(
        env.viewer.user_scn.geoms[0],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.006, 0.0, 0.0],
        pos=true_position,
        mat=np.eye(3).flatten(),
        rgba=[1.0, 0.0, 0.0, 0.85],
    )
    mujoco.mjv_initGeom(
        env.viewer.user_scn.geoms[1],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.011, 0.0, 0.0],
        pos=estimated_position,
        mat=np.eye(3).flatten(),
        rgba=[0.0, 1.0, 0.0, 0.55],
    )
    env.viewer.user_scn.ngeom = 2
    env.viewer.sync()


def run(env: SimEnv, hover_height: float = 0.02, move_duration: float = 2.0, pause_duration: float = 5.0) -> None:
    gripper(env, "open", duration=0.4)
    estimates = detect_apriltags(env)
    if not estimates:
        raise RuntimeError("No AprilTags were detected from the configured camera views.")

    print("Detected AprilTag IDs: " + ", ".join(str(tag_id) for tag_id in estimates))
    tag_items = list(estimates.items())
    for index, (tag_id, estimate) in enumerate(tag_items):
        target = estimate.world_position + np.array([0.0, 0.0, hover_height])
        truth = true_tag_position(env, tag_id) + np.array([0.0, 0.0, hover_height])
        show_tag_points(env, truth, target)
        print(
            f"agent: moving to tag {tag_id} "
            f"from camera {estimate.camera_name} "
            f"at x={estimate.world_position[0]:.4f} "
            f"y={estimate.world_position[1]:.4f} "
            f"z={estimate.world_position[2]:.4f} m"
        )
        move(env, target, gripper_position=OPEN_GRIPPER, duration=move_duration, show_marker=False)
        if index < len(tag_items) - 1:
            print(f"agent: pausing {pause_duration:.1f}s before next tag")
            hold_position(env.model, env.data, env.viewer, duration=pause_duration)
