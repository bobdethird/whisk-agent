#!/usr/bin/env python3
"""Grasp / tag-offset visualization using the same SimEnv + MuJoCo pipeline as cup pickup.

By default, renders offscreen PNGs (AprilTag detection + fusion + grasp_library,
then green / blue / yellow spheres in `mujoco.Renderer`).

Realtime options:

  --viewer     Opens the passive MuJoCo viewer with those markers frozen in the
               scene; orbit with the mouse until you close the window.

  --playback   Runs the full cup pickup motion (`run_pickup` with viewer),
               same as `mjpython mujoco_sim/run_cup_pickup.py` — robot moves
               through pregrasp → grasp → lift with markers visible.

On macOS use `mjpython` for any flag that opens the GUI (conda Python threading).

Examples:

  python scripts/visualize_grasp_offsets.py
  mjpython scripts/visualize_grasp_offsets.py --viewer
  mjpython scripts/visualize_grasp_offsets.py --playback
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco  # type: ignore[import-not-found]
import numpy as np  # type: ignore[import-not-found]

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import mujoco.viewer  # type: ignore[import-not-found]

from mujoco_sim.run_cup_pickup import (
    MODEL_PATH,
    PickupConfig,
    append_grasp_marker_geoms,
    configure_pickup_env,
    estimate_cup_target_points,
    run_pickup,
    show_cup_target_points,
)
from pose_estimation import write_rgb_png
from sim_env import create_env


def render_camera_with_markers(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera_name: str,
    width: int,
    height: int,
    object_position: np.ndarray,
    grasp_position: np.ndarray,
    pregrasp_position: np.ndarray,
) -> np.ndarray:
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        renderer.update_scene(data, camera=camera_name)
        append_grasp_marker_geoms(
            renderer.scene,
            object_position,
            grasp_position,
            pregrasp_position,
        )
        return renderer.render()


def stack_horizontal(images: list[np.ndarray]) -> np.ndarray:
    if len(images) == 1:
        return images[0]
    h0 = images[0].shape[0]
    for im in images:
        if im.shape[0] != h0 or im.ndim != 3:
            raise ValueError("All camera panels must share the same height and be RGB.")
    return np.concatenate(images, axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=ROOT_DIR / "OFFSET_CALCULATIONS_example.png",
        help="Output PNG path.",
    )
    parser.add_argument(
        "--scene",
        type=Path,
        default=MODEL_PATH,
        help="Cup MJCF (default: scene_cup.xml).",
    )
    parser.add_argument(
        "--camera",
        action="append",
        dest="cameras",
        default=None,
        help="Named MuJoCo camera (repeat for side-by-side). Default: table_observer cup_observer.",
    )
    parser.add_argument("--width", type=int, default=960, help="Render width per panel.")
    parser.add_argument("--height", type=int, default=720, help="Render height per panel.")
    parser.add_argument(
        "--allow-config-cup-position-fallback",
        action="store_true",
        help="Same as cup pickup: use configured cup pose if tags are not detected.",
    )
    parser.add_argument(
        "--cup-position",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=None,
        help="Override cup spawn position (meters).",
    )
    parser.add_argument("--cup-yaw-deg", type=float, default=None, help="Cup yaw about world Z (degrees).")
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument(
        "--viewer",
        action="store_true",
        help="Open interactive passive viewer with grasp markers (orbit camera; close window to exit).",
    )
    grp.add_argument(
        "--playback",
        action="store_true",
        help="Play full cup pickup in realtime with markers (same as run_cup_pickup with viewer).",
    )
    args = parser.parse_args()

    cfg_kw: dict = {
        "allow_config_position_fallback": args.allow_config_cup_position_fallback,
        "debug_camera_frame_dir": None,
        "use_grasp_library": True,
    }
    if args.cup_position is not None:
        cfg_kw["cup_position"] = tuple(args.cup_position)
    if args.cup_yaw_deg is not None:
        cfg_kw["cup_yaw_deg"] = float(args.cup_yaw_deg)

    config = PickupConfig(scene_path=args.scene, **cfg_kw)

    if args.playback:
        print(
            "Playback: closing the viewer window ends the run.\n"
            "Markers: green=fused object origin, blue=claw IK target, yellow=pregrasp."
        )
        run_pickup(config, launch_viewer=True)
        return

    if args.viewer:
        env = create_env(scene_path=config.scene_path)
        configure_pickup_env(env, config)
        targets = estimate_cup_target_points(env, config)
        mujoco.mj_forward(env.model, env.data)
        print(
            "Interactive viewer: drag to orbit, scroll to zoom, close window to exit.\n"
            "Markers: green=fused object origin, blue=claw IK target, yellow=pregrasp."
        )
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            env.viewer = viewer
            while viewer.is_running():
                show_cup_target_points(
                    viewer,
                    targets.object_position,
                    targets.grasp_position,
                    targets.pregrasp_position,
                )
        return

    cameras = args.cameras or ["table_observer", "cup_observer"]
    env = create_env(scene_path=config.scene_path)
    configure_pickup_env(env, config)
    targets = estimate_cup_target_points(env, config)

    obj_p = targets.object_position
    grasp_p = targets.grasp_position
    pre_p = targets.pregrasp_position

    panels = [
        render_camera_with_markers(
            env.model,
            env.data,
            name,
            args.width,
            args.height,
            obj_p,
            grasp_p,
            pre_p,
        )
        for name in cameras
    ]
    combined = stack_horizontal(panels)
    write_rgb_png(args.output, combined)
    print(f"Wrote {args.output} ({combined.shape[1]}x{combined.shape[0]} px)")
    print(
        "Markers: green=fused object origin, blue=claw IK target (after grip-pad offset), "
        "yellow=pregrasp (same convention as interactive cup pickup)."
    )


if __name__ == "__main__":
    main()
