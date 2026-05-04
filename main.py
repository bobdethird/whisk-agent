from __future__ import annotations

import argparse
from pathlib import Path

import mujoco.viewer  # type: ignore[import-not-found]

from agent import run as run_agent
from mujoco_sim.apriltag_world_config import DEFAULT_RENDER_HEIGHT, DEFAULT_RENDER_WIDTH
from sim_env import DEFAULT_CAMERA, MODEL_PATH, create_env
from so101_mujoco_utils import hold_position, hold_position_until_closed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SO-101 AprilTag-following agent in MuJoCo.")
    parser.add_argument("--scene", type=Path, default=MODEL_PATH, help="MJCF scene to load.")
    parser.add_argument("--camera", default=DEFAULT_CAMERA, help="Preferred MuJoCo camera for AprilTag detection.")
    parser.add_argument("--width", type=int, default=DEFAULT_RENDER_WIDTH, help="Rendered camera width in pixels.")
    parser.add_argument("--height", type=int, default=DEFAULT_RENDER_HEIGHT, help="Rendered camera height in pixels.")
    parser.add_argument("--hover-height", type=float, default=0.02, help="Meters above each detected tag to move to.")
    parser.add_argument("--move-duration", type=float, default=2.0, help="Seconds to spend moving between tag targets.")
    parser.add_argument("--pause-duration", type=float, default=5.0, help="Seconds to pause before moving to the next tag.")
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    env = create_env(
        scene_path=args.scene,
        camera_name=args.camera,
        render_width=args.width,
        render_height=args.height,
    )

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        env.viewer = viewer
        hold_position(env.model, env.data, viewer, duration=0.5)
        run_agent(
            env,
            hover_height=args.hover_height,
            move_duration=args.move_duration,
            pause_duration=args.pause_duration,
        )
        print("Agent run complete. Close the MuJoCo viewer window to exit.")
        hold_position_until_closed(env.model, env.data, viewer)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        if "mjpython" in str(exc):
            raise SystemExit(
                "MuJoCo viewer on macOS requires mjpython. Run:\n"
                "  cd /Users/cadenli/Documents/launchpad/whisk/agent-1\n"
                "  mjpython main.py"
            ) from exc
        raise
