from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import mujoco.viewer  # type: ignore[import-not-found]

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from mujoco_sim.openai_cup_agent import (
    DEFAULT_AGENT_CAMERAS,
    DEFAULT_AGENT_MODEL,
    DEFAULT_GRIPPER_DURATION,
    DEFAULT_MAX_AGENT_STEPS,
    DEFAULT_MOVE_DURATION,
    build_observation,
    create_agent_context,
    evaluate_pick_place_success,
    run_agent_steps_sync,
    write_json,
)
from mujoco_sim.run_cup_pickup import PickupConfig


DEFAULT_ARTIFACT_ROOT = ROOT_DIR / "openai_cup_agent_runs"
DEFAULT_MAX_ATTEMPTS = 10


def load_local_env() -> None:
    env_paths = (ROOT_DIR / ".env.local", ROOT_DIR / ".env")
    try:
        from dotenv import load_dotenv  # type: ignore[import-not-found]
    except ImportError:
        for env_path in env_paths:
            if not env_path.exists():
                continue
            for line in env_path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#") or "=" not in stripped:
                    continue
                key, value = stripped.split("=", 1)
                key = key.removeprefix("export ").strip()
                if key and key not in os.environ:
                    os.environ[key] = value.strip().strip("'\"")
        return

    for env_path in env_paths:
        load_dotenv(env_path, override=False)


def parse_camera_names(values: list[str] | None) -> tuple[str, ...]:
    if not values:
        return DEFAULT_AGENT_CAMERAS
    names: list[str] = []
    for value in values:
        names.extend(part.strip() for part in value.split(",") if part.strip())
    return tuple(dict.fromkeys(names))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the OpenAI Agents SDK cup manipulation agent in MuJoCo.")
    parser.add_argument("--scene", type=Path, default=PickupConfig.scene_path, help="MJCF cup scene to load.")
    parser.add_argument("--model", default=DEFAULT_AGENT_MODEL, help="OpenAI model name for the agent.")
    parser.add_argument(
        "--max-agent-steps",
        type=int,
        default=DEFAULT_MAX_AGENT_STEPS,
        help="Maximum observation/action steps to run.",
    )
    parser.add_argument(
        "--camera",
        action="append",
        dest="cameras",
        help="Camera to send to the model. Repeat or pass comma-separated names.",
    )
    parser.add_argument("--width", type=int, default=640, help="Rendered camera width in pixels.")
    parser.add_argument("--height", type=int, default=480, help="Rendered camera height in pixels.")
    parser.add_argument("--move-duration", type=float, default=DEFAULT_MOVE_DURATION, help="Default arm move duration.")
    parser.add_argument(
        "--gripper-duration",
        type=float,
        default=DEFAULT_GRIPPER_DURATION,
        help="Default open/close gripper duration.",
    )
    parser.add_argument("--headless", action="store_true", help="Run without opening the MuJoCo viewer.")
    parser.add_argument(
        "--no-apriltag-estimates",
        action="store_true",
        help="Skip AprilTag detection in observations; simulator truth and screenshots are still included.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and print one observation summary without calling OpenAI.",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT,
        help="Directory for timestamped run artifacts.",
    )
    parser.add_argument(
        "--improve-headless",
        action="store_true",
        help="Run bounded headless attempts until pick-and-place succeeds or attempts are exhausted.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=DEFAULT_MAX_ATTEMPTS,
        help="Maximum attempts for --improve-headless.",
    )
    return parser.parse_args()


def summarize_dry_run_observation(observation: dict[str, Any], image_urls: dict[str, str]) -> dict[str, Any]:
    return {
        "step_index": observation["step_index"],
        "scene_path": observation["scene_path"],
        "current_joints": observation["current_joints"],
        "cups": [
            {
                "label": cup["label"],
                "body_id": cup["body"]["body_id"],
                "world_position_m": cup["body"].get("world_position_m"),
                "tag_id": cup["mounted_tag"]["tag_id"],
            }
            for cup in observation["objects"]["cups"]
        ],
        "table_tags": observation["objects"]["table_tags"],
        "cameras": [
            {
                "name": camera["name"],
                "camera_id": camera["camera_id"],
                "available": camera["available"],
                "world_position_m": camera.get("world_position_m"),
                "image_data_url_bytes": len(image_urls.get(camera["name"], "")),
            }
            for camera in observation["cameras"]
        ],
        "apriltag_estimates": observation.get("apriltag_estimates"),
        "camera_frame_paths": observation.get("camera_frame_paths"),
    }


def create_run_dir(args: argparse.Namespace) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.artifact_root / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def guidance_for_attempt(previous_evaluation: dict[str, Any] | None) -> str:
    base = (
        "Follow the deterministic pickup/place example unless recovery is needed: "
        "move_arm(apriltag_id=6,target='approach'), "
        "move_arm(apriltag_id=6,target='grasp'), close_gripper(), "
        "move_arm(apriltag_id=6,target='lift'), move_arm(apriltag_id=0,target='place_above'), "
        "move_arm(apriltag_id=0,target='place'), open_gripper(). "
        "Use raw XYZ only for recovery."
    )
    if previous_evaluation is None:
        return base
    reasons = "; ".join(previous_evaluation.get("failure_reasons", []))
    if not reasons:
        return base
    return base + " Previous attempt failed because: " + reasons


def create_context(
    args: argparse.Namespace,
    camera_names: tuple[str, ...],
    *,
    realtime: bool,
    attempt_guidance: str | None = None,
):
    config = PickupConfig(scene_path=args.scene)
    return create_agent_context(
        config,
        render_width=args.width,
        render_height=args.height,
        camera_name=camera_names[0],
        realtime=realtime,
        move_duration=args.move_duration,
        gripper_duration=args.gripper_duration,
        attempt_guidance=attempt_guidance,
    )


def write_attempt_summary(
    attempt_dir: Path,
    *,
    attempt_index: int,
    evaluation: dict[str, Any],
    step_count: int,
    guidance: str | None,
) -> dict[str, Any]:
    summary = {
        "attempt_index": attempt_index,
        "success": evaluation["success"],
        "step_count": step_count,
        "guidance": guidance,
        "evaluation": evaluation,
    }
    write_json(attempt_dir / "summary.json", summary)
    return summary


def print_attempt_summary(summary: dict[str, Any], attempt_dir: Path) -> None:
    evaluation = summary["evaluation"]
    print("\n=== Attempt summary ===")
    print(f"attempt: {summary['attempt_index']}")
    print(f"success: {summary['success']}")
    print(f"steps: {summary['step_count']}")
    print(f"artifact_dir: {attempt_dir}")
    print(
        "metrics: "
        f"lift_delta={evaluation['lift_delta_m']}m "
        f"xy_error={evaluation['xy_error_m']}m "
        f"z_error={evaluation['z_error_m']}m "
        f"released={evaluation['released']}"
    )
    if evaluation["failure_reasons"]:
        print("failure reasons:")
        for reason in evaluation["failure_reasons"]:
            print(f"  - {reason}")


def run_headless(args: argparse.Namespace, camera_names: tuple[str, ...]) -> None:
    run_dir = create_run_dir(args)
    attempt_dir = run_dir / "attempt_01"
    context = create_context(args, camera_names, realtime=False, attempt_guidance=guidance_for_attempt(None))
    if args.dry_run:
        observation, image_urls, _camera_frame_paths = build_observation(
            context,
            step_index=0,
            camera_names=camera_names,
            include_apriltag_estimates=not args.no_apriltag_estimates,
            artifact_dir=attempt_dir,
        )
        summary = summarize_dry_run_observation(observation, image_urls)
        write_json(attempt_dir / "dry_run_summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        print(f"dry-run artifacts: {attempt_dir}")
        return

    step_results = run_agent_steps_sync(
        context,
        model=args.model,
        max_agent_steps=args.max_agent_steps,
        camera_names=camera_names,
        include_apriltag_estimates=not args.no_apriltag_estimates,
        artifact_dir=attempt_dir,
    )
    evaluation = evaluate_pick_place_success(context)
    summary = write_attempt_summary(
        attempt_dir,
        attempt_index=1,
        evaluation=evaluation,
        step_count=len(step_results),
        guidance=context.attempt_guidance,
    )
    print_attempt_summary(summary, attempt_dir)


def run_headless_improvement(args: argparse.Namespace, camera_names: tuple[str, ...]) -> None:
    run_dir = create_run_dir(args)
    previous_evaluation: dict[str, Any] | None = None
    summaries: list[dict[str, Any]] = []
    for attempt_index in range(1, args.max_attempts + 1):
        print(f"\n### Headless improvement attempt {attempt_index}/{args.max_attempts} ###")
        attempt_dir = run_dir / f"attempt_{attempt_index:02d}"
        guidance = guidance_for_attempt(previous_evaluation)
        context = create_context(args, camera_names, realtime=False, attempt_guidance=guidance)
        step_results = run_agent_steps_sync(
            context,
            model=args.model,
            max_agent_steps=args.max_agent_steps,
            camera_names=camera_names,
            include_apriltag_estimates=not args.no_apriltag_estimates,
            artifact_dir=attempt_dir,
        )
        evaluation = evaluate_pick_place_success(context)
        summary = write_attempt_summary(
            attempt_dir,
            attempt_index=attempt_index,
            evaluation=evaluation,
            step_count=len(step_results),
            guidance=guidance,
        )
        summaries.append(summary)
        print_attempt_summary(summary, attempt_dir)
        if evaluation["success"]:
            break
        previous_evaluation = evaluation

    run_summary = {
        "success": any(summary["success"] for summary in summaries),
        "attempts": summaries,
    }
    write_json(run_dir / "run_summary.json", run_summary)
    print(f"\nrecursive headless artifacts: {run_dir}")


def run_with_viewer(args: argparse.Namespace, camera_names: tuple[str, ...]) -> None:
    run_dir = create_run_dir(args)
    attempt_dir = run_dir / "attempt_01"
    context = create_context(args, camera_names, realtime=True, attempt_guidance=guidance_for_attempt(None))
    with mujoco.viewer.launch_passive(context.env.model, context.env.data) as viewer:
        context.viewer = viewer
        context.env.viewer = viewer
        if args.dry_run:
            observation, image_urls, _camera_frame_paths = build_observation(
                context,
                step_index=0,
                camera_names=camera_names,
                include_apriltag_estimates=not args.no_apriltag_estimates,
                artifact_dir=attempt_dir,
            )
            summary = summarize_dry_run_observation(observation, image_urls)
            write_json(attempt_dir / "dry_run_summary.json", summary)
            print(json.dumps(summary, indent=2, sort_keys=True))
            print(f"dry-run artifacts: {attempt_dir}")
            return

        step_results = run_agent_steps_sync(
            context,
            model=args.model,
            max_agent_steps=args.max_agent_steps,
            camera_names=camera_names,
            include_apriltag_estimates=not args.no_apriltag_estimates,
            artifact_dir=attempt_dir,
        )
        evaluation = evaluate_pick_place_success(context)
        summary = write_attempt_summary(
            attempt_dir,
            attempt_index=1,
            evaluation=evaluation,
            step_count=len(step_results),
            guidance=context.attempt_guidance,
        )
        print_attempt_summary(summary, attempt_dir)


def main() -> None:
    load_local_env()
    args = parse_args()
    camera_names = parse_camera_names(args.cameras)
    if args.improve_headless:
        if not args.headless:
            raise SystemExit("--improve-headless requires --headless.")
        run_headless_improvement(args, camera_names)
    elif args.headless:
        run_headless(args, camera_names)
    else:
        run_with_viewer(args, camera_names)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        if "mjpython" in str(exc):
            raise SystemExit(
                "MuJoCo viewer on macOS requires mjpython. Run with --headless or use:\n"
                "  mjpython mujoco_sim/run_openai_cup_agent.py"
            ) from exc
        raise
