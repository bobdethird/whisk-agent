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

from mujoco_sim.gemini_cup_agent import (
    DEFAULT_AGENT_CAMERAS,
    DEFAULT_GEMINI_AGENT_MODE,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_GEMINI_TEMPERATURE,
    DEFAULT_GEMINI_THINKING_BUDGET,
    DEFAULT_GRIPPER_DURATION,
    DEFAULT_MAX_AGENT_STEPS,
    DEFAULT_MAX_PLAN_ACTIONS,
    DEFAULT_MOVE_DURATION,
    DEFAULT_TASK_SPEC,
    GEMINI_TASK_SPECS,
    GeminiTaskSpec,
    build_observation,
    create_gemini_context,
    run_gemini_steps,
    write_gemini_json,
)
from mujoco_sim.run_cup_pickup import PickupConfig


DEFAULT_ARTIFACT_ROOT = ROOT_DIR / "gemini_cup_agent_runs"
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


def parse_thinking_budget(value: str) -> int | None:
    if value.lower() in {"none", "off"}:
        return None
    return int(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Gemini Robotics-ER cup manipulation planner in MuJoCo.")
    parser.add_argument("--scene", type=Path, default=PickupConfig.scene_path, help="MJCF cup scene to load.")
    parser.add_argument("--model", default=DEFAULT_GEMINI_MODEL, help="Gemini model name for the planner.")
    parser.add_argument(
        "--agent-mode",
        choices=("tools", "json"),
        default=DEFAULT_GEMINI_AGENT_MODE,
        help="Use Gemini function calling tools by default, or the legacy structured JSON planner.",
    )
    parser.add_argument(
        "--task",
        choices=tuple(GEMINI_TASK_SPECS),
        default=DEFAULT_TASK_SPEC.name,
        help="Task policy/prompt spec to use.",
    )
    parser.add_argument(
        "--task-instruction",
        help="Override attempt guidance for --task generic, e.g. 'move above tag 0 then finish'.",
    )
    parser.add_argument(
        "--max-agent-steps",
        type=int,
        default=DEFAULT_MAX_AGENT_STEPS,
        help="Maximum plan/execute cycles to run.",
    )
    parser.add_argument(
        "--max-plan-actions",
        type=int,
        default=DEFAULT_MAX_PLAN_ACTIONS,
        help="Maximum robot API calls accepted in one Gemini plan.",
    )
    parser.add_argument(
        "--camera",
        action="append",
        dest="cameras",
        help="Camera to send to Gemini. Repeat or pass comma-separated names.",
    )
    parser.add_argument("--width", type=int, default=640, help="Rendered camera width in pixels.")
    parser.add_argument("--height", type=int, default=480, help="Rendered camera height in pixels.")
    parser.add_argument("--temperature", type=float, default=DEFAULT_GEMINI_TEMPERATURE, help="Gemini sampling temperature.")
    parser.add_argument(
        "--thinking-budget",
        type=parse_thinking_budget,
        default=DEFAULT_GEMINI_THINKING_BUDGET,
        help="Gemini thinking budget. Use 'none' to omit the setting.",
    )
    parser.add_argument("--api-retries", type=int, default=2, help="Retries for retryable Gemini API errors.")
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
        help="Build and print one observation summary without calling Gemini.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Call Gemini and validate one structured action plan without executing robot actions.",
    )
    parser.add_argument(
        "--reference-plan",
        action="store_true",
        help="Use the local deterministic cup plan instead of calling Gemini, useful for executor verification.",
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
        help="Run bounded headless attempts until the selected task succeeds or attempts are exhausted.",
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


def task_spec_for_args(args: argparse.Namespace) -> GeminiTaskSpec:
    return GEMINI_TASK_SPECS[args.task]


def guidance_for_attempt(
    task_spec: GeminiTaskSpec,
    previous_evaluation: dict[str, Any] | None,
    *,
    task_instruction: str | None = None,
) -> str:
    base = task_spec.guidance(previous_evaluation)
    if task_instruction:
        return f"User task instruction: {task_instruction} {base}"
    return base


def create_context(
    args: argparse.Namespace,
    camera_names: tuple[str, ...],
    *,
    realtime: bool,
    attempt_guidance: str | None = None,
):
    config = PickupConfig(scene_path=args.scene)
    return create_gemini_context(
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
    task_name: str,
    agent_mode: str,
    evaluation: dict[str, Any],
    step_count: int,
    guidance: str | None,
    plan_only: bool,
) -> dict[str, Any]:
    summary = {
        "attempt_index": attempt_index,
        "task": task_name,
        "agent_mode": agent_mode,
        "success": evaluation["success"],
        "plan_only": plan_only,
        "step_count": step_count,
        "guidance": guidance,
        "evaluation": evaluation,
    }
    write_gemini_json(attempt_dir / "summary.json", summary)
    return summary


def print_attempt_summary(summary: dict[str, Any], attempt_dir: Path) -> None:
    evaluation = summary["evaluation"]
    print("\n=== Gemini attempt summary ===")
    print(f"attempt: {summary['attempt_index']}")
    print(f"task: {summary['task']}")
    print(f"agent_mode: {summary['agent_mode']}")
    print(f"success: {summary['success']}")
    print(f"plan_only: {summary['plan_only']}")
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


def run_dry_observation(args: argparse.Namespace, context, camera_names: tuple[str, ...], attempt_dir: Path) -> None:
    observation, image_urls, _camera_frame_paths = build_observation(
        context,
        step_index=0,
        camera_names=camera_names,
        include_apriltag_estimates=not args.no_apriltag_estimates,
        artifact_dir=attempt_dir,
    )
    summary = summarize_dry_run_observation(observation, image_urls)
    write_gemini_json(attempt_dir / "dry_run_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"dry-run artifacts: {attempt_dir}")


def run_headless(args: argparse.Namespace, camera_names: tuple[str, ...]) -> None:
    run_dir = create_run_dir(args)
    attempt_dir = run_dir / "attempt_01"
    task_spec = task_spec_for_args(args)
    context = create_context(
        args,
        camera_names,
        realtime=False,
        attempt_guidance=guidance_for_attempt(task_spec, None, task_instruction=args.task_instruction),
    )
    if args.dry_run:
        run_dry_observation(args, context, camera_names, attempt_dir)
        return

    step_results = run_gemini_steps(
        context,
        model=args.model,
        max_agent_steps=args.max_agent_steps,
        max_plan_actions=args.max_plan_actions,
        camera_names=camera_names,
        include_apriltag_estimates=not args.no_apriltag_estimates,
        artifact_dir=attempt_dir,
        execute_actions=not args.plan_only,
        temperature=args.temperature,
        thinking_budget=args.thinking_budget,
        reference_plan=args.reference_plan,
        task_spec=task_spec,
        agent_mode=args.agent_mode,
        api_retries=args.api_retries,
    )
    evaluation = task_spec.evaluator(context)
    summary = write_attempt_summary(
        attempt_dir,
        attempt_index=1,
        task_name=task_spec.name,
        agent_mode=args.agent_mode,
        evaluation=evaluation,
        step_count=len(step_results),
        guidance=context.attempt_guidance,
        plan_only=args.plan_only,
    )
    print_attempt_summary(summary, attempt_dir)


def run_headless_improvement(args: argparse.Namespace, camera_names: tuple[str, ...]) -> None:
    run_dir = create_run_dir(args)
    previous_evaluation: dict[str, Any] | None = None
    summaries: list[dict[str, Any]] = []
    task_spec = task_spec_for_args(args)
    for attempt_index in range(1, args.max_attempts + 1):
        print(f"\n### Gemini headless improvement attempt {attempt_index}/{args.max_attempts} ###")
        attempt_dir = run_dir / f"attempt_{attempt_index:02d}"
        guidance = guidance_for_attempt(task_spec, previous_evaluation, task_instruction=args.task_instruction)
        context = create_context(args, camera_names, realtime=False, attempt_guidance=guidance)
        step_results = run_gemini_steps(
            context,
            model=args.model,
            max_agent_steps=args.max_agent_steps,
            max_plan_actions=args.max_plan_actions,
            camera_names=camera_names,
            include_apriltag_estimates=not args.no_apriltag_estimates,
            artifact_dir=attempt_dir,
            execute_actions=not args.plan_only,
            temperature=args.temperature,
            thinking_budget=args.thinking_budget,
            reference_plan=args.reference_plan,
            task_spec=task_spec,
            agent_mode=args.agent_mode,
            api_retries=args.api_retries,
        )
        evaluation = task_spec.evaluator(context)
        summary = write_attempt_summary(
            attempt_dir,
            attempt_index=attempt_index,
            task_name=task_spec.name,
            agent_mode=args.agent_mode,
            evaluation=evaluation,
            step_count=len(step_results),
            guidance=guidance,
            plan_only=args.plan_only,
        )
        summaries.append(summary)
        print_attempt_summary(summary, attempt_dir)
        if args.plan_only or evaluation["success"]:
            break
        previous_evaluation = evaluation

    run_summary = {
        "success": any(summary["success"] for summary in summaries),
        "plan_only": args.plan_only,
        "task": task_spec.name,
        "agent_mode": args.agent_mode,
        "attempts": summaries,
    }
    write_gemini_json(run_dir / "run_summary.json", run_summary)
    print(f"\nGemini recursive headless artifacts: {run_dir}")


def run_with_viewer(args: argparse.Namespace, camera_names: tuple[str, ...]) -> None:
    run_dir = create_run_dir(args)
    attempt_dir = run_dir / "attempt_01"
    task_spec = task_spec_for_args(args)
    context = create_context(
        args,
        camera_names,
        realtime=True,
        attempt_guidance=guidance_for_attempt(task_spec, None, task_instruction=args.task_instruction),
    )
    with mujoco.viewer.launch_passive(context.env.model, context.env.data) as viewer:
        context.viewer = viewer
        context.env.viewer = viewer
        if args.dry_run:
            run_dry_observation(args, context, camera_names, attempt_dir)
            return

        step_results = run_gemini_steps(
            context,
            model=args.model,
            max_agent_steps=args.max_agent_steps,
            max_plan_actions=args.max_plan_actions,
            camera_names=camera_names,
            include_apriltag_estimates=not args.no_apriltag_estimates,
            artifact_dir=attempt_dir,
            execute_actions=not args.plan_only,
            temperature=args.temperature,
            thinking_budget=args.thinking_budget,
            reference_plan=args.reference_plan,
            task_spec=task_spec,
            agent_mode=args.agent_mode,
            api_retries=args.api_retries,
        )
        evaluation = task_spec.evaluator(context)
        summary = write_attempt_summary(
            attempt_dir,
            attempt_index=1,
            task_name=task_spec.name,
            agent_mode=args.agent_mode,
            evaluation=evaluation,
            step_count=len(step_results),
            guidance=context.attempt_guidance,
            plan_only=args.plan_only,
        )
        print_attempt_summary(summary, attempt_dir)


def main() -> None:
    load_local_env()
    args = parse_args()
    if args.max_plan_actions < 1:
        raise SystemExit("--max-plan-actions must be at least 1.")
    if args.api_retries < 0:
        raise SystemExit("--api-retries must be non-negative.")
    if args.reference_plan and GEMINI_TASK_SPECS[args.task].reference_plan is None:
        raise SystemExit(f"--reference-plan is not available for task {args.task!r}.")
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
                "  mjpython mujoco_sim/run_gemini_cup_agent.py"
            ) from exc
        raise
