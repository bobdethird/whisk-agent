from __future__ import annotations

"""Follower-only SO-101 runner for the matcha waypoint plan.

This script does not run MuJoCo physics on the real robot. It uses the simulator's
IK/model only to generate joint waypoints, then sends those waypoints to a
calibrated SO-101 follower arm through LeRobot.
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np  # type: ignore[import-not-found]

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from gripper import OPEN_GRIPPER
from motion import solve_ik
from mujoco_sim.run_matcha_demo import MATCHA_CLOSED_GRIPPER, MatchaConfig, configure_matcha_env
from sim_env import STARTING_POSITION, create_env
from so101_mujoco_utils import JOINT_ORDER


@dataclass(frozen=True)
class Waypoint:
    name: str
    target: dict[str, float]
    duration: float
    hardware_gripper: float | None = None


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def _map_gripper_to_hardware(sim_gripper: float, closed_gripper: float, open_gripper: float) -> float:
    sim_alpha = (sim_gripper - MATCHA_CLOSED_GRIPPER) / (OPEN_GRIPPER - MATCHA_CLOSED_GRIPPER)
    hardware_value = closed_gripper + _clamp(sim_alpha, 0.0, 1.0) * (open_gripper - closed_gripper)
    return _clamp(hardware_value, min(closed_gripper, open_gripper), max(closed_gripper, open_gripper))


def _hardware_position(
    position: dict[str, float],
    closed_gripper: float,
    open_gripper: float,
    gripper_override: float | None = None,
) -> dict[str, float]:
    """Convert the sim convention to LeRobot's follower action convention.

    Arm joints are already degrees. The sim sometimes uses a small negative gripper
    command to squeeze contacts; the real follower gripper expects 0..100.
    """
    result = {joint: float(position[joint]) for joint in JOINT_ORDER}
    result["gripper"] = (
        _clamp(gripper_override, min(closed_gripper, open_gripper), max(closed_gripper, open_gripper))
        if gripper_override is not None
        else _map_gripper_to_hardware(result["gripper"], closed_gripper, open_gripper)
    )
    return result


def _to_action(position: dict[str, float]) -> dict[str, float]:
    return {f"{joint}.pos": position[joint] for joint in JOINT_ORDER}


def _from_observation(observation: dict[str, object]) -> dict[str, float]:
    missing = [f"{joint}.pos" for joint in JOINT_ORDER if f"{joint}.pos" not in observation]
    if missing:
        raise RuntimeError(f"Follower observation is missing joint keys: {missing}")
    return {joint: float(observation[f"{joint}.pos"]) for joint in JOINT_ORDER}


def _interpolate(start: dict[str, float], target: dict[str, float], alpha: float) -> dict[str, float]:
    return {joint: (1.0 - alpha) * start[joint] + alpha * target[joint] for joint in JOINT_ORDER}


def _solve_waypoint(env, name: str, xyz: np.ndarray, gripper: float, duration: float) -> Waypoint:
    plan = solve_ik(env, xyz, gripper_position=gripper)
    target = plan.target_pose[:3, 3]
    print(
        f"planned {name:18s}: target=({target[0]:.4f}, {target[1]:.4f}, {target[2]:.4f}) m, "
        f"gripper={gripper:.1f}, IK error={plan.position_error:.6f} m"
    )
    env.current_position = dict(plan.target_position)
    return Waypoint(name=name, target=plan.target_position, duration=duration)


def build_matcha_waypoints(
    config: MatchaConfig,
    speed_scale: float,
    preclose_gripper: float,
    whisk_vertical_amplitude: float,
) -> list[Waypoint]:
    env = create_env(scene_path=config.scene_path, camera_name="matcha_observer")
    configure_matcha_env(env, config)

    grasp_offset = np.asarray(config.grasp_target_offset, dtype=float)
    whisk_body_xyz = np.array([config.whisk_position[0], config.whisk_position[1], config.grasp_height], dtype=float)
    grasp_xyz = whisk_body_xyz + grasp_offset
    approach_xyz = grasp_xyz + np.array([0.0, 0.0, config.approach_height], dtype=float)
    lift_xyz = whisk_body_xyz + np.array([0.0, 0.0, config.lift_height], dtype=float) + grasp_offset
    cup_center_xyz = (
        np.array(
            [
                config.main_cup_position[0],
                config.main_cup_position[1],
                config.whisk_tip_height + config.whisk_tip_to_body_height,
            ],
            dtype=float,
        )
        + grasp_offset
    )
    cup_approach_xyz = cup_center_xyz + np.array([0.0, 0.0, config.cup_approach_height], dtype=float)

    def scaled(duration: float) -> float:
        return duration / speed_scale

    waypoints: list[Waypoint] = [
        Waypoint("home_open", dict(STARTING_POSITION), scaled(3.0)),
        _solve_waypoint(env, "approach_whisk", approach_xyz, OPEN_GRIPPER, scaled(config.approach_duration)),
    ]

    preclose = dict(env.current_position)
    preclose["gripper"] = config.preclose_gripper
    waypoints.append(Waypoint("preclose", preclose, scaled(config.preclose_duration), hardware_gripper=preclose_gripper))
    env.current_position = dict(preclose)

    grasp_position = _solve_waypoint(env, "descend_to_grasp", grasp_xyz, MATCHA_CLOSED_GRIPPER, scaled(config.descend_duration)).target
    descend_position = dict(grasp_position)
    descend_position["gripper"] = config.preclose_gripper
    waypoints.append(
        Waypoint("descend_to_grasp", descend_position, scaled(config.descend_duration), hardware_gripper=preclose_gripper)
    )
    env.current_position = dict(descend_position)

    closed_position = dict(grasp_position)
    waypoints.append(Waypoint("close_on_whisk", closed_position, scaled(config.close_duration)))
    env.current_position = dict(closed_position)

    waypoints.append(_solve_waypoint(env, "lift_whisk", lift_xyz, MATCHA_CLOSED_GRIPPER, scaled(config.lift_duration)))
    waypoints.append(
        _solve_waypoint(env, "cup_approach", cup_approach_xyz, MATCHA_CLOSED_GRIPPER, scaled(config.approach_duration))
    )
    waypoints.append(_solve_waypoint(env, "cup_descend", cup_center_xyz, MATCHA_CLOSED_GRIPPER, scaled(config.descend_duration)))

    waypoint_count = max(2, config.whisk_strokes * 2)
    segment_duration = scaled(config.whisk_duration / waypoint_count)
    half_stroke = 0.5 * config.whisk_stroke_length
    for waypoint_index in range(1, waypoint_count + 1):
        direction = 1.0 if waypoint_index % 2 else -1.0
        small_side_jitter = 0.003 * np.sin(np.pi * waypoint_index / 2.0)
        # Add a subtle vertical lift on alternating strokes. The baseline point is still the
        # cup-descend height, so the whisk never drives deeper into the cup than the planned pose.
        vertical_lift = whisk_vertical_amplitude if waypoint_index % 2 else 0.0
        whisk_xyz = np.array(
            [
                cup_center_xyz[0] + direction * half_stroke,
                cup_center_xyz[1] + small_side_jitter,
                cup_center_xyz[2] + vertical_lift,
            ],
            dtype=float,
        )
        waypoints.append(
            _solve_waypoint(
                env,
                f"whisk_{waypoint_index:02d}",
                whisk_xyz,
                MATCHA_CLOSED_GRIPPER,
                segment_duration,
            )
        )

    final_lift_xyz = cup_center_xyz + np.array([0.0, 0.0, config.lift_height], dtype=float)
    waypoints.append(_solve_waypoint(env, "final_lift", final_lift_xyz, MATCHA_CLOSED_GRIPPER, scaled(config.lift_duration)))
    return waypoints


def print_waypoints(waypoints: list[Waypoint], closed_gripper: float, open_gripper: float) -> None:
    print("\nHardware waypoint plan:")
    for index, waypoint in enumerate(waypoints, start=1):
        target = _hardware_position(
            waypoint.target,
            closed_gripper,
            open_gripper,
            gripper_override=waypoint.hardware_gripper,
        )
        joints = ", ".join(f"{joint}={target[joint]:6.1f}" for joint in JOINT_ORDER)
        print(f"  {index:02d}. {waypoint.name:18s} {waypoint.duration:5.2f}s  {joints}")


def _confirm(message: str, assume_yes: bool) -> None:
    if assume_yes:
        return
    response = input(f"{message} [Enter to continue, q to stop] ").strip().lower()
    if response in {"q", "quit", "stop", "n", "no"}:
        raise KeyboardInterrupt("Stopped by user.")


def execute_waypoints(args: argparse.Namespace, waypoints: list[Waypoint]) -> None:
    try:
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Could not import the LeRobot SO-101 follower hardware driver. Install the Feetech extras in "
            "the environment you use for the real robot, for example:\n"
            "  pip install -e '.[feetech]'\n"
            f"Original error: {exc}"
        ) from exc

    if not args.port:
        raise SystemExit("Missing --port. Pass your follower port or set SO101_PORT.")
    if not args.robot_id:
        raise SystemExit("Missing --id. Pass the same follower id used during calibration or set SO101_ID.")

    config = SO101FollowerConfig(
        port=args.port,
        id=args.robot_id,
        max_relative_target=args.max_relative_target,
    )
    robot = SO101Follower(config)

    print(f"Connecting follower {args.robot_id!r} on {args.port!r} using existing calibration...")
    robot.connect(calibrate=False)
    try:
        current = _from_observation(robot.get_observation())
        print("Current follower position:")
        print("  " + ", ".join(f"{joint}={current[joint]:.1f}" for joint in JOINT_ORDER))

        _confirm(
            "Clear the workspace. Test without whisk/cup first. Keep power/USB reachable.",
            assume_yes=args.yes,
        )

        period = 1.0 / args.fps
        for waypoint in waypoints:
            target = _hardware_position(
                waypoint.target,
                args.closed_gripper,
                args.open_gripper,
                gripper_override=waypoint.hardware_gripper,
            )
            _confirm(f"Move to phase {waypoint.name!r} over {waypoint.duration:.2f}s?", assume_yes=args.yes)

            steps = max(1, int(round(waypoint.duration * args.fps)))
            start = dict(current)
            for step_index in range(1, steps + 1):
                alpha = step_index / steps
                command = _interpolate(start, target, alpha)
                robot.send_action(_to_action(command))
                time.sleep(period)

            robot.send_action(_to_action(target))
            current = _from_observation(robot.get_observation())
            print(
                f"Reached {waypoint.name}: "
                + ", ".join(f"{joint}={current[joint]:.1f}" for joint in JOINT_ORDER)
            )
    finally:
        robot.disconnect()


def execute_gripper_test(args: argparse.Namespace) -> None:
    try:
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Could not import the LeRobot SO-101 follower hardware driver. Install the Feetech extras in "
            "the environment you use for the real robot, for example:\n"
            "  pip install -e '.[feetech]'\n"
            f"Original error: {exc}"
        ) from exc

    if not args.port:
        raise SystemExit("Missing --port. Pass your follower port or set SO101_PORT.")
    if not args.robot_id:
        raise SystemExit("Missing --id. Pass the same follower id used during calibration or set SO101_ID.")

    robot = SO101Follower(
        SO101FollowerConfig(
            port=args.port,
            id=args.robot_id,
            max_relative_target=args.max_relative_target,
        )
    )
    sequence = [
        ("open", args.open_gripper),
        ("preclose", args.preclose_gripper),
        ("closed", args.closed_gripper),
        ("open", args.open_gripper),
    ]

    print(f"Connecting follower {args.robot_id!r} on {args.port!r} for gripper-only test...")
    robot.connect(calibrate=False)
    try:
        print("This test sends only gripper.pos; arm joints are not commanded.")
        _confirm("Keep fingers/tools clear of the gripper.", assume_yes=args.yes)
        for label, value in sequence:
            _confirm(f"Send gripper {label} command ({value:.1f})?", assume_yes=args.yes)
            robot.send_action({"gripper.pos": float(value)})
            time.sleep(args.gripper_test_hold)
            obs = robot.get_observation()
            observed = obs.get("gripper.pos", None)
            if observed is not None:
                print(f"  observed gripper.pos={float(observed):.1f}")
    finally:
        robot.disconnect()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay the matcha waypoint plan on a calibrated SO-101 follower arm.")
    parser.add_argument("--port", default=os.environ.get("SO101_PORT"), help="Follower arm serial port. Defaults to SO101_PORT.")
    parser.add_argument("--id", dest="robot_id", default=os.environ.get("SO101_ID"), help="Follower calibration id. Defaults to SO101_ID.")
    parser.add_argument("--execute", action="store_true", help="Actually connect to and move the follower arm. Omit for dry-run.")
    parser.add_argument("--yes", action="store_true", help="Do not ask for confirmation before each phase.")
    parser.add_argument("--test-gripper", action="store_true", help="Only exercise the gripper open/preclose/closed/open sequence.")
    parser.add_argument("--gripper-test-hold", type=float, default=1.5, help="Seconds to hold each gripper test command.")
    parser.add_argument("--speed-scale", type=float, default=0.15, help="Fraction of sim speed. 0.15 means about 6.7x slower.")
    parser.add_argument("--fps", type=float, default=20.0, help="Hardware command rate in Hz.")
    parser.add_argument("--max-relative-target", type=float, default=8.0, help="LeRobot per-command safety clamp in degrees/range units.")
    parser.add_argument("--closed-gripper", type=float, default=100.0, help="Real gripper command for fully closed.")
    parser.add_argument("--open-gripper", type=float, default=0.0, help="Real gripper command for fully open.")
    parser.add_argument(
        "--preclose-gripper",
        type=float,
        default=30.0,
        help="Real gripper command used while descending around the whisk before closing.",
    )
    parser.add_argument("--whisk-stroke-length", type=float, default=MatchaConfig.whisk_stroke_length)
    parser.add_argument("--whisk-strokes", type=int, default=MatchaConfig.whisk_strokes)
    parser.add_argument("--whisk-duration", type=float, default=MatchaConfig.whisk_duration)
    parser.add_argument(
        "--whisk-vertical-amplitude",
        type=float,
        default=0.008,
        help="Small upward lift, in meters, added to alternating whisk strokes.",
    )
    parser.add_argument("--main-cup-position", type=float, nargs=3, default=MatchaConfig.main_cup_position)
    parser.add_argument("--whisk-position", type=float, nargs=3, default=MatchaConfig.whisk_position)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.speed_scale <= 0.0 or args.speed_scale > 1.0:
        raise SystemExit("--speed-scale must be in the range (0, 1].")
    if args.fps <= 0.0:
        raise SystemExit("--fps must be positive.")
    if args.gripper_test_hold <= 0.0:
        raise SystemExit("--gripper-test-hold must be positive.")
    if args.whisk_vertical_amplitude < 0.0:
        raise SystemExit("--whisk-vertical-amplitude must be non-negative.")

    if args.test_gripper:
        print(
            "Gripper-only test values: "
            f"open={args.open_gripper:.1f}, preclose={args.preclose_gripper:.1f}, closed={args.closed_gripper:.1f}"
        )
        if not args.execute:
            print("Dry run only. Add --execute to connect to the follower and test the gripper.")
            return
        execute_gripper_test(args)
        return

    config = MatchaConfig(
        main_cup_position=tuple(args.main_cup_position),
        whisk_position=tuple(args.whisk_position),
        whisk_stroke_length=args.whisk_stroke_length,
        whisk_strokes=args.whisk_strokes,
        whisk_duration=args.whisk_duration,
        assisted_whisk=False,
    )
    waypoints = build_matcha_waypoints(
        config,
        speed_scale=args.speed_scale,
        preclose_gripper=args.preclose_gripper,
        whisk_vertical_amplitude=args.whisk_vertical_amplitude,
    )
    print_waypoints(waypoints, args.closed_gripper, args.open_gripper)

    if not args.execute:
        print("\nDry run only. Add --execute to connect to the follower and move the arm.")
        return

    execute_waypoints(args, waypoints)


if __name__ == "__main__":
    main()
