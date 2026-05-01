"""
Manual single-object grasp test.

Use this BEFORE running the full agent loop to verify that each object
can be grasped safely.  Every waypoint pauses for your confirmation so
you can watch the arm and press Ctrl-C if anything looks wrong.

Usage
-----
    python scripts/test_single_grasp.py --object matcha_cup
    python scripts/test_single_grasp.py --object matcha_scoop --dry-run

With --dry-run, the MockExecutor is used so no real motion occurs.

Steps covered
-------------
1. Open gripper
2. Move to each grasp waypoint one by one (pauses between each)
3. Close gripper
4. Lift to safe height
5. Open gripper and return to home
"""

from __future__ import annotations

import argparse
import os
import sys

# Allow running from project root or scripts/ directory
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from arm.executor import MockExecutor
from arm.grasp_configs import GRASP_CONFIGS
from arm.trajectory import compute_grasp_trajectory
from perception.mock_poses import MOCK_POSES


def confirm(prompt: str = "Press Enter to continue (Ctrl-C to abort)...") -> None:
    """Pause and wait for the operator to confirm before moving."""
    try:
        input(f"\n  >>> {prompt} ")
    except KeyboardInterrupt:
        print("\n[Aborted]")
        sys.exit(0)


def _build_hardware_executor(port: str, urdf_path: str, ee_frame: str):
    """
    Construct a real-hardware LeRobotExecutor and return (executor, cleanup_fn).

    The cleanup_fn disconnects the robot cleanly — the caller MUST call it
    in a try/finally so torque is released even if the test aborts.
    """
    from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig
    from arm.lerobot_executor import LeRobotExecutor, LeRobotMotionAdapter
    from arm.placo_kinematics import PlacoKinematics

    robot = SOFollower(SOFollowerRobotConfig(
        id="my_follower_arm", port=port,
    ))
    robot.connect(calibrate=False)

    kin      = PlacoKinematics(urdf_path, end_effector_frame=ee_frame)
    motion   = LeRobotMotionAdapter(robot, forward_kinematics_fn=kin.forward_kinematics)
    executor = LeRobotExecutor(kin, motion)

    def cleanup() -> None:
        try:
            robot.disconnect()
        except Exception as exc:
            print(f"[warn] disconnect failed: {exc}")

    return executor, cleanup


def run_grasp_test(
    object_name: str,
    dry_run: bool,
    safe_height: float,
    port: str | None = None,
    urdf_path: str = "assets/so101.urdf",
    ee_frame: str = "gripper",
) -> None:
    print(f"\n{'=' * 60}")
    print(f"Single-grasp test: {object_name}  (dry_run={dry_run})")
    print("=" * 60)

    if object_name not in MOCK_POSES:
        print(f"ERROR: '{object_name}' not in MOCK_POSES. Known: {list(MOCK_POSES.keys())}")
        sys.exit(1)

    if object_name not in GRASP_CONFIGS:
        print(f"ERROR: '{object_name}' not in GRASP_CONFIGS.")
        sys.exit(1)

    target_pose  = MOCK_POSES[object_name]
    grasp_config = GRASP_CONFIGS[object_name]

    cleanup = lambda: None
    if dry_run:
        executor = MockExecutor()
        print("[DRY RUN] Using MockExecutor — no real motion.")
    else:
        if not port:
            print("ERROR: --real requires --port (e.g. /dev/tty.usbmodem...).")
            sys.exit(1)
        executor, cleanup = _build_hardware_executor(port, urdf_path, ee_frame)
        print(f"[REAL] Connected on {port}; URDF={urdf_path}; EE frame={ee_frame}")

    try:
        _run_grasp_body(executor, target_pose, grasp_config, safe_height)
    finally:
        cleanup()


def _run_grasp_body(executor, target_pose, grasp_config, safe_height):
    # ---- Step 0: show plan ----
    current_pose = executor.get_end_effector_pose()
    waypoints    = compute_grasp_trajectory(current_pose, target_pose, grasp_config, safe_height)

    print(f"\nTarget pose:  {[round(v, 4) for v in target_pose]}")
    print(f"Grasp config: {grasp_config}")
    print(f"\nWaypoints ({len(waypoints)} total):")
    labels = ["lift to safe height", "translate XZ", "pre-grasp hover", "grasp contact"]
    for i, (wp, label) in enumerate(zip(waypoints, labels)):
        print(f"  {i+1}. {label:<22} → {[round(v, 4) for v in wp]}")

    confirm("Plan looks OK? Press Enter to start.")

    # ---- Step 1: open gripper ----
    print("\n[1/5] Opening gripper...")
    result = executor.set_gripper(0.08)
    print(f"      → {result}")
    confirm()

    # ---- Step 2: move through waypoints ----
    for i, wp in enumerate(waypoints):
        label = labels[i] if i < len(labels) else f"waypoint {i+1}"
        print(f"\n[{i+2}/{len(waypoints)+3}] Moving to {label}...")
        print(f"      target: {[round(v, 4) for v in wp]}")
        result = executor.move_to(wp)
        print(f"      → {result}")
        if result.get("status") != "success":
            print(f"\n[FAIL] move_to returned error: {result.get('reason')}")
            sys.exit(1)
        confirm()

    # ---- Step 3: close gripper ----
    print(f"\n[{len(waypoints)+2}/{len(waypoints)+3}] Closing gripper (grasping)...")
    result = executor.set_gripper(grasp_config.get("grip_width", 0.0))
    print(f"      → {result}")
    confirm("Object grasped OK?")

    # ---- Step 4: lift to safe height ----
    current = executor.get_end_effector_pose()
    lift_wp  = [current[0], safe_height, current[2]]
    print(f"\n[{len(waypoints)+3}/{len(waypoints)+3}] Lifting to safe height {safe_height}m...")
    result = executor.move_to(lift_wp)
    print(f"      → {result}")
    confirm("Object lifted OK? Press Enter to release and finish.")

    # ---- Step 5: release and done ----
    executor.set_gripper(0.08)
    print("\n[Done] Gripper released. Test complete.")
    print(f"\nGrasp config for '{object_name}' appears to work.")
    print("Update arm/grasp_configs.py if any offsets need adjusting.\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual single-object grasp test")
    parser.add_argument(
        "--object",
        required=True,
        choices=list(MOCK_POSES.keys()),
        help="Object to test grasping",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Use MockExecutor (no real motion) — default True until HW ready",
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="Use real hardware (requires LeRobotExecutor to be implemented)",
    )
    parser.add_argument(
        "--safe-height",
        type=float,
        default=0.30,
        help="Safe transit height in metres (default 0.30)",
    )
    parser.add_argument(
        "--port",
        default=None,
        help="Serial port for the SO-101 (required with --real)",
    )
    parser.add_argument(
        "--urdf",
        default="assets/so101.urdf",
        help="Path to SO-101 URDF (run scripts/download_urdf.py first)",
    )
    parser.add_argument(
        "--ee-frame",
        default="gripper",
        help="End-effector frame name in the URDF (grep the URDF to confirm)",
    )
    args = parser.parse_args()

    dry_run = not args.real
    run_grasp_test(
        args.object, dry_run, args.safe_height,
        port=args.port, urdf_path=args.urdf, ee_frame=args.ee_frame,
    )


if __name__ == "__main__":
    main()
