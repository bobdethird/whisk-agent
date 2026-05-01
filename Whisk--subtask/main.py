"""
Entry point for the matcha-making robot agent.

Pose source is selected with --poses flag:

    mock    Use Gaussian-noised hardcoded poses (default, no hardware needed)
    file    Read from poses.json — edit by hand or have any script write to it
    server  Poll the local HTTP pose server (teammate integration, port 5050)
    manual  Prompt user to type poses in the terminal each step

Usage
-----
    ANTHROPIC_API_KEY=sk-ant-... python main.py              # mock poses
    ANTHROPIC_API_KEY=sk-ant-... python main.py --poses file
    ANTHROPIC_API_KEY=sk-ant-... python main.py --poses server
    ANTHROPIC_API_KEY=sk-ant-... python main.py --poses manual

Arm executor
------------
    Always MockExecutor for now.  Swap for LeRobotExecutor when hardware ready.
"""

from __future__ import annotations

import argparse

from agent.loop import run_agent_loop
from agent.tools import dispatch_tool
from arm.executor import MockExecutor
from perception.pose_bridge import (
    FilePoseProvider,
    ManualPoseProvider,
    MockPoseProvider,
    ServerPoseProvider,
)


def build_executor(hardware: bool, port: str | None, urdf_path: str, ee_frame: str):
    """Return (executor, cleanup_fn).  cleanup must be called in a finally block."""
    if not hardware:
        print("[main] Executor: MockExecutor (no real motion)")
        return MockExecutor(), (lambda: None)

    if not port:
        raise SystemExit("ERROR: --hardware requires --port (e.g. /dev/tty.usbmodem...).")

    from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig
    from arm.lerobot_executor import LeRobotExecutor, LeRobotMotionAdapter
    from arm.placo_kinematics import PlacoKinematics

    print(f"[main] Executor: LeRobotExecutor on {port} (URDF={urdf_path}, EE={ee_frame})")
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
            print(f"[main] disconnect failed: {exc}")

    return executor, cleanup


TASK = (
    "Make matcha: scoop matcha powder from the scoop into the bowl, "
    "add hot water from the cup of water, add ice from the cup of ice, "
    "whisk the mixture until frothy, then pour into the matcha cup."
)


def build_pose_provider(mode: str):
    """Return the correct pose provider for *mode*."""
    if mode == "mock":
        print("[main] Pose source: mock (Gaussian noise on hardcoded values)")
        return MockPoseProvider()
    if mode == "file":
        print("[main] Pose source: poses.json  (edit the file to update poses)")
        return FilePoseProvider()
    if mode == "server":
        print("[main] Pose source: HTTP server at http://localhost:5050/poses")
        print("[main] Start the server in another terminal:")
        print("[main]   python perception/pose_server.py")
        return ServerPoseProvider()
    if mode == "manual":
        print("[main] Pose source: manual terminal input each step")
        return ManualPoseProvider()
    raise ValueError(f"Unknown pose mode '{mode}'")


def main() -> None:
    parser = argparse.ArgumentParser(description="Matcha robot agent")
    parser.add_argument(
        "--poses",
        choices=["mock", "file", "server", "manual"],
        default="mock",
        help="Pose source backend (default: mock)",
    )
    parser.add_argument(
        "--hardware",
        action="store_true",
        help="Use LeRobotExecutor on the real SO-101 (default: MockExecutor)",
    )
    parser.add_argument(
        "--port",
        default=None,
        help="Serial port for the SO-101 (required with --hardware)",
    )
    parser.add_argument(
        "--urdf",
        default="assets/so101.urdf",
        help="Path to SO-101 URDF (run scripts/download_urdf.py first)",
    )
    parser.add_argument(
        "--ee-frame",
        default="gripper",
        help="End-effector frame name in the URDF",
    )
    args = parser.parse_args()

    # ---- Pose provider ---------------------------------------------------
    provider = build_pose_provider(args.poses)

    # ---- Executor --------------------------------------------------------
    executor, cleanup = build_executor(args.hardware, args.port, args.urdf, args.ee_frame)

    def execute(tool_call: dict, pose_map: dict) -> dict:
        return dispatch_tool(tool_call, pose_map, executor)

    # ---- Run -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("MATCHA ROBOT")
    print("=" * 60)
    print(f"Task: {TASK}\n")

    try:
        history = run_agent_loop(
            task=TASK,
            get_poses_fn=provider.get_poses,
            execute_tool_fn=execute,
            verbose=True,
        )
    finally:
        cleanup()

    # ---- Summary ---------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"Completed in {len(history)} steps")
    print("=" * 60)
    for entry in history:
        action = entry.get("action")
        tool   = action.get("tool", "(none)") if action else "(none)"
        status = entry["result"]["status"]
        print(f"  Step {entry['step']:>2}: {tool:<20} → {status}")


if __name__ == "__main__":
    main()
