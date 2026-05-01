"""
One-time arm calibration helper for the SO-101.

Run this the FIRST TIME you plug in the arm, or after any mechanical
reassembly (e.g. you disassembled a joint).  You do NOT need to re-run
this every session — calibration is saved to ~/.cache/lerobot/.

Usage:
    python scripts/calibrate_arm.py --port /dev/tty.usbmodem58760431551
    python scripts/calibrate_arm.py --port COM3          # Windows

What calibration does:
    Moves each joint to its physical limits and records the motor
    encoder values at those limits.  This maps raw encoder counts
    to the radians that the rest of the software uses.

After calibration:
    Run a quick check to confirm joint readings look sane:
        python scripts/calibrate_arm.py --port /dev/tty... --check-only
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def run_calibration(port: str) -> None:
    try:
        from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig
    except ImportError:
        print("ERROR: lerobot not installed.  Run:  pip install lerobot")
        sys.exit(1)

    print(f"[calibrate] Connecting to SO-101 on {port}...")
    config = SOFollowerRobotConfig(
        id="my_follower_arm",
        port=port,
    )
    robot = SOFollower(config)

    print("[calibrate] Starting calibration — follow the on-screen prompts.")
    print("[calibrate] You will be asked to move each joint to its limits by hand.")
    print()
    robot.connect(calibrate=True)

    print()
    print("[calibrate] Calibration complete.")
    print("[calibrate] Results saved to ~/.cache/lerobot/")
    robot.disconnect()


def check_joints(port: str) -> None:
    """Read and print current joint positions — sanity check after calibration."""
    try:
        from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig
    except ImportError:
        print("ERROR: lerobot not installed.  Run:  pip install lerobot")
        sys.exit(1)

    config = SOFollowerRobotConfig(
        id="my_follower_arm",
        port=port,
    )
    robot = SOFollower(config)
    robot.connect(calibrate=False)

    obs = robot.get_observation()
    print("\nCurrent joint positions:")
    joint_keys = [k for k in obs if k.endswith(".pos")]
    for key in sorted(joint_keys):
        print(f"  {key:<25} {obs[key]:>8.4f}")

    robot.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser(description="SO-101 calibration helper")
    parser.add_argument("--port", required=True,
                        help="Serial port, e.g. /dev/tty.usbmodem... or COM3")
    parser.add_argument("--check-only", action="store_true",
                        help="Only print current joint positions, skip calibration")
    args = parser.parse_args()

    if args.check_only:
        check_joints(args.port)
    else:
        run_calibration(args.port)


if __name__ == "__main__":
    main()
