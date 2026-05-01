"""
Hardware test for open_claw and close_claw.

Cycles open → close → open once, pausing between each step so you can
watch the gripper and confirm it behaves correctly.  close_claw prints
whether a grip was detected (object in hand vs. closed on air).

Usage
-----
    python scripts/test_gripper.py --port /dev/tty.usbmodem...

    # Place an object between the jaws before the close step to test
    # grip detection.  The script will tell you whether it registered.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from robot_actions import close_claw, open_claw


def pause(seconds: int, label: str) -> None:
    """Count down visibly so you can watch the arm before it moves."""
    print(f"  [{label}] starting in ", end="", flush=True)
    for i in range(seconds, 0, -1):
        print(f"{i}...", end=" ", flush=True)
        time.sleep(1)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Hardware gripper test")
    parser.add_argument(
        "--port",
        required=True,
        help="Serial port for the SO-101 (e.g. /dev/tty.usbmodem...)",
    )
    parser.add_argument(
        "--id",
        default="my_follower_arm",
        help="Robot ID — must match the calibration file name (default: my_follower_arm)",
    )
    args = parser.parse_args()

    from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig

    robot = SOFollower(SOFollowerRobotConfig(
        id=args.id, port=args.port,
    ))
    robot.connect(calibrate=False)
    print(f"[gripper test] connected on {args.port}")

    try:
        # --- Step 1: open ---
        pause(3, "OPEN")
        print("[1/3] Opening...")
        open_claw(robot)
        print("[1/3] Done.")

        # --- Step 2: close ---
        print("\n  Place an object between the jaws now if you want to test grip detection.")
        pause(5, "CLOSE")
        print("[2/3] Closing...")
        gripped = close_claw(robot)
        if gripped:
            print("[2/3] Done — grip detected (object in hand).")
        else:
            print("[2/3] Done — no grip detected (closed on air).")

        # --- Step 3: open again ---
        pause(3, "OPEN again")
        print("[3/3] Opening...")
        open_claw(robot)
        print("[3/3] Done.")

        print("\n[gripper test] complete.")

    finally:
        try:
            robot.disconnect()
        except Exception as exc:
            print(f"[warn] disconnect failed: {exc}")


if __name__ == "__main__":
    main()
