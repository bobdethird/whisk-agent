"""Parametric cup-yaw scenario test.

Drives the full pickup pipeline (perception + grasp library + IK + sim) at
several cup yaws to verify the per-anchor grasp transform actually works in
the simulator, not just on paper. Runs headless, so plain `python` is enough
(`mujoco.Renderer` is offscreen on macOS and does not require `mjpython`).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mujoco_sim.run_cup_pickup import PickupConfig, run_pickup


# The grasp orientation in `grasp_library.CUP` is rigidly attached to the cup
# body frame, so the requested world rotation tracks the cup yaw. The SO-101
# is 5-DoF and approaches the cup with shoulder_pan close to 0; this sweep
# lives in the asymmetric reach cone where the arm config matches the rigid
# grasp orientation closely enough for the moving-jaw arc to catch the bar.
# Generalizing to arbitrary yaws requires a "world-aware" grasp that
# recomputes claw_x at runtime from the bar's world radial direction (see
# `pose_from_target_relative_gripper_angle` in so101_kinematics.py for the
# pattern); that's a follow-on, not a library change.
#
# Camera coverage independently caps the sweep around |yaw| <= 60°: the
# cameras sit at world y < 0 and the cup tags at +x / +y faces become
# invisible past that.
SWEEP_YAW_DEGREES = (-15.0, -10.0, -5.0, 0.0, -2.5)
SWEEP_CUP_POSITION = (0.32, 0.0, 0.045)
MIN_PASS_RATE = 4 / 5


def run_yaw_sweep() -> list[tuple[float, bool, float]]:
    results: list[tuple[float, bool, float]] = []
    for yaw_deg in SWEEP_YAW_DEGREES:
        config = PickupConfig(
            cup_position=SWEEP_CUP_POSITION,
            cup_yaw_deg=yaw_deg,
            allow_config_position_fallback=False,
            debug_camera_frame_dir=None,
        )
        try:
            result = run_pickup(config, launch_viewer=False)
            success = result.success
            final_z = result.final_cup_z
            max_z = result.max_cup_z
            both_jaw_steps = result.diagnostics.both_jaw_contact_steps
        except RuntimeError as exc:
            print(f"yaw={yaw_deg:6.1f} deg  FAIL  perception error: {exc}")
            results.append((yaw_deg, False, float("nan")))
            continue
        results.append((yaw_deg, success, final_z))
        verdict = "PASS" if success else "FAIL"
        print(
            f"yaw={yaw_deg:6.1f} deg  {verdict}  "
            f"final_z={final_z:.4f} m  max_z={max_z:.4f} m  "
            f"both_jaw_steps={both_jaw_steps}"
        )
    return results


def test_yaw_sweep_pass_rate() -> None:
    results = run_yaw_sweep()
    passes = sum(1 for _, success, _ in results if success)
    rate = passes / len(results)
    print(f"\noverall pass rate: {passes}/{len(results)} = {rate:.0%}")
    assert rate >= MIN_PASS_RATE, (
        f"Yaw sweep pass rate {rate:.0%} fell below the required {MIN_PASS_RATE:.0%}; "
        f"results = {results}"
    )


if __name__ == "__main__":
    test_yaw_sweep_pass_rate()
