"""High-level robot actions callable by the agent.

Each function is a complete, blocking action: it returns only after the
motion is done (or the timeout has elapsed). The caller does not need to
add sleeps.

Tuning constants are at the top of this file. Start with the defaults,
then adjust on hardware — see the inline notes for what each one controls.

Register notes (STS3215 via FeetechMotorsBus):
  Present_Velocity : sign-magnitude encoded, bit 15 is direction. Decoded
                     value is a signed int in native motor units.  0 = stopped.
  Present_Load     : sign-magnitude encoded, bit 10 is direction. Decoded
                     magnitude is 0-1023 (~0-100% of Max_Torque_Limit).
  Torque_Limit     : SRAM register (addr 48), takes effect immediately,
                     resets on power-cycle. Upper-bounded by Max_Torque_Limit
                     (EPROM, set to 500 in so_follower configure()).
  Both Present_* fields are NOT in NORMALIZED_DATA, so normalize= has no
  effect on them — the value is always the sign-magnitude decoded raw int.
"""

from __future__ import annotations

import time

# ---------------------------------------------------------------------------
# Gripper position targets (normalized 0-100 scale)
# ---------------------------------------------------------------------------

GRIPPER_OPEN  =   0.0   # fully open  — SDK: 0 = open, 100 = closed
GRIPPER_CLOSE = 100.0   # fully closed target (motor stalls before reaching
                        # this if an object is in the way)

# ---------------------------------------------------------------------------
# Torque limits (0-1000 scale; so_follower sets Max_Torque_Limit = 500)
# ---------------------------------------------------------------------------

# Applied while closing. Motor stalls at this force when it meets resistance.
# Raise toward 350 if the whisk slips during motion.
# Lower toward 150 if soft/fragile objects get deformed.
GRIPPER_HOLD_TORQUE = 200

# Restored before opening so the gripper can overcome its own friction and
# any residual grip force.  Matches the configured EPROM ceiling.
GRIPPER_OPEN_TORQUE = 500

# ---------------------------------------------------------------------------
# Stall detection thresholds
# ---------------------------------------------------------------------------

# |Present_Velocity| below this → motor is considered stopped.
# STS3215 velocity is in native encoder-ticks/s units. Start at 10 and lower
# toward 3 if the loop exits too early (noisy zero readings); raise toward 25
# if it never exits (motor never reads zero even when visibly stopped).
STALL_VELOCITY_THRESHOLD = 25

# |Present_Load| above this at stall → grip confirmed (object in hand).
# Load magnitude is 0-1023.  An empty close typically reads < 20; a gripped
# object typically reads > 50. Tune if you get false positives/negatives.
GRIP_LOAD_THRESHOLD = 50

# Number of consecutive below-threshold velocity samples required before
# declaring a stall.  Filters out single noisy zero readings mid-motion.
STALL_CONSECUTIVE_SAMPLES = 3

# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

# Delay after issuing the command before polling begins.  The motor needs a
# moment to start accelerating — without this, the first poll would read
# velocity=0 and immediately declare a false stall.
GRIPPER_SETTLE_S        = 0.6   # fixed wait for snap-shut and intermediate positions

# STS3215 Acceleration register: 0 = no ramp (instant full speed), higher = longer ramp.
# Keep at 0 for constant-speed open; non-zero values create a slow-start phase.
GRIPPER_ACCELERATION    = 0
GRIPPER_STARTUP_DELAY_S = 0.15

# How often to sample the motor during the stall-detection loop (~50 Hz).
GRIPPER_POLL_INTERVAL_S = 0.02

# Hard timeouts — the loop always exits within these bounds even if stall
# detection never triggers (e.g. serial glitch keeps returning garbage).
GRIPPER_CLOSE_TIMEOUT_S = 3.0
GRIPPER_OPEN_TIMEOUT_S  = 2.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read_gripper(robot) -> tuple[int, int]:
    """Return (velocity, load) for the gripper motor.

    Both values are sign-magnitude decoded by the LeRobot bus layer.
    velocity : signed int, 0 when stopped.
    load     : signed int, magnitude is 0-1023.
    """
    velocity = robot.bus.read("Present_Velocity", "gripper", normalize=False)
    load     = robot.bus.read("Present_Load",     "gripper", normalize=False)
    return int(velocity), int(load)


def _wait_for_stall(robot, timeout_s: float) -> int:
    """Block until the gripper motor stalls or timeout expires.

    Returns the |Present_Load| magnitude observed at the moment of stall.
    Returns 0 on timeout (so callers can treat it like a zero-load stall).

    A stall is declared after STALL_CONSECUTIVE_SAMPLES consecutive readings
    with |velocity| <= STALL_VELOCITY_THRESHOLD.
    """
    time.sleep(GRIPPER_STARTUP_DELAY_S)

    consecutive = 0
    deadline = time.monotonic() + timeout_s - GRIPPER_STARTUP_DELAY_S

    while time.monotonic() < deadline:
        velocity, load = _read_gripper(robot)

        if abs(velocity) <= STALL_VELOCITY_THRESHOLD:
            consecutive += 1
            if consecutive >= STALL_CONSECUTIVE_SAMPLES:
                return abs(load)
        else:
            consecutive = 0  # reset on any non-zero reading

        time.sleep(GRIPPER_POLL_INTERVAL_S)

    return 0  # timed out


# ---------------------------------------------------------------------------
# Public actions
# ---------------------------------------------------------------------------

def open_claw(robot) -> None:
    """Open the gripper fully, blocking until it reaches the open position."""
    robot.bus.write("Acceleration", "gripper", GRIPPER_ACCELERATION)
    robot.bus.write("Torque_Limit", "gripper", GRIPPER_OPEN_TORQUE)
    robot.send_action({"gripper.pos": GRIPPER_OPEN})
    _wait_for_stall(robot, GRIPPER_OPEN_TIMEOUT_S)


def close_claw(robot) -> bool:
    """Close the gripper with controlled force, blocking until stall or timeout.

    Sets Torque_Limit to GRIPPER_HOLD_TORQUE before commanding position 0.
    The motor closes until it meets resistance and stalls — it does NOT keep
    forcing past the object.

    Returns:
        True  — stall detected with load above GRIP_LOAD_THRESHOLD (object gripped).
        False — stall detected with low load (closed on air) or timed out.

    The return value lets the agent verify the grasp before proceeding.
    """
    robot.bus.write("Torque_Limit", "gripper", GRIPPER_HOLD_TORQUE)
    robot.send_action({"gripper.pos": GRIPPER_CLOSE})
    load_at_stall = _wait_for_stall(robot, GRIPPER_CLOSE_TIMEOUT_S)
    gripped = load_at_stall >= GRIP_LOAD_THRESHOLD

    if not gripped:
        # Nothing in the jaws — snap fully closed at full torque.
        # Use a fixed sleep rather than stall detection: the motor may already
        # be at the end stop, so velocity = 0 immediately and _wait_for_stall
        # would return before the higher torque has a chance to push further.
        robot.bus.write("Torque_Limit", "gripper", GRIPPER_OPEN_TORQUE)
        robot.send_action({"gripper.pos": GRIPPER_CLOSE})
        time.sleep(GRIPPER_SETTLE_S)

    return gripped
