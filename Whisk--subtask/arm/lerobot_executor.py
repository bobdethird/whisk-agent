"""
LeRobot SO-101 executor — real-hardware counterpart to :class:`MockExecutor`.

Implements the :class:`arm.executor.Executor` Protocol over two composable
backends:

* :class:`KinematicsBackend` — IK, FK, Jacobian.  Production uses
  :class:`arm.placo_kinematics.PlacoKinematics` (QP-based IK on a URDF).
* :class:`MotionSDK` — read joint encoders, stream joint trajectories,
  command the gripper.  Production uses the real LeRobot driver handle.

Splitting the two protocols means the kinematics backend is a pure
function of the URDF (no hardware needed) and can be unit-tested or
swapped without touching motion control.

Three responsibilities beyond the mock:

1. **Inverse kinematics.** Each ``move_to([x, y, z, yaw])`` call is solved
   to joint positions via the injected kinematics backend.  Unreachable
   poses return an error rather than silently clamping.
2. **Singularity avoidance.** Before commanding motion, the planned joint
   path is sampled; if the minimum singular value of the Jacobian falls
   below ``singularity_threshold`` at any sample, the move is rejected.
   This trades a small chance of a false-reject for a robust guarantee
   against driving into a kinematic lock-up.
3. **Trapezoidal velocity profile.** Joint commands follow an
   accel → cruise → decel shape so the arm doesn't jerk between
   waypoints.  Per-joint travel time is computed against global velocity
   and acceleration limits; the slowest joint sets the total duration
   so all joints start and stop together (coordinated motion).

Both Protocols are structural — any object exposing the right methods
works.  Tests inject a single stub that implements both; production
composes :class:`PlacoKinematics` with the LeRobot driver handle.
Neither ``placo`` nor ``lerobot`` is imported at module load, so
mock-only environments still import this module fine.
"""

from __future__ import annotations

import math
import time
from typing import Protocol, runtime_checkable

import numpy as np

from robot_actions import close_claw, open_claw


# ---------------------------------------------------------------------------
# Backend contracts
# ---------------------------------------------------------------------------

@runtime_checkable
class KinematicsBackend(Protocol):
    """
    Kinematics interface the executor calls.

    Production: :class:`arm.placo_kinematics.PlacoKinematics`.
    Tests: any object implementing the three methods below.
    """

    def solve_ik(self, pose_xyz_yaw: list[float]) -> list[float] | None:
        """Return joint positions for ``[x, y, z, yaw]``, or None if unreachable."""
        ...

    def forward_kinematics(self, joints: list[float]) -> list[float]:
        """Return ``[x, y, z, yaw]`` for the given joint positions."""
        ...

    def jacobian(self, joints: list[float]) -> np.ndarray:
        """Return the 6xN manipulator Jacobian at the given joints."""
        ...


@runtime_checkable
class MotionSDK(Protocol):
    """Motion-control interface the executor calls on the real SDK handle."""

    def get_joint_positions(self) -> list[float]:
        """Read current joint positions from encoders."""
        ...

    def execute_joint_trajectory(
        self,
        joint_waypoints: list[list[float]],
        durations: list[float],
    ) -> None:
        """
        Stream a coordinated joint-space trajectory to the controller.

        *joint_waypoints* is a dense list of joint-position samples;
        *durations[i]* is the time from sample i to sample i+1 (seconds).
        Blocks until motion completes.
        """
        ...

    def set_gripper(self, width: float) -> None:
        """Command gripper jaw opening in metres."""
        ...


# ---------------------------------------------------------------------------
# Pure helpers (unit-testable without any SDK)
# ---------------------------------------------------------------------------

def trapezoidal_duration(
    distance: float,
    v_max: float,
    a_max: float,
) -> tuple[float, float, float]:
    """
    Compute the three phase durations of a trapezoidal velocity profile.

    If the move is too short to reach *v_max*, the cruise phase collapses
    and the profile degenerates into a symmetric triangle (accel = decel,
    cruise = 0).

    Args:
        distance: Signed or unsigned distance to travel (any consistent units).
        v_max:    Peak velocity limit (> 0).
        a_max:    Acceleration limit (> 0).

    Returns:
        ``(t_accel, t_cruise, t_decel)`` in seconds.  t_accel == t_decel always.
    """
    d = abs(float(distance))
    if d < 1e-12:
        return (0.0, 0.0, 0.0)

    t_accel_full   = v_max / a_max
    d_accel_full   = 0.5 * a_max * t_accel_full * t_accel_full

    if d <= 2.0 * d_accel_full:
        # Triangular: accel until half-distance, then decel.
        t_tri = math.sqrt(d / a_max)
        return (t_tri, 0.0, t_tri)

    # Full trapezoid.
    t_cruise = (d - 2.0 * d_accel_full) / v_max
    return (t_accel_full, t_cruise, t_accel_full)


def min_singular_value(jacobian: np.ndarray) -> float:
    """Return the smallest singular value of *jacobian* — proximity to singularity."""
    return float(np.linalg.svd(jacobian, compute_uv=False).min())


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class LeRobotExecutor:
    """
    SO-101 executor implementing :class:`arm.executor.Executor` over
    a kinematics backend (e.g. Placo) and a motion SDK (e.g. LeRobot).

    The two backends may be the same object (one class implementing both
    Protocols) — tests rely on this.
    """

    def __init__(
        self,
        kinematics: KinematicsBackend,
        motion: MotionSDK,
        max_linear_velocity: float = 0.15,    # m/s at end-effector — conservative
        max_linear_accel:    float = 0.30,    # m/s^2
        max_angular_velocity: float = 1.5,    # rad/s at wrist
        singularity_threshold: float = 0.05,  # min singular value of Jacobian
        singularity_samples: int = 10,        # path samples for singularity check
        control_rate_hz: float = 100.0,       # joint-command streaming rate
    ) -> None:
        self._kin    = kinematics
        self._motion = motion
        self._max_v = float(max_linear_velocity)
        self._max_a = float(max_linear_accel)
        self._max_w = float(max_angular_velocity)
        self._singularity_threshold = float(singularity_threshold)
        self._singularity_samples   = int(singularity_samples)
        self._control_dt = 1.0 / float(control_rate_hz)

    # ------------------------------------------------------------------ move_to
    def move_to(self, waypoint: list[float]) -> dict:
        """
        Move the end-effector to ``[x, y, z, yaw]`` with a trapezoidal profile.

        Returns an error dict (no motion) if:
          - IK has no solution for *waypoint*.
          - The planned joint-space path crosses a kinematic singularity.
        """
        target_joints = self._kin.solve_ik(waypoint)
        if target_joints is None:
            return {"status": "error", "reason": "IK unreachable"}

        current_joints = self._motion.get_joint_positions()

        singular_q = self._first_singular_along_path(current_joints, target_joints)
        if singular_q is not None:
            return {
                "status": "error",
                "reason": (
                    f"path crosses singularity (min σ < "
                    f"{self._singularity_threshold:.3f})"
                ),
            }

        joint_waypoints, durations = self._plan_trapezoidal(
            current_joints, target_joints, waypoint,
        )
        self._motion.execute_joint_trajectory(joint_waypoints, durations)

        return {"status": "success", "pose": self.get_end_effector_pose()}

    # -------------------------------------------------------------- set_gripper
    def set_gripper(self, width: float) -> dict:
        self._motion.set_gripper(float(width))
        return {"status": "success", "gripper_width": float(width)}

    # -------------------------------------------------- get_end_effector_pose
    def get_end_effector_pose(self) -> list[float]:
        joints = self._motion.get_joint_positions()
        return list(self._kin.forward_kinematics(joints))

    # ======================================================================
    # Internals
    # ======================================================================

    def _first_singular_along_path(
        self,
        q_start: list[float],
        q_end:   list[float],
    ) -> list[float] | None:
        """
        Sample the straight-line joint-space path and return the first
        configuration whose Jacobian is below the singularity threshold,
        or None if the path stays clear.
        """
        n = self._singularity_samples
        for i in range(n + 1):
            t = i / n if n else 0.0
            q = [s + t * (e - s) for s, e in zip(q_start, q_end)]
            if min_singular_value(self._kin.jacobian(q)) < self._singularity_threshold:
                return q
        return None

    def _plan_trapezoidal(
        self,
        q_start: list[float],
        q_end:   list[float],
        target_pose: list[float],
    ) -> tuple[list[list[float]], list[float]]:
        """
        Build a coordinated trapezoidal trajectory in joint space.

        Total duration is set by the slowest joint under the Cartesian
        velocity/acceleration limits (converted into a rough joint-space
        bound via the path's Cartesian length — conservative but simple).
        All joints interpolate over the same duration so they finish
        together.
        """
        start_pose = self._kin.forward_kinematics(q_start)
        cart_dist = math.sqrt(
            (target_pose[0] - start_pose[0]) ** 2
            + (target_pose[1] - start_pose[1]) ** 2
            + (target_pose[2] - start_pose[2]) ** 2
        )
        yaw_delta = abs(target_pose[3] - start_pose[3]) if len(start_pose) >= 4 else 0.0

        t_a_lin, t_c_lin, t_d_lin = trapezoidal_duration(
            cart_dist, self._max_v, self._max_a,
        )
        # Angular duration: treat max_w as both v and a (≈) for simplicity.
        t_a_ang, t_c_ang, t_d_ang = trapezoidal_duration(
            yaw_delta, self._max_w, self._max_w,
        )

        lin_total = t_a_lin + t_c_lin + t_d_lin
        ang_total = t_a_ang + t_c_ang + t_d_ang
        total = max(lin_total, ang_total, self._control_dt)

        # Discretise at the control rate with a smooth trapezoidal weighting
        # mapped to joint space.  The velocity shape is applied uniformly to
        # the joint-space interpolation parameter s(t) ∈ [0, 1].
        n_steps = max(2, int(math.ceil(total / self._control_dt)))
        joint_waypoints: list[list[float]] = []
        durations: list[float] = []

        for i in range(1, n_steps + 1):
            t = i / n_steps * total
            s = _trapezoidal_s(t, t_a_lin, t_c_lin, t_d_lin) if lin_total > 0 else i / n_steps
            q = [qs + s * (qe - qs) for qs, qe in zip(q_start, q_end)]
            joint_waypoints.append(q)
            durations.append(total / n_steps)

        return joint_waypoints, durations


def _trapezoidal_s(
    t: float,
    t_accel: float,
    t_cruise: float,
    t_decel: float,
) -> float:
    """
    Evaluate the normalised position s(t) ∈ [0, 1] of a trapezoidal
    velocity profile whose three phases sum to the total travel time.

    Returns 0 at t=0 and 1 at t = t_accel + t_cruise + t_decel.
    """
    total = t_accel + t_cruise + t_decel
    if total <= 0:
        return 1.0

    # Unit-area trapezoid: scale so integral equals 1.  Peak velocity v_p
    # satisfies  v_p * (t_accel/2 + t_cruise + t_decel/2) = 1.
    denom = 0.5 * t_accel + t_cruise + 0.5 * t_decel
    if denom <= 0:
        return min(1.0, max(0.0, t / total))
    v_p = 1.0 / denom

    if t <= 0.0:
        return 0.0
    if t <= t_accel:
        return 0.5 * v_p * t * t / t_accel if t_accel > 0 else 0.0
    if t <= t_accel + t_cruise:
        s_accel = 0.5 * v_p * t_accel
        return s_accel + v_p * (t - t_accel)
    if t <= total:
        s_cruise_end = 0.5 * v_p * t_accel + v_p * t_cruise
        tau = t - t_accel - t_cruise
        return s_cruise_end + v_p * tau - 0.5 * v_p * tau * tau / t_decel if t_decel > 0 else 1.0
    return 1.0


# ---------------------------------------------------------------------------
# LeRobot SDK adapter — concrete MotionSDK implementation
# ---------------------------------------------------------------------------
#
# Workspace safety limits in the executor's y-up frame (metres, robot base).
# y is vertical (positive = up); x is forward reach; z is left/right.
# Tune these to your table BEFORE the first real run — the adapter refuses
# any joint waypoint whose FK lands outside this box.
#
# Starting values below are conservative for a desktop SO-101 with the base
# clamped at the rear of the table, matcha items on a tray in front.
SO101_WORKSPACE_LIMITS: dict[str, tuple[float, float]] = {
    "x": (0.08, 0.28),    # forward reach (avoid self-collision + low-accuracy extension)
    "y": (0.01, 0.20),    # height (1 cm floor buffer; cap below full extension)
    "z": (-0.18, 0.18),   # left/right (stay clear of singularity cones)
}

# SO-101 has 5 arm joints; the gripper is a 6th servo addressed separately.
SO101_JOINT_NAMES: list[str] = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]

# Gripper geometry: SDK scale is 0..100 where 0 = fully open, 100 = fully closed.
# The executor speaks metres, so we convert through GRIPPER_MAX_WIDTH_M.
GRIPPER_MAX_WIDTH_M: float = 0.08
GRIPPER_SETTLE_S:    float = 0.6   # servo is slower than the arm joints


class WorkspaceViolation(RuntimeError):
    """Raised when a commanded configuration would exit the safe Cartesian box."""


class LeRobotMotionAdapter:
    """
    Adapts the LeRobot ``SOFollower`` SDK to the :class:`MotionSDK` protocol.

    Pair with :class:`arm.placo_kinematics.PlacoKinematics` to get a full
    :class:`LeRobotExecutor`.  The adapter is intentionally thin: it forwards
    joint reads, streams joint-position commands at the rate the executor
    picks, and maps gripper width (m) to the SDK's 0..100 scale.

    The ``forward_kinematics_fn`` argument is optional but strongly
    recommended — when supplied, every streamed waypoint is FK-checked
    against :data:`SO101_WORKSPACE_LIMITS` before being sent, so a
    singularity-free joint path that would still drive the gripper into the
    table is rejected.  Pass ``kinematics.forward_kinematics`` from the
    Placo backend.
    """

    def __init__(
        self,
        robot,
        joint_names: list[str] = SO101_JOINT_NAMES,
        workspace_limits: dict[str, tuple[float, float]] | None = SO101_WORKSPACE_LIMITS,
        forward_kinematics_fn=None,
        gripper_max_width_m: float = GRIPPER_MAX_WIDTH_M,
        gripper_settle_s:    float = GRIPPER_SETTLE_S,
    ) -> None:
        self._robot            = robot
        self._joint_names      = list(joint_names)
        self._limits           = dict(workspace_limits) if workspace_limits else None
        self._fk               = forward_kinematics_fn
        self._gripper_max      = float(gripper_max_width_m)
        self._gripper_settle_s = float(gripper_settle_s)

    # ---------------------------------------------------------------- MotionSDK
    def get_joint_positions(self) -> list[float]:
        obs = self._robot.get_observation()
        return [float(obs[f"{j}.pos"]) for j in self._joint_names]

    def execute_joint_trajectory(
        self,
        joint_waypoints: list[list[float]],
        durations: list[float],
    ) -> None:
        """
        Stream *joint_waypoints* to the robot, sleeping *durations[i]* between
        sample i and i+1.  Blocks until the last waypoint has been sent.

        ``send_action`` on the SO follower is non-blocking, so pacing with
        sleeps is what synchronises the Python side with the servo loop.
        """
        if len(joint_waypoints) != len(durations):
            raise ValueError(
                f"waypoint/duration length mismatch: "
                f"{len(joint_waypoints)} vs {len(durations)}"
            )

        for q, dt in zip(joint_waypoints, durations):
            self._assert_in_workspace(q)
            self._robot.send_action(self._joints_to_action(q))
            if dt > 0:
                time.sleep(dt)

    def set_gripper(self, width: float) -> None:
        w = max(0.0, min(float(width), self._gripper_max))
        if w <= 0.0:
            close_claw(self._robot)
        elif w >= self._gripper_max:
            open_claw(self._robot)
        else:
            # Intermediate pre-position (e.g. pre-open to grip_width during grasp
            # approach). No stall detection needed — just position and settle.
            pos = (1.0 - w / self._gripper_max) * 100.0
            self._robot.send_action({"gripper.pos": max(0.0, min(100.0, pos))})
            time.sleep(self._gripper_settle_s)

    # ---------------------------------------------------------------- Internals
    def _joints_to_action(self, q: list[float]) -> dict:
        if len(q) != len(self._joint_names):
            raise ValueError(
                f"expected {len(self._joint_names)} joint values, got {len(q)}"
            )
        return {f"{name}.pos": float(v) for name, v in zip(self._joint_names, q)}

    def _assert_in_workspace(self, joints: list[float]) -> None:
        """Reject joint configurations whose FK lands outside the safe box."""
        if self._limits is None or self._fk is None:
            return
        pose = self._fk(joints)
        x, y, z = pose[0], pose[1], pose[2]
        for axis, val in (("x", x), ("y", y), ("z", z)):
            lo, hi = self._limits[axis]
            if not (lo <= val <= hi):
                raise WorkspaceViolation(
                    f"waypoint {axis}={val:.4f} m outside safe range "
                    f"[{lo}, {hi}] m — motion aborted."
                )
