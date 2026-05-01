"""
Tests for arm/lerobot_executor.py

No real LeRobot SDK is imported — tests inject a stub SDK handle that
satisfies the Protocol contract used by :class:`LeRobotExecutor`.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pytest

from arm.lerobot_executor import (
    LeRobotExecutor,
    LeRobotMotionAdapter,
    GRIPPER_MAX_WIDTH_M,
    min_singular_value,
    trapezoidal_duration,
)
from robot_actions import (
    GRIPPER_CLOSE,
    GRIPPER_HOLD_TORQUE,
    GRIPPER_OPEN,
    GRIPPER_OPEN_TORQUE,
)


# ===========================================================================
# Pure helpers
# ===========================================================================

class TestTrapezoidalDuration:
    def test_zero_distance_zero_time(self):
        assert trapezoidal_duration(0.0, v_max=1.0, a_max=1.0) == (0.0, 0.0, 0.0)

    def test_full_trapezoid_has_cruise_phase(self):
        # Long distance → trapezoidal, cruise > 0
        t_a, t_c, t_d = trapezoidal_duration(distance=10.0, v_max=1.0, a_max=1.0)
        assert t_c > 0
        assert t_a == pytest.approx(t_d)
        # accel-phase time = v_max / a_max = 1.0
        assert t_a == pytest.approx(1.0)
        # Distance covered: 2 * 0.5 * 1 + 1 * t_c = 10 → t_c = 9
        assert t_c == pytest.approx(9.0)

    def test_short_move_is_triangular_no_cruise(self):
        # Short distance — never reaches v_max.
        t_a, t_c, t_d = trapezoidal_duration(distance=0.1, v_max=1.0, a_max=1.0)
        assert t_c == 0.0
        assert t_a == pytest.approx(t_d)
        # Triangle: d = a * t_a^2 → t_a = sqrt(d/a) = sqrt(0.1)
        assert t_a == pytest.approx(math.sqrt(0.1))

    def test_handles_negative_distance(self):
        # Direction doesn't change the time profile.
        pos = trapezoidal_duration(1.0, 1.0, 1.0)
        neg = trapezoidal_duration(-1.0, 1.0, 1.0)
        assert pos == neg


class TestMinSingularValue:
    def test_identity_returns_one(self):
        assert min_singular_value(np.eye(3)) == pytest.approx(1.0)

    def test_rank_deficient_returns_zero(self):
        # A rank-2 3x3 matrix has one zero singular value.
        J = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ])
        assert min_singular_value(J) == pytest.approx(0.0)


# ===========================================================================
# Stub SDK used by executor tests
# ===========================================================================

class _StubSDK:
    """
    Minimal SDK implementing the _SDKHandle Protocol.

    * IK: return the pose-as-joints for reachable poses; None for poses
      whose x > 1.0 (represents "out of workspace").
    * FK: inverse of IK — returns the 4-element pose.
    * Jacobian: identity 6x6 unless a "singular joint config" is triggered.
    * execute_joint_trajectory: records what was sent.
    """

    def __init__(self):
        self._joints = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self._gripper = 0.08
        self.executed_trajectories: list[tuple[list[list[float]], list[float]]] = []
        self.singular_zone_joint_0: Optional[tuple[float, float]] = None

    def solve_ik(self, pose_xyz_yaw):
        if pose_xyz_yaw[0] > 1.0:      # out-of-workspace sentinel
            return None
        return [
            float(pose_xyz_yaw[0]),
            float(pose_xyz_yaw[1]),
            float(pose_xyz_yaw[2]),
            float(pose_xyz_yaw[3]),
            0.0,
            0.0,
        ]

    def forward_kinematics(self, joints):
        return [float(joints[0]), float(joints[1]), float(joints[2]), float(joints[3])]

    def jacobian(self, joints):
        # Normally identity (well-conditioned).  If joint[0] falls in the
        # configured singular zone, return a rank-deficient Jacobian.
        if self.singular_zone_joint_0 is not None:
            lo, hi = self.singular_zone_joint_0
            if lo <= joints[0] <= hi:
                J = np.eye(6)
                J[0, 0] = 0.0   # zero a row → rank-5 → min σ = 0
                return J
        return np.eye(6)

    def get_joint_positions(self):
        return list(self._joints)

    def execute_joint_trajectory(self, joint_waypoints, durations):
        # Record and "move" to the final commanded position.
        self.executed_trajectories.append((joint_waypoints, durations))
        if joint_waypoints:
            self._joints = list(joint_waypoints[-1])

    def set_gripper(self, width):
        self._gripper = float(width)


# ===========================================================================
# LeRobotExecutor — behavioural tests
# ===========================================================================

class TestLeRobotExecutorMoveTo:
    def _make(self, **kwargs):
        sdk = _StubSDK()
        # One stub implements both Protocols — pass it as both backends.
        ex = LeRobotExecutor(kinematics=sdk, motion=sdk, **kwargs)
        return sdk, ex

    def test_reachable_pose_succeeds_and_moves(self):
        sdk, ex = self._make()
        result = ex.move_to([0.3, 0.2, 0.4, 0.0])
        assert result["status"] == "success"
        # Stub reports pose = joints[:4]; executor drives joints to [0.3,0.2,0.4,0.0,0,0]
        assert sdk._joints[:4] == pytest.approx([0.3, 0.2, 0.4, 0.0])

    def test_unreachable_pose_errors_without_motion(self):
        sdk, ex = self._make()
        result = ex.move_to([1.5, 0.2, 0.4, 0.0])   # x > 1.0 → IK None
        assert result["status"] == "error"
        assert "IK" in result["reason"] or "unreachable" in result["reason"].lower()
        # No trajectory was streamed.
        assert sdk.executed_trajectories == []

    def test_singular_path_rejected(self):
        sdk, ex = self._make()
        # Put the singular zone between current (joint_0=0) and target (joint_0=0.5).
        sdk.singular_zone_joint_0 = (0.2, 0.3)
        result = ex.move_to([0.5, 0.0, 0.0, 0.0])
        assert result["status"] == "error"
        assert "singularity" in result["reason"].lower()
        assert sdk.executed_trajectories == []

    def test_trajectory_waypoints_are_dense(self):
        """A long move must stream more than one joint waypoint."""
        sdk, ex = self._make(control_rate_hz=100.0)
        ex.move_to([0.4, 0.0, 0.0, 0.0])
        assert len(sdk.executed_trajectories) == 1
        joint_wps, durations = sdk.executed_trajectories[0]
        assert len(joint_wps) >= 2
        # All per-step durations equal.
        assert all(d == pytest.approx(durations[0]) for d in durations)

    def test_trajectory_endpoints_match_target_joints(self):
        sdk, ex = self._make()
        ex.move_to([0.3, 0.1, 0.2, 0.5])
        joint_wps, _ = sdk.executed_trajectories[0]
        # Last streamed waypoint should reach the IK target.
        assert joint_wps[-1][:4] == pytest.approx([0.3, 0.1, 0.2, 0.5])


class TestLeRobotExecutorGripperAndPose:
    def test_set_gripper_passes_through(self):
        sdk = _StubSDK()
        ex = LeRobotExecutor(kinematics=sdk, motion=sdk)
        result = ex.set_gripper(0.04)
        assert result == {"status": "success", "gripper_width": 0.04}
        assert sdk._gripper == pytest.approx(0.04)

    def test_get_end_effector_pose_uses_fk(self):
        sdk = _StubSDK()
        sdk._joints = [0.1, 0.2, 0.3, 0.4, 0.0, 0.0]
        ex = LeRobotExecutor(kinematics=sdk, motion=sdk)
        pose = ex.get_end_effector_pose()
        assert pose == pytest.approx([0.1, 0.2, 0.3, 0.4])


# ===========================================================================
# Stub robot used by LeRobotMotionAdapter tests
# ===========================================================================

class _StubBus:
    """Minimal bus stub that records writes and returns configurable sensor values."""

    def __init__(self, velocity: int = 0, load: int = 100):
        self.writes: list[tuple] = []
        self._velocity = velocity
        self._load = load

    def write(self, field: str, motor: str, value, **kwargs) -> None:
        self.writes.append((field, motor, value))

    def read(self, field: str, motor: str, **kwargs) -> int:
        if field == "Present_Velocity":
            return self._velocity
        if field == "Present_Load":
            return self._load
        return 0


class _StubRobot:
    """Minimal robot stub for LeRobotMotionAdapter — records actions and exposes a bus."""

    def __init__(self, velocity: int = 0, load: int = 100):
        self.bus = _StubBus(velocity=velocity, load=load)
        self.actions: list[dict] = []

    def send_action(self, action: dict) -> dict:
        self.actions.append(dict(action))
        return action


# ===========================================================================
# LeRobotMotionAdapter — gripper integration tests
# ===========================================================================

class TestLeRobotMotionAdapterGripper:
    """
    Tests that LeRobotMotionAdapter.set_gripper() routes to open_claw / close_claw
    for the fully-open and fully-closed cases, and falls through to a plain
    send_action for intermediate widths.

    Timing constants in robot_actions are patched to zero so tests are instant.
    The gripper_settle_s on the adapter is also zeroed for the intermediate case.
    """

    def _make(self, monkeypatch, velocity: int = 0, load: int = 100):
        # Zero out all sleeps — we test logic, not timing.
        monkeypatch.setattr("robot_actions.GRIPPER_STARTUP_DELAY_S", 0.0)
        monkeypatch.setattr("robot_actions.GRIPPER_POLL_INTERVAL_S", 0.0)
        robot = _StubRobot(velocity=velocity, load=load)
        adapter = LeRobotMotionAdapter(
            robot,
            workspace_limits=None,   # disable FK workspace check — no kinematics stub here
            gripper_settle_s=0.0,    # suppress intermediate-path sleep
        )
        return robot, adapter

    def test_fully_open_sets_open_torque_and_position(self, monkeypatch):
        robot, adapter = self._make(monkeypatch)
        adapter.set_gripper(GRIPPER_MAX_WIDTH_M)

        torque_writes = [w for w in robot.bus.writes if w[0] == "Torque_Limit"]
        assert torque_writes == [("Torque_Limit", "gripper", GRIPPER_OPEN_TORQUE)]
        assert any(a.get("gripper.pos") == pytest.approx(GRIPPER_OPEN) for a in robot.actions)

    def test_fully_closed_sets_hold_torque_and_position(self, monkeypatch):
        robot, adapter = self._make(monkeypatch, load=100)
        adapter.set_gripper(0.0)

        torque_writes = [w for w in robot.bus.writes if w[0] == "Torque_Limit"]
        assert torque_writes == [("Torque_Limit", "gripper", GRIPPER_HOLD_TORQUE)]
        assert any(a.get("gripper.pos") == pytest.approx(GRIPPER_CLOSE) for a in robot.actions)

    def test_close_claw_detects_grip_when_load_is_high(self, monkeypatch):
        # load=100 is above GRIP_LOAD_THRESHOLD (50) → grip confirmed
        robot, adapter = self._make(monkeypatch, load=100)
        from robot_actions import close_claw
        gripped = close_claw(robot)
        assert gripped is True

    def test_close_claw_reports_no_grip_when_load_is_low(self, monkeypatch):
        # load=10 is below GRIP_LOAD_THRESHOLD (50) → closed on air
        robot, adapter = self._make(monkeypatch, load=10)
        from robot_actions import close_claw
        gripped = close_claw(robot)
        assert gripped is False

    def test_intermediate_width_bypasses_claw_functions(self, monkeypatch):
        robot, adapter = self._make(monkeypatch)
        adapter.set_gripper(0.04)   # halfway — should NOT call open_claw / close_claw

        # No Torque_Limit writes — intermediate path never touches the bus
        assert not any(w[0] == "Torque_Limit" for w in robot.bus.writes)
        # gripper.pos should be 50.0  (1 - 0.04/0.08) * 100
        assert any(a.get("gripper.pos") == pytest.approx(50.0) for a in robot.actions)
