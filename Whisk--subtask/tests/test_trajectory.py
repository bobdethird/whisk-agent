"""Tests for arm/trajectory.py"""

import math

import pytest

from arm.grasp_configs import GRASP_CONFIGS
from arm.trajectory import (
    compute_grasp_trajectory,
    compute_place_trajectory,
    compute_side_grasp_trajectory,
    compute_trajectory,
)

# Shared fixtures — [x, y, z, yaw]
CURRENT = [0.0, 0.05, 0.0, 0.0]
TARGET  = [0.3, 0.02, 0.4, 0.0]
SAFE    = 0.30


# ===========================================================================
# compute_trajectory
# ===========================================================================

class TestComputeTrajectory:
    def test_returns_exactly_3_waypoints(self):
        wps = compute_trajectory(CURRENT, TARGET, safe_height=SAFE)
        assert len(wps) == 3

    def test_waypoints_are_4_elements(self):
        wps = compute_trajectory(CURRENT, TARGET, safe_height=SAFE)
        for wp in wps:
            assert len(wp) == 4

    def test_lift_waypoint_is_at_safe_height(self):
        """First waypoint must have y == safe_height."""
        wps = compute_trajectory(CURRENT, TARGET, safe_height=SAFE)
        assert wps[0][1] == pytest.approx(SAFE)

    def test_lift_waypoint_preserves_current_xz(self):
        """Lift is straight up — x and z must not change."""
        wps = compute_trajectory(CURRENT, TARGET, safe_height=SAFE)
        assert wps[0][0] == pytest.approx(CURRENT[0])
        assert wps[0][2] == pytest.approx(CURRENT[2])

    def test_xz_translation_happens_at_safe_height(self):
        """Second waypoint (XZ transit) must stay at safe_height."""
        wps = compute_trajectory(CURRENT, TARGET, safe_height=SAFE)
        assert wps[1][1] == pytest.approx(SAFE)

    def test_xz_of_transit_waypoint_matches_target(self):
        """After transit, x and z must equal target x and z."""
        wps = compute_trajectory(CURRENT, TARGET, safe_height=SAFE)
        assert wps[1][0] == pytest.approx(TARGET[0])
        assert wps[1][2] == pytest.approx(TARGET[2])

    def test_final_waypoint_matches_target_xyz(self):
        """Last waypoint's xyz must equal the target's xyz."""
        wps = compute_trajectory(CURRENT, TARGET, safe_height=SAFE)
        assert wps[-1][:3] == pytest.approx(TARGET[:3])

    def test_yaw_threaded_through_all_waypoints(self):
        """Every waypoint carries the target yaw — wrist rotates early."""
        target = [0.3, 0.02, 0.4, math.radians(45.0)]
        wps = compute_trajectory(CURRENT, target, safe_height=SAFE)
        for wp in wps:
            assert wp[3] == pytest.approx(target[3])

    def test_safe_height_clamped_when_too_low(self):
        """safe_height below current y must be silently clamped upward."""
        high_current = [0.0, 0.50, 0.0, 0.0]
        wps = compute_trajectory(high_current, TARGET, safe_height=0.10)
        assert wps[0][1] >= high_current[1] - 1e-9
        assert wps[1][1] >= high_current[1] - 1e-9

    def test_same_start_and_end_pose_is_valid(self):
        """Edge case: current == target must not crash and must reach target."""
        pose = [0.3, 0.02, 0.4, 0.0]
        wps  = compute_trajectory(pose, pose, safe_height=SAFE)
        assert len(wps) >= 1
        assert wps[-1][:3] == pytest.approx(pose[:3])


# ===========================================================================
# compute_grasp_trajectory
# ===========================================================================

class TestComputeGraspTrajectory:
    def _config(self):
        return GRASP_CONFIGS["matcha_cup"]

    def test_returns_exactly_4_waypoints(self):
        wps = compute_grasp_trajectory(CURRENT, TARGET, self._config())
        assert len(wps) == 4

    def test_lift_waypoint_at_safe_height(self):
        wps = compute_grasp_trajectory(CURRENT, TARGET, self._config(), safe_height=SAFE)
        assert wps[0][1] == pytest.approx(SAFE)

    def test_transit_waypoint_at_safe_height(self):
        wps = compute_grasp_trajectory(CURRENT, TARGET, self._config(), safe_height=SAFE)
        assert wps[1][1] == pytest.approx(SAFE)

    def test_approach_waypoint_respects_offset(self):
        """Third waypoint (pre-grasp hover) must be exactly approach_height_offset above target."""
        config = self._config()
        wps = compute_grasp_trajectory(CURRENT, TARGET, config, safe_height=SAFE)
        expected_approach_y = TARGET[1] + config["approach_height_offset"]
        assert wps[2][1] == pytest.approx(expected_approach_y)

    def test_grasp_waypoint_at_grasp_height_offset(self):
        """Final waypoint must be exactly grasp_height_offset above target."""
        config = self._config()
        wps = compute_grasp_trajectory(CURRENT, TARGET, config, safe_height=SAFE)
        expected_grasp_y = TARGET[1] + config["grasp_height_offset"]
        assert wps[-1][1] == pytest.approx(expected_grasp_y)

    def test_all_post_lift_waypoints_share_target_xz(self):
        """Waypoints 2–4 (transit, hover, grasp) all stack above the target."""
        wps = compute_grasp_trajectory(CURRENT, TARGET, self._config(), safe_height=SAFE)
        for i, wp in enumerate(wps[1:], start=1):
            assert wp[0] == pytest.approx(TARGET[0]), f"waypoint {i} x mismatch"
            assert wp[2] == pytest.approx(TARGET[2]), f"waypoint {i} z mismatch"

    def test_wrist_angle_offsets_yaw(self):
        """Detected yaw + wrist_angle (deg→rad) must equal waypoint yaw."""
        target = [0.3, 0.02, 0.4, math.radians(10.0)]
        cfg = GRASP_CONFIGS["matcha_whisk"]  # wrist_angle = 15°
        wps = compute_grasp_trajectory(CURRENT, target, cfg)
        expected = target[3] + math.radians(cfg["wrist_angle"])
        assert wps[-1][3] == pytest.approx(expected)


# ===========================================================================
# compute_side_grasp_trajectory
# ===========================================================================

class TestComputeSideGraspTrajectory:
    def _config(self):
        return GRASP_CONFIGS["matcha_whisk"]

    def test_returns_exactly_4_waypoints(self):
        wps = compute_side_grasp_trajectory(CURRENT, TARGET, self._config())
        assert len(wps) == 4

    def test_lift_and_transit_at_safe_height(self):
        wps = compute_side_grasp_trajectory(CURRENT, TARGET, self._config(), safe_height=SAFE)
        assert wps[0][1] == pytest.approx(SAFE)
        assert wps[1][1] == pytest.approx(SAFE)

    def test_last_two_waypoints_at_same_height_horizontal_slide(self):
        """
        Final slide-in must be horizontal: waypoints 3 and 4 must share y, but
        differ in x/z.  (That's what distinguishes side-grasp from top-grasp,
        where the final two share x/z and differ in y.)
        """
        target = [0.3, 0.02, 0.4, math.radians(30.0)]
        cfg = self._config()
        wps = compute_side_grasp_trajectory(CURRENT, target, cfg)
        # y should differ (approach vs grasp offsets differ), BUT the *slide*
        # between w[2] and w[3] is what we want to verify goes horizontally
        # into the target.  Check: w[3] xz == target xz, and w[2] xz != target xz.
        assert wps[3][0] == pytest.approx(target[0])
        assert wps[3][2] == pytest.approx(target[2])
        assert not (wps[2][0] == pytest.approx(target[0])
                    and wps[2][2] == pytest.approx(target[2]))

    def test_standoff_is_perpendicular_to_yaw(self):
        """Standoff offset direction = (-sin yaw, cos yaw) in XZ."""
        target = [0.3, 0.02, 0.4, math.radians(30.0)]
        cfg = self._config()
        wps = compute_side_grasp_trajectory(CURRENT, target, cfg)

        dx = wps[2][0] - target[0]
        dz = wps[2][2] - target[2]
        # Direction vector should match ±(-sin θ, cos θ)
        expected_dx = -math.sin(target[3]) * cfg["approach_standoff"]
        expected_dz =  math.cos(target[3]) * cfg["approach_standoff"]
        assert dx == pytest.approx(expected_dx, abs=1e-9)
        assert dz == pytest.approx(expected_dz, abs=1e-9)

    def test_yaw_applied_to_all_waypoints(self):
        target = [0.3, 0.02, 0.4, math.radians(25.0)]
        cfg = self._config()
        wps = compute_side_grasp_trajectory(CURRENT, target, cfg)
        expected_yaw = target[3] + math.radians(cfg["wrist_angle"])
        for wp in wps:
            assert wp[3] == pytest.approx(expected_yaw)


# ===========================================================================
# compute_place_trajectory
# ===========================================================================

class TestComputePlaceTrajectory:
    def test_returns_exactly_3_waypoints(self):
        wps = compute_place_trajectory(CURRENT, TARGET)
        assert len(wps) == 3

    def test_lifts_to_safe_height(self):
        wps = compute_place_trajectory(CURRENT, TARGET, safe_height=SAFE)
        assert wps[0][1] == pytest.approx(SAFE)

    def test_transit_waypoint_at_safe_height(self):
        wps = compute_place_trajectory(CURRENT, TARGET, safe_height=SAFE)
        assert wps[1][1] == pytest.approx(SAFE)

    def test_final_y_is_place_height_above_target(self):
        """Arm stops at place_height_offset above target so object is set down gently."""
        offset = 0.02
        wps = compute_place_trajectory(CURRENT, TARGET, safe_height=SAFE, place_height_offset=offset)
        assert wps[-1][1] == pytest.approx(TARGET[1] + offset)

    def test_final_xz_matches_target(self):
        wps = compute_place_trajectory(CURRENT, TARGET, safe_height=SAFE)
        assert wps[-1][0] == pytest.approx(TARGET[0])
        assert wps[-1][2] == pytest.approx(TARGET[2])

    def test_custom_place_height_offset(self):
        """Caller-specified offset must be honoured."""
        wps = compute_place_trajectory(CURRENT, TARGET, safe_height=SAFE, place_height_offset=0.05)
        assert wps[-1][1] == pytest.approx(TARGET[1] + 0.05)


# ===========================================================================
# Obstacle-aware safe height
# ===========================================================================

class TestObstacleAwareCeiling:
    """Obstacles under the straight-line XZ path must raise the ceiling."""

    def test_obstacle_on_path_raises_ceiling(self):
        # Tall obstacle sits midway between current and target in XZ.
        obstacle_pose = [0.15, 0.10, 0.20, 0.0]  # y = 10 cm centroid
        obstacle_height = 0.20                    # top at y = 0.30
        # Default safe_height = 0.30, default clearance_buffer = 0.05
        # → ceiling should rise to 0.30 + 0.05 = 0.35
        wps = compute_trajectory(
            CURRENT, TARGET,
            obstacles=[(obstacle_pose, obstacle_height)],
        )
        assert wps[0][1] == pytest.approx(0.35)
        assert wps[1][1] == pytest.approx(0.35)

    def test_obstacle_off_path_leaves_ceiling_alone(self):
        # Obstacle far from the XZ segment — must not affect ceiling.
        far_obstacle = [2.0, 0.10, 2.0, 0.0]
        wps = compute_trajectory(
            CURRENT, TARGET,
            obstacles=[(far_obstacle, 0.50)],
        )
        # Default safe_height still applies.
        assert wps[0][1] == pytest.approx(SAFE)
        assert wps[1][1] == pytest.approx(SAFE)

    def test_short_obstacle_does_not_raise_ceiling(self):
        # Short obstacle under the path but already below safe_height + buffer.
        short_obstacle = [0.15, 0.02, 0.20, 0.0]
        wps = compute_trajectory(
            CURRENT, TARGET,
            obstacles=[(short_obstacle, 0.02)],
        )
        assert wps[0][1] == pytest.approx(SAFE)

    def test_clearance_buffer_is_honoured(self):
        obstacle_pose = [0.15, 0.10, 0.20, 0.0]
        obstacle_height = 0.20  # top at 0.30
        wps = compute_trajectory(
            CURRENT, TARGET,
            obstacles=[(obstacle_pose, obstacle_height)],
            clearance_buffer=0.15,
        )
        # top (0.30) + buffer (0.15) = 0.45
        assert wps[0][1] == pytest.approx(0.45)

    def test_multiple_obstacles_takes_tallest(self):
        tall = ([0.10, 0.10, 0.15, 0.0], 0.30)   # top 0.40
        short = ([0.20, 0.05, 0.30, 0.0], 0.05)  # top 0.10
        wps = compute_trajectory(
            CURRENT, TARGET,
            obstacles=[tall, short],
            clearance_buffer=0.05,
        )
        assert wps[0][1] == pytest.approx(0.45)

    def test_obstacle_aware_grasp_trajectory(self):
        """compute_grasp_trajectory also accepts and honours obstacles."""
        obstacle_pose = [0.15, 0.10, 0.20, 0.0]
        obstacle_height = 0.20
        cfg = GRASP_CONFIGS["matcha_cup"]
        wps = compute_grasp_trajectory(
            CURRENT, TARGET, cfg,
            obstacles=[(obstacle_pose, obstacle_height)],
        )
        assert wps[0][1] == pytest.approx(0.35)
        assert wps[1][1] == pytest.approx(0.35)
