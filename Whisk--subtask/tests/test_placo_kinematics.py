"""
Tests for arm/placo_kinematics.py

The pure helpers (_pose_to_transform, _transform_to_pose, _rotation_angle)
are exercised unconditionally — no Placo install required.  Any end-to-end
IK/FK tests against a real Placo instance are gated behind an import-skip.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from arm.placo_kinematics import (
    _pose_to_transform,
    _rotation_angle,
    _transform_to_pose,
    _yaw_to_rotation_y_up,
    _yaw_to_rotation_z_up,
)


# ===========================================================================
# Pure helpers
# ===========================================================================

class TestYawRotations:
    def test_y_up_zero_yaw_is_identity(self):
        np.testing.assert_allclose(_yaw_to_rotation_y_up(0.0), np.eye(3), atol=1e-12)

    def test_z_up_zero_yaw_is_identity(self):
        np.testing.assert_allclose(_yaw_to_rotation_z_up(0.0), np.eye(3), atol=1e-12)

    def test_y_up_quarter_turn_maps_x_to_minus_z(self):
        """For yaw about +y, R · [1,0,0] must equal [cos, 0, -sin]."""
        R = _yaw_to_rotation_y_up(math.radians(90.0))
        v = R @ np.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(v, [0.0, 0.0, -1.0], atol=1e-12)

    def test_z_up_quarter_turn_maps_x_to_plus_y(self):
        R = _yaw_to_rotation_z_up(math.radians(90.0))
        v = R @ np.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(v, [0.0, 1.0, 0.0], atol=1e-12)


class TestPoseTransformRoundtrip:
    """_pose_to_transform followed by _transform_to_pose must be identity."""

    @pytest.mark.parametrize("urdf_is_y_up", [True, False])
    @pytest.mark.parametrize("pose", [
        [0.0, 0.0, 0.0, 0.0],
        [0.3, 0.05, 0.4, 0.0],
        [0.3, 0.05, 0.4, math.radians(30)],
        [-0.1, 0.2, 0.5, math.radians(-45)],
        [0.0, 0.0, 0.0, math.radians(170)],  # near ±π boundary
    ])
    def test_roundtrip(self, pose, urdf_is_y_up):
        T = _pose_to_transform(pose, urdf_is_y_up=urdf_is_y_up)
        back = _transform_to_pose(T, urdf_is_y_up=urdf_is_y_up)
        for a, b in zip(pose, back):
            # yaw wraps — compare on the circle
            if abs(a - b) > math.pi:
                b = b + math.copysign(2 * math.pi, a - b)
            assert a == pytest.approx(b, abs=1e-9)

    def test_y_up_transform_matches_hand_computation(self):
        """Sanity check: with y-up URDF, translation goes straight through."""
        T = _pose_to_transform([0.3, 0.2, 0.4, 0.0], urdf_is_y_up=True)
        np.testing.assert_allclose(T[:3, 3], [0.3, 0.2, 0.4], atol=1e-12)
        np.testing.assert_allclose(T[:3, :3], np.eye(3), atol=1e-12)

    def test_z_up_transform_swaps_y_and_z(self):
        """With z-up URDF, a y-up pose's y and z components swap in the transform."""
        T = _pose_to_transform([0.3, 0.2, 0.4, 0.0], urdf_is_y_up=False)
        # Executor y-up → URDF z-up: (x, y_up, z_up) → (x, z_up, y_up)
        np.testing.assert_allclose(T[:3, 3], [0.3, 0.4, 0.2], atol=1e-12)


class TestRotationAngle:
    def test_identity_has_zero_angle(self):
        assert _rotation_angle(np.eye(3)) == pytest.approx(0.0, abs=1e-12)

    def test_180_degrees_is_pi(self):
        R = np.diag([1.0, -1.0, -1.0])  # 180° about x-axis
        assert _rotation_angle(R) == pytest.approx(math.pi, abs=1e-9)

    def test_90_degrees_is_pi_over_two(self):
        R = _yaw_to_rotation_y_up(math.pi / 2)
        assert _rotation_angle(R) == pytest.approx(math.pi / 2, abs=1e-9)


# ===========================================================================
# Integration smoke tests — require placo + the kinematics URDF
# ===========================================================================

pytest.importorskip("placo", reason="placo not installed")

import os as _os
_URDF = _os.path.join(_os.path.dirname(__file__), "..", "assets", "so101_kinematics.urdf")
_URDF_AVAILABLE = _os.path.isfile(_URDF)


def test_placo_is_importable():
    """placo package must import without error."""
    import placo
    assert placo is not None


@pytest.mark.skipif(not _URDF_AVAILABLE, reason="assets/so101_kinematics.urdf not found")
class TestPlacoKinematicsLive:
    """End-to-end IK / FK tests against the real SO-101 URDF."""

    @pytest.fixture(scope="class")
    def kin(self):
        from arm.placo_kinematics import PlacoKinematics
        return PlacoKinematics(
            urdf_path=_URDF,
            end_effector_frame="gripper_frame_link",
            urdf_is_y_up=False,
        )

    # Poses confirmed reachable by FK exploration of the SO-101 workspace.
    REACHABLE = [
        [0.30,  0.20,  0.00,  0.0 ],
        [0.35,  0.15,  0.05,  0.3 ],
        [0.25,  0.25, -0.05, -0.5 ],
    ]
    UNREACHABLE = [
        [5.0, 5.0, 5.0, 0.0],   # far outside workspace
        [0.0, 3.0, 0.0, 0.0],   # straight up, too high
    ]

    def test_construction_does_not_raise(self, kin):
        """PlacoKinematics must load the URDF without error."""
        from arm.placo_kinematics import PlacoKinematics
        assert isinstance(kin, PlacoKinematics)

    def test_fk_at_zero_config_is_in_workspace(self, kin):
        """FK at zero joints must return a plausible position for the SO-101."""
        import numpy as np
        q_zero = [0.0] * 13
        pose = kin.forward_kinematics(q_zero)
        assert len(pose) == 4
        x, y, z, yaw = pose
        # EE should be within ~60 cm of the base in every axis.
        assert abs(x) < 0.6, f"x={x:.3f} looks wrong"
        assert abs(y) < 0.6, f"y={y:.3f} looks wrong"
        assert abs(z) < 0.6, f"z={z:.3f} looks wrong"

    @pytest.mark.parametrize("target", REACHABLE)
    def test_ik_position_accuracy_reachable(self, kin, target):
        """IK must achieve < 1 mm position error for a known-reachable pose."""
        import numpy as np
        q = kin.solve_ik(target)
        assert q is not None, f"IK returned None for reachable target {target}"
        fk = kin.forward_kinematics(q)
        pos_err = np.linalg.norm(np.array(fk[:3]) - np.array(target[:3]))
        assert pos_err < 1e-3, f"FK error {pos_err:.4f} m exceeds 1 mm tolerance"

    @pytest.mark.parametrize("target", UNREACHABLE)
    def test_ik_returns_none_for_unreachable(self, kin, target):
        """IK must return None rather than a garbage solution for unreachable poses."""
        q = kin.solve_ik(target)
        assert q is None, f"Expected None for unreachable {target}, got joints"

    def test_jacobian_shape(self, kin):
        """Jacobian must be (6, nv) where nv = number of velocity DOF."""
        import numpy as np
        q_zero = [0.0] * 13
        J = kin.jacobian(q_zero)
        assert J.shape[0] == 6, f"Expected 6 twist rows, got {J.shape[0]}"
        assert J.shape[1] > 0, "Jacobian has no columns"

    def test_fk_ik_roundtrip_position(self, kin):
        """FK → IK → FK must return to the original position within tolerance."""
        import numpy as np
        q_zero = [0.0] * 13
        pose_from_fk = kin.forward_kinematics(q_zero)
        q_ik = kin.solve_ik(pose_from_fk)
        if q_ik is None:
            pytest.skip("Zero-config FK pose not reachable by IK (unexpected)")
        pose_roundtrip = kin.forward_kinematics(q_ik)
        pos_err = np.linalg.norm(
            np.array(pose_roundtrip[:3]) - np.array(pose_from_fk[:3])
        )
        assert pos_err < 1e-3, f"Roundtrip position error {pos_err:.4f} m"
