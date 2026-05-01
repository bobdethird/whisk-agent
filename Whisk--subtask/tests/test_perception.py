"""Tests for perception/mock_poses.py"""

import pytest
from perception.mock_poses import MOCK_POSES, get_mock_poses

EXPECTED_OBJECTS = {
    "matcha_cup",
    "matcha_bowl",
    "matcha_whisk",
    "matcha_scoop",
    "cup_of_ice",
    "cup_of_water",
}


# ---------------------------------------------------------------------------
# Key / schema checks
# ---------------------------------------------------------------------------

def test_mock_poses_has_all_expected_keys():
    """get_mock_poses must return exactly the expected object set."""
    poses = get_mock_poses()
    assert set(poses.keys()) == EXPECTED_OBJECTS


def test_pose_values_are_float_lists_of_length_4():
    """Each pose must be a list of four Python floats: [x, y, z, yaw]."""
    poses = get_mock_poses()
    for name, pose in poses.items():
        assert isinstance(pose, list), f"{name}: pose is not a list"
        assert len(pose) == 4, f"{name}: expected 4 coords (x,y,z,yaw), got {len(pose)}"
        for i, val in enumerate(pose):
            assert isinstance(val, float), (
                f"{name}[{i}] is {type(val).__name__}, expected float"
            )


# ---------------------------------------------------------------------------
# Noise behaviour
# ---------------------------------------------------------------------------

def test_position_noise_within_5_sigma():
    """Position noise should be statistically indistinguishable from zero beyond 5σ."""
    noise_std = 0.002
    threshold = 5 * noise_std  # 0.010 m

    for trial in range(50):
        poses = get_mock_poses(noise_std=noise_std, yaw_noise_std=0.0)
        for name, pose in poses.items():
            original = MOCK_POSES[name]
            for i in range(3):
                deviation = abs(pose[i] - original[i])
                assert deviation < threshold, (
                    f"Trial {trial}: {name}[{i}] deviated {deviation:.5f} m "
                    f"(threshold {threshold:.5f} m, 5σ)"
                )


def test_yaw_noise_within_5_sigma():
    """Yaw noise should stay within 5σ of the nominal yaw."""
    yaw_std = 0.05
    threshold = 5 * yaw_std

    for trial in range(50):
        poses = get_mock_poses(noise_std=0.0, yaw_noise_std=yaw_std)
        for name, pose in poses.items():
            original_yaw = MOCK_POSES[name][3]
            deviation = abs(pose[3] - original_yaw)
            assert deviation < threshold, (
                f"Trial {trial}: {name} yaw deviated {deviation:.4f} rad "
                f"(threshold {threshold:.4f} rad)"
            )


def test_noise_is_nonzero_on_average():
    """Sanity check: at least some noise must be added across many calls."""
    deviations = []
    for _ in range(30):
        poses = get_mock_poses(noise_std=0.002, yaw_noise_std=0.05)
        for name, pose in poses.items():
            original = MOCK_POSES[name]
            deviations.extend(abs(pose[i] - original[i]) for i in range(4))
    assert sum(deviations) / len(deviations) > 1e-6, "Noise appears to be zero"


# ---------------------------------------------------------------------------
# Isolation / immutability
# ---------------------------------------------------------------------------

def test_each_call_returns_a_new_dict():
    """Two consecutive calls must not return the same dict object."""
    a = get_mock_poses()
    b = get_mock_poses()
    assert a is not b


def test_original_mock_poses_not_mutated():
    """Noise must not leak back into the MOCK_POSES constant."""
    original_cup = list(MOCK_POSES["matcha_cup"])
    _ = get_mock_poses()
    assert MOCK_POSES["matcha_cup"] == original_cup, (
        "MOCK_POSES was mutated by get_mock_poses()"
    )
