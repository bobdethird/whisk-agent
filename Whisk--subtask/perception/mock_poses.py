"""
Mock AprilTag pose data for development and testing.

Simulates what a real camera + tag detection pipeline would return.
Poses are ``[x, y, z, yaw]`` — xyz in metres (y vertical, positive up),
yaw in radians about world-vertical (positive follows right-hand rule
about +y; 0 means the object's local forward axis aligns with +x).
"""

import math

import numpy as np

# [x, y, z, yaw] — yaw = rotation about world y-axis in radians
MOCK_POSES: dict[str, list[float]] = {
    "matcha_cup":   [0.30, 0.02, 0.40, 0.0],                   # cylinder — yaw immaterial
    "matcha_bowl":  [0.10, 0.02, 0.40, 0.0],                   # cylinder — yaw immaterial
    "matcha_whisk": [0.20, 0.02, 0.50, math.radians(30.0)],    # handle lies at 30° in XZ
    "matcha_scoop": [0.35, 0.02, 0.45, math.radians(-20.0)],   # handle lies at -20° in XZ
    "cup_of_ice":   [0.00, 0.10, 0.35, 0.0],                   # taller than table objects
    "cup_of_water": [0.05, 0.10, 0.35, 0.0],                   # same height as ice cup
}

# Per-axis noise sigma.  Position: 2 mm.  Yaw: ~3° (0.05 rad).
_POS_NOISE_STD_DEFAULT: float = 0.002
_YAW_NOISE_STD_DEFAULT: float = 0.05


def get_mock_poses(
    noise_std: float = _POS_NOISE_STD_DEFAULT,
    yaw_noise_std: float = _YAW_NOISE_STD_DEFAULT,
) -> dict[str, list[float]]:
    """
    Return a copy of MOCK_POSES with small Gaussian noise on each coordinate.

    The noise simulates realistic AprilTag detection jitter from camera
    resolution limits, tag warping, and minor calibration error.

    Args:
        noise_std:     Per-axis position noise stdev in metres (default 2 mm).
        yaw_noise_std: Yaw noise stdev in radians (default ~3°).

    Returns:
        Dict mapping object name → noisy ``[x, y, z, yaw]``.
        Each call returns a new, independently noised dict.
    """
    noisy: dict[str, list[float]] = {}
    for obj, pose in MOCK_POSES.items():
        pos_noise = np.random.normal(0.0, noise_std, size=3)
        yaw_noise = float(np.random.normal(0.0, yaw_noise_std))
        noisy[obj] = [
            float(pose[0] + pos_noise[0]),
            float(pose[1] + pos_noise[1]),
            float(pose[2] + pos_noise[2]),
            float(pose[3] + yaw_noise),
        ]
    return noisy
