"""
Real AprilTag pose detection — stub / interface definition.

This module shows the production interface.  During development, use
:func:`perception.mock_poses.get_mock_poses` instead; it is a drop-in
replacement that returns the same dict structure.

Return shape
------------
``get_poses()`` returns a mapping ``{object_name: [x, y, z, yaw]}`` where:

* ``x, y, z`` — metres in the robot base frame (y is vertical, positive up).
* ``yaw``     — radians about **world-vertical** (world +y), NOT about the
                camera's y-axis or the tag's local y-axis.  Positive yaw
                follows the right-hand rule about +y.  For rotationally
                symmetric objects (cups, bowls) yaw is reported but is
                physically meaningless; downstream planning ignores it.

Production setup expected
--------------------------
* A calibrated RGB camera mounted with known extrinsics relative to the
  robot base frame (computed offline with a checkerboard or ChArUco board).
* AprilTags (36h11 family, 4 cm printed size) affixed to each object with
  a known tag→object-centroid offset stored in ``TAG_OFFSETS``.
* The ``apriltag`` Python package (``pip install apriltag``) and OpenCV
  for image capture and PnP solving.

Tag ID → object name map
-------------------------
  0  →  matcha_cup
  1  →  matcha_bowl
  2  →  matcha_whisk
  3  →  matcha_scoop
  4  →  cup_of_ice
  5  →  cup_of_water
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class AprilTagDetector:
    """
    Detects AprilTag markers in camera frames and returns object poses
    expressed in the robot base frame.

    All methods raise :exc:`NotImplementedError` until replaced with a real
    implementation that calls ``apriltag.Detector`` and ``cv2.solvePnP``.
    """

    TAG_TO_OBJECT: dict[int, str] = {
        0: "matcha_cup",
        1: "matcha_bowl",
        2: "matcha_whisk",
        3: "matcha_scoop",
        4: "cup_of_ice",
        5: "cup_of_water",
    }

    # Position-only offset (metres) from the tag's detected centre to the
    # object's grasping centroid, expressed in the tag's local frame
    # ``[x, y, z]``.  Yaw is derived directly from the tag's detected
    # orientation about world-vertical — no per-object yaw offset is
    # currently applied (handle objects have their tag aligned with the
    # handle axis during fabrication).
    TAG_OFFSETS: dict[int, list[float]] = {
        0: [0.0,  0.04, 0.0],   # cup: tag on side, centroid 4 cm higher
        1: [0.0,  0.025, 0.0],  # bowl: tag on rim
        2: [0.0,  0.06, 0.0],   # whisk: tag at base, centroid at handle mid
        3: [0.0,  0.005, 0.0],  # scoop: lies flat, small offset
        4: [0.0,  0.05, 0.0],   # ice cup: tag on side
        5: [0.0,  0.05, 0.0],   # water cup: tag on side
    }

    def __init__(
        self,
        camera_matrix: Optional[np.ndarray] = None,
        dist_coeffs: Optional[np.ndarray] = None,
        tag_size_m: float = 0.04,
        camera_to_base_transform: Optional[np.ndarray] = None,
    ) -> None:
        """
        Args:
            camera_matrix:            3×3 intrinsic matrix from calibration.
            dist_coeffs:              Distortion coefficients from calibration.
            tag_size_m:               Physical side length of printed tag (m).
            camera_to_base_transform: 4×4 homogeneous transform from camera
                                      frame to robot base frame.
        """
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs if dist_coeffs is not None else np.zeros(5)
        self.tag_size_m = tag_size_m
        self.camera_to_base = camera_to_base_transform
        self._detector = None  # lazy-initialised on first detect() call

    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> dict[str, list[float]]:
        """
        Detect all known objects in *frame* and return their poses.

        Steps (production implementation):
        1. Convert frame to grayscale.
        2. Run ``apriltag.Detector`` to find tag corners and IDs.
        3. For each known tag ID, call ``cv2.solvePnP`` to get camera-frame pose.
        4. Apply ``camera_to_base`` transform to express pose in robot frame.
        5. Add ``TAG_OFFSETS`` to get object centroid.

        Args:
            frame: BGR image from ``cv2.VideoCapture.read()``.

        Returns:
            Dict mapping object name → ``[x, y, z, yaw]`` in robot base
            frame (xyz in metres, yaw in radians about world-vertical).
            Objects not visible in *frame* are omitted.

        Raises:
            NotImplementedError: Replace with real implementation.
        """
        raise NotImplementedError(
            "AprilTagDetector.detect() is a stub. "
            "Use perception.mock_poses.get_mock_poses() during development."
        )

    def get_poses(self) -> dict[str, list[float]]:
        """
        Capture a frame from the attached camera and return all detected poses.

        Returns:
            Dict mapping object name → ``[x, y, z, yaw]``.  See module
            docstring for the frame and yaw convention.

        Raises:
            NotImplementedError: Replace with real implementation.
        """
        raise NotImplementedError(
            "AprilTagDetector.get_poses() is a stub. "
            "Use perception.mock_poses.get_mock_poses() during development."
        )
