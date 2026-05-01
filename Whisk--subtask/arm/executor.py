"""
Executor interface and mock implementation for the SO-101 arm.

The :class:`Executor` Protocol defines the three low-level primitives that
the tool dispatcher calls.  To use a real robot, implement this Protocol
in a ``LeRobotExecutor`` class — nothing else in the codebase changes.

The :class:`MockExecutor` logs every call with a ``[MOCK]`` prefix so it
is immediately obvious that no physical motion is occurring.

Pose convention
---------------
End-effector poses are expressed as ``[x, y, z, yaw]``:

* ``x, y, z`` — metres in the robot base frame (y is vertical, positive up).
* ``yaw``     — rotation about the world-vertical axis, radians.
                yaw=0 means the wrist's natural zero (gripper jaws aligned
                with the robot's x-axis).  Positive yaw follows the
                right-hand rule about +y.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Executor(Protocol):
    """
    Minimal arm control interface.

    All methods return a result dict containing at minimum::

        {"status": "success" | "error"}

    On error the dict also contains ``"reason": str``.
    """

    def move_to(self, waypoint: list[float]) -> dict:
        """
        Command the arm to move its end-effector to *waypoint* = [x, y, z, yaw].

        The implementation is responsible for velocity limiting, joint-space
        interpolation, IK solving, and collision avoidance.  This call blocks
        until the waypoint is reached or an error occurs.

        Args:
            waypoint: Target ``[x, y, z, yaw]`` — xyz in metres (robot base
                      frame), yaw in radians about world-vertical.

        Returns:
            Result dict with ``"status"`` key.
        """
        ...

    def set_gripper(self, width: float) -> dict:
        """
        Set the gripper jaw opening to *width* metres.

        0.0 = fully closed.  The maximum useful opening is ~0.08 m for the
        SO-101 gripper.  The implementation handles force limiting internally.

        Args:
            width: Target opening in metres.

        Returns:
            Result dict with ``"status"`` key.
        """
        ...

    def get_end_effector_pose(self) -> list[float]:
        """
        Return the current end-effector pose as ``[x, y, z, yaw]``.

        xyz in metres; yaw in radians about world-vertical.  In production
        this reads from forward kinematics or a pose sensor.
        """
        ...


class MockExecutor:
    """
    Simulated executor — logs calls but performs no real motion.

    Every call prints a ``[MOCK]`` tagged line so it is impossible to confuse
    mock runs with real robot operations.  Internal state (pose, gripper width)
    is tracked so that sequential calls behave consistently.
    """

    # Safe home position above workspace: [x, y, z, yaw]
    _RESTING_POSE: list[float] = [0.0, 0.35, 0.0, 0.0]

    def __init__(self) -> None:
        self._pose: list[float] = list(self._RESTING_POSE)
        self._gripper_width: float = 0.08  # start fully open

    # ------------------------------------------------------------------
    def move_to(self, waypoint: list[float]) -> dict:
        """
        Simulate moving to *waypoint* and update internal pose tracking.

        Args:
            waypoint: Target ``[x, y, z, yaw]``; xyz in metres, yaw in radians.

        Returns:
            ``{"status": "success", "pose": [x, y, z, yaw]}``
        """
        x, y, z, yaw = waypoint
        print(
            f"[MOCK] move_to([{x:.4f}, {y:.4f}, {z:.4f}, "
            f"yaw={yaw:.3f} rad])"
        )
        self._pose = [float(x), float(y), float(z), float(yaw)]
        return {"status": "success", "pose": list(self._pose)}

    def set_gripper(self, width: float) -> dict:
        """
        Simulate opening or closing the gripper.

        Args:
            width: Jaw opening in metres (0.0 = closed, ~0.08 = fully open).

        Returns:
            ``{"status": "success", "gripper_width": width}``
        """
        action = "open" if width > 0.01 else "close"
        print(f"[MOCK] set_gripper({width:.4f})  # {action}")
        self._gripper_width = float(width)
        return {"status": "success", "gripper_width": float(width)}

    def get_end_effector_pose(self) -> list[float]:
        """
        Return the last commanded pose (mock has no sensor — uses dead reckoning).

        Returns:
            ``[x, y, z, yaw]`` — xyz in metres, yaw in radians.
        """
        pose = list(self._pose)
        print(f"[MOCK] get_end_effector_pose() → {[round(v, 4) for v in pose]}")
        return pose
