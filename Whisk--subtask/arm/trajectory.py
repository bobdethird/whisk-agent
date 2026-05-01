"""
Trajectory planning for the SO-101 arm.

Coordinate convention
---------------------
  x — horizontal, positive to the robot's right
  y — vertical, positive upward
  z — depth, positive away from robot base
  yaw — rotation about the world-vertical (y) axis, radians

All planners use a *Y-ceiling* strategy: the arm always lifts to a safe
clearance height before any XZ movement so it never sweeps through cluttered
workspace at low altitude.  Each function returns a list of
``[x, y, z, yaw]`` waypoints that the executor feeds to the arm one by one.

Obstacle-aware safe height
--------------------------
Each planner accepts an optional ``obstacles`` list of
``(pose, object_height)`` pairs.  When an obstacle's XZ centre falls within
``obstacle_radius`` of the straight-line XZ transit segment, the Y-ceiling is
raised to ``obstacle_top + clearance_buffer`` so the arm flies over it.

Dependencies: numpy only (no external robotics libraries required).
"""

from __future__ import annotations

import math

# Obstacle type: ([x, y, z[, yaw]], object_height_above_centroid_metres)
Obstacle = tuple[list[float], float]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _point_to_segment_distance_xz(
    point_xz: tuple[float, float],
    seg_start_xz: tuple[float, float],
    seg_end_xz: tuple[float, float],
) -> float:
    """
    Shortest distance from *point_xz* to the XZ-plane segment
    [seg_start_xz, seg_end_xz].  All inputs are (x, z) pairs.
    """
    px, pz = point_xz
    sx, sz = seg_start_xz
    ex, ez = seg_end_xz

    dx, dz = ex - sx, ez - sz
    seg_len_sq = dx * dx + dz * dz
    if seg_len_sq < 1e-12:
        # Degenerate segment — return point-to-point distance.
        return math.hypot(px - sx, pz - sz)

    # Project point onto segment, clamp t to [0, 1].
    t = ((px - sx) * dx + (pz - sz) * dz) / seg_len_sq
    t = max(0.0, min(1.0, t))

    nearest_x = sx + t * dx
    nearest_z = sz + t * dz
    return math.hypot(px - nearest_x, pz - nearest_z)


def _obstacle_raised_ceiling(
    safe_y: float,
    current_pose: list[float],
    target_pose: list[float],
    obstacles: list[Obstacle] | None,
    obstacle_radius: float,
    clearance_buffer: float,
) -> float:
    """
    Raise *safe_y* to clear any obstacle whose XZ centre lies within
    ``obstacle_radius`` of the straight-line XZ transit from current to target.

    Returns the (possibly raised) safe_y.
    """
    if not obstacles:
        return safe_y

    seg_start = (current_pose[0], current_pose[2])
    seg_end   = (target_pose[0],  target_pose[2])

    for obs_pose, obs_height in obstacles:
        ox, oy, oz = obs_pose[0], obs_pose[1], obs_pose[2]
        dist = _point_to_segment_distance_xz((ox, oz), seg_start, seg_end)
        if dist <= obstacle_radius:
            needed = oy + float(obs_height) + clearance_buffer
            if needed > safe_y:
                safe_y = needed
    return safe_y


def _pose_yaw(pose: list[float]) -> float:
    """Return yaw from a 3- or 4-element pose (0.0 if yaw is missing)."""
    return float(pose[3]) if len(pose) >= 4 else 0.0


# ---------------------------------------------------------------------------
# Planners
# ---------------------------------------------------------------------------

def compute_trajectory(
    current_pose: list[float],
    target_pose: list[float],
    safe_height: float = 0.30,
    obstacles: list[Obstacle] | None = None,
    clearance_buffer: float = 0.05,
    obstacle_radius: float = 0.08,
) -> list[list[float]]:
    """
    Compute a minimal Y-ceiling path from *current_pose* to *target_pose*.

    The planner always clamps *safe_height* to be at or above both the
    current and target y-coordinates, so this is safe even when the caller
    passes a value that is too low.  When *obstacles* are supplied, the
    ceiling is further raised to clear any obstacle sitting under the
    straight-line XZ transit.

    Waypoints
    ---------
    1. Lift straight up to *safe_height* — x, z unchanged; yaw rotates to target.
    2. Translate in XZ to directly above the target, staying at *safe_height*.
    3. Descend along Y to the target pose.

    Args:
        current_pose:     ``[x, y, z[, yaw]]`` of the end-effector (metres, radians).
        target_pose:      ``[x, y, z[, yaw]]`` of the destination.
        safe_height:      Minimum Y clearance during transit (metres).
        obstacles:        Optional list of ``(pose, object_height)`` pairs for
                          other objects on the workbench.
        clearance_buffer: Extra headroom above any obstructing obstacle (metres).
        obstacle_radius:  XZ exclusion radius around each obstacle (metres).

    Returns:
        List of three ``[x, y, z, yaw]`` waypoints.
    """
    cx, cy, cz = current_pose[0], current_pose[1], current_pose[2]
    tx, ty, tz = target_pose[0],  target_pose[1],  target_pose[2]
    target_yaw = _pose_yaw(target_pose)

    safe_y = float(max(safe_height, cy, ty))
    safe_y = _obstacle_raised_ceiling(
        safe_y, current_pose, target_pose,
        obstacles, obstacle_radius, clearance_buffer,
    )

    return [
        [cx, safe_y, cz, target_yaw],   # 1. lift + pre-rotate
        [tx, safe_y, tz, target_yaw],   # 2. translate XZ at ceiling
        [tx, ty,     tz, target_yaw],   # 3. descend to target
    ]


def compute_grasp_trajectory(
    current_pose: list[float],
    target_pose: list[float],
    grasp_config: dict,
    safe_height: float = 0.30,
    obstacles: list[Obstacle] | None = None,
    clearance_buffer: float = 0.05,
    obstacle_radius: float = 0.08,
) -> list[list[float]]:
    """
    Four-waypoint top-down grasp trajectory respecting the object's config.

    The extra *approach* waypoint (step 3) gives the arm time to slow down
    and ensures the gripper is aligned before making contact with the object.

    Target yaw is taken from ``target_pose`` and offset by
    ``grasp_config["wrist_angle"]`` (degrees, converted internally).

    Waypoints
    ---------
    1. Lift to *safe_height* (obstacle-aware).
    2. Translate XZ to directly above the target at *safe_height*.
    3. Descend to ``target_y + approach_height_offset`` (pre-grasp hover).
    4. Descend to ``target_y + grasp_height_offset`` (grasp contact point).

    Returns:
        List of four ``[x, y, z, yaw]`` waypoints.
    """
    cx, cy, cz = current_pose[0], current_pose[1], current_pose[2]
    tx, ty, tz = target_pose[0],  target_pose[1],  target_pose[2]

    approach_y = ty + float(grasp_config["approach_height_offset"])
    grasp_y    = ty + float(grasp_config["grasp_height_offset"])

    target_yaw = (
        _pose_yaw(target_pose)
        + math.radians(float(grasp_config.get("wrist_angle", 0.0)))
    )

    safe_y = float(max(safe_height, cy, approach_y))
    safe_y = _obstacle_raised_ceiling(
        safe_y, current_pose, target_pose,
        obstacles, obstacle_radius, clearance_buffer,
    )

    return [
        [cx, safe_y,     cz, target_yaw],   # 1. lift + pre-rotate
        [tx, safe_y,     tz, target_yaw],   # 2. translate XZ
        [tx, approach_y, tz, target_yaw],   # 3. pre-grasp hover
        [tx, grasp_y,    tz, target_yaw],   # 4. grasp contact
    ]


def compute_side_grasp_trajectory(
    current_pose: list[float],
    target_pose: list[float],
    grasp_config: dict,
    safe_height: float = 0.30,
    obstacles: list[Obstacle] | None = None,
    clearance_buffer: float = 0.05,
    obstacle_radius: float = 0.08,
) -> list[list[float]]:
    """
    Four-waypoint grasp trajectory that approaches horizontally, perpendicular
    to the object's yaw — used for thin handled objects lying on their side
    (bamboo whisk, bamboo scoop).

    The approach direction in the XZ plane is perpendicular to the detected
    yaw: for yaw θ, we stand off at::

        standoff_xz = target_xz + standoff * (-sin θ, cos θ)

    and then slide into the target along the opposite direction.  This
    brings the gripper jaws down over the handle from the side while they
    remain open, then closes around the handle at the contact waypoint.

    Waypoints
    ---------
    1. Lift to *safe_height* (obstacle-aware).
    2. Translate XZ at *safe_height* to directly above the standoff point.
    3. Descend at the standoff point to ``target_y + approach_height_offset``.
    4. Slide horizontally into ``target_xz`` at ``target_y + grasp_height_offset``.

    Returns:
        List of four ``[x, y, z, yaw]`` waypoints.
    """
    cx, cy, cz = current_pose[0], current_pose[1], current_pose[2]
    tx, ty, tz = target_pose[0],  target_pose[1],  target_pose[2]

    approach_y = ty + float(grasp_config["approach_height_offset"])
    grasp_y    = ty + float(grasp_config["grasp_height_offset"])
    standoff   = float(grasp_config.get("approach_standoff", 0.05))

    object_yaw = _pose_yaw(target_pose)
    target_yaw = object_yaw + math.radians(float(grasp_config.get("wrist_angle", 0.0)))

    # Perpendicular-to-handle direction in XZ plane.
    perp_x = -math.sin(object_yaw)
    perp_z =  math.cos(object_yaw)
    standoff_x = tx + standoff * perp_x
    standoff_z = tz + standoff * perp_z

    safe_y = float(max(safe_height, cy, approach_y))
    safe_y = _obstacle_raised_ceiling(
        safe_y, current_pose, target_pose,
        obstacles, obstacle_radius, clearance_buffer,
    )

    return [
        [cx,         safe_y,     cz,         target_yaw],  # 1. lift + pre-rotate
        [standoff_x, safe_y,     standoff_z, target_yaw],  # 2. translate to standoff XZ
        [standoff_x, approach_y, standoff_z, target_yaw],  # 3. descend at standoff
        [tx,         grasp_y,    tz,         target_yaw],  # 4. slide into contact
    ]


def compute_place_trajectory(
    current_pose: list[float],
    target_pose: list[float],
    safe_height: float = 0.30,
    place_height_offset: float = 0.02,
    obstacles: list[Obstacle] | None = None,
    clearance_buffer: float = 0.05,
    obstacle_radius: float = 0.08,
) -> list[list[float]]:
    """
    Three-waypoint place trajectory for setting an object down.

    The arm stops at *place_height_offset* above the target surface before
    the caller opens the gripper, so the object is placed gently rather than
    dropped from height.

    Waypoints
    ---------
    1. Lift to *safe_height* (obstacle-aware; clears workspace while holding).
    2. Translate XZ to directly above the drop-off point.
    3. Descend to ``target_y + place_height_offset``.

    After reaching waypoint 3 the caller should call ``open_gripper()``.

    Returns:
        List of three ``[x, y, z, yaw]`` waypoints.
    """
    cx, cy, cz = current_pose[0], current_pose[1], current_pose[2]
    tx, ty, tz = target_pose[0],  target_pose[1],  target_pose[2]
    target_yaw = _pose_yaw(target_pose)

    place_y = ty + float(place_height_offset)
    safe_y  = float(max(safe_height, cy, place_y))
    safe_y  = _obstacle_raised_ceiling(
        safe_y, current_pose, target_pose,
        obstacles, obstacle_radius, clearance_buffer,
    )

    return [
        [cx, safe_y,  cz, target_yaw],   # 1. lift
        [tx, safe_y,  tz, target_yaw],   # 2. translate XZ
        [tx, place_y, tz, target_yaw],   # 3. descend to place height (open gripper next)
    ]
