"""Per-object grasp library: maps AprilTag detections to a graspable target pose.

`TaggedObject` declares one or more `TagAnchor`s (telling us where each tag
sits inside the object's body frame) and one or more candidate `GraspPose`s
(where the gripper should pinch, also in object frame). At runtime the planner
takes the AprilTag detections, fuses them into a single object pose, and
transforms a chosen grasp into the world frame.

Frame conventions
-----------------
* Object frame: matches the MuJoCo body frame of the tagged object.
* AprilTag library frame (what `pose_estimation.detect_apriltags` returns):
  +x right of the tag image, +y down in the tag image, +z **into** the card
  (away from the camera). This is *not* the same as the MuJoCo body frame of
  the tag mesh; lib differs from body by a 180° rotation about the tag's x
  axis (i.e. `diag(1, -1, -1)`).
* Gripper / claw frame (what `motion.solve_ik` consumes): +x is the approach
  direction the gripper drives along to reach the target, +y is the
  jaw-opening direction (jaws translate along ±y to close), +z is the
  remaining right-handed axis. See `so101_kinematics.target_relative_gripper_rotation`.

The math is intentionally pure-numpy and free of any MuJoCo dependency so it
can be unit-tested without spinning up a simulator.

`pos_in_object` describes the *logical* point on the object where the gripper
should pinch (e.g. the center of a handle bar). The world claw-target that
`motion.solve_ik` consumes is offset from that pinch location by a
gripper-specific constant (`grip_pad_offset_in_claw`, default
`so101_kinematics.SO101_CLAW_TARGET_TO_GRIP_PAD_OFFSET`). Swap the offset to
support a different gripper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from so101_kinematics import SO101_CLAW_TARGET_TO_GRIP_PAD_OFFSET


# 180° rotation about x. Converts MuJoCo tag-mesh body axes to AprilTag-library
# axes (lib +x = body +x, lib +y = body -y, lib +z = body -z).
TAG_BODY_TO_LIBRARY = np.diag([1.0, -1.0, -1.0])


def _require_vec3(name: str, value: np.ndarray | tuple[float, float, float] | list[float]) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"{name} must have shape (3,); got {arr.shape}.")
    return arr


def _require_rot3(name: str, value: np.ndarray | list[list[float]]) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3, 3); got {arr.shape}.")
    return arr


@dataclass(frozen=True)
class TagAnchor:
    """A tag's static placement on a tagged object, expressed in the object frame.

    `pos_in_object` is the tag origin in the object's body frame.
    `rot_in_object` is the rotation from AprilTag-library frame to object frame
    (i.e. v_obj = rot_in_object @ v_lib). Build it from the MuJoCo body
    rotation by `R_body_in_object @ TAG_BODY_TO_LIBRARY` -- see `CUP` below.
    """

    tag_id: int
    size_m: float
    pos_in_object: np.ndarray
    rot_in_object: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "pos_in_object", _require_vec3("pos_in_object", self.pos_in_object))
        object.__setattr__(self, "rot_in_object", _require_rot3("rot_in_object", self.rot_in_object))


@dataclass(frozen=True)
class GraspPose:
    """A candidate grasp expressed in the object's body frame.

    `rot_in_object` rotates a vector from the gripper claw frame to the
    object frame (v_obj = rot_in_object @ v_claw). Column 0 of the matrix is
    therefore the approach direction expressed in the object frame; the
    pregrasp is computed by stepping back along this direction.
    """

    pos_in_object: np.ndarray
    rot_in_object: np.ndarray
    pregrasp_back_off_m: float
    gripper_open: float
    gripper_close: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "pos_in_object", _require_vec3("pos_in_object", self.pos_in_object))
        object.__setattr__(self, "rot_in_object", _require_rot3("rot_in_object", self.rot_in_object))


@dataclass(frozen=True)
class TaggedObject:
    name: str
    anchors: tuple[TagAnchor, ...]
    grasps: tuple[GraspPose, ...]
    tag_sizes: dict[int, float] = field(init=False)

    def __post_init__(self) -> None:
        if not self.anchors:
            raise ValueError(f"TaggedObject {self.name!r} must have at least one anchor.")
        if not self.grasps:
            raise ValueError(f"TaggedObject {self.name!r} must have at least one grasp.")
        seen: set[int] = set()
        for anchor in self.anchors:
            if anchor.tag_id in seen:
                raise ValueError(f"Duplicate tag_id {anchor.tag_id} on object {self.name!r}.")
            seen.add(anchor.tag_id)
        object.__setattr__(self, "tag_sizes", {a.tag_id: a.size_m for a in self.anchors})

    def anchor_for(self, tag_id: int) -> TagAnchor | None:
        for anchor in self.anchors:
            if anchor.tag_id == tag_id:
                return anchor
        return None


# ---------------------------------------------------------------------------
# Pose math
# ---------------------------------------------------------------------------


def object_pose_from_tag(
    tag_pos_world: np.ndarray,
    tag_rot_world: np.ndarray,
    anchor: TagAnchor,
) -> tuple[np.ndarray, np.ndarray]:
    """Recover an object's world pose from a single tag observation.

    `tag_rot_world` must be the rotation from the AprilTag library frame to
    the world frame (i.e. exactly what `pose_estimation.TagPoseEstimate.world_rotation`
    returns).
    """

    tag_pos_world = _require_vec3("tag_pos_world", tag_pos_world)
    tag_rot_world = _require_rot3("tag_rot_world", tag_rot_world)
    obj_rot = tag_rot_world @ anchor.rot_in_object.T
    obj_pos = tag_pos_world - obj_rot @ anchor.pos_in_object
    return obj_pos, obj_rot


def _rotation_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    # Shepperd's method; returns (w, x, y, z) with w >= 0.
    m = rotation
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    quat = np.array([w, x, y, z], dtype=float)
    if quat[0] < 0.0:
        quat = -quat
    return quat / np.linalg.norm(quat)


def _quaternion_to_rotation(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _average_rotation(rotations: list[np.ndarray]) -> np.ndarray:
    # Sign-aligned quaternion mean. Cheap and adequate when inputs are close.
    if not rotations:
        raise ValueError("rotations must be non-empty.")
    quats = np.stack([_rotation_to_quaternion(r) for r in rotations], axis=0)
    reference = quats[0]
    aligned = np.where((quats @ reference)[:, None] < 0, -quats, quats)
    mean = aligned.mean(axis=0)
    norm = np.linalg.norm(mean)
    if norm < 1e-9:
        # Antipodal disagreement; fall back to the first rotation.
        return rotations[0]
    return _quaternion_to_rotation(mean / norm)


@dataclass(frozen=True)
class TagDetection:
    """Minimal subset of `pose_estimation.TagPoseEstimate` used by the library.

    Lets unit tests run without depending on the full simulator pipeline.
    """

    tag_id: int
    world_position: np.ndarray
    world_rotation: np.ndarray


def fuse_object_pose(
    detections: Iterable[TagDetection],
    obj: TaggedObject,
) -> tuple[np.ndarray, np.ndarray]:
    """Combine tag observations into a single (position, rotation) for the object."""

    candidate_positions: list[np.ndarray] = []
    candidate_rotations: list[np.ndarray] = []
    seen_ids: set[int] = set()
    for detection in detections:
        anchor = obj.anchor_for(detection.tag_id)
        if anchor is None or detection.tag_id in seen_ids:
            continue
        seen_ids.add(detection.tag_id)
        pos, rot = object_pose_from_tag(detection.world_position, detection.world_rotation, anchor)
        candidate_positions.append(pos)
        candidate_rotations.append(rot)
    if not candidate_positions:
        raise RuntimeError(
            f"No detections in {[d.tag_id for d in detections]} match anchors for {obj.name!r} "
            f"({[a.tag_id for a in obj.anchors]})."
        )
    if len(candidate_positions) == 1:
        return candidate_positions[0], candidate_rotations[0]
    return (
        np.mean(np.stack(candidate_positions, axis=0), axis=0),
        _average_rotation(candidate_rotations),
    )


def world_grasp_from_object(
    grasp: GraspPose,
    obj_pos_world: np.ndarray,
    obj_rot_world: np.ndarray,
    grip_pad_offset_in_claw: np.ndarray = SO101_CLAW_TARGET_TO_GRIP_PAD_OFFSET,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transform a body-frame grasp to a world-frame IK target plus pregrasp.

    `grasp.pos_in_object` is the *logical* pinch location: the point on the
    object where the gripper's grip pads should land. Because the IK solver's
    "claw_target" frame is offset from where the pads actually sit (see
    `grip_pad_offset_in_claw`), the returned world-frame position is biased
    along the grasp's own axes by exactly that offset, so commanding the IK to
    the returned position lands the pads on the requested pinch point.

    Returns `(claw_target_world, claw_rot_world, pregrasp_world)` ready to
    feed to `motion.solve_ik(env, xyz, rotation=...)`.
    """

    obj_pos_world = _require_vec3("obj_pos_world", obj_pos_world)
    obj_rot_world = _require_rot3("obj_rot_world", obj_rot_world)
    grip_pad_offset_in_claw = _require_vec3("grip_pad_offset_in_claw", grip_pad_offset_in_claw)

    pinch_pos_world = obj_pos_world + obj_rot_world @ grasp.pos_in_object
    grasp_rot_world = obj_rot_world @ grasp.rot_in_object
    # Grip pad sits at claw_target + R_claw @ offset, so to put the pad at
    # `pinch_pos_world` we subtract the offset.
    claw_target_world = pinch_pos_world - grasp_rot_world @ grip_pad_offset_in_claw
    approach_world = grasp_rot_world[:, 0]
    pregrasp_pos_world = claw_target_world - grasp.pregrasp_back_off_m * approach_world
    return claw_target_world, grasp_rot_world, pregrasp_pos_world


# ---------------------------------------------------------------------------
# Object library
# ---------------------------------------------------------------------------


def _rotation_about_x(deg: float) -> np.ndarray:
    rad = np.deg2rad(deg)
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def _rotation_about_y(deg: float) -> np.ndarray:
    rad = np.deg2rad(deg)
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def _rotation_about_z(deg: float) -> np.ndarray:
    rad = np.deg2rad(deg)
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


# Cup tag #6 sits on the cup -x face. Its MuJoCo body rotation in cup frame
# is R_y(-90°): mesh +z (the printed-face normal) maps to cup -x. The library
# frame is the body frame post-multiplied by TAG_BODY_TO_LIBRARY.
_TAG6_ROT_IN_CUP = _rotation_about_y(-90.0) @ TAG_BODY_TO_LIBRARY
_TAG7_ROT_IN_CUP = _rotation_about_x(90.0) @ TAG_BODY_TO_LIBRARY


# Horizontal grasp of the +y handle. The SO-101 gripper joint is a hinge
# rotating about claw_y, so the moving jaw swings in the claw_x-claw_z plane
# while claw_y stays constant. Pick claw_y = +obj_z so the moving jaw arc
# stays at constant world z (the bar's vertical extent already covers that
# height). claw_z = claw_x × claw_y = -obj_y points the moving jaw toward
# obj -y at OPEN, which sweeps through the bar's location during closing
# (the gripper's grip-pad offset shift puts the gripper *past* the bar in
# +y, so the bar lies between the open and closed jaw positions).
_HANDLE_GRASP_ROT_IN_CUP = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=float,
)


CUP = TaggedObject(
    name="cup",
    anchors=(
        TagAnchor(
            tag_id=6,
            size_m=0.024,
            pos_in_object=np.array([-0.026, 0.0, 0.067]),
            rot_in_object=_TAG6_ROT_IN_CUP,
        ),
        TagAnchor(
            tag_id=7,
            size_m=0.024,
            pos_in_object=np.array([0.0, -0.026, 0.0]),
            rot_in_object=_TAG7_ROT_IN_CUP,
        ),
    ),
    grasps=(
        GraspPose(
            # Logical pinch point: middle of the +y handle bar. The bar sits
            # 6 cm out on +obj_y; the cup wall ends at radius 0.023 so there
            # is ~3.7 cm of clearance between bar and cup body, which is
            # enough room for one gripper jaw to fit between them while the
            # other jaw closes from outside.
            pos_in_object=np.array([0.0, 0.060, 0.012]),
            rot_in_object=_HANDLE_GRASP_ROT_IN_CUP,
            pregrasp_back_off_m=0.04,
            gripper_open=50.0,
            gripper_close=-5.0,
        ),
    ),
)


# Three identical cups stacked as separate bodies (`scene_cup_stack.xml`). Tag ID 8 is on the
# **bottom** cup only so it stays visible for vision while removing top → mid → bottom.
# Nominal cup spacing along +z in the bottom cup frame: mid center +0.09 m, top +0.18 m
# (center-to-center = 2 * cylinder half_height).
THREE_CUP_STACK = TaggedObject(
    name="three_cup_stack",
    anchors=(
        TagAnchor(
            tag_id=8,
            size_m=0.024,
            pos_in_object=np.array([-0.026, 0.0, 0.067]),
            rot_in_object=_TAG6_ROT_IN_CUP,
        ),
    ),
    grasps=(
        GraspPose(
            pos_in_object=np.array([0.0, 0.060, 0.012]),
            rot_in_object=_HANDLE_GRASP_ROT_IN_CUP,
            pregrasp_back_off_m=0.04,
            gripper_open=50.0,
            gripper_close=-5.0,
        ),
    ),
)

# Removal order (top → mid → bottom).
THREE_CUP_STACK_REMOVAL_ORDER: tuple[str, ...] = ("cup_top", "cup_mid", "cup_bottom")

# Nominal spacing for documentation / pure-math tests (bottom origin → member origin in bottom frame).
THREE_CUP_STACK_NOMINAL_OFFSETS_M: dict[str, np.ndarray] = {
    "cup_top": np.array([0.0, 0.0, 0.18]),
    "cup_mid": np.array([0.0, 0.0, 0.09]),
    "cup_bottom": np.array([0.0, 0.0, 0.0]),
}


def stack_member_pose_from_bottom_pose(
    bottom_pos_world: np.ndarray,
    bottom_rot_world: np.ndarray,
    offset_bottom_origin_to_member_origin: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """World pose of a stacked cup given fused bottom-cup pose and nominal stack offset."""

    offset_bottom_origin_to_member_origin = _require_vec3(
        "offset_bottom_origin_to_member_origin", offset_bottom_origin_to_member_origin
    )
    member_pos = bottom_pos_world + bottom_rot_world @ offset_bottom_origin_to_member_origin
    # Stacked upright: member cup shares bottom cup yaw (small-angle approximation).
    return member_pos, bottom_rot_world


# Fixed placement pad (`scene_cup_stack.xml`): tag ID 9 on a separate body from the stack (tag 8).
# Same tag mesh mount as the bottom cup so fusion math matches the MJCF.
PLACEMENT_PAD = TaggedObject(
    name="placement_pad",
    anchors=(
        TagAnchor(
            tag_id=9,
            size_m=0.024,
            pos_in_object=np.array([-0.026, 0.0, 0.067]),
            rot_in_object=_TAG6_ROT_IN_CUP,
        ),
    ),
    grasps=(
        GraspPose(
            pos_in_object=np.array([0.0, 0.060, 0.012]),
            rot_in_object=_HANDLE_GRASP_ROT_IN_CUP,
            pregrasp_back_off_m=0.04,
            gripper_open=50.0,
            gripper_close=-5.0,
        ),
    ),
)


# Spoon (`scene_spoon.xml`): handle is a thin capsule along object +x at z=0;
# tag 10 sits flat on top of the handle (mesh body axes match spoon body axes,
# so R_body_in_object = I and rot_in_object = TAG_BODY_TO_LIBRARY).
#
# Top-down grasp: the SO-101 moving jaw rotates about claw_y, so the closing
# arc lies in the claw_x-claw_z plane. To clamp a thin horizontal handle, the
# handle's long axis must align with claw_y so the arc sweeps perpendicular to
# it. With approach claw_x = -obj_z (down) and claw_y = +obj_x (along handle),
# claw_z = claw_x x claw_y = -obj_y.
_TAG10_ROT_IN_SPOON = TAG_BODY_TO_LIBRARY  # tag mesh axes == spoon body axes

# Grasp 0 — top-down pinch: claw_x = -obj_z (down), claw_y = +obj_x (along
# handle), claw_z = -obj_y. Moving-jaw arc sweeps perpendicular to handle.
_SPOON_TOPDOWN_GRASP_ROT = np.array(
    [
        [0.0, 1.0, 0.0],
        [0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0],
    ],
    dtype=float,
)

# Grasp 1 — scooping grip: approach at 45° between top-down and side (+y).
# claw_x = (0, 1/√2, -1/√2), claw_y = +obj_x (handle), claw_z = (0, -1/√2, -1/√2).
# Achievable on the 5-DoF arm; spoon is tilted ~45° when lifted — natural
# orientation for holding a spoon to eat / scoop.
_S = float(np.sqrt(0.5))
_SPOON_SCOOP_GRASP_ROT = np.array(
    [
        [0.0, 1.0, 0.0],
        [_S, 0.0, -_S],
        [-_S, 0.0, -_S],
    ],
    dtype=float,
)


SPOON = TaggedObject(
    name="spoon",
    anchors=(
        TagAnchor(
            tag_id=10,
            size_m=0.024,
            pos_in_object=np.array([-0.020, 0.0, 0.020]),
            rot_in_object=_TAG10_ROT_IN_SPOON,
        ),
    ),
    grasps=(
        GraspPose(
            pos_in_object=np.array([-0.015, 0.0, 0.009]),
            rot_in_object=_SPOON_TOPDOWN_GRASP_ROT,
            pregrasp_back_off_m=0.06,
            gripper_open=50.0,
            gripper_close=-15.0,
        ),
        GraspPose(
            # Same handle point; approach from +y so bowl faces forward after lift.
            pos_in_object=np.array([-0.015, 0.0, 0.009]),
            rot_in_object=_SPOON_SCOOP_GRASP_ROT,
            pregrasp_back_off_m=0.06,
            gripper_open=50.0,
            gripper_close=-15.0,
        ),
    ),
)


OBJECTS: tuple[TaggedObject, ...] = (CUP, THREE_CUP_STACK, PLACEMENT_PAD, SPOON)


def _build_tag_index() -> dict[int, tuple[TaggedObject, TagAnchor]]:
    index: dict[int, tuple[TaggedObject, TagAnchor]] = {}
    for obj in OBJECTS:
        for anchor in obj.anchors:
            if anchor.tag_id in index:
                raise ValueError(
                    f"Tag id {anchor.tag_id} is registered to more than one TaggedObject."
                )
            index[anchor.tag_id] = (obj, anchor)
    return index


TAG_TO_OBJECT: dict[int, tuple[TaggedObject, TagAnchor]] = _build_tag_index()
