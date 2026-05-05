"""Pure-math tests for grasp_library — no MuJoCo dependency."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grasp_library import (
    CUP,
    PLACEMENT_PAD,
    THREE_CUP_STACK,
    GraspPose,
    TAG_BODY_TO_LIBRARY,
    TagAnchor,
    TagDetection,
    fuse_object_pose,
    object_pose_from_tag,
    world_grasp_from_object,
)


def _R_x(deg: float) -> np.ndarray:
    r = np.deg2rad(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _R_y(deg: float) -> np.ndarray:
    r = np.deg2rad(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _R_z(deg: float) -> np.ndarray:
    r = np.deg2rad(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _synth_detection(obj_pos: np.ndarray, obj_rot: np.ndarray, anchor: TagAnchor) -> TagDetection:
    """Build the TagDetection that perfect detection of `anchor` would produce."""
    tag_pos_world = obj_pos + obj_rot @ anchor.pos_in_object
    tag_rot_world = obj_rot @ anchor.rot_in_object
    return TagDetection(tag_id=anchor.tag_id, world_position=tag_pos_world, world_rotation=tag_rot_world)


def test_placement_pad_tag_roundtrip():
    """Placement pad uses tag 9 with same anchor geometry as the stack bottom tag."""
    obj_pos = np.array([0.32, -0.17, 0.003])
    obj_rot = _R_z(5.0)
    anchor = PLACEMENT_PAD.anchors[0]
    assert anchor.tag_id == 9
    det = _synth_detection(obj_pos, obj_rot, anchor)
    recovered_pos, recovered_rot = object_pose_from_tag(
        det.world_position, det.world_rotation, anchor
    )
    assert np.allclose(recovered_pos, obj_pos, atol=1e-9)
    assert np.allclose(recovered_rot, obj_rot, atol=1e-9)


def test_cup_stack_single_tag_roundtrip():
    """Stack object: one tag ID recovers full-body pose (generalization smoke test)."""
    obj_pos = np.array([0.31, -0.02, 0.05])
    obj_rot = _R_z(-12.0)
    anchor = THREE_CUP_STACK.anchors[0]
    det = _synth_detection(obj_pos, obj_rot, anchor)
    recovered_pos, recovered_rot = object_pose_from_tag(
        det.world_position, det.world_rotation, anchor
    )
    assert np.allclose(recovered_pos, obj_pos, atol=1e-9)
    assert np.allclose(recovered_rot, obj_rot, atol=1e-9)


def test_object_pose_from_single_tag_roundtrip():
    obj_pos = np.array([0.32, 0.05, 0.045])
    obj_rot = _R_z(30.0)
    for anchor in CUP.anchors:
        det = _synth_detection(obj_pos, obj_rot, anchor)
        recovered_pos, recovered_rot = object_pose_from_tag(
            det.world_position, det.world_rotation, anchor
        )
        assert np.allclose(recovered_pos, obj_pos, atol=1e-9), (
            f"position roundtrip failed for tag {anchor.tag_id}"
        )
        assert np.allclose(recovered_rot, obj_rot, atol=1e-9), (
            f"rotation roundtrip failed for tag {anchor.tag_id}"
        )


def test_fuse_two_tags_recovers_yaw():
    obj_pos = np.array([0.30, 0.10, 0.045])
    obj_rot = _R_z(30.0)
    detections = [_synth_detection(obj_pos, obj_rot, anchor) for anchor in CUP.anchors]
    fused_pos, fused_rot = fuse_object_pose(detections, CUP)
    assert np.allclose(fused_pos, obj_pos, atol=1e-9)
    assert np.allclose(fused_rot, obj_rot, atol=1e-9)


def test_fuse_one_noisy_tag_does_not_collapse():
    obj_pos = np.array([0.30, 0.0, 0.045])
    obj_rot = _R_z(15.0)
    rng = np.random.default_rng(0)
    clean = _synth_detection(obj_pos, obj_rot, CUP.anchors[0])
    noisy = _synth_detection(obj_pos, obj_rot, CUP.anchors[1])
    noisy = TagDetection(
        tag_id=noisy.tag_id,
        world_position=noisy.world_position + rng.normal(scale=0.002, size=3),
        world_rotation=noisy.world_rotation @ _R_z(0.5),
    )
    fused_pos, fused_rot = fuse_object_pose([clean, noisy], CUP)
    pos_err = np.linalg.norm(fused_pos - obj_pos)
    yaw_err = abs(np.degrees(np.arctan2(fused_rot[1, 0], fused_rot[0, 0])) - 15.0)
    assert pos_err < 0.0015, f"fused pos error {pos_err} is too large"
    assert yaw_err < 0.4, f"fused yaw error {yaw_err} deg is too large"


def test_fuse_ignores_unrelated_tags():
    obj_pos = np.array([0.30, 0.0, 0.045])
    obj_rot = np.eye(3)
    cup_det = _synth_detection(obj_pos, obj_rot, CUP.anchors[0])
    foreign = TagDetection(
        tag_id=99,
        world_position=np.array([1.0, 2.0, 3.0]),
        world_rotation=_R_x(45.0),
    )
    fused_pos, fused_rot = fuse_object_pose([foreign, cup_det], CUP)
    assert np.allclose(fused_pos, obj_pos, atol=1e-9)
    assert np.allclose(fused_rot, obj_rot, atol=1e-9)


def test_fuse_raises_when_no_anchor_tags_match():
    foreign = TagDetection(
        tag_id=42,
        world_position=np.zeros(3),
        world_rotation=np.eye(3),
    )
    try:
        fuse_object_pose([foreign], CUP)
    except RuntimeError:
        return
    raise AssertionError("Expected RuntimeError when no anchor tags are detected.")


def test_world_grasp_translates_with_object():
    obj_pos = np.array([0.40, -0.10, 0.045])
    obj_rot = np.eye(3)
    grasp = CUP.grasps[0]
    # zero gripper offset → returned claw_target equals the logical pinch
    g_pos, g_rot, pre_pos = world_grasp_from_object(
        grasp, obj_pos, obj_rot, grip_pad_offset_in_claw=np.zeros(3)
    )
    expected_pinch_pos = obj_pos + grasp.pos_in_object
    assert np.allclose(g_pos, expected_pinch_pos, atol=1e-9)
    assert np.allclose(g_rot, grasp.rot_in_object, atol=1e-9)
    approach = g_rot[:, 0]
    expected_pre = expected_pinch_pos - grasp.pregrasp_back_off_m * approach
    assert np.allclose(pre_pos, expected_pre, atol=1e-9)


def test_world_grasp_rotates_with_object():
    obj_pos = np.array([0.32, 0.0, 0.045])
    obj_rot = _R_z(90.0)
    grasp = CUP.grasps[0]
    g_pos, g_rot, pre_pos = world_grasp_from_object(
        grasp, obj_pos, obj_rot, grip_pad_offset_in_claw=np.zeros(3)
    )
    delta = g_pos - obj_pos
    assert np.allclose(delta, obj_rot @ grasp.pos_in_object, atol=1e-9)
    assert np.allclose(g_rot, obj_rot @ grasp.rot_in_object, atol=1e-9)
    approach_world = g_rot[:, 0]
    assert np.allclose(pre_pos, g_pos - grasp.pregrasp_back_off_m * approach_world, atol=1e-9)


def test_world_grasp_applies_grip_pad_offset():
    obj_pos = np.array([0.32, 0.0, 0.045])
    obj_rot = np.eye(3)
    grasp = CUP.grasps[0]
    custom_offset = np.array([-0.02, 0.0, -0.01])
    g_pos, g_rot, _ = world_grasp_from_object(
        grasp, obj_pos, obj_rot, grip_pad_offset_in_claw=custom_offset
    )
    # The pinch point in world is obj_pos + grasp.pos_in_object; the IK
    # claw_target is shifted *opposite* the grip-pad offset so the pads land
    # on the pinch point.
    expected_pinch_pos = obj_pos + grasp.pos_in_object
    expected_claw_target = expected_pinch_pos - g_rot @ custom_offset
    assert np.allclose(g_pos, expected_claw_target, atol=1e-9)


def test_grasp_pose_validates_shapes():
    try:
        GraspPose(
            pos_in_object=np.array([0.0, 0.0]),
            rot_in_object=np.eye(3),
            pregrasp_back_off_m=0.05,
            gripper_open=50.0,
            gripper_close=-5.0,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("GraspPose should reject non-(3,) position.")


def test_tag_body_to_library_is_self_inverse():
    # diag(1, -1, -1) is its own inverse — applying it twice gets us back
    # to the body frame.
    assert np.allclose(TAG_BODY_TO_LIBRARY @ TAG_BODY_TO_LIBRARY, np.eye(3))


def test_tag_to_object_index_round_trips_anchors():
    from grasp_library import TAG_TO_OBJECT  # local import for clarity

    for tag_id in (6, 7):
        obj, anchor = TAG_TO_OBJECT[tag_id]
        assert obj is CUP
        assert anchor.tag_id == tag_id

    obj8, anchor8 = TAG_TO_OBJECT[8]
    assert obj8 is THREE_CUP_STACK
    assert anchor8.tag_id == 8

    obj9, anchor9 = TAG_TO_OBJECT[9]
    assert obj9 is PLACEMENT_PAD
    assert anchor9.tag_id == 9


if __name__ == "__main__":
    # Lightweight runner: pytest-free, prints PASS/FAIL.
    fns = {name: fn for name, fn in globals().items() if name.startswith("test_") and callable(fn)}
    failed = []
    for name, fn in fns.items():
        try:
            fn()
        except AssertionError as exc:
            failed.append((name, str(exc)))
            print(f"FAIL {name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed.append((name, repr(exc)))
            print(f"ERROR {name}: {exc!r}")
        else:
            print(f"PASS {name}")
    if failed:
        sys.exit(1)
