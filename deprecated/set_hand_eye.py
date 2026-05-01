"""Hand-author T_flange_camera and overwrite hand_eye_calib.npz.

Edit the values below and run:

    python set_hand_eye.py

Then re-launch wrist_extrinsic.py -- it loads the new T_flange_camera on
startup via hand_eye_calib.load_hand_eye_calib().

You specify the camera offset (and orientation) in any URDF link's
coordinate frame, not just the flange's. The script reads the URDF,
verifies the chosen reference frame is rigidly attached to the flange
(only fixed joints in between -- otherwise T_flange_camera wouldn't be
constant as the arm moves), and composes the final T_flange_camera as

    T_flange_camera = T_flange_link @ T_link_camera

Frame conventions:
    flange frame        :  whatever placo computes for `gripper_frame_link`
                           from SO101/so101_new_calib.urdf
    reference frame     :  any URDF link rigidly fixed to the flange. For
                           SO-101 that's `gripper_link` or `gripper_frame_link`.
    camera frame        :  OpenCV convention -- +Z out of the lens,
                           +X right in the image, +Y down in the image
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from hand_eye_calib import (
    build_T_flange_camera,
    load_hand_eye_calib,
    save_hand_eye_calib,
)
from render_robot import load_robot_model

URDF_PATH = "SO101/so101_new_calib.urdf"
FLANGE_LINK = "gripper_frame_link"

# ---------------------------------------------------------------------------
# Reference frame: which URDF link the offset/rotation are measured in.
# Must be rigidly connected to FLANGE_LINK (fixed joints only) so the
# resulting T_flange_camera is constant across joint motion.
#
# For SO-101 the only valid choices are:
#   "gripper_frame_link"  : the flange itself (offsets interpreted directly)
#   "gripper_link"        : sits 98 mm "above" the flange along flange -Z
# ---------------------------------------------------------------------------
REFERENCE_FRAME = "gripper_link"

# ---------------------------------------------------------------------------
# Translation: where the camera optical centre sits in REFERENCE_FRAME.
#
# Useful landmarks in `gripper_link` coords (mm):
#   gripper_link origin            : (  0.0,   0.0,   0.0)
#   gripper motor body (sts3215)   : ( +7.7,  +0.1, -23.4)   <-- pre-filled
#   gripper joint axis (jaw pivot) : (+20.2, +18.8, -23.4)
#   wrist_roll motor body          : (+12.7,  +7.3, +93.5)
#
# Useful landmarks in `gripper_frame_link` coords (mm):
#   flange origin                  : (  0.0,   0.0,   0.0)
#   gripper_link origin            : ( -7.9,  +0.2, -98.1)
#   wrist_roll motor body          : ( -7.3, +12.7, -116.8)
# ---------------------------------------------------------------------------
TX_MM = 7.7
TY_MM = 100.1   # gripper motor body (0.1) + 35 mm "perpendicularly above" along gripper_link +Y
TZ_MM = -23.4

# ---------------------------------------------------------------------------
# Camera orientation strategy. Choose one:
#
#   "keep_old" : keep R_flange_camera EXACTLY as it is on disk. Only the
#                translation is updated. Use when relocating the camera
#                without changing where it's pointing.
#   "flange"   : apply ROTATION_CASE / TILT_X_DEG / FREEFORM_RPY_DEG below
#                as a rotation in the FLANGE frame (gripper_frame_link).
#                Easiest for "I want a 30 deg pitch about flange X" tweaks.
#   "link"     : same params, but interpreted in REFERENCE_FRAME and then
#                composed with T_flange_link. Use when the camera's natural
#                axes line up with the reference link (e.g. identity in
#                gripper_link means camera axes == gripper_link axes).
#
# In all three modes the TRANSLATION (TX_MM/TY_MM/TZ_MM) is interpreted in
# REFERENCE_FRAME -- only the rotation interpretation changes.
# ---------------------------------------------------------------------------
ROTATION_STRATEGY = "flange"

# Pick ONE rotation_case from hand_eye_calib.ROTATION_CASES (used by both
# "flange" and "link" strategies, ignored by "keep_old"):
#   "A_aligned"     : camera axes == frame axes (identity).
#   "B_back"        : 180 deg rotation about frame Y.
#   "C_down_neg_z"  : -90 deg rotation about frame X.
#   "D_tilt_x"      : pure pitch about frame X by tilt_x_deg.
#   "freeform_rpy"  : arbitrary roll/pitch/yaw (deg) about frame X/Y/Z
#                     applied as Rx @ Ry @ Rz.
#
# The values below reproduce the original D_tilt_x(25 deg) calibration in
# the flange frame. Bump TILT_X_DEG to tweak the pitch.
ROTATION_CASE = "D_tilt_x"
TILT_X_DEG = 19.0
FREEFORM_RPY_DEG = (0.0, 0.0, 0.0)


def _flange_to_link_transform(reference_frame: str) -> np.ndarray:
    """Return the constant transform T_flange_link from URDF FK.

    Validates that `reference_frame` is rigidly connected to FLANGE_LINK
    (fixed joints only) so the result is invariant under joint motion.
    """
    model = load_robot_model(Path(URDF_PATH).resolve())

    if reference_frame not in {v.link_name for v in model.visuals} | {
        FLANGE_LINK
    } and reference_frame not in {j.parent for j in model.joints} | {
        j.child for j in model.joints
    }:
        raise ValueError(
            f"reference_frame {reference_frame!r} is not a link in {URDF_PATH}"
        )

    visited = {FLANGE_LINK}
    queue = [FLANGE_LINK]
    while queue:
        cur = queue.pop(0)
        for joint in model.joints:
            if joint.joint_type != "fixed":
                continue
            if joint.parent == cur and joint.child not in visited:
                visited.add(joint.child)
                queue.append(joint.child)
            elif joint.child == cur and joint.parent not in visited:
                visited.add(joint.parent)
                queue.append(joint.parent)

    if reference_frame not in visited:
        raise ValueError(
            f"reference_frame {reference_frame!r} is NOT rigidly connected "
            f"to flange {FLANGE_LINK!r}; rigidly-connected links are "
            f"{sorted(visited)}. Pick one of those, otherwise the resulting "
            "T_flange_camera would change as the arm moves."
        )

    zeros = {
        joint.name: 0.0
        for joint in model.joints
        if joint.joint_type in {"revolute", "continuous"}
    }
    poses = model.forward_kinematics(zeros)
    return np.linalg.inv(poses[FLANGE_LINK]) @ poses[reference_frame]


def main() -> None:
    if ROTATION_STRATEGY not in {"keep_old", "flange", "link"}:
        raise ValueError(
            f"ROTATION_STRATEGY must be one of 'keep_old', 'flange', 'link' "
            f"-- got {ROTATION_STRATEGY!r}"
        )

    T_flange_link = _flange_to_link_transform(REFERENCE_FRAME)
    T_old = load_hand_eye_calib()

    # Translation: ALWAYS in REFERENCE_FRAME, composed through T_flange_link.
    t_link_camera = np.array([TX_MM, TY_MM, TZ_MM], dtype=np.float64) / 1000.0
    t_flange_camera = T_flange_link[:3, :3] @ t_link_camera + T_flange_link[:3, 3]

    # Rotation: depends on strategy.
    rotation_params_T = build_T_flange_camera(
        tx_mm=0.0,
        ty_mm=0.0,
        tz_mm=0.0,
        rotation_case=ROTATION_CASE,
        tilt_x_deg=TILT_X_DEG,
        freeform_rpy_deg=FREEFORM_RPY_DEG,
    )
    R_user = rotation_params_T[:3, :3]
    if ROTATION_STRATEGY == "keep_old":
        R_flange_camera = T_old[:3, :3].copy()
    elif ROTATION_STRATEGY == "flange":
        R_flange_camera = R_user
    else:  # "link"
        R_flange_camera = T_flange_link[:3, :3] @ R_user

    T_flange_camera = np.eye(4, dtype=np.float64)
    T_flange_camera[:3, :3] = R_flange_camera
    T_flange_camera[:3, 3] = t_flange_camera

    np.set_printoptions(precision=5, suppress=True)
    print(f"reference frame      = {REFERENCE_FRAME!r}")
    print(f"flange link          = {FLANGE_LINK!r}")
    print(f"rotation_strategy    = {ROTATION_STRATEGY!r}")
    if ROTATION_STRATEGY != "keep_old":
        print(f"  rotation_case      = {ROTATION_CASE!r}")
        print(f"  tilt_x_deg         = {TILT_X_DEG}")
        print(f"  freeform_rpy_deg   = {FREEFORM_RPY_DEG}")
    print()
    print(f"T_flange_link (constant, from URDF FK):")
    print(T_flange_link)
    print(f"  translation (mm) = {T_flange_link[:3, 3] * 1000.0}")
    print()
    print(f"old T_flange_camera (on disk):")
    print(T_old)
    print(f"  translation (mm) = {T_old[:3, 3] * 1000.0}")
    print()
    print(f"new T_flange_camera (saved):")
    print(T_flange_camera)
    print(f"  translation (mm) = {T_flange_camera[:3, 3] * 1000.0}")
    print()

    delta_t_mm = (T_flange_camera[:3, 3] - T_old[:3, 3]) * 1000.0
    delta_R = T_old[:3, :3].T @ T_flange_camera[:3, :3]
    angle_deg = float(
        np.degrees(np.arccos(np.clip((np.trace(delta_R) - 1.0) / 2.0, -1.0, 1.0)))
    )
    print(
        f"delta translation (mm) = {delta_t_mm}  | "
        f"delta rotation = {angle_deg:.3f} deg"
    )

    save_hand_eye_calib(T_flange_camera)
    print("\nwrote hand_eye_calib.npz")


if __name__ == "__main__":
    main()
