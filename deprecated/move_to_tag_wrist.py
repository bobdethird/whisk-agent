"""Spacebar-latched IK move to an AprilTag center using the WRIST camera.

This is the wrist-camera analog of `move_to_tag.py`. Instead of an
overhead iPhone whose pose in the base frame is fixed and solved once at
startup, the wrist camera moves with the arm, so we recompute its pose
in the base frame every frame from FK + hand-eye:

    T_base_camera = T_base_flange @ T_flange_camera   (FK + hand_eye_calib.npz)

Combined with PnP on the tag,

    T_base_tag    = T_base_camera @ T_camera_tag      (per-frame)

On SPACE the most recent T_base_tag is latched, lerobot's
`RobotKinematics.inverse_kinematics` solves safe z-ceiling waypoints that
put an offset-correct gripper frame above the tag while aligning the tool
axis with the tag normal, and the follower streams those waypoints.

One-shot per SPACE press: the target is frozen at the moment the key is
pressed, so moving the tag (or the arm) afterwards does NOT change the
in-flight command.

Controls:
    SPACE : latch the current target-tag pose, solve IK, send action
    q/ESC : quit

Usage:
    python move_to_tag_wrist.py
    python move_to_tag_wrist.py --tag-id 4
    python move_to_tag_wrist.py --hover-z-mm 80
    python move_to_tag_wrist.py --dry-run        # perception + IK, no send_action
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np
from lerobot.model.kinematics import RobotKinematics
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

from camera_utils import disable_autofocus
from detector import (
    TAG_SIZE_M,
    detect_tags,
    make_detector,
    origin_center_error_px,
    render_overlay,
)
from hand_eye_calib import load_hand_eye_calib
from move_to_tag import (
    FK_JOINT_NAMES,
    GRIPPER_OBS_KEY,
    HOVER_Z_M,
    IK_POSITION_WEIGHT,
    MAX_IK_RESIDUAL_MM,
    ROBOT_ID,
    ROBOT_PORT,
    TAG_STALE_S,
    TARGET_FRAME,
    URDF_PATH,
    WORKSPACE_MAX,
    WORKSPACE_MIN,
    _format_xyz_mm,
    _read_joints,
)
from so101_tag_motion import (
    DEFAULT_MOVE_DURATION_S,
    DEFAULT_MOVE_RATE_HZ,
    DEFAULT_ORIENTATION_WEIGHT,
    DEFAULT_POSITION_PRIORITY_MM,
    DEFAULT_SAFE_Z_M,
    MotionConfig,
    solve_and_execute_tag_waypoints,
)

# ---------------------------------------------------------------------------
# Wrist-camera config (mirrors wrist_extrinsic.py)
# ---------------------------------------------------------------------------
WRIST_CAMERA_INDEX = 0
WRIST_CALIB_FILE = "wrist_camera_calib.npz"
CAMERA_BUFFER_SIZE = 1

TARGET_TAG_ID = 4

PREVIEW_WINDOW = "move_to_tag_wrist"

# The shared overhead-camera script uses orientation_weight=0.0 so IK is free
# to choose any wrist roll that reaches the point. For the wrist camera flow we
# want the gripper frame's tool axis to follow the AprilTag blue (+Z) axis.
DEFAULT_WRIST_ORIENTATION_WEIGHT = DEFAULT_ORIENTATION_WEIGHT


def _open_wrist_camera(index: int, expected_size: tuple[int, int]) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise SystemExit(f"failed to open wrist camera at index {index}")
    disable_autofocus(cap, label="wrist")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, CAMERA_BUFFER_SIZE)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, expected_size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, expected_size[1])
    ok, frame = cap.read()
    if not ok:
        raise SystemExit(f"wrist camera at index {index} returned no frame")
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    h, w = frame.shape[:2]
    if (w, h) != expected_size:
        raise SystemExit(
            f"wrist camera delivered {(w, h)} but calibration is for "
            f"{expected_size}"
        )
    return cap


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run the full perception + IK pipeline but skip send_action",
    )
    parser.add_argument(
        "--tag-id",
        type=int,
        default=TARGET_TAG_ID,
        help="id of the AprilTag to move toward",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=WRIST_CAMERA_INDEX,
        help="OpenCV index for the wrist camera",
    )
    parser.add_argument(
        "--hover-z-mm",
        type=float,
        default=HOVER_Z_M * 1000.0,
        help="vertical offset above the tag/contact point (mm)",
    )
    parser.add_argument(
        "--safe-z-mm",
        type=float,
        default=DEFAULT_SAFE_Z_M * 1000.0,
        help="z-up ceiling height (mm) for lift/translate/descent waypoints",
    )
    parser.add_argument(
        "--max-residual-mm",
        type=float,
        default=MAX_IK_RESIDUAL_MM,
        help=(
            "IK residual warning threshold. In default best-effort mode the "
            "closest finite IK solution is still used; with --strict-residual "
            "the move is aborted above this threshold."
        ),
    )
    parser.add_argument(
        "--position-priority-mm",
        type=float,
        default=DEFAULT_POSITION_PRIORITY_MM,
        help=(
            "retry a waypoint with orientation_weight=0 if the first IK solve "
            "misses position by more than this many mm"
        ),
    )
    parser.add_argument(
        "--move-duration-s",
        type=float,
        default=DEFAULT_MOVE_DURATION_S,
        help=(
            "total wall-clock seconds to ramp from the current pose to the "
            "IK solution via smoothstep interpolation. Larger = slower / "
            "smoother. Set to 0 to fall back to a single send_action() shot."
        ),
    )
    parser.add_argument(
        "--move-rate-hz",
        type=float,
        default=DEFAULT_MOVE_RATE_HZ,
        help=(
            "rate at which intermediate Goal_Positions are streamed to the "
            "motors during the move. 30 Hz mirrors lerobot's record/replay "
            "cadence and is plenty smooth for the SO-101."
        ),
    )
    parser.add_argument(
        "--orientation-weight",
        type=float,
        default=DEFAULT_WRIST_ORIENTATION_WEIGHT,
        help=(
            "IK weight for aligning gripper_frame_link +Z with the tag blue "
            "(+Z) axis. Lower favors exact position; higher favors alignment."
        ),
    )
    parser.add_argument(
        "--flip-tag-z",
        action="store_true",
        help=(
            "align gripper_frame_link +Z with -tag Z instead of +tag Z. Use "
            "this if the claw is parallel but points the opposite useful way."
        ),
    )
    parser.add_argument(
        "--tool-offset-mm",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("X", "Y", "Z"),
        help=(
            "offset from gripper_frame_link origin to the desired contact/aim "
            "point, expressed in gripper_frame_link coordinates (mm)"
        ),
    )
    parser.add_argument(
        "--workspace-min-mm",
        type=float,
        nargs=3,
        default=tuple(WORKSPACE_MIN * 1000.0),
        metavar=("X", "Y", "Z"),
        help="minimum allowed IK-frame xyz target in base frame (mm)",
    )
    parser.add_argument(
        "--workspace-max-mm",
        type=float,
        nargs=3,
        default=tuple(WORKSPACE_MAX * 1000.0),
        metavar=("X", "Y", "Z"),
        help="maximum allowed IK-frame xyz target in base frame (mm)",
    )
    parser.add_argument(
        "--strict-residual",
        action="store_true",
        help="abort instead of moving to the closest IK solution above max residual",
    )
    parser.add_argument(
        "--hardcoded-forward-mm",
        type=float,
        default=None,
        help=(
            "debug mode: ignore AprilTags and, on SPACE, synthesize a target "
            "this many mm in front of the current pose"
        ),
    )
    parser.add_argument(
        "--hardcoded-forward-frame",
        choices=("base-x", "gripper-z", "camera-z"),
        default="camera-z",
        help=(
            "frame used by --hardcoded-forward-mm: base +X, current "
            "gripper_frame_link +Z, or current wrist camera +Z"
        ),
    )
    parser.add_argument(
        "--hardcoded-base-mm",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help=(
            "debug mode: ignore AprilTags and, on SPACE, synthesize an "
            "absolute base-frame target at this xyz position in mm"
        ),
    )
    args = parser.parse_args()
    if args.hardcoded_forward_mm is not None and args.hardcoded_base_mm is not None:
        raise SystemExit(
            "choose only one of --hardcoded-forward-mm or --hardcoded-base-mm"
        )

    # --- intrinsics + hand-eye -----------------------------------------------
    calib = np.load(WRIST_CALIB_FILE)
    K = calib["K"].astype(np.float64)
    dist = calib["dist"].astype(np.float64)
    size = (int(calib["image_size"][0]), int(calib["image_size"][1]))
    hover_z_m = float(args.hover_z_mm) / 1000.0
    safe_z_m = float(args.safe_z_mm) / 1000.0
    tool_offset_m = np.asarray(args.tool_offset_mm, dtype=np.float64) / 1000.0
    workspace_min = np.asarray(args.workspace_min_mm, dtype=np.float64) / 1000.0
    workspace_max = np.asarray(args.workspace_max_mm, dtype=np.float64) / 1000.0

    try:
        T_flange_camera = load_hand_eye_calib()
    except FileNotFoundError:
        raise SystemExit(
            "hand_eye_calib.npz not found -- run calibrate_hand_eye.py first"
        )
    print(
        f"[move-wrist] loaded hand_eye_calib.npz: "
        f"T_flange_camera.t = {_format_xyz_mm(T_flange_camera[:3, 3])}"
    )

    # --- robot + IK ----------------------------------------------------------
    kinematics = RobotKinematics(
        urdf_path=URDF_PATH,
        target_frame_name=TARGET_FRAME,
        joint_names=FK_JOINT_NAMES,
    )
    robot = SO101Follower(SO101FollowerConfig(id=ROBOT_ID, port=ROBOT_PORT))
    robot.connect()
    print(f"[move-wrist] robot connected on {ROBOT_PORT}")
    if args.dry_run:
        print("[move-wrist] --dry-run: send_action will be SKIPPED")

    # --- camera + detector ---------------------------------------------------
    detector = make_detector()
    cap = _open_wrist_camera(args.camera_index, size)
    print(f"[move-wrist] opened wrist camera idx={args.camera_index} size={size}")
    print(
        f"[move-wrist] calibration={WRIST_CALIB_FILE} "
        f"tag_size={TAG_SIZE_M * 1000:.1f} mm"
    )

    cv2.namedWindow(PREVIEW_WINDOW, cv2.WINDOW_AUTOSIZE)

    print(
        f"[move-wrist] target tag id={args.tag_id}   "
        f"hover = {hover_z_m * 1000:+.1f} mm above tag   "
        f"safe_z = {safe_z_m * 1000:+.1f} mm   "
        f"[SPACE]=move  [q]/[ESC]=quit"
    )
    if args.hardcoded_forward_mm is not None:
        print(
            f"[move-wrist] hardcoded target mode: SPACE ignores AprilTags and "
            f"uses {args.hardcoded_forward_mm:+.1f} mm along "
            f"{args.hardcoded_forward_frame}"
        )
    if args.hardcoded_base_mm is not None:
        print(
            f"[move-wrist] hardcoded absolute target mode: SPACE ignores "
            f"AprilTags and uses base xyz "
            f"{_format_xyz_mm(np.asarray(args.hardcoded_base_mm) / 1000.0)}"
        )

    latest_T_base_tag: np.ndarray | None = None
    latest_tag_time: float = 0.0
    latest_origin_center_err_px: float | None = None
    latest_joint_deg: np.ndarray | None = None
    latest_gripper: float = 0.0
    last_joint_err_t: float = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            now = time.monotonic()

            # --- live FK -> T_base_camera (camera moves with the arm) -------
            T_base_camera: np.ndarray | None = None
            try:
                joint_deg, gripper = _read_joints(robot)
                latest_joint_deg = joint_deg
                latest_gripper = gripper
                T_base_flange = np.asarray(
                    kinematics.forward_kinematics(joint_deg), dtype=np.float64
                )
                T_base_camera = T_base_flange @ T_flange_camera
            except Exception as e:
                if now - last_joint_err_t > 1.0:
                    print(f"[move-wrist] joint/FK read failed: {e}")
                    last_joint_err_t = now

            # --- tag detection + base-frame composition ---------------------
            detections = detect_tags(frame, detector, K, dist, TAG_SIZE_M)
            target_hit = next(
                (d for d in detections if d.id == args.tag_id), None
            )
            if target_hit is not None and T_base_camera is not None:
                latest_T_base_tag = T_base_camera @ target_hit.T_camera_tag
                latest_tag_time = now
                latest_origin_center_err_px = origin_center_error_px(
                    target_hit,
                    K,
                    dist,
                )

            # --- preview overlay --------------------------------------------
            render_overlay(frame, detections, K, dist, TAG_SIZE_M)

            fresh = (
                latest_T_base_tag is not None
                and (now - latest_tag_time) <= TAG_STALE_S
            )
            if latest_T_base_tag is None:
                status = f"tag {args.tag_id}: never seen"
                status_color = (0, 0, 255)
            else:
                age_ms = int((now - latest_tag_time) * 1000)
                state = "FRESH" if fresh else "STALE"
                status = (
                    f"tag {args.tag_id} base="
                    f"{_format_xyz_mm(latest_T_base_tag[:3, 3])} "
                    f"[{state} {age_ms}ms]"
                )
                status_color = (0, 255, 0) if fresh else (0, 0, 255)
            cv2.putText(
                frame, status, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2,
            )
            cv2.imshow(PREVIEW_WINDOW, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key != ord(" "):
                continue

            if latest_joint_deg is None:
                print("[move-wrist] no joint observation yet; refusing to move")
                continue

            T_base_flange_current = np.asarray(
                kinematics.forward_kinematics(latest_joint_deg),
                dtype=np.float64,
            )
            hardcoded_target = (
                args.hardcoded_forward_mm is not None
                or args.hardcoded_base_mm is not None
            )
            if not hardcoded_target:
                # --- SPACE: latch + IK + (optional) send_action -------------
                if not fresh or latest_T_base_tag is None:
                    print(
                        f"[move-wrist] no fresh sighting of tag {args.tag_id}; "
                        f"refusing to move"
                    )
                    continue

                T_base_tag_latched = latest_T_base_tag.copy()
                latched_age_ms = int((now - latest_tag_time) * 1000)
                print(
                    f"[move-wrist] SPACE: latched tag {args.tag_id} @ "
                    f"{_format_xyz_mm(T_base_tag_latched[:3, 3])} "
                    f"(age {latched_age_ms} ms)"
                )
            elif args.hardcoded_base_mm is not None:
                T_base_tag_latched = T_base_flange_current.copy()
                T_base_tag_latched[:3, 3] = (
                    np.asarray(args.hardcoded_base_mm, dtype=np.float64) / 1000.0
                )
                print(
                    f"[move-wrist] SPACE: hardcoded absolute base target @ "
                    f"{_format_xyz_mm(T_base_tag_latched[:3, 3])}"
                )
                print(
                    "[move-wrist] note: --hover-z-mm is still applied after "
                    "this synthetic target; use --hover-z-mm 0 for exact point"
                )
            else:
                T_base_camera_current = T_base_flange_current @ T_flange_camera
                if args.hardcoded_forward_frame == "base-x":
                    forward = np.array([1.0, 0.0, 0.0], dtype=np.float64)
                elif args.hardcoded_forward_frame == "gripper-z":
                    forward = T_base_flange_current[:3, 2]
                else:
                    forward = T_base_camera_current[:3, 2]
                forward = forward / max(float(np.linalg.norm(forward)), 1e-9)

                T_base_tag_latched = T_base_flange_current.copy()
                T_base_tag_latched[:3, 3] = (
                    T_base_flange_current[:3, 3]
                    + forward * (float(args.hardcoded_forward_mm) / 1000.0)
                )
                print(
                    f"[move-wrist] SPACE: hardcoded target @ "
                    f"{_format_xyz_mm(T_base_tag_latched[:3, 3])} "
                    f"({args.hardcoded_forward_mm:+.1f} mm along "
                    f"{args.hardcoded_forward_frame})"
                )
                print(
                    "[move-wrist] note: --hover-z-mm is still applied after "
                    "this synthetic target; use --hover-z-mm 0 for exact point"
                )

            if (
                not hardcoded_target
                and latest_origin_center_err_px is not None
            ):
                print(
                    f"[move-wrist] PnP origin vs detector center at latch: "
                    f"{latest_origin_center_err_px:.1f} px"
                )

            config = MotionConfig(
                fk_joint_names=FK_JOINT_NAMES,
                gripper_obs_key=GRIPPER_OBS_KEY,
                workspace_min=workspace_min,
                workspace_max=workspace_max,
                position_weight=IK_POSITION_WEIGHT,
                orientation_weight=float(args.orientation_weight),
                max_residual_mm=float(args.max_residual_mm),
                move_duration_s=float(args.move_duration_s),
                move_rate_hz=float(args.move_rate_hz),
                fallback_residual_mm=float(args.position_priority_mm),
                safe_z_m=safe_z_m,
                best_effort=not bool(args.strict_residual),
                dry_run=bool(args.dry_run),
                label="move-wrist",
            )
            solve_and_execute_tag_waypoints(
                kinematics=kinematics,
                robot=robot,
                current_joints_deg=latest_joint_deg,
                gripper=latest_gripper,
                T_base_tag=T_base_tag_latched,
                hover_z_m=hover_z_m,
                tag_z_sign=-1.0 if args.flip_tag_z else 1.0,
                tool_offset_m=tool_offset_m,
                config=config,
                keepalive=lambda: cv2.waitKey(1),
            )
    finally:
        try:
            cap.release()
        except Exception:
            pass
        cv2.destroyAllWindows()
        try:
            robot.disconnect()
        except Exception as e:
            print(f"[move-wrist] disconnect warning: {e}")


if __name__ == "__main__":
    main()
