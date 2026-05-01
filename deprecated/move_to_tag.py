"""Spacebar-latched IK move to an AprilTag center.

Runs the overhead iPhone AprilTag detection, maintains a live base-frame
pose for a target tag, and on SPACE commands the SO-101 follower through
safe z-ceiling waypoints to an offset-correct hover/contact pose.

One-shot per SPACE press: the target pose is frozen at the moment the
key is pressed, so moving the tag afterwards does NOT change the
in-flight command.

Controls:
    SPACE : latch the current target-tag pose, solve IK, send action
    q/ESC : quit

Usage:
    python move_to_tag.py
    python move_to_tag.py --dry-run       # perception + IK, no send_action
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from camera_utils import disable_autofocus
from detector import (
    TAG_SIZE_M,
    detect_tags,
    make_detector,
    origin_center_error_px,
    render_overlay,
)
from iphone_extrinsic import ReferenceTagNotFound, solve_iphone_extrinsic_avg
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
# Config (edit to match your setup)
# ---------------------------------------------------------------------------
IPHONE_CAMERA_INDEX = 3
IPHONE_CALIB_FILE = "iphone_ultrawide_camera_calib.npz"
CAMERA_BUFFER_SIZE = 1

ROBOT_PORT = "/dev/tty.usbmodem5AE60557941"
ROBOT_ID = "my_awesome_follower_arm"

URDF_PATH = "SO101/so101_new_calib.urdf"
TARGET_FRAME = "gripper_frame_link"

# FK joints that affect T_base_flange. Gripper is excluded from FK because
# gripper_frame_link is attached via a fixed joint before the revolute
# 'gripper' joint, and the gripper observation is normalized 0..100 rather
# than degrees.
FK_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]
FK_JOINT_OBS_KEYS = [f"{n}.pos" for n in FK_JOINT_NAMES]
GRIPPER_OBS_KEY = "gripper.pos"

# Reference AprilTag: stationary, known pose in the SO-101 base frame
# (measured with a ruler/tape). Must NOT move during the session; used at
# startup to solve T_base_iphone via iphone_extrinsic.
REFERENCE_TAG_ID = 4
REFERENCE_TAG_X_MM = 0.0
REFERENCE_TAG_Y_MM = 200.0
REFERENCE_TAG_Z_MM = 0.0
# Reference tag orientation in the base frame. Default assumes the tag is
# laid flat on the tabletop with +x_tag -> +x_base and +y_tag -> +y_base
# (so +z_tag points up out of the table). Edit if the real tag is rotated.
R_BASE_TAG_REF = np.eye(3, dtype=np.float64)

# Target AprilTag: the one the robot will hover above when SPACE is pressed.
TARGET_TAG_ID = 4

# Hover offset above the target tag center, along +Z in the base frame.
HOVER_Z_M = 0.05

# IK weight. Orientation is controlled by the shared waypoint executor.
IK_POSITION_WEIGHT = 1.0

# Pose freshness: refuse to command a move if the latest detection of the
# target tag is older than this at the moment SPACE is pressed (seconds).
TAG_STALE_S = 0.5

# Largest allowed IK-reconstruction position residual (mm). If the solver
# cannot bring the flange within this of the desired hover point, abort
# instead of sending the arm somewhere unexpected.
MAX_IK_RESIDUAL_MM = 15.0

# Startup extrinsic solve.
EXTRINSIC_N_FRAMES = 15
EXTRINSIC_TIMEOUT_S = 20.0

# Name of the shared live-camera preview window. Used both during the
# startup extrinsic solve and in the main loop so the view is continuous.
PREVIEW_WINDOW = "move_to_tag"

# Workspace bounds (base frame, metres). These are guardrails for tag targets,
# not motor calibration limits. They should be large enough to cover the URDF
# reach envelope while keeping obviously unsafe below-table requests out.
WORKSPACE_MIN = np.array([-0.40, -0.45, 0.00], dtype=np.float64)
WORKSPACE_MAX = np.array([0.48, 0.45, 0.55], dtype=np.float64)


# ---------------------------------------------------------------------------
# LeRobot import probing (mirrors calibrate_hand_eye.py)
# ---------------------------------------------------------------------------
def _probe_robot_kinematics():
    errors: list[str] = []
    for mod_path in (
        "lerobot.model.kinematics",
        "lerobot.kinematics",
        "lerobot.common.model.kinematics",
        "lerobot.common.kinematics",
    ):
        try:
            mod = __import__(mod_path, fromlist=["RobotKinematics"])
            return mod.RobotKinematics
        except ImportError as e:
            errors.append(f"    {mod_path}: {e}")
    raise ImportError(
        "could not locate RobotKinematics in any expected module.\n"
        + "\n".join(errors)
    )


def _probe_so101_follower():
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    return SO101Follower, SO101FollowerConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_reference_tag_pose() -> np.ndarray:
    """Assemble T_base_tag_ref from the ruler survey constants above."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_BASE_TAG_REF
    T[:3, 3] = np.array([
        REFERENCE_TAG_X_MM,
        REFERENCE_TAG_Y_MM,
        REFERENCE_TAG_Z_MM,
    ], dtype=np.float64) / 1000.0
    return T


def _open_camera(index: int, expected_size: tuple[int, int]) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise SystemExit(f"failed to open iPhone camera at index {index}")
    disable_autofocus(cap, label="iphone")

    cap.set(cv2.CAP_PROP_BUFFERSIZE, CAMERA_BUFFER_SIZE)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, expected_size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, expected_size[1])

    ok, frame = cap.read()
    if not ok:
        raise SystemExit(f"camera index {index} opened but returned no frame")

    h, w = frame.shape[:2]
    if (w, h) != expected_size:
        raise SystemExit(
            f"camera at index {index} delivered {(w, h)} but calibration is "
            f"for {expected_size}"
        )
    return cap


def _read_joints(robot) -> tuple[np.ndarray, float]:
    """Read current FK joints (deg) + gripper pos (0..100) from the follower."""
    obs = robot.get_observation()
    joint_deg = np.asarray(
        [float(obs[k]) for k in FK_JOINT_OBS_KEYS], dtype=np.float64
    )
    gripper = float(obs.get(GRIPPER_OBS_KEY, 0.0))
    return joint_deg, gripper


def _format_xyz_mm(xyz: np.ndarray) -> str:
    mm = np.asarray(xyz, dtype=np.float64).ravel() * 1000.0
    return f"({mm[0]:+7.1f}, {mm[1]:+7.1f}, {mm[2]:+7.1f}) mm"


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
        help="seconds to stream each waypoint segment",
    )
    parser.add_argument(
        "--move-rate-hz",
        type=float,
        default=DEFAULT_MOVE_RATE_HZ,
        help="rate for streamed intermediate Goal_Position commands",
    )
    parser.add_argument(
        "--orientation-weight",
        type=float,
        default=DEFAULT_ORIENTATION_WEIGHT,
        help="IK weight for keeping gripper_frame_link +Z parallel to tag Z",
    )
    parser.add_argument(
        "--flip-tag-z",
        action="store_true",
        help="align gripper_frame_link +Z with -tag Z instead of +tag Z",
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
    args = parser.parse_args()

    iphone_calib = np.load(IPHONE_CALIB_FILE)
    K = iphone_calib["K"].astype(np.float64)
    dist = iphone_calib["dist"].astype(np.float64)
    iphone_size = (
        int(iphone_calib["image_size"][0]),
        int(iphone_calib["image_size"][1]),
    )
    hover_z_m = float(args.hover_z_mm) / 1000.0
    safe_z_m = float(args.safe_z_mm) / 1000.0
    tool_offset_m = np.asarray(args.tool_offset_mm, dtype=np.float64) / 1000.0
    workspace_min = np.asarray(args.workspace_min_mm, dtype=np.float64) / 1000.0
    workspace_max = np.asarray(args.workspace_max_mm, dtype=np.float64) / 1000.0

    RobotKinematics = _probe_robot_kinematics()
    kinematics = RobotKinematics(
        urdf_path=URDF_PATH,
        target_frame_name=TARGET_FRAME,
        joint_names=FK_JOINT_NAMES,
    )

    SO101Follower, SO101FollowerConfig = _probe_so101_follower()
    robot = SO101Follower(SO101FollowerConfig(id=ROBOT_ID, port=ROBOT_PORT))
    robot.connect()
    print(f"[move] robot connected on {ROBOT_PORT}")
    if args.dry_run:
        print("[move] --dry-run: send_action will be SKIPPED")

    detector = make_detector()
    cap = _open_camera(IPHONE_CAMERA_INDEX, iphone_size)
    print(f"[move] opened iPhone camera idx={IPHONE_CAMERA_INDEX}")
    print(
        f"[move] calibration={IPHONE_CALIB_FILE} tag_size={TAG_SIZE_M * 1000:.1f} mm"
    )

    cv2.namedWindow(PREVIEW_WINDOW, cv2.WINDOW_AUTOSIZE)

    try:
        T_base_tag_ref = _build_reference_tag_pose()
        print(
            f"[move] reference tag id={REFERENCE_TAG_ID} @ "
            f"{_format_xyz_mm(T_base_tag_ref[:3, 3])}"
        )
        print(
            f"[move] solving T_base_iphone over {EXTRINSIC_N_FRAMES} frames; "
            f"keep reference tag visible (q/ESC aborts)..."
        )
        try:
            T_base_iphone = solve_iphone_extrinsic_avg(
                cap=cap,
                K=K,
                dist=dist,
                tag_id=REFERENCE_TAG_ID,
                T_base_tag_ref=T_base_tag_ref,
                n_frames=EXTRINSIC_N_FRAMES,
                timeout_s=EXTRINSIC_TIMEOUT_S,
                detector=detector,
                verbose=True,
                preview_window=PREVIEW_WINDOW,
            )
        except ReferenceTagNotFound as e:
            raise SystemExit(f"[move] extrinsic solve failed: {e}")

        print(
            f"[move] T_base_iphone translation = "
            f"{_format_xyz_mm(T_base_iphone[:3, 3])}"
        )
        print(
            f"[move] target tag id={args.tag_id}   "
            f"hover = {hover_z_m * 1000:+.1f} mm   "
            f"safe_z = {safe_z_m * 1000:+.1f} mm   "
            f"[SPACE]=move  [q]/[ESC]=quit"
        )

        latest_T_base_tag: np.ndarray | None = None
        latest_tag_time: float = 0.0
        latest_origin_center_err_px: float | None = None

        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            now = time.monotonic()
            detections = detect_tags(frame, detector, K, dist, TAG_SIZE_M)
            target_hit = next(
                (d for d in detections if d.id == args.tag_id), None
            )
            if target_hit is not None:
                latest_T_base_tag = T_base_iphone @ target_hit.T_camera_tag
                latest_tag_time = now
                latest_origin_center_err_px = origin_center_error_px(
                    target_hit,
                    K,
                    dist,
                )

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

            # --- SPACE: latch pose, compute IK, optionally send action -----
            if not fresh or latest_T_base_tag is None:
                print(
                    f"[move] no fresh sighting of tag {args.tag_id}; "
                    f"refusing to move"
                )
                continue

            T_base_tag_latched = latest_T_base_tag.copy()
            latched_age_ms = int((now - latest_tag_time) * 1000)
            print(
                f"[move] SPACE: latched tag {args.tag_id} @ "
                f"{_format_xyz_mm(T_base_tag_latched[:3, 3])} "
                f"(age {latched_age_ms} ms)"
            )
            if latest_origin_center_err_px is not None:
                print(
                    f"[move] PnP origin vs detector center at latch: "
                    f"{latest_origin_center_err_px:.1f} px"
                )

            try:
                joint_deg, gripper = _read_joints(robot)
            except KeyError as e:
                print(f"[move] joint observation missing key {e}; aborting")
                continue

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
                label="move",
            )
            solve_and_execute_tag_waypoints(
                kinematics=kinematics,
                robot=robot,
                current_joints_deg=joint_deg,
                gripper=gripper,
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
            print(f"[move] disconnect warning: {e}")


if __name__ == "__main__":
    main()
