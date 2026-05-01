"""Live AprilTag detection -> press SPACE -> move SO101 claw tip to the tag.

Pipeline per frame:
    cv2.VideoCapture        -> BGR frame
    cv2.rotate(ROTATE_180)  -> wrist camera is mounted upside-down
    apriltag.detect         -> per-tag image corners
    cv2.solvePnPGeneric     -> tag pose in the *camera* frame
    T_BASE_CAMERA           -> tag pose in the *base* frame
    HUD overlay             -> live X, Y, Z in both frames

When the user presses SPACE, the camera is closed and `move_arm` from
move_arm.py is called with (x, y, z) of the target tag's center plus
HOVER_Z_M, so the claw tip lands above the tag instead of crashing into it.

T_BASE_CAMERA is the rigid transform that maps a 3D point in the camera
frame into the base_link frame (p_base = R @ p_camera + t). The wrist
camera moves with the arm, so this transform depends on the joint angles
through the FK chain:

    T_base_camera(q) = T_base_flange(q) @ T_flange_camera

T_flange_camera is constant (it's the hand-eye calibration). Its
translation is given in `gripper_link` coordinates and the rotation is a
pitch about the flange's X axis. The constants below were transcribed from
deprecated/set_hand_eye.py.

Because the camera moves with the arm, this script reads the joints once
at startup, snapshots T_base_camera at the current pose, and assumes the
arm holds still during detection. Pressing SPACE then moves the arm using
that snapshot.

Coordinate frames:
    Camera frame  : OpenCV / AprilTag convention.
                        +X right, +Y down, +Z forward (out the lens)
    Base frame    : SO101 URDF base_link.
                        +X forward, +Y left, +Z up

Controls:
    SPACE   capture target tag's base-frame XYZ, close camera, run move_arm
    q/ESC   quit without moving
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from apriltag import apriltag

from lerobot.model.kinematics import RobotKinematics
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

from move_arm import move_arm


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CAMERA_INDEX = 0
CALIB_FILE = Path(__file__).parent / "camera_calib.npz"
TAG_FAMILY = "tagStandard41h12"
TAG_SIZE_M = 0.027

TARGET_TAG_ID = 0

HOVER_Z_M = 0.05
MOVE_DURATION_S = 2.0
MOVE_HZ = 50.0

# Robot / URDF setup. Mirrors move_arm.py.
URDF_PATH = Path(__file__).parent / "SO101" / "so101_new_calib.urdf"
PORT = "/dev/tty.usbmodem5AE60557941"
ROBOT_ID = "follower-1"
ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
MOTOR_NAMES = ARM_JOINTS + ["gripper"]
FLANGE_LINK = "gripper_frame_link"

# Wrist-camera mount, transcribed from deprecated/set_hand_eye.py.
# Translation is in `gripper_link` coordinates (a fixed-joint child of the
# flange); rotation is a pure pitch about the flange's X axis.
CAMERA_REF_FRAME = "gripper_link"
CAMERA_TX_MM = 7.7
CAMERA_TY_MM = 100.1
CAMERA_TZ_MM = -23.4
CAMERA_TILT_X_DEG = 19.0


# ---------------------------------------------------------------------------
# Camera-to-base extrinsic (computed at startup from the current arm pose)
# ---------------------------------------------------------------------------
def _Rx(rad: float) -> np.ndarray:
    """3x3 rotation matrix about +X by `rad` radians."""
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _read_motors_deg(robot: SO101Follower) -> np.ndarray:
    obs = robot.get_observation()
    return np.array([float(obs[f"{m}.pos"]) for m in MOTOR_NAMES])


def compute_T_base_camera_at_current_pose() -> np.ndarray:
    """Snapshot T_base_camera using the arm's *current* joint pose.

    The wrist camera moves with the arm, so this transform only stays
    valid while the arm holds the pose it was in at startup. Don't move
    the arm during AprilTag detection.

    Composition (matches deprecated/set_hand_eye.py + deprecated/wrist_extrinsic.py):
        T_base_camera = T_base_flange(q) @ T_flange_camera
        T_flange_camera.t = T_flange_link[:3,:3] @ (TX, TY, TZ)/1000
                          + T_flange_link[:3, 3]   # translation in gripper_link
        T_flange_camera.R = Rx(CAMERA_TILT_X_DEG)  # pitch in flange frame
    """
    robot = SO101Follower(
        SO101FollowerConfig(port=PORT, id=ROBOT_ID, disable_torque_on_disconnect=False)
    )
    kinematics = RobotKinematics(
        urdf_path=str(URDF_PATH),
        target_frame_name=FLANGE_LINK,
        joint_names=ARM_JOINTS,
    )
    robot.connect()
    try:
        joint_deg = _read_motors_deg(robot)[: len(ARM_JOINTS)]
        T_base_flange = np.asarray(
            kinematics.forward_kinematics(joint_deg), dtype=np.float64
        )
        T_base_gripper_link = np.asarray(
            kinematics.robot.get_T_world_frame(CAMERA_REF_FRAME), dtype=np.float64
        )
    finally:
        robot.disconnect()

    T_flange_link = np.linalg.inv(T_base_flange) @ T_base_gripper_link
    t_link_camera = np.array(
        [CAMERA_TX_MM, CAMERA_TY_MM, CAMERA_TZ_MM], dtype=np.float64
    ) / 1000.0
    t_flange_camera = T_flange_link[:3, :3] @ t_link_camera + T_flange_link[:3, 3]
    R_flange_camera = _Rx(np.deg2rad(CAMERA_TILT_X_DEG))

    T_flange_camera = np.eye(4, dtype=np.float64)
    T_flange_camera[:3, :3] = R_flange_camera
    T_flange_camera[:3, 3] = t_flange_camera
    return T_base_flange @ T_flange_camera


# Computed at startup in main(); identity until then.
T_BASE_CAMERA = np.eye(4, dtype=np.float64)


# ---------------------------------------------------------------------------
# AprilTag detection
# ---------------------------------------------------------------------------
# Axis flip from OpenCV IPPE_SQUARE's tag-local frame (x right, y up,
# z out of tag) to AprilTag's documented frame (x right, y down,
# z into tag). Diagonal -1 on Y and Z; involutory.
R_IPPE_TO_APRILTAG = np.diag([1.0, -1.0, -1.0]).astype(np.float64)

# solvePnPRefineLM termination. EPS well below corner noise; 20 iters is
# plenty after IPPE has seeded near the optimum.
_REFINE_LM_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1e-6)


def _tag_object_points(tag_size_m: float) -> np.ndarray:
    """4 coplanar tag corners in the order SOLVEPNP_IPPE_SQUARE expects."""
    half = tag_size_m / 2.0
    return np.array([
        [-half,  half, 0.0],   # top-left
        [ half,  half, 0.0],   # top-right
        [ half, -half, 0.0],   # bottom-right
        [-half, -half, 0.0],   # bottom-left
    ], dtype=np.float64)


def detect_tags(
    frame: np.ndarray,
    detector: apriltag,
    K: np.ndarray,
    dist: np.ndarray,
    tag_size_m: float = TAG_SIZE_M,
) -> list[dict]:
    """Run AprilTag detection + IPPE PnP on a BGR frame.

    Returns a list of dicts: id, corners (4x2 px), center (cx, cy), rvec, tvec,
    T_camera_tag (4x4). Tags whose PnP fails are silently dropped.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    obj = _tag_object_points(tag_size_m)

    out: list[dict] = []
    for raw in detector.detect(gray):
        # apriltag returns (lb, rb, rt, lt); IPPE wants (lt, rt, rb, lb)
        # so we reverse.
        corners = np.asarray(raw["lb-rb-rt-lt"], dtype=np.float64)
        corners_ippe = corners[[3, 2, 1, 0]]

        retval, rvecs, tvecs, errs = cv2.solvePnPGeneric(
            obj, corners_ippe, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        if retval < 1 or rvecs is None or len(rvecs) == 0:
            continue

        idx = 0
        if len(rvecs) == 2 and errs[1].item() < errs[0].item():
            idx = 1
        rvec, tvec = rvecs[idx].copy(), tvecs[idx].copy()

        rvec, tvec = cv2.solvePnPRefineLM(
            obj, corners_ippe, K, dist, rvec, tvec, _REFINE_LM_CRITERIA
        )

        R_ippe, _ = cv2.Rodrigues(rvec)
        R_at = R_ippe @ R_IPPE_TO_APRILTAG
        T_camera_tag = np.eye(4, dtype=np.float64)
        T_camera_tag[:3, :3] = R_at
        T_camera_tag[:3, 3] = tvec.ravel()
        rvec_at, _ = cv2.Rodrigues(R_at)

        out.append({
            "id": int(raw["id"]),
            "corners": corners,
            "center": (float(raw["center"][0]), float(raw["center"][1])),
            "rvec": rvec_at,
            "tvec": tvec.reshape(3, 1),
            "T_camera_tag": T_camera_tag,
        })
    return out


def transform_camera_to_base(xyz_camera: np.ndarray) -> np.ndarray:
    """Apply T_BASE_CAMERA to a 3-vector expressed in the camera frame."""
    p = np.asarray(xyz_camera, dtype=np.float64).reshape(3)
    return T_BASE_CAMERA[:3, :3] @ p + T_BASE_CAMERA[:3, 3]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    global T_BASE_CAMERA

    print("[main] reading robot joints to snapshot T_base_camera...")
    T_BASE_CAMERA = compute_T_base_camera_at_current_pose()
    cam_xyz_mm = T_BASE_CAMERA[:3, 3] * 1000.0
    print(
        f"[main] T_base_camera.t = ({cam_xyz_mm[0]:+7.1f},"
        f" {cam_xyz_mm[1]:+7.1f}, {cam_xyz_mm[2]:+7.1f}) mm"
    )
    print(
        "[main] keep the arm still during detection; T_base_camera is a "
        "single snapshot at this pose."
    )

    calib = np.load(CALIB_FILE)
    K = calib["K"].astype(np.float64)
    dist = calib["dist"].astype(np.float64)
    calib_size = (int(calib["image_size"][0]), int(calib["image_size"][1]))

    detector = apriltag(TAG_FAMILY, decimate=1.0, threads=4, refine_edges=True)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise SystemExit(f"failed to open camera at index {CAMERA_INDEX}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, calib_size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, calib_size[1])

    ok, frame = cap.read()
    if not ok:
        cap.release()
        raise SystemExit(f"camera idx={CAMERA_INDEX} returned no frame")
    # Camera is mounted upside-down on the wrist (matches deprecated/
    # wrist_extrinsic.py); rotate every frame so the calibration and the
    # detector see the image in the same orientation it was captured for.
    frame = cv2.rotate(frame, cv2.ROTATE_180)

    h, w = frame.shape[:2]
    if (w, h) != calib_size:
        cap.release()
        raise SystemExit(
            f"camera delivered {(w, h)} but calibration is for {calib_size}; "
            "recapture camera_calib.npz at the runtime resolution or pick a "
            "camera that delivers the calibration resolution."
        )

    print(f"[main] camera idx={CAMERA_INDEX} @ {calib_size} (rotated 180)")
    print(f"[main] target tag id={TARGET_TAG_ID}, hover {HOVER_Z_M*1000:.0f} mm above")
    print("[main] [SPACE] move arm to target tag   [q]/[ESC] quit")

    target_xyz_base: np.ndarray | None = None
    window_name = "apriltag -> move_arm"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.rotate(frame, cv2.ROTATE_180)

        detections = detect_tags(frame, detector, K, dist, TAG_SIZE_M)
        target = next((d for d in detections if d["id"] == TARGET_TAG_ID), None)

        for d in detections:
            color = (0, 255, 0) if d["id"] == TARGET_TAG_ID else (180, 180, 180)
            pts = d["corners"].astype(np.int32)
            cv2.polylines(frame, [pts], True, color, 2)
            cv2.drawFrameAxes(frame, K, dist, d["rvec"], d["tvec"], TAG_SIZE_M * 0.5, 2)

            xyz_base = transform_camera_to_base(d["tvec"].ravel())
            cx, cy = int(d["center"][0]), int(d["center"][1])
            label = (
                f"id={d['id']}  "
                f"({xyz_base[0]:+.3f}, {xyz_base[1]:+.3f}, {xyz_base[2]:+.3f}) m"
            )
            cv2.putText(
                frame, label, (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
            )

        if target is not None:
            xyz_cam = target["tvec"].ravel()
            xyz_base = transform_camera_to_base(xyz_cam)
            status = (
                f"tag {TARGET_TAG_ID} VISIBLE   [SPACE] move arm"
            )
            cam_line = (
                f"camera frame: ({xyz_cam[0]:+.3f}, {xyz_cam[1]:+.3f},"
                f" {xyz_cam[2]:+.3f}) m"
            )
            base_line = (
                f"base   frame: ({xyz_base[0]:+.3f}, {xyz_base[1]:+.3f},"
                f" {xyz_base[2]:+.3f}) m"
            )
            status_color = (0, 255, 0)
        else:
            status = (
                f"tag {TARGET_TAG_ID} NOT visible "
                f"({len(detections)} other tag(s) seen)"
            )
            cam_line = ""
            base_line = ""
            status_color = (0, 0, 255)
        cv2.putText(
            frame, status, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2,
        )
        if cam_line:
            cv2.putText(
                frame, cam_line, (10, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 2,
            )
            cv2.putText(
                frame, base_line, (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2,
            )

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            if target is None:
                print(f"[main] target tag {TARGET_TAG_ID} not visible; ignoring SPACE")
                continue
            target_xyz_base = transform_camera_to_base(target["tvec"].ravel()).copy()
            break

    cap.release()
    cv2.destroyAllWindows()

    if target_xyz_base is None:
        print("[main] quit before SPACE; not moving")
        return

    target_x = float(target_xyz_base[0])
    target_y = float(target_xyz_base[1])
    target_z = float(target_xyz_base[2]) + HOVER_Z_M
    print(
        f"[main] moving claw tip to "
        f"({target_x:+.3f}, {target_y:+.3f}, {target_z:+.3f}) m  "
        f"(tag center + {HOVER_Z_M*1000:.0f} mm hover, vertical)"
    )
    move_arm(
        x=target_x,
        y=target_y,
        z=target_z,
        vertical=True,
        duration=MOVE_DURATION_S,
        hz=MOVE_HZ,
    )
    print("[main] move complete")


if __name__ == "__main__":
    main()
