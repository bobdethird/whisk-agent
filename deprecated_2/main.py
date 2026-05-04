"""Live AprilTag detection -> press SPACE -> simulate/move SO101 claw tip.

Pipeline per frame:
    cv2.VideoCapture        -> BGR frame
    cv2.rotate(ROTATE_180)  -> wrist camera is mounted upside-down
    apriltag.detect         -> per-tag image corners
    cv2.solvePnPGeneric     -> tag pose in the *camera* frame
    T_BASE_CAMERA           -> tag pose in the *base* frame
    HUD overlay             -> live X, Y, Z in both frames

When the user presses SPACE, the camera is closed and `move_arm` from
move_arm_v4.py is called with a base_link target computed from the target
tag pose. The target point is expressed in the tag-local frame first, then
transformed through T_base_tag so tag orientation is respected. By default
this runs as a V4 simulation; pass `--execute-move` to send commands to the
physical robot.

T_BASE_CAMERA is the rigid transform that maps a 3D point in the camera
frame into the base_link frame (p_base = R @ p_camera + t). The wrist
camera moves with the arm, so this transform depends on the joint angles
through the FK chain:

    T_base_camera(q) = T_base_flange(q) @ T_flange_camera

T_flange_camera is constant (it's the hand-eye calibration). Its
translation is given in `gripper_link` coordinates and the rotation is a
pitch about the flange's X axis. The constants below were transcribed from
deprecated/set_hand_eye.py.

Because the camera moves with the arm, this script keeps the robot connected
during detection and recomputes T_base_camera from live encoder readings every
frame. Pressing SPACE takes a fresh frame and fresh encoder snapshot before
running MoveArmV4.

Coordinate frames:
    Camera frame  : OpenCV / AprilTag convention.
                        +X right, +Y down, +Z forward (out the lens)
    Base frame    : SO101 URDF base_link.
                        +X forward, +Y left, +Z up

Controls:
    SPACE   capture target tag-local target in base_link, run MoveArmV4
    q/ESC   quit without moving

For a camera-free V4 simulation, run:
    python main.py --xyz 0.20 0.00 0.15
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from lerobot.model.kinematics import RobotKinematics
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

from move_arm_v4 import ARM_JOINTS as V4_ARM_JOINTS
from move_arm_v4 import MAX_FINAL_RESIDUAL_MM
from move_arm_v4 import move_arm as move_arm_v4


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CAMERA_INDEX = 0
CALIB_FILE = Path(__file__).parent / "camera_calib.npz"
TAG_FAMILY = "tagStandard41h12"
TAG_SIZE_M = 0.027

TARGET_TAG_ID = 3

HOVER_Z_M = 0.0
TAG_LOCAL_OFFSET_M = np.zeros(3, dtype=np.float64)
MOVE_DURATION_S = 2.0
MOVE_HZ = 50.0

# Robot / URDF setup. Mirrors move_arm_v4.py.
URDF_PATH = Path(__file__).parent / "SO101" / "so101_new_calib.urdf"
PORT = "/dev/tty.usbmodem5AE60557941"
ROBOT_ID = "follower-1"
ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
MOTOR_NAMES = ARM_JOINTS + ["gripper"]
FREE_DRIVE_MOTORS = ARM_JOINTS
FLANGE_LINK = "gripper_frame_link"

# Wrist-camera mount, transcribed from deprecated/set_hand_eye.py.
# Translation is in `gripper_link` coordinates (a fixed-joint child of the
# flange); rotation is a pure pitch about the flange's X axis.
CAMERA_REF_FRAME = "gripper_link"
CAMERA_TX_MM = 7.7
CAMERA_TY_MM = 100.1
CAMERA_TZ_MM = -23.4
CAMERA_TILT_X_DEG = 19.0

cv2 = None
apriltag = None


def _load_camera_deps() -> None:
    """Import camera-only dependencies after the --xyz fast path."""
    global cv2, apriltag
    if cv2 is not None and apriltag is not None:
        return
    try:
        import cv2 as cv2_module
        from apriltag import apriltag as apriltag_class
    except ModuleNotFoundError as e:
        raise SystemExit(
            "camera mode requires OpenCV and apriltag. For camera-free V4 "
            "simulation use --xyz, or install the missing dependency: "
            f"{e.name}"
        ) from e
    cv2 = cv2_module
    apriltag = apriltag_class


# ---------------------------------------------------------------------------
# Camera-to-base extrinsic (computed from live encoder readings)
# ---------------------------------------------------------------------------
def _Rx(rad: float) -> np.ndarray:
    """3x3 rotation matrix about +X by `rad` radians."""
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _read_motors_deg(robot: SO101Follower) -> np.ndarray:
    obs = robot.get_observation()
    return np.array([float(obs[f"{m}.pos"]) for m in MOTOR_NAMES])


def _compute_T_base_camera_from_joints(
    kinematics: RobotKinematics,
    joint_deg: np.ndarray,
) -> np.ndarray:
    """Compute T_base_camera from an encoder snapshot.

    Composition (matches deprecated/set_hand_eye.py + deprecated/wrist_extrinsic.py):
        T_base_camera = T_base_flange(q) @ T_flange_camera
        T_flange_camera.t = T_flange_link[:3,:3] @ (TX, TY, TZ)/1000
                          + T_flange_link[:3, 3]   # translation in gripper_link
        T_flange_camera.R = Rx(CAMERA_TILT_X_DEG)  # pitch in flange frame
    """
    arm_joint_deg = np.asarray(joint_deg, dtype=np.float64).reshape(-1)[
        : len(ARM_JOINTS)
    ]
    T_base_flange = np.asarray(
        kinematics.forward_kinematics(arm_joint_deg), dtype=np.float64
    )
    T_base_gripper_link = np.asarray(
        kinematics.robot.get_T_world_frame(CAMERA_REF_FRAME), dtype=np.float64
    )

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


class LiveCameraPoseTracker:
    """Keeps robot IO open while converting encoder snapshots to T_base_camera."""

    def __init__(self, disable_torque: bool = False) -> None:
        self.disable_torque = disable_torque
        self.robot = SO101Follower(
            SO101FollowerConfig(
                port=PORT,
                id=ROBOT_ID,
                disable_torque_on_disconnect=False,
            )
        )
        self.kinematics = RobotKinematics(
            urdf_path=str(URDF_PATH),
            target_frame_name=FLANGE_LINK,
            joint_names=ARM_JOINTS,
        )
        self.connected = False

    def connect(self) -> None:
        self.robot.connect()
        if self.disable_torque:
            self.robot.bus.disable_torque(FREE_DRIVE_MOTORS)
        self.connected = True

    def read(self) -> tuple[np.ndarray, np.ndarray]:
        joint_deg = _read_motors_deg(self.robot)[: len(ARM_JOINTS)]
        return (
            _compute_T_base_camera_from_joints(self.kinematics, joint_deg),
            joint_deg.copy(),
        )

    def disconnect(self) -> None:
        if self.connected:
            self.robot.disconnect()
            self.connected = False


def compute_T_base_camera_at_current_pose() -> tuple[np.ndarray, np.ndarray]:
    """One-shot T_base_camera read for scripts/tests that do not need live tracking."""
    tracker = LiveCameraPoseTracker()
    tracker.connect()
    try:
        return tracker.read()
    finally:
        tracker.disconnect()


# Updated from live encoder readings in main(); identity until then.
T_BASE_CAMERA = np.eye(4, dtype=np.float64)


# ---------------------------------------------------------------------------
# AprilTag detection
# ---------------------------------------------------------------------------
# Axis flip from OpenCV IPPE_SQUARE's tag-local frame (x right, y up,
# z out of tag) to AprilTag's documented frame (x right, y down,
# z into tag). Diagonal -1 on Y and Z; involutory.
R_IPPE_TO_APRILTAG = np.diag([1.0, -1.0, -1.0]).astype(np.float64)

_REFINE_LM_MAX_ITER = 20
_REFINE_LM_EPS = 1e-6


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
    # solvePnPRefineLM termination. EPS well below corner noise; 20 iters is
    # plenty after IPPE has seeded near the optimum.
    refine_lm_criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        _REFINE_LM_MAX_ITER,
        _REFINE_LM_EPS,
    )

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
            obj, corners_ippe, K, dist, rvec, tvec, refine_lm_criteria
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
    """Map a camera-frame 3-vector into the robot `base_link` frame."""
    p = np.asarray(xyz_camera, dtype=np.float64).reshape(3)
    return T_BASE_CAMERA[:3, :3] @ p + T_BASE_CAMERA[:3, 3]


def transform_tag_pose_to_base(T_camera_tag: np.ndarray) -> np.ndarray:
    """Map a camera-frame AprilTag pose into the robot `base_link` frame."""
    return T_BASE_CAMERA @ np.asarray(T_camera_tag, dtype=np.float64).reshape(4, 4)


def transform_tag_point_to_base(
    T_base_tag: np.ndarray,
    xyz_tag_local: np.ndarray,
) -> np.ndarray:
    """Map a tag-local point into the robot `base_link` frame.

    Tag-local axes use the AprilTag convention produced by detect_tags():
    +X right, +Y down, +Z into the printed tag.
    """
    T = np.asarray(T_base_tag, dtype=np.float64).reshape(4, 4)
    p_tag = np.asarray(xyz_tag_local, dtype=np.float64).reshape(3)
    return T[:3, :3] @ p_tag + T[:3, 3]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(
    target_tag_id: int = TARGET_TAG_ID,
    hover_z_m: float = HOVER_Z_M,
    tag_local_offset_m: np.ndarray | None = None,
    move_duration_s: float = MOVE_DURATION_S,
    move_hz: float = MOVE_HZ,
    simulate_move: bool = True,
    direct_xyz: np.ndarray | None = None,
    sim_current_joints_deg: np.ndarray | None = None,
    sim_from_robot: bool = False,
    visualize: bool = False,
    save_plot: Path | None = None,
    max_final_residual_mm: float = MAX_FINAL_RESIDUAL_MM,
    hold_wrist_roll: bool = True,
    closed_loop_execution: bool = True,
    debug_fk_chain: bool = False,
    disable_torque: bool = False,
) -> None:
    global T_BASE_CAMERA

    tag_local_offset = (
        TAG_LOCAL_OFFSET_M.copy()
        if tag_local_offset_m is None
        else np.asarray(tag_local_offset_m, dtype=np.float64).reshape(3).copy()
    )
    # AprilTag +Z points into the tag plane after R_IPPE_TO_APRILTAG, so a
    # positive hover moves outward from the printed tag along -tag Z.
    tag_local_offset[2] -= hover_z_m

    if direct_xyz is not None:
        target = np.asarray(direct_xyz, dtype=np.float64).reshape(3)
        mode = "simulation" if simulate_move else "execution"
        print(
            f"[main] MoveArmV4 direct {mode}: base_link TCP target "
            f"({target[0]:+.3f}, {target[1]:+.3f}, {target[2]:+.3f}) m"
        )
        move_arm_v4(
            x=float(target[0]),
            y=float(target[1]),
            z=float(target[2]),
            duration=move_duration_s,
            hz=move_hz,
            simulate=simulate_move,
            current_joints_deg=sim_current_joints_deg,
            simulate_from_robot=sim_from_robot,
            visualize=visualize,
            save_plot=save_plot,
            max_final_residual_mm=max_final_residual_mm,
            hold_wrist_roll=hold_wrist_roll,
            closed_loop_execution=closed_loop_execution,
            debug_fk_chain=debug_fk_chain,
        )
        print("[main] MoveArmV4 direct run complete")
        return

    target_xyz_base: np.ndarray | None = None
    snapshot_arm_joints: np.ndarray | None = None
    cap = None
    pose_tracker = LiveCameraPoseTracker(disable_torque=disable_torque)
    print("[main] connecting robot for live T_base_camera from encoders...")
    pose_tracker.connect()
    try:
        T_BASE_CAMERA, snapshot_arm_joints = pose_tracker.read()
        cam_xyz_mm = T_BASE_CAMERA[:3, 3] * 1000.0
        print(
            f"[main] T_base_camera.t = ({cam_xyz_mm[0]:+7.1f},"
            f" {cam_xyz_mm[1]:+7.1f}, {cam_xyz_mm[2]:+7.1f}) mm"
        )
        print("[main] updating T_base_camera from current encoders every frame")
        if disable_torque:
            print(
                "[main] arm joint torque disabled for hand positioning; "
                "encoder readings stay live"
            )

        _load_camera_deps()

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
            raise SystemExit(f"camera idx={CAMERA_INDEX} returned no frame")
        # Camera is mounted upside-down on the wrist (matches deprecated/
        # wrist_extrinsic.py); rotate every frame so the calibration and the
        # detector see the image in the same orientation it was captured for.
        frame = cv2.rotate(frame, cv2.ROTATE_180)

        h, w = frame.shape[:2]
        if (w, h) != calib_size:
            raise SystemExit(
                f"camera delivered {(w, h)} but calibration is for {calib_size}; "
                "recapture camera_calib.npz at the runtime resolution or pick a "
                "camera that delivers the calibration resolution."
            )

        print(f"[main] camera idx={CAMERA_INDEX} @ {calib_size} (rotated 180)")
        offset_mm = tag_local_offset * 1000.0
        print(
            f"[main] target tag id={target_tag_id}, tag-local offset "
            f"({offset_mm[0]:+.1f}, {offset_mm[1]:+.1f}, {offset_mm[2]:+.1f}) mm"
        )
        if simulate_move:
            print("[main] MoveArmV4 simulation mode; pass --execute-move to command robot")
        print("[main] [SPACE] run MoveArmV4 for target tag   [q]/[ESC] quit")

        window_name = "apriltag -> MoveArmV4"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            T_BASE_CAMERA, snapshot_arm_joints = pose_tracker.read()

            detections = detect_tags(frame, detector, K, dist, TAG_SIZE_M)
            target = next((d for d in detections if d["id"] == target_tag_id), None)

            for d in detections:
                color = (0, 255, 0) if d["id"] == target_tag_id else (180, 180, 180)
                pts = d["corners"].astype(np.int32)
                cv2.polylines(frame, [pts], True, color, 2)
                cv2.drawFrameAxes(
                    frame, K, dist, d["rvec"], d["tvec"], TAG_SIZE_M * 0.5, 2
                )

                T_base_tag = transform_tag_pose_to_base(d["T_camera_tag"])
                xyz_base = T_base_tag[:3, 3]
                cx, cy = int(d["center"][0]), int(d["center"][1])
                label = (
                    f"id={d['id']} base_link "
                    f"({xyz_base[0]:+.3f}, {xyz_base[1]:+.3f}, "
                    f"{xyz_base[2]:+.3f}) m"
                )
                cv2.putText(
                    frame, label, (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
                )

            if target is not None:
                T_base_tag = transform_tag_pose_to_base(target["T_camera_tag"])
                tag_center_base = T_base_tag[:3, 3]
                target_point_base = transform_tag_point_to_base(T_base_tag, tag_local_offset)
                torque_state = "   TORQUE OFF" if disable_torque else ""
                status = f"tag {target_tag_id} VISIBLE   [SPACE] run MoveArmV4{torque_state}"
                tag_center_line = (
                    f"base_link tag: ({tag_center_base[0]:+.3f},"
                    f" {tag_center_base[1]:+.3f}, {tag_center_base[2]:+.3f}) m"
                )
                target_line = (
                    f"base_link target: ({target_point_base[0]:+.3f},"
                    f" {target_point_base[1]:+.3f}, {target_point_base[2]:+.3f}) m"
                )
                status_color = (0, 255, 0)
            else:
                status = (
                    f"tag {target_tag_id} NOT visible "
                    f"({len(detections)} other tag(s) seen)"
                )
                tag_center_line = ""
                target_line = ""
                status_color = (0, 0, 255)
            cv2.putText(
                frame, status, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2,
            )
            if tag_center_line:
                cv2.putText(
                    frame, tag_center_line, (10, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 2,
                )
                cv2.putText(
                    frame, target_line, (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2,
                )

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                if target is None:
                    print(f"[main] target tag {target_tag_id} not visible; ignoring SPACE")
                    continue
                ok, capture_frame = cap.read()
                if not ok:
                    print("[main] camera returned no frame at capture; ignoring SPACE")
                    continue
                capture_frame = cv2.rotate(capture_frame, cv2.ROTATE_180)
                T_BASE_CAMERA, snapshot_arm_joints = pose_tracker.read()
                capture_detections = detect_tags(capture_frame, detector, K, dist, TAG_SIZE_M)
                capture_target = next(
                    (d for d in capture_detections if d["id"] == target_tag_id),
                    None,
                )
                if capture_target is None:
                    print(f"[main] target tag {target_tag_id} not visible at capture")
                    continue
                T_base_tag = transform_tag_pose_to_base(capture_target["T_camera_tag"])
                target_xyz_base = transform_tag_point_to_base(
                    T_base_tag,
                    tag_local_offset,
                ).copy()
                break
    finally:
        if cap is not None:
            cap.release()
        if cv2 is not None:
            cv2.destroyAllWindows()
        pose_tracker.disconnect()

    if target_xyz_base is None:
        print("[main] quit before SPACE; not moving")
        return

    target_x = float(target_xyz_base[0])
    target_y = float(target_xyz_base[1])
    target_z = float(target_xyz_base[2])
    print(
        f"[main] MoveArmV4 base_link TCP target "
        f"({target_x:+.3f}, {target_y:+.3f}, {target_z:+.3f}) m  "
        f"(tag-local offset transformed through full T_base_tag)"
    )
    move_arm_v4(
        x=target_x,
        y=target_y,
        z=target_z,
        duration=move_duration_s,
        hz=move_hz,
        simulate=simulate_move,
        current_joints_deg=snapshot_arm_joints if simulate_move else None,
        visualize=visualize,
        save_plot=save_plot,
        max_final_residual_mm=max_final_residual_mm,
        hold_wrist_roll=hold_wrist_roll,
        closed_loop_execution=closed_loop_execution,
        debug_fk_chain=debug_fk_chain,
    )
    print("[main] MoveArmV4 run complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Detect AprilTags from the wrist camera and run MoveArmV4 for "
            "a tag on SPACE. Defaults to simulation unless --execute-move is set."
        )
    )
    parser.add_argument(
        "--xyz",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help=(
            "Skip camera detection and run MoveArmV4 directly for this "
            "base-frame gripper-tip target in meters. Defaults to simulation."
        ),
    )
    parser.add_argument(
        "--execute-move",
        action="store_true",
        help="send commands to the physical robot; default is MoveArmV4 simulation",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="show a Matplotlib 3D visualization of the MoveArmV4 plan",
    )
    parser.add_argument(
        "--save-plot",
        type=Path,
        help="save the MoveArmV4 visualization PNG to this path",
    )
    parser.add_argument(
        "--sim-current-joints",
        nargs=len(V4_ARM_JOINTS),
        type=float,
        metavar="DEG",
        help=(
            "simulation start joints in degrees for --xyz: "
            + ", ".join(V4_ARM_JOINTS)
            + "; defaults to all zeros"
        ),
    )
    parser.add_argument(
        "--sim-from-robot",
        action="store_true",
        help=(
            "with --xyz simulation, read current arm joints from the robot "
            "without commanding motion"
        ),
    )
    parser.add_argument(
        "--disable-torque",
        action="store_true",
        help=(
            "in AprilTag camera mode, disable arm joint torque after connecting "
            "so the wrist camera can be moved by hand while encoders stay live"
        ),
    )
    parser.add_argument(
        "--allow-wrist-roll",
        action="store_true",
        help="do not hold wrist_roll at its starting value during MoveArmV4 IK",
    )
    parser.add_argument(
        "--open-loop-execution",
        action="store_true",
        help="execute preplanned joint waypoints instead of closed-loop live-observation IK",
    )
    parser.add_argument(
        "--debug-fk-chain",
        action="store_true",
        help="print base-frame FK positions for each arm link before moving",
    )
    parser.add_argument(
        "--tag-id",
        type=int,
        default=TARGET_TAG_ID,
        help=f"AprilTag id to target (default {TARGET_TAG_ID})",
    )
    parser.add_argument(
        "--hover-mm",
        type=float,
        default=HOVER_Z_M * 1000.0,
        help=(
            "Shorthand outward hover from the tag plane in millimetres "
            f"(default {HOVER_Z_M * 1000.0:.0f}). Applied along tag-local -Z, "
            "not base_link Z."
        ),
    )
    parser.add_argument(
        "--tag-offset-mm",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        default=TAG_LOCAL_OFFSET_M * 1000.0,
        help=(
            "Target point offset from the tag center in tag-local millimetres. "
            "AprilTag axes: +X right, +Y down, +Z into the printed tag."
        ),
    )
    parser.add_argument(
        "--move-duration",
        type=float,
        default=MOVE_DURATION_S,
        help=f"MoveArmV4 duration in seconds (default {MOVE_DURATION_S:.1f})",
    )
    parser.add_argument(
        "--move-hz",
        type=float,
        default=MOVE_HZ,
        help=f"MoveArmV4 planning/control rate in Hz (default {MOVE_HZ:.1f})",
    )
    parser.add_argument(
        "--max-final-residual-mm",
        type=float,
        default=MAX_FINAL_RESIDUAL_MM,
        help=(
            "mark/refuse MoveArmV4 plans whose final FK tip misses the target "
            f"by more than this many mm (default {MAX_FINAL_RESIDUAL_MM:.1f})"
        ),
    )
    args = parser.parse_args()
    main(
        target_tag_id=args.tag_id,
        hover_z_m=args.hover_mm / 1000.0,
        tag_local_offset_m=np.array(args.tag_offset_mm, dtype=np.float64) / 1000.0,
        move_duration_s=args.move_duration,
        move_hz=args.move_hz,
        simulate_move=not args.execute_move,
        direct_xyz=None if args.xyz is None else np.array(args.xyz, dtype=np.float64),
        sim_current_joints_deg=(
            None
            if args.sim_current_joints is None
            else np.array(args.sim_current_joints, dtype=np.float64)
        ),
        sim_from_robot=args.sim_from_robot,
        visualize=args.visualize,
        save_plot=args.save_plot,
        max_final_residual_mm=args.max_final_residual_mm,
        hold_wrist_roll=not args.allow_wrist_roll,
        closed_loop_execution=not args.open_loop_execution,
        debug_fk_chain=args.debug_fk_chain,
        disable_torque=args.disable_torque,
    )
