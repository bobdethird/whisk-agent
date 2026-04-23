"""Localize the SO-101 base in a reference AprilTag's frame, using ONLY the
wrist camera plus FK plus hand-eye calibration.

This is the wrist-camera analog of `iphone_extrinsic.py`. With the iPhone
camera the camera itself is fixed in the base frame, so solving
T_base_iphoneCam once per session is enough. The wrist camera moves with
the arm, so T_base_wristCam is NOT fixed -- but the robot base IS fixed
relative to a tag taped to the table, so:

        T_tag_base   (pose of SO-101 base in the tag frame)

is the quantity that should stay constant as the arm moves around.

Transform chain (from a single wrist frame + one joint observation):

    T_base_flange    via FK                (URDF + joint angles, placo)
    T_flange_camera  from hand_eye_calib.npz
    T_camera_tag     via AprilTag PnP      (detector.py)

Composed:

    T_base_camera    = T_base_flange  @ T_flange_camera
    T_base_tag       = T_base_camera  @ T_camera_tag
    T_tag_base       = inv(T_base_tag)          # the thing that should be fixed
    T_tag_camera     = inv(T_camera_tag)        # moves as the arm moves
    T_tag_flange     = T_tag_base @ T_base_flange  # moves as the arm moves

The live 2D overlay labels each tag with its base-frame coordinates and the
current T_tag_base translation. The running stats window reports the
mean/std of T_tag_base[:3, 3] over the last few seconds -- small std
(sub-centimetre) confirms the hand-eye + FK + PnP chain is self-consistent.

The 3D viz puts the tag at the origin and draws the robot base, FK arm
chain, and wrist camera all expressed in the tag frame. If the arm moves
and the base "wobbles", something in the chain is wrong.

Controls:
    q/ESC : quit
    r     : reset running statistics

Usage:
    python wrist_extrinsic.py
    python wrist_extrinsic.py --tag-id 4
    python wrist_extrinsic.py --disable-torque    # pose arm by hand
"""

from __future__ import annotations

import argparse
import time
from collections import deque

import cv2
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

from camera_utils import disable_autofocus
from detector import TAG_SIZE_M, TagDetection, detect_tags, make_detector, render_overlay
from hand_eye_calib import load_hand_eye_calib
from iphone_extrinsic import (
    _invert_transform,
    _matrix_to_quat,
    _quat_to_matrix,
    average_quaternions,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WRIST_CAMERA_INDEX = 0
WRIST_CALIB_FILE = "wrist_camera_calib.npz"
CAMERA_BUFFER_SIZE = 1

ROBOT_PORT = "/dev/tty.usbmodem5AE60557941"
ROBOT_ID = "my_awesome_follower_arm"

URDF_PATH = "SO101/so101_new_calib.urdf"
TARGET_FRAME = "gripper_frame_link"

# FK joints that affect T_base_flange. Mirrors calibrate_hand_eye.py and
# move_to_tag.py -- gripper excluded because gripper_frame_link is attached
# via a fixed joint and the gripper obs is a 0..100 percent.
FK_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]
FK_JOINT_OBS_KEYS = [f"{n}.pos" for n in FK_JOINT_NAMES]

# Stationary reference tag -- the script treats this as the world origin.
REFERENCE_TAG_ID = 4
TAG_SIZE = TAG_SIZE_M

# Arm link chain rendered in the 3D viz (base -> gripper). Mirrors
# calibrate_hand_eye.py; missing links are silently skipped.
ARM_LINK_CHAIN = [
    "base_link",
    "shoulder_link",
    "upper_arm_link",
    "lower_arm_link",
    "wrist_link",
    "gripper_link",
    "gripper_frame_link",
]

# Running-stats window (how many recent T_tag_base samples to average).
STATS_WINDOW = 60

# 3D viz refresh cadence (every N camera frames).
VIZ_EVERY = 3

PREVIEW_WINDOW = "wrist extrinsic (tag frame)"


# ---------------------------------------------------------------------------
# LeRobot import probing (mirrors the other scripts)
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
# Torque helpers (optional -- enabled via --disable-torque for hand posing)
# ---------------------------------------------------------------------------
def _try_disable_torque(robot) -> bool:
    try:
        robot.bus.disable_torque()
        return True
    except Exception as e:
        print(f"[wrist] disable_torque() failed: {e}")
    try:
        for motor in robot.bus.motors:
            robot.bus.write("Torque_Enable", motor, 0)
        return True
    except Exception as e:
        print(f"[wrist] per-motor Torque_Enable=0 failed: {e}")
    return False


def _try_enable_torque(robot) -> bool:
    try:
        robot.bus.enable_torque()
        return True
    except Exception:
        pass
    try:
        for motor in robot.bus.motors:
            robot.bus.write("Torque_Enable", motor, 1)
        return True
    except Exception:
        pass
    return False


# ---------------------------------------------------------------------------
# Camera setup
# ---------------------------------------------------------------------------
def _open_camera(index: int, expected_size: tuple[int, int]) -> cv2.VideoCapture:
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
    h, w = frame.shape[:2]
    if (w, h) != expected_size:
        raise SystemExit(
            f"wrist camera delivered {(w, h)} but calibration is for "
            f"{expected_size}"
        )
    return cap


def _read_joint_deg(robot) -> np.ndarray:
    obs = robot.get_observation()
    return np.asarray(
        [float(obs[k]) for k in FK_JOINT_OBS_KEYS], dtype=np.float64
    )


def _get_arm_link_poses(kinematics) -> dict[str, np.ndarray]:
    """Query T_base_link for every link in ARM_LINK_CHAIN.

    Must be called AFTER kinematics.forward_kinematics() so placo's internal
    state reflects the latest joint positions.
    """
    poses: dict[str, np.ndarray] = {}
    for name in ARM_LINK_CHAIN:
        try:
            T = np.asarray(
                kinematics.robot.get_T_world_frame(name), dtype=np.float64
            )
            poses[name] = T
        except Exception:
            pass
    return poses


# ---------------------------------------------------------------------------
# Stats window: average T_tag_base across the last STATS_WINDOW sightings.
# Rotations averaged via Markley quaternion mean (imported from iphone_extrinsic).
# ---------------------------------------------------------------------------
class TagBaseStats:
    """Rolling mean of T_tag_base over the last `window` sightings."""

    def __init__(self, window: int = STATS_WINDOW):
        self.window = window
        self.quats: deque[np.ndarray] = deque(maxlen=window)
        self.trans: deque[np.ndarray] = deque(maxlen=window)

    def push(self, T_tag_base: np.ndarray) -> None:
        self.quats.append(_matrix_to_quat(T_tag_base[:3, :3]))
        self.trans.append(T_tag_base[:3, 3].copy())

    def reset(self) -> None:
        self.quats.clear()
        self.trans.clear()

    def count(self) -> int:
        return len(self.trans)

    def mean(self) -> np.ndarray | None:
        if not self.trans:
            return None
        mean_q = average_quaternions(np.stack(list(self.quats), axis=0))
        mean_t = np.mean(np.stack(list(self.trans), axis=0), axis=0)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = _quat_to_matrix(mean_q)
        T[:3, 3] = mean_t
        return T

    def translation_std_mm(self) -> np.ndarray | None:
        if not self.trans:
            return None
        return np.std(np.stack(list(self.trans), axis=0), axis=0) * 1000.0


# ---------------------------------------------------------------------------
# 3D viz (tag at origin, everything else in the tag frame)
# ---------------------------------------------------------------------------
def _draw_triad(
    ax,
    T: np.ndarray,
    scale: float = 0.03,
    label: str | None = None,
) -> None:
    origin = T[:3, 3]
    for i, color in enumerate("rgb"):
        end = origin + T[:3, i] * scale
        ax.plot(
            [origin[0], end[0]],
            [origin[1], end[1]],
            [origin[2], end[2]],
            color=color,
            lw=2,
        )
    if label:
        ax.text(origin[0], origin[1], origin[2], f"  {label}", fontsize=8)


def _draw_camera_frustum(
    ax,
    T: np.ndarray,
    depth: float = 0.05,
    scale: float = 0.03,
    color: str = "crimson",
    label: str | None = None,
) -> None:
    """Tiny pyramid for the wrist camera, apex at T origin, looking +Z."""
    apex = T[:3, 3]
    R = T[:3, :3]
    corners_cam = np.array([
        [-scale, -scale, depth],
        [ scale, -scale, depth],
        [ scale,  scale, depth],
        [-scale,  scale, depth],
    ])
    corners_world = (R @ corners_cam.T).T + apex
    segs = [[apex, c] for c in corners_world]
    segs += [[corners_world[i], corners_world[(i + 1) % 4]] for i in range(4)]
    ax.add_collection3d(Line3DCollection(segs, colors=color, linewidths=1.5))
    ax.scatter(*apex, color=color, s=20, zorder=10)
    if label:
        ax.text(apex[0], apex[1], apex[2], f"  {label}", fontsize=8, color=color)


def _draw_tag_patch(
    ax,
    T_tag_tag: np.ndarray,
    tag_size_m: float = TAG_SIZE,
    face_color: str = "cornflowerblue",
    edge_color: str = "navy",
) -> None:
    """Draw the reference tag as a flat square at the origin."""
    half = tag_size_m / 2.0
    local = np.array([
        [-half,  half, 0.0],
        [ half,  half, 0.0],
        [ half, -half, 0.0],
        [-half, -half, 0.0],
    ])
    world = (T_tag_tag[:3, :3] @ local.T).T + T_tag_tag[:3, 3]
    ax.add_collection3d(Poly3DCollection(
        [world],
        alpha=0.45,
        facecolor=face_color,
        edgecolor=edge_color,
    ))
    _draw_triad(ax, T_tag_tag, scale=tag_size_m * 0.8, label="tag (origin)")


def _refresh_tag_frame_viz(
    ax,
    T_tag_base: np.ndarray | None,
    T_tag_camera: np.ndarray | None,
    tag_frame_link_poses: dict[str, np.ndarray],
    mean_T_tag_base: np.ndarray | None,
    std_mm: np.ndarray | None,
    sample_count: int,
    joint_deg: np.ndarray | None,
) -> None:
    ax.cla()
    # Fit a generous box around the SO-101 workspace (~half a metre).
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(-0.6, 0.6)
    ax.set_zlim(-0.1, 0.8)
    ax.set_box_aspect((1.2, 1.2, 0.9))
    ax.set_xlabel("X_tag (m)")
    ax.set_ylabel("Y_tag (m)")
    ax.set_zlabel("Z_tag (m)")
    ax.set_title("everything in the tag frame (tag at origin)")

    _draw_tag_patch(ax, np.eye(4), TAG_SIZE)

    if T_tag_base is not None:
        _draw_triad(ax, T_tag_base, scale=0.08, label="base (live)")

    # Mean base pose (faded) lets you see how much the live base is wandering
    # around the true (fixed) value.
    if mean_T_tag_base is not None:
        _draw_triad(ax, mean_T_tag_base, scale=0.06, label="base (mean)")

    chain_positions: list[np.ndarray] = []
    for name in ARM_LINK_CHAIN:
        T = tag_frame_link_poses.get(name)
        if T is None:
            continue
        chain_positions.append(T[:3, 3])
        _draw_triad(ax, T, scale=0.025)
    if len(chain_positions) >= 2:
        segs = [
            [chain_positions[i], chain_positions[i + 1]]
            for i in range(len(chain_positions) - 1)
        ]
        ax.add_collection3d(Line3DCollection(segs, colors="gray", linewidths=3))

    if T_tag_camera is not None:
        _draw_camera_frustum(ax, T_tag_camera, label="wrist cam")

    lines = [f"samples: {sample_count}"]
    if T_tag_base is not None:
        t_mm = T_tag_base[:3, 3] * 1000.0
        lines.append(
            f"T_tag_base.t  = ({t_mm[0]:+7.1f},"
            f"{t_mm[1]:+7.1f},{t_mm[2]:+7.1f}) mm  [live]"
        )
    else:
        lines.append("T_tag_base.t  = (tag not visible)")
    if mean_T_tag_base is not None:
        m_mm = mean_T_tag_base[:3, 3] * 1000.0
        lines.append(
            f"              mean=({m_mm[0]:+7.1f},"
            f"{m_mm[1]:+7.1f},{m_mm[2]:+7.1f}) mm"
        )
    if std_mm is not None:
        lines.append(
            f"              std =({std_mm[0]:6.2f},"
            f"{std_mm[1]:6.2f},{std_mm[2]:6.2f}) mm"
        )
    lines.append("")
    if joint_deg is not None:
        for name, val in zip(FK_JOINT_NAMES, joint_deg):
            lines.append(f"{name:>14} = {val:+7.2f} deg")
    ax.text2D(
        0.02, 0.98, "\n".join(lines),
        transform=ax.transAxes, fontsize=8, va="top", family="monospace",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _format_xyz_mm(xyz_m: np.ndarray) -> str:
    mm = np.asarray(xyz_m, dtype=np.float64).ravel() * 1000.0
    return f"({mm[0]:+7.1f},{mm[1]:+7.1f},{mm[2]:+7.1f}) mm"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tag-id", type=int, default=REFERENCE_TAG_ID,
        help="id of the reference AprilTag treated as world origin",
    )
    parser.add_argument(
        "--camera-index", type=int, default=WRIST_CAMERA_INDEX,
        help="OpenCV index for the wrist camera",
    )
    parser.add_argument(
        "--disable-torque", action="store_true",
        help="disable motor torque so you can pose the arm by hand",
    )
    parser.add_argument(
        "--stats-window", type=int, default=STATS_WINDOW,
        help="how many recent T_tag_base samples to average for stats",
    )
    args = parser.parse_args()

    calib = np.load(WRIST_CALIB_FILE)
    K = calib["K"].astype(np.float64)
    dist = calib["dist"].astype(np.float64)
    size = (int(calib["image_size"][0]), int(calib["image_size"][1]))

    try:
        T_flange_camera = load_hand_eye_calib()
    except FileNotFoundError:
        raise SystemExit(
            "hand_eye_calib.npz not found -- run calibrate_hand_eye.py first"
        )
    print("[wrist] loaded hand_eye_calib.npz:")
    print(
        f"[wrist]   T_flange_camera.t = {_format_xyz_mm(T_flange_camera[:3, 3])}"
    )

    RobotKinematics = _probe_robot_kinematics()
    kinematics = RobotKinematics(
        urdf_path=URDF_PATH,
        target_frame_name=TARGET_FRAME,
        joint_names=FK_JOINT_NAMES,
    )

    SO101Follower, SO101FollowerConfig = _probe_so101_follower()
    robot = SO101Follower(SO101FollowerConfig(id=ROBOT_ID, port=ROBOT_PORT))
    robot.connect()
    print(f"[wrist] robot connected on {ROBOT_PORT}")

    torque_off = False
    if args.disable_torque:
        torque_off = _try_disable_torque(robot)
        print(
            f"[wrist] disable_torque requested; "
            f"{'disabled' if torque_off else 'NOT disabled'}"
        )

    detector = make_detector()
    cap = _open_camera(args.camera_index, size)
    print(f"[wrist] opened wrist camera idx={args.camera_index} size={size}")
    print(
        f"[wrist] treating tag id={args.tag_id} as world origin; "
        f"stats window = {args.stats_window} samples"
    )
    print("[wrist] controls: [q]/[ESC]=quit   [r]=reset running stats")

    cv2.namedWindow(PREVIEW_WINDOW, cv2.WINDOW_AUTOSIZE)

    plt.ion()
    fig3d = plt.figure("robot in tag frame", figsize=(7, 7))
    ax3d = fig3d.add_subplot(111, projection="3d")

    stats = TagBaseStats(window=args.stats_window)
    fps_window: deque[float] = deque(maxlen=30)
    last_print_t = time.monotonic()
    frame_i = 0

    try:
        while True:
            loop_start = time.monotonic()

            ok, frame = cap.read()
            if not ok:
                continue

            # --- joints + FK (needed every frame to build T_base_camera) -----
            joint_deg: np.ndarray | None = None
            T_base_flange: np.ndarray | None = None
            link_poses_base: dict[str, np.ndarray] = {}
            try:
                joint_deg = _read_joint_deg(robot)
                T_base_flange = np.asarray(
                    kinematics.forward_kinematics(joint_deg), dtype=np.float64
                )
                link_poses_base = _get_arm_link_poses(kinematics)
            except Exception as e:
                if frame_i % 60 == 0:
                    print(f"[wrist] joint/FK read hiccup: {e}")

            # --- tag detection ---------------------------------------------
            detections: list[TagDetection] = detect_tags(
                frame, detector, K, dist, TAG_SIZE
            )
            hit = next((d for d in detections if d.id == args.tag_id), None)

            T_tag_base: np.ndarray | None = None
            T_tag_camera: np.ndarray | None = None
            T_base_camera: np.ndarray | None = None
            if T_base_flange is not None:
                T_base_camera = T_base_flange @ T_flange_camera

            if hit is not None and T_base_camera is not None:
                T_camera_tag = hit.T_camera_tag
                T_base_tag = T_base_camera @ T_camera_tag
                T_tag_base = _invert_transform(T_base_tag)
                T_tag_camera = _invert_transform(T_camera_tag)
                stats.push(T_tag_base)

            # --- 2D overlay -------------------------------------------------
            render_overlay(frame, detections, K, dist, TAG_SIZE)
            if T_tag_base is not None:
                status = (
                    f"tag {args.tag_id} VISIBLE   "
                    f"T_tag_base.t = {_format_xyz_mm(T_tag_base[:3, 3])}   "
                    f"samples {stats.count()}"
                )
                color = (0, 255, 0)
            else:
                if hit is None:
                    reason = f"tag {args.tag_id} MISSING"
                else:
                    reason = "joint/FK unavailable"
                status = (
                    f"{reason}   samples {stats.count()}   "
                    f"T_flange_camera.t = {_format_xyz_mm(T_flange_camera[:3, 3])}"
                )
                color = (0, 0, 255)
            cv2.putText(
                frame, status, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
            )

            mean_T = stats.mean()
            std_mm = stats.translation_std_mm()
            if mean_T is not None and std_mm is not None:
                line2 = (
                    f"mean_T_tag_base.t = {_format_xyz_mm(mean_T[:3, 3])}   "
                    f"std = ({std_mm[0]:5.2f},{std_mm[1]:5.2f},{std_mm[2]:5.2f}) mm"
                )
                cv2.putText(
                    frame, line2, (10, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 0), 2,
                )

            cv2.imshow(PREVIEW_WINDOW, frame)

            # --- 3D viz in tag frame ---------------------------------------
            if frame_i % VIZ_EVERY == 0:
                # Express each arm link in the tag frame for drawing.
                link_poses_tag: dict[str, np.ndarray] = {}
                if T_tag_base is not None:
                    for name, T_base_link in link_poses_base.items():
                        link_poses_tag[name] = T_tag_base @ T_base_link
                _refresh_tag_frame_viz(
                    ax3d,
                    T_tag_base=T_tag_base,
                    T_tag_camera=T_tag_camera,
                    tag_frame_link_poses=link_poses_tag,
                    mean_T_tag_base=mean_T,
                    std_mm=std_mm,
                    sample_count=stats.count(),
                    joint_deg=joint_deg,
                )
                plt.pause(0.001)

            # --- periodic console print ------------------------------------
            dt = time.monotonic() - loop_start
            fps_window.append(1.0 / max(dt, 1e-6))
            if time.monotonic() - last_print_t > 1.0:
                if T_tag_base is not None:
                    extra = (
                        f"T_tag_base.t={_format_xyz_mm(T_tag_base[:3, 3])}"
                    )
                    if mean_T is not None and std_mm is not None:
                        extra += (
                            f"  mean={_format_xyz_mm(mean_T[:3, 3])}"
                            f"  std=({std_mm[0]:5.2f},{std_mm[1]:5.2f},"
                            f"{std_mm[2]:5.2f}) mm"
                        )
                else:
                    extra = "tag not visible"
                print(
                    f"[wrist] frame {frame_i} | "
                    f"FPS {np.mean(fps_window):4.1f} | "
                    f"samples {stats.count():3d} | {extra}"
                )
                last_print_t = time.monotonic()

            frame_i += 1

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in (ord("r"), ord("R")):
                stats.reset()
                print("[wrist] running stats reset")
    finally:
        try:
            cap.release()
        except Exception:
            pass
        cv2.destroyAllWindows()
        plt.close("all")

        if torque_off:
            re_on = _try_enable_torque(robot)
            print(f"[wrist] torque re-enabled: {re_on}")

        try:
            robot.disconnect()
        except Exception as e:
            print(f"[wrist] robot disconnect warning: {e}")

    # -----------------------------------------------------------------------
    # Final averaged result (mirrors iphone_extrinsic's printout)
    # -----------------------------------------------------------------------
    mean_T = stats.mean()
    std_mm = stats.translation_std_mm()
    if mean_T is None or std_mm is None:
        print("[wrist] no tag sightings captured; nothing to summarize")
        return

    print(
        f"\n[wrist] final averaged T_tag_base over last {stats.count()} samples:"
    )
    print(np.array2string(mean_T, precision=5, suppress_small=True))
    print(
        f"[wrist] translation (mm) = {_format_xyz_mm(mean_T[:3, 3])}   "
        f"std = ({std_mm[0]:5.2f},{std_mm[1]:5.2f},{std_mm[2]:5.2f}) mm"
    )
    T_base_tag = _invert_transform(mean_T)
    print(
        f"[wrist] equivalent T_base_tag.t = {_format_xyz_mm(T_base_tag[:3, 3])}"
    )


if __name__ == "__main__":
    main()
