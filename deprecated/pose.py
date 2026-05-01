"""Fixed iPhone AprilTag world-pose tracker.

This simplified version of `pose.py` assumes:
    - only the iPhone camera is used
    - the iPhone is rigidly mounted
    - the table defines the world frame, with z=0 on the tabletop
    - AprilTags may move freely on or above the table

Usage:
    python pose.py
"""

from __future__ import annotations

import time
from collections import deque

import cv2
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

from camera_utils import disable_autofocus
from detector import TAG_SIZE_M, TagDetection, detect_tags, make_detector, render_overlay

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
IPHONE_CAMERA_INDEX = 1
IPHONE_CALIB_FILE = "camera_calib.npz"
CAMERA_BUFFER_SIZE = 1

# World frame:
#   - origin: point on the table directly under the iPhone lens
#   - +x: camera-right projected onto the table
#   - +y: across the table
#   - +z: upward, away from the table surface
#
# Assumption for the supplied mount measurement:
#   - the phone is 35 cm above the table
#   - the phone is tilted by -22 deg about its x/right axis relative to the
#     straight-down view
#   - the phone is not yawed relative to the table axes
IPHONE_X_M = 0.0
IPHONE_Y_M = 0.0
IPHONE_HEIGHT_M = 0.35
IPHONE_TILT_X_DEG = -22.0
IPHONE_YAW_DEG = 0.0

PLOT_EVERY = 1
PRINT_EVERY = 30
TABLE_DRAW_SIZE_M = 0.90


def _rotation_x_deg(theta_deg: float) -> np.ndarray:
    theta = np.deg2rad(theta_deg)
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, c, -s],
        [0.0, s, c],
    ], dtype=np.float64)


def _rotation_z_deg(theta_deg: float) -> np.ndarray:
    theta = np.deg2rad(theta_deg)
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return np.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def build_T_world_iphone(
    x_m: float = IPHONE_X_M,
    y_m: float = IPHONE_Y_M,
    height_m: float = IPHONE_HEIGHT_M,
    tilt_x_deg: float = IPHONE_TILT_X_DEG,
    yaw_deg: float = IPHONE_YAW_DEG,
) -> np.ndarray:
    """Build the fixed world<-iPhone transform from a simple mount survey.

    The base orientation is a straight-down camera:
        x_cam -> +x_world
        y_cam -> -y_world
        z_cam -> -z_world

    A tilt about the camera x axis then lets the optical axis lean across the
    table. With the current convention, negative tilt points the optical axis
    toward negative world y.
    """
    T_world_iphone = np.eye(4, dtype=np.float64)

    R_world_iphone_down = np.array([
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ], dtype=np.float64)
    T_world_iphone[:3, :3] = (
        _rotation_z_deg(yaw_deg)
        @ R_world_iphone_down
        @ _rotation_x_deg(tilt_x_deg)
    )
    T_world_iphone[:3, 3] = np.array([x_m, y_m, height_m], dtype=np.float64)
    return T_world_iphone


def _format_xyz_mm(xyz_m: np.ndarray) -> str:
    xyz_mm = np.asarray(xyz_m, dtype=np.float64).ravel() * 1000.0
    return f"({xyz_mm[0]:+7.1f}, {xyz_mm[1]:+7.1f}, {xyz_mm[2]:+7.1f}) mm"


def get_tag_poses_world(
    frame: np.ndarray,
    T_world_iphone: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    detector,
    tag_size_m: float = TAG_SIZE_M,
) -> tuple[list[TagDetection], dict[int, np.ndarray]]:
    """Detect AprilTags and return their current world-frame poses."""
    detections = detect_tags(frame, detector, K, dist, tag_size_m)
    world_poses: dict[int, np.ndarray] = {}
    for det in detections:
        world_poses[det.id] = T_world_iphone @ det.T_camera_tag
    return detections, world_poses


def _annotate_world_overlay(
    frame: np.ndarray,
    detections: list[TagDetection],
    world_poses: dict[int, np.ndarray],
) -> None:
    for det in detections:
        T_world_tag = world_poses.get(det.id)
        if T_world_tag is None:
            continue
        xyz = T_world_tag[:3, 3]
        label = f"world=({xyz[0]:+.3f}, {xyz[1]:+.3f}, {xyz[2]:+.3f})m"
        cx = int(det.center[0])
        cy = int(det.center[1]) + 22
        cv2.putText(
            frame,
            label,
            (cx, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 0),
            2,
        )


def _print_snapshot(world_poses: dict[int, np.ndarray]) -> None:
    if not world_poses:
        print("    no visible tags")
        return

    for tag_id, T_world_tag in sorted(world_poses.items()):
        print(f"    tag {tag_id:>3} world_xyz = {_format_xyz_mm(T_world_tag[:3, 3])}")
        print(np.array2string(T_world_tag, precision=5, suppress_small=True))


def _draw_triad(
    ax,
    T: np.ndarray,
    scale: float = 0.05,
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
    T_world_iphone: np.ndarray,
    K: np.ndarray,
    img_size: tuple[int, int],
    depth: float = 0.18,
    color: str = "dodgerblue",
) -> None:
    w, h = img_size
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    px = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
    rays_cam = np.stack([
        (px[:, 0] - cx) / fx,
        (px[:, 1] - cy) / fy,
        np.ones(4),
    ], axis=1) * depth
    rays_world = (T_world_iphone[:3, :3] @ rays_cam.T).T + T_world_iphone[:3, 3]
    origin = T_world_iphone[:3, 3]
    segs = [[origin, ray] for ray in rays_world]
    segs += [[rays_world[i], rays_world[(i + 1) % 4]] for i in range(4)]
    ax.add_collection3d(Line3DCollection(segs, colors=color, linewidths=1))


def _draw_table(ax, size_m: float = TABLE_DRAW_SIZE_M) -> None:
    half = size_m / 2.0
    table = np.array([
        [-half, -half, 0.0],
        [half, -half, 0.0],
        [half, half, 0.0],
        [-half, half, 0.0],
    ], dtype=np.float64)
    ax.add_collection3d(Poly3DCollection(
        [table],
        alpha=0.12,
        facecolor="lightgray",
        edgecolor="silver",
    ))


def _draw_tag(
    ax,
    T_world_tag: np.ndarray,
    tag_id: int,
    tag_size_m: float = TAG_SIZE_M,
) -> None:
    half = tag_size_m / 2.0
    local = np.array([
        [-half, -half, 0.0],
        [half, -half, 0.0],
        [half, half, 0.0],
        [-half, half, 0.0],
    ], dtype=np.float64)
    world = (T_world_tag[:3, :3] @ local.T).T + T_world_tag[:3, 3]
    ax.add_collection3d(Poly3DCollection(
        [world],
        alpha=0.45,
        facecolor="mediumseagreen",
        edgecolor="navy",
    ))
    _draw_triad(ax, T_world_tag, scale=tag_size_m * 0.7)
    ax.text(
        T_world_tag[0, 3],
        T_world_tag[1, 3],
        T_world_tag[2, 3],
        f"  id={tag_id}",
        fontsize=8,
    )


def refresh_3d(
    ax,
    world_poses: dict[int, np.ndarray],
    T_world_iphone: np.ndarray,
    K: np.ndarray,
    img_size: tuple[int, int],
) -> None:
    ax.cla()
    ax.set_xlim(-0.45, 0.45)
    ax.set_ylim(-0.45, 0.45)
    ax.set_zlim(0.0, 0.55)
    ax.set_box_aspect((1.0, 1.0, 0.7))
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z up (m)")

    _draw_table(ax)
    _draw_triad(ax, np.eye(4, dtype=np.float64), scale=0.08, label="world")
    _draw_triad(ax, T_world_iphone, scale=0.06, label="iphone")
    _draw_camera_frustum(ax, T_world_iphone, K, img_size)

    for tag_id, T_world_tag in sorted(world_poses.items()):
        _draw_tag(ax, T_world_tag, tag_id)


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
            f"camera at index {index} delivered {(w, h)} but calibration is for "
            f"{expected_size}. Either force the camera to the calibrated "
            f"resolution or recalibrate at {(w, h)}."
        )
    return cap


def _shutdown(cap: cv2.VideoCapture) -> None:
    try:
        cap.release()
    except Exception:
        pass
    cv2.destroyAllWindows()
    plt.close("all")


def main() -> None:
    iphone_calib = np.load(IPHONE_CALIB_FILE)
    iphone_K = iphone_calib["K"].astype(np.float64)
    iphone_dist = iphone_calib["dist"].astype(np.float64)
    iphone_size = (
        int(iphone_calib["image_size"][0]),
        int(iphone_calib["image_size"][1]),
    )

    T_world_iphone = build_T_world_iphone()
    print("[pose] world frame: table surface is z=0")
    print("[pose] world origin: point directly below the iPhone lens")
    print(f"[pose] iPhone height = {IPHONE_HEIGHT_M:.3f} m")
    print(f"[pose] iPhone tilt_x = {IPHONE_TILT_X_DEG:+.1f} deg")
    print(f"[pose] iPhone yaw = {IPHONE_YAW_DEG:+.1f} deg")
    print("[pose] T_world_iphone =")
    print(np.array2string(T_world_iphone, precision=5, suppress_small=True))

    detector = make_detector()
    cap = _open_camera(IPHONE_CAMERA_INDEX, iphone_size)
    print(f"[pose] opened iPhone camera idx={IPHONE_CAMERA_INDEX}")

    plt.ion()
    fig = plt.figure("table/world frame", figsize=(7, 7))
    ax3d = fig.add_subplot(111, projection="3d")

    fps_window: deque[float] = deque(maxlen=30)
    last_print_t = time.monotonic()
    frame_i = 0

    try:
        while True:
            loop_start = time.monotonic()

            ok, frame = cap.read()
            if not ok:
                continue

            detections, world_poses = get_tag_poses_world(
                frame,
                T_world_iphone,
                iphone_K,
                iphone_dist,
                detector,
            )

            render_overlay(frame, detections, iphone_K, iphone_dist)
            _annotate_world_overlay(frame, detections, world_poses)
            cv2.imshow("iphone", frame)

            if frame_i % PLOT_EVERY == 0:
                refresh_3d(ax3d, world_poses, T_world_iphone, iphone_K, iphone_size)
                plt.pause(0.001)

            dt = time.monotonic() - loop_start
            fps_window.append(1.0 / max(dt, 1e-6))
            if time.monotonic() - last_print_t > 1.0:
                visible_ids = sorted(world_poses)
                print(
                    f"[pose] frame {frame_i} | FPS {np.mean(fps_window):.1f} | "
                    f"visible tags {visible_ids}"
                )
                last_print_t = time.monotonic()

            if frame_i % PRINT_EVERY == 0:
                print(f"[pose] measurement snapshot @ frame {frame_i}")
                _print_snapshot(world_poses)

            frame_i += 1
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        _shutdown(cap)


if __name__ == "__main__":
    main()
