"""AX=XB hand-eye calibration for the SO-101 wrist camera.

Runs cv2.calibrateHandEye (Tsai method) over a set of snapped
(T_base_flange, T_camera_tag) pairs to produce T_flange_camera, the
fixed transform from `gripper_frame_link` to the wrist camera's optical
frame. Output is written to hand_eye_calib.npz.

Procedure
---------
1. Tape a single AprilTag (default id = TAG_ID, size TAG_SIZE_M) to a
   fixed, rigid spot in the workspace. The tag must NOT move during the
   calibration session.
2. Run:
       python calibrate_hand_eye.py
3. Torque is disabled on the follower so you can physically pose it by
   hand. At each pose, make sure the tag is visible in the wrist camera
   (the overlay shows "VISIBLE"/"MISSING"), hold the arm still, and
   press SPACE to snapshot the pair.
4. Vary BOTH position and orientation (tilts, rotations). Aim for
   REQUESTED_POSES (25) distinct poses; MIN_POSES (15) is the minimum
   the Tsai solver needs for a well-conditioned fit.
5. Press ENTER to run the solver. Press ESC to abort.

The script then prints a residual analysis -- for each captured pose it
reconstructs T_base_tag = T_base_flange @ X @ T_camera_tag; if the
calibration is good, these should all agree to within a few millimetres.

Conventions
-----------
- cv2.calibrateHandEye inputs:
    R/t _gripper2base : T_base_gripper (gripper-pose-in-base) from FK
    R/t _target2cam   : T_camera_tag   (tag-pose-in-camera)  from PnP
- Output:
    R/t _cam2gripper  = T_gripper_camera = T_flange_camera

"""

from __future__ import annotations

import math
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from camera_utils import disable_autofocus
from detector import TAG_SIZE_M, detect_tags, make_detector, render_overlay
from hand_eye_calib import load_hand_eye_calib, save_hand_eye_calib

# ---------------------------------------------------------------------------
# Config (edit to match your setup)
# ---------------------------------------------------------------------------
TAG_ID = 4                    # id of the stationary calibration tag
MIN_POSES = 15                # Tsai typically needs >= ~10; 15+ is safer
REQUESTED_POSES = 25          # target count before user should press ENTER

ROBOT_PORT = "/dev/tty.usbmodem5AE60557941"
ROBOT_ID = "my_awesome_follower_arm"

URDF_PATH = "SO101/so101_new_calib.urdf"
TARGET_FRAME = "gripper_frame_link"

# FK joints that affect T_base_flange. gripper_frame_link is attached to
# gripper_link by a *fixed* joint BEFORE the revolute 'gripper' joint, so
# the gripper value does not influence T_base_flange. We also omit gripper
# because its observation is normalized to 0..100 (not degrees), which would
# be interpreted incorrectly by forward_kinematics.
FK_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]
FK_JOINT_OBS_KEYS = [f"{n}.pos" for n in FK_JOINT_NAMES]

WRIST_CAMERA_INDEX = 0
WRIST_CALIB_FILE = "wrist_camera_calib.npz"
TAG_SIZE = TAG_SIZE_M

STILLNESS_SETTLE_S = 0.2      # short sleep after SPACE to let motion settle

# Rough guess of where the wrist camera lens is relative to gripper_frame_link
# (the TCP between closed jaws). Used ONLY by the 3D viz when no prior
# hand_eye_calib.npz exists. Does NOT affect cv2.calibrateHandEye.
#
# Defaults: 35.6 mm along +Y from the TCP, with the optical axis tilted 25 deg
# about +X (complement of the 65-deg-from-vertical physical measurement) so
# the camera looks forward-and-slightly-down along the approach direction.
VIZ_FALLBACK_CAMERA_OFFSET_MM = (0.0, 35.6, 0.0)   # (x, y, z) in gripper_frame_link
VIZ_FALLBACK_CAMERA_TILT_X_DEG = 25.0              # rotation about +X; +ve = tip optical +Z toward -Y (down)

# Arm links in base-to-tip chain order, used by the live 3D viz so you can
# visually verify that FK is tracking the real arm.
ARM_LINK_CHAIN = [
    "base_link",
    "shoulder_link",
    "upper_arm_link",
    "lower_arm_link",
    "wrist_link",
    "gripper_link",
    "gripper_frame_link",
]
VIZ_EVERY = 3                 # refresh 3D arm viz every N camera frames
VIZ_FIT_MARGIN_M = 0.06       # padding around robot/tag in the live 3D view
VIZ_MIN_AXIS_SPAN_M = 0.18    # avoid excessive zoom before all points are known


# ---------------------------------------------------------------------------
# LeRobot import probing
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
# Torque helpers (defensive -- exact bus API can drift)
# ---------------------------------------------------------------------------
def _try_disable_torque(robot) -> bool:
    try:
        robot.bus.disable_torque()
        return True
    except Exception as e:
        print(f"[calib] robot.bus.disable_torque() failed: {e}")
    try:
        for motor in robot.bus.motors:
            robot.bus.write("Torque_Enable", motor, 0)
        return True
    except Exception as e:
        print(f"[calib] per-motor Torque_Enable=0 failed: {e}")
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
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, expected_size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, expected_size[1])
    ok, frame = cap.read()
    if not ok:
        raise SystemExit(f"wrist camera at index {index} returned no frame")
    h, w = frame.shape[:2]
    if (w, h) != expected_size:
        raise SystemExit(
            f"wrist camera delivered {(w, h)} but calibration is for {expected_size}"
        )
    return cap


# ---------------------------------------------------------------------------
# Live arm visualization
# ---------------------------------------------------------------------------
def _get_arm_link_poses(kinematics) -> dict[str, np.ndarray]:
    """Query T_base_link for every link in ARM_LINK_CHAIN.

    Must be called AFTER kinematics.forward_kinematics() so placo's internal
    state reflects the latest joint positions. Missing links are skipped.
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


def _invert_transform(T: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def _draw_camera_frustum(ax, T: np.ndarray, depth: float = 0.05, scale: float = 0.04,
                         color: str = "black", label: str | None = None) -> None:
    """Tiny pyramid representing a camera (apex at T origin, looking +Z).

    Uses the OpenCV convention where the optical axis is +Z and image right/down
    is +X/+Y -- same convention the wrist camera uses.
    """
    apex = T[:3, 3]
    Rm = T[:3, :3]
    corners_cam = np.array([
        [-scale, -scale, depth],
        [ scale, -scale, depth],
        [ scale,  scale, depth],
        [-scale,  scale, depth],
    ])
    corners_base = (Rm @ corners_cam.T).T + apex
    segs = [[apex, c] for c in corners_base]
    segs += [[corners_base[i], corners_base[(i + 1) % 4]] for i in range(4)]
    ax.add_collection3d(Line3DCollection(segs, colors=color, linewidths=1.5))
    ax.scatter(*apex, color=color, s=20, zorder=10)
    if label:
        ax.text(apex[0], apex[1], apex[2], f"  {label}", fontsize=8, color=color)


def _draw_triad(ax, T: np.ndarray, scale: float = 0.03, label: str | None = None) -> None:
    origin = T[:3, 3]
    for i, color in enumerate("rgb"):
        end = origin + T[:3, i] * scale
        ax.plot([origin[0], end[0]], [origin[1], end[1]], [origin[2], end[2]],
                color=color, lw=2)
    if label:
        ax.text(origin[0], origin[1], origin[2], f"  {label}", fontsize=8)


def _draw_tag(ax, T_base_tag: np.ndarray, tag_id: int, tag_size_m: float,
              face_color: str = "cornflowerblue",
              edge_color: str = "navy",
              alpha: float = 0.5,
              wire_only: bool = False,
              label_prefix: str = "") -> None:
    """Draw a square patch at the tag's base-frame pose, with id label.

    When `wire_only=True`, only the 4 outline segments are drawn (no
    Poly3DCollection). Useful for the reference-ghost render style --
    matplotlib 3.10 chokes on an alpha-zero Poly3DCollection.
    """
    half = tag_size_m / 2.0
    local = np.array([
        [-half,  half, 0.0],
        [ half,  half, 0.0],
        [ half, -half, 0.0],
        [-half, -half, 0.0],
    ])
    world = (T_base_tag[:3, :3] @ local.T).T + T_base_tag[:3, 3]

    if wire_only:
        segs = [[world[i], world[(i + 1) % 4]] for i in range(4)]
        ax.add_collection3d(Line3DCollection(segs, colors=edge_color, linewidths=1.5))
    else:
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        ax.add_collection3d(Poly3DCollection(
            [world], alpha=alpha, facecolor=face_color, edgecolor=edge_color,
        ))

    _draw_triad(ax, T_base_tag, scale=tag_size_m * 0.8)
    ax.text(T_base_tag[0, 3], T_base_tag[1, 3], T_base_tag[2, 3],
            f"  {label_prefix}id={tag_id}", fontsize=9, color=edge_color, weight="bold")


def _tag_corners(T_base_tag: np.ndarray, tag_size_m: float) -> np.ndarray:
    half = tag_size_m / 2.0
    local = np.array([
        [-half,  half, 0.0],
        [ half,  half, 0.0],
        [ half, -half, 0.0],
        [-half, -half, 0.0],
    ])
    return (T_base_tag[:3, :3] @ local.T).T + T_base_tag[:3, 3]


def _fit_axes_to_points(
    ax,
    points: list[np.ndarray],
    margin_m: float = VIZ_FIT_MARGIN_M,
    min_axis_span_m: float = VIZ_MIN_AXIS_SPAN_M,
) -> None:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    center = (lo + hi) / 2.0
    span = np.maximum(hi - lo + margin_m * 2.0, min_axis_span_m)

    ax.set_xlim(center[0] - span[0] / 2.0, center[0] + span[0] / 2.0)
    ax.set_ylim(center[1] - span[1] / 2.0, center[1] + span[1] / 2.0)
    ax.set_zlim(center[2] - span[2] / 2.0, center[2] + span[2] / 2.0)
    ax.set_box_aspect(tuple(span))


def _refresh_arm_viz(
    ax,
    link_poses: dict[str, np.ndarray],
    joint_deg: np.ndarray | None,
    capture_count: int,
    anchor_tag: tuple[int, np.ndarray] | None = None,
    T_base_camera_fk: np.ndarray | None = None,
    T_base_camera_pnp: np.ndarray | None = None,
    tag_size_m: float = TAG_SIZE_M,
    he_is_estimate: bool = True,
) -> None:
    """Render the arm + stationary tag + FK/PnP camera estimates.

    The physical tag doesn't move (it's taped down). What "moves" in world
    coordinates is the wrist camera as the arm reposes. Two cameras are drawn:
      - FK camera: T_base_flange @ T_flange_camera_viz  (what we think it is)
      - PnP camera: T_base_tag_anchor @ inv(T_camera_tag)  (what reality says)
    A dashed line between them is the hand-eye calibration error.
    """
    ax.cla()
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z up (m)")
    ax.set_title("SO-101: tag is fixed; FK vs PnP camera disagreement = calib error")

    fit_points: list[np.ndarray] = [np.zeros(3, dtype=np.float64)]
    _draw_triad(ax, np.eye(4), scale=0.08, label="base")

    chain_positions: list[np.ndarray] = []
    for name in ARM_LINK_CHAIN:
        T = link_poses.get(name)
        if T is None:
            continue
        chain_positions.append(T[:3, 3])
        fit_points.append(T[:3, 3])
        _draw_triad(ax, T, scale=0.03, label=name.replace("_link", ""))

    if len(chain_positions) >= 2:
        segs = [[chain_positions[i], chain_positions[i + 1]]
                for i in range(len(chain_positions) - 1)]
        ax.add_collection3d(Line3DCollection(segs, colors="gray", linewidths=3))

    # Stationary tag at the anchor. This doesn't move between frames.
    if anchor_tag is not None:
        anchor_id, anchor_T = anchor_tag
        fit_points.extend(_tag_corners(anchor_T, tag_size_m))
        _draw_tag(ax, anchor_T, anchor_id, tag_size_m,
                  face_color="cornflowerblue", edge_color="navy", alpha=0.5)

    # "FK camera" -- where we currently think the camera is.
    if T_base_camera_fk is not None:
        fit_points.append(T_base_camera_fk[:3, 3])
        _draw_camera_frustum(ax, T_base_camera_fk, depth=0.05, scale=0.03,
                             color="dimgray", label="FK cam")

    # "PnP camera" -- where reality says the camera is (given stationary tag).
    if T_base_camera_pnp is not None:
        fit_points.append(T_base_camera_pnp[:3, 3])
        _draw_camera_frustum(ax, T_base_camera_pnp, depth=0.05, scale=0.03,
                             color="crimson", label="PnP cam")

    # Dashed-style error line between the two.
    error_mm: float | None = None
    if T_base_camera_fk is not None and T_base_camera_pnp is not None:
        segs = [[T_base_camera_fk[:3, 3], T_base_camera_pnp[:3, 3]]]
        ax.add_collection3d(Line3DCollection(
            segs, colors="crimson", linewidths=2, linestyles=(0, (4, 2)),
        ))
        error_mm = float(
            np.linalg.norm(T_base_camera_fk[:3, 3] - T_base_camera_pnp[:3, 3]) * 1000.0
        )

    _fit_axes_to_points(ax, fit_points)

    lines = [f"captures: {capture_count}"]
    if anchor_tag is not None:
        anchor_id, anchor_T = anchor_tag
        xyz_mm = anchor_T[:3, 3] * 1000.0
        est_suffix = " (from identity T_fc)" if he_is_estimate else ""
        lines.append(
            f"tag id={anchor_id:<3} base=({xyz_mm[0]:+6.1f},"
            f"{xyz_mm[1]:+6.1f},{xyz_mm[2]:+6.1f}) mm{est_suffix}"
        )
    else:
        lines.append("tag: waiting for first sighting...")
    if error_mm is not None:
        lines.append(f"FK<->PnP camera gap: {error_mm:6.1f} mm  (0 = good calib)")
    lines.append("")
    if joint_deg is not None:
        for name, val in zip(FK_JOINT_NAMES, joint_deg):
            lines.append(f"{name:>14} = {val:+7.2f} deg")
    else:
        lines.append("(no joint reading)")
    ax.text2D(0.02, 0.98, "\n".join(lines), transform=ax.transAxes,
              fontsize=8, va="top", family="monospace")


# ---------------------------------------------------------------------------
# Residual analysis
# ---------------------------------------------------------------------------
def _residual_report(
    T_flange_camera: np.ndarray,
    base_flanges: list[np.ndarray],
    camera_tags: list[np.ndarray],
) -> None:
    """For each captured pose, reconstruct T_base_tag and report spread."""
    reconstructed = np.stack([
        fb @ T_flange_camera @ ct for fb, ct in zip(base_flanges, camera_tags)
    ])
    positions = reconstructed[:, :3, 3]
    mean_pos = positions.mean(axis=0)
    residuals = positions - mean_pos
    norms_mm = np.linalg.norm(residuals, axis=1) * 1000.0

    print("\n[calib] tag reconstruction consistency across poses:")
    print(f"  mean T_base_tag (mm): {np.round(mean_pos * 1000.0, 2)}")
    print(f"  per-pose residual (mm):")
    for i, n in enumerate(norms_mm):
        print(f"    pose {i + 1:>2}: {n:6.2f}")
    print(f"  mean residual: {norms_mm.mean():.2f} mm")
    print(f"  max  residual: {norms_mm.max():.2f} mm")
    if norms_mm.max() > 10.0:
        print("[calib] WARNING: max residual > 10 mm. Consider:")
        print("         - adding more pose diversity (especially rotation around different axes)")
        print("         - ensuring the tag did not move during capture")
        print("         - double-checking the wrist camera intrinsic calibration")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main() -> None:
    calib = np.load(WRIST_CALIB_FILE)
    K = calib["K"].astype(np.float64)
    dist = calib["dist"].astype(np.float64)
    size = (int(calib["image_size"][0]), int(calib["image_size"][1]))

    RobotKinematics = _probe_robot_kinematics()
    kinematics = RobotKinematics(
        urdf_path=URDF_PATH,
        target_frame_name=TARGET_FRAME,
        joint_names=FK_JOINT_NAMES,
    )

    SO101Follower, SO101FollowerConfig = _probe_so101_follower()
    robot = SO101Follower(SO101FollowerConfig(id=ROBOT_ID, port=ROBOT_PORT))
    robot.connect()
    print(f"[calib] robot connected on {ROBOT_PORT}")

    torque_off = _try_disable_torque(robot)
    if torque_off:
        print("[calib] torque disabled -- arm is free to pose by hand")
    else:
        print("[calib] WARNING: torque NOT disabled. You may be fighting the motors.")

    cap = _open_camera(WRIST_CAMERA_INDEX, size)
    detector = make_detector()

    # Pull in a previous calibration if it exists so the live viz is
    # geometrically correct. Otherwise fall back to the rough offset + tilt
    # above so the FK camera renders in a plausible spot (and orbits when the
    # wrist rolls, rather than spinning in place).
    try:
        T_flange_camera_viz = load_hand_eye_calib()
        viz_is_estimate = False
        print("[calib] loaded previous hand_eye_calib.npz for live viz")
    except FileNotFoundError:
        tilt = math.radians(VIZ_FALLBACK_CAMERA_TILT_X_DEG)
        c, s = math.cos(tilt), math.sin(tilt)
        R_fallback = np.array([
            [1.0, 0.0, 0.0],
            [0.0,   c,  -s],
            [0.0,   s,   c],
        ], dtype=np.float64)
        T_flange_camera_viz = np.eye(4, dtype=np.float64)
        T_flange_camera_viz[:3, :3] = R_fallback
        T_flange_camera_viz[:3, 3] = np.asarray(VIZ_FALLBACK_CAMERA_OFFSET_MM) / 1000.0
        viz_is_estimate = True
        print(
            f"[calib] no previous hand_eye_calib.npz; using fallback camera "
            f"offset {VIZ_FALLBACK_CAMERA_OFFSET_MM} mm, tilt "
            f"{VIZ_FALLBACK_CAMERA_TILT_X_DEG:+.1f} deg about +X for viz only"
        )

    # 3D arm viz window
    plt.ion()
    fig3d = plt.figure("SO-101 arm (live)", figsize=(7, 7))
    ax3d = fig3d.add_subplot(111, projection="3d")

    # Per-pose captures, kept parallel for calibrateHandEye + residual check.
    R_g2b: list[np.ndarray] = []
    t_g2b: list[np.ndarray] = []
    R_t2c: list[np.ndarray] = []
    t_t2c: list[np.ndarray] = []
    full_base_flanges: list[np.ndarray] = []
    full_camera_tags: list[np.ndarray] = []

    print(f"\n[calib] target: {REQUESTED_POSES} poses (minimum {MIN_POSES}).")
    print(f"[calib] pose the arm so tag id={TAG_ID} is visible in the wrist camera.")
    print(f"[calib] vary BOTH translation and orientation between poses for good conditioning.")
    print(f"[calib] controls: [SPACE]=capture  [ENTER]=solve  [R]=reset ref tag  [ESC]=abort\n")

    aborted = False
    frame_i = 0
    # Latched the first time we see the tag. The physical tag is stationary,
    # so once we set this we keep drawing it here; what moves in the viz is
    # the camera (both FK's claim and PnP's truth).
    anchor_tag_pose: tuple[int, np.ndarray] | None = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            # --- live FK for the 3D arm viz (runs every frame) ---
            live_joint_deg: np.ndarray | None = None
            live_link_poses: dict[str, np.ndarray] = {}
            live_T_base_flange: np.ndarray | None = None
            try:
                live_obs = robot.get_observation()
                live_joint_deg = np.asarray(
                    [float(live_obs[k]) for k in FK_JOINT_OBS_KEYS],
                    dtype=np.float64,
                )
                # forward_kinematics() updates placo's internal state so we can
                # then query any link's pose with get_T_world_frame(name).
                live_T_base_flange = np.asarray(
                    kinematics.forward_kinematics(live_joint_deg),
                    dtype=np.float64,
                )
                live_link_poses = _get_arm_link_poses(kinematics)
            except Exception as e:
                # Don't crash the loop over a transient bus read or FK hiccup;
                # the viz just won't update this frame.
                if frame_i % 60 == 0:
                    print(f"[calib] live FK/obs read hiccup: {e}")

            detections = detect_tags(frame, detector, K, dist, TAG_SIZE)
            hit = next((d for d in detections if d.id == TAG_ID), None)

            # Compute the two camera positions for the viz:
            #   FK cam: where T_base_flange @ T_flange_camera_viz says it is.
            #   PnP cam: where the camera must be if the tag is stationary at
            #           the anchor: T_base_tag_anchor @ inv(T_camera_tag).
            T_base_camera_fk: np.ndarray | None = None
            T_base_camera_pnp: np.ndarray | None = None
            if live_T_base_flange is not None:
                T_base_camera_fk = live_T_base_flange @ T_flange_camera_viz

            if hit is not None and live_T_base_flange is not None:
                # First sighting: latch the anchor. We use T_base_camera_fk as
                # our best initial guess of where the camera is, then back out
                # T_base_tag from that plus T_camera_tag.
                if anchor_tag_pose is None and T_base_camera_fk is not None:
                    T_base_tag_init = T_base_camera_fk @ hit.T_camera_tag
                    anchor_tag_pose = (hit.id, T_base_tag_init.copy())
                    print(f"[calib] tag anchor latched at "
                          f"{np.round(T_base_tag_init[:3, 3] * 1000.0, 1)} mm "
                          f"(id={hit.id})")

                if anchor_tag_pose is not None:
                    anchor_id, anchor_T = anchor_tag_pose
                    if hit.id == anchor_id:
                        T_base_camera_pnp = (
                            anchor_T @ _invert_transform(hit.T_camera_tag)
                        )

            render_overlay(frame, detections, K, dist, TAG_SIZE)
            n = len(R_g2b)
            color = (0, 255, 0) if hit else (0, 0, 255)
            status = (
                f"captures {n}/{REQUESTED_POSES} (min {MIN_POSES})   "
                f"tag {TAG_ID}: {'VISIBLE' if hit else 'MISSING'}   "
                f"[SPACE]=capture [ENTER]=solve [ESC]=abort"
            )
            cv2.putText(frame, status, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            cv2.imshow("hand-eye calibration", frame)

            if frame_i % VIZ_EVERY == 0:
                _refresh_arm_viz(
                    ax3d, live_link_poses, live_joint_deg, n,
                    anchor_tag=anchor_tag_pose,
                    T_base_camera_fk=T_base_camera_fk,
                    T_base_camera_pnp=T_base_camera_pnp,
                    tag_size_m=TAG_SIZE,
                    he_is_estimate=viz_is_estimate,
                )
                plt.pause(0.001)
            frame_i += 1

            key = cv2.waitKey(1) & 0xFF
            if key == 27:                   # ESC
                aborted = True
                break
            if key in (10, 13):             # ENTER
                if n < MIN_POSES:
                    print(f"[calib] need at least {MIN_POSES} poses, have {n}")
                    continue
                break
            if key in (ord("r"), ord("R")):
                anchor_tag_pose = None
                print("[calib] tag anchor cleared; will re-latch next sighting")
                continue
            if key != ord(" "):
                continue

            if hit is None:
                print("[calib] tag not visible, skipping")
                continue

            # Small settle so the user's hand is off before we read joints.
            time.sleep(STILLNESS_SETTLE_S)

            obs = robot.get_observation()
            try:
                joint_deg = np.asarray(
                    [float(obs[k]) for k in FK_JOINT_OBS_KEYS], dtype=np.float64
                )
            except KeyError as e:
                print(f"[calib] joint observation missing key {e}; skipping capture")
                continue
            T_base_flange = np.asarray(
                kinematics.forward_kinematics(joint_deg), dtype=np.float64
            )

            # Re-read one more camera frame so the capture pair is co-temporal
            # with the freshly-read joints (rather than several frames stale).
            ok2, frame2 = cap.read()
            if not ok2:
                print("[calib] post-settle camera read failed; skipping capture")
                continue
            detections2 = detect_tags(frame2, detector, K, dist, TAG_SIZE)
            hit2 = next((d for d in detections2 if d.id == TAG_ID), None)
            if hit2 is None:
                print("[calib] tag disappeared during settle; skipping capture")
                continue

            R_g2b.append(T_base_flange[:3, :3].copy())
            t_g2b.append(T_base_flange[:3, 3].copy())
            R_t2c.append(hit2.T_camera_tag[:3, :3].copy())
            t_t2c.append(hit2.T_camera_tag[:3, 3].copy())
            full_base_flanges.append(T_base_flange.copy())
            full_camera_tags.append(hit2.T_camera_tag.copy())

            print(f"[calib] captured pose {len(R_g2b)}  "
                  f"t_flange={np.round(T_base_flange[:3, 3] * 1000, 1)} mm  "
                  f"t_tag_cam={np.round(hit2.tvec.ravel() * 1000, 1)} mm")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        plt.close("all")

        # Make sure the arm is holding itself again before we hand it back.
        if torque_off:
            re_enabled = _try_enable_torque(robot)
            print(f"[calib] torque re-enabled: {re_enabled}")

    if aborted:
        print("[calib] aborted by user; nothing written")
        try:
            robot.disconnect()
        except Exception:
            pass
        return

    if len(R_g2b) < MIN_POSES:
        print(f"[calib] only {len(R_g2b)} captures (< {MIN_POSES}); aborting without writing.")
        try:
            robot.disconnect()
        except Exception:
            pass
        return

    # ----- solve -----
    print(f"\n[calib] running cv2.calibrateHandEye (Tsai) on {len(R_g2b)} pairs...")
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_g2b, t_g2b,
        R_t2c, t_t2c,
        method=cv2.CALIB_HAND_EYE_TSAI,
    )
    T_flange_camera = np.eye(4, dtype=np.float64)
    T_flange_camera[:3, :3] = R_cam2gripper
    T_flange_camera[:3, 3] = t_cam2gripper.ravel()

    print("\n[calib] T_flange_camera =")
    print(np.array2string(T_flange_camera, precision=5, suppress_small=True))
    t_mm = T_flange_camera[:3, 3] * 1000.0
    print(f"[calib] translation (mm): ({t_mm[0]:+.2f}, {t_mm[1]:+.2f}, {t_mm[2]:+.2f})")

    _residual_report(T_flange_camera, full_base_flanges, full_camera_tags)

    save_hand_eye_calib(T_flange_camera)
    print(f"\n[calib] wrote hand_eye_calib.npz")

    try:
        robot.disconnect()
    except Exception as e:
        print(f"[calib] robot disconnect warning: {e}")


if __name__ == "__main__":
    main()
