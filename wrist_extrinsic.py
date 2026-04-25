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
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pyvista as pv
import vtk

from camera_utils import disable_autofocus
from detector import TAG_SIZE_M, TagDetection, detect_tags, make_detector, render_overlay
from hand_eye_calib import load_hand_eye_calib
from iphone_extrinsic import (
    _invert_transform,
    _matrix_to_quat,
    _quat_to_matrix,
    average_quaternions,
)
from render_robot import (
    RobotCadScene,
    _fit_camera,
    _segments_to_polydata,
    load_robot_model,
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
    "moving_jaw_so101_v1_link",
]

# Running-stats window (how many recent T_tag_base samples to average).
STATS_WINDOW = 60

# 3D viz refresh cadence (max Hz; capped per-frame regardless of camera FPS).
# The CAD render is the most expensive thing in the loop, so we cap it
# independently from the camera/detect pipeline. ~15 Hz feels live but leaves
# >50 ms per frame for vision work.
VIZ_MAX_HZ = 15.0

# Background joint-reader poll rate (Hz). Lerobot's USB read is the dominant
# blocking cost; running it off the main loop unsticks the camera pipeline.
JOINT_POLL_HZ = 80.0

VIZ_FIT_MARGIN_M = 0.06       # padding around robot/tag in the live 3D view
VIZ_MIN_AXIS_SPAN_M = 0.18    # avoid excessive zoom before all points are known

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
    frame = cv2.rotate(frame, cv2.ROTATE_180)
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
# Background joint reader: keeps the slow USB get_observation() + placo FK +
# per-link queries OFF the camera/detect/render hot loop. Without this the
# main loop stalls for 20-40 ms per iteration just talking to the bus.
#
# Both forward_kinematics() and get_T_world_frame() touch placo's internal
# state, so they MUST run on the same thread that owns `kinematics`. We make
# the reader thread the sole owner of `kinematics` and only ever publish
# precomputed numpy arrays to the main thread.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WristJointSample:
    joint_deg: np.ndarray
    T_base_flange: np.ndarray
    link_poses_base: dict[str, np.ndarray]
    timestamp: float
    read_ms: float
    fk_ms: float


class WristJointReader:
    """Continuously poll robot + FK and keep only the newest snapshot."""

    def __init__(self, robot, kinematics, poll_hz: float = JOINT_POLL_HZ) -> None:
        self._robot = robot
        self._kinematics = kinematics
        self._min_dt = 1.0 / max(poll_hz, 1e-6)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="wrist-joint-reader", daemon=True
        )
        self._latest: WristJointSample | None = None

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)

    def latest(self) -> WristJointSample | None:
        with self._lock:
            return self._latest

    def _run(self) -> None:
        last_error_t = 0.0
        while not self._stop.is_set():
            loop_start = time.monotonic()
            try:
                read_start = time.monotonic()
                joint_deg = _read_joint_deg(self._robot)
                read_end = time.monotonic()
                T_base_flange = np.asarray(
                    self._kinematics.forward_kinematics(joint_deg),
                    dtype=np.float64,
                )
                link_poses = _get_arm_link_poses(self._kinematics)
                fk_end = time.monotonic()
                sample = WristJointSample(
                    joint_deg=joint_deg,
                    T_base_flange=T_base_flange,
                    link_poses_base=link_poses,
                    timestamp=fk_end,
                    read_ms=(read_end - read_start) * 1000.0,
                    fk_ms=(fk_end - read_end) * 1000.0,
                )
                with self._lock:
                    self._latest = sample
            except Exception as e:
                now = time.monotonic()
                if now - last_error_t > 1.0:
                    print(f"[wrist] joint/FK read failed: {e}")
                    last_error_t = now

            elapsed = time.monotonic() - loop_start
            if elapsed < self._min_dt:
                self._stop.wait(self._min_dt - elapsed)


# ---------------------------------------------------------------------------
# Stats window: average T_tag_base across the last STATS_WINDOW sightings.
# Rotations averaged via Markley quaternion mean (imported from iphone_extrinsic).
# ---------------------------------------------------------------------------
class TagBaseStats:
    """Rolling mean of T_tag_base over the last `window` sightings.

    The mean/std queries are O(N) eigh + reductions, so we cache results
    against the push counter and only recompute when new data has come in.
    """

    def __init__(self, window: int = STATS_WINDOW):
        self.window = window
        self.quats: deque[np.ndarray] = deque(maxlen=window)
        self.trans: deque[np.ndarray] = deque(maxlen=window)
        self._dirty_seq = 0
        self._cache_seq = -1
        self._cache_mean: np.ndarray | None = None
        self._cache_std_mm: np.ndarray | None = None

    def push(self, T_tag_base: np.ndarray) -> None:
        self.quats.append(_matrix_to_quat(T_tag_base[:3, :3]))
        self.trans.append(T_tag_base[:3, 3].copy())
        self._dirty_seq += 1

    def reset(self) -> None:
        self.quats.clear()
        self.trans.clear()
        self._dirty_seq += 1
        self._cache_seq = -1
        self._cache_mean = None
        self._cache_std_mm = None

    def count(self) -> int:
        return len(self.trans)

    def _refresh_cache(self) -> None:
        if self._cache_seq == self._dirty_seq:
            return
        if not self.trans:
            self._cache_mean = None
            self._cache_std_mm = None
        else:
            trans = np.stack(list(self.trans), axis=0)
            mean_q = average_quaternions(np.stack(list(self.quats), axis=0))
            mean_t = trans.mean(axis=0)
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = _quat_to_matrix(mean_q)
            T[:3, 3] = mean_t
            self._cache_mean = T
            self._cache_std_mm = trans.std(axis=0) * 1000.0
        self._cache_seq = self._dirty_seq

    def mean(self) -> np.ndarray | None:
        self._refresh_cache()
        return self._cache_mean

    def translation_std_mm(self) -> np.ndarray | None:
        self._refresh_cache()
        return self._cache_std_mm


# ---------------------------------------------------------------------------
# PyVista 3D viz (tag at origin, everything else in the tag frame)
# ---------------------------------------------------------------------------
def _tag_corners(T_tag_tag: np.ndarray, tag_size_m: float) -> np.ndarray:
    half = tag_size_m / 2.0
    local = np.array([
        [-half,  half, 0.0],
        [ half,  half, 0.0],
        [ half, -half, 0.0],
        [-half, -half, 0.0],
    ])
    return (T_tag_tag[:3, :3] @ local.T).T + T_tag_tag[:3, 3]


def _tag_mesh(tag_size_m: float) -> pv.PolyData:
    corners = _tag_corners(np.eye(4, dtype=np.float64), tag_size_m)
    faces = np.array([4, 0, 1, 2, 3], dtype=np.int64)
    return pv.PolyData(corners, faces=faces)


def _frame_segments(T: np.ndarray, scale: float) -> list[tuple[np.ndarray, np.ndarray]]:
    origin = T[:3, 3]
    return [(origin, origin + T[:3, i] * scale) for i in range(3)]


def _camera_frustum_segments(
    T: np.ndarray,
    depth: float = 0.05,
    scale: float = 0.03,
) -> list[tuple[np.ndarray, np.ndarray]]:
    apex = T[:3, 3]
    R = T[:3, :3]
    corners_cam = np.array([
        [-scale, -scale, depth],
        [ scale, -scale, depth],
        [ scale,  scale, depth],
        [-scale,  scale, depth],
    ])
    corners = (R @ corners_cam.T).T + apex
    segs = [(apex, c) for c in corners]
    segs.extend((corners[i], corners[(i + 1) % 4]) for i in range(4))
    return segs


class _PersistentLineOverlay:
    """Pre-allocated vtkPolyData line set; mutate points in place to update.

    Adding/removing VTK actors is dramatically slower than mutating an
    existing actor's polydata in place. We pre-allocate `num_segments`
    line segments and just rewrite the point array each viz frame. When
    fewer real segments exist than allocated, we collapse the extras to
    zero-length lines so they render invisibly.
    """

    def __init__(
        self,
        plotter: pv.Plotter,
        num_segments: int,
        color: str,
        line_width: float,
        name: str,
    ) -> None:
        self._num_segments = num_segments
        self._max_points = num_segments * 2

        points = np.zeros((self._max_points, 3), dtype=np.float64)
        lines = np.empty(num_segments * 3, dtype=np.int64)
        for i in range(num_segments):
            lines[3 * i] = 2
            lines[3 * i + 1] = 2 * i
            lines[3 * i + 2] = 2 * i + 1
        self._poly = pv.PolyData(points, lines=lines)
        self._actor = plotter.add_mesh(
            self._poly,
            color=color,
            line_width=line_width,
            render_lines_as_tubes=True,
            name=name,
            render=False,
        )
        self._actor.SetVisibility(False)
        self._visible = False

    def set_segments(
        self, segments: list[tuple[np.ndarray, np.ndarray]]
    ) -> None:
        n = min(len(segments), self._num_segments)
        if n == 0:
            self.hide()
            return
        flat = np.empty((self._max_points, 3), dtype=np.float64)
        for i in range(n):
            flat[2 * i] = segments[i][0]
            flat[2 * i + 1] = segments[i][1]
        if n < self._num_segments:
            # Collapse leftover lines to a zero-length point at the last
            # visible vertex so they don't draw.
            flat[2 * n :] = flat[2 * n - 1]
        self._poly.points = flat
        self._poly.Modified()
        if not self._visible:
            self._actor.SetVisibility(True)
            self._visible = True

    def hide(self) -> None:
        if self._visible:
            self._actor.SetVisibility(False)
            self._visible = False


class WristExtrinsicScene:
    """PyVista scene for the wrist extrinsic tag-frame visualization."""

    def __init__(
        self,
        urdf_path: str,
        tag_size_m: float,
        max_triangles_per_visual: int = 1500,
        show_cad: bool = True,
    ) -> None:
        self.plotter = pv.Plotter(window_size=(900, 800), title="robot in tag frame")
        self.plotter.set_background("white")
        self.plotter.add_axes()
        self.plotter.show_grid(color="lightgray")
        # show_grid creates a vtkCubeAxesActor with default bounds (-1, +1) m
        # on every axis. Any subsequent VTK ResetCamera() (e.g. the camera
        # orientation widget calls one when its snap animation ends) would
        # otherwise blow parallel_scale up to ~sqrt(3) m and shrink the
        # robot to a postage stamp. Excluding the grid from bounds keeps
        # ResetCamera() honest while still drawing the grid.
        self._cube_axes_actor = self._find_cube_axes_actor()
        if self._cube_axes_actor is not None:
            self._cube_axes_actor.SetUseBounds(False)
        # Screen convention for this view: Z is vertical, Y is horizontal, and
        # X is depth toward/away from the viewer.
        self.plotter.camera_position = [
            (0.45, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, -1.0),
        ]

        self.robot_cad: RobotCadScene | None = None
        if show_cad:
            model = load_robot_model(Path(urdf_path).resolve())
            self.robot_cad = RobotCadScene(
                self.plotter,
                model,
                max_triangles_per_visual=max_triangles_per_visual,
            )
        self._fit_done = False
        self._fit_requested = False
        self._closed = False
        self._correcting_camera = False
        self._view_distance_m = 0.45
        # Cached state from the last successful auto-fit. snap_view() and the
        # orientation widget's EndInteractionEvent re-use these to reproduce
        # the tight fit instead of falling back to VTK defaults.
        self._latest_fit_points: list[np.ndarray] | None = None
        self._fit_focal: np.ndarray | None = None
        self._fit_parallel_scale: float | None = None

        self.plotter.add_mesh(
            _tag_mesh(tag_size_m),
            color="cornflowerblue",
            opacity=0.45,
            show_edges=True,
            edge_color="navy",
            name="reference_tag",
        )

        # Static origin triad at the tag frame -- never changes, build once.
        self.plotter.add_mesh(
            _segments_to_polydata(
                _frame_segments(np.eye(4, dtype=np.float64), TAG_SIZE * 0.8)
            ),
            color="black",
            line_width=2.0,
            render_lines_as_tubes=True,
            name="tag_origin_triad",
            render=False,
        )

        # Persistent dynamic overlays. Pre-allocated to their maximum
        # segment count so we never have to add/remove VTK actors during
        # the live loop -- only mutate point arrays in place.
        self._live_axes = _PersistentLineOverlay(
            self.plotter, num_segments=3, color="black", line_width=3.0,
            name="live_T_tag_base_axes",
        )
        self._mean_axes = _PersistentLineOverlay(
            self.plotter, num_segments=3, color="gray", line_width=2.0,
            name="mean_T_tag_base_axes",
        )
        # Arm chain: max segments = links - 1.
        self._arm_chain = _PersistentLineOverlay(
            self.plotter,
            num_segments=max(1, len(ARM_LINK_CHAIN) - 1),
            color="gray",
            line_width=4.0,
            name="arm_chain",
        )
        # Camera frustum: 4 apex-to-corner + 4 corner-to-corner = 8.
        self._frustum = _PersistentLineOverlay(
            self.plotter, num_segments=8, color="crimson", line_width=2.0,
            name="wrist_camera_frustum",
        )

        # Persistent text actor; SetInput() is much cheaper than rebuilding
        # via plotter.add_text() every viz frame.
        self._status_actor = self.plotter.add_text(
            "",
            position="upper_left",
            font_size=10,
            name="status",
        )

        self.plotter.add_key_event("q", self.close)
        self.plotter.add_key_event("Escape", self.close)
        self.plotter.add_key_event("x", lambda: self.snap_view("+x"))
        self.plotter.add_key_event("X", lambda: self.snap_view("-x"))
        self.plotter.add_key_event("y", lambda: self.snap_view("+y"))
        self.plotter.add_key_event("Y", lambda: self.snap_view("-y"))
        self.plotter.add_key_event("z", lambda: self.snap_view("-z"))
        self.plotter.add_key_event("Z", lambda: self.snap_view("+z"))
        self.plotter.show(interactive_update=True, auto_close=False)
        self._install_camera_roll_corrector()
        self._add_view_cube()

    def _find_cube_axes_actor(self):
        try:
            collection = self.plotter.renderer.GetActors()
            collection.InitTraversal()
            for _ in range(collection.GetNumberOfItems()):
                actor = collection.GetNextActor()
                if actor is None:
                    break
                try:
                    if actor.IsA("vtkCubeAxesActor"):
                        return actor
                except Exception:
                    continue
        except Exception:
            pass
        return None

    @staticmethod
    def _snap_views() -> dict[str, tuple[np.ndarray, np.ndarray]]:
        return {
            "+x": (np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, -1.0])),
            "-x": (np.array([-1.0, 0.0, 0.0]), np.array([0.0, 0.0, -1.0])),
            "+y": (np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, -1.0])),
            "-y": (np.array([0.0, -1.0, 0.0]), np.array([0.0, 0.0, -1.0])),
            "+z": (np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0])),
            "-z": (np.array([0.0, 0.0, -1.0]), np.array([0.0, 1.0, 0.0])),
        }

    def _install_camera_roll_corrector(self) -> None:
        try:
            self.plotter.camera.AddObserver("ModifiedEvent", self._on_camera_modified)
        except Exception as e:
            print(f"[wrist] camera roll corrector unavailable: {e}")

    def _on_camera_modified(self, *_args) -> None:
        if self._correcting_camera:
            return
        self._correct_axis_aligned_camera_roll()

    def _correct_axis_aligned_camera_roll(self) -> None:
        """Re-roll VTK view-cube snaps to match the AprilTag axis convention."""
        camera = self.plotter.camera
        position = np.asarray(camera.GetPosition(), dtype=np.float64)
        focal = np.asarray(camera.GetFocalPoint(), dtype=np.float64)
        view_from = position - focal
        norm = float(np.linalg.norm(view_from))
        if norm <= 1e-9:
            return
        view_from /= norm

        for snap_direction, desired_up in self._snap_views().values():
            if float(np.dot(view_from, snap_direction)) < 0.995:
                continue
            corrected_up = desired_up - snap_direction * float(np.dot(desired_up, snap_direction))
            up_norm = float(np.linalg.norm(corrected_up))
            if up_norm <= 1e-9:
                return
            corrected_up /= up_norm

            current_up = np.asarray(camera.GetViewUp(), dtype=np.float64)
            if float(np.dot(current_up, corrected_up)) > 0.999:
                return

            self._correcting_camera = True
            try:
                camera.SetViewUp(tuple(corrected_up))
                camera.OrthogonalizeViewUp()
            finally:
                self._correcting_camera = False
            return

    def _add_view_cube(self) -> None:
        """Add a CAD-style orientation cube when the local VTK build supports it."""
        try:
            widget = vtk.vtkCameraOrientationWidget()
            widget.SetParentRenderer(self.plotter.renderer)
            widget.On()
            # The widget's snap animation ends with a ResetCamera() call. Even
            # with the grid removed from bounds, that gives a slightly looser
            # fit than our parallel-aware auto-fit, so trigger a re-fit on the
            # next update.
            try:
                widget.AddObserver("EndInteractionEvent", self._on_widget_end_interaction)
            except Exception as e:
                print(f"[wrist] orientation widget end-interaction observer unavailable: {e}")
            self._view_cube_widget = widget
            return
        except Exception as e:
            print(f"[wrist] VTK camera orientation widget unavailable: {e}")

        try:
            cube = pv.Cube()
            self.plotter.add_orientation_widget(
                cube,
                viewport=(0.82, 0.02, 0.98, 0.18),
            )
        except Exception as e:
            print(f"[wrist] fallback orientation widget unavailable: {e}")

    def _on_widget_end_interaction(self, *_args) -> None:
        # The orientation widget kicks off its animation right after this
        # event fires, so deferring the re-fit to the next update() lets us
        # apply our scale on the post-animation state.
        self._fit_requested = True

    def close(self) -> None:
        self._closed = True

    def request_fit(self) -> None:
        self._fit_requested = True

    def snap_view(self, view: str) -> None:
        """Snap around the cached scene center with Z kept vertical.

        Falls back to the AprilTag/world origin only before the first auto-fit.
        Re-runs the auto-fit on the next update() so the new view direction
        gets a tight parallel scale.
        """
        direction, view_up = self._snap_views()[view]
        if self._fit_focal is not None:
            focal = self._fit_focal.astype(np.float64, copy=True)
        else:
            focal = np.zeros(3, dtype=np.float64)
        position = focal + direction * self._view_distance_m
        self.plotter.camera_position = [
            tuple(position),
            tuple(focal),
            tuple(view_up),
        ]
        self._fit_requested = True
        self.plotter.update()

    def is_closed(self) -> bool:
        if self._closed:
            return True
        try:
            return bool(self.plotter.iren is not None and self.plotter.iren.interactor.GetDone())
        except Exception:
            return False

    def shutdown(self) -> None:
        try:
            self.plotter.close()
        except Exception:
            pass

    def update(
        self,
        T_tag_base: np.ndarray | None,
        T_tag_camera: np.ndarray | None,
        tag_frame_link_poses: dict[str, np.ndarray],
        mean_T_tag_base: np.ndarray | None,
        std_mm: np.ndarray | None,
        sample_count: int,
        joint_deg: np.ndarray | None,
    ) -> None:
        robot_fit_points: list[np.ndarray] = []
        if self.robot_cad is not None:
            robot_fit_points = self.robot_cad.update(
                tag_frame_link_poses,
                auto_fit_once=False,
            )

        fit_points: list[np.ndarray] = [np.zeros(3, dtype=np.float64)]
        fit_points.extend(_tag_corners(np.eye(4, dtype=np.float64), TAG_SIZE))
        fit_points.extend(robot_fit_points)

        if T_tag_base is not None:
            fit_points.append(T_tag_base[:3, 3])
            self._live_axes.set_segments(_frame_segments(T_tag_base, 0.08))
        else:
            self._live_axes.hide()

        if mean_T_tag_base is not None:
            fit_points.append(mean_T_tag_base[:3, 3])
            self._mean_axes.set_segments(_frame_segments(mean_T_tag_base, 0.06))
        else:
            self._mean_axes.hide()

        chain_positions: list[np.ndarray] = []
        for name in ARM_LINK_CHAIN:
            T = tag_frame_link_poses.get(name)
            if T is None:
                continue
            chain_positions.append(T[:3, 3])
            fit_points.append(T[:3, 3])
        if len(chain_positions) >= 2:
            self._arm_chain.set_segments(
                [
                    (chain_positions[i], chain_positions[i + 1])
                    for i in range(len(chain_positions) - 1)
                ]
            )
        else:
            self._arm_chain.hide()

        if T_tag_camera is not None:
            fit_points.append(T_tag_camera[:3, 3])
            self._frustum.set_segments(_camera_frustum_segments(T_tag_camera))
        else:
            self._frustum.hide()

        self._latest_fit_points = list(fit_points)

        if self._fit_requested or (not self._fit_done and robot_fit_points):
            # zoom=1.0 + 20 mm margin keeps both the AprilTag (origin) and the
            # robot fully on-screen. zoom>1 would shrink parallel_scale below
            # the constrained-axis span and crop whichever subject is at the
            # edge -- usually the tag, which is small and far from the robot.
            _fit_camera(
                self.plotter,
                fit_points,
                margin_m=0.02,
                zoom=1.0,
                parallel=True,
            )
            cam = self.plotter.camera
            cam_pos = np.asarray(cam.position, dtype=np.float64)
            self._view_distance_m = max(float(np.linalg.norm(cam_pos)), 0.08)
            self._fit_focal = np.asarray(cam.GetFocalPoint(), dtype=np.float64)
            try:
                self._fit_parallel_scale = float(cam.GetParallelScale())
            except Exception:
                self._fit_parallel_scale = None
            self._fit_done = True
            self._fit_requested = False

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
        lines.append("")
        lines.append("q/ESC: quit  |  r: reset/refit")
        lines.append("snap: x=front/depth view, y=side view, z=top view")
        # Mutate the existing text actor in place rather than rebuilding via
        # plotter.add_text() each viz frame.
        try:
            self._status_actor.SetInput("\n".join(lines))
        except Exception:
            self._status_actor = self.plotter.add_text(
                "\n".join(lines),
                position="upper_left",
                font_size=10,
                name="status",
            )
        self.plotter.update()


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
    parser.add_argument(
        "--max-triangles-per-visual",
        type=int,
        default=1500,
        help="mesh decimation cap per URDF visual; 0 keeps full-resolution STL assets",
    )
    parser.add_argument(
        "--full-mesh",
        action="store_true",
        help="render full-resolution STL assets instead of simplified meshes",
    )
    parser.add_argument(
        "--no-cad",
        action="store_true",
        help="hide the CAD mesh and show only tag-frame lines plus the wrist camera frustum",
    )
    parser.add_argument(
        "--preview-every",
        type=int,
        default=1,
        help="show the camera preview every N frames; detection still runs every frame",
    )
    parser.add_argument(
        "--preview-scale",
        type=float,
        default=1.0,
        help="scale factor for displayed camera preview only",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="disable the OpenCV camera preview window",
    )
    parser.add_argument(
        "--detector-decimate",
        type=float,
        default=2.0,
        help=(
            "AprilTag detector decimation factor (>=1.0). At 1920x1080, "
            "decimate=2.0 roughly halves detection time with no observable "
            "accuracy loss for the wrist tag setup. Use 1.0 for full-res."
        ),
    )
    parser.add_argument(
        "--viz-hz",
        type=float,
        default=VIZ_MAX_HZ,
        help=(
            "max 3D viz refresh rate in Hz; the CAD render is the most "
            "expensive thing in the loop and capping it independently keeps "
            "the camera/detect pipeline fast."
        ),
    )
    parser.add_argument(
        "--joint-poll-hz",
        type=float,
        default=JOINT_POLL_HZ,
        help="background joint+FK polling rate (Hz) for the off-loop reader",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="print per-stage timing (cap/detect/viz/preview) every second",
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

    detector_decimate = max(1.0, float(args.detector_decimate))
    detector = make_detector(decimate=detector_decimate)
    print(
        f"[wrist] AprilTag detector decimate={detector_decimate:.2f} "
        f"(higher = faster, lower = more accurate; 2.0 is a good default at 1080p)"
    )
    cap = _open_camera(args.camera_index, size)
    print(f"[wrist] opened wrist camera idx={args.camera_index} size={size}")
    print(
        f"[wrist] treating tag id={args.tag_id} as world origin; "
        f"stats window = {args.stats_window} samples"
    )
    print("[wrist] controls: [q]/[ESC]=quit   [r]=reset running stats")

    preview_every = max(1, int(args.preview_every))
    preview_scale = max(0.05, float(args.preview_scale))
    if not args.no_preview:
        cv2.namedWindow(PREVIEW_WINDOW, cv2.WINDOW_AUTOSIZE)

    max_triangles_per_visual = 0 if args.full_mesh else args.max_triangles_per_visual
    scene = WristExtrinsicScene(
        URDF_PATH,
        TAG_SIZE,
        max_triangles_per_visual=max_triangles_per_visual,
        show_cad=not args.no_cad,
    )

    # Background joint+FK reader. With this off the main loop, the camera
    # pipeline no longer stalls on the 15-30 ms USB read per iteration.
    joint_reader = WristJointReader(robot, kinematics, poll_hz=args.joint_poll_hz)
    joint_reader.start()
    print(f"[wrist] background joint+FK polling started at {args.joint_poll_hz:.1f} Hz")

    stats = TagBaseStats(window=args.stats_window)
    fps_window: deque[float] = deque(maxlen=30)
    last_print_t = time.monotonic()
    frame_i = 0

    viz_min_dt = 1.0 / max(args.viz_hz, 1e-6)
    last_viz_t = 0.0

    # Per-stage timing accumulators for --profile.
    prof_n = 0
    prof_cap_ms = 0.0
    prof_detect_ms = 0.0
    prof_viz_ms = 0.0
    prof_preview_ms = 0.0
    prof_loop_ms = 0.0

    try:
        while True:
            loop_start = time.monotonic()

            t_cap = time.monotonic()
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            cap_ms = (time.monotonic() - t_cap) * 1000.0

            # --- joints + FK from the background reader ----------------------
            joint_sample = joint_reader.latest()
            joint_deg: np.ndarray | None = None
            T_base_flange: np.ndarray | None = None
            link_poses_base: dict[str, np.ndarray] = {}
            if joint_sample is not None:
                joint_deg = joint_sample.joint_deg
                T_base_flange = joint_sample.T_base_flange
                link_poses_base = joint_sample.link_poses_base

            # --- tag detection ---------------------------------------------
            t_det = time.monotonic()
            detections: list[TagDetection] = detect_tags(
                frame, detector, K, dist, TAG_SIZE
            )
            detect_ms = (time.monotonic() - t_det) * 1000.0
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

            # mean/std are O(N) so fetch lazily; the cache short-circuits when
            # nothing's been pushed since last call.
            preview_ms = 0.0
            if not args.no_preview and frame_i % preview_every == 0:
                t_prev = time.monotonic()
                mean_T_for_preview = stats.mean()
                std_mm_for_preview = stats.translation_std_mm()
                preview = frame.copy()
                render_overlay(preview, detections, K, dist, TAG_SIZE)
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
                    preview, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
                )
                if mean_T_for_preview is not None and std_mm_for_preview is not None:
                    line2 = (
                        f"mean_T_tag_base.t = {_format_xyz_mm(mean_T_for_preview[:3, 3])}   "
                        f"std = ({std_mm_for_preview[0]:5.2f},"
                        f"{std_mm_for_preview[1]:5.2f},{std_mm_for_preview[2]:5.2f}) mm"
                    )
                    cv2.putText(
                        preview, line2, (10, 56),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 0), 2,
                    )
                if preview_scale != 1.0:
                    preview = cv2.resize(
                        preview,
                        None,
                        fx=preview_scale,
                        fy=preview_scale,
                        interpolation=cv2.INTER_AREA,
                    )
                cv2.imshow(PREVIEW_WINDOW, preview)
                preview_ms = (time.monotonic() - t_prev) * 1000.0

            # --- 3D viz in tag frame (rate-limited; CAD render is expensive)
            viz_ms = 0.0
            now = time.monotonic()
            if now - last_viz_t >= viz_min_dt:
                t_viz = time.monotonic()
                # Express each arm link in the tag frame for drawing.
                link_poses_tag: dict[str, np.ndarray] = {}
                if T_tag_base is not None:
                    for name, T_base_link in link_poses_base.items():
                        link_poses_tag[name] = T_tag_base @ T_base_link
                scene.update(
                    T_tag_base=T_tag_base,
                    T_tag_camera=T_tag_camera,
                    tag_frame_link_poses=link_poses_tag,
                    mean_T_tag_base=stats.mean(),
                    std_mm=stats.translation_std_mm(),
                    sample_count=stats.count(),
                    joint_deg=joint_deg,
                )
                if scene.is_closed():
                    break
                viz_ms = (time.monotonic() - t_viz) * 1000.0
                last_viz_t = now

            # --- periodic console print ------------------------------------
            dt = time.monotonic() - loop_start
            fps_window.append(1.0 / max(dt, 1e-6))
            prof_n += 1
            prof_cap_ms += cap_ms
            prof_detect_ms += detect_ms
            prof_viz_ms += viz_ms
            prof_preview_ms += preview_ms
            prof_loop_ms += dt * 1000.0
            if time.monotonic() - last_print_t > 1.0:
                mean_T_for_print = stats.mean()
                std_mm_for_print = stats.translation_std_mm()
                if T_tag_base is not None:
                    extra = (
                        f"T_tag_base.t={_format_xyz_mm(T_tag_base[:3, 3])}"
                    )
                    if mean_T_for_print is not None and std_mm_for_print is not None:
                        extra += (
                            f"  mean={_format_xyz_mm(mean_T_for_print[:3, 3])}"
                            f"  std=({std_mm_for_print[0]:5.2f},"
                            f"{std_mm_for_print[1]:5.2f},"
                            f"{std_mm_for_print[2]:5.2f}) mm"
                        )
                else:
                    extra = "tag not visible"
                print(
                    f"[wrist] frame {frame_i} | "
                    f"FPS {np.mean(fps_window):4.1f} | "
                    f"samples {stats.count():3d} | {extra}"
                )
                if args.profile and prof_n > 0:
                    inv_n = 1.0 / prof_n
                    print(
                        f"[wrist][profile] avg over {prof_n:3d} frames | "
                        f"loop {prof_loop_ms * inv_n:5.1f} ms | "
                        f"cap {prof_cap_ms * inv_n:5.1f} | "
                        f"detect {prof_detect_ms * inv_n:5.1f} | "
                        f"viz {prof_viz_ms * inv_n:5.1f} | "
                        f"preview {prof_preview_ms * inv_n:5.1f}"
                    )
                    prof_n = 0
                    prof_cap_ms = 0.0
                    prof_detect_ms = 0.0
                    prof_viz_ms = 0.0
                    prof_preview_ms = 0.0
                    prof_loop_ms = 0.0
                last_print_t = time.monotonic()

            frame_i += 1

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in (ord("r"), ord("R")):
                stats.reset()
                scene.request_fit()
                print("[wrist] running stats reset")
    finally:
        try:
            joint_reader.stop()
        except Exception:
            pass
        try:
            cap.release()
        except Exception:
            pass
        cv2.destroyAllWindows()
        scene.shutdown()

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
