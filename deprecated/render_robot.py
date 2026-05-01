"""Live SO-101 CAD renderer using PyVista/VTK.

Connects to the follower arm, reads the current six motor observations, and
renders the URDF visual meshes in an OpenGL-accelerated 3D viewer.

Usage:
    python render_robot.py
    python render_robot.py --disable-torque       # hand-pose the arm
    python render_robot.py --rate-hz 60

Controls:
    q/ESC : quit the PyVista viewer
"""

from __future__ import annotations

import argparse
import math
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv
import vtk


ROBOT_PORT = "/dev/tty.usbmodem5AE60557941"
ROBOT_ID = "my_awesome_follower_arm"
URDF_PATH = "SO101/so101_new_calib.urdf"

ARM_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]
GRIPPER_JOINT_NAME = "gripper"
JOINT_NAMES = [*ARM_JOINT_NAMES, GRIPPER_JOINT_NAME]

DEFAULT_MATERIAL_RGBA = (0.8, 0.8, 0.8, 1.0)
PRINT_EVERY_S = 1.0
STATUS_EVERY_N_FRAMES = 10


@dataclass(frozen=True)
class RenderRobotConfig:
    port: str = ROBOT_PORT
    robot_id: str = ROBOT_ID
    urdf: str = URDF_PATH
    rate_hz: float = 60.0
    poll_hz: float = 100.0
    sync_reads: bool = False
    disable_torque: bool = False
    restore_torque_on_exit: bool = False
    gripper_units: str = "percent"
    invert_gripper: bool = False
    max_triangles_per_visual: int = 1500
    full_mesh: bool = False
    no_auto_fit: bool = False
    show_skeleton: bool = False
    show_joint_axes: bool = False
    verbose: bool = False


@dataclass(frozen=True)
class JointSpec:
    name: str
    joint_type: str
    parent: str
    child: str
    T_parent_joint: np.ndarray
    axis: np.ndarray
    lower: float | None = None
    upper: float | None = None


@dataclass(frozen=True)
class VisualSpec:
    link_name: str
    mesh_path: Path
    T_link_visual: np.ndarray
    rgba: tuple[float, float, float, float]


@dataclass
class VisualActor:
    link_name: str
    T_link_visual: np.ndarray
    actor: Any
    local_bbox_corners: np.ndarray


@dataclass(frozen=True)
class JointSample:
    joint_positions_rad: dict[str, float]
    raw_values: dict[str, float]
    timestamp: float
    read_ms: float
    seq: int


class RobotModel:
    def __init__(
        self,
        base_link: str,
        joints: list[JointSpec],
        visuals: list[VisualSpec],
    ) -> None:
        self.base_link = base_link
        self.joints = joints
        self.visuals = visuals
        self.joints_by_name = {joint.name: joint for joint in joints}
        self.children_by_parent: dict[str, list[JointSpec]] = {}
        for joint in joints:
            self.children_by_parent.setdefault(joint.parent, []).append(joint)

    def forward_kinematics(
        self,
        joint_positions_rad: dict[str, float],
    ) -> dict[str, np.ndarray]:
        """Return T_base_link for every reachable link in the URDF tree."""
        poses = {self.base_link: np.eye(4, dtype=np.float64)}
        queue = [self.base_link]

        while queue:
            parent = queue.pop(0)
            T_base_parent = poses[parent]
            for joint in self.children_by_parent.get(parent, []):
                T_parent_child = joint.T_parent_joint.copy()
                if joint.joint_type in {"revolute", "continuous"}:
                    q = float(joint_positions_rad.get(joint.name, 0.0))
                    T_parent_child = T_parent_child @ _axis_angle_transform(joint.axis, q)
                elif joint.joint_type != "fixed":
                    raise ValueError(
                        f"unsupported joint type {joint.joint_type!r} for {joint.name}"
                    )

                poses[joint.child] = T_base_parent @ T_parent_child
                queue.append(joint.child)

        return poses


def _probe_so101_follower():
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    return SO101Follower, SO101FollowerConfig


def _try_disable_torque(robot) -> bool:
    try:
        robot.bus.disable_torque()
        return True
    except Exception as e:
        print(f"[render] robot.bus.disable_torque() failed: {e}")
    try:
        for motor in robot.bus.motors:
            robot.bus.write("Torque_Enable", motor, 0)
        return True
    except Exception as e:
        print(f"[render] per-motor Torque_Enable=0 failed: {e}")
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


def _parse_xyz(text: str | None, default: tuple[float, float, float]) -> np.ndarray:
    if not text:
        return np.asarray(default, dtype=np.float64)
    return np.asarray([float(v) for v in text.split()], dtype=np.float64)


def _rotation_x(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ],
        dtype=np.float64,
    )


def _rotation_y(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=np.float64,
    )


def _rotation_z(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _transform_from_xyz_rpy(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    roll, pitch, yaw = [float(v) for v in rpy]
    T[:3, :3] = _rotation_z(yaw) @ _rotation_y(pitch) @ _rotation_x(roll)
    T[:3, 3] = xyz
    return T


def _axis_angle_transform(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-12:
        R = np.eye(3, dtype=np.float64)
    else:
        x, y, z = axis / norm
        c = math.cos(angle)
        s = math.sin(angle)
        C = 1.0 - c
        R = np.array(
            [
                [x * x * C + c, x * y * C - z * s, x * z * C + y * s],
                [y * x * C + z * s, y * y * C + c, y * z * C - x * s],
                [z * x * C - y * s, z * y * C + x * s, z * z * C + c],
            ],
            dtype=np.float64,
        )
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    return T


def _transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    return (T[:3, :3] @ points.T).T + T[:3, 3]


def _parse_materials(root: ET.Element) -> dict[str, tuple[float, float, float, float]]:
    materials: dict[str, tuple[float, float, float, float]] = {}
    for material in root.findall("material"):
        name = material.get("name")
        color = material.find("color")
        rgba_text = color.get("rgba") if color is not None else None
        if name and rgba_text:
            rgba = tuple(float(v) for v in rgba_text.split())
            if len(rgba) == 4:
                materials[name] = rgba  # type: ignore[assignment]
    return materials


def _visual_rgba(
    visual: ET.Element,
    materials: dict[str, tuple[float, float, float, float]],
) -> tuple[float, float, float, float]:
    material = visual.find("material")
    if material is None:
        return DEFAULT_MATERIAL_RGBA

    color = material.find("color")
    if color is not None and color.get("rgba"):
        rgba = tuple(float(v) for v in color.get("rgba", "").split())
        if len(rgba) == 4:
            return rgba  # type: ignore[return-value]

    name = material.get("name")
    if name and name in materials:
        return materials[name]
    return DEFAULT_MATERIAL_RGBA


def load_robot_model(urdf_path: Path) -> RobotModel:
    root = ET.parse(urdf_path).getroot()
    materials = _parse_materials(root)
    urdf_dir = urdf_path.parent

    link_names = {link.get("name", "") for link in root.findall("link")}
    child_links: set[str] = set()
    joints: list[JointSpec] = []
    visuals: list[VisualSpec] = []

    for link in root.findall("link"):
        link_name = link.get("name")
        if not link_name:
            continue
        for visual in link.findall("visual"):
            mesh = visual.find("geometry/mesh")
            filename = mesh.get("filename") if mesh is not None else None
            if not filename:
                continue
            origin = visual.find("origin")
            xyz = _parse_xyz(
                origin.get("xyz") if origin is not None else None,
                (0.0, 0.0, 0.0),
            )
            rpy = _parse_xyz(
                origin.get("rpy") if origin is not None else None,
                (0.0, 0.0, 0.0),
            )
            visuals.append(
                VisualSpec(
                    link_name=link_name,
                    mesh_path=(urdf_dir / filename).resolve(),
                    T_link_visual=_transform_from_xyz_rpy(xyz, rpy),
                    rgba=_visual_rgba(visual, materials),
                )
            )

    for joint in root.findall("joint"):
        name = joint.get("name")
        joint_type = joint.get("type", "fixed")
        parent_el = joint.find("parent")
        child_el = joint.find("child")
        parent = parent_el.get("link") if parent_el is not None else None
        child = child_el.get("link") if child_el is not None else None
        if not name or not parent or not child:
            continue

        origin = joint.find("origin")
        xyz = _parse_xyz(
            origin.get("xyz") if origin is not None else None,
            (0.0, 0.0, 0.0),
        )
        rpy = _parse_xyz(
            origin.get("rpy") if origin is not None else None,
            (0.0, 0.0, 0.0),
        )
        axis_el = joint.find("axis")
        axis = _parse_xyz(axis_el.get("xyz") if axis_el is not None else None, (0.0, 0.0, 1.0))

        limit_el = joint.find("limit")
        lower = float(limit_el.get("lower")) if limit_el is not None and limit_el.get("lower") else None
        upper = float(limit_el.get("upper")) if limit_el is not None and limit_el.get("upper") else None

        joints.append(
            JointSpec(
                name=name,
                joint_type=joint_type,
                parent=parent,
                child=child,
                T_parent_joint=_transform_from_xyz_rpy(xyz, rpy),
                axis=axis,
                lower=lower,
                upper=upper,
            )
        )
        child_links.add(child)

    base_link = "base_link" if "base_link" in link_names else next(iter(link_names - child_links))
    return RobotModel(base_link=base_link, joints=joints, visuals=visuals)


def _mesh_bbox_corners(mesh: pv.DataSet) -> np.ndarray:
    xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
    return np.array(
        [
            [xmin, ymin, zmin],
            [xmin, ymin, zmax],
            [xmin, ymax, zmin],
            [xmin, ymax, zmax],
            [xmax, ymin, zmin],
            [xmax, ymin, zmax],
            [xmax, ymax, zmin],
            [xmax, ymax, zmax],
        ],
        dtype=np.float64,
    )


def _make_vtk_matrix(T: np.ndarray) -> vtk.vtkMatrix4x4:
    matrix = vtk.vtkMatrix4x4()
    for r in range(4):
        for c in range(4):
            matrix.SetElement(r, c, float(T[r, c]))
    return matrix


def _set_actor_transform(actor: Any, T: np.ndarray) -> None:
    # VTK expects a homogeneous matrix in world coordinates for the actor.
    actor.SetUserMatrix(_make_vtk_matrix(T))


def build_visual_actors(
    plotter: pv.Plotter,
    model: RobotModel,
    max_triangles_per_visual: int,
) -> list[VisualActor]:
    mesh_cache: dict[Path, pv.DataSet] = {}
    actors: list[VisualActor] = []

    for visual in model.visuals:
        if visual.mesh_path not in mesh_cache:
            mesh = pv.read(visual.mesh_path)
            if max_triangles_per_visual > 0 and mesh.n_cells > max_triangles_per_visual:
                target_reduction = 1.0 - max_triangles_per_visual / float(mesh.n_cells)
                try:
                    mesh = mesh.decimate(target_reduction, inplace=False)
                except Exception as e:
                    print(f"[render] decimate failed for {visual.mesh_path.name}: {e}")
            mesh_cache[visual.mesh_path] = mesh

        mesh = mesh_cache[visual.mesh_path]
        rgba = visual.rgba
        actor = plotter.add_mesh(
            mesh,
            color=rgba[:3],
            opacity=rgba[3],
            smooth_shading=False,
            show_edges=False,
            name=f"{visual.link_name}:{visual.mesh_path.name}:{len(actors)}",
        )
        actors.append(
            VisualActor(
                link_name=visual.link_name,
                T_link_visual=visual.T_link_visual,
                actor=actor,
                local_bbox_corners=_mesh_bbox_corners(mesh),
            )
        )

    return actors


def _map_gripper_observation(
    raw_value: float,
    joint: JointSpec,
    units: str,
    invert: bool,
) -> float:
    if units == "rad":
        return raw_value
    if units == "deg":
        return math.radians(raw_value)

    lower = joint.lower if joint.lower is not None else 0.0
    upper = joint.upper if joint.upper is not None else 1.0
    pct = float(np.clip(raw_value, 0.0, 100.0)) / 100.0
    if invert:
        pct = 1.0 - pct
    return lower + pct * (upper - lower)


def read_joint_positions(
    robot,
    model: RobotModel,
    gripper_units: str,
    invert_gripper: bool,
) -> tuple[dict[str, float], dict[str, float]]:
    obs = robot.get_observation()
    positions_rad: dict[str, float] = {}
    raw_values: dict[str, float] = {}

    for name in ARM_JOINT_NAMES:
        raw = float(obs[f"{name}.pos"])
        raw_values[name] = raw
        positions_rad[name] = math.radians(raw)

    gripper_raw = float(obs.get(f"{GRIPPER_JOINT_NAME}.pos", 0.0))
    raw_values[GRIPPER_JOINT_NAME] = gripper_raw
    positions_rad[GRIPPER_JOINT_NAME] = _map_gripper_observation(
        gripper_raw,
        model.joints_by_name[GRIPPER_JOINT_NAME],
        gripper_units,
        invert_gripper,
    )

    return positions_rad, raw_values


class LatestJointReader:
    """Continuously poll the robot and keep only the newest joint sample."""

    def __init__(
        self,
        robot,
        model: RobotModel,
        gripper_units: str,
        invert_gripper: bool,
        poll_hz: float,
    ) -> None:
        self._robot = robot
        self._model = model
        self._gripper_units = gripper_units
        self._invert_gripper = invert_gripper
        self._min_dt = 1.0 / max(poll_hz, 1e-6)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="joint-reader", daemon=True)
        self._latest: JointSample | None = None
        self._seq = 0

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)

    def latest(self) -> JointSample | None:
        with self._lock:
            sample = self._latest
            if sample is None:
                return None
            return JointSample(
                joint_positions_rad=sample.joint_positions_rad.copy(),
                raw_values=sample.raw_values.copy(),
                timestamp=sample.timestamp,
                read_ms=sample.read_ms,
                seq=sample.seq,
            )

    def _run(self) -> None:
        last_error_t = 0.0
        while not self._stop.is_set():
            loop_start = time.monotonic()
            try:
                positions, raw = read_joint_positions(
                    self._robot,
                    self._model,
                    self._gripper_units,
                    self._invert_gripper,
                )
                now = time.monotonic()
                sample = JointSample(
                    joint_positions_rad=positions,
                    raw_values=raw,
                    timestamp=now,
                    read_ms=(now - loop_start) * 1000.0,
                    seq=self._seq,
                )
                self._seq += 1
                with self._lock:
                    self._latest = sample
            except Exception as e:
                now = time.monotonic()
                if now - last_error_t > 1.0:
                    print(f"[render] joint read failed: {e}")
                    last_error_t = now

            elapsed = time.monotonic() - loop_start
            if elapsed < self._min_dt:
                self._stop.wait(self._min_dt - elapsed)


def _joint_axis_segments(
    model: RobotModel,
    link_poses: dict[str, np.ndarray],
    length_m: float = 0.04,
) -> list[tuple[np.ndarray, np.ndarray]]:
    segments: list[tuple[np.ndarray, np.ndarray]] = []
    for name in JOINT_NAMES:
        joint = model.joints_by_name.get(name)
        if joint is None or joint.parent not in link_poses:
            continue
        axis_norm = float(np.linalg.norm(joint.axis))
        if axis_norm <= 1e-12:
            continue
        T_base_joint = link_poses[joint.parent] @ joint.T_parent_joint
        start = T_base_joint[:3, 3]
        axis = T_base_joint[:3, :3] @ (joint.axis / axis_norm)
        segments.append((start, start + axis * length_m))
    return segments


def _skeleton_segments(
    model: RobotModel,
    link_poses: dict[str, np.ndarray],
) -> list[tuple[np.ndarray, np.ndarray]]:
    segments: list[tuple[np.ndarray, np.ndarray]] = []
    for joint in model.joints:
        if joint.parent in link_poses and joint.child in link_poses:
            segments.append((link_poses[joint.parent][:3, 3], link_poses[joint.child][:3, 3]))
    return segments


def _segments_to_polydata(segments: list[tuple[np.ndarray, np.ndarray]]) -> pv.PolyData:
    if not segments:
        points = np.zeros((2, 3), dtype=np.float64)
        lines = np.array([2, 0, 1], dtype=np.int64)
        return pv.PolyData(points, lines=lines)

    points = np.asarray([p for seg in segments for p in seg], dtype=np.float64)
    lines = []
    for i in range(len(segments)):
        lines.extend([2, 2 * i, 2 * i + 1])
    return pv.PolyData(points, lines=np.asarray(lines, dtype=np.int64))


def _fit_camera(
    plotter: pv.Plotter,
    points: list[np.ndarray],
    margin_m: float = 0.03,
    zoom: float = 1.35,
    parallel: bool = False,
) -> None:
    if not points:
        return
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    lo = pts.min(axis=0) - margin_m
    hi = pts.max(axis=0) + margin_m
    bounds = (lo[0], hi[0], lo[1], hi[1], lo[2], hi[2])
    plotter.reset_camera(bounds=bounds)
    plotter.camera.Zoom(zoom)
    if not parallel:
        return

    camera = plotter.camera
    # Use the bounding-box midpoint (not the mean) so points that are sparse
    # in the cloud (e.g. a small AprilTag at the origin) don't get pushed
    # off-screen by an over-represented cluster (e.g. dense robot CAD bbox
    # corners). With midpoint-as-focal, parallel_scale = span / 2 fits both
    # extremes symmetrically.
    center = (lo + margin_m + hi - margin_m) / 2.0
    view_from = np.asarray(camera.GetPosition(), dtype=np.float64) - np.asarray(
        camera.GetFocalPoint(), dtype=np.float64,
    )
    view_norm = float(np.linalg.norm(view_from))
    if view_norm <= 1e-9:
        return
    view_from /= view_norm

    view_up = np.asarray(camera.GetViewUp(), dtype=np.float64)
    view_up -= view_from * float(np.dot(view_up, view_from))
    up_norm = float(np.linalg.norm(view_up))
    if up_norm <= 1e-9:
        return
    view_up /= up_norm
    view_right = np.cross(view_up, view_from)

    centered = pts - center
    vertical_span = float(np.ptp(centered @ view_up)) + margin_m * 2.0
    horizontal_span = float(np.ptp(centered @ view_right)) + margin_m * 2.0
    width, height = plotter.window_size
    aspect = max(float(width) / max(float(height), 1.0), 1e-6)
    parallel_scale = max(vertical_span / 2.0, horizontal_span / (2.0 * aspect))
    parallel_scale = max(parallel_scale / max(zoom, 1e-6), 1e-4)

    distance = max(view_norm, float(np.linalg.norm(hi - lo)) * 2.0, 0.1)
    camera.SetParallelProjection(True)
    camera.SetFocalPoint(tuple(center))
    camera.SetPosition(tuple(center + view_from * distance))
    camera.SetViewUp(tuple(view_up))
    camera.SetParallelScale(parallel_scale)
    try:
        plotter.renderer.ResetCameraClippingRange()
    except Exception:
        try:
            plotter.reset_camera_clipping_range()
        except Exception:
            pass


class RobotCadScene:
    """Persistent PyVista actors for the SO-101 URDF visual meshes."""

    def __init__(
        self,
        plotter: pv.Plotter,
        model: RobotModel,
        max_triangles_per_visual: int = 1500,
    ) -> None:
        self.plotter = plotter
        self.model = model
        self.visual_actors = build_visual_actors(
            plotter,
            model,
            max_triangles_per_visual,
        )
        self._fit_done = False

    def update(
        self,
        link_poses: dict[str, np.ndarray],
        auto_fit_once: bool = False,
    ) -> list[np.ndarray]:
        fit_points: list[np.ndarray] = []
        for visual_actor in self.visual_actors:
            T_scene_link = link_poses.get(visual_actor.link_name)
            if T_scene_link is None:
                continue
            T_scene_visual = T_scene_link @ visual_actor.T_link_visual
            _set_actor_transform(visual_actor.actor, T_scene_visual)
            fit_points.extend(
                _transform_points(T_scene_visual, visual_actor.local_bbox_corners)
            )

        if auto_fit_once and not self._fit_done:
            _fit_camera(self.plotter, fit_points)
            self._fit_done = True
        return fit_points


def update_view(
    plotter: pv.Plotter,
    model: RobotModel,
    visual_actors: list[VisualActor],
    skeleton_actor: Any | None,
    axis_actor: Any | None,
    link_poses: dict[str, np.ndarray],
    joint_positions_rad: dict[str, float],
    raw_values: dict[str, float],
    fps: float,
    frame_i: int,
    sample_age_ms: float,
    auto_fit: bool,
    show_skeleton: bool,
    show_joint_axes: bool,
) -> tuple[Any | None, Any | None]:
    fit_points: list[np.ndarray] = []
    for visual_actor in visual_actors:
        T_base_link = link_poses.get(visual_actor.link_name)
        if T_base_link is None:
            continue
        T_base_visual = T_base_link @ visual_actor.T_link_visual
        _set_actor_transform(visual_actor.actor, T_base_visual)
        fit_points.extend(_transform_points(T_base_visual, visual_actor.local_bbox_corners))

    if auto_fit and frame_i == 0:
        _fit_camera(plotter, fit_points)

    if show_skeleton:
        if skeleton_actor is not None:
            plotter.remove_actor(skeleton_actor, render=False)
        skeleton_actor = plotter.add_mesh(
            _segments_to_polydata(_skeleton_segments(model, link_poses)),
            color="black",
            line_width=2,
            opacity=0.35,
            render_lines_as_tubes=True,
            name="skeleton",
            render=False,
        )

    if show_joint_axes:
        if axis_actor is not None:
            plotter.remove_actor(axis_actor, render=False)
        axis_actor = plotter.add_mesh(
            _segments_to_polydata(_joint_axis_segments(model, link_poses)),
            color="red",
            line_width=5,
            render_lines_as_tubes=True,
            name="joint_axes",
            render=False,
        )

    if frame_i % STATUS_EVERY_N_FRAMES != 0:
        return skeleton_actor, axis_actor

    lines = [f"SO-101 live CAD | FPS {fps:4.1f} | sample age {sample_age_ms:5.1f} ms", ""]
    for name in ARM_JOINT_NAMES:
        lines.append(f"{name:>14}: {raw_values[name]:+7.2f} deg")
    gripper_rad = joint_positions_rad[GRIPPER_JOINT_NAME]
    lines.append(
        f"{GRIPPER_JOINT_NAME:>14}: {raw_values[GRIPPER_JOINT_NAME]:+7.2f} raw "
        f"({math.degrees(gripper_rad):+6.1f} deg URDF)"
    )
    lines.append("")
    if show_joint_axes:
        lines.append("red tubes: the 6 motor axes")
    else:
        lines.append("debug axes hidden; use --show-joint-axes")
    lines.append("q/ESC: quit")
    plotter.add_text("\n".join(lines), position="upper_left", font_size=10, name="status")

    return skeleton_actor, axis_actor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default=ROBOT_PORT, help="SO-101 follower serial port")
    parser.add_argument("--robot-id", default=ROBOT_ID, help="SO-101 follower id")
    parser.add_argument("--urdf", default=URDF_PATH, help="path to the SO-101 URDF")
    parser.add_argument("--rate-hz", type=float, default=60.0, help="target render/read rate")
    parser.add_argument(
        "--poll-hz",
        type=float,
        default=100.0,
        help="background robot joint polling rate",
    )
    parser.add_argument(
        "--sync-reads",
        action="store_true",
        help="read robot joints in the render loop instead of using latest-only background polling",
    )
    parser.add_argument(
        "--disable-torque",
        action="store_true",
        help="disable motor torque so the rendered robot follows hand posing",
    )
    parser.add_argument(
        "--restore-torque-on-exit",
        action="store_true",
        help="re-enable torque on exit if --disable-torque was used",
    )
    parser.add_argument(
        "--gripper-units",
        choices=("percent", "deg", "rad"),
        default="percent",
        help="interpret gripper.pos as percent, degrees, or radians",
    )
    parser.add_argument(
        "--invert-gripper",
        action="store_true",
        help="reverse the percent-to-URDF mapping for gripper.pos",
    )
    parser.add_argument(
        "--max-triangles-per-visual",
        type=int,
        default=1500,
        help=(
            "mesh decimation cap per URDF visual; "
            "0 keeps full-resolution STL assets"
        ),
    )
    parser.add_argument(
        "--full-mesh",
        action="store_true",
        help="render full-resolution STL assets instead of simplified meshes",
    )
    parser.add_argument(
        "--no-auto-fit",
        action="store_true",
        help="do not fit the camera to the robot on the first frame",
    )
    parser.add_argument(
        "--show-skeleton",
        action="store_true",
        help="draw the FK link skeleton overlay; costs some FPS",
    )
    parser.add_argument(
        "--show-joint-axes",
        action="store_true",
        help="draw the six joint-axis overlay; costs some FPS",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print startup details and periodic FPS/latency telemetry",
    )
    return parser.parse_args()


def _config_from_args(args: argparse.Namespace) -> RenderRobotConfig:
    return RenderRobotConfig(
        port=args.port,
        robot_id=args.robot_id,
        urdf=args.urdf,
        rate_hz=args.rate_hz,
        poll_hz=args.poll_hz,
        sync_reads=args.sync_reads,
        disable_torque=args.disable_torque,
        restore_torque_on_exit=args.restore_torque_on_exit,
        gripper_units=args.gripper_units,
        invert_gripper=args.invert_gripper,
        max_triangles_per_visual=args.max_triangles_per_visual,
        full_mesh=args.full_mesh,
        no_auto_fit=args.no_auto_fit,
        show_skeleton=args.show_skeleton,
        show_joint_axes=args.show_joint_axes,
        verbose=args.verbose,
    )


def render_robot(
    config: RenderRobotConfig | None = None,
    *,
    robot: Any | None = None,
    connect_robot: bool = True,
    disconnect_robot: bool | None = None,
) -> None:
    """Run the live SO-101 renderer until the viewer is closed.

    `robot` can be an existing SO101Follower-like object. If no robot is
    supplied, this function creates, connects, and disconnects one.
    """
    cfg = RenderRobotConfig() if config is None else config
    owns_robot = robot is None
    should_disconnect = owns_robot if disconnect_robot is None else disconnect_robot

    urdf_path = Path(cfg.urdf).resolve()
    model = load_robot_model(urdf_path)
    missing = [name for name in JOINT_NAMES if name not in model.joints_by_name]
    if missing:
        raise SystemExit(f"URDF is missing expected joints: {missing}")

    if robot is None:
        SO101Follower, SO101FollowerConfig = _probe_so101_follower()
        robot = SO101Follower(SO101FollowerConfig(id=cfg.robot_id, port=cfg.port))

    if connect_robot:
        robot.connect()
    if cfg.verbose:
        print(f"[render] robot connected on {cfg.port}")
        print(f"[render] loaded {len(model.visuals)} visual meshes from {urdf_path}")

    torque_disabled = False
    if cfg.disable_torque:
        torque_disabled = _try_disable_torque(robot)
        if cfg.verbose:
            print(
                f"[render] disable_torque requested; "
                f"{'disabled' if torque_disabled else 'NOT disabled'}"
            )

    plotter = pv.Plotter(window_size=(1100, 900), title="SO-101 live CAD")
    plotter.set_background("white")
    plotter.add_axes()
    plotter.show_grid(color="lightgray")
    plotter.camera_position = [(0.45, -0.55, 0.38), (0.0, 0.0, 0.08), (0.0, 0.0, 1.0)]

    max_triangles_per_visual = 0 if cfg.full_mesh else cfg.max_triangles_per_visual
    visual_actors = build_visual_actors(plotter, model, max_triangles_per_visual)
    if max_triangles_per_visual > 0:
        if cfg.verbose:
            print(
                f"[render] simplified meshes to <= {max_triangles_per_visual} "
                "triangles per visual for realtime rendering"
            )
    skeleton_actor = None
    axis_actor = None
    should_quit = {"value": False}

    def quit_viewer() -> None:
        should_quit["value"] = True

    plotter.add_key_event("q", quit_viewer)
    plotter.add_key_event("Escape", quit_viewer)
    plotter.show(interactive_update=True, auto_close=False)

    fps = 0.0
    last_print_t = time.monotonic()
    frame_i = 0
    min_dt = 1.0 / max(cfg.rate_hz, 1e-6)
    read_ms = 0.0
    render_ms = 0.0
    sample_age_ms = 0.0
    reader: LatestJointReader | None = None

    if not cfg.sync_reads:
        reader = LatestJointReader(
            robot,
            model,
            cfg.gripper_units,
            cfg.invert_gripper,
            cfg.poll_hz,
        )
        reader.start()
        if cfg.verbose:
            print(f"[render] background joint polling enabled at {cfg.poll_hz:.1f} Hz")

    try:
        while not should_quit["value"] and not plotter.render_window.GetNeverRendered():
            loop_start = time.monotonic()
            if plotter.iren is not None and plotter.iren.interactor.GetDone():
                break

            if reader is None:
                try:
                    read_start = time.monotonic()
                    joint_positions_rad, raw_values = read_joint_positions(
                        robot,
                        model,
                        cfg.gripper_units,
                        cfg.invert_gripper,
                    )
                    sample_time = time.monotonic()
                    read_ms = (sample_time - read_start) * 1000.0
                    sample_age_ms = (time.monotonic() - sample_time) * 1000.0
                except Exception as e:
                    print(f"[render] joint read failed: {e}")
                    time.sleep(0.1)
                    continue
            else:
                sample = reader.latest()
                if sample is None:
                    time.sleep(0.001)
                    continue
                joint_positions_rad = sample.joint_positions_rad
                raw_values = sample.raw_values
                read_ms = sample.read_ms
                sample_age_ms = (time.monotonic() - sample.timestamp) * 1000.0

            link_poses = model.forward_kinematics(joint_positions_rad)
            render_start = time.monotonic()
            skeleton_actor, axis_actor = update_view(
                plotter,
                model,
                visual_actors,
                skeleton_actor,
                axis_actor,
                link_poses,
                joint_positions_rad,
                raw_values,
                fps,
                frame_i,
                sample_age_ms,
                auto_fit=not cfg.no_auto_fit,
                show_skeleton=cfg.show_skeleton,
                show_joint_axes=cfg.show_joint_axes,
            )
            plotter.update()
            render_ms = (time.monotonic() - render_start) * 1000.0

            elapsed = time.monotonic() - loop_start
            if elapsed < min_dt:
                time.sleep(min_dt - elapsed)
            total_dt = time.monotonic() - loop_start
            inst_fps = 1.0 / max(total_dt, 1e-6)
            fps = 0.9 * fps + 0.1 * inst_fps if fps > 0 else inst_fps

            if cfg.verbose and time.monotonic() - last_print_t >= PRINT_EVERY_S:
                print(
                    f"[render] FPS {fps:4.1f} | read {read_ms:5.1f} ms | "
                    f"age {sample_age_ms:5.1f} ms | render {render_ms:5.1f} ms | "
                    + " | ".join(f"{name}={raw_values[name]:+.1f}" for name in JOINT_NAMES)
                )
                last_print_t = time.monotonic()

            frame_i += 1
    finally:
        if reader is not None:
            reader.stop()
        if torque_disabled and cfg.restore_torque_on_exit:
            restored = _try_enable_torque(robot)
            print(f"[render] restore torque: {'ok' if restored else 'failed'}")
        if should_disconnect:
            try:
                robot.disconnect()
            except Exception:
                pass
        plotter.close()


def main() -> None:
    render_robot(_config_from_args(parse_args()))


if __name__ == "__main__":
    main()
