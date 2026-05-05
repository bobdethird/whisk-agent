from __future__ import annotations

import math
import struct
import zlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import mujoco  # type: ignore[import-not-found]
import numpy as np  # type: ignore[import-not-found]

from mujoco_sim.apriltag_world_config import APRILTAGS, CAMERAS, TAG_FAMILY
from sim_env import SimEnv


OPENCV_TO_MUJOCO_CAMERA = np.diag([1.0, -1.0, -1.0])


@dataclass(frozen=True)
class TagPoseEstimate:
    tag_id: int
    world_position: np.ndarray
    world_rotation: np.ndarray
    camera_position: np.ndarray
    pose_error: float
    corners: np.ndarray
    camera_name: str


@dataclass(frozen=True)
class RawTagDetection:
    tag_id: int
    pose_R: np.ndarray
    pose_t: np.ndarray
    pose_error: float
    corners: np.ndarray


def require_named_id(model: mujoco.MjModel, obj_type: mujoco.mjtObj, name: str) -> int:
    obj_id = mujoco.mj_name2id(model, obj_type, name)
    if obj_id < 0:
        raise ValueError(f"Could not find {obj_type.name} named {name!r}.")
    return obj_id


def grayscale(image: np.ndarray) -> np.ndarray:
    return np.clip(
        0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2],
        0,
        255,
    ).astype(np.uint8)


def render_camera(model: mujoco.MjModel, data: mujoco.MjData, camera_name: str, width: int, height: int) -> np.ndarray:
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        renderer.update_scene(data, camera=camera_name)
        return renderer.render()


def png_chunk(kind: bytes, payload: bytes) -> bytes:
    checksum = zlib.crc32(kind + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)


def write_rgb_png(path: Path, image: np.ndarray) -> None:
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected an RGB uint8 image.")

    height, width, _ = image.shape
    raw_rows = b"".join(b"\x00" + image[row].tobytes() for row in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    data = (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", header)
        + png_chunk(b"IDAT", zlib.compress(raw_rows, 9))
        + png_chunk(b"IEND", b"")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def camera_params_from_fovy(model: mujoco.MjModel, camera_id: int, width: int, height: int) -> tuple[float, float, float, float]:
    fovy_rad = math.radians(float(model.cam_fovy[camera_id]))
    fy = 0.5 * height / math.tan(0.5 * fovy_rad)
    fx = fy
    cx = 0.5 * width
    cy = 0.5 * height
    return fx, fy, cx, cy


def load_apriltag_detector():
    try:
        from pupil_apriltags import Detector  # type: ignore[import-not-found]
    except ImportError as pupil_error:
        try:
            from pyapriltags import Detector  # type: ignore[import-not-found]
        except ImportError as py_error:
            raise ImportError("Install pupil-apriltags to run camera-based AprilTag pose estimation.") from py_error
        return Detector, pupil_error
    return Detector, None


def detect_raw_tags(
    image: np.ndarray,
    camera_params: tuple[float, float, float, float],
    tag_size: float,
    tag_ids: set[int],
) -> list[RawTagDetection]:
    Detector, fallback_error = load_apriltag_detector()
    if fallback_error is not None:
        print("pupil-apriltags was not found; using pyapriltags instead.")

    detector = Detector(
        families=TAG_FAMILY,
        nthreads=1,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0,
    )
    detections = detector.detect(
        grayscale(image),
        estimate_tag_pose=True,
        camera_params=camera_params,
        tag_size=tag_size,
    )
    return [
        RawTagDetection(
            tag_id=int(detection.tag_id),
            pose_R=np.asarray(detection.pose_R, dtype=float),
            pose_t=np.asarray(detection.pose_t, dtype=float),
            pose_error=float(getattr(detection, "pose_err", 0.0) or 0.0),
            corners=np.asarray(detection.corners, dtype=float),
        )
        for detection in detections
        if int(detection.tag_id) in tag_ids
    ]


def tag_pose_camera_to_world(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera_name: str,
    detection: RawTagDetection,
) -> TagPoseEstimate:
    camera_id = require_named_id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    camera_world_position = data.cam_xpos[camera_id].copy()
    camera_world_rotation = data.cam_xmat[camera_id].reshape(3, 3).copy()

    pose_t = detection.pose_t.reshape(3)
    pose_R = detection.pose_R.reshape(3, 3)
    mujoco_camera_position = OPENCV_TO_MUJOCO_CAMERA @ pose_t
    mujoco_camera_rotation = OPENCV_TO_MUJOCO_CAMERA @ pose_R

    return TagPoseEstimate(
        tag_id=detection.tag_id,
        world_position=camera_world_position + camera_world_rotation @ mujoco_camera_position,
        world_rotation=camera_world_rotation @ mujoco_camera_rotation,
        camera_position=pose_t,
        pose_error=detection.pose_error,
        corners=detection.corners,
        camera_name=camera_name,
    )


def camera_candidates(preferred_camera: str, camera_names: tuple[str, ...] | None = None) -> tuple[str, ...]:
    configured_cameras = camera_names or tuple(camera.name for camera in CAMERAS)
    candidates = [preferred_camera]
    candidates.extend(camera_name for camera_name in configured_cameras if camera_name != preferred_camera)
    return tuple(candidates)


def tag_size_groups(tag_sizes: Mapping[int, float] | None = None) -> dict[float, set[int]]:
    groups: dict[float, set[int]] = {}
    if tag_sizes is None:
        tag_sizes = {tag.tag_id: tag.size_m for tag in APRILTAGS}
    for tag_id, tag_size in tag_sizes.items():
        groups.setdefault(float(tag_size), set()).add(int(tag_id))
    return groups


def detect_apriltags(
    env: SimEnv,
    camera_name: str | None = None,
    tag_sizes: Mapping[int, float] | None = None,
    camera_names: tuple[str, ...] | None = None,
    debug_frame_dir: Path | None = None,
) -> dict[int, TagPoseEstimate]:
    preferred_camera = camera_name or env.camera_name
    estimates: dict[int, TagPoseEstimate] = {}
    size_groups = tag_size_groups(tag_sizes)
    target_ids = {tag_id for tag_ids in size_groups.values() for tag_id in tag_ids}

    for candidate in camera_candidates(preferred_camera, camera_names):
        if target_ids.issubset(estimates):
            break
        camera_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, candidate)
        if camera_id < 0:
            continue
        image = render_camera(env.model, env.data, candidate, env.render_width, env.render_height)
        if debug_frame_dir is not None:
            write_rgb_png(debug_frame_dir / f"{candidate}.png", image)
        camera_params = camera_params_from_fovy(env.model, camera_id, env.render_width, env.render_height)
        for tag_size, tag_ids in size_groups.items():
            remaining_ids = tag_ids.difference(estimates)
            if not remaining_ids:
                continue
            for detection in detect_raw_tags(image, camera_params, tag_size, remaining_ids):
                estimates[detection.tag_id] = tag_pose_camera_to_world(env.model, env.data, candidate, detection)

    return dict(sorted(estimates.items()))
