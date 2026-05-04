import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import mujoco  # type: ignore[import-not-found]
import mujoco.viewer  # type: ignore[import-not-found]
import numpy as np  # type: ignore[import-not-found]

from mujoco_sim.apriltag_world_config import APRILTAGS, TAG_FAMILY, TAG_THICKNESS_M
from so101_kinematics import (
    SO101Kinematics,
    pose_from_position_rotation,
    pose_from_target_relative_gripper_angle,
    rotation_error_rad,
)
from so101_mujoco_utils import hold_position, move_to_pose, set_initial_pose


ROOT_DIR = Path(__file__).parent
MODEL_PATH = ROOT_DIR / "simulation_code" / "model" / "scene.xml"
DEFAULT_CAMERA = "table_observer"
DEFAULT_TAG = APRILTAGS[0]
DEFAULT_TAG_NAME = DEFAULT_TAG.name
DEFAULT_TAG_SITE_NAME = f"{DEFAULT_TAG.name}_site"
DEFAULT_POSITION_WEIGHT = 1.0
DEFAULT_ORIENTATION_WEIGHT = 0.01
TARGETED_POSITION_WEIGHT = 1.0
TARGETED_ORIENTATION_WEIGHT = 0.01
TARGETED_MAX_ITERATIONS = 100
POSITION_WARNING_THRESHOLD_M = 0.01
ORIENTATION_WARNING_THRESHOLD_DEG = 5.0
OPENCV_TO_MUJOCO_CAMERA = np.diag([1.0, -1.0, -1.0])

STARTING_POSITION = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -45.0,
    "elbow_flex": 90.0,
    "wrist_flex": -45.0,
    "wrist_roll": 0.0,
    "gripper": 50.0,
}


@dataclass(frozen=True)
class TagPoseEstimate:
    tag_id: int
    world_position: np.ndarray
    world_rotation: np.ndarray
    camera_position: np.ndarray
    pose_error: float
    corners: np.ndarray


@dataclass(frozen=True)
class RawTagDetection:
    tag_id: int
    pose_R: np.ndarray
    pose_t: np.ndarray
    pose_error: float
    corners: np.ndarray


@dataclass(frozen=True)
class IKPlan:
    target_pose: np.ndarray
    target_position: dict[str, float]
    position_error: float
    orientation_error: float


def show_target(viewer, target_pose: np.ndarray) -> None:
    """Single marker: end-effector IK target (legacy helper for `move`)."""
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[0],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.015, 0.0, 0.0],
        pos=target_pose[:3, 3],
        mat=np.eye(3).flatten(),
        rgba=[0.0, 1.0, 0.0, 0.45],
    )
    viewer.user_scn.ngeom = 1
    viewer.sync()


def show_vision_hover_markers(
    viewer,
    estimated_hover_m: np.ndarray,
    ground_truth_hover_m: np.ndarray,
) -> None:
    """Green: hover point from vision (estimated tag center + vertical offset).
    Red: same offset from MuJoCo ground-truth tag site (simulation truth).
    """
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[0],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.015, 0.0, 0.0],
        pos=estimated_hover_m,
        mat=np.eye(3).flatten(),
        rgba=[0.0, 1.0, 0.0, 0.55],
    )
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[1],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.015, 0.0, 0.0],
        pos=ground_truth_hover_m,
        mat=np.eye(3).flatten(),
        rgba=[1.0, 0.0, 0.0, 0.55],
    )
    viewer.user_scn.ngeom = 2
    viewer.sync()


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


def detect_tag_pose(
    image: np.ndarray,
    camera_params: tuple[float, float, float, float],
    tag_size: float,
    tag_id: int,
) -> RawTagDetection:
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
    matching = [detection for detection in detections if detection.tag_id == tag_id]
    if not matching:
        found = ", ".join(str(detection.tag_id) for detection in detections) or "none"
        raise RuntimeError(f"AprilTag id {tag_id} was not detected. Detected ids: {found}.")
    detection = min(matching, key=lambda detection: float(getattr(detection, "pose_err", 0.0) or 0.0))
    return RawTagDetection(
        tag_id=int(detection.tag_id),
        pose_R=np.asarray(detection.pose_R, dtype=float),
        pose_t=np.asarray(detection.pose_t, dtype=float),
        pose_error=float(getattr(detection, "pose_err", 0.0) or 0.0),
        corners=np.asarray(detection.corners, dtype=float),
    )


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
    )


def estimate_tag_world_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera_name: str,
    width: int,
    height: int,
    tag_id: int,
    tag_size: float,
) -> TagPoseEstimate:
    camera_id = require_named_id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    image = render_camera(model, data, camera_name, width, height)
    camera_params = camera_params_from_fovy(model, camera_id, width, height)
    detection = detect_tag_pose(image, camera_params, tag_size=tag_size, tag_id=tag_id)
    return tag_pose_camera_to_world(model, data, camera_name, detection)


def randomize_apriltag_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    rng: np.random.Generator,
    tag_name: str,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    yaw_range_degrees: tuple[float, float],
) -> tuple[np.ndarray, float]:
    body_id = require_named_id(model, mujoco.mjtObj.mjOBJ_BODY, tag_name)
    x = rng.uniform(*x_range)
    y = rng.uniform(*y_range)
    yaw_degrees = rng.uniform(*yaw_range_degrees)
    yaw_rad = math.radians(yaw_degrees)

    model.body_pos[body_id] = np.array([x, y, TAG_THICKNESS_M], dtype=float)
    model.body_quat[body_id] = np.array([math.cos(0.5 * yaw_rad), 0.0, 0.0, math.sin(0.5 * yaw_rad)])
    mujoco.mj_forward(model, data)
    return model.body_pos[body_id].copy(), yaw_degrees


def get_tag_ground_truth_position(model: mujoco.MjModel, data: mujoco.MjData, site_name: str) -> np.ndarray:
    site_id = require_named_id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    return data.site_xpos[site_id].copy()


def solve_ik_target(
    target_world_position: np.ndarray,
    gripper_angle_degrees: float | None,
    kinematics: SO101Kinematics | None = None,
) -> IKPlan:
    if kinematics is None:
        kinematics = SO101Kinematics()

    starting_ee_pose = kinematics.forward_kinematics(STARTING_POSITION, frame="mujoco")
    position_weight = DEFAULT_POSITION_WEIGHT
    orientation_weight = DEFAULT_ORIENTATION_WEIGHT
    max_iterations = 25
    if gripper_angle_degrees is None:
        target_ee_pose = pose_from_position_rotation(target_world_position, starting_ee_pose[:3, :3])
    else:
        target_ee_pose = pose_from_target_relative_gripper_angle(
            target_world_position,
            gripper_angle_degrees,
        )
        position_weight = TARGETED_POSITION_WEIGHT
        orientation_weight = TARGETED_ORIENTATION_WEIGHT
        max_iterations = TARGETED_MAX_ITERATIONS

    target_position = kinematics.inverse_kinematics(
        STARTING_POSITION,
        target_ee_pose,
        position_weight=position_weight,
        orientation_weight=orientation_weight,
        gripper=STARTING_POSITION["gripper"],
        max_iterations=max_iterations,
    )
    solved_pose = kinematics.forward_kinematics(target_position, frame="mujoco")
    return IKPlan(
        target_pose=target_ee_pose,
        target_position=target_position,
        position_error=float(np.linalg.norm(target_ee_pose[:3, 3] - solved_pose[:3, 3])),
        orientation_error=float(rotation_error_rad(target_ee_pose[:3, :3], solved_pose[:3, :3])),
    )


def print_ik_summary(ik_plan: IKPlan, gripper_angle_degrees: float | None) -> None:
    target_position = ik_plan.target_pose[:3, 3]
    print(
        "Target world coordinates: "
        f"x={target_position[0]:.3f} m, y={target_position[1]:.3f} m, z={target_position[2]:.3f} m"
    )
    print(f"IK position error: {ik_plan.position_error:.6f} m")
    print(f"IK orientation error: {np.rad2deg(ik_plan.orientation_error):.3f} deg")
    if gripper_angle_degrees is not None:
        print(f"Target gripper angle: {gripper_angle_degrees:.1f} deg from horizontal")
    if (
        ik_plan.position_error > POSITION_WARNING_THRESHOLD_M
        or np.rad2deg(ik_plan.orientation_error) > ORIENTATION_WARNING_THRESHOLD_DEG
    ):
        print("Warning: target pose may be near or outside the arm's reachable workspace.")


def move(x: float, y: float, z: float, gripper_angle_degrees: float | None = None) -> None:
    """Move to a world-space end-effector position in meters.

    When provided, gripper_angle_degrees is measured from horizontal in the
    target-relative vertical plane. Positive angles point up; negative angles
    point down.
    """
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    set_initial_pose(model, data, STARTING_POSITION)

    target_world_position = np.array([x, y, z], dtype=float)
    ik_plan = solve_ik_target(target_world_position, gripper_angle_degrees)
    print_ik_summary(ik_plan, gripper_angle_degrees)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        show_target(viewer, ik_plan.target_pose)
        hold_position(model, data, viewer, duration=1.0)
        move_to_pose(model, data, viewer, ik_plan.target_position, duration=3.0)
        hold_position(model, data, viewer, duration=3.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Move SO-101 to an AprilTag pose estimated from a MuJoCo camera.")
    parser.add_argument("--scene", type=Path, default=MODEL_PATH, help="MJCF scene to load.")
    parser.add_argument("--camera", default=DEFAULT_CAMERA, help="Named MuJoCo camera used for tag detection.")
    parser.add_argument("--width", type=int, default=640, help="Rendered camera width in pixels.")
    parser.add_argument("--height", type=int, default=480, help="Rendered camera height in pixels.")
    parser.add_argument("--random-tag", action="store_true", help="Randomize the AprilTag pose before detection.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible tag placement.")
    parser.add_argument("--tag-name", default=DEFAULT_TAG_NAME, help="MuJoCo body name for the AprilTag.")
    parser.add_argument("--tag-site", default=DEFAULT_TAG_SITE_NAME, help="MuJoCo site name at the AprilTag center.")
    parser.add_argument("--tag-id", type=int, default=DEFAULT_TAG.tag_id, help="AprilTag id to detect.")
    parser.add_argument("--tag-size", type=float, default=DEFAULT_TAG.size_m, help="AprilTag black-square edge size in meters.")
    parser.add_argument("--tag-x-range", type=float, nargs=2, default=(0.16, 0.32), metavar=("MIN", "MAX"))
    parser.add_argument("--tag-y-range", type=float, nargs=2, default=(-0.12, 0.12), metavar=("MIN", "MAX"))
    parser.add_argument("--tag-yaw-range", type=float, nargs=2, default=(-180.0, 180.0), metavar=("MIN", "MAX"))
    parser.add_argument(
        "--hover-height",
        type=float,
        default=0.05,
        help="Meters above the tag center (world +Z) for the hover target.",
    )
    parser.add_argument(
        "--gripper-angle",
        type=float,
        default=-90.0,
        help="Target gripper angle in degrees from horizontal.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Estimate pose and solve IK without opening the viewer.")
    return parser.parse_args()


def validate_range(name: str, values: tuple[float, float]) -> tuple[float, float]:
    if len(values) != 2 or values[0] > values[1]:
        raise ValueError(f"{name} must be two values in ascending order.")
    return float(values[0]), float(values[1])


def run_detected_tag_motion(args: argparse.Namespace) -> None:
    model = mujoco.MjModel.from_xml_path(str(args.scene))
    data = mujoco.MjData(model)
    set_initial_pose(model, data, STARTING_POSITION)

    rng = np.random.default_rng(args.seed)
    if args.random_tag:
        tag_position, yaw_degrees = randomize_apriltag_pose(
            model,
            data,
            rng,
            tag_name=args.tag_name,
            x_range=validate_range("--tag-x-range", tuple(args.tag_x_range)),
            y_range=validate_range("--tag-y-range", tuple(args.tag_y_range)),
            yaw_range_degrees=validate_range("--tag-yaw-range", tuple(args.tag_yaw_range)),
        )
        print(
            "Randomized AprilTag pose: "
            f"x={tag_position[0]:.4f} y={tag_position[1]:.4f} z={tag_position[2]:.4f} yaw={yaw_degrees:.1f} deg"
        )

    ground_truth_position = get_tag_ground_truth_position(model, data, args.tag_site)
    estimate = estimate_tag_world_pose(
        model,
        data,
        camera_name=args.camera,
        width=args.width,
        height=args.height,
        tag_id=args.tag_id,
        tag_size=args.tag_size,
    )
    estimate_error = np.linalg.norm(estimate.world_position - ground_truth_position)
    print(
        "Estimated AprilTag world position: "
        f"x={estimate.world_position[0]:.4f} y={estimate.world_position[1]:.4f} z={estimate.world_position[2]:.4f} m"
    )
    print(
        "Ground-truth AprilTag world position: "
        f"x={ground_truth_position[0]:.4f} y={ground_truth_position[1]:.4f} z={ground_truth_position[2]:.4f} m"
    )
    print(f"AprilTag pose-estimation position error: {estimate_error:.6f} m")
    print(f"AprilTag detector pose error: {estimate.pose_error:.6f}")

    offset = np.array([0.0, 0.0, args.hover_height])
    target_world_position = estimate.world_position + offset
    ground_truth_hover = ground_truth_position + offset
    ik_plan = solve_ik_target(target_world_position, args.gripper_angle)
    print_ik_summary(ik_plan, args.gripper_angle)
    print(
        "Green viewer marker: camera-estimated tag center + hover offset "
        f"(z+{args.hover_height:.3f} m), same as IK position target."
    )
    print(
        "Red viewer marker: MuJoCo ground-truth tag site + same hover offset "
        f"(z+{args.hover_height:.3f} m)."
    )

    if args.dry_run:
        return

    with mujoco.viewer.launch_passive(model, data) as viewer:
        show_vision_hover_markers(viewer, target_world_position, ground_truth_hover)
        hold_position(model, data, viewer, duration=1.0)
        move_to_pose(model, data, viewer, ik_plan.target_position, duration=3.0)
        hold_position(model, data, viewer, duration=3.0)


if __name__ == "__main__":
    run_detected_tag_motion(parse_args())
