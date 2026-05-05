from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import mujoco  # type: ignore[import-not-found]
import mujoco.viewer  # type: ignore[import-not-found]
import numpy as np  # type: ignore[import-not-found]

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from gripper import CLOSED_GRIPPER, OPEN_GRIPPER
from motion import solve_ik
from mujoco_sim.cup_scene_config import (
    CUP_SCENE_PATH,
    CUP_TAG_CAMERA_NAMES,
    CUP_TAG_SIZE,
    CUP_TAG_TO_CUP_CENTER_OFFSET,
    DEFAULT_CUP_FRICTION,
    DEFAULT_CUP_HALF_HEIGHT,
    DEFAULT_CUP_MASS,
    DEFAULT_CUP_RADIUS,
    DEFAULT_CUP_RIM_HALF_HEIGHT,
    DEFAULT_CUP_RIM_OVERHANG,
    DEFAULT_JAW_FRICTION,
    PLACE_TAG,
    PLACE_TAG_CAMERA_NAMES,
    PRIMARY_CUP,
    SECOND_CUP,
    TOP_DOWN_CAMERA_NAME,
    WRIST_CAMERA_NAME,
    CupObjectSpec,
)
from pose_estimation import TagPoseEstimate, detect_apriltags
from sim_env import SimEnv, create_env
from so101_mujoco_utils import JOINT_ORDER, convert_to_dictionary, send_position_command


MODEL_PATH = CUP_SCENE_PATH
CUP_BODY_NAME = PRIMARY_CUP.body_name
CUP_FREEJOINT_NAME = PRIMARY_CUP.freejoint_name
SECOND_CUP_BODY_NAME = SECOND_CUP.body_name
SECOND_CUP_FREEJOINT_NAME = SECOND_CUP.freejoint_name
FIXED_JAW_BODY_NAME = "gripper"
MOVING_JAW_BODY_NAME = "moving_jaw_so101_v1"
DEFAULT_CUP_POSITION = PRIMARY_CUP.initial_position
DEFAULT_SECOND_CUP_POSITION = SECOND_CUP.initial_position
DEFAULT_PLACE_TAG_POSITION = PLACE_TAG.pos
DEFAULT_SWEEP_JAW_FRICTIONS = (0.8, 1.0, 1.5, 2.0)
DEFAULT_SWEEP_CUP_MASSES = (0.03, 0.045, 0.07)
DEFAULT_SWEEP_GRIPPER_FORCES = (1.5, 2.0, 2.94)
PLACE_TAG_ID = PLACE_TAG.tag_id
SECOND_CUP_TAG_ID = SECOND_CUP.tag.tag_id
CUP_TAG_ID = PRIMARY_CUP.tag.tag_id
CUP_DEBUG_FRAME_DIR = ROOT_DIR / "cup_camera_debug_frames"
DEFAULT_TAG_TO_CUP_CENTER_OFFSET = CUP_TAG_TO_CUP_CENTER_OFFSET
CAMERA_FOV_VISUALIZATION_DISTANCE = 0.25
CAMERA_FOV_VISUALIZATION_ASPECT = 4.0 / 3.0
CAMERA_FOV_LINE_RADIUS = 0.001


@dataclass(frozen=True)
class PickupConfig:
    scene_path: Path = MODEL_PATH
    cup_position: tuple[float, float, float] = DEFAULT_CUP_POSITION
    second_cup_position: tuple[float, float, float] = DEFAULT_SECOND_CUP_POSITION
    cup_radius: float = DEFAULT_CUP_RADIUS
    cup_half_height: float = DEFAULT_CUP_HALF_HEIGHT
    cup_rim_overhang: float = DEFAULT_CUP_RIM_OVERHANG
    cup_rim_half_height: float = DEFAULT_CUP_RIM_HALF_HEIGHT
    cup_mass: float = DEFAULT_CUP_MASS
    cup_friction: tuple[float, float, float] = DEFAULT_CUP_FRICTION
    jaw_friction: tuple[float, float, float] = DEFAULT_JAW_FRICTION
    gripper_force: float = 1.5
    approach_height: float = 0.07
    side_grasp_offset: float = 0.004
    lateral_grasp_offset: float = 0.005
    first_waypoint_forward_offset: float = 0.02
    first_waypoint_left_offset: float = 0.03
    grasp_height: float = 0.070
    lift_height: float = 0.09
    approach_duration: float = 2.0
    descend_duration: float = 1.0
    close_duration: float = 1.0
    squeeze_duration: float = 0.5
    lift_duration: float = 1.5
    final_hold_duration: float = 1.0
    success_lift_delta: float = 0.035
    cup_tag_id: int = CUP_TAG_ID
    second_cup_tag_id: int = SECOND_CUP_TAG_ID
    cup_tag_size: float = CUP_TAG_SIZE
    cup_tag_camera_names: tuple[str, ...] = CUP_TAG_CAMERA_NAMES
    tag_to_cup_center_offset: tuple[float, float, float] = DEFAULT_TAG_TO_CUP_CENTER_OFFSET
    place_tag_id: int = PLACE_TAG_ID
    place_tag_size: float = CUP_TAG_SIZE
    place_tag_position: tuple[float, float, float] = DEFAULT_PLACE_TAG_POSITION
    place_tag_camera_names: tuple[str, ...] = PLACE_TAG_CAMERA_NAMES
    place_approach_height: float = 0.08
    place_lateral_retreat: float = 0.06
    place_success_xy_tolerance: float = 0.04
    place_success_z_tolerance: float = 0.04
    debug_camera_frame_dir: Path | None = CUP_DEBUG_FRAME_DIR
    allow_config_position_fallback: bool = False


@dataclass(frozen=True)
class CupSpec:
    label: str
    body_name: str
    freejoint_name: str
    side_geom_name: str
    rim_geom_name: str
    visual_geom_name: str
    initial_position: tuple[float, float, float]
    tag_id: int
    tag_to_cup_center_offset: tuple[float, float, float]


@dataclass
class ContactDiagnostics:
    model: mujoco.MjModel
    cup_label: str
    cup_body_id: int
    fixed_jaw_body_id: int
    moving_jaw_body_id: int
    cup_geom_ids: set[int]
    steps: int = 0
    cup_contact_steps: int = 0
    fixed_contact_steps: int = 0
    moving_contact_steps: int = 0
    both_jaw_contact_steps: int = 0
    max_contacts: int = 0
    max_cup_z: float = -math.inf
    contact_pairs: set[tuple[str, str]] = field(default_factory=set)

    def update(self, data: mujoco.MjData) -> None:
        self.steps += 1
        cup_z = float(data.xpos[self.cup_body_id, 2])
        self.max_cup_z = max(self.max_cup_z, cup_z)

        cup_contacts = 0
        touching_fixed = False
        touching_moving = False

        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if geom1 in self.cup_geom_ids:
                cup_geom_id, other_geom_id = geom1, geom2
            elif geom2 in self.cup_geom_ids:
                cup_geom_id, other_geom_id = geom2, geom1
            else:
                continue

            cup_contacts += 1
            other_body_id = int(self.model.geom_bodyid[other_geom_id])
            touching_fixed = touching_fixed or other_body_id == self.fixed_jaw_body_id
            touching_moving = touching_moving or other_body_id == self.moving_jaw_body_id
            self.contact_pairs.add(
                (
                    _object_name(self.model, mujoco.mjtObj.mjOBJ_GEOM, cup_geom_id),
                    _object_name(self.model, mujoco.mjtObj.mjOBJ_GEOM, other_geom_id),
                )
            )

        if cup_contacts:
            self.cup_contact_steps += 1
            self.max_contacts = max(self.max_contacts, cup_contacts)
        if touching_fixed:
            self.fixed_contact_steps += 1
        if touching_moving:
            self.moving_contact_steps += 1
        if touching_fixed and touching_moving:
            self.both_jaw_contact_steps += 1


@dataclass(frozen=True)
class PickupResult:
    success: bool
    initial_cup_z: float
    final_cup_z: float
    max_cup_z: float
    diagnostics: ContactDiagnostics
    cup_center: np.ndarray
    grasp_xyz: np.ndarray
    grasp_offset: np.ndarray


@dataclass(frozen=True)
class PlacementResult:
    success: bool
    target_center: np.ndarray
    final_center: np.ndarray
    xy_error: float
    z_error: float
    diagnostics: ContactDiagnostics


@dataclass(frozen=True)
class CupStackSequenceResult:
    first_pickup: PickupResult
    first_place: PlacementResult | None
    second_pickup: PickupResult | None
    stack_place: PlacementResult | None

    @property
    def success(self) -> bool:
        return (
            self.first_pickup.success
            and self.first_place is not None
            and self.first_place.success
            and self.second_pickup is not None
            and self.second_pickup.success
            and self.stack_place is not None
            and self.stack_place.success
        )


@dataclass(frozen=True)
class CupTargetPoints:
    green_center: np.ndarray
    blue_pregrasp: np.ndarray
    tag_estimate: TagPoseEstimate | None


@dataclass(frozen=True)
class ViewerDebugOverlay:
    green_center: np.ndarray | None = None
    blue_pregrasp: np.ndarray | None = None
    camera_names: tuple[str, ...] = CUP_TAG_CAMERA_NAMES


def _object_name(model: mujoco.MjModel, obj_type: mujoco.mjtObj, obj_id: int) -> str:
    name = mujoco.mj_id2name(model, obj_type, obj_id)
    return name or f"{obj_type.name.lower()}_{obj_id}"


def cup_spec_from_scene(
    cup_scene: CupObjectSpec,
    initial_position: tuple[float, float, float],
    tag_id: int,
    tag_to_cup_center_offset: tuple[float, float, float],
) -> CupSpec:
    return CupSpec(
        label=cup_scene.label,
        body_name=cup_scene.body_name,
        freejoint_name=cup_scene.freejoint_name,
        side_geom_name=cup_scene.side_geom_name,
        rim_geom_name=cup_scene.rim_geom_name,
        visual_geom_name=cup_scene.visual_geom_name,
        initial_position=initial_position,
        tag_id=tag_id,
        tag_to_cup_center_offset=tag_to_cup_center_offset,
    )


def primary_cup_spec(config: PickupConfig) -> CupSpec:
    return cup_spec_from_scene(
        PRIMARY_CUP,
        initial_position=config.cup_position,
        tag_id=config.cup_tag_id,
        tag_to_cup_center_offset=config.tag_to_cup_center_offset,
    )


def second_cup_spec(config: PickupConfig) -> CupSpec:
    return cup_spec_from_scene(
        SECOND_CUP,
        initial_position=config.second_cup_position,
        tag_id=config.second_cup_tag_id,
        tag_to_cup_center_offset=config.tag_to_cup_center_offset,
    )


def _require_id(model: mujoco.MjModel, obj_type: mujoco.mjtObj, name: str) -> int:
    obj_id = mujoco.mj_name2id(model, obj_type, name)
    if obj_id < 0:
        raise ValueError(f"Could not find {obj_type.name.lower()} named {name!r}.")
    return obj_id


def _body_geom_ids(model: mujoco.MjModel, body_id: int, contact_only: bool = False) -> set[int]:
    geom_ids: set[int] = set()
    for geom_id in range(model.ngeom):
        if int(model.geom_bodyid[geom_id]) != body_id:
            continue
        if contact_only and model.geom_contype[geom_id] == 0 and model.geom_conaffinity[geom_id] == 0:
            continue
        geom_ids.add(geom_id)
    return geom_ids


def _set_freejoint_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_name: str,
    position: tuple[float, float, float],
) -> None:
    joint_id = _require_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    qpos_address = int(model.jnt_qposadr[joint_id])
    qvel_address = int(model.jnt_dofadr[joint_id])
    data.qpos[qpos_address : qpos_address + 7] = [*position, 1.0, 0.0, 0.0, 0.0]
    data.qvel[qvel_address : qvel_address + 6] = 0.0
    mujoco.mj_forward(model, data)


def _scale_body_mass(model: mujoco.MjModel, body_id: int, target_mass: float) -> None:
    current_mass = float(model.body_mass[body_id])
    if current_mass <= 0.0:
        raise ValueError("Cannot scale a body with non-positive compiled mass.")
    inertia_scale = target_mass / current_mass
    model.body_mass[body_id] = target_mass
    model.body_inertia[body_id] *= inertia_scale


def _set_cup_size(model: mujoco.MjModel, cup: CupSpec, radius: float, half_height: float) -> None:
    for geom_name, visual_offset in ((cup.side_geom_name, 0.0), (cup.visual_geom_name, 0.0005)):
        geom_id = _require_id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        model.geom_size[geom_id, 0] = radius + visual_offset
        model.geom_size[geom_id, 1] = half_height + visual_offset


def _set_cup_rim(
    model: mujoco.MjModel,
    cup: CupSpec,
    radius: float,
    half_height: float,
    overhang: float,
    rim_half_height: float,
) -> None:
    rim_geom_id = _require_id(model, mujoco.mjtObj.mjOBJ_GEOM, cup.rim_geom_name)
    model.geom_size[rim_geom_id, 0] = radius + overhang
    model.geom_size[rim_geom_id, 1] = rim_half_height
    model.geom_pos[rim_geom_id, 2] = half_height - rim_half_height


def _set_actuator_force(model: mujoco.MjModel, actuator_name: str, force: float) -> None:
    actuator_id = _require_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
    model.actuator_forcerange[actuator_id] = [-force, force]


def _set_geom_friction(
    model: mujoco.MjModel,
    geom_ids: set[int],
    friction: tuple[float, float, float],
) -> None:
    for geom_id in geom_ids:
        model.geom_friction[geom_id] = friction


def configure_cup(env: SimEnv, config: PickupConfig, cup: CupSpec) -> ContactDiagnostics:
    cup_body_id = _require_id(env.model, mujoco.mjtObj.mjOBJ_BODY, cup.body_name)
    fixed_jaw_body_id = _require_id(env.model, mujoco.mjtObj.mjOBJ_BODY, FIXED_JAW_BODY_NAME)
    moving_jaw_body_id = _require_id(env.model, mujoco.mjtObj.mjOBJ_BODY, MOVING_JAW_BODY_NAME)

    _set_cup_size(env.model, cup, config.cup_radius, config.cup_half_height)
    _set_cup_rim(
        env.model,
        cup,
        config.cup_radius,
        config.cup_half_height,
        config.cup_rim_overhang,
        config.cup_rim_half_height,
    )
    _set_freejoint_pose(env.model, env.data, cup.freejoint_name, cup.initial_position)
    _scale_body_mass(env.model, cup_body_id, config.cup_mass)
    _set_actuator_force(env.model, "gripper", config.gripper_force)

    cup_geom_ids = _body_geom_ids(env.model, cup_body_id, contact_only=True)
    fixed_jaw_geom_ids = _body_geom_ids(env.model, fixed_jaw_body_id, contact_only=True)
    moving_jaw_geom_ids = _body_geom_ids(env.model, moving_jaw_body_id, contact_only=True)
    _set_geom_friction(env.model, cup_geom_ids, config.cup_friction)
    _set_geom_friction(env.model, fixed_jaw_geom_ids | moving_jaw_geom_ids, config.jaw_friction)
    mujoco.mj_forward(env.model, env.data)

    return ContactDiagnostics(
        model=env.model,
        cup_label=cup.label,
        cup_body_id=cup_body_id,
        fixed_jaw_body_id=fixed_jaw_body_id,
        moving_jaw_body_id=moving_jaw_body_id,
        cup_geom_ids=cup_geom_ids,
    )


def configure_cups(env: SimEnv, config: PickupConfig) -> dict[str, ContactDiagnostics]:
    cups = (primary_cup_spec(config), second_cup_spec(config))
    return {cup.label: configure_cup(env, config, cup) for cup in cups}


def configure_pickup_env(env: SimEnv, config: PickupConfig) -> ContactDiagnostics:
    return configure_cups(env, config)[primary_cup_spec(config).label]


def _interpolate_position(
    start_position: dict[str, float],
    target_position: dict[str, float],
    alpha: float,
) -> dict[str, float]:
    return {
        joint: (1.0 - alpha) * start_position[joint] + alpha * target_position[joint]
        for joint in JOINT_ORDER
    }


def _step_once(
    env: SimEnv,
    diagnostics: ContactDiagnostics,
    viewer: mujoco.viewer.Handle | None,
    realtime: bool,
    debug_overlay: ViewerDebugOverlay | None = None,
) -> bool:
    step_start = time.time()
    mujoco.mj_step(env.model, env.data)
    diagnostics.update(env.data)

    if viewer is not None:
        if debug_overlay is not None:
            refresh_viewer_debug_overlays(viewer, env, debug_overlay)
        viewer.sync()
        if not viewer.is_running():
            return False

    if realtime:
        sleep_time = env.model.opt.timestep - (time.time() - step_start)
        if sleep_time > 0:
            time.sleep(sleep_time)
    return True


def command_motion(
    env: SimEnv,
    target_position: dict[str, float],
    duration: float,
    diagnostics: ContactDiagnostics,
    viewer: mujoco.viewer.Handle | None = None,
    realtime: bool = False,
    debug_overlay: ViewerDebugOverlay | None = None,
) -> bool:
    start_position = convert_to_dictionary(env.data.qpos.copy())
    steps = max(1, math.ceil(duration / env.model.opt.timestep))

    for step_index in range(steps):
        alpha = (step_index + 1) / steps
        command = _interpolate_position(start_position, target_position, alpha)
        send_position_command(env.data, command)
        if not _step_once(env, diagnostics, viewer, realtime, debug_overlay):
            return False

    env.current_position = dict(target_position)
    return True


def hold_command(
    env: SimEnv,
    target_position: dict[str, float],
    duration: float,
    diagnostics: ContactDiagnostics,
    viewer: mujoco.viewer.Handle | None = None,
    realtime: bool = False,
    debug_overlay: ViewerDebugOverlay | None = None,
) -> bool:
    steps = max(1, math.ceil(duration / env.model.opt.timestep))
    for _ in range(steps):
        send_position_command(env.data, target_position)
        if not _step_once(env, diagnostics, viewer, realtime, debug_overlay):
            return False
    env.current_position = dict(target_position)
    return True


def solve_target(env: SimEnv, xyz: np.ndarray, gripper_position: float) -> dict[str, float]:
    plan = solve_ik(env, xyz, gripper_position=gripper_position)
    target = plan.target_pose[:3, 3]
    print(
        "target: "
        f"x={target[0]:.4f} y={target[1]:.4f} z={target[2]:.4f} m, "
        f"gripper={gripper_position:.1f}, IK error={plan.position_error:.6f} m"
    )
    return plan.target_position


def _format_xyz(position: np.ndarray) -> str:
    return f"({position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f})"


def _camera_world_position(env: SimEnv, camera_name: str) -> np.ndarray | None:
    camera_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if camera_id < 0:
        return None
    return env.data.cam_xpos[camera_id].copy()


def _detect_tag_estimates(
    env: SimEnv,
    tag_id: int,
    tag_size: float,
    camera_names: tuple[str, ...],
    debug_frame_dir: Path | None,
) -> list[TagPoseEstimate]:
    camera_names = tuple(dict.fromkeys(camera_name for camera_name in camera_names if camera_name))
    if not camera_names:
        raise ValueError("At least one AprilTag camera must be configured.")

    estimates: list[TagPoseEstimate] = []
    for camera_name in camera_names:
        camera_estimates = detect_apriltags(
            env,
            camera_name=camera_name,
            tag_sizes={tag_id: tag_size},
            camera_names=(camera_name,),
            debug_frame_dir=debug_frame_dir,
        )
        estimate = camera_estimates.get(tag_id)
        if estimate is not None:
            estimates.append(estimate)

    return estimates


def _estimate_for_offset_rotation(estimates: list[TagPoseEstimate]) -> TagPoseEstimate:
    for estimate in estimates:
        if estimate.camera_name == WRIST_CAMERA_NAME:
            return estimate
    return estimates[0]


def _fuse_equal_weight_tag_estimates(estimates: list[TagPoseEstimate]) -> TagPoseEstimate:
    if not estimates:
        raise ValueError("Cannot fuse an empty list of tag estimates.")
    if len(estimates) == 1:
        return estimates[0]

    rotation_source = _estimate_for_offset_rotation(estimates)
    tag_ids = {estimate.tag_id for estimate in estimates}
    if len(tag_ids) != 1:
        raise ValueError(f"Cannot fuse mismatched tag IDs: {sorted(tag_ids)}")

    return TagPoseEstimate(
        tag_id=rotation_source.tag_id,
        world_position=np.mean([estimate.world_position for estimate in estimates], axis=0),
        world_rotation=rotation_source.world_rotation.copy(),
        camera_position=rotation_source.camera_position.copy(),
        pose_error=float(np.mean([estimate.pose_error for estimate in estimates])),
        corners=rotation_source.corners.copy(),
        camera_name="+".join(estimate.camera_name for estimate in estimates),
    )


def estimate_cup_center_from_tag(
    env: SimEnv,
    config: PickupConfig,
    cup: CupSpec,
) -> tuple[np.ndarray, TagPoseEstimate | None]:
    try:
        estimates = _detect_tag_estimates(
            env,
            tag_id=cup.tag_id,
            tag_size=config.cup_tag_size,
            camera_names=config.cup_tag_camera_names,
            debug_frame_dir=config.debug_camera_frame_dir,
        )
    except Exception as exc:
        if not config.allow_config_position_fallback:
            raise
        print(f"{cup.label} tag detection failed ({exc}); using configured cup position fallback")
        return np.array(cup.initial_position, dtype=float), None

    if not estimates:
        if config.allow_config_position_fallback:
            print(f"{cup.label} tag not detected; using configured cup position fallback")
            return np.array(cup.initial_position, dtype=float), None
        raise RuntimeError(
            f"{cup.label.title()} AprilTag {cup.tag_id} was not detected from cameras: "
            + ", ".join(config.cup_tag_camera_names)
        )

    estimate = _fuse_equal_weight_tag_estimates(estimates)
    tag_to_cup_center = np.asarray(cup.tag_to_cup_center_offset, dtype=float)
    if tag_to_cup_center.shape != (3,):
        raise ValueError("tag_to_cup_center_offset must contain exactly three values.")

    green_center = estimate.world_position + estimate.world_rotation @ tag_to_cup_center
    print(
        f"{cup.label} tag: "
        f"id={estimate.tag_id} cameras={estimate.camera_name} "
        f"tag={_format_xyz(estimate.world_position)} m "
        f"green_center={_format_xyz(green_center)} m"
    )
    if len(estimates) > 1:
        print(f"  fused {len(estimates)} camera estimates with equal weights")
    for contributor in estimates:
        camera_position = _camera_world_position(env, contributor.camera_name)
        camera_position_text = "unavailable" if camera_position is None else _format_xyz(camera_position)
        print(
            f"  camera={contributor.camera_name} "
            f"camera_pos={camera_position_text} m "
            f"tag={_format_xyz(contributor.world_position)} m "
            f"error={contributor.pose_error:.6f}"
        )
    return green_center, estimate


def estimate_place_target_center(env: SimEnv, config: PickupConfig) -> tuple[np.ndarray, TagPoseEstimate | None]:
    try:
        estimates = _detect_tag_estimates(
            env,
            tag_id=config.place_tag_id,
            tag_size=config.place_tag_size,
            camera_names=config.place_tag_camera_names,
            debug_frame_dir=config.debug_camera_frame_dir,
        )
    except Exception as exc:
        if not config.allow_config_position_fallback:
            raise
        print(f"place tag detection failed ({exc}); using configured place tag fallback")
        place_tag = np.array(config.place_tag_position, dtype=float)
        return np.array([place_tag[0], place_tag[1], config.cup_half_height], dtype=float), None

    if not estimates:
        if config.allow_config_position_fallback:
            print("place tag not detected; using configured place tag fallback")
            place_tag = np.array(config.place_tag_position, dtype=float)
            return np.array([place_tag[0], place_tag[1], config.cup_half_height], dtype=float), None
        raise RuntimeError(
            f"Place AprilTag {config.place_tag_id} was not detected from cameras: "
            + ", ".join(config.place_tag_camera_names)
        )

    estimate = _fuse_equal_weight_tag_estimates(estimates)
    target_center = np.array(
        [estimate.world_position[0], estimate.world_position[1], config.cup_half_height],
        dtype=float,
    )
    print(
        "place tag: "
        f"id={estimate.tag_id} cameras={estimate.camera_name} "
        f"tag={_format_xyz(estimate.world_position)} m "
        f"cup_center_target={_format_xyz(target_center)} m"
    )
    return target_center, estimate


def target_points_from_cup_center(
    green_center: np.ndarray,
    config: PickupConfig,
    tag_estimate: TagPoseEstimate | None,
) -> CupTargetPoints:
    cup_xy = np.array(green_center[:2], dtype=float)
    radial_norm = np.linalg.norm(cup_xy)
    if radial_norm == 0.0:
        raise ValueError("Cup center must not be directly above the robot base.")
    radial = cup_xy / radial_norm
    left = np.array([-radial[1], radial[0]], dtype=float)
    pregrasp_xy = cup_xy + radial * config.first_waypoint_forward_offset + left * config.first_waypoint_left_offset
    blue_pregrasp = np.array([pregrasp_xy[0], pregrasp_xy[1], green_center[2]], dtype=float)
    return CupTargetPoints(green_center=green_center, blue_pregrasp=blue_pregrasp, tag_estimate=tag_estimate)


def estimate_cup_target_points(env: SimEnv, config: PickupConfig, cup: CupSpec | None = None) -> CupTargetPoints:
    cup = primary_cup_spec(config) if cup is None else cup
    green_center, tag_estimate = estimate_cup_center_from_tag(env, config, cup)
    return target_points_from_cup_center(green_center, config, tag_estimate)


def _add_debug_sphere(
    viewer: mujoco.viewer.Handle,
    geom_index: int,
    position: np.ndarray,
    radius: float,
    rgba: list[float],
) -> int:
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[geom_index],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[radius, 0.0, 0.0],
        pos=position,
        mat=np.eye(3).flatten(),
        rgba=rgba,
    )
    return geom_index + 1


def _add_debug_line(
    viewer: mujoco.viewer.Handle,
    geom_index: int,
    start: np.ndarray,
    end: np.ndarray,
    rgba: list[float],
    radius: float = CAMERA_FOV_LINE_RADIUS,
) -> int:
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[geom_index],
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        size=[radius, 0.0, 0.0],
        pos=np.zeros(3),
        mat=np.eye(3).flatten(),
        rgba=rgba,
    )
    mujoco.mjv_connector(
        viewer.user_scn.geoms[geom_index],
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        radius,
        start,
        end,
    )
    viewer.user_scn.geoms[geom_index].rgba[:] = rgba
    return geom_index + 1


def _camera_frustum_corners(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera_id: int,
    distance: float = CAMERA_FOV_VISUALIZATION_DISTANCE,
    aspect: float = CAMERA_FOV_VISUALIZATION_ASPECT,
) -> tuple[np.ndarray, np.ndarray]:
    camera_position = data.cam_xpos[camera_id].copy()
    camera_rotation = data.cam_xmat[camera_id].reshape(3, 3).copy()
    half_height = distance * math.tan(0.5 * math.radians(float(model.cam_fovy[camera_id])))
    half_width = half_height * aspect
    local_corners = np.array(
        [
            [-half_width, -half_height, -distance],
            [half_width, -half_height, -distance],
            [half_width, half_height, -distance],
            [-half_width, half_height, -distance],
        ],
        dtype=float,
    )
    return camera_position, camera_position + local_corners @ camera_rotation.T


def _add_camera_fov_overlays(
    viewer: mujoco.viewer.Handle,
    env: SimEnv,
    geom_index: int,
    camera_names: tuple[str, ...] = CUP_TAG_CAMERA_NAMES,
) -> int:
    camera_rgba = [1.0, 0.65, 0.0, 0.85]
    frustum_rgba = [1.0, 0.65, 0.0, 0.45]
    forward_rgba = [1.0, 0.25, 0.0, 0.7]

    for camera_name in camera_names:
        camera_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if camera_id < 0:
            continue

        camera_position, corners = _camera_frustum_corners(env.model, env.data, camera_id)
        geom_index = _add_debug_sphere(viewer, geom_index, camera_position, 0.008, camera_rgba)

        center = np.mean(corners, axis=0)
        geom_index = _add_debug_line(viewer, geom_index, camera_position, center, forward_rgba, radius=0.0015)

        for corner in corners:
            geom_index = _add_debug_line(viewer, geom_index, camera_position, corner, frustum_rgba)
        for start, end in zip(corners, np.roll(corners, shift=-1, axis=0), strict=True):
            geom_index = _add_debug_line(viewer, geom_index, start, end, frustum_rgba)

    return geom_index


def show_viewer_debug_overlays(
    viewer: mujoco.viewer.Handle | None,
    env: SimEnv,
    green_center: np.ndarray,
    blue_pregrasp: np.ndarray,
    camera_names: tuple[str, ...] = CUP_TAG_CAMERA_NAMES,
) -> None:
    if viewer is None:
        return

    refresh_viewer_debug_overlays(
        viewer,
        env,
        ViewerDebugOverlay(
            green_center=green_center,
            blue_pregrasp=blue_pregrasp,
            camera_names=camera_names,
        ),
    )
    viewer.sync()


def refresh_viewer_debug_overlays(
    viewer: mujoco.viewer.Handle,
    env: SimEnv,
    overlay: ViewerDebugOverlay,
) -> None:
    geom_index = 0
    if overlay.green_center is not None:
        geom_index = _add_debug_sphere(viewer, geom_index, overlay.green_center, 0.010, [0.0, 1.0, 0.0, 0.55])
    if overlay.blue_pregrasp is not None:
        geom_index = _add_debug_sphere(viewer, geom_index, overlay.blue_pregrasp, 0.008, [0.0, 0.2, 1.0, 0.75])
    geom_index = _add_camera_fov_overlays(viewer, env, geom_index, overlay.camera_names)
    viewer.user_scn.ngeom = geom_index


def execute_pickup(
    env: SimEnv,
    config: PickupConfig,
    diagnostics: ContactDiagnostics,
    cup: CupSpec | None = None,
    viewer: mujoco.viewer.Handle | None = None,
    realtime: bool = False,
) -> PickupResult:
    cup = primary_cup_spec(config) if cup is None else cup
    target_points = estimate_cup_target_points(env, config, cup)
    pregrasp_xyz = target_points.blue_pregrasp
    approach_xyz = pregrasp_xyz + np.array([0.0, 0.0, config.approach_height], dtype=float)
    lift_xyz = pregrasp_xyz + np.array([0.0, 0.0, config.lift_height], dtype=float)

    initial_cup_z = float(env.data.xpos[diagnostics.cup_body_id, 2])
    debug_overlay = ViewerDebugOverlay(
        green_center=target_points.green_center,
        blue_pregrasp=pregrasp_xyz,
        camera_names=config.cup_tag_camera_names,
    )
    show_viewer_debug_overlays(
        viewer,
        env,
        target_points.green_center,
        pregrasp_xyz,
        config.cup_tag_camera_names,
    )

    approach_position = solve_target(env, approach_xyz, OPEN_GRIPPER)
    if not command_motion(
        env,
        approach_position,
        config.approach_duration,
        diagnostics,
        viewer,
        realtime,
        debug_overlay,
    ):
        return _pickup_result(env, config, diagnostics, initial_cup_z, target_points)

    pregrasp_position = solve_target(env, pregrasp_xyz, OPEN_GRIPPER)
    if not command_motion(
        env,
        pregrasp_position,
        config.descend_duration,
        diagnostics,
        viewer,
        realtime,
        debug_overlay,
    ):
        return _pickup_result(env, config, diagnostics, initial_cup_z, target_points)

    closed_position = dict(pregrasp_position)
    closed_position["gripper"] = CLOSED_GRIPPER
    if not command_motion(
        env,
        closed_position,
        config.close_duration,
        diagnostics,
        viewer,
        realtime,
        debug_overlay,
    ):
        return _pickup_result(env, config, diagnostics, initial_cup_z, target_points)
    if not hold_command(
        env,
        closed_position,
        config.squeeze_duration,
        diagnostics,
        viewer,
        realtime,
        debug_overlay,
    ):
        return _pickup_result(env, config, diagnostics, initial_cup_z, target_points)

    lift_position = solve_target(env, lift_xyz, CLOSED_GRIPPER)
    if not command_motion(
        env,
        lift_position,
        config.lift_duration,
        diagnostics,
        viewer,
        realtime,
        debug_overlay,
    ):
        return _pickup_result(env, config, diagnostics, initial_cup_z, target_points)
    hold_command(
        env,
        lift_position,
        config.final_hold_duration,
        diagnostics,
        viewer,
        realtime,
        debug_overlay,
    )
    return _pickup_result(env, config, diagnostics, initial_cup_z, target_points)


def execute_place(
    env: SimEnv,
    config: PickupConfig,
    diagnostics: ContactDiagnostics,
    pickup_result: PickupResult,
    target_center: np.ndarray,
    viewer: mujoco.viewer.Handle | None = None,
    realtime: bool = False,
) -> PlacementResult:
    release_xyz = target_center + pickup_result.grasp_offset
    approach_xyz = release_xyz + np.array([0.0, 0.0, config.place_approach_height], dtype=float)
    debug_overlay = ViewerDebugOverlay(
        green_center=target_center,
        blue_pregrasp=release_xyz,
        camera_names=config.place_tag_camera_names,
    )
    if viewer is not None:
        refresh_viewer_debug_overlays(viewer, env, debug_overlay)
        viewer.sync()

    approach_position = solve_target(env, approach_xyz, CLOSED_GRIPPER)
    if not command_motion(
        env,
        approach_position,
        config.approach_duration,
        diagnostics,
        viewer,
        realtime,
        debug_overlay,
    ):
        return _placement_result(env, config, diagnostics, target_center)

    release_position = solve_target(env, release_xyz, CLOSED_GRIPPER)
    if not command_motion(
        env,
        release_position,
        config.descend_duration,
        diagnostics,
        viewer,
        realtime,
        debug_overlay,
    ):
        return _placement_result(env, config, diagnostics, target_center)

    open_position = dict(release_position)
    open_position["gripper"] = OPEN_GRIPPER
    if not command_motion(
        env,
        open_position,
        config.close_duration,
        diagnostics,
        viewer,
        realtime,
        debug_overlay,
    ):
        return _placement_result(env, config, diagnostics, target_center)
    if not hold_command(
        env,
        open_position,
        config.squeeze_duration,
        diagnostics,
        viewer,
        realtime,
        debug_overlay,
    ):
        return _placement_result(env, config, diagnostics, target_center)

    retreat_xy = pickup_result.grasp_offset[:2]
    retreat_norm = np.linalg.norm(retreat_xy)
    lateral_retreat_xyz = release_xyz.copy()
    if retreat_norm > 1e-6:
        lateral_retreat_xyz[:2] += retreat_xy / retreat_norm * config.place_lateral_retreat

    lateral_retreat_position = solve_target(env, lateral_retreat_xyz, OPEN_GRIPPER)
    if not command_motion(
        env,
        lateral_retreat_position,
        config.descend_duration,
        diagnostics,
        viewer,
        realtime,
        debug_overlay,
    ):
        return _placement_result(env, config, diagnostics, target_center)

    retreat_xyz = lateral_retreat_xyz + np.array([0.0, 0.0, config.place_approach_height], dtype=float)
    retreat_position = solve_target(env, retreat_xyz, OPEN_GRIPPER)
    command_motion(
        env,
        retreat_position,
        config.lift_duration,
        diagnostics,
        viewer,
        realtime,
        debug_overlay,
    )
    hold_command(
        env,
        retreat_position,
        config.final_hold_duration,
        diagnostics,
        viewer,
        realtime,
        debug_overlay,
    )
    return _placement_result(env, config, diagnostics, target_center)


def _pickup_result(
    env: SimEnv,
    config: PickupConfig,
    diagnostics: ContactDiagnostics,
    initial_cup_z: float,
    target_points: CupTargetPoints,
) -> PickupResult:
    final_cup_z = float(env.data.xpos[diagnostics.cup_body_id, 2])
    success = final_cup_z >= initial_cup_z + config.success_lift_delta
    return PickupResult(
        success=success,
        initial_cup_z=initial_cup_z,
        final_cup_z=final_cup_z,
        max_cup_z=diagnostics.max_cup_z,
        diagnostics=diagnostics,
        cup_center=target_points.green_center,
        grasp_xyz=target_points.blue_pregrasp,
        grasp_offset=target_points.blue_pregrasp - target_points.green_center,
    )


def _placement_result(
    env: SimEnv,
    config: PickupConfig,
    diagnostics: ContactDiagnostics,
    target_center: np.ndarray,
) -> PlacementResult:
    final_center = env.data.xpos[diagnostics.cup_body_id].copy()
    xy_error = float(np.linalg.norm(final_center[:2] - target_center[:2]))
    z_error = float(abs(final_center[2] - target_center[2]))
    return PlacementResult(
        success=xy_error <= config.place_success_xy_tolerance and z_error <= config.place_success_z_tolerance,
        target_center=target_center.copy(),
        final_center=final_center,
        xy_error=xy_error,
        z_error=z_error,
        diagnostics=diagnostics,
    )


def run_pickup(config: PickupConfig, launch_viewer: bool) -> PickupResult:
    env = create_env(scene_path=config.scene_path)
    diagnostics = configure_pickup_env(env, config)

    if not launch_viewer:
        return execute_pickup(env, config, diagnostics, realtime=False)

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        env.viewer = viewer
        return execute_pickup(env, config, diagnostics, viewer=viewer, realtime=True)


def _stack_target_center(env: SimEnv, config: PickupConfig, lower_cup_diagnostics: ContactDiagnostics) -> np.ndarray:
    lower_center = env.data.xpos[lower_cup_diagnostics.cup_body_id].copy()
    return np.array(
        [lower_center[0], lower_center[1], lower_center[2] + 2.0 * config.cup_half_height],
        dtype=float,
    )


def execute_pick_place_stack_sequence(
    env: SimEnv,
    config: PickupConfig,
    diagnostics_by_label: dict[str, ContactDiagnostics],
    viewer: mujoco.viewer.Handle | None = None,
    realtime: bool = False,
) -> CupStackSequenceResult:
    first_cup = primary_cup_spec(config)
    second_cup = second_cup_spec(config)
    first_diagnostics = diagnostics_by_label[first_cup.label]
    second_diagnostics = diagnostics_by_label[second_cup.label]

    ground_target_center, _ = estimate_place_target_center(env, config)
    first_pickup = execute_pickup(env, config, first_diagnostics, first_cup, viewer=viewer, realtime=realtime)
    if not first_pickup.success:
        return CupStackSequenceResult(first_pickup, None, None, None)

    first_place = execute_place(
        env,
        config,
        first_diagnostics,
        first_pickup,
        ground_target_center,
        viewer=viewer,
        realtime=realtime,
    )
    if not first_place.success:
        return CupStackSequenceResult(first_pickup, first_place, None, None)

    second_pickup = execute_pickup(env, config, second_diagnostics, second_cup, viewer=viewer, realtime=realtime)
    if not second_pickup.success:
        return CupStackSequenceResult(first_pickup, first_place, second_pickup, None)

    stack_target_center = _stack_target_center(env, config, first_diagnostics)
    stack_place = execute_place(
        env,
        config,
        second_diagnostics,
        second_pickup,
        stack_target_center,
        viewer=viewer,
        realtime=realtime,
    )
    return CupStackSequenceResult(first_pickup, first_place, second_pickup, stack_place)


def run_pick_place_stack_sequence(config: PickupConfig, launch_viewer: bool) -> CupStackSequenceResult:
    env = create_env(scene_path=config.scene_path)
    diagnostics_by_label = configure_cups(env, config)

    if not launch_viewer:
        return execute_pick_place_stack_sequence(env, config, diagnostics_by_label, realtime=False)

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        env.viewer = viewer
        return execute_pick_place_stack_sequence(
            env,
            config,
            diagnostics_by_label,
            viewer=viewer,
            realtime=True,
        )


def print_result(result: PickupResult) -> None:
    diagnostics = result.diagnostics
    print(f"Pickup result ({diagnostics.cup_label}): " + ("PASS" if result.success else "FAIL"))
    print(
        f"  cup z: initial={result.initial_cup_z:.4f} m "
        f"final={result.final_cup_z:.4f} m max={result.max_cup_z:.4f} m"
    )
    print(
        "  contacts: "
        f"cup_steps={diagnostics.cup_contact_steps}/{diagnostics.steps}, "
        f"fixed_jaw_steps={diagnostics.fixed_contact_steps}, "
        f"moving_jaw_steps={diagnostics.moving_contact_steps}, "
        f"both_jaw_steps={diagnostics.both_jaw_contact_steps}, "
        f"max_contacts={diagnostics.max_contacts}"
    )
    if diagnostics.contact_pairs:
        print("  contact pairs:")
        for cup_geom, other_geom in sorted(diagnostics.contact_pairs):
            print(f"    {cup_geom} <-> {other_geom}")


def print_placement_result(label: str, result: PlacementResult | None) -> None:
    if result is None:
        print(f"{label}: SKIPPED")
        return
    print(f"{label}: " + ("PASS" if result.success else "FAIL"))
    print(
        f"  target={_format_xyz(result.target_center)} m "
        f"final={_format_xyz(result.final_center)} m "
        f"xy_error={result.xy_error:.4f} m z_error={result.z_error:.4f} m"
    )


def print_sequence_result(result: CupStackSequenceResult) -> None:
    print("Cup sequence result: " + ("PASS" if result.success else "FAIL"))
    print_result(result.first_pickup)
    print_placement_result("Place first cup on ground tag", result.first_place)
    if result.second_pickup is None:
        print("Pickup result (second cup): SKIPPED")
    else:
        print_result(result.second_pickup)
    print_placement_result("Place second cup on first cup", result.stack_place)


def parse_friction(values: list[float]) -> tuple[float, float, float]:
    if len(values) != 3:
        raise argparse.ArgumentTypeError("friction must have exactly three values.")
    return float(values[0]), float(values[1]), float(values[2])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a focused SO101 cup pickup test in MuJoCo.")
    parser.add_argument("--scene", type=Path, default=MODEL_PATH, help="MJCF scene to load.")
    parser.add_argument("--headless", action="store_true", help="Run without opening the MuJoCo viewer.")
    parser.add_argument(
        "--sequence",
        choices=("pickup", "place-stack"),
        default="place-stack",
        help="Run the original single pickup or the pick/place/stack sequence.",
    )
    parser.add_argument("--sweep", action="store_true", help="Run a headless jaw-friction/cup-mass sweep.")
    parser.add_argument("--cup-radius", type=float, default=DEFAULT_CUP_RADIUS, help="Cup side collision radius in meters.")
    parser.add_argument("--cup-mass", type=float, default=DEFAULT_CUP_MASS, help="Cup mass in kg.")
    parser.add_argument("--gripper-force", type=float, default=1.5, help="Symmetric gripper actuator force range.")
    parser.add_argument("--cup-friction", type=float, nargs=3, default=DEFAULT_CUP_FRICTION, metavar=("SLIDE", "TORSION", "ROLL"))
    parser.add_argument("--jaw-friction", type=float, nargs=3, default=DEFAULT_JAW_FRICTION, metavar=("SLIDE", "TORSION", "ROLL"))
    parser.add_argument("--sweep-jaw-friction", type=float, nargs="*", default=list(DEFAULT_SWEEP_JAW_FRICTIONS))
    parser.add_argument("--sweep-cup-mass", type=float, nargs="*", default=list(DEFAULT_SWEEP_CUP_MASSES))
    parser.add_argument("--sweep-gripper-force", type=float, nargs="*", default=list(DEFAULT_SWEEP_GRIPPER_FORCES))
    parser.add_argument("--cup-position", type=float, nargs=3, default=DEFAULT_CUP_POSITION, metavar=("X", "Y", "Z"))
    parser.add_argument(
        "--second-cup-position",
        type=float,
        nargs=3,
        default=DEFAULT_SECOND_CUP_POSITION,
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument("--side-grasp-offset", type=float, default=0.004, help="Meters to offset target toward the robot from cup center.")
    parser.add_argument("--lateral-grasp-offset", type=float, default=0.005, help="Meters to offset final grasp target left of cup center.")
    parser.add_argument("--first-waypoint-forward-offset", type=float, default=0.02, help="Meters to place the first low waypoint forward from cup center.")
    parser.add_argument("--first-waypoint-left-offset", type=float, default=0.03, help="Meters to place the first low waypoint left from cup center.")
    parser.add_argument("--grasp-height", type=float, default=0.070, help="World z target for the claw center.")
    parser.add_argument("--lift-height", type=float, default=0.09, help="Meters to lift above the grasp target.")
    parser.add_argument("--cup-tag-id", type=int, default=CUP_TAG_ID, help="AprilTag ID mounted on the cup.")
    parser.add_argument("--second-cup-tag-id", type=int, default=SECOND_CUP_TAG_ID, help="AprilTag ID mounted on the second cup.")
    parser.add_argument("--cup-tag-size", type=float, default=CUP_TAG_SIZE, help="Black-square size of the cup AprilTag in meters.")
    parser.add_argument(
        "--cup-tag-cameras",
        nargs="+",
        default=list(CUP_TAG_CAMERA_NAMES),
        metavar="CAMERA",
        help="Named MuJoCo cameras to use for cup tag detection; visible estimates are fused equally.",
    )
    parser.add_argument("--place-tag-id", type=int, default=PLACE_TAG_ID, help="Flat AprilTag ID used as the first placement target.")
    parser.add_argument("--place-tag-size", type=float, default=CUP_TAG_SIZE, help="Black-square size of the flat placement AprilTag in meters.")
    parser.add_argument(
        "--place-tag-position",
        type=float,
        nargs=3,
        default=DEFAULT_PLACE_TAG_POSITION,
        metavar=("X", "Y", "Z"),
        help="Configured placement tag position used only when fallback is enabled.",
    )
    parser.add_argument(
        "--place-tag-cameras",
        nargs="+",
        default=list(PLACE_TAG_CAMERA_NAMES),
        metavar="CAMERA",
        help="Named MuJoCo cameras to use for flat placement tag detection.",
    )
    parser.add_argument("--place-approach-height", type=float, default=0.08, help="Meters to retreat above each placement target.")
    parser.add_argument("--place-lateral-retreat", type=float, default=0.06, help="Meters to move away from a placed cup before lifting the open gripper.")
    parser.add_argument("--place-success-xy-tolerance", type=float, default=0.04, help="Allowed horizontal placement error in meters.")
    parser.add_argument("--place-success-z-tolerance", type=float, default=0.04, help="Allowed vertical placement error in meters.")
    parser.add_argument(
        "--tag-to-cup-center-offset",
        type=float,
        nargs=3,
        default=DEFAULT_TAG_TO_CUP_CENTER_OFFSET,
        metavar=("X", "Y", "Z"),
        help="Fixed offset from cup AprilTag frame to green cup-center point in meters.",
    )
    parser.add_argument(
        "--allow-config-cup-position-fallback",
        action="store_true",
        help="Use --cup-position if the cup AprilTag is not detected.",
    )
    parser.add_argument(
        "--debug-camera-frame-dir",
        type=Path,
        default=CUP_DEBUG_FRAME_DIR,
        help="Directory for rendered camera PNGs used during cup AprilTag detection.",
    )
    parser.add_argument(
        "--no-debug-camera-frames",
        action="store_true",
        help="Do not save rendered camera PNGs during cup AprilTag detection.",
    )
    return parser.parse_args()


def config_from_args(
    args: argparse.Namespace,
    jaw_sliding_friction: float | None = None,
    cup_mass: float | None = None,
    gripper_force: float | None = None,
) -> PickupConfig:
    jaw_friction = tuple(args.jaw_friction)
    if jaw_sliding_friction is not None:
        jaw_friction = (jaw_sliding_friction, jaw_friction[1], jaw_friction[2])

    return PickupConfig(
        scene_path=args.scene,
        cup_position=tuple(args.cup_position),
        second_cup_position=tuple(args.second_cup_position),
        cup_radius=args.cup_radius,
        cup_mass=args.cup_mass if cup_mass is None else cup_mass,
        cup_friction=tuple(args.cup_friction),
        jaw_friction=jaw_friction,
        gripper_force=args.gripper_force if gripper_force is None else gripper_force,
        side_grasp_offset=args.side_grasp_offset,
        lateral_grasp_offset=args.lateral_grasp_offset,
        first_waypoint_forward_offset=args.first_waypoint_forward_offset,
        first_waypoint_left_offset=args.first_waypoint_left_offset,
        grasp_height=args.grasp_height,
        lift_height=args.lift_height,
        cup_tag_id=args.cup_tag_id,
        second_cup_tag_id=args.second_cup_tag_id,
        cup_tag_size=args.cup_tag_size,
        cup_tag_camera_names=tuple(args.cup_tag_cameras),
        tag_to_cup_center_offset=tuple(args.tag_to_cup_center_offset),
        place_tag_id=args.place_tag_id,
        place_tag_size=args.place_tag_size,
        place_tag_position=tuple(args.place_tag_position),
        place_tag_camera_names=tuple(args.place_tag_cameras),
        place_approach_height=args.place_approach_height,
        place_lateral_retreat=args.place_lateral_retreat,
        place_success_xy_tolerance=args.place_success_xy_tolerance,
        place_success_z_tolerance=args.place_success_z_tolerance,
        debug_camera_frame_dir=None if args.no_debug_camera_frames else args.debug_camera_frame_dir,
        allow_config_position_fallback=args.allow_config_cup_position_fallback,
    )


def run_sweep(args: argparse.Namespace) -> None:
    print("Running headless pickup sweep...")
    for cup_mass in args.sweep_cup_mass:
        for gripper_force in args.sweep_gripper_force:
            for jaw_friction in args.sweep_jaw_friction:
                config = config_from_args(
                    args,
                    jaw_sliding_friction=jaw_friction,
                    cup_mass=cup_mass,
                    gripper_force=gripper_force,
                )
                result = run_pickup(config, launch_viewer=False)
                print(
                    f"{'PASS' if result.success else 'FAIL'} "
                    f"cup_mass={cup_mass:.3f}kg "
                    f"gripper_force={gripper_force:.2f} "
                    f"jaw_slide={jaw_friction:.2f} "
                    f"max_z={result.max_cup_z:.4f}m "
                    f"final_z={result.final_cup_z:.4f}m "
                    f"both_jaw_steps={result.diagnostics.both_jaw_contact_steps}"
                )


def main() -> None:
    args = parse_args()
    if args.sweep:
        run_sweep(args)
        return

    config = config_from_args(args)
    if args.sequence == "pickup":
        result = run_pickup(config, launch_viewer=not args.headless)
        print_result(result)
    else:
        result = run_pick_place_stack_sequence(config, launch_viewer=not args.headless)
        print_sequence_result(result)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        if "mjpython" in str(exc):
            raise SystemExit(
                "MuJoCo viewer on macOS requires mjpython. Run:\n"
                "  conda activate whisk-agent\n"
                "  cd /Users/cadenli/Documents/launchpad/whisk/agent-1\n"
                "  mjpython mujoco_sim/run_cup_pickup.py"
            ) from exc
        raise
