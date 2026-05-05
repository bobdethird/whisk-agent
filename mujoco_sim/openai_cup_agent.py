from __future__ import annotations

import asyncio
import base64
import json
import math
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco  # type: ignore[import-not-found]
import numpy as np  # type: ignore[import-not-found]

from gripper import CLOSED_GRIPPER, OPEN_GRIPPER
from mujoco_sim.cup_scene_config import (
    CUP_SCENE_CAMERAS,
    PLACE_TAG,
    PRIMARY_CUP,
    SECOND_CUP,
)
from mujoco_sim.run_cup_pickup import (
    AprilTagPoseCache,
    ContactDiagnostics,
    CupSpec,
    PickupConfig,
    WORK_TABLE_BODY_NAME,
    _detect_tag_estimates,
    command_motion,
    configure_scene,
    cup_center_from_tag_estimate,
    current_gripper_cup_contacts,
    hold_command,
    place_target_center_from_tag_estimate,
    planner_context_for_cup,
    primary_cup_spec,
    scene_apriltag_sizes,
    second_cup_spec,
    solve_target,
    target_points_from_cup_center,
)
from pose_estimation import TagPoseEstimate, render_camera
from sim_env import HORIZONTAL_WRIST_ROLL_DEGREES, SimEnv, create_env
from so101_kinematics import (
    FIXED_JAW_TOOL_POINT,
    ToolPointName,
    gripperframe_pose_to_tool_target_pose,
)
from so101_mujoco_utils import JOINT_ORDER, convert_to_dictionary


DEFAULT_AGENT_MODEL = "gpt-5.5"
DEFAULT_MAX_AGENT_STEPS = 12
DEFAULT_AGENT_CAMERAS = ("wrist_cam", "table_observer", "cup_observer")
DEFAULT_MOVE_DURATION = 1.0
DEFAULT_GRIPPER_DURATION = 0.5
MOVE_ARM_TARGETS = ("custom", "approach", "grasp", "lift", "place_above", "place")


@dataclass
class CupAgentContext:
    env: SimEnv
    config: PickupConfig
    diagnostics_by_label: dict[str, ContactDiagnostics]
    tag_pose_cache: AprilTagPoseCache
    viewer: mujoco.viewer.Handle | None = None  # type: ignore[name-defined]
    realtime: bool = False
    move_duration: float = DEFAULT_MOVE_DURATION
    gripper_duration: float = DEFAULT_GRIPPER_DURATION
    held_cup_label: str | None = None
    held_grasp_offset: np.ndarray | None = None
    last_tool_result: dict[str, Any] | None = None
    attempt_guidance: str | None = None
    completed_pour: dict[str, Any] | None = None


@dataclass(frozen=True)
class AgentStepResult:
    step_index: int
    final_output: str
    observation: dict[str, Any]
    tool_output: dict[str, Any] | None = None
    camera_frame_paths: dict[str, str] | None = None


def _round_float(value: float, digits: int = 5) -> float:
    if not math.isfinite(value):
        return value
    return round(float(value), digits)


def _round_list(values: np.ndarray | list[float] | tuple[float, ...], digits: int = 5) -> list[float]:
    return [_round_float(float(value), digits) for value in values]


def _object_name(model: mujoco.MjModel, obj_type: mujoco.mjtObj, obj_id: int) -> str | None:
    if obj_id < 0:
        return None
    return mujoco.mj_id2name(model, obj_type, obj_id)


def _named_id(model: mujoco.MjModel, obj_type: mujoco.mjtObj, name: str) -> int | None:
    obj_id = mujoco.mj_name2id(model, obj_type, name)
    return None if obj_id < 0 else int(obj_id)


def _body_observation(env: SimEnv, name: str) -> dict[str, Any]:
    body_id = _named_id(env.model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id is None:
        return {"name": name, "body_id": None, "available": False}
    return {
        "name": name,
        "body_id": body_id,
        "available": True,
        "world_position_m": _round_list(env.data.xpos[body_id]),
        "world_quaternion_wxyz": _round_list(env.data.xquat[body_id]),
    }


def _site_observation(env: SimEnv, name: str) -> dict[str, Any]:
    site_id = _named_id(env.model, mujoco.mjtObj.mjOBJ_SITE, name)
    if site_id is None:
        return {"name": name, "site_id": None, "available": False}
    return {
        "name": name,
        "site_id": site_id,
        "available": True,
        "world_position_m": _round_list(env.data.site_xpos[site_id]),
        "world_rotation_matrix": _round_list(env.data.site_xmat[site_id].reshape(3, 3).flatten()),
    }


def _camera_pose_observation(env: SimEnv, camera_name: str) -> dict[str, Any]:
    camera_id = _named_id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if camera_id is None:
        return {"name": camera_name, "camera_id": None, "available": False}
    return {
        "name": camera_name,
        "camera_id": camera_id,
        "available": True,
        "world_position_m": _round_list(env.data.cam_xpos[camera_id]),
        "world_rotation_matrix": _round_list(env.data.cam_xmat[camera_id].reshape(3, 3).flatten()),
        "fovy_degrees": _round_float(float(env.model.cam_fovy[camera_id]), 3),
    }


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    checksum = zlib.crc32(kind + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)


def rgb_png_bytes(image: np.ndarray) -> bytes:
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected an RGB uint8 image.")

    height, width, _ = image.shape
    raw_rows = b"".join(b"\x00" + image[row].tobytes() for row in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(raw_rows, 9))
        + _png_chunk(b"IEND", b"")
    )


def image_data_url(image: np.ndarray) -> str:
    return "data:image/png;base64," + base64.b64encode(rgb_png_bytes(image)).decode("ascii")


def _safe_camera_filename(camera_name: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in camera_name)


def _render_camera_observations(
    env: SimEnv,
    camera_names: tuple[str, ...],
    *,
    frame_dir: Path | None = None,
    step_index: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, str]]:
    camera_observations: list[dict[str, Any]] = []
    image_urls: dict[str, str] = {}
    camera_frame_paths: dict[str, str] = {}
    for camera_name in camera_names:
        camera_observation = _camera_pose_observation(env, camera_name)
        if not camera_observation["available"]:
            camera_observations.append(camera_observation)
            continue

        image = render_camera(env.model, env.data, camera_name, env.render_width, env.render_height)
        png_bytes = rgb_png_bytes(image)
        camera_observation["image"] = {
            "width": env.render_width,
            "height": env.render_height,
            "content_type": "image/png",
            "png_bytes": len(png_bytes),
        }
        if frame_dir is not None and step_index is not None:
            frame_dir.mkdir(parents=True, exist_ok=True)
            frame_path = frame_dir / f"step_{step_index:03d}_{_safe_camera_filename(camera_name)}.png"
            frame_path.write_bytes(png_bytes)
            camera_observation["image"]["artifact_path"] = str(frame_path)
            camera_frame_paths[camera_name] = str(frame_path)
        image_urls[camera_name] = "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")
        camera_observations.append(camera_observation)
    return camera_observations, image_urls, camera_frame_paths


def _tag_estimate_observation(estimate: TagPoseEstimate) -> dict[str, Any]:
    return {
        "tag_id": estimate.tag_id,
        "camera_name": estimate.camera_name,
        "world_position_m": _round_list(estimate.world_position),
        "world_rotation_matrix": _round_list(estimate.world_rotation.flatten()),
        "camera_position_m": _round_list(estimate.camera_position),
        "pose_error": _round_float(estimate.pose_error, 6),
    }


def _cup_for_apriltag_id(config: PickupConfig, apriltag_id: int) -> CupSpec | None:
    for cup in (primary_cup_spec(config), second_cup_spec(config)):
        if cup.tag_id == apriltag_id:
            return cup
    return None


def _cup_center_for_tool(context: CupAgentContext, cup: CupSpec) -> tuple[np.ndarray, str]:
    diagnostics = context.diagnostics_by_label.get(cup.label)
    if diagnostics is not None:
        return context.env.data.xpos[diagnostics.cup_body_id].copy(), "simulator_body_pose"

    estimate = context.tag_pose_cache.get_estimate(cup.tag_id)
    if estimate is not None:
        return cup_center_from_tag_estimate(cup, estimate), "cached_apriltag_estimate"

    raise ValueError(f"No diagnostics or AprilTag estimate are available for {cup.label!r}.")


def _place_center_for_tool(context: CupAgentContext) -> tuple[np.ndarray, str]:
    body_id = _named_id(context.env.model, mujoco.mjtObj.mjOBJ_BODY, PLACE_TAG.name)
    if body_id is not None:
        place_tag = context.env.data.xpos[body_id]
        return np.array([place_tag[0], place_tag[1], context.config.cup_half_height], dtype=float), "simulator_place_tag_body_pose"

    place_tag = np.array(context.config.place_tag_position, dtype=float)
    return np.array([place_tag[0], place_tag[1], context.config.cup_half_height], dtype=float), "configured_place_tag"


def _semantic_target_for_apriltag(
    context: CupAgentContext,
    apriltag_id: int,
    target: str,
    tool_point: ToolPointName = FIXED_JAW_TOOL_POINT,
) -> tuple[np.ndarray, CupSpec, ContactDiagnostics, str, tuple[str, ...]]:
    if target not in MOVE_ARM_TARGETS or target == "custom":
        raise ValueError(f"Semantic move target must be one of {MOVE_ARM_TARGETS[1:]}, got {target!r}.")

    cup = _cup_for_apriltag_id(context.config, apriltag_id)
    if cup is not None:
        cup_center, source = _cup_center_for_tool(context, cup)
        if target in {"approach", "grasp", "lift"}:
            target_points = target_points_from_cup_center(cup_center, context.config, None)
            if target == "approach":
                target_xyz = target_points.blue_pregrasp + np.array([0.0, 0.0, context.config.approach_height])
            elif target == "grasp":
                target_xyz = target_points.blue_pregrasp
            else:
                target_xyz = target_points.blue_pregrasp + np.array([0.0, 0.0, context.config.lift_height])
            diagnostics = context.diagnostics_by_label[cup.label]
            return target_xyz, cup, diagnostics, source, ()

        if target in {"place_above", "place"}:
            if context.held_cup_label is None:
                raise ValueError("Cannot place on a cup target because no cup is currently held.")
            held_cup, diagnostics = _active_cup_and_diagnostics(context)
            if held_cup.label == cup.label:
                raise ValueError("Cannot stack a held cup onto itself.")
            target_center = cup_center + np.array([0.0, 0.0, 2.0 * context.config.cup_half_height], dtype=float)
            release_xyz = target_center + _held_tool_offset(context, held_cup, diagnostics, tool_point)
            if target == "place_above":
                target_xyz = release_xyz + np.array([0.0, 0.0, context.config.place_approach_height])
            else:
                target_xyz = release_xyz
            return target_xyz, held_cup, diagnostics, f"simulator_stack_on_{source}", (cup.body_name,)

    if apriltag_id != context.config.place_tag_id:
        raise ValueError(f"AprilTag ID {apriltag_id} is not a configured cup or placement tag.")
    target_center, source = _place_center_for_tool(context)
    active_cup, diagnostics = _active_cup_and_diagnostics(context)
    place_offset = (
        context.held_grasp_offset.copy()
        if context.held_grasp_offset is not None
        else _held_tool_offset(context, active_cup, diagnostics, tool_point)
    )
    release_xyz = target_center + place_offset
    if target == "place_above":
        target_xyz = release_xyz + np.array([0.0, 0.0, context.config.place_approach_height])
    elif target == "place":
        target_xyz = release_xyz
    else:
        target_xyz = release_xyz
    return target_xyz, active_cup, diagnostics, source, (WORK_TABLE_BODY_NAME,)


def _detect_scene_tags(
    env: SimEnv,
    config: PickupConfig,
    camera_names: tuple[str, ...],
    tag_pose_cache: AprilTagPoseCache,
) -> dict[str, Any]:
    tag_sizes = scene_apriltag_sizes(config)
    detected: dict[int, list[dict[str, Any]]] = {}
    errors: list[str] = []
    for tag_id, tag_size in tag_sizes.items():
        try:
            estimates = _detect_tag_estimates(
                env,
                tag_id=tag_id,
                tag_size=tag_size,
                camera_names=camera_names,
                debug_frame_dir=None,
                tag_pose_cache=tag_pose_cache,
                tag_sizes=tag_sizes,
            )
        except Exception as exc:  # Vision is diagnostic; the agent still gets simulator truth.
            errors.append(f"tag {tag_id}: {exc}")
            continue
        detected[tag_id] = [_tag_estimate_observation(estimate) for estimate in estimates]
    return {
        "detected_tags": {str(tag_id): estimates for tag_id, estimates in sorted(detected.items())},
        "errors": errors,
    }


def _cup_observation(
    env: SimEnv,
    config: PickupConfig,
    cup: CupSpec,
    diagnostics: ContactDiagnostics | None,
) -> dict[str, Any]:
    scene_cup = PRIMARY_CUP if cup.body_name == PRIMARY_CUP.body_name else SECOND_CUP
    observation = {
        "label": cup.label,
        "body_name": cup.body_name,
        "freejoint_name": cup.freejoint_name,
        "visual_geom_name": cup.visual_geom_name,
        "configured_initial_position_m": _round_list(cup.initial_position),
        "dimensions_m": {
            "radius": _round_float(config.cup_radius),
            "half_height": _round_float(config.cup_half_height),
            "height": _round_float(2.0 * config.cup_half_height),
            "first_waypoint_clearance": _round_float(config.first_waypoint_clearance),
            "side_grasp_offset": _round_float(config.side_grasp_offset),
            "lateral_grasp_offset": _round_float(config.lateral_grasp_offset),
            "approach_height": _round_float(config.approach_height),
            "lift_height": _round_float(config.lift_height),
        },
        "mounted_tag": {
            "tag_id": cup.tag_id,
            "body_name": scene_cup.tag.name,
            "site_name": f"{scene_cup.tag.name}_site",
            "tag_to_cup_center_offset_m": _round_list(cup.tag_to_cup_center_offset),
        },
        "body": _body_observation(env, cup.body_name),
        "center_site": _site_observation(env, scene_cup.site_name),
    }
    if diagnostics is not None:
        observation["contact_diagnostics"] = _diagnostics_observation(env, diagnostics)
    return observation


def _diagnostics_observation(env: SimEnv, diagnostics: ContactDiagnostics) -> dict[str, Any]:
    return {
        "cup_label": diagnostics.cup_label,
        "steps": diagnostics.steps,
        "cup_contact_steps": diagnostics.cup_contact_steps,
        "fixed_contact_steps": diagnostics.fixed_contact_steps,
        "moving_contact_steps": diagnostics.moving_contact_steps,
        "both_jaw_contact_steps": diagnostics.both_jaw_contact_steps,
        "max_contacts": diagnostics.max_contacts,
        "max_cup_z_m": _round_float(diagnostics.max_cup_z),
        "current_gripper_cup_contacts": [
            {"cup_geom": cup_geom, "gripper_geom": gripper_geom}
            for cup_geom, gripper_geom in current_gripper_cup_contacts(env, diagnostics)
        ],
        "contact_pairs_seen": [
            {"cup_geom": cup_geom, "other_geom": other_geom}
            for cup_geom, other_geom in sorted(diagnostics.contact_pairs)
        ],
    }


def build_observation(
    context: CupAgentContext,
    step_index: int,
    camera_names: tuple[str, ...],
    include_apriltag_estimates: bool = True,
    artifact_dir: Path | None = None,
) -> tuple[dict[str, Any], dict[str, str], dict[str, str]]:
    env = context.env
    env.sync_current_position()
    frame_dir = artifact_dir / "frames" if artifact_dir is not None else None
    camera_observations, image_urls, camera_frame_paths = _render_camera_observations(
        env,
        camera_names,
        frame_dir=frame_dir,
        step_index=step_index,
    )

    primary_cup = primary_cup_spec(context.config)
    second_cup = second_cup_spec(context.config)
    observation: dict[str, Any] = {
        "step_index": step_index,
        "scene_path": str(env.scene_path),
        "model_timestep_s": _round_float(float(env.model.opt.timestep), 6),
        "joint_order": list(JOINT_ORDER),
        "current_joints": {
            joint_name: _round_float(value)
            for joint_name, value in convert_to_dictionary(env.data.qpos.copy()).items()
        },
        "gripper": {
            "open_value": OPEN_GRIPPER,
            "closed_value": CLOSED_GRIPPER,
            "held_cup_label": context.held_cup_label,
        },
        "objects": {
            "cups": [
                _cup_observation(env, context.config, primary_cup, context.diagnostics_by_label.get(primary_cup.label)),
                _cup_observation(env, context.config, second_cup, context.diagnostics_by_label.get(second_cup.label)),
            ],
            "table_tags": [
                {
                    "label": "placement tag",
                    "tag_id": context.config.place_tag_id,
                    "body_name": PLACE_TAG.name,
                    "body": _body_observation(env, PLACE_TAG.name),
                    "configured_position_m": _round_list(context.config.place_tag_position),
                }
            ],
            "robot": {
                "fixed_jaw": _body_observation(env, "gripper"),
                "moving_jaw": _body_observation(env, "moving_jaw_so101_v1"),
            },
        },
        "cameras": camera_observations,
        "camera_frame_paths": camera_frame_paths,
        "last_tool_result": context.last_tool_result,
        "attempt_guidance": context.attempt_guidance,
        "task_state": _cup_pour_task_state(context),
    }

    if include_apriltag_estimates:
        observation["apriltag_estimates"] = _detect_scene_tags(
            env,
            context.config,
            camera_names,
            context.tag_pose_cache,
        )

    return observation, image_urls, camera_frame_paths


def observation_to_input_items(
    observation: dict[str, Any],
    image_urls: dict[str, str],
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [
        {
            "type": "input_text",
            "text": (
                "You are controlling a MuJoCo SO-101 robot arm in simulation. "
                "Use exactly one tool when you want to act. Current observation JSON:\n"
                + json.dumps(observation, sort_keys=True, separators=(",", ":"))
            ),
        }
    ]
    for camera_name, data_url in image_urls.items():
        content.append({"type": "input_text", "text": f"Camera view: {camera_name}"})
        content.append({"type": "input_image", "image_url": data_url, "detail": "high"})
    return [{"role": "user", "content": content}]


def _tool_result(success: bool, action: str, **extra: Any) -> str:
    result = {"success": success, "action": action, **extra}
    return json.dumps(result, sort_keys=True)


def _active_cup_and_diagnostics(context: CupAgentContext) -> tuple[CupSpec, ContactDiagnostics]:
    cups = (primary_cup_spec(context.config), second_cup_spec(context.config))
    if context.held_cup_label is not None:
        for cup in cups:
            if cup.label == context.held_cup_label:
                return cup, context.diagnostics_by_label[cup.label]
    cup = cups[0]
    return cup, context.diagnostics_by_label[cup.label]


def _current_tool_target_xyz(context: CupAgentContext, tool_point: ToolPointName) -> np.ndarray:
    gripperframe_pose = context.env.kinematics.forward_kinematics(context.env.current_position, frame="mujoco")
    tool_pose = gripperframe_pose_to_tool_target_pose(gripperframe_pose, tool_point)
    return tool_pose[:3, 3].copy()


def _held_tool_offset(
    context: CupAgentContext,
    held_cup: CupSpec,
    diagnostics: ContactDiagnostics,
    tool_point: ToolPointName,
) -> np.ndarray:
    if context.held_cup_label != held_cup.label:
        if context.held_grasp_offset is not None:
            return context.held_grasp_offset.copy()
        return np.zeros(3, dtype=float)
    held_center = context.env.data.xpos[diagnostics.cup_body_id].copy()
    return _current_tool_target_xyz(context, tool_point) - held_center


def _update_held_cup_after_close(context: CupAgentContext) -> None:
    for label, diagnostics in context.diagnostics_by_label.items():
        if current_gripper_cup_contacts(context.env, diagnostics):
            context.held_cup_label = label
            cup = primary_cup_spec(context.config) if label == primary_cup_spec(context.config).label else second_cup_spec(context.config)
            cup_center, _source = _cup_center_for_tool(context, cup)
            target_points = target_points_from_cup_center(cup_center, context.config, None)
            context.held_grasp_offset = target_points.blue_pregrasp - cup_center
            return


def _retreat_after_release(context: CupAgentContext, cup: CupSpec, diagnostics: ContactDiagnostics) -> bool:
    retreat_xyz = _current_fixed_jaw_target_xyz(context)
    cup_center = context.env.data.xpos[diagnostics.cup_body_id].copy()
    retreat_xy = retreat_xyz[:2] - cup_center[:2]
    retreat_norm = float(np.linalg.norm(retreat_xy))
    if retreat_norm < 1e-6:
        fallback_xy = cup_center[:2]
        fallback_norm = float(np.linalg.norm(fallback_xy))
        retreat_xy = -fallback_xy / fallback_norm if fallback_norm > 1e-6 else np.array([-1.0, 0.0], dtype=float)
    else:
        retreat_xy = retreat_xy / retreat_norm

    retreat_xyz[:2] += retreat_xy * context.config.release_clearance
    retreat_xyz[2] += context.config.place_approach_height
    retreat_position = solve_target(context.env, retreat_xyz, OPEN_GRIPPER, FIXED_JAW_TOOL_POINT)
    return command_motion(
        context.env,
        retreat_position,
        context.move_duration,
        diagnostics,
        viewer=context.viewer,
        realtime=context.realtime,
        planner_config=context.config.motion_planner,
        collision_context=planner_context_for_cup(diagnostics, cup, allow_gripper_cup_contact=True),
    )


def _current_fixed_jaw_target_xyz(context: CupAgentContext) -> np.ndarray:
    return _current_tool_target_xyz(context, FIXED_JAW_TOOL_POINT)


def _last_move_was_stack_place(context: CupAgentContext) -> bool:
    result = context.last_tool_result or {}
    return (
        result.get("action") == "move_arm"
        and result.get("target") == "place"
        and str(result.get("target_source", "")).startswith("simulator_stack_on_")
    )


def _joint_range_degrees(context: CupAgentContext, joint_name: str) -> tuple[float, float]:
    joint_id = _named_id(context.env.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id is None:
        raise ValueError(f"Could not find MuJoCo joint named {joint_name!r}.")
    lower, upper = context.env.model.jnt_range[joint_id]
    return float(np.rad2deg(lower)), float(np.rad2deg(upper))


def _clamp_joint_degrees(context: CupAgentContext, joint_name: str, value: float) -> float:
    lower, upper = _joint_range_degrees(context, joint_name)
    return float(np.clip(value, lower, upper))


def _pour_tool_target_xyz(
    context: CupAgentContext,
    receiver_cup: CupSpec,
    held_cup: CupSpec,
    held_diagnostics: ContactDiagnostics,
    tool_point: ToolPointName,
) -> tuple[np.ndarray, np.ndarray]:
    receiver_center, _source = _cup_center_for_tool(context, receiver_cup)
    held_center = context.env.data.xpos[held_diagnostics.cup_body_id].copy()
    offset_xy = held_center[:2] - receiver_center[:2]
    offset_norm = float(np.linalg.norm(offset_xy))
    if offset_norm <= 1e-6:
        offset_xy = receiver_center[:2].copy()
        offset_norm = float(np.linalg.norm(offset_xy))
    pour_direction = offset_xy / offset_norm if offset_norm > 1e-6 else np.array([1.0, 0.0], dtype=float)
    pour_center = receiver_center.copy()
    # Wrist roll tilts the held cup around its center, so keep the center
    # half a cup-height back from the receiver. After the tilt, the rim/top
    # sweeps over the receiver opening instead of the cup center hovering there.
    rim_alignment_offset = context.config.cup_half_height + context.config.pour_lateral_offset
    pour_center[:2] = receiver_center[:2] + pour_direction * rim_alignment_offset
    pour_center[2] = receiver_center[2] + context.config.cup_half_height + context.config.pour_height
    return pour_center + _held_tool_offset(context, held_cup, held_diagnostics, tool_point), receiver_center


def execute_pour_into(
    context: CupAgentContext,
    apriltag_id: int,
    *,
    duration: float | None = None,
    tilt_degrees: float | None = None,
) -> dict[str, Any]:
    """Move the held cup above a receiver cup and roll the wrist for a pour gesture."""
    if context.held_cup_label is None:
        context.last_tool_result = {
            "success": False,
            "action": "pour_into",
            "error": "Cannot pour because no cup is currently held.",
        }
        return dict(context.last_tool_result)

    receiver_cup = _cup_for_apriltag_id(context.config, apriltag_id)
    if receiver_cup is None:
        context.last_tool_result = {
            "success": False,
            "action": "pour_into",
            "error": f"AprilTag {apriltag_id} is not a cup tag.",
        }
        return dict(context.last_tool_result)

    held_cup, held_diagnostics = _active_cup_and_diagnostics(context)
    if held_cup.label == receiver_cup.label:
        context.last_tool_result = {
            "success": False,
            "action": "pour_into",
            "error": "Cannot pour a held cup into itself.",
        }
        return dict(context.last_tool_result)

    try:
        tool_point = FIXED_JAW_TOOL_POINT
        pour_xyz, receiver_center = _pour_tool_target_xyz(context, receiver_cup, held_cup, held_diagnostics, tool_point)
        gripper_position = float(context.env.current_position.get("gripper", CLOSED_GRIPPER))
        approach_position = solve_target(context.env, pour_xyz, gripper_position, tool_point)
        collision_context = planner_context_for_cup(
            held_diagnostics,
            held_cup,
            attach_cup=True,
            allow_gripper_cup_contact=True,
            allowed_support_body_names=(receiver_cup.body_name,),
        )
        success = command_motion(
            context.env,
            approach_position,
            context.move_duration if duration is None else float(duration),
            held_diagnostics,
            viewer=context.viewer,
            realtime=context.realtime,
            planner_config=context.config.motion_planner,
            collision_context=collision_context,
        )
        tilt_position = dict(context.env.current_position)
        if success:
            requested_tilt = context.config.pour_tilt_degrees if tilt_degrees is None else float(tilt_degrees)
            tilt_position["wrist_roll"] = _clamp_joint_degrees(
                context,
                "wrist_roll",
                HORIZONTAL_WRIST_ROLL_DEGREES + requested_tilt,
            )
            success = command_motion(
                context.env,
                tilt_position,
                context.config.pour_duration,
                held_diagnostics,
                viewer=context.viewer,
                realtime=context.realtime,
                collision_context=collision_context,
            )
        if success:
            success = hold_command(
                context.env,
                tilt_position,
                context.config.pour_hold_duration,
                held_diagnostics,
                viewer=context.viewer,
                realtime=context.realtime,
                collision_context=collision_context,
            )

        pour_record = {
            "source_cup_label": held_cup.label,
            "receiver_cup_label": receiver_cup.label,
            "receiver_apriltag_id": apriltag_id,
            "pour_xyz_m": _round_list(pour_xyz),
            "receiver_center_m": _round_list(receiver_center),
            "tilt_degrees": _round_float(tilt_position["wrist_roll"] - HORIZONTAL_WRIST_ROLL_DEGREES),
        }
        if success:
            context.completed_pour = pour_record
        context.last_tool_result = {"success": success, "action": "pour_into", **pour_record}
        return dict(context.last_tool_result)
    except Exception as exc:
        context.last_tool_result = {"success": False, "action": "pour_into", "error": str(exc)}
        return dict(context.last_tool_result)


def parse_tool_output(output: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(output)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) and isinstance(payload.get("action"), str) else None


def evaluate_pick_place_success(context: CupAgentContext) -> dict[str, Any]:
    primary_cup = primary_cup_spec(context.config)
    diagnostics = context.diagnostics_by_label[primary_cup.label]
    final_center = context.env.data.xpos[diagnostics.cup_body_id].copy()
    initial_center = np.array(primary_cup.initial_position, dtype=float)
    place_tag = np.array(context.config.place_tag_position, dtype=float)
    target_center = np.array([place_tag[0], place_tag[1], context.config.cup_half_height], dtype=float)
    xy_error = float(np.linalg.norm(final_center[:2] - target_center[:2]))
    z_error = float(abs(final_center[2] - target_center[2]))
    lift_delta = float(diagnostics.max_cup_z - initial_center[2])
    current_joints = convert_to_dictionary(context.env.data.qpos.copy())
    gripper_open = bool(current_joints["gripper"] >= (OPEN_GRIPPER + CLOSED_GRIPPER) * 0.5)
    released = bool(context.held_cup_label is None and not current_gripper_cup_contacts(context.env, diagnostics))
    lifted = bool(lift_delta >= context.config.success_lift_delta)
    placed = bool(xy_error <= context.config.place_success_xy_tolerance and z_error <= context.config.place_success_z_tolerance)
    success = bool(lifted and placed and gripper_open and released)
    failure_reasons: list[str] = []
    if not lifted:
        failure_reasons.append(
            f"cup lift delta {lift_delta:.4f} m below threshold {context.config.success_lift_delta:.4f} m"
        )
    if not placed:
        failure_reasons.append(
            "cup final pose outside placement tolerance "
            f"(xy_error={xy_error:.4f} m, z_error={z_error:.4f} m)"
        )
    if not gripper_open:
        failure_reasons.append(f"gripper is not open enough (gripper={current_joints['gripper']:.2f})")
    if not released:
        failure_reasons.append("cup still appears held or touching the gripper")

    return {
        "success": success,
        "failure_reasons": failure_reasons,
        "primary_cup_label": primary_cup.label,
        "final_center_m": _round_list(final_center),
        "target_center_m": _round_list(target_center),
        "xy_error_m": _round_float(xy_error, 6),
        "z_error_m": _round_float(z_error, 6),
        "lift_delta_m": _round_float(lift_delta, 6),
        "max_cup_z_m": _round_float(diagnostics.max_cup_z),
        "initial_cup_z_m": _round_float(initial_center[2]),
        "gripper_position": _round_float(current_joints["gripper"]),
        "gripper_open": gripper_open,
        "released": released,
        "held_cup_label": context.held_cup_label,
        "contact_diagnostics": _diagnostics_observation(context.env, diagnostics),
    }


def evaluate_cup_stack_success(context: CupAgentContext) -> dict[str, Any]:
    first_cup = primary_cup_spec(context.config)
    second_cup = second_cup_spec(context.config)
    first_diagnostics = context.diagnostics_by_label[first_cup.label]
    second_diagnostics = context.diagnostics_by_label[second_cup.label]

    first_center = context.env.data.xpos[first_diagnostics.cup_body_id].copy()
    second_center = context.env.data.xpos[second_diagnostics.cup_body_id].copy()
    first_initial = np.array(first_cup.initial_position, dtype=float)
    second_initial = np.array(second_cup.initial_position, dtype=float)
    place_tag = np.array(context.config.place_tag_position, dtype=float)
    first_target_center = np.array([place_tag[0], place_tag[1], context.config.cup_half_height], dtype=float)
    second_target_center = np.array(
        [first_center[0], first_center[1], first_center[2] + 2.0 * context.config.cup_half_height],
        dtype=float,
    )

    first_xy_error = float(np.linalg.norm(first_center[:2] - first_target_center[:2]))
    first_z_error = float(abs(first_center[2] - first_target_center[2]))
    stack_xy_error = float(np.linalg.norm(second_center[:2] - first_center[:2]))
    stack_z_error = float(abs((second_center[2] - first_center[2]) - 2.0 * context.config.cup_half_height))
    first_lift_delta = float(first_diagnostics.max_cup_z - first_initial[2])
    second_lift_delta = float(second_diagnostics.max_cup_z - second_initial[2])

    current_joints = convert_to_dictionary(context.env.data.qpos.copy())
    gripper_open = bool(current_joints["gripper"] >= (OPEN_GRIPPER + CLOSED_GRIPPER) * 0.5)
    released = bool(
        context.held_cup_label is None
        and not current_gripper_cup_contacts(context.env, first_diagnostics)
        and not current_gripper_cup_contacts(context.env, second_diagnostics)
    )
    first_lifted = bool(first_lift_delta >= context.config.success_lift_delta)
    second_lifted = bool(second_lift_delta >= context.config.success_lift_delta)
    first_placed = bool(
        first_xy_error <= context.config.place_success_xy_tolerance
        and first_z_error <= context.config.place_success_z_tolerance
    )
    second_stacked = bool(
        stack_xy_error <= context.config.place_success_xy_tolerance
        and stack_z_error <= context.config.place_success_z_tolerance
    )
    success = bool(first_lifted and first_placed and second_lifted and second_stacked and gripper_open and released)

    failure_reasons: list[str] = []
    if not first_lifted:
        failure_reasons.append(
            f"first cup lift delta {first_lift_delta:.4f} m below threshold {context.config.success_lift_delta:.4f} m"
        )
    if not first_placed:
        failure_reasons.append(
            "first cup final pose outside placement tolerance "
            f"(xy_error={first_xy_error:.4f} m, z_error={first_z_error:.4f} m)"
        )
    if not second_lifted:
        failure_reasons.append(
            f"second cup lift delta {second_lift_delta:.4f} m below threshold {context.config.success_lift_delta:.4f} m"
        )
    if not second_stacked:
        failure_reasons.append(
            "second cup is not stacked on the first cup within tolerance "
            f"(xy_error={stack_xy_error:.4f} m, z_error={stack_z_error:.4f} m)"
        )
    if not gripper_open:
        failure_reasons.append(f"gripper is not open enough (gripper={current_joints['gripper']:.2f})")
    if not released:
        failure_reasons.append("one or more cups still appear held or touching the gripper")

    return {
        "success": success,
        "failure_reasons": failure_reasons,
        "task": "cup_stack",
        "first_cup_label": first_cup.label,
        "second_cup_label": second_cup.label,
        "first_final_center_m": _round_list(first_center),
        "first_target_center_m": _round_list(first_target_center),
        "second_final_center_m": _round_list(second_center),
        "second_target_center_m": _round_list(second_target_center),
        "first_xy_error_m": _round_float(first_xy_error, 6),
        "first_z_error_m": _round_float(first_z_error, 6),
        "stack_xy_error_m": _round_float(stack_xy_error, 6),
        "stack_z_error_m": _round_float(stack_z_error, 6),
        "first_lift_delta_m": _round_float(first_lift_delta, 6),
        "second_lift_delta_m": _round_float(second_lift_delta, 6),
        "lift_delta_m": _round_float(min(first_lift_delta, second_lift_delta), 6),
        "xy_error_m": _round_float(max(first_xy_error, stack_xy_error), 6),
        "z_error_m": _round_float(max(first_z_error, stack_z_error), 6),
        "first_max_cup_z_m": _round_float(first_diagnostics.max_cup_z),
        "second_max_cup_z_m": _round_float(second_diagnostics.max_cup_z),
        "gripper_position": _round_float(current_joints["gripper"]),
        "gripper_open": gripper_open,
        "released": released,
        "held_cup_label": context.held_cup_label,
        "first_contact_diagnostics": _diagnostics_observation(context.env, first_diagnostics),
        "second_contact_diagnostics": _diagnostics_observation(context.env, second_diagnostics),
    }


def evaluate_cup_pour_success(context: CupAgentContext) -> dict[str, Any]:
    first_cup = primary_cup_spec(context.config)
    second_cup = second_cup_spec(context.config)
    first_diagnostics = context.diagnostics_by_label[first_cup.label]
    second_diagnostics = context.diagnostics_by_label[second_cup.label]

    first_center = context.env.data.xpos[first_diagnostics.cup_body_id].copy()
    second_center = context.env.data.xpos[second_diagnostics.cup_body_id].copy()
    first_initial = np.array(first_cup.initial_position, dtype=float)
    second_initial = np.array(second_cup.initial_position, dtype=float)
    place_tag = np.array(context.config.place_tag_position, dtype=float)
    first_target_center = np.array([place_tag[0], place_tag[1], context.config.cup_half_height], dtype=float)

    first_xy_error = float(np.linalg.norm(first_center[:2] - first_target_center[:2]))
    first_z_error = float(abs(first_center[2] - first_target_center[2]))
    pour_xy_error = float(np.linalg.norm(second_center[:2] - first_center[:2]))
    first_lift_delta = float(first_diagnostics.max_cup_z - first_initial[2])
    second_lift_delta = float(second_diagnostics.max_cup_z - second_initial[2])

    current_joints = convert_to_dictionary(context.env.data.qpos.copy())
    gripper_open = bool(current_joints["gripper"] >= (OPEN_GRIPPER + CLOSED_GRIPPER) * 0.5)
    released = bool(context.held_cup_label is None)
    first_lifted = bool(first_lift_delta >= context.config.success_lift_delta)
    second_lifted = bool(second_lift_delta >= context.config.success_lift_delta)
    first_placed = bool(
        first_xy_error <= context.config.place_success_xy_tolerance
        and first_z_error <= context.config.place_success_z_tolerance
    )
    pour_record = context.completed_pour or {}
    second_poured = bool(
        pour_record.get("source_cup_label") == second_cup.label
        and pour_record.get("receiver_cup_label") == first_cup.label
    )
    success = bool(first_lifted and first_placed and second_lifted and second_poured)

    failure_reasons: list[str] = []
    if not first_lifted:
        failure_reasons.append(
            f"first cup lift delta {first_lift_delta:.4f} m below threshold {context.config.success_lift_delta:.4f} m"
        )
    if not first_placed:
        failure_reasons.append(
            "first cup final pose outside placement tolerance "
            f"(xy_error={first_xy_error:.4f} m, z_error={first_z_error:.4f} m)"
        )
    if not second_lifted:
        failure_reasons.append(
            f"second cup lift delta {second_lift_delta:.4f} m below threshold {context.config.success_lift_delta:.4f} m"
        )
    if not second_poured:
        failure_reasons.append("second cup has not completed a pour gesture into the first cup")

    return {
        "success": success,
        "failure_reasons": failure_reasons,
        "task": "cup_pour",
        "first_cup_label": first_cup.label,
        "second_cup_label": second_cup.label,
        "first_final_center_m": _round_list(first_center),
        "first_target_center_m": _round_list(first_target_center),
        "second_final_center_m": _round_list(second_center),
        "first_xy_error_m": _round_float(first_xy_error, 6),
        "first_z_error_m": _round_float(first_z_error, 6),
        "pour_xy_error_m": _round_float(pour_xy_error, 6),
        "first_lift_delta_m": _round_float(first_lift_delta, 6),
        "second_lift_delta_m": _round_float(second_lift_delta, 6),
        "lift_delta_m": _round_float(min(first_lift_delta, second_lift_delta), 6),
        "xy_error_m": _round_float(max(first_xy_error, pour_xy_error), 6),
        "z_error_m": _round_float(first_z_error, 6),
        "first_max_cup_z_m": _round_float(first_diagnostics.max_cup_z),
        "second_max_cup_z_m": _round_float(second_diagnostics.max_cup_z),
        "gripper_position": _round_float(current_joints["gripper"]),
        "gripper_open": gripper_open,
        "released": released,
        "held_cup_label": context.held_cup_label,
        "completed_pour": pour_record or None,
        "second_poured_into_first": second_poured,
        "first_contact_diagnostics": _diagnostics_observation(context.env, first_diagnostics),
        "second_contact_diagnostics": _diagnostics_observation(context.env, second_diagnostics),
    }


def _cup_pour_task_state(context: CupAgentContext) -> dict[str, Any]:
    evaluation = evaluate_cup_pour_success(context)
    config = context.config
    first_placed = bool(
        evaluation["first_xy_error_m"] <= config.place_success_xy_tolerance
        and evaluation["first_z_error_m"] <= config.place_success_z_tolerance
    )
    second_poured = bool(evaluation["second_poured_into_first"])
    if evaluation["success"]:
        next_phase = "done"
        allowed_next_cup_tag_id = None
    elif context.held_cup_label == evaluation["first_cup_label"]:
        next_phase = "place_first_cup_on_tag_0"
        allowed_next_cup_tag_id = None
    elif context.held_cup_label == evaluation["second_cup_label"]:
        next_phase = "pour_second_cup_into_tag_6"
        allowed_next_cup_tag_id = config.cup_tag_id
    elif not first_placed:
        next_phase = "pick_first_cup_tag_6"
        allowed_next_cup_tag_id = config.cup_tag_id
    elif not second_poured:
        next_phase = "pick_second_cup_tag_1"
        allowed_next_cup_tag_id = config.second_cup_tag_id
    else:
        next_phase = "done"
        allowed_next_cup_tag_id = None

    return {
        "next_phase": next_phase,
        "held_cup_label": context.held_cup_label,
        "first_placed_on_tag_0": first_placed,
        "second_poured_into_first": second_poured,
        "allowed_next_cup_tag_id": allowed_next_cup_tag_id,
        "first_cup_tag_id": config.cup_tag_id,
        "second_cup_tag_id": config.second_cup_tag_id,
        "place_tag_id": config.place_tag_id,
        "pour_receiver_tag_id": config.cup_tag_id,
        "pour_tilt_degrees": config.pour_tilt_degrees,
    }


def _cup_stack_task_state(context: CupAgentContext) -> dict[str, Any]:
    evaluation = evaluate_cup_stack_success(context)
    config = context.config
    first_placed = bool(
        evaluation["first_xy_error_m"] <= config.place_success_xy_tolerance
        and evaluation["first_z_error_m"] <= config.place_success_z_tolerance
    )
    second_stacked = bool(
        evaluation["stack_xy_error_m"] <= config.place_success_xy_tolerance
        and evaluation["stack_z_error_m"] <= config.place_success_z_tolerance
    )
    if evaluation["success"]:
        next_phase = "done"
        allowed_next_cup_tag_id = None
    elif context.held_cup_label == evaluation["first_cup_label"]:
        next_phase = "place_first_cup_on_tag_0"
        allowed_next_cup_tag_id = None
    elif context.held_cup_label == evaluation["second_cup_label"]:
        next_phase = "stack_second_cup_on_tag_6"
        allowed_next_cup_tag_id = config.cup_tag_id
    elif not first_placed:
        next_phase = "pick_first_cup_tag_6"
        allowed_next_cup_tag_id = config.cup_tag_id
    elif not second_stacked:
        next_phase = "pick_second_cup_tag_1"
        allowed_next_cup_tag_id = config.second_cup_tag_id
    else:
        next_phase = "done"
        allowed_next_cup_tag_id = None

    return {
        "next_phase": next_phase,
        "held_cup_label": context.held_cup_label,
        "first_placed_on_tag_0": first_placed,
        "second_stacked_on_first": second_stacked,
        "allowed_next_cup_tag_id": allowed_next_cup_tag_id,
        "first_cup_tag_id": config.cup_tag_id,
        "second_cup_tag_id": config.second_cup_tag_id,
        "place_tag_id": config.place_tag_id,
        "stack_place_tool_point": "claw_center",
        "stack_open_duration_s": config.stack_release_open_duration,
    }


def _compact_observation_summary(observation: dict[str, Any]) -> dict[str, Any]:
    cups = observation["objects"]["cups"]
    return {
        "step_index": observation["step_index"],
        "joints": observation["current_joints"],
        "held_cup_label": observation["gripper"]["held_cup_label"],
        "task_state": observation.get("task_state"),
        "cups": [
            {
                "label": cup["label"],
                "tag_id": cup["mounted_tag"]["tag_id"],
                "world_position_m": cup["body"].get("world_position_m"),
                "contacts": cup.get("contact_diagnostics", {}).get("current_gripper_cup_contacts", []),
            }
            for cup in cups
        ],
        "cameras": [
            {
                "name": camera["name"],
                "available": camera["available"],
                "image": camera.get("image"),
            }
            for camera in observation["cameras"]
        ],
        "last_tool_result": observation.get("last_tool_result"),
    }


def pretty_print_agent_step(
    *,
    step_index: int,
    observation: dict[str, Any],
    final_output: str,
    tool_output: dict[str, Any] | None,
    evaluation: dict[str, Any],
) -> None:
    print(f"\n=== Agent step {step_index:03d} ===")
    camera_parts = []
    for camera in observation["cameras"]:
        image = camera.get("image") or {}
        if camera["available"]:
            camera_parts.append(
                f"{camera['name']} {image.get('width')}x{image.get('height')} "
                f"{image.get('png_bytes', 0)} bytes"
            )
        else:
            camera_parts.append(f"{camera['name']} unavailable")
    print("cameras: " + ", ".join(camera_parts))
    print(f"held cup: {observation['gripper']['held_cup_label']}")
    for cup in observation["objects"]["cups"]:
        diagnostics = cup.get("contact_diagnostics", {})
        contacts = diagnostics.get("current_gripper_cup_contacts", [])
        print(
            f"cup {cup['label']} tag={cup['mounted_tag']['tag_id']} "
            f"pos={cup['body'].get('world_position_m')} contacts={len(contacts)}"
        )
    if tool_output is None:
        print(f"model final: {final_output}")
    else:
        print(
            "tool: "
            + json.dumps(
                {
                    key: tool_output.get(key)
                    for key in ("action", "success", "apriltag_id", "target", "target_xyz_m", "held_cup_label")
                    if key in tool_output
                },
                sort_keys=True,
            )
        )
    print(
        "metrics: "
        f"lift_delta={evaluation['lift_delta_m']}m "
        f"xy_error={evaluation['xy_error_m']}m "
        f"z_error={evaluation['z_error_m']}m "
        f"released={evaluation['released']}"
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def create_cup_agent(model: str = DEFAULT_AGENT_MODEL) -> Any:
    from agents import Agent, RunContextWrapper, function_tool  # type: ignore[import-not-found]

    globals()["RunContextWrapper"] = RunContextWrapper

    @function_tool
    def move_arm(
        ctx: RunContextWrapper[CupAgentContext],
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        apriltag_id: int | None = None,
        target: str = "custom",
        duration: float | None = None,
        tool_point: ToolPointName = FIXED_JAW_TOOL_POINT,
    ) -> str:
        """Move the robot arm tool point using a semantic AprilTag target or raw XYZ.

        Args:
            x: Optional world-frame X target in meters. Use only for custom fallback moves.
            y: Optional world-frame Y target in meters. Use only for custom fallback moves.
            z: Optional world-frame Z target in meters. Use only for custom fallback moves.
            apriltag_id: Recommended semantic target ID. Use cup tags with target approach/grasp/lift.
            target: One of custom, approach, grasp, lift, or place_above. Prefer semantic targets over raw XYZ.
            duration: Optional motion duration in seconds.
            tool_point: Tool point to place at the target. Use fixed_jaw_tip unless there is a clear reason.
        """
        context = ctx.context
        try:
            allowed_support_body_names: tuple[str, ...] = ()
            if apriltag_id is not None and target != "custom":
                target_xyz, active_cup, diagnostics, target_source, allowed_support_body_names = _semantic_target_for_apriltag(
                    context,
                    apriltag_id,
                    target,
                    tool_point,
                )
            else:
                if x is None or y is None or z is None:
                    raise ValueError("Raw XYZ moves require x, y, and z. Prefer apriltag_id with a semantic target.")
                target_xyz = np.array([x, y, z], dtype=float)
                active_cup, diagnostics = _active_cup_and_diagnostics(context)
                target_source = "raw_xyz"

            gripper_position = float(context.env.current_position.get("gripper", OPEN_GRIPPER))
            target_position = solve_target(context.env, target_xyz, gripper_position, tool_point)
            collision_context = planner_context_for_cup(
                diagnostics,
                active_cup,
                attach_cup=context.held_cup_label == active_cup.label,
                allow_gripper_cup_contact=True,
                allowed_support_body_names=allowed_support_body_names,
            )
            success = command_motion(
                context.env,
                target_position,
                context.move_duration if duration is None else float(duration),
                diagnostics,
                viewer=context.viewer,
                realtime=context.realtime,
                planner_config=context.config.motion_planner,
                collision_context=collision_context,
            )
            context.last_tool_result = {
                "action": "move_arm",
                "success": success,
                "target_xyz_m": _round_list(target_xyz),
                "tool_point": tool_point,
                "apriltag_id": apriltag_id,
                "target": target,
                "target_source": target_source,
            }
            return _tool_result(
                success,
                "move_arm",
                target_xyz_m=_round_list(target_xyz),
                tool_point=tool_point,
                apriltag_id=apriltag_id,
                target=target,
                target_source=target_source,
            )
        except Exception as exc:
            context.last_tool_result = {"action": "move_arm", "success": False, "error": str(exc)}
            return _tool_result(False, "move_arm", error=str(exc))

    @function_tool
    def open_gripper(ctx: RunContextWrapper[CupAgentContext], duration: float | None = None) -> str:
        """Open the gripper and release any currently held cup."""
        context = ctx.context
        target_position = dict(context.env.current_position)
        target_position["gripper"] = OPEN_GRIPPER
        was_holding = context.held_cup_label is not None
        stack_release = _last_move_was_stack_place(context)
        cup, diagnostics = _active_cup_and_diagnostics(context)
        try:
            release_duration = (
                context.config.stack_release_open_duration
                if duration is None and stack_release
                else context.gripper_duration if duration is None else float(duration)
            )
            success = command_motion(
                context.env,
                target_position,
                release_duration,
                diagnostics,
                viewer=context.viewer,
                realtime=context.realtime,
            )
            if success:
                context.held_cup_label = None
                context.held_grasp_offset = None
            release_retreat = False
            release_settle = False
            if success and stack_release:
                release_settle = hold_command(
                    context.env,
                    target_position,
                    max(context.config.release_contact_settle_duration, context.config.stack_release_settle_duration),
                    diagnostics,
                    viewer=context.viewer,
                    realtime=context.realtime,
                )
                success = release_settle
            elif success and was_holding:
                release_retreat = _retreat_after_release(context, cup, diagnostics)
                success = release_retreat
            context.last_tool_result = {
                "action": "open_gripper",
                "success": success,
                "release_retreat": release_retreat,
                "stack_release": stack_release,
                "release_settle": release_settle,
            }
            return _tool_result(
                success,
                "open_gripper",
                release_retreat=release_retreat,
                stack_release=stack_release,
                release_settle=release_settle,
            )
        except Exception as exc:
            context.last_tool_result = {"action": "open_gripper", "success": False, "error": str(exc)}
            return _tool_result(False, "open_gripper", error=str(exc))

    @function_tool
    def close_gripper(
        ctx: RunContextWrapper[CupAgentContext],
        duration: float | None = None,
        squeeze_duration: float | None = None,
    ) -> str:
        """Close the gripper around the object currently between the jaws."""
        context = ctx.context
        target_position = dict(context.env.current_position)
        target_position["gripper"] = CLOSED_GRIPPER
        cup, diagnostics = _active_cup_and_diagnostics(context)
        try:
            success = command_motion(
                context.env,
                target_position,
                context.gripper_duration if duration is None else float(duration),
                diagnostics,
                viewer=context.viewer,
                realtime=context.realtime,
            )
            if success:
                success = hold_command(
                    context.env,
                    target_position,
                    context.config.squeeze_duration if squeeze_duration is None else float(squeeze_duration),
                    diagnostics,
                    viewer=context.viewer,
                    realtime=context.realtime,
                )
            if success:
                _update_held_cup_after_close(context)
            context.last_tool_result = {
                "action": "close_gripper",
                "success": success,
                "held_cup_label": context.held_cup_label,
            }
            return _tool_result(success, "close_gripper", held_cup_label=context.held_cup_label)
        except Exception as exc:
            context.last_tool_result = {"action": "close_gripper", "success": False, "error": str(exc)}
            return _tool_result(False, "close_gripper", error=str(exc))

    @function_tool
    def pour_into(
        ctx: RunContextWrapper[CupAgentContext],
        apriltag_id: int,
        duration: float | None = None,
        tilt_degrees: float | None = None,
    ) -> str:
        """Pour from the currently held cup into the cup with the given AprilTag ID."""
        return json.dumps(
            execute_pour_into(ctx.context, apriltag_id, duration=duration, tilt_degrees=tilt_degrees),
            sort_keys=True,
        )

    instructions = (
        "You are a careful robot manipulation agent for a MuJoCo SO-101 cup scene. "
        "At each step you receive object poses, IDs, joint state, contact diagnostics, and camera screenshots. "
        "Choose at most one action tool per step. Use this deterministic reference policy as your default "
        "example for how to pick and place the first cup: "
        "1. move_arm(apriltag_id=6, target='approach') to approach above the pregrasp point with the gripper open. "
        "2. move_arm(apriltag_id=6, target='grasp') to descend to the pregrasp point. "
        "3. close_gripper() to close and squeeze the cup. "
        "4. move_arm(apriltag_id=6, target='lift') to lift the held cup. "
        "5. move_arm(apriltag_id=0, target='place_above') to move above the flat placement tag. "
        "6. move_arm(apriltag_id=0, target='place') to descend to the release height using the stored grasp offset. "
        "7. open_gripper() to release the cup at the placement target. "
        "Prefer this semantic sequence over raw XYZ because the tools compute cup dimensions, tag offsets, "
        "and gripper offsets for approach, grasp, lift, and placement motions. "
        "For two-cup pouring tasks, place the first cup on the tag, pick and lift the second cup, then call "
        "pour_into(apriltag_id=6) while holding the second cup instead of placing it on top of the first. "
        "The task is to pick up the first cup, move it to the placement tag, and release it there using only "
        "move_arm, open_gripper, close_gripper, and pour_into. "
        "If the task is complete or impossible, respond with a concise status message instead of calling a tool."
    )
    return Agent[CupAgentContext](
        name="MuJoCo Cup Manipulation Agent",
        instructions=instructions,
        model=model,
        tools=[move_arm, open_gripper, close_gripper, pour_into],
        tool_use_behavior="stop_on_first_tool",
    )


async def run_agent_steps(
    context: CupAgentContext,
    *,
    model: str = DEFAULT_AGENT_MODEL,
    max_agent_steps: int = DEFAULT_MAX_AGENT_STEPS,
    camera_names: tuple[str, ...] = DEFAULT_AGENT_CAMERAS,
    include_apriltag_estimates: bool = True,
    artifact_dir: Path | None = None,
    pretty_logs: bool = True,
    stop_on_success: bool = True,
) -> list[AgentStepResult]:
    from agents import Runner  # type: ignore[import-not-found]

    agent = create_cup_agent(model)
    step_results: list[AgentStepResult] = []
    steps_log_path = artifact_dir / "steps.jsonl" if artifact_dir is not None else None
    if artifact_dir is not None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
    for step_index in range(max_agent_steps):
        observation, image_urls, camera_frame_paths = build_observation(
            context,
            step_index,
            camera_names,
            include_apriltag_estimates=include_apriltag_estimates,
            artifact_dir=artifact_dir,
        )
        try:
            result = await Runner.run(
                starting_agent=agent,
                input=observation_to_input_items(observation, image_urls),
                context=context,
                max_turns=2,
            )
            final_output = str(result.final_output)
        except Exception as exc:
            final_output = f"ERROR: {type(exc).__name__}: {exc}"
            context.last_tool_result = {"action": "agent_error", "success": False, "error": final_output}
        tool_output = parse_tool_output(final_output)
        evaluation = evaluate_pick_place_success(context)
        step_results.append(
            AgentStepResult(
                step_index=step_index,
                final_output=final_output,
                observation=observation,
                tool_output=tool_output,
                camera_frame_paths=camera_frame_paths,
            )
        )
        if steps_log_path is not None:
            with steps_log_path.open("a", encoding="utf-8") as steps_log:
                steps_log.write(
                    json.dumps(
                        {
                            "step_index": step_index,
                            "observation_summary": _compact_observation_summary(observation),
                            "tool_output": tool_output,
                            "final_output": final_output,
                            "camera_frame_paths": camera_frame_paths,
                            "evaluation": evaluation,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
        if pretty_logs:
            pretty_print_agent_step(
                step_index=step_index,
                observation=observation,
                final_output=final_output,
                tool_output=tool_output,
                evaluation=evaluation,
            )
        else:
            print(f"agent step {step_index}: {final_output}")
        if stop_on_success and evaluation["success"]:
            print("pick-and-place success detected; stopping early")
            break
        if tool_output is None:
            break
    return step_results


def _looks_like_tool_json(output: str) -> bool:
    return parse_tool_output(output) is not None


def create_agent_context(
    config: PickupConfig,
    *,
    render_width: int,
    render_height: int,
    camera_name: str = "table_observer",
    viewer: mujoco.viewer.Handle | None = None,  # type: ignore[name-defined]
    realtime: bool = False,
    move_duration: float = DEFAULT_MOVE_DURATION,
    gripper_duration: float = DEFAULT_GRIPPER_DURATION,
    attempt_guidance: str | None = None,
) -> CupAgentContext:
    env = create_env(
        scene_path=config.scene_path,
        camera_name=camera_name,
        render_width=render_width,
        render_height=render_height,
    )
    diagnostics_by_label = configure_scene(env, config)
    env.viewer = viewer
    return CupAgentContext(
        env=env,
        config=config,
        diagnostics_by_label=diagnostics_by_label,
        tag_pose_cache=AprilTagPoseCache(),
        viewer=viewer,
        realtime=realtime,
        move_duration=move_duration,
        gripper_duration=gripper_duration,
        attempt_guidance=attempt_guidance,
    )


def available_cup_camera_names() -> tuple[str, ...]:
    generated_cameras = tuple(camera.name for camera in CUP_SCENE_CAMERAS)
    return tuple(dict.fromkeys((*DEFAULT_AGENT_CAMERAS, *generated_cameras)))


def run_agent_steps_sync(context: CupAgentContext, **kwargs: Any) -> list[AgentStepResult]:
    return asyncio.run(run_agent_steps(context, **kwargs))
