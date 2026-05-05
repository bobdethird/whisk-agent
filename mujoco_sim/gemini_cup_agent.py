from __future__ import annotations

import base64
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import mujoco  # type: ignore[import-not-found]
import numpy as np  # type: ignore[import-not-found]

from gripper import CLOSED_GRIPPER, OPEN_GRIPPER
from mujoco_sim.openai_cup_agent import (
    DEFAULT_AGENT_CAMERAS,
    DEFAULT_GRIPPER_DURATION,
    DEFAULT_MAX_AGENT_STEPS,
    DEFAULT_MOVE_DURATION,
    MOVE_ARM_TARGETS,
    CupAgentContext,
    _active_cup_and_diagnostics,
    _compact_observation_summary,
    _current_fixed_jaw_target_xyz,
    _round_list,
    _semantic_target_for_apriltag,
    _retreat_after_release,
    _stabilize_released_cup,
    _update_held_cup_after_close,
    build_observation,
    create_agent_context,
    evaluate_cup_stack_success,
    evaluate_pick_place_success,
    write_json,
)
from mujoco_sim.run_cup_pickup import (
    PickupConfig,
    command_motion,
    hold_command,
    planner_context_for_cup,
    solve_target,
)
from sim_env import STARTING_POSITION
from so101_kinematics import CLAW_CENTER_TOOL_POINT, FIXED_JAW_TOOL_POINT, ToolPointName


DEFAULT_GEMINI_MODEL = "gemini-robotics-er-1.6-preview"
DEFAULT_GEMINI_TEMPERATURE = 0.5
DEFAULT_GEMINI_THINKING_BUDGET = 0
DEFAULT_MAX_PLAN_ACTIONS = 16
ACTION_NAMES = ("move_arm", "open_gripper", "close_gripper", "return_to_origin", "finish")
TOOL_POINTS = (CLAW_CENTER_TOOL_POINT, FIXED_JAW_TOOL_POINT)


ACTION_PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "reasoning": {
            "type": "string",
            "description": "Brief task-level reasoning. Keep it concise.",
        },
        "done": {
            "type": "boolean",
            "description": "True only when no more robot actions are needed.",
        },
        "status": {
            "type": "string",
            "description": "Short status summary for logs.",
        },
        "actions": {
            "type": "array",
            "description": "Ordered robot API calls to validate and execute locally.",
            "items": {
                "type": "object",
                "properties": {
                    "function": {
                        "type": "string",
                        "enum": list(ACTION_NAMES),
                    },
                    "args": {
                        "type": "object",
                        "description": (
                            "Named function arguments. For semantic move_arm use apriltag_id and target. "
                            "For raw move_arm use x, y, z, and target='custom'. Gripper and finish actions may use {}."
                        ),
                        "properties": {
                            "apriltag_id": {
                                "type": "integer",
                                "description": "AprilTag target ID. Use 6 for the first cup, 1 for the second cup, and 0 for the placement tag.",
                            },
                            "target": {
                                "type": "string",
                                "enum": list(MOVE_ARM_TARGETS),
                                "description": "Semantic move target, or custom when x/y/z are supplied.",
                            },
                            "x": {"type": "number", "description": "World-frame X target in meters for custom moves only."},
                            "y": {"type": "number", "description": "World-frame Y target in meters for custom moves only."},
                            "z": {"type": "number", "description": "World-frame Z target in meters for custom moves only."},
                            "duration": {"type": "number", "description": "Optional action duration in seconds."},
                            "tool_point": {
                                "type": "string",
                                "enum": list(TOOL_POINTS),
                                "description": "Optional IK tool point. Prefer fixed_jaw_tip for cup side grasps.",
                            },
                            "squeeze_duration": {
                                "type": "number",
                                "description": "Optional close_gripper squeeze hold duration in seconds.",
                            },
                        },
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Optional reason for this specific action.",
                    },
                },
                "required": ["function", "args"],
            },
        },
    },
    "required": ["reasoning", "done", "status", "actions"],
}


@dataclass(frozen=True)
class GraspStrategy:
    """Object-scale grasp policy used by the validator/executor boundary."""

    name: Literal["side_grasp", "top_down_pinch"]
    approach_height_m: float
    lift_height_m: float
    preferred_tool_point: ToolPointName


@dataclass(frozen=True)
class ObjectSpec:
    """Provider-neutral object description for cups now and small objects later."""

    label: str
    category: Literal["cup", "small_object", "container", "placement_marker"]
    pose_source: Literal["apriltag", "vision_point", "vision_box", "simulator"]
    grasp_strategy: GraspStrategy
    radius_m: float | None = None
    height_m: float | None = None
    apriltag_id: int | None = None


@dataclass(frozen=True)
class PlacementTarget:
    label: str
    pose_source: Literal["apriltag", "vision_point", "vision_box", "simulator"]
    support_body_name: str
    apriltag_id: int | None = None


CUP_SIDE_GRASP = GraspStrategy(
    name="side_grasp",
    approach_height_m=0.07,
    lift_height_m=0.09,
    preferred_tool_point=FIXED_JAW_TOOL_POINT,
)
SMALL_OBJECT_TOP_DOWN_GRASP = GraspStrategy(
    name="top_down_pinch",
    approach_height_m=0.04,
    lift_height_m=0.06,
    preferred_tool_point=CLAW_CENTER_TOOL_POINT,
)


@dataclass(frozen=True)
class RobotAction:
    function: str
    args: dict[str, Any]
    reasoning: str | None = None


@dataclass(frozen=True)
class RobotActionPlan:
    reasoning: str
    status: str
    done: bool
    actions: tuple[RobotAction, ...]
    raw_payload: dict[str, Any]


@dataclass(frozen=True)
class GeminiStepResult:
    step_index: int
    prompt: str
    raw_response: str
    plan: RobotActionPlan | None
    validation_errors: tuple[str, ...]
    action_results: tuple[dict[str, Any], ...]
    observation: dict[str, Any]
    camera_frame_paths: dict[str, str]
    evaluation: dict[str, Any]


class PlanValidationError(ValueError):
    pass


def compact_scene_state_for_gemini(observation: dict[str, Any]) -> dict[str, Any]:
    """Keep prompts compact while preserving the state needed for action choices."""

    return {
        "step_index": observation["step_index"],
        "scene_path": observation["scene_path"],
        "current_joints": observation["current_joints"],
        "gripper": observation["gripper"],
        "objects": observation["objects"],
        "apriltag_estimates": observation.get("apriltag_estimates"),
        "last_tool_result": observation.get("last_tool_result"),
        "attempt_guidance": observation.get("attempt_guidance"),
    }


def build_gemini_prompt(observation: dict[str, Any], *, max_actions: int = DEFAULT_MAX_PLAN_ACTIONS) -> str:
    scene_state = compact_scene_state_for_gemini(observation)
    return (
        "You are controlling a MuJoCo SO-101 robot arm in a cup pick-and-place scene.\n"
        "Use the camera images and scene JSON to choose robot API calls. Return only JSON matching the provided schema.\n\n"
        "Task: stack cups. First pick up the first cup with AprilTag 6, move it to the flat placement tag "
        "with AprilTag 0, and release it there. Then pick up the second cup with AprilTag 1, move it above "
        "the first cup using AprilTag 6 as the support target, and release it on top of the first cup.\n\n"
        "Available robot API:\n"
        "- move_arm(apriltag_id:int,target:string,duration?:number,tool_point?:string)\n"
        "  Semantic targets are approach, grasp, lift for cup tags, plus place_above/place for the flat placement "
        "tag or a support cup tag while holding another cup. "
        "Every semantic move_arm action must include both apriltag_id and target inside args.\n"
        "- move_arm(x:number,y:number,z:number,target:\"custom\",duration?:number,tool_point?:string)\n"
        "  Raw XYZ is a recovery fallback only. Empty args are invalid for move_arm.\n"
        "- open_gripper(duration?:number)\n"
        "- close_gripper(duration?:number,squeeze_duration?:number)\n"
        "- return_to_origin(duration?:number) only after the cup has been released.\n"
        "- finish() when the task is complete or impossible.\n\n"
        "Default deterministic policy for this cup scene, written in the exact JSON action format:\n"
        "{\"function\":\"move_arm\",\"args\":{\"apriltag_id\":6,\"target\":\"approach\"}}\n"
        "{\"function\":\"move_arm\",\"args\":{\"apriltag_id\":6,\"target\":\"grasp\"}}\n"
        "{\"function\":\"close_gripper\",\"args\":{}}\n"
        "{\"function\":\"move_arm\",\"args\":{\"apriltag_id\":6,\"target\":\"lift\"}}\n"
        "{\"function\":\"move_arm\",\"args\":{\"apriltag_id\":0,\"target\":\"place_above\"}}\n"
        "{\"function\":\"move_arm\",\"args\":{\"apriltag_id\":0,\"target\":\"place\"}}\n"
        "{\"function\":\"open_gripper\",\"args\":{}}\n"
        "{\"function\":\"return_to_origin\",\"args\":{}}\n"
        "{\"function\":\"move_arm\",\"args\":{\"apriltag_id\":1,\"target\":\"approach\"}}\n"
        "{\"function\":\"move_arm\",\"args\":{\"apriltag_id\":1,\"target\":\"grasp\"}}\n"
        "{\"function\":\"close_gripper\",\"args\":{}}\n"
        "{\"function\":\"move_arm\",\"args\":{\"apriltag_id\":1,\"target\":\"lift\"}}\n"
        "{\"function\":\"move_arm\",\"args\":{\"apriltag_id\":6,\"target\":\"place_above\"}}\n"
        "{\"function\":\"move_arm\",\"args\":{\"apriltag_id\":6,\"target\":\"place\"}}\n"
        "{\"function\":\"open_gripper\",\"args\":{}}\n\n"
        "Prefer semantic calls because local code computes simulator object centers, cup dimensions, side-grasp "
        "offsets, tag/support offsets, IK, collision guards, and release behavior. Do not aim raw XYZ at a cup "
        "center or top. For cup grasps, the fixed jaw tip is placed at side height with a small depth offset "
        "toward the robot and a left offset of cup_radius + lateral_grasp_offset so the moving jaw closes around "
        "the cup wall. "
        "Produce no more than "
        f"{max_actions} actions. Use finish with no other action if the task is already complete.\n\n"
        "Scene JSON:\n"
        + json.dumps(scene_state, sort_keys=True, separators=(",", ":"))
    )


def _strip_json_fence(raw_response: str) -> str:
    stripped = raw_response.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def parse_json_payload(raw_response: str) -> dict[str, Any]:
    try:
        payload = json.loads(_strip_json_fence(raw_response))
    except json.JSONDecodeError as exc:
        raise PlanValidationError(f"Gemini response was not valid JSON: {exc}") from exc
    if isinstance(payload, list):
        payload = {
            "reasoning": "Model returned a bare action list.",
            "done": False,
            "status": "action_list",
            "actions": payload,
        }
    if not isinstance(payload, dict):
        raise PlanValidationError("Gemini response must be a JSON object or action list.")
    return payload


def _coerce_args(raw_args: Any, function_name: str) -> dict[str, Any]:
    if raw_args is None:
        return {}
    if isinstance(raw_args, dict):
        return dict(raw_args)
    if isinstance(raw_args, list) and not raw_args:
        return {}
    raise PlanValidationError(f"{function_name}.args must be an object.")


def _optional_finite_float(args: dict[str, Any], key: str) -> float | None:
    if key not in args or args[key] is None:
        return None
    value = args[key]
    if not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise PlanValidationError(f"{key} must be a finite number.")
    return float(value)


def _required_finite_float(args: dict[str, Any], key: str) -> float:
    value = _optional_finite_float(args, key)
    if value is None:
        raise PlanValidationError(f"{key} is required.")
    return value


def _optional_int(args: dict[str, Any], key: str) -> int | None:
    if key not in args or args[key] is None:
        return None
    value = args[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise PlanValidationError(f"{key} must be an integer.")
    return int(value)


def _tool_point(args: dict[str, Any]) -> ToolPointName:
    value = args.get("tool_point", FIXED_JAW_TOOL_POINT)
    if value not in TOOL_POINTS:
        raise PlanValidationError(f"tool_point must be one of {TOOL_POINTS}.")
    return value


def _validate_move_arm_args(context: CupAgentContext, args: dict[str, Any]) -> dict[str, Any]:
    target = args.get("target", "custom")
    if not isinstance(target, str) or target not in MOVE_ARM_TARGETS:
        raise PlanValidationError(f"move_arm.target must be one of {MOVE_ARM_TARGETS}.")
    duration = _optional_finite_float(args, "duration")
    tool_point = _tool_point(args)

    apriltag_id = _optional_int(args, "apriltag_id")
    if target != "custom":
        if apriltag_id is None:
            raise PlanValidationError("Semantic move_arm calls require apriltag_id.")
        allowed_tag_ids = {
            context.config.cup_tag_id,
            context.config.second_cup_tag_id,
            context.config.place_tag_id,
        }
        if apriltag_id not in allowed_tag_ids:
            raise PlanValidationError(f"Unsupported apriltag_id {apriltag_id}.")
        if apriltag_id == context.config.place_tag_id and target not in {"place_above", "place"}:
            raise PlanValidationError("Placement tag moves must use target place_above or place.")
        if apriltag_id != context.config.place_tag_id and target not in {"approach", "grasp", "lift", "place_above", "place"}:
            raise PlanValidationError("Cup tag moves must use approach, grasp, lift, place_above, or place.")
        validated: dict[str, Any] = {
            "apriltag_id": apriltag_id,
            "target": target,
            "tool_point": tool_point,
        }
    else:
        validated = {
            "x": _required_finite_float(args, "x"),
            "y": _required_finite_float(args, "y"),
            "z": _required_finite_float(args, "z"),
            "target": "custom",
            "tool_point": tool_point,
        }
    if duration is not None:
        validated["duration"] = duration
    return validated


def _validate_action(context: CupAgentContext, raw_action: Any, index: int) -> RobotAction:
    if not isinstance(raw_action, dict):
        raise PlanValidationError(f"Action {index} must be an object.")
    function_name = raw_action.get("function")
    if not isinstance(function_name, str) or function_name not in ACTION_NAMES:
        raise PlanValidationError(f"Action {index} function must be one of {ACTION_NAMES}.")
    args = _coerce_args(raw_action.get("args", {}), function_name)
    reasoning = raw_action.get("reasoning")
    if reasoning is not None and not isinstance(reasoning, str):
        raise PlanValidationError(f"Action {index} reasoning must be a string.")

    if function_name == "move_arm":
        args = _validate_move_arm_args(context, args)
    elif function_name in {"open_gripper", "close_gripper", "return_to_origin"}:
        validated_args: dict[str, Any] = {}
        duration = _optional_finite_float(args, "duration")
        if duration is not None:
            validated_args["duration"] = duration
        if function_name == "close_gripper":
            squeeze_duration = _optional_finite_float(args, "squeeze_duration")
            if squeeze_duration is not None:
                validated_args["squeeze_duration"] = squeeze_duration
        args = validated_args
    elif function_name == "finish":
        args = {}
    return RobotAction(function=function_name, args=args, reasoning=reasoning)


def validate_action_plan(
    context: CupAgentContext,
    payload: dict[str, Any],
    *,
    max_actions: int = DEFAULT_MAX_PLAN_ACTIONS,
) -> RobotActionPlan:
    reasoning = payload.get("reasoning", "")
    status = payload.get("status", "")
    done = payload.get("done", False)
    raw_actions = payload.get("actions", [])
    if not isinstance(reasoning, str):
        raise PlanValidationError("reasoning must be a string.")
    if not isinstance(status, str):
        raise PlanValidationError("status must be a string.")
    if not isinstance(done, bool):
        raise PlanValidationError("done must be a boolean.")
    if not isinstance(raw_actions, list):
        raise PlanValidationError("actions must be a list.")
    if len(raw_actions) > max_actions:
        raise PlanValidationError(f"actions must contain no more than {max_actions} items.")

    actions = tuple(_validate_action(context, action, index) for index, action in enumerate(raw_actions))
    if done and actions:
        raise PlanValidationError("done=true plans must not include actions.")
    if not done and not actions:
        raise PlanValidationError("Plans must either set done=true or include at least one action.")
    if any(action.function == "finish" for action in actions) and len(actions) > 1:
        raise PlanValidationError("finish must be the only action when used.")
    return RobotActionPlan(
        reasoning=reasoning,
        status=status,
        done=done,
        actions=actions,
        raw_payload=payload,
    )


def parse_and_validate_action_plan(
    context: CupAgentContext,
    raw_response: str,
    *,
    max_actions: int = DEFAULT_MAX_PLAN_ACTIONS,
) -> RobotActionPlan:
    return validate_action_plan(context, parse_json_payload(raw_response), max_actions=max_actions)


def _decode_data_url(data_url: str) -> tuple[bytes, str]:
    header, encoded = data_url.split(",", 1)
    mime_type = header.removeprefix("data:").split(";", 1)[0] or "image/png"
    return base64.b64decode(encoded), mime_type


def _gemini_config(types: Any, *, temperature: float, thinking_budget: int | None) -> Any:
    config_kwargs: dict[str, Any] = {
        "temperature": temperature,
        "response_mime_type": "application/json",
        "response_schema": ACTION_PLAN_SCHEMA,
    }
    if thinking_budget is not None:
        config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=thinking_budget)
    try:
        return types.GenerateContentConfig(**config_kwargs)
    except TypeError:
        config_kwargs["response_json_schema"] = config_kwargs.pop("response_schema")
        return types.GenerateContentConfig(**config_kwargs)


def call_gemini_action_planner(
    prompt: str,
    image_urls: dict[str, str],
    *,
    model: str = DEFAULT_GEMINI_MODEL,
    temperature: float = DEFAULT_GEMINI_TEMPERATURE,
    thinking_budget: int | None = DEFAULT_GEMINI_THINKING_BUDGET,
) -> str:
    from google import genai  # type: ignore[import-not-found]
    from google.genai import types  # type: ignore[import-not-found]

    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key) if api_key else genai.Client()
    contents: list[Any] = [prompt]
    for camera_name, data_url in image_urls.items():
        image_bytes, mime_type = _decode_data_url(data_url)
        contents.append(f"Camera view: {camera_name}")
        contents.append(types.Part.from_bytes(data=image_bytes, mime_type=mime_type))

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=_gemini_config(types, temperature=temperature, thinking_budget=thinking_budget),
    )
    return response.text or ""


def deterministic_reference_plan_payload() -> dict[str, Any]:
    return {
        "reasoning": "Follow the known-good semantic cup stacking policy.",
        "done": False,
        "status": "reference_cup_stack_plan",
        "actions": [
            {"function": "move_arm", "args": {"apriltag_id": 6, "target": "approach"}},
            {"function": "move_arm", "args": {"apriltag_id": 6, "target": "grasp"}},
            {"function": "close_gripper", "args": {}},
            {"function": "move_arm", "args": {"apriltag_id": 6, "target": "lift"}},
            {"function": "move_arm", "args": {"apriltag_id": 0, "target": "place_above"}},
            {"function": "move_arm", "args": {"apriltag_id": 0, "target": "place"}},
            {"function": "open_gripper", "args": {}},
            {"function": "return_to_origin", "args": {}},
            {"function": "move_arm", "args": {"apriltag_id": 1, "target": "approach"}},
            {"function": "move_arm", "args": {"apriltag_id": 1, "target": "grasp"}},
            {"function": "close_gripper", "args": {}},
            {"function": "move_arm", "args": {"apriltag_id": 1, "target": "lift"}},
            {"function": "move_arm", "args": {"apriltag_id": 6, "target": "place_above"}},
            {"function": "move_arm", "args": {"apriltag_id": 6, "target": "place"}},
            {"function": "open_gripper", "args": {}},
        ],
    }


def _result(success: bool, action: str, **extra: Any) -> dict[str, Any]:
    return {"success": success, "action": action, **extra}


def execute_move_arm(context: CupAgentContext, args: dict[str, Any]) -> dict[str, Any]:
    try:
        target = args.get("target", "custom")
        allowed_support_body_names: tuple[str, ...] = ()
        if args.get("apriltag_id") is not None and target != "custom":
            target_xyz, active_cup, diagnostics, target_source, allowed_support_body_names = _semantic_target_for_apriltag(
                context,
                int(args["apriltag_id"]),
                str(target),
            )
        else:
            target_xyz = np.array([float(args["x"]), float(args["y"]), float(args["z"])], dtype=float)
            active_cup, diagnostics = _active_cup_and_diagnostics(context)
            target_source = "raw_xyz"

        gripper_position = float(context.env.current_position.get("gripper", OPEN_GRIPPER))
        tool_point = args.get("tool_point", FIXED_JAW_TOOL_POINT)
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
            context.move_duration if args.get("duration") is None else float(args["duration"]),
            diagnostics,
            viewer=context.viewer,
            realtime=context.realtime,
            planner_config=context.config.motion_planner,
            collision_context=collision_context,
        )
        context.last_tool_result = _result(
            success,
            "move_arm",
            target_xyz_m=_round_list(target_xyz),
            tool_point=tool_point,
            apriltag_id=args.get("apriltag_id"),
            target=target,
            target_source=target_source,
        )
    except Exception as exc:
        context.last_tool_result = _result(False, "move_arm", error=str(exc))
    return dict(context.last_tool_result)


def execute_open_gripper(context: CupAgentContext, args: dict[str, Any]) -> dict[str, Any]:
    target_position = dict(context.env.current_position)
    target_position["gripper"] = OPEN_GRIPPER
    was_holding = context.held_cup_label is not None
    release_center = None
    if was_holding and context.held_grasp_offset is not None:
        release_center = _current_fixed_jaw_target_xyz(context) - context.held_grasp_offset
    _cup, diagnostics = _active_cup_and_diagnostics(context)
    try:
        success = command_motion(
            context.env,
            target_position,
            context.gripper_duration if args.get("duration") is None else float(args["duration"]),
            diagnostics,
            viewer=context.viewer,
            realtime=context.realtime,
        )
        if success:
            context.held_cup_label = None
            context.held_grasp_offset = None
        release_retreat = False
        if success and was_holding:
            release_retreat = _retreat_after_release(context, _cup, diagnostics)
            success = release_retreat
            if success and release_center is not None:
                _stabilize_released_cup(context, _cup, release_center)
        context.last_tool_result = _result(success, "open_gripper", release_retreat=release_retreat)
    except Exception as exc:
        context.last_tool_result = _result(False, "open_gripper", error=str(exc))
    return dict(context.last_tool_result)


def execute_close_gripper(context: CupAgentContext, args: dict[str, Any]) -> dict[str, Any]:
    target_position = dict(context.env.current_position)
    target_position["gripper"] = CLOSED_GRIPPER
    _cup, diagnostics = _active_cup_and_diagnostics(context)
    try:
        success = command_motion(
            context.env,
            target_position,
            context.gripper_duration if args.get("duration") is None else float(args["duration"]),
            diagnostics,
            viewer=context.viewer,
            realtime=context.realtime,
        )
        if success:
            success = hold_command(
                context.env,
                target_position,
                context.config.squeeze_duration if args.get("squeeze_duration") is None else float(args["squeeze_duration"]),
                diagnostics,
                viewer=context.viewer,
                realtime=context.realtime,
            )
        if success:
            _update_held_cup_after_close(context)
        context.last_tool_result = _result(success, "close_gripper", held_cup_label=context.held_cup_label)
    except Exception as exc:
        context.last_tool_result = _result(False, "close_gripper", error=str(exc))
    return dict(context.last_tool_result)


def execute_return_to_origin(context: CupAgentContext, args: dict[str, Any]) -> dict[str, Any]:
    if context.held_cup_label is not None:
        context.last_tool_result = _result(False, "return_to_origin", error="Cannot return to origin while holding a cup.")
        return dict(context.last_tool_result)
    cup, diagnostics = _active_cup_and_diagnostics(context)
    target_position = dict(STARTING_POSITION)
    try:
        success = command_motion(
            context.env,
            target_position,
            context.move_duration if args.get("duration") is None else float(args["duration"]),
            diagnostics,
            viewer=context.viewer,
            realtime=context.realtime,
            planner_config=context.config.motion_planner,
            collision_context=planner_context_for_cup(diagnostics, cup, allow_gripper_cup_contact=False),
        )
        context.last_tool_result = _result(success, "return_to_origin")
    except Exception as exc:
        context.last_tool_result = _result(False, "return_to_origin", error=str(exc))
    return dict(context.last_tool_result)


def execute_robot_action(context: CupAgentContext, action: RobotAction) -> dict[str, Any]:
    if action.function == "move_arm":
        return execute_move_arm(context, action.args)
    if action.function == "open_gripper":
        return execute_open_gripper(context, action.args)
    if action.function == "close_gripper":
        return execute_close_gripper(context, action.args)
    if action.function == "return_to_origin":
        return execute_return_to_origin(context, action.args)
    if action.function == "finish":
        context.last_tool_result = _result(True, "finish", status=action.reasoning)
        return dict(context.last_tool_result)
    raise ValueError(f"Unknown action {action.function!r}.")


def pretty_print_gemini_step(result: GeminiStepResult) -> None:
    print(f"\n=== Gemini step {result.step_index:03d} ===")
    if result.plan is None:
        print("plan: invalid")
    else:
        print(f"plan status: {result.plan.status}")
        print(f"actions: {len(result.plan.actions)}")
    if result.validation_errors:
        print("validation errors:")
        for error in result.validation_errors:
            print(f"  - {error}")
    for action_result in result.action_results:
        print("action: " + json.dumps(action_result, sort_keys=True))
    print(
        "metrics: "
        f"lift_delta={result.evaluation['lift_delta_m']}m "
        f"xy_error={result.evaluation['xy_error_m']}m "
        f"z_error={result.evaluation['z_error_m']}m "
        f"released={result.evaluation['released']}"
    )


def write_gemini_step_log(path: Path, result: GeminiStepResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as steps_log:
        steps_log.write(
            json.dumps(
                {
                    "step_index": result.step_index,
                    "observation_summary": _compact_observation_summary(result.observation),
                    "prompt": result.prompt,
                    "raw_response": result.raw_response,
                    "plan": result.plan.raw_payload if result.plan is not None else None,
                    "validation_errors": list(result.validation_errors),
                    "action_results": list(result.action_results),
                    "camera_frame_paths": result.camera_frame_paths,
                    "evaluation": result.evaluation,
                },
                sort_keys=True,
            )
            + "\n"
        )


def run_gemini_steps(
    context: CupAgentContext,
    *,
    model: str = DEFAULT_GEMINI_MODEL,
    max_agent_steps: int = DEFAULT_MAX_AGENT_STEPS,
    max_plan_actions: int = DEFAULT_MAX_PLAN_ACTIONS,
    camera_names: tuple[str, ...] = DEFAULT_AGENT_CAMERAS,
    include_apriltag_estimates: bool = True,
    artifact_dir: Path | None = None,
    execute_actions: bool = True,
    pretty_logs: bool = True,
    stop_on_success: bool = True,
    temperature: float = DEFAULT_GEMINI_TEMPERATURE,
    thinking_budget: int | None = DEFAULT_GEMINI_THINKING_BUDGET,
    reference_plan: bool = False,
) -> list[GeminiStepResult]:
    results: list[GeminiStepResult] = []
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
        prompt = build_gemini_prompt(observation, max_actions=max_plan_actions)
        prompt_path = artifact_dir / f"prompt_{step_index:03d}.txt" if artifact_dir is not None else None
        if prompt_path is not None:
            prompt_path.write_text(prompt + "\n", encoding="utf-8")

        validation_errors: tuple[str, ...] = ()
        plan: RobotActionPlan | None = None
        action_results: list[dict[str, Any]] = []
        try:
            if reference_plan:
                raw_response = json.dumps(deterministic_reference_plan_payload())
            else:
                raw_response = call_gemini_action_planner(
                    prompt,
                    image_urls,
                    model=model,
                    temperature=temperature,
                    thinking_budget=thinking_budget,
                )
            plan = parse_and_validate_action_plan(context, raw_response, max_actions=max_plan_actions)
        except Exception as exc:
            raw_response = locals().get("raw_response", "")
            validation_errors = (f"{type(exc).__name__}: {exc}",)
            context.last_tool_result = _result(False, "gemini_plan", error=validation_errors[0])

        if execute_actions and plan is not None and not plan.done:
            for action in plan.actions:
                action_result = execute_robot_action(context, action)
                action_results.append(action_result)
                if not action_result.get("success", False) or action.function == "finish":
                    break

        evaluation = evaluate_cup_stack_success(context)
        step_result = GeminiStepResult(
            step_index=step_index,
            prompt=prompt,
            raw_response=raw_response,
            plan=plan,
            validation_errors=validation_errors,
            action_results=tuple(action_results),
            observation=observation,
            camera_frame_paths=camera_frame_paths,
            evaluation=evaluation,
        )
        results.append(step_result)

        if steps_log_path is not None:
            write_gemini_step_log(steps_log_path, step_result)
        if pretty_logs:
            pretty_print_gemini_step(step_result)

        if not execute_actions or validation_errors or plan is None:
            break
        if stop_on_success and evaluation["success"]:
            print("cup-stack success detected; stopping early")
            break
        if plan.done or any(action.function == "finish" for action in plan.actions):
            break
    return results


def create_gemini_context(
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
    return create_agent_context(
        config,
        render_width=render_width,
        render_height=render_height,
        camera_name=camera_name,
        viewer=viewer,
        realtime=realtime,
        move_duration=move_duration,
        gripper_duration=gripper_duration,
        attempt_guidance=attempt_guidance,
    )


def write_gemini_json(path: Path, payload: dict[str, Any]) -> None:
    write_json(path, payload)
