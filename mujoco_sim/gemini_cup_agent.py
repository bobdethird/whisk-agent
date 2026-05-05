from __future__ import annotations

import base64
import json
import math
import os
import re
import time
from collections.abc import Callable
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
    _round_list,
    _last_move_was_stack_place,
    _semantic_target_for_apriltag,
    _retreat_after_release,
    _update_held_cup_after_close,
    build_observation,
    create_agent_context,
    evaluate_cup_pour_success,
    execute_pour_into,
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
DEFAULT_MAX_PLAN_ACTIONS = 4
DEFAULT_GEMINI_AGENT_MODE = "tools"
ACTION_NAMES = ("move_arm", "open_gripper", "close_gripper", "pour_into", "return_to_origin", "finish")
TOOL_POINTS = (CLAW_CENTER_TOOL_POINT, FIXED_JAW_TOOL_POINT)
GeminiAgentMode = Literal["tools", "json"]


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
                            "For raw move_arm use x, y, z, and target='custom'. For pour_into use the receiver "
                            "cup apriltag_id. Gripper and finish actions may use {}."
                        ),
                        "properties": {
                            "apriltag_id": {
                                "type": "integer",
                                "description": "Configured AprilTag target ID from the task and Scene JSON.",
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
                            "tilt_degrees": {
                                "type": "number",
                                "description": "Optional pour wrist-roll tilt; omit for the configured default.",
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


TaskEvaluationFn = Callable[[CupAgentContext], dict[str, Any]]
TaskReferencePlanFn = Callable[[CupAgentContext], dict[str, Any]]
TaskAllowedTagsFn = Callable[[CupAgentContext], set[int]]
TaskGuidanceFn = Callable[[dict[str, Any] | None], str]


@dataclass(frozen=True)
class GeminiTaskSpec:
    """Task-specific policy layered on top of the generic Gemini robot API loop."""

    name: str
    title: str
    instruction: str
    phase_policy: str
    semantic_guidance: str
    completion_criteria: str
    evaluator: TaskEvaluationFn
    reference_plan: TaskReferencePlanFn | None
    allowed_apriltag_ids: TaskAllowedTagsFn
    guidance: TaskGuidanceFn


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
        "task_state": observation.get("task_state"),
    }


def build_gemini_prompt(
    observation: dict[str, Any],
    *,
    max_actions: int = DEFAULT_MAX_PLAN_ACTIONS,
    task_spec: GeminiTaskSpec | None = None,
    agent_mode: GeminiAgentMode = DEFAULT_GEMINI_AGENT_MODE,
) -> str:
    task_spec = task_spec or DEFAULT_TASK_SPEC
    scene_state = compact_scene_state_for_gemini(observation)
    if agent_mode == "tools":
        response_contract = (
            "Use the provided Gemini function-calling tools when you want the robot to act. "
            "Do not emit JSON action lists in text. You may call no more than "
            f"{max_actions} robot tools during this agent step, then summarize what happened."
        )
    else:
        response_contract = (
            "Return only JSON matching the provided schema. The JSON actions will be validated "
            f"and executed locally. Produce no more than {max_actions} actions."
        )
    return (
        "You are controlling a MuJoCo SO-101 robot arm.\n"
        "Use the camera images and Scene JSON to choose robot API calls. "
        "Work closed-loop: choose only the next short phase, then stop. "
        "After the phase executes, you will receive fresh object poses and camera images before planning again.\n\n"
        f"Task ({task_spec.name}): {task_spec.title}\n"
        f"{task_spec.instruction}\n\n"
        f"Response contract: {response_contract}\n\n"
        "Available robot API:\n"
        "- move_arm(apriltag_id:int,target:string,duration?:number,tool_point?:string)\n"
        "  Semantic targets are approach, grasp, lift for cup tags, plus place_above/place for the flat placement "
        "tag or a support cup tag while holding another cup. "
        "Every semantic move_arm action must include both apriltag_id and target inside args.\n"
        "- move_arm(x:number,y:number,z:number,target:\"custom\",duration?:number,tool_point?:string)\n"
        "  Raw XYZ is a recovery fallback only. Empty args are invalid for move_arm.\n"
        "- open_gripper(duration?:number)\n"
        "- close_gripper(duration?:number,squeeze_duration?:number)\n"
        "- pour_into(apriltag_id:int,duration?:number,tilt_degrees?:number) while holding one cup, to pour into "
        "the receiver cup with the given AprilTag.\n"
        "- return_to_origin(duration?:number) only after the cup has been released.\n"
        "- finish() when the task is complete or impossible.\n\n"
        "Task policy:\n"
        f"{task_spec.phase_policy}\n\n"
        "Prefer semantic calls because local code computes simulator object centers, cup dimensions, side-grasp "
        "offsets, tag/support offsets, IK, collision guards, and release behavior. Do not aim raw XYZ at a cup "
        "center or top. For cup grasps, the fixed jaw tip is placed at side height with a small depth offset "
        "toward the robot and a left offset of cup_radius + lateral_grasp_offset so the moving jaw closes around "
        "the cup wall.\n"
        f"{task_spec.semantic_guidance}\n"
        f"Completion criteria: {task_spec.completion_criteria}\n"
        "Do not plan later phases early. Use finish with no other robot action if the task is already complete or impossible.\n\n"
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


def _validate_move_arm_args(
    context: CupAgentContext,
    args: dict[str, Any],
    *,
    task_spec: GeminiTaskSpec | None = None,
) -> dict[str, Any]:
    task_spec = task_spec or DEFAULT_TASK_SPEC
    target = args.get("target", "custom")
    if not isinstance(target, str) or target not in MOVE_ARM_TARGETS:
        raise PlanValidationError(f"move_arm.target must be one of {MOVE_ARM_TARGETS}.")
    duration = _optional_finite_float(args, "duration")
    explicit_tool_point = args.get("tool_point") is not None
    tool_point = _tool_point(args)

    apriltag_id = _optional_int(args, "apriltag_id")
    if target != "custom":
        if apriltag_id is None:
            raise PlanValidationError("Semantic move_arm calls require apriltag_id.")
        allowed_tag_ids = task_spec.allowed_apriltag_ids(context)
        if apriltag_id not in allowed_tag_ids:
            raise PlanValidationError(f"Unsupported apriltag_id {apriltag_id}.")
        if apriltag_id == context.config.place_tag_id and target not in {"place_above", "place"}:
            raise PlanValidationError("Placement tag moves must use target place_above or place.")
        if apriltag_id != context.config.place_tag_id and target not in {"approach", "grasp", "lift", "place_above", "place"}:
            raise PlanValidationError("Cup tag moves must use approach, grasp, lift, place_above, or place.")
        if (
            not explicit_tool_point
            and apriltag_id != context.config.place_tag_id
            and target in {"place_above", "place"}
            and context.held_cup_label is not None
        ):
            tool_point = CLAW_CENTER_TOOL_POINT
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


def _validate_action(
    context: CupAgentContext,
    raw_action: Any,
    index: int,
    *,
    task_spec: GeminiTaskSpec | None = None,
) -> RobotAction:
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
        args = _validate_move_arm_args(context, args, task_spec=task_spec)
    elif function_name == "pour_into":
        apriltag_id = _optional_int(args, "apriltag_id")
        if apriltag_id is None:
            raise PlanValidationError("pour_into requires receiver cup apriltag_id.")
        if apriltag_id not in task_spec.allowed_apriltag_ids(context) or apriltag_id == context.config.place_tag_id:
            raise PlanValidationError(f"Unsupported pour receiver apriltag_id {apriltag_id}.")
        validated_args = {"apriltag_id": apriltag_id}
        duration = _optional_finite_float(args, "duration")
        if duration is not None:
            validated_args["duration"] = duration
        tilt_degrees = _optional_finite_float(args, "tilt_degrees")
        if tilt_degrees is not None:
            validated_args["tilt_degrees"] = tilt_degrees
        args = validated_args
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
    task_spec: GeminiTaskSpec | None = None,
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

    actions = tuple(_validate_action(context, action, index, task_spec=task_spec) for index, action in enumerate(raw_actions))
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
    task_spec: GeminiTaskSpec | None = None,
) -> RobotActionPlan:
    return validate_action_plan(context, parse_json_payload(raw_response), max_actions=max_actions, task_spec=task_spec)


def _decode_data_url(data_url: str) -> tuple[bytes, str]:
    header, encoded = data_url.split(",", 1)
    mime_type = header.removeprefix("data:").split(";", 1)[0] or "image/png"
    return base64.b64decode(encoded), mime_type


def _retry_delay_seconds(exc: Exception, attempt_index: int) -> float:
    match = re.search(r"retry in ([0-9.]+)s", str(exc), flags=re.IGNORECASE)
    if match:
        return min(30.0, max(1.0, float(match.group(1))))
    return min(30.0, float(2**attempt_index))


def _is_retryable_gemini_error(exc: Exception) -> bool:
    message = str(exc)
    return any(token in message for token in ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "500"))


def _generate_with_retries(client: Any, *, model: str, contents: list[Any], config: Any, api_retries: int) -> Any:
    for attempt_index in range(api_retries + 1):
        try:
            return client.models.generate_content(model=model, contents=contents, config=config)
        except Exception as exc:
            if attempt_index >= api_retries or not _is_retryable_gemini_error(exc):
                raise
            time.sleep(_retry_delay_seconds(exc, attempt_index))
    raise RuntimeError("Gemini request retry loop exited unexpectedly.")


def _gemini_config(
    types: Any,
    *,
    temperature: float,
    thinking_budget: int | None,
    response_schema: dict[str, Any] | None = None,
    tools: list[Any] | None = None,
) -> Any:
    config_kwargs: dict[str, Any] = {
        "temperature": temperature,
    }
    if response_schema is not None:
        config_kwargs["response_mime_type"] = "application/json"
        config_kwargs["response_schema"] = response_schema
    if tools is not None:
        config_kwargs["tools"] = tools
    if thinking_budget is not None:
        config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=thinking_budget)
    try:
        return types.GenerateContentConfig(**config_kwargs)
    except TypeError:
        if "response_schema" in config_kwargs:
            config_kwargs["response_json_schema"] = config_kwargs.pop("response_schema")
        return types.GenerateContentConfig(**config_kwargs)


def call_gemini_action_planner(
    prompt: str,
    image_urls: dict[str, str],
    *,
    model: str = DEFAULT_GEMINI_MODEL,
    temperature: float = DEFAULT_GEMINI_TEMPERATURE,
    thinking_budget: int | None = DEFAULT_GEMINI_THINKING_BUDGET,
    api_retries: int = 2,
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

    response = _generate_with_retries(
        client,
        model=model,
        contents=contents,
        config=_gemini_config(
            types,
            temperature=temperature,
            thinking_budget=thinking_budget,
            response_schema=ACTION_PLAN_SCHEMA,
        ),
        api_retries=api_retries,
    )
    return response.text or ""


def _action_payload(action: RobotAction) -> dict[str, Any]:
    payload = {"function": action.function, "args": dict(action.args)}
    if action.reasoning is not None:
        payload["reasoning"] = action.reasoning
    return payload


def call_gemini_tool_agent(
    context: CupAgentContext,
    prompt: str,
    image_urls: dict[str, str],
    *,
    model: str = DEFAULT_GEMINI_MODEL,
    temperature: float = DEFAULT_GEMINI_TEMPERATURE,
    thinking_budget: int | None = DEFAULT_GEMINI_THINKING_BUDGET,
    max_tool_calls: int = DEFAULT_MAX_PLAN_ACTIONS,
    execute_actions: bool = True,
    task_spec: GeminiTaskSpec | None = None,
    api_retries: int = 2,
) -> tuple[str, RobotActionPlan, tuple[dict[str, Any], ...]]:
    """Run one Gemini function-calling turn and return the robot calls it made."""

    from google import genai  # type: ignore[import-not-found]
    from google.genai import types  # type: ignore[import-not-found]

    task_spec = task_spec or DEFAULT_TASK_SPEC
    actions: list[RobotAction] = []
    action_results: list[dict[str, Any]] = []

    def record_robot_action(function_name: str, args: dict[str, Any], reasoning: str | None = None) -> dict[str, Any]:
        if len(actions) >= max_tool_calls:
            result = _result(False, function_name, error=f"Tool-call budget exceeded ({max_tool_calls}).")
            context.last_tool_result = result
            action_results.append(result)
            return result
        try:
            action = _validate_action(
                context,
                {"function": function_name, "args": args, "reasoning": reasoning},
                len(actions),
                task_spec=task_spec,
            )
        except Exception as exc:
            result = _result(False, function_name, error=f"{type(exc).__name__}: {exc}")
            context.last_tool_result = result
            action_results.append(result)
            return result

        actions.append(action)
        if execute_actions:
            result = execute_robot_action(context, action)
        else:
            result = _result(True, action.function, planned_only=True, args=action.args)
            context.last_tool_result = result
        action_results.append(result)
        return result

    def handle_move_arm(raw_args: dict[str, Any]) -> dict[str, Any]:
        args = {
            "apriltag_id": raw_args.get("apriltag_id"),
            "target": raw_args.get("target"),
        }
        if raw_args.get("duration") not in {None, 0, 0.0}:
            args["duration"] = raw_args["duration"]
        if raw_args.get("tool_point"):
            args["tool_point"] = raw_args["tool_point"]
        return record_robot_action("move_arm", args)

    def handle_move_arm_xyz(raw_args: dict[str, Any]) -> dict[str, Any]:
        args = {
            "x": raw_args.get("x"),
            "y": raw_args.get("y"),
            "z": raw_args.get("z"),
            "target": "custom",
        }
        if raw_args.get("duration") not in {None, 0, 0.0}:
            args["duration"] = raw_args["duration"]
        if raw_args.get("tool_point"):
            args["tool_point"] = raw_args["tool_point"]
        return record_robot_action("move_arm", args, reasoning="raw XYZ recovery move")

    def handle_open_gripper(raw_args: dict[str, Any]) -> dict[str, Any]:
        args: dict[str, Any] = {}
        if raw_args.get("duration") not in {None, 0, 0.0}:
            args["duration"] = raw_args["duration"]
        return record_robot_action("open_gripper", args)

    def handle_close_gripper(raw_args: dict[str, Any]) -> dict[str, Any]:
        args: dict[str, Any] = {}
        if raw_args.get("duration") not in {None, 0, 0.0}:
            args["duration"] = raw_args["duration"]
        if raw_args.get("squeeze_duration") not in {None, 0, 0.0}:
            args["squeeze_duration"] = raw_args["squeeze_duration"]
        return record_robot_action("close_gripper", args)

    def handle_pour_into(raw_args: dict[str, Any]) -> dict[str, Any]:
        args: dict[str, Any] = {"apriltag_id": raw_args.get("apriltag_id")}
        if raw_args.get("duration") not in {None, 0, 0.0}:
            args["duration"] = raw_args["duration"]
        if raw_args.get("tilt_degrees") not in {None, 0, 0.0}:
            args["tilt_degrees"] = raw_args["tilt_degrees"]
        return record_robot_action("pour_into", args)

    def handle_return_to_origin(raw_args: dict[str, Any]) -> dict[str, Any]:
        args: dict[str, Any] = {}
        if raw_args.get("duration") not in {None, 0, 0.0}:
            args["duration"] = raw_args["duration"]
        return record_robot_action("return_to_origin", args)

    def handle_finish(raw_args: dict[str, Any]) -> dict[str, Any]:
        return record_robot_action("finish", {}, reasoning=str(raw_args.get("status") or "finished"))

    tool_handlers: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
        "move_arm": handle_move_arm,
        "move_arm_xyz": handle_move_arm_xyz,
        "open_gripper": handle_open_gripper,
        "close_gripper": handle_close_gripper,
        "pour_into": handle_pour_into,
        "return_to_origin": handle_return_to_origin,
        "finish": handle_finish,
    }

    function_declarations = [
        types.FunctionDeclaration(
            name="move_arm",
            description="Move the robot arm to a semantic AprilTag-relative target.",
            parameters={
                "type": "object",
                "properties": {
                    "apriltag_id": {"type": "integer", "description": "Configured target AprilTag ID from Scene JSON."},
                    "target": {"type": "string", "description": "One of approach, grasp, lift, place_above, or place."},
                    "duration": {"type": "number", "description": "Optional move duration in seconds."},
                    "tool_point": {"type": "string", "description": "Optional IK tool point: fixed_jaw_tip or claw_center."},
                },
                "required": ["apriltag_id", "target"],
            },
        ),
        types.FunctionDeclaration(
            name="move_arm_xyz",
            description="Move the robot arm to a raw world-frame XYZ target for recovery only.",
            parameters={
                "type": "object",
                "properties": {
                    "x": {"type": "number", "description": "World-frame X target in meters."},
                    "y": {"type": "number", "description": "World-frame Y target in meters."},
                    "z": {"type": "number", "description": "World-frame Z target in meters."},
                    "duration": {"type": "number", "description": "Optional move duration in seconds."},
                    "tool_point": {"type": "string", "description": "Optional IK tool point: fixed_jaw_tip or claw_center."},
                },
                "required": ["x", "y", "z"],
            },
        ),
        types.FunctionDeclaration(
            name="open_gripper",
            description="Open the gripper.",
            parameters={
                "type": "object",
                "properties": {"duration": {"type": "number", "description": "Optional gripper motion duration in seconds."}},
            },
        ),
        types.FunctionDeclaration(
            name="close_gripper",
            description="Close the gripper and optionally hold the squeeze.",
            parameters={
                "type": "object",
                "properties": {
                    "duration": {"type": "number", "description": "Optional gripper motion duration in seconds."},
                    "squeeze_duration": {"type": "number", "description": "Optional post-close squeeze hold duration in seconds."},
                },
            },
        ),
        types.FunctionDeclaration(
            name="pour_into",
            description="While holding one cup, move above a receiver cup and tilt the wrist to pour into it.",
            parameters={
                "type": "object",
                "properties": {
                    "apriltag_id": {"type": "integer", "description": "Receiver cup AprilTag ID."},
                    "duration": {"type": "number", "description": "Optional move-to-pour duration in seconds."},
                    "tilt_degrees": {"type": "number", "description": "Optional wrist-roll pour tilt in degrees."},
                },
                "required": ["apriltag_id"],
            },
        ),
        types.FunctionDeclaration(
            name="return_to_origin",
            description="Return the robot to the starting pose after releasing any held object.",
            parameters={
                "type": "object",
                "properties": {"duration": {"type": "number", "description": "Optional move duration in seconds."}},
            },
        ),
        types.FunctionDeclaration(
            name="finish",
            description="Finish the current task when complete or impossible.",
            parameters={
                "type": "object",
                "properties": {"status": {"type": "string", "description": "Concise reason why no more robot actions are needed."}},
                "required": ["status"],
            },
        ),
    ]

    def function_response_part(function_call: Any, result: dict[str, Any]) -> Any:
        response = {"result": result}
        try:
            return types.Part.from_function_response(
                id=getattr(function_call, "id", None),
                name=function_call.name,
                response=response,
            )
        except TypeError:
            return types.Part.from_function_response(
                name=function_call.name,
                response=response,
            )
        except AttributeError:
            return types.Part(
                function_response=types.FunctionResponse(
                    id=getattr(function_call, "id", None),
                    name=function_call.name,
                    response=response,
                )
            )

    def dispatch_function_call(function_call: Any) -> dict[str, Any]:
        raw_args = dict(function_call.args or {})
        handler = tool_handlers.get(function_call.name)
        if handler is None:
            result = _result(False, function_call.name, error="Unknown Gemini function call.")
            context.last_tool_result = result
            action_results.append(result)
            return result
        return handler(raw_args)

    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key) if api_key else genai.Client()
    initial_parts: list[Any] = [types.Part(text=prompt)]
    for camera_name, data_url in image_urls.items():
        image_bytes, mime_type = _decode_data_url(data_url)
        initial_parts.append(types.Part(text=f"Camera view: {camera_name}"))
        initial_parts.append(types.Part.from_bytes(data=image_bytes, mime_type=mime_type))

    contents: list[Any] = [types.Content(role="user", parts=initial_parts)]
    tool_config = types.Tool(function_declarations=function_declarations)
    tool_loop_exhausted = False
    raw_response = ""
    while True:
        response = _generate_with_retries(
            client,
            model=model,
            contents=contents,
            config=_gemini_config(
                types,
                temperature=temperature,
                thinking_budget=thinking_budget,
                tools=None if tool_loop_exhausted else [tool_config],
            ),
            api_retries=api_retries,
        )
        function_calls = list(getattr(response, "function_calls", None) or [])
        if not function_calls:
            raw_response = response.text or ""
            break

        contents.append(response.candidates[0].content)
        response_parts = []
        for function_call in function_calls:
            result = dispatch_function_call(function_call)
            response_parts.append(function_response_part(function_call, result))
        contents.append(types.Content(role="user", parts=response_parts))

        if len(actions) >= max_tool_calls or any(not result.get("success", False) for result in action_results):
            contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            text=(
                                "Stop calling robot tools for this agent step. "
                                "Summarize the tool result and wait for the next observation."
                            )
                        )
                    ],
                )
            )
            tool_loop_exhausted = True

    done = bool(actions and actions[-1].function == "finish") or not actions
    payload = {
        "reasoning": raw_response,
        "done": done,
        "status": "tool_loop_done" if done else "tool_loop_actions",
        "actions": [_action_payload(action) for action in actions],
        "mode": "tools",
    }
    plan = RobotActionPlan(
        reasoning=raw_response,
        status=payload["status"],
        done=done,
        actions=tuple(actions),
        raw_payload=payload,
    )
    return raw_response, plan, tuple(action_results)


def deterministic_reference_plan_payload(context: CupAgentContext) -> dict[str, Any]:
    evaluation = evaluate_cup_pour_success(context)
    config = context.config
    if evaluation["success"]:
        return {
            "reasoning": "Cup pour is already complete.",
            "done": True,
            "status": "reference_done",
            "actions": [],
        }

    if context.held_cup_label == evaluation["first_cup_label"]:
        return {
            "reasoning": "Fresh observation shows the first cup is held; place it on the flat tag.",
            "done": False,
            "status": "reference_place_first_cup",
            "actions": [
                {"function": "move_arm", "args": {"apriltag_id": config.place_tag_id, "target": "place_above"}},
                {"function": "move_arm", "args": {"apriltag_id": config.place_tag_id, "target": "place"}},
                {"function": "open_gripper", "args": {}},
                {"function": "return_to_origin", "args": {}},
            ],
        }

    if context.held_cup_label == evaluation["second_cup_label"]:
        return {
            "reasoning": "Fresh observation shows the second cup is held; pour it into the first cup.",
            "done": False,
            "status": "reference_pour_second_cup",
            "actions": [
                {"function": "pour_into", "args": {"apriltag_id": config.cup_tag_id}},
            ],
        }

    first_placed = (
        evaluation["first_xy_error_m"] <= config.place_success_xy_tolerance
        and evaluation["first_z_error_m"] <= config.place_success_z_tolerance
    )
    if not first_placed:
        return {
            "reasoning": "Fresh observation shows no cup is held and the first cup still needs to be picked.",
            "done": False,
            "status": "reference_pick_first_cup",
            "actions": [
                {"function": "move_arm", "args": {"apriltag_id": config.cup_tag_id, "target": "approach"}},
                {"function": "move_arm", "args": {"apriltag_id": config.cup_tag_id, "target": "grasp"}},
                {"function": "close_gripper", "args": {}},
                {"function": "move_arm", "args": {"apriltag_id": config.cup_tag_id, "target": "lift"}},
            ],
        }

    return {
        "reasoning": "Fresh observation shows the first cup is placed; pick the second cup next.",
        "done": False,
        "status": "reference_pick_second_cup",
        "actions": [
            {"function": "move_arm", "args": {"apriltag_id": config.second_cup_tag_id, "target": "approach"}},
            {"function": "move_arm", "args": {"apriltag_id": config.second_cup_tag_id, "target": "grasp"}},
            {"function": "close_gripper", "args": {}},
            {"function": "move_arm", "args": {"apriltag_id": config.second_cup_tag_id, "target": "lift"}},
        ],
    }


def _configured_apriltag_ids(context: CupAgentContext) -> set[int]:
    return {
        context.config.cup_tag_id,
        context.config.second_cup_tag_id,
        context.config.place_tag_id,
    }


def _cup_pour_guidance(previous_evaluation: dict[str, Any] | None) -> str:
    base = (
        "Follow the deterministic cup pouring policy one closed-loop phase at a time unless recovery is needed: "
        "pick/lift first cup; after fresh observation, place/release first cup on tag 0 and return to origin; "
        "after fresh observation, pick/lift second cup; after fresh observation, call pour_into(apriltag_id=6) "
        "while holding the second cup. Do not place or release the second cup on top of the first. "
        "Use raw XYZ only for recovery. Return only the next phase, not the full task."
    )
    if previous_evaluation is None:
        return base
    reasons = "; ".join(previous_evaluation.get("failure_reasons", []))
    if not reasons:
        return base
    return base + " Previous attempt failed because: " + reasons


def _generic_guidance(previous_evaluation: dict[str, Any] | None) -> str:
    base = (
        "Use the configured robot API tools to make conservative closed-loop progress on the requested task. "
        "Prefer semantic AprilTag moves when an object or target tag is available. Use raw XYZ only for recovery. "
        "Call finish with a status when the requested task is complete or impossible."
    )
    if previous_evaluation is None:
        return base
    reasons = "; ".join(previous_evaluation.get("failure_reasons", []))
    if not reasons:
        return base
    return base + " Previous attempt failed because: " + reasons


def evaluate_prompt_task(context: CupAgentContext) -> dict[str, Any]:
    pour_evaluation = evaluate_cup_pour_success(context)
    last_result = context.last_tool_result or {}
    success = bool(last_result.get("success") and last_result.get("action") == "finish")
    return {
        **pour_evaluation,
        "task": "prompt_task",
        "success": success,
        "failure_reasons": [] if success else ["generic prompt task has not been finished by the agent"],
    }


CUP_POUR_TASK_SPEC = GeminiTaskSpec(
    name="cup_pour",
    title="Pour from the second cup into the first.",
    instruction=(
        "The Scene JSON field task_state.next_phase is authoritative; choose the phase named there and do not "
        "repeat an earlier phase after it is marked complete. First pick up the first cup with AprilTag 6, move "
        "it to the flat placement tag with AprilTag 0, and release it there. Then pick up the second cup with "
        "AprilTag 1, move it above the first cup, and pour into the first cup using pour_into(apriltag_id=6). "
        "Do not place or release the second cup on top of the first."
    ),
    phase_policy=(
        "Default closed-loop phases for this cup scene:\n"
        "Phase 1, if no cup is held and the first cup is not at the placement tag: "
        "move_arm(tag 6, approach), move_arm(tag 6, grasp), close_gripper(), move_arm(tag 6, lift).\n"
        "Phase 2, if holding the first cup: move_arm(tag 0, place_above), move_arm(tag 0, place), "
        "open_gripper(), return_to_origin().\n"
        "Phase 3, if no cup is held, the first cup is placed, and the second cup has not poured: "
        "move_arm(tag 1, approach), move_arm(tag 1, grasp), close_gripper(), move_arm(tag 1, lift).\n"
        "Phase 4, if holding the second cup: pour_into(tag 6)."
    ),
    semantic_guidance=(
        "If task_state.next_phase is pick_second_cup_tag_1, the next pick actions must use apriltag_id 1, not 6. "
        "Use apriltag_id 6 after that only as the receiver target for pour_into while holding the second cup."
    ),
    completion_criteria="The first cup is released on the placement tag, and the held second cup has completed a pour gesture into the first cup.",
    evaluator=evaluate_cup_pour_success,
    reference_plan=deterministic_reference_plan_payload,
    allowed_apriltag_ids=_configured_apriltag_ids,
    guidance=_cup_pour_guidance,
)

GENERIC_PROMPT_TASK_SPEC = GeminiTaskSpec(
    name="generic",
    title="Follow the user-provided manipulation instruction.",
    instruction=(
        "Use the available robot APIs and Scene JSON to make conservative progress on the user instruction in "
        "attempt_guidance. The current simulator exposes cup tags and a flat placement tag; use those semantic "
        "targets when the instruction refers to them, otherwise use raw XYZ recovery moves only when the target "
        "is unambiguous from the scene."
    ),
    phase_policy=(
        "Plan only the next small action or short phase. Re-observe after every phase before continuing. "
        "Do not assume cup pouring is the goal unless the instruction explicitly asks for it."
    ),
    semantic_guidance="The configured AprilTags in Scene JSON are the available semantic references for this scene.",
    completion_criteria="The user-provided instruction has been satisfied, or the agent has determined it cannot be completed safely.",
    evaluator=evaluate_prompt_task,
    reference_plan=None,
    allowed_apriltag_ids=_configured_apriltag_ids,
    guidance=_generic_guidance,
)

GEMINI_TASK_SPECS: dict[str, GeminiTaskSpec] = {
    CUP_POUR_TASK_SPEC.name: CUP_POUR_TASK_SPEC,
    "cup_stack": CUP_POUR_TASK_SPEC,
    GENERIC_PROMPT_TASK_SPEC.name: GENERIC_PROMPT_TASK_SPEC,
}
DEFAULT_TASK_SPEC = CUP_POUR_TASK_SPEC


def _result(success: bool, action: str, **extra: Any) -> dict[str, Any]:
    return {"success": success, "action": action, **extra}


def execute_move_arm(context: CupAgentContext, args: dict[str, Any]) -> dict[str, Any]:
    try:
        target = args.get("target", "custom")
        allowed_support_body_names: tuple[str, ...] = ()
        if args.get("apriltag_id") is not None and target != "custom":
            tool_point = args.get("tool_point", FIXED_JAW_TOOL_POINT)
            target_xyz, active_cup, diagnostics, target_source, allowed_support_body_names = _semantic_target_for_apriltag(
                context,
                int(args["apriltag_id"]),
                str(target),
                tool_point,
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
    stack_release = _last_move_was_stack_place(context)
    _cup, diagnostics = _active_cup_and_diagnostics(context)
    try:
        release_duration = (
            context.config.stack_release_open_duration
            if args.get("duration") is None and stack_release
            else context.gripper_duration if args.get("duration") is None else float(args["duration"])
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
            release_retreat = _retreat_after_release(context, _cup, diagnostics)
            success = release_retreat
        context.last_tool_result = _result(
            success,
            "open_gripper",
            release_retreat=release_retreat,
            stack_release=stack_release,
            release_settle=release_settle,
        )
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
    if action.function == "pour_into":
        return execute_pour_into(
            context,
            int(action.args["apriltag_id"]),
            duration=action.args.get("duration"),
            tilt_degrees=action.args.get("tilt_degrees"),
        )
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
        print(f"tool/actions: {len(result.plan.actions)}")
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
    task_spec: GeminiTaskSpec | None = None,
    agent_mode: GeminiAgentMode = DEFAULT_GEMINI_AGENT_MODE,
    api_retries: int = 2,
) -> list[GeminiStepResult]:
    task_spec = task_spec or DEFAULT_TASK_SPEC
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
        prompt = build_gemini_prompt(
            observation,
            max_actions=max_plan_actions,
            task_spec=task_spec,
            agent_mode=agent_mode,
        )
        prompt_path = artifact_dir / f"prompt_{step_index:03d}.txt" if artifact_dir is not None else None
        if prompt_path is not None:
            prompt_path.write_text(prompt + "\n", encoding="utf-8")

        validation_errors: tuple[str, ...] = ()
        plan: RobotActionPlan | None = None
        action_results: list[dict[str, Any]] = []
        try:
            if reference_plan:
                if task_spec.reference_plan is None:
                    raise PlanValidationError(f"Task {task_spec.name!r} does not provide a reference plan.")
                raw_response = json.dumps(task_spec.reference_plan(context))
                plan = parse_and_validate_action_plan(
                    context,
                    raw_response,
                    max_actions=max_plan_actions,
                    task_spec=task_spec,
                )
            elif agent_mode == "json":
                raw_response = call_gemini_action_planner(
                    prompt,
                    image_urls,
                    model=model,
                    temperature=temperature,
                    thinking_budget=thinking_budget,
                    api_retries=api_retries,
                )
                plan = parse_and_validate_action_plan(
                    context,
                    raw_response,
                    max_actions=max_plan_actions,
                    task_spec=task_spec,
                )
            else:
                raw_response, plan, tool_results = call_gemini_tool_agent(
                    context,
                    prompt,
                    image_urls,
                    model=model,
                    temperature=temperature,
                    thinking_budget=thinking_budget,
                    max_tool_calls=max_plan_actions,
                    execute_actions=execute_actions,
                    task_spec=task_spec,
                    api_retries=api_retries,
                )
                action_results.extend(tool_results)
        except Exception as exc:
            raw_response = locals().get("raw_response", "")
            validation_errors = (f"{type(exc).__name__}: {exc}",)
            context.last_tool_result = _result(False, "gemini_plan", error=validation_errors[0])

        execute_plan_actions = reference_plan or agent_mode == "json"
        if execute_plan_actions and execute_actions and plan is not None and not plan.done:
            for action in plan.actions:
                action_result = execute_robot_action(context, action)
                action_results.append(action_result)
                if not action_result.get("success", False) or action.function == "finish":
                    break

        evaluation = task_spec.evaluator(context)
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
            print("cup-pour success detected; stopping early")
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
