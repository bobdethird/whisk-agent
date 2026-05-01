"""
Tool dispatch layer between the LLM agent loop and the arm executor.

All tool calls from the LLM pass through :func:`validate_tool_call` then
:func:`dispatch_tool`.  The *executor* argument is typed via the
:class:`arm.executor.Executor` Protocol so that swapping MockExecutor for
a real LeRobot executor requires zero changes here.
"""

from __future__ import annotations

import time

from arm.grasp_configs import GRASP_CONFIGS
from arm.trajectory import (
    compute_grasp_trajectory,
    compute_place_trajectory,
    compute_side_grasp_trajectory,
    compute_trajectory,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_TOOLS: frozenset[str] = frozenset(
    {"move_arm", "open_gripper", "close_gripper", "wait", "done"}
)
VALID_ACTIONS: frozenset[str] = frozenset({"grasp", "place", "hover"})

_GRIPPER_OPEN_M:   float = 0.08   # fully open — 8 cm
_GRIPPER_CLOSED_M: float = 0.0    # fully closed


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_tool_call(tool_call: dict) -> tuple[bool, str]:
    """
    Validate a parsed tool-call dict before dispatching it.

    Checks:
    - *tool_call* is a dict with ``"tool"`` and ``"args"`` keys.
    - ``tool`` is a known tool name.
    - ``args`` is a dict.
    - ``move_arm`` has ``target_object`` (str) and ``action`` (valid string).
    - ``wait`` has ``seconds`` (numeric).

    Args:
        tool_call: Parsed JSON from the LLM response.

    Returns:
        ``(True, "")`` on success.
        ``(False, error_message)`` on the first schema violation found.
    """
    if not isinstance(tool_call, dict):
        return False, "tool_call must be a JSON object (dict)"

    tool = tool_call.get("tool")
    if tool is None:
        return False, "missing required key 'tool'"
    if not isinstance(tool, str):
        return False, f"'tool' must be a string, got {type(tool).__name__}"
    if tool not in VALID_TOOLS:
        return False, f"unknown tool '{tool}'; valid tools: {sorted(VALID_TOOLS)}"

    args = tool_call.get("args")
    if args is None:
        return False, "missing required key 'args'"
    if not isinstance(args, dict):
        return False, f"'args' must be a JSON object (dict), got {type(args).__name__}"

    if tool == "move_arm":
        if "target_object" not in args:
            return False, "move_arm requires 'target_object' in args"
        if not isinstance(args["target_object"], str):
            return False, "'target_object' must be a string"
        if "action" not in args:
            return False, "move_arm requires 'action' in args"
        if args["action"] not in VALID_ACTIONS:
            return False, (
                f"invalid action '{args['action']}'; "
                f"valid: {sorted(VALID_ACTIONS)}"
            )

    if tool == "wait":
        if "seconds" not in args:
            return False, "wait requires 'seconds' in args"
        if not isinstance(args["seconds"], (int, float)):
            return False, f"'seconds' must be numeric, got {type(args['seconds']).__name__}"
        if args["seconds"] < 0:
            return False, "'seconds' must be non-negative"

    return True, ""


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def dispatch_tool(tool_call: dict, pose_map: dict, executor) -> dict:
    """
    Route a validated tool call to the appropriate executor function.

    This function assumes :func:`validate_tool_call` has already passed.
    Unknown tools that slip through return an error dict rather than raising.

    Args:
        tool_call: Dict with ``"tool"`` and ``"args"`` keys.
        pose_map:  Current ``{object_name: [x, y, z]}`` map.
        executor:  Object implementing :class:`arm.executor.Executor` Protocol.

    Returns:
        Result dict containing at minimum ``{"status": "success"|"error"|"done"}``.
        On error also contains ``{"reason": str}``.
    """
    tool = tool_call["tool"]
    args = tool_call.get("args", {})

    # ------------------------------------------------------------------ done
    if tool == "done":
        return {"status": "done"}

    # ---------------------------------------------------------- open_gripper
    if tool == "open_gripper":
        return executor.set_gripper(_GRIPPER_OPEN_M)

    # --------------------------------------------------------- close_gripper
    if tool == "close_gripper":
        return executor.set_gripper(_GRIPPER_CLOSED_M)

    # --------------------------------------------------------------- wait
    if tool == "wait":
        seconds = float(args["seconds"])
        time.sleep(seconds)
        return {"status": "success", "waited_seconds": seconds}

    # ------------------------------------------------------------- move_arm
    if tool == "move_arm":
        target_obj = args["target_object"]
        action     = args["action"]

        if target_obj not in pose_map:
            known = sorted(pose_map.keys())
            return {
                "status": "error",
                "reason": f"'{target_obj}' not in pose map; known objects: {known}",
            }

        target_pose  = pose_map[target_obj]
        current_pose = executor.get_end_effector_pose()

        # Every other object on the workbench is a potential obstacle.
        # Pair each with its object_height (defaulting to 0 if unknown).
        obstacles = [
            (pose, float(GRASP_CONFIGS.get(name, {}).get("object_height", 0.0)))
            for name, pose in pose_map.items()
            if name != target_obj
        ]

        if action == "grasp":
            if target_obj not in GRASP_CONFIGS:
                return {
                    "status": "error",
                    "reason": f"no grasp config for '{target_obj}'",
                }
            cfg = GRASP_CONFIGS[target_obj]

            # Pre-open gripper to the per-object grip_width so jaws are
            # already close to the correct opening when descent begins.
            executor.set_gripper(float(cfg.get("grip_width", _GRIPPER_OPEN_M)))

            approach_axis = cfg.get("approach_axis", "y")
            if approach_axis == "y":
                waypoints = compute_grasp_trajectory(
                    current_pose, target_pose, cfg, obstacles=obstacles,
                )
            else:  # "yaw_aligned", "x", "z" — all side-approach variants
                waypoints = compute_side_grasp_trajectory(
                    current_pose, target_pose, cfg, obstacles=obstacles,
                )

        elif action == "place":
            waypoints = compute_place_trajectory(
                current_pose, target_pose, obstacles=obstacles,
            )

        else:  # hover
            waypoints = compute_trajectory(
                current_pose, target_pose, obstacles=obstacles,
            )

        # Execute each waypoint; abort on first failure.
        for idx, wp in enumerate(waypoints):
            result = executor.move_to(wp)
            if result.get("status") != "success":
                return {
                    "status": "error",
                    "reason": (
                        f"move_to failed at waypoint {idx}: "
                        f"{result.get('reason', 'unknown error')}"
                    ),
                }

        return {
            "status": "success",
            "action": action,
            "target": target_obj,
            "waypoints_executed": len(waypoints),
        }

    # Should never reach here after validation.
    return {"status": "error", "reason": f"unhandled tool '{tool}'"}
