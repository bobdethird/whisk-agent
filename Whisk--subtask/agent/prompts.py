"""
Prompt templates and world-state formatting for the matcha-making agent.

Keeping token counts low is important: the state prompt is sent every step,
so format_state() deliberately truncates and summarises rather than dumps
the full history.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT: str = """\
You are the planning brain of a robotic arm making matcha tea.
Your job: issue exactly ONE tool call per turn as a raw JSON object.

Rules
-----
1. Output ONLY valid JSON — no prose, no markdown, no explanation.
2. Format: {"tool": "<name>", "args": {<key>: <value>}}
3. If a tool takes no arguments use "args": {}.
4. Work through the SUBTASKS in order. Tick off each one before moving on.
5. A subtask is done when the required move_arm actions completed successfully.
6. Call done() only after ALL subtasks are marked complete.
7. Do NOT repeat a subtask you have already completed.\
"""

# ---------------------------------------------------------------------------
# Tool catalogue (injected into the system message)
# ---------------------------------------------------------------------------

AVAILABLE_TOOLS: str = """\
AVAILABLE TOOLS
===============
move_arm(target_object: str, action: str)
    Move the arm to an object and perform an action.
    target_object: one of the object names shown in OBJECTS.
    action: "grasp"  — pick up the object (gripper closes on it)
            "place"  — set held object down at target location (gripper opens)
            "hover"  — move above object without gripping

open_gripper()
    Fully open the gripper. Call before grasping a new object.

close_gripper()
    Fully close the gripper.

wait(seconds: float)
    Pause (e.g. let powder dissolve or water settle).

done()
    Signal that ALL subtasks are complete. Call this last.\
"""

# ---------------------------------------------------------------------------
# Fixed subtask list shown to the model every step
# ---------------------------------------------------------------------------

SUBTASKS: list[str] = [
    "1. Grasp matcha_scoop → place at matcha_bowl (scooping powder)",
    "2. Grasp cup_of_water → place at matcha_bowl (adding hot water)",
    "3. Grasp cup_of_ice   → place at matcha_bowl (adding ice)",
    "4. Grasp matcha_whisk → place at matcha_bowl (whisk until frothy)",
    "5. Grasp matcha_bowl  → place at matcha_cup  (pour into cup)",
]


# ---------------------------------------------------------------------------
# State formatter
# ---------------------------------------------------------------------------

def format_state(
    poses: dict[str, list[float]],
    completed_steps: list[dict],
    last_result: dict | None,
    task_description: str,
    gripper_open: bool = True,
    holding: str | None = None,
) -> str:
    """
    Format current world state into a concise LLM prompt (< 500 tokens).

    Conciseness strategies:
    - Poses rounded to 3 decimal places.
    - Only the last 5 completed steps are shown.
    - Result dicts summarised to status + optional reason.

    Args:
        poses:            Object-name → [x, y, z] from the detector.
        completed_steps:  All steps completed so far; only the last 5 shown.
        last_result:      Result dict from the previous tool call, or None.
        task_description: Natural-language task description (used as header).
        gripper_open:     Whether the gripper is currently open.
        holding:          Name of object currently held, or None.

    Returns:
        Formatted string ready to be sent as the user message.
    """
    lines: list[str] = []

    # --- Subtask checklist --------------------------------------------------
    lines.append("SUBTASKS (work through these in order):")
    completed_count = _count_completed_subtasks(completed_steps)
    for i, subtask in enumerate(SUBTASKS):
        marker = "[x]" if i < completed_count else "[ ]"
        lines.append(f"  {marker} {subtask}")

    # --- Gripper state ------------------------------------------------------
    gripper_str = "OPEN" if gripper_open else "CLOSED"
    held_str    = f", holding: {holding}" if holding else ""
    lines.append(f"\nGRIPPER: {gripper_str}{held_str}")

    # --- Object poses -------------------------------------------------------
    # Yaw is omitted from the prompt: the LLM doesn't pick approach geometry
    # (GRASP_CONFIGS does), so rendering yaw would just burn tokens.
    lines.append("OBJECTS:")
    for name, pose in poses.items():
        x, y, z = pose[0], pose[1], pose[2]
        lines.append(f"  {name}: x={x:.3f} y={y:.3f} z={z:.3f}")

    # --- Recent history (last 5 steps) -------------------------------------
    recent = completed_steps[-5:]
    if recent:
        lines.append("\nRECENT ACTIONS:")
        for entry in recent:
            action = entry.get("action") or {}
            tool   = action.get("tool", "?")
            args   = action.get("args", {})
            status = entry.get("result", {}).get("status", "?")
            args_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
            lines.append(f"  [{entry['step']}] {tool}({args_str}) → {status}")

    # --- Last result --------------------------------------------------------
    if last_result is not None:
        status = last_result.get("status", "?")
        reason = last_result.get("reason", "")
        summary = status + (f" — {reason}" if reason else "")
        lines.append(f"\nLAST RESULT: {summary}")

    return "\n".join(lines)


def _count_completed_subtasks(completed_steps: list[dict]) -> int:
    """
    Infer how many subtasks are done by scanning the action history.

    Subtask N is considered complete when the required grasp→place pair
    appears in completed_steps in order.  This is a simple heuristic
    sufficient for the mock run — a real system would track explicit state.
    """
    # Each subtask is defined by (grasp target, place target)
    subtask_signatures = [
        ("matcha_scoop", "matcha_bowl"),
        ("cup_of_water", "matcha_bowl"),
        ("cup_of_ice",   "matcha_bowl"),
        ("matcha_whisk", "matcha_bowl"),
        ("matcha_bowl",  "matcha_cup"),
    ]

    actions = [
        (e["action"].get("args", {}).get("target_object"),
         e["action"].get("args", {}).get("action"))
        for e in completed_steps
        if e.get("action") and e["action"].get("tool") == "move_arm"
    ]

    completed = 0
    action_idx = 0
    for grasp_target, place_target in subtask_signatures:
        # Find a grasp of grasp_target followed by a place of place_target
        found_grasp = False
        while action_idx < len(actions):
            obj, act = actions[action_idx]
            action_idx += 1
            if obj == grasp_target and act == "grasp":
                found_grasp = True
                break
        if not found_grasp:
            break
        found_place = False
        while action_idx < len(actions):
            obj, act = actions[action_idx]
            action_idx += 1
            if obj == place_target and act == "place":
                found_place = True
                break
        if not found_place:
            break
        completed += 1

    return completed
