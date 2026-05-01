"""
Main agent loop for the matcha-making robot.

Each iteration observes the world, asks Claude for the next tool call, validates
and dispatches it, then records the result.  The loop terminates when the model
calls done(), the step budget is exhausted, or 3 consecutive errors occur.

No API key is hard-coded — the Anthropic client reads ANTHROPIC_API_KEY from
the environment automatically.
"""

from __future__ import annotations

import json
import logging
from typing import Callable

import anthropic

from agent.prompts import AVAILABLE_TOOLS, SYSTEM_PROMPT, format_state
from agent.tools import validate_tool_call

logger = logging.getLogger(__name__)

_SYSTEM = SYSTEM_PROMPT + "\n\n" + AVAILABLE_TOOLS
_MAX_CONSECUTIVE_ERRORS = 3


def _strip_code_fence(text: str) -> str:
    """Remove markdown code fences that some models wrap JSON in."""
    text = text.strip()
    if text.startswith("```"):
        # Drop the opening fence line (```json or just ```)
        text = text.split("\n", 1)[-1]
        # Drop the closing fence
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
    return text.strip()


def run_agent_loop(
    task: str,
    get_poses_fn: Callable[[], dict],
    execute_tool_fn: Callable[[dict, dict], dict],
    model: str = "claude-haiku-4-5-20251001",
    max_steps: int = 20,
    verbose: bool = True,
) -> list[dict]:
    """
    Run the LLM-driven planning loop until done or the budget is exhausted.

    Each iteration:
      1. Fetch current object poses via *get_poses_fn*.
      2. Build a state prompt from poses + history.
      3. Call the Claude model; expect raw JSON text.
      4. Parse JSON — on failure record error and retry.
      5. Validate the tool-call schema.
      6. Dispatch via *execute_tool_fn(tool_call, pose_map) → result*.
      7. Append ``{"step", "action", "result"}`` to history.
      8. Break on done(), 3 consecutive errors, or step budget.

    Args:
        task:            Natural-language task description.
        get_poses_fn:    Callable returning ``{object_name: [x, y, z]}`` dict.
        execute_tool_fn: Callable ``(tool_call: dict, pose_map: dict) → dict``.
                         Wraps :func:`agent.tools.dispatch_tool` with a bound
                         executor.  Swap this closure to change the executor.
        model:           Anthropic model ID.
        max_steps:       Hard iteration cap.
        verbose:         Print per-step progress to stdout when True.

    Returns:
        List of ``{"step": int, "action": dict|None, "result": dict}`` entries
        in chronological order.  Includes failed/parse-error entries so callers
        can audit every LLM response.
    """
    client = anthropic.Anthropic()  # ANTHROPIC_API_KEY read from environment

    history:         list[dict]  = []
    completed_steps: list[dict]  = []
    last_result:     dict | None = None
    consecutive_errors: int      = 0
    gripper_open:    bool        = True   # track gripper state for the prompt
    holding:         str | None  = None   # object currently held

    for step in range(1, max_steps + 1):
        if verbose:
            print(f"\n{'─' * 60}")
            print(f"[Step {step}/{max_steps}]")

        # 1. Observe current world state
        poses = get_poses_fn()

        # 2. Format state prompt (include gripper state so model knows what it holds)
        state_prompt = format_state(
            poses, completed_steps, last_result, task,
            gripper_open=gripper_open, holding=holding,
        )
        if verbose:
            print(state_prompt)

        # 3. Call the LLM
        try:
            response = client.messages.create(
                model=model,
                max_tokens=256,
                system=_SYSTEM,
                messages=[{"role": "user", "content": state_prompt}],
            )
            raw_text = _strip_code_fence(response.content[0].text)
        except Exception as exc:
            last_result = {"status": "error", "reason": f"LLM API error: {exc}"}
            history.append({"step": step, "action": None, "result": last_result})
            consecutive_errors += 1
            logger.warning("LLM API error at step %d: %s", step, exc)
            if consecutive_errors >= _MAX_CONSECUTIVE_ERRORS:
                if verbose:
                    print(f"[Abort] {_MAX_CONSECUTIVE_ERRORS} consecutive failures.")
                break
            continue

        if verbose:
            print(f"LLM → {raw_text}")

        # 4. Parse JSON
        try:
            tool_call: dict = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            last_result = {"status": "error", "reason": f"invalid json, retry: {exc}"}
            history.append({"step": step, "action": None, "result": last_result})
            consecutive_errors += 1
            if verbose:
                print(f"[JSON error] {exc}")
            if consecutive_errors >= _MAX_CONSECUTIVE_ERRORS:
                if verbose:
                    print(f"[Abort] {_MAX_CONSECUTIVE_ERRORS} consecutive failures.")
                break
            continue

        # 5. Validate schema
        is_valid, error_msg = validate_tool_call(tool_call)
        if not is_valid:
            last_result = {"status": "error", "reason": error_msg}
            history.append({"step": step, "action": tool_call, "result": last_result})
            consecutive_errors += 1
            if verbose:
                print(f"[Validation error] {error_msg}")
            if consecutive_errors >= _MAX_CONSECUTIVE_ERRORS:
                if verbose:
                    print(f"[Abort] {_MAX_CONSECUTIVE_ERRORS} consecutive failures.")
                break
            continue

        # 6. Dispatch
        result = execute_tool_fn(tool_call, poses)
        last_result = result
        history.append({"step": step, "action": tool_call, "result": result})

        if verbose:
            status = result.get("status", "?")
            print(
                f"Tool: {tool_call['tool']}({tool_call.get('args', {})}) → {status}"
            )

        # 6b. Update gripper / holding state for the next prompt
        if result.get("status") == "success":
            tool = tool_call["tool"]
            args = tool_call.get("args", {})
            if tool == "open_gripper":
                gripper_open = True
                holding = None
            elif tool == "close_gripper":
                gripper_open = False
            elif tool == "move_arm":
                action = args.get("action")
                if action == "grasp":
                    gripper_open = False
                    holding = args.get("target_object")
                elif action == "place":
                    gripper_open = True
                    holding = None

        # 7. Check termination
        if result.get("status") == "done":
            if verbose:
                print("\n[Done] Task complete.")
            break

        if result.get("status") == "error":
            consecutive_errors += 1
            if consecutive_errors >= _MAX_CONSECUTIVE_ERRORS:
                if verbose:
                    print(f"[Abort] {_MAX_CONSECUTIVE_ERRORS} consecutive failures.")
                break
        else:
            consecutive_errors = 0
            # Only record successful steps in the history fed back to the LLM
            completed_steps.append(
                {"step": step, "action": tool_call, "result": result}
            )

    return history
