"""
Tests for agent/loop.py

All Anthropic API calls are mocked — no real network requests are made.
The patch target is 'agent.loop.anthropic.Anthropic' because that is the
name as imported inside the loop module.
"""

from __future__ import annotations

import json
import pytest
from unittest.mock import MagicMock, patch

from agent.loop import run_agent_loop

# ---------------------------------------------------------------------------
# Shared fixtures and helpers
# ---------------------------------------------------------------------------

SAMPLE_POSES: dict[str, list[float]] = {
    "matcha_cup":  [0.30, 0.02, 0.40, 0.0],
    "matcha_bowl": [0.10, 0.02, 0.40, 0.0],
}


def _make_llm_response(tool_call_dict: dict) -> MagicMock:
    """Build a fake Anthropic message response that returns JSON text."""
    mock_resp = MagicMock()
    mock_resp.content = [MagicMock()]
    mock_resp.content[0].text = json.dumps(tool_call_dict)
    return mock_resp


def _make_invalid_json_response() -> MagicMock:
    """Return a fake response containing unparseable text."""
    mock_resp = MagicMock()
    mock_resp.content = [MagicMock()]
    mock_resp.content[0].text = "not valid json {{{"
    return mock_resp


def _success_executor(tool_call: dict, pose_map: dict) -> dict:
    """Succeeds for all tools; returns done status for done()."""
    if tool_call.get("tool") == "done":
        return {"status": "done"}
    return {"status": "success"}


def _error_executor(tool_call: dict, pose_map: dict) -> dict:
    """Always returns error — used to trigger the abort path."""
    return {"status": "error", "reason": "mock failure"}


# ---------------------------------------------------------------------------
# Test: done() terminates the loop
# ---------------------------------------------------------------------------

@patch("agent.loop.anthropic.Anthropic")
def test_agent_calls_done_and_terminates(mock_cls):
    """Loop must stop after the first done() response with a done status."""
    mock_client = MagicMock()
    mock_cls.return_value = mock_client
    mock_client.messages.create.return_value = _make_llm_response(
        {"tool": "done", "args": {}}
    )

    history = run_agent_loop(
        task="test",
        get_poses_fn=lambda: dict(SAMPLE_POSES),
        execute_tool_fn=_success_executor,
        verbose=False,
    )

    assert len(history) == 1
    assert history[0]["result"]["status"] == "done"
    # LLM should have been called exactly once
    assert mock_client.messages.create.call_count == 1


# ---------------------------------------------------------------------------
# Test: graceful handling of invalid JSON
# ---------------------------------------------------------------------------

@patch("agent.loop.anthropic.Anthropic")
def test_agent_handles_invalid_json_gracefully(mock_cls):
    """Invalid JSON from the LLM produces error entries, not exceptions."""
    mock_client = MagicMock()
    mock_cls.return_value = mock_client
    mock_client.messages.create.side_effect = [
        _make_invalid_json_response(),
        _make_invalid_json_response(),
        _make_llm_response({"tool": "done", "args": {}}),
    ]

    history = run_agent_loop(
        task="test",
        get_poses_fn=lambda: dict(SAMPLE_POSES),
        execute_tool_fn=_success_executor,
        verbose=False,
    )

    assert len(history) == 3

    # First two entries are JSON errors
    assert history[0]["result"]["status"] == "error"
    assert "json" in history[0]["result"]["reason"].lower()
    assert history[1]["result"]["status"] == "error"

    # Third entry is the successful done
    assert history[2]["result"]["status"] == "done"


# ---------------------------------------------------------------------------
# Test: retries on error result from executor
# ---------------------------------------------------------------------------

@patch("agent.loop.anthropic.Anthropic")
def test_agent_retries_after_executor_error(mock_cls):
    """An error result from the executor should not stop the loop prematurely."""
    mock_client = MagicMock()
    mock_cls.return_value = mock_client
    mock_client.messages.create.side_effect = [
        _make_llm_response(
            {"tool": "move_arm", "args": {"target_object": "matcha_cup", "action": "hover"}}
        ),
        _make_llm_response({"tool": "done", "args": {}}),
    ]

    call_log: list[str] = []

    def mixed_executor(tool_call: dict, pose_map: dict) -> dict:
        call_log.append(tool_call["tool"])
        if tool_call["tool"] == "done":
            return {"status": "done"}
        # First non-done call fails
        return {"status": "error", "reason": "first attempt failed"}

    history = run_agent_loop(
        task="test",
        get_poses_fn=lambda: dict(SAMPLE_POSES),
        execute_tool_fn=mixed_executor,
        verbose=False,
    )

    # History must have the error then the done
    assert history[0]["result"]["status"] == "error"
    assert history[-1]["result"]["status"] == "done"
    assert "move_arm" in call_log
    assert "done" in call_log


# ---------------------------------------------------------------------------
# Test: abort after 3 consecutive failures
# ---------------------------------------------------------------------------

@patch("agent.loop.anthropic.Anthropic")
def test_agent_aborts_after_3_consecutive_failures(mock_cls):
    """Loop must stop after exactly 3 consecutive error results."""
    mock_client = MagicMock()
    mock_cls.return_value = mock_client
    # LLM always returns a valid-schema move_arm; executor always errors
    mock_client.messages.create.return_value = _make_llm_response(
        {"tool": "move_arm", "args": {"target_object": "matcha_cup", "action": "hover"}}
    )

    history = run_agent_loop(
        task="test",
        get_poses_fn=lambda: dict(SAMPLE_POSES),
        execute_tool_fn=_error_executor,
        verbose=False,
        max_steps=20,
    )

    assert len(history) == 3
    for entry in history:
        assert entry["result"]["status"] == "error"


# ---------------------------------------------------------------------------
# Test: history structure
# ---------------------------------------------------------------------------

@patch("agent.loop.anthropic.Anthropic")
def test_history_entries_have_required_keys(mock_cls):
    """Every history entry must contain 'step', 'action', and 'result'."""
    mock_client = MagicMock()
    mock_cls.return_value = mock_client
    mock_client.messages.create.side_effect = [
        _make_llm_response({"tool": "open_gripper", "args": {}}),
        _make_llm_response({"tool": "wait", "args": {"seconds": 0.0}}),
        _make_llm_response({"tool": "done", "args": {}}),
    ]

    history = run_agent_loop(
        task="test",
        get_poses_fn=lambda: dict(SAMPLE_POSES),
        execute_tool_fn=_success_executor,
        verbose=False,
    )

    assert len(history) >= 3
    for entry in history:
        assert "step"   in entry, f"missing 'step' key in {entry}"
        assert "action" in entry, f"missing 'action' key in {entry}"
        assert "result" in entry, f"missing 'result' key in {entry}"
        assert isinstance(entry["step"], int)
        assert "status" in entry["result"]

    # Steps must be strictly increasing
    steps = [e["step"] for e in history]
    assert steps == sorted(steps)
    assert steps == list(range(1, len(steps) + 1))


# ---------------------------------------------------------------------------
# Test: max_steps cap
# ---------------------------------------------------------------------------

@patch("agent.loop.anthropic.Anthropic")
def test_loop_respects_max_steps(mock_cls):
    """Loop must never exceed max_steps iterations."""
    mock_client = MagicMock()
    mock_cls.return_value = mock_client
    # LLM always returns open_gripper (success) — never done
    mock_client.messages.create.return_value = _make_llm_response(
        {"tool": "open_gripper", "args": {}}
    )

    max_steps = 5
    history = run_agent_loop(
        task="test",
        get_poses_fn=lambda: dict(SAMPLE_POSES),
        execute_tool_fn=_success_executor,
        verbose=False,
        max_steps=max_steps,
    )

    assert len(history) <= max_steps
