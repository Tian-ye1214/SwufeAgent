from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import logger
from runtime_state import AgentRunPolicy, TRACE_STORE, current_turn_id, turn_context


def test_agent_run_policy_uses_compatible_defaults() -> None:
    policy = AgentRunPolicy.from_config({})

    assert policy.max_worker_concurrent == 3
    assert policy.max_tool_output_chars > 100
    assert policy.max_command_timeout_seconds == 60


def test_tool_wrapper_truncates_long_string_and_records_trace() -> None:
    TRACE_STORE.clear()
    policy = AgentRunPolicy(max_worker_concurrent=2, max_tool_output_chars=12, max_command_timeout_seconds=5)

    def noisy_tool() -> str:
        return "abcdefghijklmnopqrstuvwxyz"

    wrapped = logger.wrap_tools_for_user_notify([noisy_tool], policy=policy)[0]
    with turn_context("turn-abc"):
        result = wrapped()

    assert result.startswith("abcdefghijkl")
    assert "truncated" in result
    events = TRACE_STORE.events_for_turn("turn-abc")
    assert len(events) == 1
    assert events[0]["kind"] == "tool_call"
    assert events[0]["tool_name"] == "noisy_tool"
    assert events[0]["success"] is True
    assert current_turn_id() is None


def test_tool_wrapper_records_failures() -> None:
    TRACE_STORE.clear()

    def broken_tool() -> str:
        raise RuntimeError("boom")

    wrapped = logger.wrap_tools_for_user_notify([broken_tool])[0]
    try:
        with turn_context("turn-fail"):
            wrapped()
    except RuntimeError:
        pass

    events = TRACE_STORE.events_for_turn("turn-fail")
    assert len(events) == 1
    assert events[0]["success"] is False
    assert "RuntimeError" in events[0]["error"]
