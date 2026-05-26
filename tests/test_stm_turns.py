"""短期记忆：按轮次切分消息的单元测试。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    UserPromptPart,
)
from tools.memory import (
    split_messages_into_turns,
    turn_has_agent_content,
    turn_texts_from_messages,
)


def test_split_single_turn():
    msgs = [
        ModelRequest(parts=[UserPromptPart(content="hello")]),
        ModelResponse(parts=[TextPart(content="hi there")]),
    ]
    turns = split_messages_into_turns(msgs)
    assert len(turns) == 1
    assert turn_has_agent_content(turns[0])
    texts = turn_texts_from_messages(msgs)
    assert len(texts) == 1
    assert "[USER]: hello" in texts[0]
    assert "[ASSISTANT]: hi there" in texts[0]


def test_split_two_turns():
    msgs = [
        ModelRequest(parts=[UserPromptPart(content="q1")]),
        ModelResponse(parts=[TextPart(content="a1")]),
        ModelRequest(parts=[UserPromptPart(content="q2")]),
        ModelResponse(parts=[ToolCallPart(tool_name="t", args="{}")]),
    ]
    turns = split_messages_into_turns(msgs)
    assert len(turns) == 2
    assert "[USER]: q1" in turn_texts_from_messages(msgs)[0]
    assert "[TOOL_CALL:t]" in turn_texts_from_messages(msgs)[1]


def test_skip_user_only_turn():
    msgs = [
        ModelRequest(parts=[UserPromptPart(content="alone")]),
    ]
    assert split_messages_into_turns(msgs) == []


def test_empty_messages():
    assert split_messages_into_turns([]) == []


if __name__ == "__main__":
    test_split_single_turn()
    test_split_two_turns()
    test_skip_user_only_turn()
    test_empty_messages()
    print("ok")
