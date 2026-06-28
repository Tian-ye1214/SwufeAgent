from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from pydantic_ai.messages import (
    BaseToolReturnPart,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    UserPromptPart,
)


@dataclass(frozen=True)
class TurnMemoryEntry:
    text: str
    token_reference: int


def pydantic_messages_to_text(messages: list) -> str:
    """Convert visible pydantic-ai messages to text for memory ingestion."""
    lines: list[str] = []
    for msg in messages:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, UserPromptPart):
                    content = part.content
                    if isinstance(content, str):
                        lines.append(f"[USER]: {content}")
                elif isinstance(part, BaseToolReturnPart):
                    c = part.model_response_str() if hasattr(part, "model_response_str") else str(part.content)
                    lines.append(f"[TOOL_RESULT:{part.tool_name}]: {c[:500]}")
        elif isinstance(msg, ModelResponse):
            for part in msg.parts:
                if isinstance(part, TextPart):
                    lines.append(f"[ASSISTANT]: {part.content}")
                elif isinstance(part, ToolCallPart):
                    args_str = part.args if isinstance(part.args, str) else str(part.args)
                    lines.append(f"[TOOL_CALL:{part.tool_name}]: {args_str[:300]}")
    return "\n\n".join(lines)


def _safe_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _usage_detail_int(usage: Any, key: str) -> int:
    details = getattr(usage, "details", None)
    if not isinstance(details, dict):
        return 0
    return _safe_int(details.get(key))


def _usage_attr_int(usage: Any, key: str) -> int:
    return _safe_int(getattr(usage, key, 0))


def _response_input_tokens(msg: ModelResponse) -> int:
    usage = getattr(msg, "usage", None)
    if usage is None:
        return 0
    return _usage_attr_int(usage, "input_tokens")


def _response_cache_read_tokens(usage: Any) -> int:
    cache_read = _usage_attr_int(usage, "cache_read_tokens")
    if cache_read > 0:
        return cache_read
    return _usage_detail_int(usage, "prompt_cache_hit_tokens")


def _response_visible_output_tokens(usage: Any) -> int:
    output_tokens = _usage_attr_int(usage, "output_tokens")
    reasoning_tokens = _usage_detail_int(usage, "reasoning_tokens")
    return max(0, output_tokens - reasoning_tokens)


def _response_prompt_delta(
    msg: ModelResponse,
    previous_response_input_tokens: int | None,
) -> int:
    usage = getattr(msg, "usage", None)
    if usage is None:
        return 0
    input_tokens = _response_input_tokens(msg)
    if input_tokens <= 0:
        return 0
    cache_read_tokens = _response_cache_read_tokens(usage)
    if cache_read_tokens > 0:
        return max(0, input_tokens - cache_read_tokens)
    if previous_response_input_tokens is not None:
        return max(0, input_tokens - previous_response_input_tokens)
    return 0


def _estimate_text_tokens(text: str) -> int:
    stripped = (text or "").strip()
    if not stripped:
        return 0
    ascii_words = len(re.findall(r"[A-Za-z0-9_]+", stripped))
    cjk_chars = len(re.findall(r"[\u4e00-\u9fff]", stripped))
    non_space_chars = sum(1 for ch in stripped if not ch.isspace())
    char_floor = (non_space_chars + 3) // 4
    return max(1, ascii_words + cjk_chars, char_floor)


def _turn_token_reference_and_next_input(
    messages: list,
    previous_response_input_tokens: int | None,
    text: str,
) -> tuple[int, int | None]:
    usage_reference = 0
    last_input = previous_response_input_tokens
    for msg in messages:
        if not isinstance(msg, ModelResponse):
            continue
        usage = getattr(msg, "usage", None)
        usage_reference += _response_prompt_delta(msg, last_input)
        if usage is not None:
            usage_reference += _response_visible_output_tokens(usage)
        input_tokens = _response_input_tokens(msg)
        if input_tokens > 0:
            last_input = input_tokens
    return max(usage_reference, _estimate_text_tokens(text)), last_input


def turn_token_reference(messages: list) -> int:
    """Return a current-turn token reference without counting full history."""
    text = pydantic_messages_to_text(messages)
    ref, _ = _turn_token_reference_and_next_input(messages, None, text)
    return ref


def user_prompts_to_text(messages: list) -> str:
    out: list[str] = []
    for msg in messages:
        if not isinstance(msg, ModelRequest):
            continue
        for part in msg.parts:
            if not isinstance(part, UserPromptPart):
                continue
            c = part.content
            items = [c] if isinstance(c, str) else c if isinstance(c, (list, tuple)) else []
            out += [s.strip() for s in items if isinstance(s, str) and s.strip()]
    return "\n\n".join(out)


def message_has_user_prompt(msg: Any) -> bool:
    return isinstance(msg, ModelRequest) and any(
        isinstance(p, UserPromptPart) for p in msg.parts
    )


def turn_has_agent_content(messages: list) -> bool:
    for msg in messages:
        if isinstance(msg, ModelResponse):
            for part in msg.parts:
                if isinstance(part, (TextPart, ToolCallPart)):
                    return True
    return False


def split_messages_into_turns(messages: list) -> list[list]:
    """Split complete user turns by UserPromptPart boundaries."""
    if not messages:
        return []
    turns: list[list] = []
    i = 0
    n = len(messages)
    while i < n:
        if not message_has_user_prompt(messages[i]):
            i += 1
            continue
        start = i
        i += 1
        while i < n and not message_has_user_prompt(messages[i]):
            i += 1
        chunk = messages[start:i]
        if turn_has_agent_content(chunk):
            turns.append(chunk)
    return turns


def turn_entries_from_messages(messages: list) -> list[TurnMemoryEntry]:
    entries: list[TurnMemoryEntry] = []
    previous_response_input_tokens: int | None = None
    current_turn: list | None = None

    def finish_turn(turn: list) -> None:
        nonlocal previous_response_input_tokens
        text = pydantic_messages_to_text(turn)
        token_reference, previous_response_input_tokens = (
            _turn_token_reference_and_next_input(
                turn, previous_response_input_tokens, text
            )
        )
        if turn_has_agent_content(turn):
            entries.append(TurnMemoryEntry(text=text, token_reference=token_reference))

    for msg in messages:
        if message_has_user_prompt(msg):
            if current_turn is not None:
                finish_turn(current_turn)
            current_turn = [msg]
            continue
        if current_turn is not None:
            current_turn.append(msg)
            continue
        if isinstance(msg, ModelResponse):
            input_tokens = _response_input_tokens(msg)
            if input_tokens > 0:
                previous_response_input_tokens = input_tokens

    if current_turn is not None:
        finish_turn(current_turn)
    return entries


def turn_texts_from_messages(messages: list) -> list[str]:
    return [entry.text for entry in turn_entries_from_messages(messages)]
