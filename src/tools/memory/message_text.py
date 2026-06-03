from __future__ import annotations

from typing import Any

from pydantic_ai.messages import (
    BaseToolReturnPart,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    UserPromptPart,
)


def pydantic_messages_to_text(messages: list) -> str:
    """将 pydantic-ai 消息对象列表转为可读文本，供长期记忆合并等使用。"""
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
    """按 USER 消息边界切分完整轮次（含工具链）。"""
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


def turn_texts_from_messages(messages: list) -> list[str]:
    return [pydantic_messages_to_text(t) for t in split_messages_into_turns(messages)]


def pack_messages_to_chunk_texts(
    messages: list[Any],
    chunk_max_tokens: int,
    chars_per_token: float,
) -> list[str]:
    from ModelGateway.ModelChecker import estimate_message_tokens

    if not messages:
        return []
    out: list[str] = []
    buf: list[Any] = []
    buf_tokens = 0
    for m in messages:
        t = estimate_message_tokens(m, chars_per_token)
        if buf and buf_tokens + t > chunk_max_tokens:
            out.append(pydantic_messages_to_text(buf))
            buf = []
            buf_tokens = 0
        buf.append(m)
        buf_tokens += t
    if buf:
        out.append(pydantic_messages_to_text(buf))
    return out
