from __future__ import annotations

import asyncio
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic_ai.messages import ModelMessagesTypeAdapter

import logger
from logger import LOG_DIR


def _safe_segment(s: str, max_len: int = 80) -> str:
    s = (s or "").strip().replace("\n", " ")
    return ("".join(c if c.isalnum() or c in ("_", "-", ".") else "_" for c in s)[:max_len]) or "default"


class ConversationLog:
    """对话落盘。save() 写入两份文件：人类可读的 .json 与可加载历史的 .model_messages.json。"""

    def __init__(self, name: str, date: str, topic: str, *, sub_id: str | None = None) -> None:
        self._name = _safe_segment(name, 40)
        self._date = _safe_segment(date, 16)
        self._topic = _safe_segment(topic, 80)
        path = LOG_DIR / "conversations" / self._name / self._date / self._topic
        if sub_id:
            path = path / _safe_segment(sub_id, 60)
        self._dir = path
        self._run_base: Path | None = None
        self._lock = threading.Lock()

    def save(self, model_messages: list[Any], *, extra: dict[str, Any] | None = None) -> None:
        """模型返回后调用；异步落盘，不阻塞事件循环。同一会话多次调用覆盖同一对文件。"""
        with self._lock:
            if self._run_base is None:
                self._dir.mkdir(parents=True, exist_ok=True)
                self._run_base = self._dir / f"messages_{datetime.now().strftime('%H%M%S')}"
        base = self._run_base
        snap = list(model_messages)

        async def _job() -> None:
            try:
                await asyncio.to_thread(self._write, base, snap, extra)
            except Exception as e:
                logger.get_logger().debug("conversation_log 写入失败: %s", e)

        try:
            asyncio.get_running_loop().create_task(_job())
        except RuntimeError:
            pass

    def _write(self, base: Path, model_messages: list[Any], extra: dict[str, Any] | None) -> None:
        raw = ModelMessagesTypeAdapter.dump_python(model_messages, mode="json")
        saved_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        meta: dict[str, Any] = {"agent": self._name, "date": self._date, "topic": self._topic}
        if extra:
            meta.update(extra)
        with base.with_suffix(".json").open("w", encoding="utf-8") as f:
            json.dump({"saved_at": saved_at, "meta": meta, "messages": self._to_readable(raw)}, f, ensure_ascii=False, indent=2)
        with base.with_suffix(".model_messages.json").open("w", encoding="utf-8") as f:
            json.dump({"saved_at": saved_at, "meta": meta, "model_messages": raw}, f, ensure_ascii=False, indent=2)

    def _to_readable(self, raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        for msg in raw:
            parts = msg.get("parts") or []
            if msg.get("kind") == "request":
                for p in parts:
                    if p.get("part_kind") == "user-prompt":
                        messages.append({"role": "user", "content": self._str_content(p.get("content"))})
                    elif p.get("part_kind") == "tool-return":
                        messages.append({"role": "tool", "name": p.get("tool_name"), "tool_call_id": p.get("tool_call_id"), "content": p.get("content")})
            elif msg.get("kind") == "response":
                texts = [p.get("content") or "" for p in parts if p.get("part_kind") == "text"]
                tool_calls = [{"id": p.get("tool_call_id"), "name": p.get("tool_name"), "arguments": p.get("args")} for p in parts if p.get("part_kind") == "tool-call"]
                entry: dict[str, Any] = {"role": "assistant"}
                if text := "\n".join(t for t in texts if t).strip():
                    entry["content"] = text
                if tool_calls:
                    entry["tool_calls"] = tool_calls
                if len(entry) > 1:
                    messages.append(entry)
        return messages

    def _str_content(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks = []
            for item in content:
                if isinstance(item, str):
                    chunks.append(item)
                elif isinstance(item, dict) and item.get("kind") == "binary":
                    chunks.append(f"<binary {item.get('media_type', '?')} base64_len={len(item.get('data', ''))}>")
                else:
                    chunks.append(str(item))
            return "\n".join(chunks)
        return str(content)


class SessionConversationLogs:
    """一次用户任务内各 Agent 的落盘日志；共享同一日期和主题目录，按角色名懒创建 ConversationLog。

    on_activate(date, topic) -- 首次 ensure() 成功时调用
    on_reset()               -- reset() 时调用
    """

    __slots__ = ("_date", "_topic", "_logs", "_on_activate", "_on_reset")

    def __init__(
        self,
        on_activate: Any = None,
        on_reset: Any = None,
    ) -> None:
        self._date: str | None = None
        self._topic: str | None = None
        self._logs: dict[str, ConversationLog] = {}
        self._on_activate = on_activate
        self._on_reset = on_reset

    def ensure(self, topic_hint: str) -> None:
        """绑定本次任务的日期与主题；在 reset() 之前重复调用无效。首次绑定后触发 on_activate。"""
        if self._date is not None:
            return
        self._date = _safe_segment(datetime.now().strftime("%Y%m%d"), 16)
        self._topic = _safe_segment((topic_hint.strip() or "default")[:200], 80)
        if self._on_activate:
            self._on_activate(self._date, self._topic)

    def reset(self) -> None:
        self._date = None
        self._topic = None
        self._logs.clear()
        if self._on_reset:
            self._on_reset()

    def for_agent(self, name: str) -> ConversationLog:
        key = _safe_segment(name, 40)
        if key not in self._logs:
            self._logs[key] = ConversationLog(key, self._date, self._topic)
        return self._logs[key]