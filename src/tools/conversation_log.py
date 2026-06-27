from __future__ import annotations

import asyncio
import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic_ai.messages import ModelMessagesTypeAdapter

from config.app_config import settings
from infra.persist_utils import atomic_write_json, iso_utc_now, safe_segment
from infra import logger
from workspace.workspace import (
    MODEL_MESSAGES_SUFFIX,
    conversations_root,
    snapshot_base_from_loadable,
    snapshot_basename,
)

_PENDING_SAVE_TASKS: set[asyncio.Task[None]] = set()


def read_saved_model_messages_file(path: Path) -> tuple[list[Any], dict[str, Any]]:
    """从 `*_ModelMessages.json` 读取 `model_messages`，校验并还原为 pydantic-ai 消息对象。"""
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    raw = data.get("model_messages")
    if not isinstance(raw, list):
        raise ValueError("文件格式无效：缺少 model_messages 数组")
    meta_raw = data.get("meta")
    meta: dict[str, Any] = dict(meta_raw) if isinstance(meta_raw, dict) else {}
    messages = ModelMessagesTypeAdapter.validate_python(raw)
    return messages, meta


def dump_validated_model_messages(model_messages: list[Any]) -> list[dict[str, Any]]:
    raw = ModelMessagesTypeAdapter.dump_python(model_messages, mode="json")
    ModelMessagesTypeAdapter.validate_python(raw)
    return raw


async def drain_pending_saves(timeout: float = 10.0) -> bool:
    tasks = [t for t in list(_PENDING_SAVE_TASKS) if not t.done()]
    if not tasks:
        return True
    try:
        await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=timeout,
        )
        return True
    except asyncio.TimeoutError:
        logger.warning("conversation_log drain timed out after %.1fs", timeout)
        return False


class ConversationLog:
    """对话落盘。save() 写入工作区 .redlotus 下两份文件：可读 .json 与 *_ModelMessages.json。"""

    def __init__(
        self,
        name: str,
        date: str | None,
        topic: str | None,
        *,
        sub_id: str | None = None,
        existing_run_base: Path | None = None,
    ) -> None:
        self._name = safe_segment(name, 40)
        self._date = safe_segment(date or "", 16)
        self._topic = safe_segment(topic or "", 80)
        self._sub_id = safe_segment(sub_id, 60) if sub_id else None
        self._root = conversations_root()
        self._run_base: Path | None = existing_run_base
        self._init_lock = threading.Lock()
        self._write_lock: asyncio.Lock | None = None

    def _write_lock_for_loop(self) -> asyncio.Lock:
        if self._write_lock is None:
            with self._init_lock:
                if self._write_lock is None:
                    self._write_lock = asyncio.Lock()
        return self._write_lock

    def _snapshot_paths(self, base: Path) -> tuple[Path, Path]:
        return (
            base.parent / f"{base.name}.json",
            base.parent / f"{base.name}{MODEL_MESSAGES_SUFFIX}",
        )

    def model_messages_path(self) -> Path | None:
        if self._run_base is None:
            return None
        return self._snapshot_paths(self._run_base)[1]

    def save(self, model_messages: list[Any], *, extra: dict[str, Any] | None = None) -> None:
        """模型返回后调用；异步落盘，不阻塞事件循环。同一会话多次调用覆盖同一对文件。"""
        with self._init_lock:
            if self._run_base is None:
                if not self._date or not self._topic:
                    return
                self._root.mkdir(parents=True, exist_ok=True)
                stem = snapshot_basename(
                    self._name,
                    self._date,
                    self._topic,
                    sub_id=self._sub_id,
                )
                self._run_base = self._root / stem
        base = self._run_base
        if base is None:
            return
        snap = list(model_messages)

        async def _job() -> None:
            try:
                async with self._write_lock_for_loop():
                    await asyncio.to_thread(self._write, base, snap, extra)
            except Exception as e:
                logger.error("conversation_log 写入失败: %s", e)

        try:
            task = asyncio.get_running_loop().create_task(_job())
            _PENDING_SAVE_TASKS.add(task)
            task.add_done_callback(_PENDING_SAVE_TASKS.discard)
        except RuntimeError:
            pass

    def _write(self, base: Path, model_messages: list[Any], extra: dict[str, Any] | None) -> None:
        raw = dump_validated_model_messages(model_messages)
        saved_at = iso_utc_now()
        meta: dict[str, Any] = {"agent": self._name, "date": self._date, "topic": self._topic}
        if self._sub_id:
            if self._name == "worker":
                meta["sub_id"] = self._sub_id
            else:
                meta["session_id"] = self._sub_id
        if extra:
            meta.update(extra)
        readable_path, model_path = self._snapshot_paths(base)
        readable_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            readable_path,
            {"saved_at": saved_at, "meta": meta, "messages": self._to_readable(raw)},
        )
        atomic_write_json(
            model_path,
            {"saved_at": saved_at, "meta": meta, "model_messages": raw},
        )

    def _to_readable(self, raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        include_thinking = self._save_model_thinking_chain_enabled()
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
                thinking_parts = [
                    p.get("content") or ""
                    for p in parts
                    if p.get("part_kind") in ("thinking", "reasoning", "reasoning-content")
                ]
                tool_calls = [{"id": p.get("tool_call_id"), "name": p.get("tool_name"), "arguments": p.get("args")} for p in parts if p.get("part_kind") == "tool-call"]
                entry: dict[str, Any] = {"role": "assistant"}
                if text := "\n".join(t for t in texts if t).strip():
                    entry["content"] = text
                if include_thinking:
                    reasoning_content = msg.get("reasoning_content")
                    chains: list[str] = []
                    if isinstance(reasoning_content, str) and reasoning_content.strip():
                        chains.append(reasoning_content.strip())
                    chains.extend(t.strip() for t in thinking_parts if isinstance(t, str) and t.strip())
                    if chains:
                        entry["thinking_chain"] = "\n\n".join(chains)
                if tool_calls:
                    entry["tool_calls"] = tool_calls
                if len(entry) > 1:
                    messages.append(entry)
        return messages

    def _save_model_thinking_chain_enabled(self) -> bool:
        raw = settings().get("conversation_log")
        if not isinstance(raw, dict):
            return False
        flag = raw.get("save_model_thinking_chain")
        if isinstance(flag, bool):
            return flag
        if isinstance(flag, str):
            return flag.strip().lower() in ("1", "true", "yes", "on")
        return bool(flag)

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

    __slots__ = ("_date", "_topic", "_session_id", "_logs", "_on_activate", "_on_reset")

    def __init__(
        self,
        on_activate: Any = None,
        on_reset: Any = None,
    ) -> None:
        self._date: str | None = None
        self._topic: str | None = None
        self._session_id: str | None = None
        self._logs: dict[str, ConversationLog] = {}
        self._on_activate = on_activate
        self._on_reset = on_reset

    def ensure(self, topic_hint: str) -> None:
        """绑定本次任务的日期与主题；在 reset() 之前重复调用无效。首次绑定后触发 on_activate。"""
        if self._date is not None:
            return
        self._date = safe_segment(datetime.now().strftime("%Y%m%d"), 16)
        self._topic = safe_segment((topic_hint.strip() or "default")[:200], 80)
        self._session_id = safe_segment(datetime.now().strftime("%H%M%S%f"), 16)
        if self._on_activate:
            self._on_activate(self._date, self._topic)

    def reset(self) -> None:
        self._date = None
        self._topic = None
        self._session_id = None
        self._logs.clear()
        if self._on_reset:
            self._on_reset()

    def bind_loaded_snapshot(self, agent_name: str, load_path: Path, meta: dict[str, Any]) -> None:
        """将会话日志绑定到 /load 的原始快照文件，后续保存继续覆盖该文件。"""
        p = Path(load_path)
        base = snapshot_base_from_loadable(p)
        date = safe_segment(str(meta.get("date") or ""), 16)
        topic = safe_segment(str(meta.get("topic") or ""), 80)
        if not date:
            date = safe_segment(datetime.now().strftime("%Y%m%d"), 16)
        if not topic:
            topic = safe_segment(base.name, 80)
        self._date = date
        self._topic = topic
        self._logs.clear()
        key = safe_segment(agent_name, 40)
        instance_raw = meta.get("session_id") or meta.get("sub_id")
        instance_id = safe_segment(str(instance_raw), 60) if instance_raw else None
        self._session_id = instance_id
        self._logs[key] = ConversationLog(
            key,
            self._date,
            self._topic,
            sub_id=instance_id,
            existing_run_base=base,
        )
        if self._on_activate:
            self._on_activate(self._date, self._topic)

    def session_key(self) -> str | None:
        if self._date is not None and self._topic is not None:
            return f"{self._date}/{self._topic}"
        return None

    def for_agent(self, name: str) -> ConversationLog:
        key = safe_segment(name, 40)
        if key not in self._logs:
            self._logs[key] = ConversationLog(
                key,
                self._date,
                self._topic,
                sub_id=self._session_id,
            )
        return self._logs[key]
