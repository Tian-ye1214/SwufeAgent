from __future__ import annotations

import asyncio
import json
import threading
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic_ai.messages import ModelMessagesTypeAdapter

import logger as logger_mod

from logger import LOG_DIR

if TYPE_CHECKING:
    from tools.Memory import ChatHistory


class ConversationLog:
    """对话落盘：可读快照 + 完整 ``model_messages``；调度写入时在后台线程执行。

    同一次「会话」（同一 ``ConversationLog`` 实例、且未调用 ``reset_session``）只占用一对文件
    （``*.json`` / ``*.model_messages.json``）；每轮对话在内存中累积的 ``model_messages``
    会整份写回同一路径，效果上为在同一文件中持续追加对话内容。
    """

    def __init__(self, conversation_dir: Path):
        self._conversation_dir = Path(conversation_dir)
        self._run_base: Path | None = None
        self._session_lock = threading.Lock()
        self._disk_write_lock = threading.Lock()

    def _safe_filename_segment(self, s: str, max_len: int = 40) -> str:
        s = (s or "turn").strip().replace("\n", " ")
        out = "".join(c if c.isalnum() or c in ("_", "-", ".") else "_" for c in s)[:max_len]
        return out or "turn"

    def _stringify_user_content(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks: list[str] = []
            for item in content:
                if isinstance(item, str):
                    chunks.append(item)
                elif isinstance(item, dict) and item.get("kind") == "binary":
                    b64 = item.get("data") or ""
                    mt = item.get("media_type") or "?"
                    chunks.append(f"<binary {mt} base64_len={len(b64)}>")
                else:
                    chunks.append(str(item))
            return "\n".join(chunks)
        return str(content)

    def _stringify_tool_content(self, content: Any) -> str | dict | list | None:
        if content is None:
            return None
        if isinstance(content, (str, int, float, bool)):
            return str(content) if not isinstance(content, str) else content
        if isinstance(content, (dict, list)):
            return content
        return str(content)

    def simple_messages_from_raw(self, raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """由已 dump 的 model_messages 生成可读的 messages（保留 tool / tool_calls）。"""
        simple: list[dict[str, Any]] = []
        for msg in raw:
            kind = msg.get("kind")
            parts = msg.get("parts") or []
            if kind == "request":
                for p in parts:
                    pk = p.get("part_kind")
                    if pk == "user-prompt":
                        simple.append(
                            {"role": "user", "content": self._stringify_user_content(p.get("content"))}
                        )
                    elif pk == "tool-return":
                        simple.append(
                            {
                                "role": "tool",
                                "name": p.get("tool_name"),
                                "tool_call_id": p.get("tool_call_id"),
                                "content": self._stringify_tool_content(p.get("content")),
                            }
                        )
            elif kind == "response":
                text_chunks: list[str] = []
                tool_calls: list[dict[str, Any]] = []
                for p in parts:
                    pk = p.get("part_kind")
                    if pk == "text":
                        text_chunks.append(p.get("content") or "")
                    elif pk == "tool-call":
                        tool_calls.append(
                            {
                                "id": p.get("tool_call_id"),
                                "name": p.get("tool_name"),
                                "arguments": p.get("args"),
                            }
                        )
                entry: dict[str, Any] = {"role": "assistant"}
                text = "\n".join(t for t in text_chunks if t).strip()
                if text:
                    entry["content"] = text
                if tool_calls:
                    entry["tool_calls"] = tool_calls
                if text or tool_calls:
                    simple.append(entry)
        return simple

    def build_conversation_record(
        self,
        model_messages: list[Any],
        *,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """仅生成可读快照（不含 model_messages）。兼容旧调用方。"""
        raw = ModelMessagesTypeAdapter.dump_python(model_messages, mode="json")
        saved_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        record: dict[str, Any] = {
            "saved_at": saved_at,
            "messages": self.simple_messages_from_raw(raw),
        }
        if extra:
            record["meta"] = extra
        return record

    async def abuild_conversation_record(
        self,
        model_messages: list[Any],
        *,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """在线程中执行序列化与摘要，避免在事件循环中阻塞。"""
        return await asyncio.to_thread(
            partial(self.build_conversation_record, model_messages, extra=extra)
        )

    def _build_model_messages_record(
        self,
        raw: list[dict[str, Any]],
        *,
        saved_at: str,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        rec: dict[str, Any] = {"saved_at": saved_at, "model_messages": raw}
        if extra:
            rec["meta"] = extra
        return rec

    def _build_and_write(self, base_path: Path, model_messages: list[Any], extra: dict[str, Any] | None) -> None:
        """base_path 无后缀，例如 .../conversations/foo_20260416104550 → 写入 .json 与 .model_messages.json"""
        raw = ModelMessagesTypeAdapter.dump_python(model_messages, mode="json")
        saved_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        readable: dict[str, Any] = {
            "saved_at": saved_at,
            "messages": self.simple_messages_from_raw(raw),
        }
        if extra:
            readable["meta"] = extra
        full = self._build_model_messages_record(raw, saved_at=saved_at, extra=extra)

        readable_path = base_path.with_suffix(".json")
        raw_path = base_path.with_suffix(".model_messages.json")
        base_path.parent.mkdir(parents=True, exist_ok=True)
        with readable_path.open("w", encoding="utf-8") as f:
            json.dump(readable, f, ensure_ascii=False, indent=2)
        with raw_path.open("w", encoding="utf-8") as f:
            json.dump(full, f, ensure_ascii=False, indent=2)
        log = logger_mod.get_logger()
        log.debug("conversation_log · %s + %s", readable_path.name, raw_path.name)

    def _build_and_write_locked(
        self, base_path: Path, model_messages: list[Any], extra: dict[str, Any] | None
    ) -> None:
        with self._disk_write_lock:
            self._build_and_write(base_path, model_messages, extra)

    def reset_session(self) -> None:
        """下次保存时使用新的文件名（例如用户清空对话、开始新任务时）。"""
        with self._session_lock:
            self._run_base = None

    def _ensure_run_base(self, name_hint: str) -> Path:
        if self._run_base is not None:
            return self._run_base
        with self._session_lock:
            if self._run_base is not None:
                return self._run_base
            ts = datetime.now().strftime("%Y%m%d%H%M%S")
            seg = self._safe_filename_segment(name_hint)
            self._run_base = self._conversation_dir / f"{seg}_{ts}"
            return self._run_base

    def load_chat_history_from_file(self, path: str | Path) -> "ChatHistory":
        """读取含 ``model_messages`` 的快照，或可读 ``*.json`` 时自动查找同 stem 的配对文件。"""
        from tools.Memory import ChatHistory

        p = Path(path)
        with p.open(encoding="utf-8") as f:
            payload = json.load(f)
        raw = payload.get("model_messages")
        if not raw:
            sibling = p.with_suffix(".model_messages.json")
            if sibling.is_file():
                with sibling.open(encoding="utf-8") as f2:
                    payload = json.load(f2)
                raw = payload.get("model_messages")
        if not raw:
            raise ValueError(
                "未找到 model_messages：请传入 *.model_messages.json，"
                "或与可读快照同名的 *.model_messages.json（例如 foo.json 与 foo.model_messages.json）。"
            )
        return ChatHistory.from_model_messages_json(raw)

    async def aload_chat_history_from_file(self, path: str | Path) -> "ChatHistory":
        """在线程中执行读盘与反序列化，避免阻塞事件循环。"""
        return await asyncio.to_thread(self.load_chat_history_from_file, path)

    def schedule_save_conversation_turn(
        self,
        model_messages: list[Any],
        *,
        name_hint: str = "",
        extra: dict[str, Any] | None = None,
    ) -> None:
        """在当前事件循环中排队一次磁盘写入，不阻塞。同会话内多次调用写入同一对文件。"""
        if not model_messages:
            return
        base = self._ensure_run_base(name_hint)
        snap = list(model_messages)

        async def _job() -> None:
            try:
                await asyncio.to_thread(self._build_and_write_locked, base, snap, extra)
            except Exception as e:
                logger_mod.get_logger().debug("conversation_log 写入失败: %s", e)

        try:
            asyncio.get_running_loop().create_task(_job())
        except RuntimeError:
            pass


class ConversationLogFactory:
    """对外入口：默认目录为 ``LOG_DIR / conversations``，也可用 ``create`` 指定目录（例如测试）。"""
    def default(self) -> ConversationLog:
        return ConversationLog(LOG_DIR / "conversations")

    def create(self, conversation_dir: Path | str) -> ConversationLog:
        return ConversationLog(Path(conversation_dir))