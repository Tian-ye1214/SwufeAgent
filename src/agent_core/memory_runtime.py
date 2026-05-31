from __future__ import annotations

from collections.abc import Callable, Coroutine
from typing import Any

import logger
from app_config import settings
from tools.conversation_log import SessionConversationLogs
from tools.memory import ChatHistory, LongTermMemory, ShortTermMemory


class MemoryRuntime:
    """Owns long/short-term memory resources for one AgentSystem instance."""

    def __init__(self) -> None:
        self.long_term = LongTermMemory()
        self.long_term.refresh_from_disk_sync()
        self.short_term = ShortTermMemory(
            settings()["short_term_memory"],
            log_root=logger.LOG_DIR,
        )
        self._injection_snapshot: str | None = None

    @property
    def worker_tools(self) -> list:
        return [
            self.short_term.query_short_term_memory,
            self.long_term.add,
            self.long_term.remove,
            self.long_term.list_memory,
        ]

    def injection_for_session(self) -> str:
        if self._injection_snapshot is None:
            self._injection_snapshot = self.long_term.get_injection()
        return self._injection_snapshot

    def reset_injection_snapshot(self) -> None:
        self._injection_snapshot = None

    def schedule_after_coordinator_turn(
        self,
        history: ChatHistory,
        session_logs: SessionConversationLogs,
        spawn_background: Callable[[Coroutine[Any, Any, Any]], None],
    ) -> None:
        msgs = list(history.messages)
        spawn_background(self.long_term.consolidate_from_messages(msgs, silent=True))
        spawn_background(self.long_term.consolidate_from_logs(logger.LOG_DIR, silent=True))

        coord_log = session_logs.for_agent("coordinator")
        messages_path = coord_log.model_messages_path()
        session_key = session_logs.session_key()
        if messages_path is None or session_key is None:
            return

        root = logger.LOG_DIR.resolve()
        try:
            log_key = messages_path.resolve().relative_to(root).as_posix()
        except ValueError:
            log_key = messages_path.resolve().as_posix()

        self.short_term.schedule_ingest_after_turn(
            msgs, log_key, "coordinator", session_key
        )

    async def close(self) -> None:
        await self.short_term.drain()
        await self.short_term.close()
