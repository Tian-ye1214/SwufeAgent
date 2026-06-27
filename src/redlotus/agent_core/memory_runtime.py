from __future__ import annotations

from collections.abc import Callable, Coroutine
from typing import Any

from redlotus.config.app_config import settings
from redlotus.tools.conversation_log import SessionConversationLogs
from redlotus.tools.memory import ChatHistory, LongTermMemory, ShortTermMemory


class MemoryRuntime:
    """Owns long/short-term memory resources for one AgentSystem instance."""

    def __init__(self) -> None:
        self.long_term = LongTermMemory()
        self.long_term.refresh_from_disk_sync()
        self.short_term = ShortTermMemory(settings()["short_term_memory"])
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

    async def long_term_snapshot(self) -> dict[str, dict[str, Any]]:
        return await self.long_term.snapshot()

    async def short_term_snapshot(self) -> dict[str, Any]:
        return await self.short_term.snapshot()

    async def clear_long_term(self) -> None:
        await self.long_term.clear_all()
        self.reset_injection_snapshot()

    async def clear_short_term(self) -> None:
        await self.short_term.clear_index_state()

    def schedule_after_coordinator_turn(
        self,
        history: ChatHistory,
        session_logs: SessionConversationLogs,
        spawn_background: Callable[[Coroutine[Any, Any, Any]], None],
    ) -> None:
        msgs = list(history.messages)
        spawn_background(self.long_term.consolidate_from_messages(msgs, silent=True))

        from redlotus.workspace.workspace import stm_log_key

        coord_log = session_logs.for_agent("coordinator")
        messages_path = coord_log.model_messages_path()
        session_key = session_logs.session_key()
        if messages_path is None or session_key is None:
            return

        log_key = stm_log_key(messages_path)

        self.short_term.schedule_ingest_after_turn(
            msgs, log_key, "coordinator", session_key
        )

    async def close(self) -> None:
        await self.short_term.drain()
        await self.short_term.flush()
        await self.short_term.close()
