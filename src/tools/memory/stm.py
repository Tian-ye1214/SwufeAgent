from __future__ import annotations

import asyncio
import json
import threading
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import logger
from app_config import get_env
from persist_utils import load_json_state, rel_key, save_locked_json
from RAG.RAG import RAG as RAGEngineCls
from tools.conversation_log import read_saved_model_messages_file
from tools.memory.message_text import turn_texts_from_messages


def _stm_rag_from_config(cfg: dict[str, Any]) -> Any:
    index_raw = cfg.get("index")
    index_cfg = dict(index_raw) if isinstance(index_raw, dict) else None
    vector_dim = int(get_env("RAG_EMBED_DIM", default="1024", warn=False) or "1024")
    return RAGEngineCls(
        db_path=str(cfg["db_path"]),
        table_name=str(cfg["table_name"]),
        chunk_size=0,
        overlap=0,
        vector_search_limit=int(cfg["vector_search_limit"]),
        final_top_k=int(cfg["final_top_k"]),
        vector_dim=vector_dim,
        use_rerank=bool(cfg["use_rerank"]),
        extended_schema=True,
        index_config=index_cfg,
    )


class _AsyncThreadLock:
    """跨主事件循环与 STM 入库线程的互斥锁。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    async def __aenter__(self) -> _AsyncThreadLock:
        await asyncio.to_thread(self._lock.acquire)
        return self

    async def __aexit__(self, *args: Any) -> None:
        self._lock.release()

    def __enter__(self) -> _AsyncThreadLock:
        self._lock.acquire()
        return self

    def __exit__(self, *args: Any) -> None:
        self._lock.release()


class ShortTermMemory:
    """短期记忆：每轮 user→agent（含工具）单向量；回合结束后后台线程异步入库，供 Worker 检索。"""

    def __init__(
        self,
        stm_config: dict[str, Any],
        rag: Any | None = None,
        log_root: Path | None = None,
    ):
        self._cfg = dict(stm_config)
        self._rag = rag
        self._log_root = log_root
        self._lock = _AsyncThreadLock()
        self._verbose_ingest = bool(self._cfg.get("verbose_ingest", False))
        self._pending_tasks: set[asyncio.Task[None]] = set()
        self._reconcile_on_query = bool(self._cfg["reconcile_on_query"])

    def _get_rag(self) -> Any:
        if self._rag is None:
            self._rag = _stm_rag_from_config(self._cfg)
        return self._rag

    def _ingest_log_context(self):
        if self._verbose_ingest:
            return nullcontext()
        return logger.stm_ingest_console_quiet()

    def _log_root_resolved(self) -> Path:
        return (self._log_root if self._log_root is not None else logger.LOG_DIR).resolve()

    def _db_dir(self) -> Path:
        p = Path(str(self._cfg["db_path"]))
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        return p

    def _stm_state_path(self) -> Path:
        return self._db_dir() / "conversation_stm_state.json"

    def _failures_path(self) -> Path:
        return self._db_dir() / "conversation_stm_failures.jsonl"

    def _rel_log_key(self, path: Path, root: Path) -> str:
        return rel_key(path, root)

    def _load_stm_state_sync(self) -> dict[str, Any]:
        return load_json_state(self._stm_state_path(), 2)

    def _save_stm_state_sync(self, state: dict[str, Any]) -> None:
        save_locked_json(self._stm_state_path(), state)

    def _turns_done_for_key(self, sources: dict[str, Any], log_key: str) -> int:
        ent = sources.get(log_key)
        if not isinstance(ent, dict):
            return 0
        try:
            done = int(ent.get("turns_done", ent.get("chunks_done", 0)))
        except (TypeError, ValueError):
            return 0
        return max(0, done)

    def _append_failure_sync(self, source: str, error: str) -> None:
        p = self._failures_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(
            {
                "source": source,
                "error": error,
                "at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            },
            ensure_ascii=False,
        )
        with p.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def _collect_pending_turn_rows(
        self,
        log_key: str,
        turn_texts: list[str],
        turns_done: int,
        agent: str,
        session_key: str,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        for i in range(turns_done, len(turn_texts)):
            text = turn_texts[i].strip()
            if not text:
                continue
            source = f"{log_key}#t{i}"
            rows.append(
                {
                    "id": source,
                    "source": source,
                    "text": text,
                    "agent": agent,
                    "session_key": session_key,
                    "created_at": now,
                }
            )
        return rows

    async def _ingest_rows_unlocked(
        self,
        log_key: str,
        rows: list[dict[str, Any]],
        new_turns_done: int,
        rag: Any,
    ) -> None:
        if not rows:
            return
        await rag.connect()
        try:
            await rag.ingest_turn_rows(rows)
        except Exception as e:
            for r in rows:
                await asyncio.to_thread(
                    self._append_failure_sync, r.get("source", log_key), str(e)
                )
            raise
        state = await asyncio.to_thread(self._load_stm_state_sync)
        sources = state.setdefault("sources", {})
        sources[log_key] = {"turns_done": new_turns_done}
        await asyncio.to_thread(self._save_stm_state_sync, state)
        if self._verbose_ingest:
            logger.info(
                "STM ingest: log_key=%s turns=%d new_rows=%d",
                log_key,
                new_turns_done,
                len(rows),
            )

    async def ingest_after_turn(
        self,
        messages: list,
        log_key: str,
        agent: str,
        session_key: str,
    ) -> None:
        """回合结束后异步入库自上次 turns_done 起的新轮。"""
        if not log_key or not messages:
            return
        with self._ingest_log_context():
            async with self._lock:
                turn_texts = turn_texts_from_messages(messages)
                state = await asyncio.to_thread(self._load_stm_state_sync)
                sources = state.setdefault("sources", {})
                done = self._turns_done_for_key(sources, log_key)
                if len(turn_texts) < done:
                    done = 0
                rows = self._collect_pending_turn_rows(
                    log_key, turn_texts, done, agent, session_key
                )
                if not rows:
                    return
                rag = _stm_rag_from_config(self._cfg)
                try:
                    await self._ingest_rows_unlocked(
                        log_key, rows, len(turn_texts), rag
                    )
                finally:
                    await rag.close()

    async def _ingest_background(
        self,
        messages: list,
        log_key: str,
        agent: str,
        session_key: str,
    ) -> None:
        try:
            await self.ingest_after_turn(messages, log_key, agent, session_key)
        except Exception as e:
            logger.warning("STM ingest 后台失败: %s", e)

    def schedule_ingest_after_turn(
        self,
        messages: list,
        log_key: str,
        agent: str,
        session_key: str,
    ) -> None:
        if not log_key or not messages:
            return
        msgs = list(messages)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.warning("STM ingest 后台失败: no running event loop")
            return
        task = loop.create_task(
            self._ingest_background(msgs, log_key, agent, session_key),
            name="stm-ingest",
        )
        self._pending_tasks = {t for t in self._pending_tasks if not t.done()}
        self._pending_tasks.add(task)
        task.add_done_callback(self._pending_tasks.discard)

    async def drain(self, *, timeout: float = 60.0) -> None:
        tasks = [t for t in self._pending_tasks if not t.done()]
        if not tasks:
            return
        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            logger.warning("STM ingest task 未在 %.0fs 内结束，继续 shutdown", timeout)

    async def close(self) -> None:
        if self._rag is not None:
            await self._rag.close()

    def _sync_reconcile_file(
        self,
        fp: Path,
        root: Path,
        state: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], str, int]:
        sources = state.setdefault("sources", {})
        log_key = self._rel_log_key(fp, root)
        messages, meta = read_saved_model_messages_file(fp)
        turn_texts = turn_texts_from_messages(messages)
        done = self._turns_done_for_key(sources, log_key)
        if len(turn_texts) < done:
            done = 0
        agent = str(meta.get("agent", "") or "")
        if not agent and "conversations/" in log_key:
            parts = log_key.split("/")
            if len(parts) > 1:
                agent = parts[1]
        date = str(meta.get("date", "") or "")
        topic = str(meta.get("topic", "") or "")
        session_key = f"{date}/{topic}" if date and topic else ""
        rows = self._collect_pending_turn_rows(
            log_key, turn_texts, done, agent, session_key
        )
        return rows, log_key, len(turn_texts)

    async def _reconcile_from_logs(self) -> None:
        root = self._log_root_resolved()
        if not root.is_dir():
            return
        conv = root / "conversations"
        if not conv.is_dir():
            return

        def _scan() -> tuple[dict[str, Any], list[dict[str, Any]]]:
            state = self._load_stm_state_sync()
            all_rows: list[dict[str, Any]] = []
            sources = state.setdefault("sources", {})
            for fp in sorted(conv.rglob("messages_*.model_messages.json"), key=str):
                rows, log_key, new_done = self._sync_reconcile_file(fp, root, state)
                all_rows.extend(rows)
                if rows:
                    sources[log_key] = {"turns_done": new_done}
            return state, all_rows

        final_state, all_rows = await asyncio.to_thread(_scan)
        if not all_rows:
            return
        rag = self._get_rag()
        await rag.connect()
        try:
            await rag.ingest_turn_rows(all_rows)
            await asyncio.to_thread(self._save_stm_state_sync, final_state)
        except Exception as e:
            for r in all_rows:
                await asyncio.to_thread(
                    self._append_failure_sync,
                    r.get("source", ""),
                    str(e),
                )

    async def query_short_term_memory(self, query: str) -> str:
        """
        在历史对话向量索引中检索与 query 相关的片段；结果为模型可用的短期记忆上下文。

        Parameters:
            query: 描述想要查找的详细对话内容
        """
        q = query.strip()
        inner = ""
        if not q:
            return f"<ShortTermMemory>\n{inner}\n</ShortTermMemory>"
        async with self._lock:
            if self._reconcile_on_query:
                await self._reconcile_from_logs()
            rag = self._get_rag()
            await rag.connect()
            hits = await rag.retrieve(q)
            if hits:
                inner = "\n\n".join(
                    f"[{i + 1}] (source: {h.get('source', '')})\n{h.get('text', '')}"
                    for i, h in enumerate(hits)
                )
        return f"<ShortTermMemory>\n{inner}\n</ShortTermMemory>"
