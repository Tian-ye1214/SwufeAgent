from __future__ import annotations

import asyncio
import hashlib
import json
import math
import threading
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from infra import logger
from config.app_config import get_env
from infra.persist_utils import iso_utc_now, load_json_state, save_locked_json
from RAG.RAG import RAG as RAGEngineCls
from tools.conversation_log import read_saved_model_messages_file
from workspace.workspace import MODEL_MESSAGES_GLOB, conversations_root, stm_log_key
from tools.memory.message_text import TurnMemoryEntry, turn_entries_from_messages


def _turn_hash(text: str) -> str:
    """turn 内容指纹：跨压缩（历史被重排/截断）稳定标识一个 turn，替代易错的位置游标。"""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


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
        min_similarity=float(cfg.get("min_similarity", 0.0) or 0.0),
        extended_schema=True,
        index_config=index_cfg,
    )


class _AsyncThreadLock:
    """跨主事件循环与 STM 入库线程的互斥锁。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    async def __aenter__(self) -> _AsyncThreadLock:
        while not self._lock.acquire(blocking=False):
            await asyncio.sleep(0.01)
        return self

    async def __aexit__(self, *args: Any) -> None:
        self._lock.release()


class ShortTermMemory:
    """Short-term memory for complete user turns, split into chunk rows when needed."""

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
        self._turn_token_limit = max(1, int(self._cfg.get("turn_token_limit", 8192)))
        overlap = max(0, int(self._cfg.get("turn_chunk_overlap_tokens", 512)))
        self._turn_chunk_overlap = min(overlap, max(0, self._turn_token_limit - 1))

    def _get_rag(self) -> Any:
        if self._rag is None:
            self._rag = _stm_rag_from_config(self._cfg)
        return self._rag

    def _ingest_log_context(self):
        if self._verbose_ingest:
            return nullcontext()
        return logger.stm_ingest_console_quiet()

    def _log_root_resolved(self) -> Path:
        return conversations_root().resolve()

    def _stm_source_key(self, path: Path) -> str:
        return stm_log_key(path)

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
        return self._stm_source_key(path)

    def _load_stm_state_sync(self) -> dict[str, Any]:
        return load_json_state(self._stm_state_path(), 2)

    def _save_stm_state_sync(self, state: dict[str, Any]) -> None:
        save_locked_json(self._stm_state_path(), state)

    def _ingested_for_key(
        self,
        sources: dict[str, Any],
        log_key: str,
        turn_entries: list[TurnMemoryEntry],
    ) -> set[str]:
        """已入库的 turn 指纹集合；兼容旧版按位置计数的 turns_done。"""
        ent = sources.get(log_key)
        if not isinstance(ent, dict):
            return set()
        raw = ent.get("ingested")
        if isinstance(raw, list):
            return {str(h) for h in raw}
        try:
            legacy = int(ent.get("turns_done", ent.get("chunks_done", 0)))
        except (TypeError, ValueError):
            legacy = 0
        return {_turn_hash(e.text) for e in turn_entries[: max(0, legacy)]}

    def _append_failure_sync(self, source: str, error: str) -> None:
        p = self._failures_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(
            {
                "source": source,
                "error": error,
                "at": iso_utc_now(),
            },
            ensure_ascii=False,
        )
        with p.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def _snap_chunk_boundary(self, text: str, pos: int) -> int:
        if pos <= 0 or pos >= len(text):
            return max(0, min(len(text), pos))
        radius = max(20, min(240, len(text) // 80))
        lo = max(1, pos - radius)
        hi = min(len(text) - 1, pos + radius)
        for sep in ("\n\n", "\n", " "):
            best = -1
            best_dist = radius + 1
            idx = text.find(sep, lo, hi)
            while idx != -1:
                boundary = idx + len(sep)
                dist = abs(boundary - pos)
                if dist < best_dist:
                    best = boundary
                    best_dist = dist
                idx = text.find(sep, idx + 1, hi)
            if best != -1:
                return best
        return pos

    def _split_turn_text(self, text: str, token_reference: int) -> list[str]:
        text = text.strip()
        if not text:
            return []
        ref = max(0, int(token_reference or 0))
        limit = self._turn_token_limit
        overlap = self._turn_chunk_overlap
        if ref <= limit or ref <= 0:
            return [text]

        stride = max(1, limit - overlap)
        chunk_count = max(1, math.ceil((ref - overlap) / stride))
        if chunk_count <= 1:
            return [text]

        side_overlap = overlap / 2
        text_len = len(text)

        def token_to_char(token_pos: float) -> int:
            return max(0, min(text_len, round((token_pos / ref) * text_len)))

        chunks: list[str] = []
        for chunk_index in range(chunk_count):
            core_start = (ref * chunk_index) / chunk_count
            core_end = (ref * (chunk_index + 1)) / chunk_count
            start_token = 0 if chunk_index == 0 else max(0, core_start - side_overlap)
            end_token = ref if chunk_index == chunk_count - 1 else min(ref, core_end + side_overlap)
            raw_start = token_to_char(start_token)
            raw_end = token_to_char(end_token)
            start = self._snap_chunk_boundary(text, raw_start)
            end = self._snap_chunk_boundary(text, raw_end)
            if start >= end:
                start, end = raw_start, raw_end
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
        return chunks or [text]

    def _collect_turn_rows(
        self,
        log_key: str,
        entry: TurnMemoryEntry,
        turn_hash: str,
        agent: str,
        session_key: str,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        now = iso_utc_now()
        chunks = self._split_turn_text(entry.text, entry.token_reference)
        for chunk_index, chunk in enumerate(chunks):
            source = (
                f"{log_key}#{turn_hash}"
                if len(chunks) == 1
                else f"{log_key}#{turn_hash}#c{chunk_index}"
            )
            rows.append(
                {
                    "id": source,
                    "source": source,
                    "text": chunk,
                    "agent": agent,
                    "session_key": session_key,
                    "created_at": now,
                }
            )
        return rows

    async def _ingest_pending_turns(
        self,
        log_key: str,
        turn_entries: list[TurnMemoryEntry],
        ingested: set[str],
        agent: str,
        session_key: str,
        rag: Any,
    ) -> None:
        """Ingest turns whose content fingerprint isn't recorded yet; persist per turn.

        指纹集每次入库都裁剪到当前存活的 turn（压缩丢弃的 turn 不会再出现），
        既避免重复入库，也避免压缩后新增 turn 被位置游标漏记。
        """
        live = {_turn_hash(e.text) for e in turn_entries}
        done = set(ingested) & live
        for entry in turn_entries:
            h = _turn_hash(entry.text)
            if h in done:
                continue
            rows = self._collect_turn_rows(log_key, entry, h, agent, session_key)
            if rows:
                try:
                    await rag.ingest_turn_rows(rows)
                except Exception as e:
                    for r in rows:
                        await asyncio.to_thread(
                            self._append_failure_sync, r.get("source", log_key), str(e)
                        )
                    raise
            done.add(h)
            state = await asyncio.to_thread(self._load_stm_state_sync)
            sources = state.setdefault("sources", {})
            sources[log_key] = {"ingested": sorted(done)}
            await asyncio.to_thread(self._save_stm_state_sync, state)
            if self._verbose_ingest:
                logger.info(
                    "STM ingest: log_key=%s turns=%d new_rows=%d",
                    log_key,
                    len(done),
                    len(rows),
                )

    async def ingest_after_turn(
        self,
        messages: list,
        log_key: str,
        agent: str,
        session_key: str,
    ) -> None:
        """Ingest complete user turns saved since the last STM cursor."""
        if not log_key or not messages:
            return
        with self._ingest_log_context():
            async with self._lock:
                turn_entries = turn_entries_from_messages(messages)
                state = await asyncio.to_thread(self._load_stm_state_sync)
                sources = state.setdefault("sources", {})
                ingested = self._ingested_for_key(sources, log_key, turn_entries)
                if all(_turn_hash(e.text) in ingested for e in turn_entries):
                    return
                rag = _stm_rag_from_config(self._cfg)
                try:
                    await rag.connect()
                    await self._ingest_pending_turns(
                        log_key, turn_entries, ingested, agent, session_key, rag
                    )
                finally:
                    await rag.close()

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
        loop = asyncio.get_running_loop()
        task = loop.create_task(
            asyncio.to_thread(
                lambda: asyncio.run(
                    self.ingest_after_turn(msgs, log_key, agent, session_key)
                )
            ),
            name="stm-ingest",
        )
        self._pending_tasks = {t for t in self._pending_tasks if not t.done()}
        self._pending_tasks.add(task)
        task.add_done_callback(self._pending_tasks.discard)

    async def drain(self, *, timeout: float = 60.0) -> None:
        ok = await self.wait_idle(timeout=timeout)
        if not ok:
            logger.warning("STM ingest task 未在 %.0fs 内结束，继续 shutdown", timeout)

    async def wait_idle(self, *, timeout: float = 60.0) -> bool:
        tasks = [t for t in self._pending_tasks if not t.done()]
        if not tasks:
            return True
        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=timeout,
            )
            return True
        except asyncio.TimeoutError:
            return False

    async def close(self) -> None:
        if self._rag is not None:
            await self._rag.close()

    async def snapshot(self) -> dict[str, Any]:
        rag = self._get_rag()
        await rag.connect()
        row_count = await rag.row_count()
        rag_db = getattr(getattr(rag, "_db", None), "db_path", None)
        db_path = Path(str(getattr(rag, "db_path", rag_db or self._db_dir())))
        state_path = self._stm_state_path()
        failures_path = self._failures_path()
        return {
            "db_path": db_path,
            "table_name": str(self._cfg["table_name"]),
            "row_count": row_count,
            "state_path": state_path,
            "state_exists": state_path.exists(),
            "failures_path": failures_path,
            "failures_exists": failures_path.exists(),
            "use_rerank": bool(self._cfg["use_rerank"]),
            "reconcile_on_query": bool(self._cfg["reconcile_on_query"]),
        }

    async def clear_index_state(self) -> None:
        async with self._lock:
            state = await asyncio.to_thread(self._current_log_sources_state_sync)
            rag = self._get_rag()
            try:
                await rag.connect()
                await rag.clear_table()
            finally:
                await rag.close()
                self._rag = None
            await asyncio.to_thread(self._save_stm_state_sync, state)

    def _current_log_sources_state_sync(self) -> dict[str, Any]:
        root = self._log_root_resolved()
        sources: dict[str, Any] = {}
        if root.is_dir():
            for fp in sorted(root.glob(MODEL_MESSAGES_GLOB), key=str):
                messages, _ = read_saved_model_messages_file(fp)
                entries = turn_entries_from_messages(messages)
                sources[self._stm_source_key(fp)] = {
                    "ingested": sorted(_turn_hash(e.text) for e in entries)
                }
        return {"version": 2, "sources": sources}

    def _sync_reconcile_file(
        self,
        fp: Path,
        root: Path,
        state: dict[str, Any],
    ) -> tuple[list[TurnMemoryEntry], str, set[str], str, str]:
        """Read one saved model_messages log and return STM cursor context."""
        sources = state.setdefault("sources", {})
        log_key = self._rel_log_key(fp, root)
        messages, meta = read_saved_model_messages_file(fp)
        turn_entries = turn_entries_from_messages(messages)
        ingested = self._ingested_for_key(sources, log_key, turn_entries)
        agent = str(meta.get("agent", "") or "")
        if not agent and "conversations/" in log_key:
            parts = log_key.split("/")
            if len(parts) > 1:
                agent = parts[1]
        date = str(meta.get("date", "") or "")
        topic = str(meta.get("topic", "") or "")
        session_key = f"{date}/{topic}" if date and topic else ""
        return turn_entries, log_key, ingested, agent, session_key

    async def _reconcile_from_logs(self, *, flush: bool = False) -> None:
        root = self._log_root_resolved()
        if not root.is_dir():
            return

        files = sorted(root.glob(MODEL_MESSAGES_GLOB), key=str)
        rag: Any | None = None
        for fp in files:
            state = await asyncio.to_thread(self._load_stm_state_sync)
            turn_entries, log_key, ingested, agent, session_key = await asyncio.to_thread(
                self._sync_reconcile_file, fp, root, state
            )
            if all(_turn_hash(e.text) in ingested for e in turn_entries):
                continue
            if rag is None:
                rag = self._get_rag()
                await rag.connect()
            await self._ingest_pending_turns(
                log_key, turn_entries, ingested, agent, session_key, rag
            )

    async def _flush_inner(self) -> None:
        async with self._lock:
            await self._reconcile_from_logs(flush=True)

    async def flush(self, *, timeout: float = 30.0) -> None:
        """Best-effort shutdown flush for saved complete user turns."""
        try:
            await asyncio.wait_for(self._flush_inner(), timeout)
        except asyncio.TimeoutError:
            logger.warning("STM flush 未在 %.0fs 内完成，跳过剩余 user turn", timeout)
        except Exception as e:
            logger.warning("STM flush 失败: %s", e)

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
                try:
                    await self._reconcile_from_logs(flush=False)
                except Exception as e:
                    logger.warning("STM on-query reconcile 失败，降级为直接检索: %s", e)
            rag = self._get_rag()
            await rag.connect()
            hits = await rag.retrieve(q)
            if hits:
                inner = "\n\n".join(
                    f"[{i + 1}] (source: {h.get('source', '')})\n{h.get('text', '')}"
                    for i, h in enumerate(hits)
                )
        return f"<ShortTermMemory>\n{inner}\n</ShortTermMemory>"
