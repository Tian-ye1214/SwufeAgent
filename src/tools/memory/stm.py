from __future__ import annotations

import asyncio
import json
import threading
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import logger
from app_config import get_env
from persist_utils import iso_utc_now, load_json_state, rel_key, save_locked_json
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
        self._embed_batch = max(1, int(self._cfg.get("embed_batch_turns", 64)))

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
                "at": iso_utc_now(),
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
        *,
        end: int | None = None,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        now = iso_utc_now()
        stop = len(turn_texts) if end is None else end
        for i in range(turns_done, stop):
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

    async def _ingest_in_batches(
        self,
        log_key: str,
        turn_texts: list[str],
        turns_done: int,
        agent: str,
        session_key: str,
        rag: Any,
        *,
        flush: bool,
    ) -> None:
        """按 embed_batch 切窗入库：每批至多 batch 轮，逐批推进并持久化游标。

        非 flush 时只处理完整批，剩余尾批（< batch）延后；flush 时连尾批一并入库。
        每批落盘游标，因此已嵌入的轮永不重嵌，中途崩溃也不丢已完成批。
        """
        total = len(turn_texts)
        batch = self._embed_batch
        end = total if flush else turns_done + ((total - turns_done) // batch) * batch
        a = turns_done
        while a < end:
            b = min(a + batch, end)
            rows = self._collect_pending_turn_rows(
                log_key, turn_texts, a, agent, session_key, end=b
            )
            if rows:
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
            sources[log_key] = {"turns_done": b}
            await asyncio.to_thread(self._save_stm_state_sync, state)
            if self._verbose_ingest:
                logger.info(
                    "STM ingest: log_key=%s turns=%d new_rows=%d",
                    log_key,
                    b,
                    len(rows),
                )
            a = b

    async def ingest_after_turn(
        self,
        messages: list,
        log_key: str,
        agent: str,
        session_key: str,
    ) -> None:
        """回合结束后异步入库自上次 turns_done 起的新轮（仅完整批，尾批延后）。"""
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
                # 不足一个完整批：延后（不建 RAG，避免每轮连/关抖动）。
                if len(turn_texts) - done < self._embed_batch:
                    return
                rag = _stm_rag_from_config(self._cfg)
                try:
                    await rag.connect()
                    await self._ingest_in_batches(
                        log_key, turn_texts, done, agent, session_key, rag, flush=False
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
    ) -> tuple[list[str], str, int, str, str]:
        """从单个日志文件解析出入库所需上下文（不构造 rows，留给批量入库逐窗构造）。"""
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
        return turn_texts, log_key, done, agent, session_key

    async def _reconcile_from_logs(self, *, flush: bool = False) -> None:
        root = self._log_root_resolved()
        if not root.is_dir():
            return
        conv = root / "conversations"
        if not conv.is_dir():
            return

        files = sorted(conv.rglob("messages_*.model_messages.json"), key=str)
        rag: Any | None = None
        for fp in files:
            state = await asyncio.to_thread(self._load_stm_state_sync)
            turn_texts, log_key, done, agent, session_key = await asyncio.to_thread(
                self._sync_reconcile_file, fp, root, state
            )
            pending = len(turn_texts) - done
            if pending <= 0:
                continue
            # 非 flush 且不足一个完整批：延后尾批。
            if not flush and pending < self._embed_batch:
                continue
            if rag is None:
                rag = self._get_rag()
                await rag.connect()
            await self._ingest_in_batches(
                log_key, turn_texts, done, agent, session_key, rag, flush=flush
            )

    async def _flush_inner(self) -> None:
        async with self._lock:
            await self._reconcile_from_logs(flush=True)

    async def flush(self, *, timeout: float = 30.0) -> None:
        """关闭前尽力收尾：把所有未入库的轮（含不足一批的尾批）批量嵌入并落盘游标。"""
        try:
            await asyncio.wait_for(self._flush_inner(), timeout)
        except asyncio.TimeoutError:
            logger.warning("STM flush 未在 %.0fs 内完成，跳过剩余尾批", timeout)
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
                await self._reconcile_from_logs(flush=False)
            rag = self._get_rag()
            await rag.connect()
            hits = await rag.retrieve(q)
            if hits:
                inner = "\n\n".join(
                    f"[{i + 1}] (source: {h.get('source', '')})\n{h.get('text', '')}"
                    for i, h in enumerate(hits)
                )
        return f"<ShortTermMemory>\n{inner}\n</ShortTermMemory>"
