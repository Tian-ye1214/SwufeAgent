from __future__ import annotations

import asyncio
import os
import shutil
from pathlib import Path
from typing import Any

import lancedb
from infra import logger
from lancedb.pydantic import LanceModel, Vector
from RAG.storage_path import resolve_lancedb_dir


def vector_record_schema(dim: int, *, extended: bool = False) -> type[LanceModel]:
    """按向量维度生成 Lance 表 schema（与 BGE-M3 等固定维度模型对齐）。"""

    if extended:

        class VectorRecord(LanceModel):
            vector: Vector(dim)
            text: str
            source: str = ""
            id: str = ""
            created_at: str = ""
            agent: str = ""
            session_key: str = ""

        VectorRecord.__name__ = f"VectorRecordExt_{dim}"
        return VectorRecord

    class VectorRecord(LanceModel):
        vector: Vector(dim)
        text: str
        source: str = ""

    VectorRecord.__name__ = f"VectorRecord_{dim}"
    return VectorRecord


class EmbedDataBase:
    """LanceDB 向量存储：对外仅暴露 async API；同步 SDK 全部在 worker 线程中执行。"""

    def __init__(
        self,
        db_path: str,
        table_name: str,
        vector_dim: int,
        *,
        extended_schema: bool = False,
        index_config: dict[str, Any] | None = None,
    ):
        self.db_path = resolve_lancedb_dir(db_path)
        self.table_name = table_name
        self.vector_dim = vector_dim
        self._extended_schema = extended_schema
        self._schema = vector_record_schema(self.vector_dim, extended=extended_schema)
        self._index_config = dict(index_config) if index_config else {}
        self._db = None
        self._table = None
        self._rows_since_index = 0

    def _table_lance_dir(self) -> Path:
        return Path(self.db_path) / f"{self.table_name}.lance"

    def _table_names_sync(self, db) -> list[str]:
        if hasattr(db, "list_tables"):
            raw = db.list_tables()
            tables = getattr(raw, "tables", raw)
            return [str(name) for name in tables]
        return list(db.table_names())

    def _remove_orphan_table_dir_sync(self) -> None:
        d = self._table_lance_dir()
        if d.is_dir():
            shutil.rmtree(d)

    def _open_table_sync(self, db) -> Any | None:
        names = self._table_names_sync(db)
        if self.table_name not in names:
            self._remove_orphan_table_dir_sync()
            return None
        try:
            return db.open_table(self.table_name)
        except Exception:
            try:
                db.drop_table(self.table_name)
            except Exception:
                pass
            self._remove_orphan_table_dir_sync()
            return None

    async def connect(self) -> None:
        def _connect() -> None:
            os.makedirs(self.db_path, exist_ok=True)
            self._db = lancedb.connect(self.db_path)
            self._table = self._open_table_sync(self._db)

        await asyncio.to_thread(_connect)
        logger.debug(
            "RAG DB: connect path=%s, table=%s, has_table=%s",
            self.db_path,
            self.table_name,
            self._table is not None,
        )

    async def ensure_connected(self) -> None:
        if self._db is None:
            await self.connect()

    def _require_db(self):
        if self._db is None:
            raise RuntimeError("数据库未连接，请先调用 await connect()")
        return self._db

    def _ensure_table_sync(self, db) -> Any | None:
        if self._table is not None:
            return self._table
        self._table = self._open_table_sync(db)
        return self._table

    def _row_to_pydantic(self, r: dict[str, Any]):
        if self._extended_schema:
            return self._schema(
                vector=r["vector"],
                text=r["text"],
                source=r.get("source", "") or "",
                id=r.get("id", "") or r.get("source", "") or "",
                created_at=r.get("created_at", "") or "",
                agent=r.get("agent", "") or "",
                session_key=r.get("session_key", "") or "",
            )
        return self._schema(
            vector=r["vector"],
            text=r["text"],
            source=r.get("source", "") or "",
        )

    async def add_vectors(self, rows: list[dict[str, Any]]) -> int:
        """rows 每项需含 vector, text；扩展 schema 时还需 id 等元数据。"""

        if not rows:
            return 0

        def _add() -> int:
            db = self._require_db()
            pydantic_rows = [self._row_to_pydantic(r) for r in rows]
            if self._table is None:
                opened = self._ensure_table_sync(db)
                if opened is not None:
                    self._table.add(pydantic_rows)
                else:
                    self._remove_orphan_table_dir_sync()
                    self._table = db.create_table(
                        self.table_name,
                        data=pydantic_rows,
                        schema=self._schema,
                    )
            else:
                self._table.add(pydantic_rows)
            return len(pydantic_rows)

        n = await asyncio.to_thread(_add)
        self._rows_since_index += n
        logger.debug("RAG DB: add_vectors, rows=%d, table=%s", n, self.table_name)
        return n

    async def row_count(self) -> int:
        await self.ensure_connected()

        def _count() -> int:
            db = self._require_db()
            if self._ensure_table_sync(db) is None:
                return 0
            return int(self._table.count_rows())

        return await asyncio.to_thread(_count)

    async def drop_table(self) -> None:
        """Drop the configured table and clear any orphaned Lance directory."""
        await self.ensure_connected()

        def _drop() -> None:
            db = self._require_db()
            try:
                if self.table_name in self._table_names_sync(db):
                    db.drop_table(self.table_name)
            finally:
                self._table = None
                self._rows_since_index = 0
                self._remove_orphan_table_dir_sync()

        await asyncio.to_thread(_drop)

    async def delete_where(self, where: str) -> None:
        """按 SQL 谓词删除行（如 source LIKE 'x%' ESCAPE '\\'）。表不存在则 no-op。"""
        await self.ensure_connected()

        def _del() -> None:
            db = self._require_db()
            if self._ensure_table_sync(db) is None:
                return
            self._table.delete(where)

        await asyncio.to_thread(_del)
        logger.debug("RAG DB: delete_where table=%s where=%s", self.table_name, where)

    async def ensure_vector_index(self) -> bool:
        """达阈值后创建 IVF_PQ 索引；返回是否触发了建索引。"""
        if not self._index_config:
            return False
        await self.ensure_connected()
        min_rows = int(self._index_config["min_rows"])
        rebuild_every = int(self._index_config["rebuild_every_n_adds"])
        count = await self.row_count()
        if count < min_rows:
            return False
        has_index = await asyncio.to_thread(self._has_vector_index_sync)
        if has_index and self._rows_since_index < rebuild_every:
            return False

        metric = str(self._index_config["metric"])
        num_partitions = max(1, count // 4096)
        num_sub_vectors = max(1, self.vector_dim // 8)

        def _build() -> None:
            db = self._require_db()
            if self._ensure_table_sync(db) is None:
                return
            self._table.create_index(
                metric=metric,
                vector_column_name="vector",
                index_type="IVF_PQ",
                num_partitions=num_partitions,
                num_sub_vectors=num_sub_vectors,
                replace=True,
            )

        await asyncio.to_thread(_build)
        self._rows_since_index = 0
        logger.info(
            "RAG DB: vector index built table=%s rows=%d metric=%s partitions=%d",
            self.table_name,
            count,
            metric,
            num_partitions,
        )
        return True

    def _has_vector_index_sync(self) -> bool:
        db = self._require_db()
        if self._ensure_table_sync(db) is None:
            return False
        try:
            indices = self._table.list_indices()
            return len(indices) > 0
        except Exception:
            return False

    async def vector_search(
        self, query_embedding: list[float], top_k: int
    ) -> list[dict[str, Any]]:
        """向量检索；在 worker 线程内完成查询与结果序列化，避免阻塞事件循环。"""

        await self.ensure_connected()
        await self.ensure_vector_index()
        metric = str((self._index_config or {}).get("metric") or "cosine")

        def _search() -> list[dict[str, Any]]:
            db = self._require_db()
            if self._ensure_table_sync(db) is None:
                return []
            df = self._table.search(query_embedding).metric(metric).limit(top_k).to_pandas()
            out: list[dict[str, Any]] = []
            for _, row in df.iterrows():
                dist = row.get("_distance")
                out.append(
                    {
                        "text": row.get("text", ""),
                        "source": row.get("source", "") or "",
                        "created_at": row.get("created_at", "") or "",
                        "_distance": float(dist) if dist is not None and dist == dist else None,
                    }
                )
            return out

        out = await asyncio.to_thread(_search)
        logger.debug("RAG DB: vector_search top_k=%s, results=%d", top_k, len(out))
        return out

    async def close(self) -> None:
        def _close() -> None:
            self._table = None
            self._db = None

        await asyncio.to_thread(_close)
