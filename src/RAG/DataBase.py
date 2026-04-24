from __future__ import annotations

import asyncio
import os
from typing import Any

import lancedb
from lancedb.pydantic import LanceModel, Vector


def vector_record_schema(dim: int) -> type[LanceModel]:
    """按向量维度生成 Lance 表 schema（与 BGE-M3 等固定维度模型对齐）。"""

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
        table_name: str = "knowledge_chunks",
        vector_dim: int | None = None,
    ):
        self.db_path = db_path
        self.table_name = table_name
        self.vector_dim = vector_dim or int(os.environ.get("RAG_EMBED_DIM", "1024"))
        self._schema = vector_record_schema(self.vector_dim)
        self._db = None
        self._table = None

    async def connect(self) -> None:
        def _connect() -> None:
            os.makedirs(self.db_path, exist_ok=True)
            self._db = lancedb.connect(self.db_path)
            if self.table_name in self._db.table_names():
                self._table = self._db.open_table(self.table_name)
            else:
                self._table = None

        await asyncio.to_thread(_connect)

    async def ensure_connected(self) -> None:
        if self._db is None:
            await self.connect()

    def _require_table(self):
        if self._db is None:
            raise RuntimeError("数据库未连接，请先调用 await connect()")
        return self._db

    async def add_vectors(self, rows: list[dict[str, Any]]) -> None:
        """rows每项需含 vector, text；可选 source。"""

        if not rows:
            return

        def _add() -> None:
            db = self._require_table()
            pydantic_rows = [
                self._schema(
                    vector=r["vector"],
                    text=r["text"],
                    source=r.get("source", "") or "",
                )
                for r in rows
            ]
            if self._table is None:
                if self.table_name in db.table_names():
                    self._table = db.open_table(self.table_name)
                    self._table.add(pydantic_rows)
                else:
                    self._table = db.create_table(
                        self.table_name,
                        data=pydantic_rows,
                        schema=self._schema,
                    )
            else:
                self._table.add(pydantic_rows)

        await asyncio.to_thread(_add)

    async def vector_search(
        self, query_embedding: list[float], top_k: int = 10
    ) -> list[dict[str, Any]]:
        """向量检索；在 worker 线程内完成查询与结果序列化，避免阻塞事件循环。"""

        await self.ensure_connected()

        def _search() -> list[dict[str, Any]]:
            if self._table is None:
                if self._db is not None and self.table_name in self._db.table_names():
                    self._table = self._db.open_table(self.table_name)
            if self._table is None:
                return []
            df = self._table.search(query_embedding).limit(top_k).to_pandas()
            out: list[dict[str, Any]] = []
            for _, row in df.iterrows():
                dist = row.get("_distance")
                out.append(
                    {
                        "text": row.get("text", ""),
                        "source": row.get("source", "") or "",
                        "_distance": float(dist) if dist is not None and dist == dist else None,
                    }
                )
            return out

        return await asyncio.to_thread(_search)

    async def drop_table(self) -> None:
        await self.ensure_connected()

        def _drop() -> None:
            db = self._require_table()
            if self.table_name in db.table_names():
                db.drop_table(self.table_name)
            self._table = None

        await asyncio.to_thread(_drop)
