from __future__ import annotations

import os
from typing import Any

import logger
from app_config import get_env
from persist_utils import iso_utc_now
from RAG.DataBase import EmbedDataBase
from RAG.embedding_function import embed_texts, rerank_documents


class RAG:
    def __init__(
        self,
        db_path: str | None,
        table_name: str,
        chunk_size: int,
        overlap: int,
        vector_search_limit: int,
        final_top_k: int,
        vector_dim: int,
        *,
        use_rerank: bool,
        extended_schema: bool = False,
        index_config: dict | None = None,
    ):
        base = db_path or get_env("RAG_DB_PATH", warn=False) or os.path.join(
            os.getcwd(), "data", "rag_lancedb"
        )
        self._db = EmbedDataBase(
            base,
            table_name=table_name,
            vector_dim=vector_dim,
            extended_schema=extended_schema,
            index_config=index_config,
        )
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.vector_search_limit = vector_search_limit
        self.final_top_k = final_top_k
        self._use_rerank = use_rerank

    async def connect(self) -> None:
        await self._db.connect()
        logger.info(
            "RAG: 已连接 LanceDB, path=%s, table=%s, rerank=%s",
            self._db.db_path,
            self._db.table_name,
            self._use_rerank,
        )

    async def close(self) -> None:
        await self._db.close()

    async def row_count(self) -> int:
        return await self._db.row_count()

    async def clear_table(self) -> None:
        await self._db.drop_table()

    def format_instruction(
        self,
        instruction: str | None,
        query: str,
        doc: str | None = None,
        type: str = "embedding",
    ) -> str:
        if instruction is None:
            instruction = "Given a search query, retrieve relevant passages that answer the query"
        if type == "embedding":
            return f"Instruct: {instruction}\nQuery: {query}"
        return "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
            instruction=instruction,
            query=query,
            doc=doc or "",
        )

    async def _embed_query(self, query: str) -> list[float]:
        instruct_text = self.format_instruction(None, query, None, type="embedding")
        vectors = await embed_texts(instruct_text)
        return vectors[0]

    async def ingest_turn_rows(self, rows: list[dict[str, Any]]) -> int:
        """短期记忆：每条含 text、source、id，及可选 agent、session_key、created_at。"""
        if not rows:
            return 0
        await self._db.ensure_connected()
        texts = [r["text"] for r in rows]
        vectors = await embed_texts(texts)
        now = iso_utc_now()
        built: list[dict[str, Any]] = []
        for i, r in enumerate(rows):
            built.append(
                {
                    "vector": vectors[i],
                    "text": texts[i],
                    "source": r.get("source", "") or "",
                    "id": r.get("id", "") or r.get("source", "") or "",
                    "created_at": r.get("created_at", "") or now,
                    "agent": r.get("agent", "") or "",
                    "session_key": r.get("session_key", "") or "",
                }
            )
        n = await self._db.add_vectors(built)
        await self._db.ensure_vector_index()
        logger.info("RAG ingest_turn_rows: rows=%d", n)
        return n

    async def retrieve(self, query: str) -> list[dict[str, Any]]:
        if not query.strip():
            return []

        qprev = (query[:120] + "…") if len(query) > 120 else query
        try:
            query_vector = await self._embed_query(query)
            candidates = await self._db.vector_search(
                query_vector, top_k=self.vector_search_limit
            )
            n_cand = len(candidates)
            logger.info(
                "RAG retrieve: query_len=%d, query_preview=%r, vector_hits=%d, rerank=%s",
                len(query),
                qprev,
                n_cand,
                self._use_rerank,
            )
            if not candidates:
                return []

            if not self._use_rerank:
                cap = min(self.final_top_k, len(candidates))
                out = [
                    {
                        **candidates[i],
                        "relevance_score": 1.0 - (i / max(cap, 1)) * 0.01,
                    }
                    for i in range(cap)
                ]
                logger.info("RAG retrieve: 无 rerank，返回条数=%d", len(out))
                return out

            texts = [c["text"] for c in candidates]
            top_n = min(self.final_top_k, len(texts))
            ranked = await rerank_documents(query, texts, top_n=top_n)

            results: list[dict[str, Any]] = []
            for r in ranked:
                idx = int(r["index"])
                if 0 <= idx < len(candidates):
                    merged = {**candidates[idx], "relevance_score": r["relevance_score"]}
                    results.append(merged)
            logger.info("RAG retrieve: rerank 后返回条数=%d", len(results))
            return results
        except Exception as e:
            logger.error("RAG retrieve 失败: %s", e, exc_info=True)
            raise
