from __future__ import annotations

import asyncio
import os
from typing import Any

import logger
from app_config import get_env
from RAG.DataBase import EmbedDataBase
from RAG.embedding_function import embed_texts, rerank_documents


class RAG:
    def __init__(
        self,
        db_path: str | None = None,
        table_name: str = "knowledge_chunks",
        chunk_size: int = 1024,
        overlap: int = 128,
        vector_search_limit: int | None = 50,
        final_top_k: int | None = 10,
        vector_dim: int | None = 1024,
        *,
        use_rerank: bool | None = None,
    ):
        base = db_path or get_env("RAG_DB_PATH", warn=False) or os.path.join(
            os.getcwd(), "data", "rag_lancedb"
        )
        self._db = EmbedDataBase(base, table_name=table_name, vector_dim=vector_dim)
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

    def split_text(
        self,
        text: str,
        chunk_size: int | None = None,
        special_chars: list[str] | None = None,
        overlap: int | None = None,
    ) -> list[str]:
        chunk_size = chunk_size or self.chunk_size
        overlap = overlap if overlap is not None else self.overlap
        if special_chars:
            segments: list[str] = []
            current_pos = 0
            parts: list[str] = []
            for char in special_chars:
                parts = text[current_pos:].split(char)
                for i, part in enumerate(parts[:-1]):
                    if part.strip():
                        segments.append(part.strip())
                    current_pos += len(part) + len(char)
            if parts and parts[-1].strip():
                segments.append(parts[-1].strip())

            chunks = []
            for segment in segments:
                if len(segment) > chunk_size:
                    start = 0
                    while start < len(segment):
                        end = min(start + chunk_size, len(segment))
                        chunks.append(segment[start:end])
                        start = end
                        if overlap > 0 and start < len(segment):
                            start = max(0, start - overlap)
                else:
                    chunks.append(segment)
            return chunks

        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])
            start = end
            if overlap > 0 and start < len(text):
                start = max(0, start - overlap)
        return chunks

    async def split_text_async(
        self,
        text: str,
        chunk_size: int | None = None,
        special_chars: list[str] | None = None,
        overlap: int | None = None,
    ) -> list[str]:
        """长文本分块在 worker 线程执行，避免阻塞事件循环。"""
        return await asyncio.to_thread(
            self.split_text, text, chunk_size, special_chars, overlap
        )

    async def _embed_query(self, query: str) -> list[float]:
        instruct_text = self.format_instruction(None, query, None, type="embedding")
        vectors = await embed_texts(instruct_text)
        return vectors[0]

    async def ingest_text(
        self,
        text: str,
        source: str = "inline",
        *,
        special_chars: list[str] | None = None,
    ) -> int:
        """将长文本分块、取向量并写入库；返回写入块数。"""
        chunks = await self.split_text_async(text, special_chars=special_chars)
        if not chunks:
            return 0
        await self._db.ensure_connected()
        vectors = await embed_texts(chunks)
        rows = [
            {"vector": vectors[i], "text": chunks[i], "source": source}
            for i in range(len(chunks))
        ]
        await self._db.add_vectors(rows)
        n = len(chunks)
        logger.info("RAG ingest: source=%s, chunks=%d", source, n)
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
                out = [
                    {
                        **candidates[i],
                        "relevance_score": 1.0 - (i / max(self.vector_search_limit, 1)) * 0.01,
                    }
                    for i in range(self.vector_search_limit)
                ]
                logger.info("RAG retrieve: 无 rerank，返回条数=%d", len(out))
                return out

            texts = [c["text"] for c in candidates]
            top_n = min(self.final_top_k, len(texts)) if self.final_top_k is not None else len(texts)
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
