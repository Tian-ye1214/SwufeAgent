from __future__ import annotations

import asyncio
import os
from typing import Any

from RAG.DataBase import EmbedDataBase
from RAG.embedding_function import embed_texts, rerank_documents


class RAG:
    """分块、写入 LanceDB、向量检索 + 异步重排序；对外均为 async，CPU/同步 IO 走 to_thread。"""

    def __init__(
        self,
        db_path: str | None = None,
        table_name: str = "knowledge_chunks",
        chunk_size: int = 1024,
        overlap: int = 128,
        vector_search_limit: int = 20,
        final_top_k: int = 5,
        vector_dim: int | None = None,
    ):
        base = db_path or os.environ.get("RAG_DB_PATH") or os.path.join(
            os.getcwd(), "data", "rag_lancedb"
        )
        self._db = EmbedDataBase(base, table_name=table_name, vector_dim=vector_dim)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.vector_search_limit = vector_search_limit
        self.final_top_k = final_top_k

    async def connect(self) -> None:
        await self._db.connect()

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
        return len(chunks)

    async def retrieve(
        self,
        query: str,
        *,
        vector_top_k: int | None = None,
        rerank_top_n: int | None = None,
    ) -> list[dict[str, Any]]:
        """向量召回 + 异步 rerank，返回带分数的段落列表。"""
        if not query.strip():
            return []

        top_vec = vector_top_k or self.vector_search_limit
        top_final = rerank_top_n or self.final_top_k

        query_vector = await self._embed_query(query)
        candidates = await self._db.vector_search(query_vector, top_k=top_vec)
        if not candidates:
            return []

        texts = [c["text"] for c in candidates]
        ranked = await rerank_documents(query, texts, top_n=min(top_final, len(texts)))

        results: list[dict[str, Any]] = []
        for r in ranked:
            idx = int(r["index"])
            if 0 <= idx < len(candidates):
                merged = {**candidates[idx], "relevance_score": r["relevance_score"]}
                results.append(merged)
        return results
