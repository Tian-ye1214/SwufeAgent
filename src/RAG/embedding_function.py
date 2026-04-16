from __future__ import annotations

import os
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv()

_SILICONFLOW_BASE = os.environ.get("SILICONFLOW_API_BASE", "https://api.siliconflow.cn/v1").rstrip(
    "/"
)
_DEFAULT_EMBED_MODEL = os.environ.get("SILICONFLOW_EMBED_MODEL", "BAAI/bge-m3")
_DEFAULT_RERANK_MODEL = os.environ.get("SILICONFLOW_RERANK_MODEL", "Qwen/Qwen3-Reranker-0.6B")


def _api_key() -> str:
    key = os.environ.get("API_KEY") or os.environ.get("SILICONFLOW_API_KEY")
    if not key:
        raise RuntimeError("未配置 API_KEY 或 SILICONFLOW_API_KEY，无法调用 SiliconFlow。")
    return key


async def embed_texts(
    texts: str | list[str],
    *,
    model: str | None = None,
    timeout: float = 60.0,
) -> list[list[float]]:
    """异步获取文本向量；支持单条字符串或多条批量。"""
    if isinstance(texts, str):
        payload_input: str | list[str] = texts
    else:
        if not texts:
            return []
        payload_input = texts

    url = f"{_SILICONFLOW_BASE}/embeddings"
    body = {"model": model or _DEFAULT_EMBED_MODEL, "input": payload_input}

    async with httpx.AsyncClient(http2=True, timeout=timeout) as client:
        response = await client.post(
            url,
            headers={
                "Authorization": f"Bearer {_api_key()}",
                "Content-Type": "application/json",
            },
            json=body,
        )
        response.raise_for_status()
        data = response.json()

    items = data.get("data") or []
    items = sorted(items, key=lambda x: x.get("index", 0))
    return [row["embedding"] for row in items]


async def embedding(query: str, **kwargs: Any) -> list[float]:
    """兼容旧接口名：单查询 -> 单向量。"""
    vecs = await embed_texts(query, **kwargs)
    return vecs[0]


async def rerank_documents(
    query: str,
    documents: list[str],
    *,
    model: str | None = None,
    top_n: int | None = None,
    instruction: str | None = None,
    timeout: float = 60.0,
) -> list[dict[str, Any]]:
    """调用 SiliconFlow /v1/rerank，返回按相关度排序的结果（含原始下标与分数）。"""
    if not documents:
        return []

    url = f"{_SILICONFLOW_BASE}/rerank"
    body: dict[str, Any] = {
        "model": model or _DEFAULT_RERANK_MODEL,
        "query": query,
        "documents": documents,
        "return_documents": False,
    }
    if top_n is not None:
        body["top_n"] = top_n
    if instruction:
        body["instruction"] = instruction

    async with httpx.AsyncClient(http2=True, timeout=timeout) as client:
        response = await client.post(
            url,
            headers={
                "Authorization": f"Bearer {_api_key()}",
                "Content-Type": "application/json",
            },
            json=body,
        )
        response.raise_for_status()
        data = response.json()

    out: list[dict[str, Any]] = []
    for row in data.get("results", []):
        idx = int(row["index"])
        doc_wrap = row.get("document")
        text = ""
        if isinstance(doc_wrap, dict):
            text = doc_wrap.get("text") or ""
        if not text and 0 <= idx < len(documents):
            text = documents[idx]
        out.append(
            {
                "index": idx,
                "text": text,
                "relevance_score": float(row.get("relevance_score", 0.0)),
            }
        )
    return out


async def reranker(query: str, document: list[str], **kwargs: Any) -> str:
    """兼容旧接口：返回相关性最高的单段文本。"""
    ranked = await rerank_documents(query, document, **kwargs)
    if not ranked:
        return document[0] if document else ""
    return ranked[0]["text"]
