from __future__ import annotations

from typing import Any

import httpx
from redlotus.infra import logger
from redlotus.config.app_config import get_env, settings
from redlotus.infra.shared_http import get_client

_HTTP_KEY = "rag"

def _require_rag_model(role: str) -> str:
    m = settings().get("RAG_models")
    if not isinstance(m, dict):
        raise RuntimeError("config.json 中须包含 RAG_models 对象。")
    name = (m.get(role) or "").strip()
    if not name:
        raise RuntimeError(f"config.json 的 RAG_models 中须配置非空的 {role!r}。")
    return name


def _client_kwargs(timeout: float) -> dict[str, Any]:
    base = get_env("SILICONFLOW_BASE", warn=False).strip().rstrip("/")
    return {
        "base_url": base,
        "http2": True,
        "timeout": httpx.Timeout(timeout),
    }


def _get_shared_client() -> httpx.AsyncClient:
    """进程级共享 embedding/rerank 客户端：复用连接池，免去每次请求的 TLS/HTTP2 握手。"""
    return get_client(_HTTP_KEY, lambda: httpx.AsyncClient(**_client_kwargs(60.0)))


async def _rag_api_post(endpoint: str, body: dict[str, Any], *, timeout: float) -> dict[str, Any]:
    """RAG 接口统一 POST：构造鉴权头、校验状态、解析 JSON。"""
    response = await _get_shared_client().post(
        endpoint,
        headers={
            "Authorization": f"Bearer {get_env('SILICONFLOW_KEY', warn=False).strip()}",
            "Content-Type": "application/json",
        },
        json=body,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


_EMBED_MAX_BATCH = 32


async def embed_texts(
    texts: str | list[str],
    *,
    timeout: float = 60.0,
) -> list[list[float]]:
    """异步获取文本向量；支持单条字符串或多条批量，超过上限自动分批请求。"""
    if isinstance(texts, str):
        texts = [texts]
    logger.debug("RAG embed: batch_size=%d", len(texts))
    model = _require_rag_model("embedding")
    vectors: list[list[float]] = []
    for start in range(0, len(texts), _EMBED_MAX_BATCH):
        body = {"model": model, "input": texts[start:start + _EMBED_MAX_BATCH]}
        data = await _rag_api_post("/embeddings", body, timeout=timeout)
        items = data.get("data") or []
        n = len(items)
        if n > 1 and [x.get("index", 0) for x in items] != list(range(n)):
            items = sorted(items, key=lambda x: x.get("index", 0))
        vectors.extend(row["embedding"] for row in items)
    return vectors


async def rerank_documents(
    query: str,
    documents: list[str],
    *,
    top_n: int | None = None,
    timeout: float = 60.0,
) -> list[dict[str, Any]]:
    """调用与 OpenAI 兼容的 /v1/rerank，返回按相关度排序的结果（含原始下标与分数）。"""
    if not documents:
        return []
    logger.debug("RAG rerank: n_docs=%d, top_n=%s", len(documents), top_n)
    model = _require_rag_model("reranker")
    body: dict[str, Any] = {
        "model": model,
        "query": query,
        "documents": documents,
        "return_documents": False,
    }
    if top_n is not None:
        body["top_n"] = top_n

    data = await _rag_api_post("/rerank", body, timeout=timeout)

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
