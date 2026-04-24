from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import httpx

try:
    from app_config import get_config
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from app_config import get_config

_http_state: tuple[httpx.AsyncClient, str] | None = None


def _require_rag_model(role: str) -> str:
    m = get_config().get("RAG_models")
    if not isinstance(m, dict):
        raise RuntimeError("config.json 中须包含 RAG_models 对象。")
    name = (m.get(role) or "").strip()
    if not name:
        raise RuntimeError(f"config.json 的 RAG_models 中须配置非空的 {role!r}。")
    return name


async def _get_http_client() -> httpx.AsyncClient:
    global _http_state
    base = get_config().get("SILICONFLOW_base").strip().rstrip("/")
    if _http_state is not None and _http_state[1] == base:
        return _http_state[0]
    async with asyncio.Lock():
        if _http_state is not None and _http_state[1] == base:
            return _http_state[0]
        if _http_state is not None:
            await _http_state[0].aclose()
        client = httpx.AsyncClient(
            base_url=base,
            http2=True,
            timeout=httpx.Timeout(60.0),
        )
        _http_state = (client, base)
    return _http_state[0]


async def embed_texts(
    texts: str | list[str],
    *,
    timeout: float = 60.0,
) -> list[list[float]]:
    """异步获取文本向量；支持单条字符串或多条批量。"""
    if isinstance(texts, str):
       texts = [texts]

    model = _require_rag_model("embedding")
    body = {"model": model, "input": texts}
    client = await _get_http_client()
    response = await client.post(
        "/embeddings",
        headers={
            "Authorization": f"Bearer {get_config().get('SILICONFLOW_key').strip()}",
            "Content-Type": "application/json",
        },
        json=body,
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()

    items = data.get("data") or []
    n = len(items)
    if n > 1 and [x.get("index", 0) for x in items] != list(range(n)):
        items = sorted(items, key=lambda x: x.get("index", 0))
    return [row["embedding"] for row in items]


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

    model = _require_rag_model("reranker")
    body: dict[str, Any] = {
        "model": model,
        "query": query,
        "documents": documents,
        "return_documents": False,
    }
    if top_n is not None:
        body["top_n"] = top_n

    client = await _get_http_client()
    response = await client.post(
        "/rerank",
        headers={
            "Authorization": f"Bearer {get_config().get('SILICONFLOW_key').strip()}",
            "Content-Type": "application/json",
        },
        json=body,
        timeout=timeout,
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

if __name__ == "__main__":
    async def _demo() -> None:
        text = "Silicon flow embedding online: fast, affordable, and high-quality embedding services. come try it out!"
        embedding = await embed_texts(text)
        print(embedding)

        query = "Apple"
        documents = ["apple", "banana", "fruit", "vegetable"]
        result = await rerank_documents(query, documents)
        print(result)

    asyncio.run(_demo())
