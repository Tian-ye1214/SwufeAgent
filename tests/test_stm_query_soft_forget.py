import pytest

from tools.memory.stm import ShortTermMemory


class StubRAG:
    def __init__(self, hits):
        self._hits = hits

    async def connect(self): ...
    async def close(self): ...
    async def retrieve(self, q):
        return list(self._hits)


def _cfg(tmp_path, max_age_days=30):
    return {
        "db_path": str(tmp_path / "stm"), "table_name": "t",
        "vector_search_limit": 10, "final_top_k": 5, "use_rerank": False,
        "min_similarity": 0.0, "reconcile_on_query": False,
        "turn_token_limit": 8192, "turn_chunk_overlap_tokens": 512,
        "index": {"min_rows": 256, "metric": "cosine", "rebuild_every_n_adds": 100},
        "soft_forget": {"enabled": True, "max_age_days": max_age_days,
                        "grace_until_first_use": True},
    }


@pytest.mark.asyncio
async def test_query_filters_forgotten_keeps_order(tmp_path, monkeypatch):
    stm = ShortTermMemory(_cfg(tmp_path))
    hits = [
        {"source": "lk#a", "text": "A", "created_at": "2026-06-20T00:00:00Z"},
        {"source": "lk#b", "text": "B", "created_at": "2026-06-20T00:00:00Z"},
    ]
    stm._rag = StubRAG(hits)
    # 预置 meta：a 很久未访问(应遗忘)，b 最近(保留)；并标记上线时刻早于 created_at
    stm._soft_meta = {
        "lk#a": {"hit": 0, "last_accessed": "2026-01-01T00:00:00Z"},
        "lk#b": {"hit": 0, "last_accessed": "2026-06-25T00:00:00Z"},
    }
    stm._soft_since = "2026-05-01T00:00:00Z"
    out = await stm.query_short_term_memory("q")
    assert "B" in out and "A" not in out


@pytest.mark.asyncio
async def test_query_bumps_hit_in_process(tmp_path):
    stm = ShortTermMemory(_cfg(tmp_path))
    stm._rag = StubRAG([{"source": "lk#x", "text": "X", "created_at": "2026-06-25T00:00:00Z"}])
    stm._soft_since = "2026-05-01T00:00:00Z"
    await stm.query_short_term_memory("q")
    await stm.query_short_term_memory("q")
    assert stm._soft_meta_deltas["lk#x"]["hit"] >= 2
