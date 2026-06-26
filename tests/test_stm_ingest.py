import json

import pytest

from tools.memory.stm import ShortTermMemory


class FakeRAG:
    def __init__(self):
        self.embed_calls = 0
        self.deleted = []
        self.rows = []

    async def connect(self): ...
    async def close(self): ...

    async def ingest_turn_rows(self, rows):
        self.embed_calls += 1
        self.rows.extend(rows)
        return len(rows)

    async def delete_by_source_prefix(self, prefix):
        self.deleted.append(prefix)


def _cfg(tmp_path):
    return {
        "db_path": str(tmp_path / "stm"),
        "table_name": "t", "vector_search_limit": 10, "final_top_k": 5,
        "use_rerank": False, "min_similarity": 0.0, "reconcile_on_query": False,
        "turn_token_limit": 8192, "turn_chunk_overlap_tokens": 512,
        "index": {"min_rows": 256, "metric": "cosine", "rebuild_every_n_adds": 100},
        "soft_forget": {"enabled": True, "max_age_days": 30, "grace_until_first_use": True},
    }


def _msgs(texts):
    # 构造每条独立 user turn 的最小 pydantic-ai 消息序列（user + assistant 配对，turn_has_agent_content=True）
    from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
    msgs = []
    for t in texts:
        msgs.append(ModelRequest(parts=[UserPromptPart(content=t)]))
        msgs.append(ModelResponse(parts=[TextPart(content="ok")]))
    return msgs


@pytest.mark.asyncio
async def test_reingest_same_turns_no_new_embeddings(tmp_path, monkeypatch):
    stm = ShortTermMemory(_cfg(tmp_path))
    fake = FakeRAG()
    monkeypatch.setattr(stm, "_rag", fake, raising=False)
    monkeypatch.setattr("tools.memory.stm._stm_rag_from_config", lambda cfg: fake)

    msgs = _msgs(["第一条记忆", "第二条记忆"])
    await stm.ingest_after_turn(msgs, "lk", "coordinator", "d/t")
    first = fake.embed_calls
    assert first >= 1
    await stm.ingest_after_turn(msgs, "lk", "coordinator", "d/t")
    assert fake.embed_calls == first  # 重复 ingest 0 新 embedding 请求


@pytest.mark.asyncio
async def test_edited_turn_ingests_new_and_keeps_old(tmp_path, monkeypatch):
    stm = ShortTermMemory(_cfg(tmp_path))
    fake = FakeRAG()
    monkeypatch.setattr("tools.memory.stm._stm_rag_from_config", lambda cfg: fake)
    await stm.ingest_after_turn(_msgs(["原文"]), "lk", "coordinator", "d/t")
    n1 = fake.embed_calls
    await stm.ingest_after_turn(_msgs(["改过的文"]), "lk", "coordinator", "d/t")
    assert fake.embed_calls > n1     # edited turn ingested as a new row
    assert fake.deleted == []        # old row NOT deleted (kept for retrieval)


@pytest.mark.asyncio
async def test_per_turn_ingest_capped(tmp_path, monkeypatch):
    # 每个 turn 单独一次 ingest 调用（不合并成一个巨型请求）
    stm = ShortTermMemory(_cfg(tmp_path))
    fake = FakeRAG()
    monkeypatch.setattr("tools.memory.stm._stm_rag_from_config", lambda cfg: fake)
    await stm.ingest_after_turn(_msgs(["a", "b", "c"]), "lk", "coordinator", "d/t")
    assert fake.embed_calls == 3


@pytest.mark.asyncio
async def test_reconcile_triggers_migration(tmp_path, monkeypatch):
    stm = ShortTermMemory(_cfg(tmp_path))
    # make the reconcile root exist (empty) so _reconcile_from_logs proceeds past the dir guard
    root = tmp_path / "convroot"
    root.mkdir()
    monkeypatch.setattr(stm, "_log_root_resolved", lambda: root)
    # pre-write a legacy version-2 state file (pre-upgrade shape)
    sp = stm._stm_state_path()
    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text(json.dumps({"version": 2, "sources": {}}), encoding="utf-8")
    await stm._reconcile_from_logs()
    after = json.loads(sp.read_text(encoding="utf-8"))
    assert after["version"] == stm._STATE_VERSION      # migrated
    assert "soft_forget_since" in after                # migration stamped the boundary
