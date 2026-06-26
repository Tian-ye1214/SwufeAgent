import pytest

from tools.memory.ltm import LongTermMemory


def _msgs(texts):
    from pydantic_ai.messages import ModelRequest, UserPromptPart
    return [ModelRequest(parts=[UserPromptPart(content=t)]) for t in texts]


@pytest.mark.asyncio
async def test_unchanged_transcript_skips_llm(tmp_path, monkeypatch):
    ltm = LongTermMemory()
    monkeypatch.setattr(type(ltm), "_MEMORY_DIR", tmp_path)
    calls = {"n": 0}

    async def fake_chat(*a, **k):
        calls["n"] += 1
        return {"soul": None, "user": None}

    monkeypatch.setattr(ltm, "_consolidation_chat_and_parse", fake_chat)
    # 模板存在
    (tmp_path / "soul_user_consolidation.md").write_text("TPL", encoding="utf-8")

    msgs = _msgs(["记住我喜欢简洁"])
    await ltm.consolidate_from_messages(msgs, silent=True)
    first = calls["n"]
    # 同一转写再来一次 -> 命中"unchanged"短路，不再调 LLM
    await ltm.consolidate_from_messages(msgs, silent=True)
    assert calls["n"] == first
