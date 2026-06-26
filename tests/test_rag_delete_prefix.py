from pathlib import Path

import pytest

from RAG.RAG import RAG

DIM = 4


def _row(src):
    return {"vector": [0.1, 0.2, 0.3, 0.4], "text": f"t-{src}", "source": src,
            "id": src, "created_at": "2026-06-01T00:00:00Z", "agent": "", "session_key": ""}


async def _rag(tmp_path: Path):
    r = RAG(db_path=str(tmp_path / "db"), table_name="t", chunk_size=0, overlap=0,
            vector_search_limit=10, final_top_k=10, vector_dim=DIM, use_rerank=False,
            extended_schema=True)
    await r.connect()
    return r


@pytest.mark.asyncio
async def test_prefix_deletes_only_target_turn_not_chunk_siblings(tmp_path):
    r = await _rag(tmp_path)
    # log_key 含下划线；同 turn 多 chunk；另有别的 turn
    await r._db.add_vectors([
        _row("conv_1#aaaa"), _row("conv_1#aaaa#c0"), _row("conv_1#aaaa#c1"),
        _row("conv_1#bbbb"),
    ])
    await r.delete_by_source_prefix("conv_1#aaaa")
    assert await r._db.row_count() == 1  # 只剩 conv_1#bbbb
    await r.close()


@pytest.mark.asyncio
async def test_underscore_not_treated_as_wildcard(tmp_path):
    r = await _rag(tmp_path)
    # prefix has a literal '_'; if '_' were an unescaped LIKE wildcard,
    # pattern conv_1#h% would also match convA1#h (the _ matching 'A') and wrongly delete it.
    await r._db.add_vectors([_row("conv_1#h"), _row("convA1#h")])
    await r.delete_by_source_prefix("conv_1#h")
    assert await r._db.row_count() == 1   # convA1#h must survive (escaped _ is literal)
    await r.close()
