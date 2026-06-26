from pathlib import Path

import pytest

from RAG.DataBase import EmbedDataBase

DIM = 4


def _row(src, created="2026-06-01T00:00:00Z"):
    return {"vector": [0.1, 0.2, 0.3, 0.4], "text": f"t-{src}", "source": src,
            "id": src, "created_at": created, "agent": "", "session_key": ""}


async def _db(tmp_path: Path):
    db = EmbedDataBase(str(tmp_path / "db"), table_name="t", vector_dim=DIM,
                       extended_schema=True)
    await db.connect()
    return db


@pytest.mark.asyncio
async def test_delete_where_removes_matching(tmp_path):
    db = await _db(tmp_path)
    await db.add_vectors([_row("a#1"), _row("a#2"), _row("b#1")])
    await db.delete_where("source = 'a#1'")
    assert await db.row_count() == 2
    await db.close()


@pytest.mark.asyncio
async def test_vector_search_returns_created_at(tmp_path):
    db = await _db(tmp_path)
    await db.add_vectors([_row("a#1", created="2026-06-09T00:00:00Z")])
    hits = await db.vector_search([0.1, 0.2, 0.3, 0.4], top_k=5)
    assert hits and hits[0]["created_at"] == "2026-06-09T00:00:00Z"
    await db.close()
