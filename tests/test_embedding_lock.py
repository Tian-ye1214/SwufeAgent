from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import RAG.embedding_function as ef


@pytest.mark.asyncio
async def test_http_client_lock_is_singleton() -> None:
    ef._http_state = None
    created = 0

    class FakeClient:
        async def aclose(self) -> None:
            return None

    def fake_get_env(key: str, *, warn: bool = False, default: str = "") -> str:
        return "https://api.example.com" if key == "SILICONFLOW_BASE" else "k"

    with patch.object(ef, "get_env", side_effect=fake_get_env):
        with patch("RAG.embedding_function.httpx.AsyncClient") as mock_cls:
            def _make_client(**_kw: object) -> FakeClient:
                nonlocal created
                created += 1
                return FakeClient()

            mock_cls.side_effect = _make_client

            async def one() -> None:
                await ef._get_http_client()

            await asyncio.gather(one(), one(), one())
    assert created == 1
