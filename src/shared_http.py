from __future__ import annotations

from collections.abc import Callable
import httpx

import logger

_clients: dict[str, httpx.AsyncClient] = {}


def get_client(
    key: str,
    factory: Callable[[], httpx.AsyncClient],
) -> httpx.AsyncClient:
    """Return a named process-level AsyncClient, creating it via factory when needed."""
    client = _clients.get(key)
    if client is None or client.is_closed:
        client = factory()
        _clients[key] = client
    return client


async def close_client(key: str) -> None:
    global _clients
    client = _clients.pop(key, None)
    if client is not None and not client.is_closed:
        try:
            await client.aclose()
        except Exception as e:
            logger.debug("关闭 HTTP 客户端 %r 时忽略异常: %s", key, e)


async def close_all_clients() -> None:
    """Close every registered shared client (process exit only)."""
    keys = list(_clients)
    for key in keys:
        await close_client(key)
