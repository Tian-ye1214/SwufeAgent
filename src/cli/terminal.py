"""Coordination helpers for prompt_toolkit input and terminal output."""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Callable
from typing import TypeVar

from prompt_toolkit.application import in_terminal
from prompt_toolkit.application.current import get_app_or_none
from prompt_toolkit.application.current import set_app

_T = TypeVar("_T")
_prompt_app: object | None = None
_prompt_loop: asyncio.AbstractEventLoop | None = None


def register_prompt_app(app: object | None, loop: asyncio.AbstractEventLoop | None) -> None:
    """Register the prompt application that currently owns the terminal."""
    global _prompt_app, _prompt_loop
    _prompt_app = app
    _prompt_loop = loop


def is_prompt_active() -> bool:
    """Return True while a prompt_toolkit application owns the terminal."""
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return False
    app = get_app_or_none()
    if app is not None and getattr(app, "is_running", False):
        return True
    return bool(_prompt_app is not None and getattr(_prompt_app, "is_running", False))


async def _run_with_prompt_app(app: object, func: Callable[[], _T]) -> None:
    with set_app(app):  # type: ignore[arg-type]
        async with in_terminal(render_cli_done=False):
            func()


def run_above_prompt(func: Callable[[], _T]) -> _T | None:
    """
    Run terminal output above the active prompt without patching stdout.

    prompt_toolkit temporarily hides the input area, runs ``func`` against the
    real terminal, then redraws the prompt at the bottom. When no prompt is
    active, the function runs immediately.
    """
    app = _prompt_app or get_app_or_none()
    if is_prompt_active() and app is not None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        coro = _run_with_prompt_app(app, func)
        if loop is not None and (loop is _prompt_loop or _prompt_loop is None):
            loop.create_task(coro)
        elif _prompt_loop is not None and _prompt_loop.is_running():
            prompt_loop = _prompt_loop
            prompt_loop.call_soon_threadsafe(lambda: prompt_loop.create_task(coro))
        else:
            coro.close()
            return func()
        return None
    return func()
