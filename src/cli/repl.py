"""增强型 REPL：prompt_toolkit 输入循环。"""

from __future__ import annotations

import asyncio
import sys
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings

from cli.completer import AgentCompleter
from cli.render import print_success, print_warning

# 连续两次 Ctrl+C（空输入）间隔内退出
_INTERRUPT_EXIT_WINDOW_SEC = 1.5
_INPUT_LABEL = "📝 请输入您的任务:"


def _history_path() -> Path:
    base = Path.home() / ".redlotus"
    base.mkdir(parents=True, exist_ok=True)
    return base / "history"


def _create_prompt_session() -> PromptSession:
    kb = KeyBindings()

    @kb.add("c-c")
    def _interrupt(event) -> None:
        buffer = event.app.current_buffer
        if buffer.text:
            buffer.reset()
            return
        event.app.exit(exception=KeyboardInterrupt())

    return PromptSession(
        history=FileHistory(str(_history_path())),
        completer=AgentCompleter(),
        complete_while_typing=False,
        key_bindings=kb,
        bottom_toolbar=HTML(f" <b>{_INPUT_LABEL}</b> "),
    )


class InteractiveRepl:
    """TTY 交互循环；非 TTY 回退到标准 input。"""

    def __init__(self, *, prompt: str = "> ") -> None:
        self.prompt = prompt
        self._session: PromptSession | None = None
        self._interrupt_hits = 0
        self._last_interrupt_at = 0.0

    def _get_session(self) -> PromptSession:
        if self._session is None:
            self._session = _create_prompt_session()
        return self._session

    def _on_keyboard_interrupt(self) -> bool:
        """处理空行 Ctrl+C。返回 True 表示应退出 REPL。"""
        now = time.monotonic()
        if now - self._last_interrupt_at > _INTERRUPT_EXIT_WINDOW_SEC:
            self._interrupt_hits = 1
        else:
            self._interrupt_hits += 1
        self._last_interrupt_at = now

        if self._interrupt_hits >= 2:
            return True
        print_warning("再次按 Ctrl+C 退出，或输入 /exit、quit。")
        return False

    async def read_line(self, *, stop_event: asyncio.Event | None = None) -> str | None:
        if sys.stdin.isatty() and sys.stdout.isatty():
            session = self._get_session()
            try:
                read_coro = session.prompt_async(self.prompt)
                if stop_event is None:
                    return (await read_coro).strip()
                read_task = asyncio.create_task(read_coro)
                stop_task = asyncio.create_task(stop_event.wait())
                done, pending = await asyncio.wait(
                    {read_task, stop_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for t in pending:
                    t.cancel()
                    try:
                        await t
                    except asyncio.CancelledError:
                        pass
                if stop_task in done:
                    return None
                return read_task.result().strip()
            except EOFError:
                return None
            except asyncio.CancelledError:
                return None
        try:
            return (await asyncio.to_thread(input, self.prompt)).strip()
        except EOFError:
            return None

    async def run(
        self,
        handler: Callable[[str], Awaitable[str]],
        *,
        stop_event: asyncio.Event | None = None,
    ) -> None:
        """
        handler 返回 "continue" | "break"。
        空行 Ctrl+C：连按两次退出；有内容时 Ctrl+C 仅清空输入行。
        """
        await self._loop(handler, stop_event)

    async def _loop(
        self,
        handler: Callable[[str], Awaitable[str]],
        stop_event: asyncio.Event | None,
    ) -> None:
        while True:
            if stop_event is not None and stop_event.is_set():
                break
            try:
                line = await self.read_line(stop_event=stop_event)
                if line is None:
                    break
                self._interrupt_hits = 0
                action = await handler(line)
                if action == "break":
                    break
            except KeyboardInterrupt:
                if self._on_keyboard_interrupt():
                    print_success("再见！")
                    break
            except asyncio.CancelledError:
                break
