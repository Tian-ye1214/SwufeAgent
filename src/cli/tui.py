"""Textual-based CLI pilot UI."""

from __future__ import annotations

import asyncio
import threading
from typing import Any

from textual.binding import Binding
from rich.ansi import AnsiDecoder
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.suggester import Suggester
from textual.widgets import Footer, Input, RichLog, Static

from cli.completer import COMMANDS, _iter_file_completions
from cli.output import OutputSink, set_output_sink
from app_config import get_agent_roles


class AgentInputSuggester(Suggester):
    async def get_suggestion(self, value: str) -> str | None:
        if not value:
            return None

        if value.startswith("/") and " " not in value:
            return next((cmd for cmd in COMMANDS if cmd.startswith(value) and cmd != value), None)

        if value.startswith("/agent "):
            parts = value.split()
            prefix = parts[-1] if len(parts) > 1 else ""
            if len(parts) == 2:
                for role in get_agent_roles():
                    if role.startswith(prefix) and role != prefix:
                        return value[: -len(prefix)] + role if prefix else value + role
            return None

        for command in ("/cd ", "/load "):
            if value.startswith(command):
                fragment = value[len(command) :]
                return _complete_path_value(value, fragment)

        at_index = value.rfind("@")
        if at_index != -1:
            fragment = value[at_index + 1 :]
            if " " in fragment:
                return None
            return _complete_path_value(value, fragment)
        return None


def _complete_path_value(value: str, fragment: str) -> str | None:
    for completion in _iter_file_completions(fragment, at_mode=False) or ():
        candidate = str(completion.text)
        if candidate != fragment:
            return value[: -len(fragment)] + candidate if fragment else value + candidate
    return None


class AgentInput(Input):
    BINDINGS = [*Input.BINDINGS, Binding("tab", "cursor_right", "Complete", show=False)]


class TextualOutputSink(OutputSink):
    def __init__(self, app: "RedLotusTui", log: RichLog, status: Static) -> None:
        self._app = app
        self._log = log
        self._status = status
        self._thread_id = threading.get_ident()
        self._ansi_decoder = AnsiDecoder()

    def _run_ui(self, fn) -> None:
        if threading.get_ident() == self._thread_id:
            fn()
            return
        try:
            self._app.call_from_thread(fn)
        except RuntimeError:
            pass

    def emit(self, renderable: Any) -> None:
        if isinstance(renderable, str) and "\x1b[" in renderable:
            parts = list(self._ansi_decoder.decode(renderable))
            def _write_parts() -> None:
                for part in parts:
                    self._log.write(part, scroll_end=True)

            self._run_ui(_write_parts)
            return
        self._run_ui(lambda: self._log.write(renderable, scroll_end=True))

    def rule(self, title: str) -> None:
        self.emit(Text(title, style="dim"))

    def set_status(self, message: str) -> None:
        self._run_ui(lambda: self._app.set_status(message))

    def clear_status(self) -> None:
        self._run_ui(self._app.refresh_status)


class RedLotusTui(App[None]):
    CSS = """
    Screen {
        layout: vertical;
    }

    #output {
        height: 1fr;
        border: round $accent;
    }

    #status {
        height: 1;
        padding: 0 1;
        background: $surface;
        color: $text-muted;
    }

    #input {
        height: 3;
        border: round $primary;
    }

    #input.ask {
        border: thick $warning;
    }
    """

    BINDINGS = [
        ("ctrl+c", "stop_or_quit", "Stop"),
        ("ctrl+q", "quit", "Quit"),
    ]

    def __init__(self, system: Any, stop_event: asyncio.Event | None = None) -> None:
        super().__init__()
        self.system = system
        self.stop_event = stop_event
        self.state = system.new_cli_session_state()
        self._ask_future: asyncio.Future[str] | None = None
        self._ask_question = ""
        self._status_message = "就绪"

    def compose(self) -> ComposeResult:
        with Vertical():
            yield RichLog(id="output", wrap=True, markup=False, highlight=False)
            yield Static("就绪", id="status")
            yield AgentInput(
                placeholder="📝 请输入您的任务:",
                id="input",
                suggester=AgentInputSuggester(case_sensitive=True, use_cache=False),
            )
            yield Footer()

    async def on_mount(self) -> None:
        log = self.query_one("#output", RichLog)
        status = self.query_one("#status", Static)
        set_output_sink(TextualOutputSink(self, log, status))
        self.system.set_ask_user_handler(self.ask_user)
        self.set_interval(0.5, self.refresh_status)
        self.query_one("#input", AgentInput).focus()
        await self.system.prepare_cli_session()
        if self.stop_event is not None:
            asyncio.create_task(self._watch_stop_event())

    async def _watch_stop_event(self) -> None:
        assert self.stop_event is not None
        await self.stop_event.wait()
        self.exit()

    def set_status(self, message: str) -> None:
        self._status_message = message or "就绪"
        self.query_one("#status", Static).update(self._status_text())

    def refresh_status(self) -> None:
        self.query_one("#status", Static).update(self._status_text())

    def _status_text(self) -> str:
        if self._ask_future is not None and not self._ask_future.done():
            return f"需要用户回复: {self._ask_question}"
        if self.system.has_current_turn:
            return self._status_message if self._status_message != "就绪" else "任务运行中... 输入 /status、/tasks 或 /stop"
        return self._status_message or "就绪"

    async def ask_user(self, question: str) -> str:
        if self._ask_future is not None and not self._ask_future.done():
            return "(已有用户提问等待回复)"
        self._ask_question = question.strip()
        self._ask_future = asyncio.get_running_loop().create_future()
        inp = self.query_one("#input", AgentInput)
        inp.add_class("ask")
        inp.suggester = None
        inp.placeholder = f"🤔 {self._ask_question}"
        inp.value = ""
        inp.focus()
        self.refresh_status()
        try:
            return await self._ask_future
        finally:
            self._ask_future = None
            self._ask_question = ""
            inp.remove_class("ask")
            inp.placeholder = "📝 请输入您的任务:"
            inp.suggester = AgentInputSuggester(case_sensitive=True, use_cache=False)
            self.refresh_status()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        value = event.value.strip()
        event.input.value = ""
        if not value:
            return
        if self._ask_future is not None and not self._ask_future.done():
            self._ask_future.set_result(value)
            return
        asyncio.create_task(self._handle_line(value))

    async def _handle_line(self, value: str) -> None:
        self.set_status("处理中...")
        try:
            action = await self.system.process_cli_line(
                value, self.state, wait_for_turn=False
            )
            if action == "break":
                self.exit()
        finally:
            self.refresh_status()

    async def action_stop_or_quit(self) -> None:
        if self.system.has_current_turn:
            msg = await self.system.cancel_current_turn()
            from cli.render import print_warning

            print_warning(msg)
        else:
            self.exit()


async def run_textual_tui(system: Any, *, stop_event: asyncio.Event | None = None) -> None:
    app = RedLotusTui(system, stop_event=stop_event)
    try:
        await app.run_async()
    finally:
        set_output_sink(None)
        system.set_ask_user_handler(None)
        await system.shutdown()
