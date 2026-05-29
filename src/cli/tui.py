from __future__ import annotations

import asyncio
import math
import queue
import threading
from typing import Any

from textual.binding import Binding
from rich.ansi import AnsiDecoder
from rich.align import Align
from rich.panel import Panel
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.suggester import Suggester
from textual.widgets import Footer, Input, RichLog, Static

from cli.completer import COMMANDS, _iter_file_completions
from cli.output import ContextUsageItem, OutputSink, set_output_sink
from app_config import get_agent_roles

READY_LABEL = "就绪"
WORKING_LABEL = "工作中"
WORKING_FRAMES = ("", ".", "..", "...")
STREAM_PREVIEW_MAX_LINES = 10
STREAM_PREVIEW_MAX_CHARS = 6000


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

    @property
    def supports_model_stream(self) -> bool:
        return True

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
        self._run_ui(self._app.clear_status)

    def set_context_usage(self, items: list[ContextUsageItem]) -> None:
        self._run_ui(lambda: self._app.set_context_usage(items))

    def clear_context_usage(self) -> None:
        self._run_ui(self._app.clear_context_usage)

    def begin_model_stream(self, title: str) -> None:
        self._run_ui(lambda: self._app.begin_model_stream(title))

    def append_model_stream_delta(self, text: str) -> None:
        self._run_ui(lambda: self._app.append_model_stream_delta(text))

    def clear_model_stream(self) -> None:
        self._run_ui(self._app.clear_model_stream)


class RedLotusTui(App[None]):
    CSS = """
    Screen { layout: vertical; }
    #output { height: 1fr; border: round $accent; }
    #context-usage { display: none; height: 1; padding: 0 1; color: $text-muted; }
    #stream-preview { display: none; height: 12; max-height: 12; padding: 0 1; }
    #status { height: 1; padding: 0 1; background: $surface; color: $text-muted; }
    #input { height: 3; border: round $primary; }
    #input.ask { border: thick $warning; }
    """

    BINDINGS = [
        ("ctrl+c", "stop_or_quit", "Stop"),
        ("ctrl+q", "stop_or_quit", "Quit"),
    ]

    def __init__(self, system: Any, stop_event: asyncio.Event | None = None) -> None:
        super().__init__()
        self.system = system
        self.stop_event = stop_event
        self.state = system.new_cli_session_state()
        self._ask_future: asyncio.Future[str] | None = None
        self._ask_question = ""
        self._active_line_handlers = 0
        self._status_is_working = False
        self._working_frame = 0
        self._model_stream_title = ""
        self._model_stream_text = ""

    def compose(self) -> ComposeResult:
        with Vertical():
            yield RichLog(id="output", wrap=True, markup=False, highlight=False)
            yield Static("", id="context-usage")
            yield Static("", id="stream-preview")
            yield Static(READY_LABEL, id="status")
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
        self.system.set_ask_user_handler(self._make_ask_user_bridge())
        self.set_interval(0.5, self.refresh_status)
        self.query_one("#input", AgentInput).focus()
        await self.system.prepare_cli_session()
        if self.stop_event is not None:
            asyncio.create_task(self._watch_stop_event())

    def _make_ask_user_bridge(self):
        _ASK_TIMEOUT = 60  # 用户回复超时（秒）

        def ask_user_bridge(question: str) -> str:
            result_queue: queue.Queue[str | Exception] = queue.Queue()

            def _schedule_ask() -> None:
                """在 Textual 事件循环线程中执行。"""
                async def _do_ask() -> None:
                    try:
                        answer = await self.ask_user(question)
                        result_queue.put(answer)
                    except asyncio.CancelledError:
                        result_queue.put(
                            RuntimeError("ask_user cancelled during TUI shutdown")
                        )
                    except Exception as e:
                        result_queue.put(e)

                asyncio.create_task(_do_ask())

            try:
                self.call_from_thread(_schedule_ask)
            except RuntimeError:
                raise RuntimeError("Textual app is no longer running, cannot ask user")

            try:
                result = result_queue.get(timeout=_ASK_TIMEOUT)
            except queue.Empty:
                raise RuntimeError(
                    f"ask_user timed out after {_ASK_TIMEOUT}s (no reply from TUI)"
                )

            if isinstance(result, Exception):
                raise result
            return result

        return ask_user_bridge

    async def _watch_stop_event(self) -> None:
        assert self.stop_event is not None
        await self.stop_event.wait()
        self.exit()

    def set_status(self, message: str) -> None:
        self._status_is_working = bool(message and message != READY_LABEL)
        self.query_one("#status", Static).update(self._status_text())

    def clear_status(self) -> None:
        self._status_is_working = False
        self.refresh_status()

    @staticmethod
    def _context_usage_bar(percent: float, width: int = 8) -> str:
        clamped = max(0.0, min(100.0, float(percent)))
        filled = 0 if clamped <= 0 else math.ceil(width * clamped / 100.0)
        filled = max(0, min(width, filled))
        return "[" + "=" * filled + "." * (width - filled) + "]"

    @classmethod
    def _context_usage_renderable(cls, items: list[ContextUsageItem]) -> Align:
        parts = [
            f"{item.role_label} {cls._context_usage_bar(item.percent)} {item.percent:.0f}%"
            for item in items
        ]
        return Align.right(Text("  ".join(parts), style="dim"))

    def set_context_usage(self, items: list[ContextUsageItem]) -> None:
        if not items:
            self.clear_context_usage()
            return
        widget = self.query_one("#context-usage", Static)
        widget.update(self._context_usage_renderable(items))
        widget.display = True

    def clear_context_usage(self) -> None:
        widget = self.query_one("#context-usage", Static)
        widget.update("")
        widget.display = False

    def refresh_status(self) -> None:
        self.query_one("#status", Static).update(self._status_text())

    def _is_working(self) -> bool:
        if self._ask_future is not None and not self._ask_future.done():
            return True
        return bool(
            self._active_line_handlers > 0
            or self._status_is_working
            or self.system.has_current_turn
        )

    def _status_text(self) -> str:
        if not self._is_working():
            self._working_frame = 0
            return READY_LABEL
        suffix = WORKING_FRAMES[self._working_frame % len(WORKING_FRAMES)]
        self._working_frame += 1
        return f"{WORKING_LABEL}{suffix}"

    @staticmethod
    def _user_input_renderable(value: str) -> Panel:
        return Panel(
            Text(value, style="bold white"), title="用户",
            title_align="left", border_style="bright_blue",
            padding=(0, 1), expand=False,
        )

    def _write_user_input(self, value: str) -> None:
        self.query_one("#output", RichLog).write(
            self._user_input_renderable(value), scroll_end=True,
        )

    @staticmethod
    def _user_reply_renderable(value: str) -> Panel:
        return Panel(
            Text(value, style="bold white"), title="用户回复",
            title_align="left", border_style="bright_blue",
            padding=(0, 1), expand=False,
        )

    def _write_user_reply(self, value: str) -> None:
        self.query_one("#output", RichLog).write(
            self._user_reply_renderable(value), scroll_end=True,
        )

    @staticmethod
    def _model_stream_renderable(title: str, text: str) -> Panel:
        return Panel(
            Text(text or " ", style="white"), title=title,
            title_align="left", border_style="cyan",
            padding=(0, 1), expand=False,
        )

    def _stream_preview(self) -> Static:
        return self.query_one("#stream-preview", Static)

    @staticmethod
    def _model_stream_visible_text(text: str) -> str:
        body = text or ""
        if len(body) > STREAM_PREVIEW_MAX_CHARS:
            body = body[-STREAM_PREVIEW_MAX_CHARS :].lstrip("\n")
        lines = body.splitlines()
        if len(lines) > STREAM_PREVIEW_MAX_LINES:
            body = "\n".join(lines[-STREAM_PREVIEW_MAX_LINES:])
        return body

    def _refresh_model_stream(self) -> None:
        self._stream_preview().update(
            self._model_stream_renderable(
                self._model_stream_title,
                self._model_stream_visible_text(self._model_stream_text),
            )
        )

    def begin_model_stream(self, title: str) -> None:
        self._model_stream_title = title
        self._model_stream_text = ""
        preview = self._stream_preview()
        preview.display = True
        self._refresh_model_stream()

    def append_model_stream_delta(self, text: str) -> None:
        if not text:
            return
        self._model_stream_text += text
        if not self._model_stream_title:
            self._model_stream_title = "模型正在回复"
        preview = self._stream_preview()
        preview.display = True
        self._refresh_model_stream()

    def clear_model_stream(self) -> None:
        self._model_stream_title = ""
        self._model_stream_text = ""
        preview = self._stream_preview()
        preview.update("")
        preview.display = False

    def _cancel_pending_ask(self) -> bool:
        fut = self._ask_future
        if fut is not None and not fut.done():
            fut.cancel()
            return True
        return False

    async def ask_user(self, question: str) -> str:
        """在 Textual 事件循环中弹出用户提问界面（内部方法）。"""
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
            self._write_user_reply(value)
            self._ask_future.set_result(value)
            return
        if value.lower() == "/api":
            from cli.render import print_warning
            print_warning("TUI does not support interactive /api changes; use the legacy CLI or config file.")
            return
        if self._active_line_handlers > 0 and self._should_echo_as_user_input(value):
            from cli.render import print_warning
            print_warning("A previous input is still being scheduled; please wait.")
            return
        if self._should_echo_as_user_input(value):
            self._write_user_input(value)
        self._active_line_handlers += 1
        self.refresh_status()
        asyncio.create_task(self._handle_line(value))

    def _should_echo_as_user_input(self, value: str) -> bool:
        if value.startswith("/"):
            return False
        if value.lower() in ("quit", "exit", "退出"):
            return False
        return True

    async def _handle_line(self, value: str) -> None:
        try:
            action = await self.system.process_cli_line(value, self.state, wait_for_turn=False)
            if action == "break":
                self.exit()
        finally:
            self._active_line_handlers = max(0, self._active_line_handlers - 1)
            self.refresh_status()

    async def action_stop_or_quit(self) -> None:
        ask_cancelled = self._cancel_pending_ask()
        if self.system.has_current_turn:
            msg = await self.system.cancel_current_turn()
            from cli.render import print_warning
            print_warning(msg)
        elif not ask_cancelled:
            self.exit()


async def run_textual_tui(system: Any, *, stop_event: asyncio.Event | None = None) -> None:
    app = RedLotusTui(system, stop_event=stop_event)
    try:
        await app.run_async()
    finally:
        app._cancel_pending_ask()
        set_output_sink(None)
        system.set_ask_user_handler(None)
        await system.shutdown()
