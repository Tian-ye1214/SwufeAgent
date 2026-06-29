from __future__ import annotations

import asyncio
import math
import queue
import threading
from enum import Enum
from typing import Any

from textual.binding import Binding
from rich.ansi import AnsiDecoder
from rich.align import Align
from rich.panel import Panel
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.suggester import Suggester
from textual.widgets import Footer, Input, Label, OptionList, ProgressBar, RichLog, Sparkline, Static
from textual.widgets.option_list import Option

from redlotus.cli.completer import _iter_file_completions
from redlotus.cli.completion import (
    completion_for_input,
    iter_agent_role_completions,
    iter_command_completions,
    iter_effort_value_completions,
    iter_literal_choice_completions,
)
from redlotus.cli.output import ContextUsageItem, OutputSink, set_output_sink
from redlotus.cli.panel import PanelSnapshotCache, build_panel_snapshot, render_panel
from redlotus.cli.pending_review import compute_hunks
from redlotus.infra import logger
from redlotus.runtime.runtime_state import current_short_agent_id
from redlotus.workspace.workspace import WorkspaceSnapshot

READY_LABEL = "就绪"
WORKING_LABEL = "工作中"
WORKING_FRAMES = ("", ".", "..", "...")
STREAM_PREVIEW_MAX_LINES = 10
STREAM_PREVIEW_MAX_CHARS = 6000


class TuiRunMode(str, Enum):
    REVIEW = "review"
    PASS = "pass"
    GOAL = "goal"

    def next(self) -> "TuiRunMode":
        if self == TuiRunMode.REVIEW:
            return TuiRunMode.PASS
        if self == TuiRunMode.PASS:
            return TuiRunMode.GOAL
        return TuiRunMode.REVIEW


class AgentInputSuggester(Suggester):
    async def get_suggestion(self, value: str) -> str | None:
        if not value:
            return None
        ctx = completion_for_input(value)
        if ctx is None:
            return None
        if ctx.kind == "command":
            return next(
                (cmd for cmd in iter_command_completions(value) if cmd != value),
                None,
            )
        if ctx.kind == "agent_role":
            for role in iter_agent_role_completions(ctx.prefix):
                if role != ctx.prefix:
                    return value[: -len(ctx.prefix)] + role if ctx.prefix else value + role
            return None
        if ctx.kind == "effort_value":
            for effort in iter_effort_value_completions(ctx.prefix, ctx.role):
                if effort != ctx.prefix:
                    return value[: -len(ctx.prefix)] + effort if ctx.prefix else value + effort
            return None
        if ctx.kind == "literal_choice":
            for choice in iter_literal_choice_completions(ctx.prefix, ctx.choices):
                if choice != ctx.prefix:
                    return value[: -len(ctx.prefix)] + choice if ctx.prefix else value + choice
            return None
        if ctx.kind == "file_path":
            return _complete_path_value(value, ctx.prefix)
        return None


def _complete_path_value(value: str, fragment: str) -> str | None:
    for completion in _iter_file_completions(fragment, at_mode=False) or ():
        candidate = str(completion.text)
        if candidate != fragment:
            return value[: -len(fragment)] + candidate if fragment else value + candidate
    return None


class AgentInput(Input):
    BINDINGS = [*Input.BINDINGS, Binding("tab", "cursor_right", "Complete", show=False)]


class SnapshotPickScreen(ModalScreen[WorkspaceSnapshot | None]):
    BINDINGS = [Binding("escape", "cancel", "取消", show=False)]

    DEFAULT_CSS = """
    SnapshotPickScreen {
        align: center middle;
    }
    #snapshot-dialog {
        width: 90%;
        max-width: 120;
        height: auto;
        max-height: 80%;
        border: thick $primary;
        background: $surface;
        padding: 1 2;
    }
    #snapshot-title {
        text-style: bold;
        margin-bottom: 1;
    }
    .snapshot-hint {
        color: $text-muted;
        margin-bottom: 1;
    }
    #snapshot-list {
        height: auto;
        max-height: 24;
        min-height: 5;
    }
    """

    def __init__(self, snapshots: list[WorkspaceSnapshot]) -> None:
        super().__init__()
        self._snapshots = snapshots

    def compose(self) -> ComposeResult:
        with Vertical(id="snapshot-dialog"):
            yield Static("选择要加载的对话快照", id="snapshot-title")
            yield Static("↑↓ 选择 · Enter 确认 · Esc 取消", classes="snapshot-hint")
            yield OptionList(
                *[
                    Option(snapshot.label, id=str(index))
                    for index, snapshot in enumerate(self._snapshots)
                ],
                id="snapshot-list",
            )

    def on_mount(self) -> None:
        self.query_one("#snapshot-list", OptionList).focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        option_id = event.option.id
        if option_id is None:
            self.dismiss(None)
            return
        self.dismiss(self._snapshots[int(option_id)])

    def action_cancel(self) -> None:
        self.dismiss(None)


class TextualOutputSink(OutputSink):
    def __init__(self, app: "RedLotusTui", log: RichLog) -> None:
        self._app = app
        self._log = log
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
    #review-view { display: none; height: 1fr; border: round $warning; padding: 0 1; }
    #panel-view { display: none; height: 1fr; border: round $success; padding: 0 1; }
    .panel-chart-title { color: $text-muted; text-style: bold; margin-top: 1; }
    #panel-trend { height: 3; }
    .panel-bar-row { height: 1; width: 1fr; }
    .panel-bar-label { width: 12; }
    .panel-bar-row ProgressBar { width: 1fr; }
    #panel-task-progress { width: 1fr; }
    #panel-task-counts { height: 1; }
    .hunk-view { height: auto; padding: 0 0 1 0; }
    .hunk-view.-current { background: $boost; }
    #context-usage { display: none; height: 1; padding: 0 1; color: $text-muted; }
    #stream-preview { display: none; height: 12; max-height: 12; padding: 0 1; }
    #status { height: 1; padding: 0 1; background: $surface; color: $text-muted; }
    #input { height: 3; border: round $primary; }
    #input.ask { border: thick $warning; }
    """

    BINDINGS = [
        ("ctrl+c", "stop_or_quit", "Stop"),
        ("ctrl+q", "stop_or_quit", "Quit"),
        Binding("shift+tab", "toggle_mode", "切换模式", priority=True),
        ("ctrl+r", "review", "审查更改"),
        Binding("y", "review_keep", "保留", show=False),
        Binding("n", "review_undo", "撤销", show=False),
        Binding("up", "review_prev", "上一处", show=False),
        Binding("down", "review_next", "下一处", show=False),
        Binding("escape", "escape", "退出面板/审查", show=False),
    ]

    def __init__(self, system: Any, stop_event: asyncio.Event | None = None) -> None:
        super().__init__()
        self.system = system
        self.stop_event = stop_event
        self.state = system.new_cli_session_state()
        self._ask_future: asyncio.Future[str] | None = None
        self._ask_lock: asyncio.Lock | None = None
        self._ask_question = ""
        self._api_state: dict[str, Any] | None = None
        self._active_line_handlers = 0
        self._status_is_working = False
        self._working_frame = 0
        self._model_stream_title = ""
        self._model_stream_text = ""
        self._ui_thread_id = 0
        self._run_mode = TuiRunMode.REVIEW
        self._review_mode = False
        self._review_items: list = []  # [(key, name, hunk), ...] 当前未决定的改动
        self._review_idx = 0
        self._review_widgets: list = []
        self._pending_count = 0
        self._rv_snapshots: dict[str, str] = {}
        self._rv_decided: dict[str, set[int]] = {}
        self._rv_rejected: dict[str, set[int]] = {}
        self._panel_mode = False
        self._panel_include_all = False
        self._panel_cache = PanelSnapshotCache()
        self._panel_timer = None
        self._panel_refresh_task = None

    def compose(self) -> ComposeResult:
        with Vertical():
            yield RichLog(id="output", wrap=True, markup=False, highlight=False)
            yield VerticalScroll(id="review-view")
            with VerticalScroll(id="panel-view"):
                yield Static("", id="panel-content")
                yield Label("Token 趋势（按会话 旧→新）", classes="panel-chart-title")
                yield Sparkline(id="panel-trend")
                yield Label("Token 占比（Input / Output / Reasoning）", classes="panel-chart-title")
                with Horizontal(classes="panel-bar-row"):
                    yield Label("Input", classes="panel-bar-label")
                    yield ProgressBar(id="panel-comp-input", show_eta=False)
                with Horizontal(classes="panel-bar-row"):
                    yield Label("Output", classes="panel-bar-label")
                    yield ProgressBar(id="panel-comp-output", show_eta=False)
                with Horizontal(classes="panel-bar-row"):
                    yield Label("Reasoning", classes="panel-bar-label")
                    yield ProgressBar(id="panel-comp-reasoning", show_eta=False)
                yield Label("任务进度", classes="panel-chart-title")
                yield ProgressBar(id="panel-task-progress", show_eta=False)
                yield Static("", id="panel-task-counts")
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
        set_output_sink(TextualOutputSink(self, log))
        self.system.set_ask_user_handler(self._make_ask_user_bridge())
        self._ui_thread_id = threading.get_ident()
        if self._run_mode == TuiRunMode.REVIEW:
            self.system.review_store.activate(self._on_reviews_changed)
        self.set_interval(0.5, self.refresh_status)
        self.query_one("#input", AgentInput).focus()
        await self._prepare_cli_session()
        controller = self.system._cli_controller
        controller._active_session_state = self.state
        controller.set_snapshot_picker(self.pick_snapshot)
        if self.stop_event is not None:
            asyncio.create_task(self._watch_stop_event())
        self.call_after_refresh(self._schedule_workspace_enter)

    def _schedule_workspace_enter(self) -> None:
        self.run_worker(self._enter_workspace_after_mount, exclusive=True)

    async def _prepare_cli_session(self) -> None:
        missing = await self.system.prepare_cli_session()
        if missing:
            self._begin_api_input()

    async def _enter_workspace_after_mount(self) -> None:
        loaded = await self.system._cli_controller.enter_current_workspace()
        if loaded:
            self.state.is_first_input = False

    async def pick_snapshot(
        self,
        snapshots: list[WorkspaceSnapshot],
    ) -> WorkspaceSnapshot | None:
        if not snapshots:
            return None
        if len(snapshots) == 1:
            return snapshots[0]

        loop = asyncio.get_running_loop()
        future: asyncio.Future[WorkspaceSnapshot | None] = loop.create_future()

        def _on_result(result: WorkspaceSnapshot | None) -> None:
            if not future.done():
                future.set_result(result)

        screen = SnapshotPickScreen(snapshots)
        await self.push_screen(screen, callback=_on_result, wait_for_dismiss=False)
        return await future

    def _make_ask_user_bridge(self):
        _ASK_TIMEOUT = 60  # 用户回复超时（秒）

        def ask_user_bridge(question: str) -> str:
            result_queue: queue.Queue[str | BaseException] = queue.Queue()
            who = current_short_agent_id()  # 在提问 agent 的上下文（worker 线程）内读取
            tagged = f"[{who}] {question}" if who else question

            def _schedule_ask() -> None:
                """在 Textual 事件循环线程中执行。"""
                async def _do_ask() -> None:
                    try:
                        answer = await self.ask_user(tagged)
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
            except queue.Empty as e:
                raise TimeoutError("Timed out waiting for TUI ask_user response") from e
            if isinstance(result, BaseException):
                raise result
            return result

        return ask_user_bridge

    async def _watch_stop_event(self) -> None:
        assert self.stop_event is not None
        await self.stop_event.wait()
        self.exit()

    def _on_reviews_changed(self) -> None:
        """store 回调（可能来自 worker 线程）：更新待审查计数。"""
        if threading.get_ident() == self._ui_thread_id:
            self._update_pending()
        else:
            try:
                self.call_from_thread(self._update_pending)
            except RuntimeError:
                pass

    def _pending_hunks(self) -> list:
        items: list = []
        entries = self.system.review_store.entries()
        valid = {str(e.path) for e in entries}
        for key in [k for k in self._rv_decided if k not in valid]:
            self._rv_decided.pop(key, None)
            self._rv_rejected.pop(key, None)
            self._rv_snapshots.pop(key, None)
        for entry in entries:
            key = str(entry.path)
            if self._rv_snapshots.get(key) != entry.snapshot:  # 新文件或 agent 又改了
                self._rv_snapshots[key] = entry.snapshot
                self._rv_decided[key] = set()
                self._rv_rejected[key] = set()
            decided = self._rv_decided.get(key, set())
            for hunk in compute_hunks(entry.baseline, entry.snapshot):
                if hunk.index not in decided:
                    items.append((key, entry.name, hunk))
        return items

    def _update_pending(self) -> None:
        self._pending_count = len(self._pending_hunks())
        self.refresh_status()

    def action_review(self) -> None:
        """Ctrl+R：审查模式临时占用输出区，全量展示所有待审查改动；y/n 决定，完成后还原输出区。"""
        self._review_items = self._pending_hunks()
        if not self._review_items:
            return
        self._review_mode = True
        self._review_idx = 0
        view = self.query_one("#review-view", VerticalScroll)
        view.remove_children()
        self._review_widgets = [
            Static(self._render_hunk(name, hunk), classes="hunk-view")
            for _key, name, hunk in self._review_items
        ]
        view.mount(*self._review_widgets)
        view.border_title = "审查改动 · y 保留 · n 撤销 · ↑↓ 切换 · Esc 退出"
        self.query_one("#output", RichLog).display = False
        view.display = True
        self.set_focus(None)
        self._highlight_current()
        self.refresh_status()

    def _render_hunk(self, name: str, hunk, status: str | None = None) -> Text:
        """一处改动的完整展示：位置 + 全部 +/- 内容（不截断）+ 决定状态。"""
        t = Text()
        t.append(f"{name}  {hunk.location}\n", style="bold")
        for ln in hunk.old_lines:
            t.append(f"  - {ln.rstrip()}\n", style="red")
        for ln in hunk.new_lines:
            t.append(f"  + {ln.rstrip()}\n", style="green")
        if status == "keep":
            t.append("  → ✓ 已保留", style="bold green")
        elif status == "undo":
            t.append("  → ✗ 已撤销", style="bold red")
        else:
            t.append("  …待决定（y 保留 / n 撤销）", style="dim")
        return t

    def _highlight_current(self) -> None:
        for i, widget in enumerate(self._review_widgets):
            widget.set_class(i == self._review_idx, "-current")
        if 0 <= self._review_idx < len(self._review_widgets):
            self._review_widgets[self._review_idx].scroll_visible()

    def _decide_current(self, reject: bool) -> None:
        if not self._review_mode or not self._review_items:
            return
        if not (0 <= self._review_idx < len(self._review_items)):
            return
        key, name, hunk = self._review_items[self._review_idx]
        self._rv_decided.setdefault(key, set()).add(hunk.index)
        rejected = self._rv_rejected.setdefault(key, set())
        if reject:
            rejected.add(hunk.index)
        else:
            rejected.discard(hunk.index)  # 改判为保留：撤销之前的"撤销"
        self.system.review_store.apply(key, set(rejected))
        self._review_widgets[self._review_idx].update(
            self._render_hunk(name, hunk, "undo" if reject else "keep")
        )
        was_last = self._review_idx == len(self._review_items) - 1
        if was_last and self._all_decided():
            self._exit_review()
            return
        self._review_idx = min(self._review_idx + 1, len(self._review_items) - 1)
        self._highlight_current()
        self.refresh_status()

    def _all_decided(self) -> bool:
        return all(
            hunk.index in self._rv_decided.get(key, set())
            for key, _name, hunk in self._review_items
        )

    def _finish_if_done(self, key: str) -> None:
        entry = self.system.review_store.get(key)
        if entry is None:
            return
        decided = self._rv_decided.get(key, set())
        if all(h.index in decided for h in compute_hunks(entry.baseline, entry.snapshot)):
            self.system.review_store.finish(key)

    def _exit_review(self) -> None:
        for key in list(self._rv_decided.keys()):  # 退出时：全部块都决定完的文件 → 清出暂存区
            self._finish_if_done(key)
        self._review_mode = False
        self._review_items = []
        self._review_idx = 0
        self._review_widgets = []
        view = self.query_one("#review-view", VerticalScroll)
        view.remove_children()
        view.display = False
        self.query_one("#output", RichLog).display = True  # 还原输出区（内容原样保留）
        self._pending_count = len(self._pending_hunks())
        self.query_one("#input", AgentInput).focus()
        self.refresh_status()

    async def open_panel(self, *, include_all: bool = False) -> None:
        if self._review_mode:
            self._exit_review()
        self._panel_mode = True
        self._panel_include_all = include_all
        self.query_one("#output", RichLog).display = False
        panel_view = self.query_one("#panel-view", VerticalScroll)
        panel_view.border_title = "工作区总览 · 每 3 秒刷新 · Esc 退出"
        panel_view.display = True
        self._schedule_panel_refresh()
        self._ensure_panel_timer()
        self.query_one("#input", AgentInput).focus()
        self.refresh_status()

    async def _refresh_panel(self) -> None:
        if not self._panel_mode:
            return
        try:
            from redlotus.workspace.workspace import conversations_root

            snapshot = await build_panel_snapshot(
                log_root=conversations_root(),
                system=self.system,
                coordinator_history=self.state.history,
                manager_history=getattr(self.system, "_manager_history", None),
                include_all=self._panel_include_all,
                cache=self._panel_cache,
            )
            self.query_one("#panel-content", Static).update(render_panel(snapshot))
            self._update_panel_charts(snapshot)
        except Exception as e:
            logger.error(f"刷新工作区面板失败: {type(e).__name__}: {e}", exc_info=True)

    def _update_panel_charts(self, snapshot: Any) -> None:
        """就地更新面板内的原生图表控件（趋势 Sparkline、占比与任务 ProgressBar），避免重建。"""
        trend = self.query_one("#panel-trend", Sparkline)
        series = [float(v) for v in (snapshot.token_trend or [])]
        if series and any(series):
            trend.data = series
            trend.display = True
        else:
            trend.data = []
            trend.display = False
        history = snapshot.history
        total = (history.input_tokens or 0) + (history.output_tokens or 0) + (history.reasoning_tokens or 0)
        for widget_id, value in (
            ("#panel-comp-input", history.input_tokens or 0),
            ("#panel-comp-output", history.output_tokens or 0),
            ("#panel-comp-reasoning", history.reasoning_tokens or 0),
        ):
            self.query_one(widget_id, ProgressBar).update(total=total or 1, progress=value)
        tasks = snapshot.runtime.tasks
        task_total = tasks.total or 0
        self.query_one("#panel-task-progress", ProgressBar).update(
            total=task_total or 1, progress=tasks.completed or 0
        )
        counts = Text()
        counts.append(f"✓ Completed {tasks.completed}/{task_total}", style="green")
        counts.append("   ")
        counts.append(f"⟳ Running {tasks.running}", style="cyan")
        counts.append("   ")
        counts.append(f"✗ Failed {tasks.failed}", style="red")
        counts.append("   ")
        counts.append(f"… Pending {tasks.pending}", style="yellow")
        self.query_one("#panel-task-counts", Static).update(counts)

    def _schedule_panel_refresh(self) -> None:
        if not self._panel_mode:
            return
        task = self._panel_refresh_task
        if task is not None and not task.done():
            return
        self._panel_refresh_task = asyncio.create_task(self._refresh_panel())

    def _ensure_panel_timer(self) -> None:
        if self._panel_timer is not None:
            return
        self._panel_timer = self.set_interval(3.0, self._schedule_panel_refresh)

    def _stop_panel_timer(self) -> None:
        timer = self._panel_timer
        self._panel_timer = None
        task = self._panel_refresh_task
        self._panel_refresh_task = None
        if task is not None and not task.done():
            task.cancel()
        if timer is None:
            return
        stop = getattr(timer, "stop", None)
        if callable(stop):
            stop()

    def _exit_panel(self) -> None:
        if not self._panel_mode:
            return
        self._panel_mode = False
        self._stop_panel_timer()
        self.query_one("#panel-view", VerticalScroll).display = False
        self.query_one("#output", RichLog).display = True
        self.query_one("#input", AgentInput).focus()
        self.refresh_status()

    def action_review_keep(self) -> None:
        if self._review_mode:
            self._decide_current(False)

    def action_review_undo(self) -> None:
        if self._review_mode:
            self._decide_current(True)

    def action_review_prev(self) -> None:
        """↑：回退到上一处，可重新选择 y/n。"""
        if self._review_mode and self._review_items:
            self._review_idx = max(0, self._review_idx - 1)
            self._highlight_current()
            self.refresh_status()

    def action_review_next(self) -> None:
        if self._review_mode and self._review_items:
            self._review_idx = min(len(self._review_items) - 1, self._review_idx + 1)
            self._highlight_current()
            self.refresh_status()

    def action_escape(self) -> None:
        if self._panel_mode:
            self._exit_panel()
            return
        if self._review_mode:
            self._exit_review()

    def action_toggle_mode(self) -> None:
        """Shift+Tab：审查模式 -> 放行模式 -> 目标模式 -> 审查模式。"""
        if self._review_mode or self._panel_mode:
            return  # 浮层中不切换全局模式
        if self.system.has_current_goal_turn:
            return
        self._run_mode = self._run_mode.next()
        if self._run_mode == TuiRunMode.REVIEW:
            self.system.review_store.activate(self._on_reviews_changed)
        else:
            self.system.review_store.deactivate()  # 清空待审查；后续写入直接放行
        self._update_pending()

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

    def _mode_chip(self) -> Text:
        if self._run_mode == TuiRunMode.REVIEW:
            return Text(" ⏵ 审查模式 ", style="bold black on cyan")
        if self._run_mode == TuiRunMode.PASS:
            return Text(" ⏵⏵ 放行模式 ", style="bold black on green")
        return Text(" ◎ 目标模式 ", style="bold black on yellow")

    def _status_text(self) -> Any:
        if self._panel_mode:
            return Text("Panel 总览    ·    每 3 秒刷新    ·    Esc 退出", style="bold")
        if self._review_mode:
            return Text("审查改动中    ·    y 保留    ·    n 撤销    ·    ↑↓ 切换    ·    Esc 退出", style="bold")
        if not self._is_working():
            self._working_frame = 0
            t = Text()
            t.append(self._mode_chip())
            t.append("  ")
            t.append(READY_LABEL, style="dim")
            if self._run_mode == TuiRunMode.REVIEW and self._pending_count > 0:
                t.append("       ")
                t.append(f" ⚑ 待审查 {self._pending_count} 处 · 按 Ctrl+R 审查 ", style="bold black on yellow")
            return t
        suffix = WORKING_FRAMES[self._working_frame % len(WORKING_FRAMES)]
        self._working_frame += 1
        t = Text()
        t.append(self._mode_chip())
        t.append("  ")
        if self.system.has_current_goal_turn:
            iteration = self.system.current_goal_iteration
            label = f"目标循环第 {iteration} 轮" if iteration else "目标循环"
            t.append(f"{label}{suffix}")
        else:
            t.append(f"{WORKING_LABEL}{suffix}")
        return t

    @staticmethod
    def _panel_renderable(
        content: str,
        title: str,
        *,
        text_style: str = "bold white",
        border_style: str = "bright_blue",
    ) -> Panel:
        return Panel(
            Text(content or " ", style=text_style),
            title=title,
            title_align="left",
            border_style=border_style,
            padding=(0, 1),
            expand=False,
        )

    def _write_user_input(self, value: str) -> None:
        self.query_one("#output", RichLog).write(
            self._panel_renderable(value, "用户"), scroll_end=True,
        )

    def _write_user_reply(self, value: str) -> None:
        self.query_one("#output", RichLog).write(
            self._panel_renderable(value, "用户回复"), scroll_end=True,
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
            self._panel_renderable(
                self._model_stream_visible_text(self._model_stream_text),
                self._model_stream_title,
                text_style="white",
                border_style="cyan",
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
        if self._ask_lock is None:
            self._ask_lock = asyncio.Lock()
        async with self._ask_lock:  # 多个并行提问按 FIFO 串行排队，互不丢弃
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

    def _set_api_prompt(self, prompt: str) -> None:
        inp = self.query_one("#input", AgentInput)
        inp.add_class("ask")
        inp.suggester = None
        inp.placeholder = prompt
        inp.value = ""
        inp.focus()
        self.refresh_status()

    def _restore_api_prompt(self) -> None:
        inp = self.query_one("#input", AgentInput)
        inp.remove_class("ask")
        inp.placeholder = "📝 请输入您的任务:"
        inp.suggester = AgentInputSuggester(case_sensitive=True, use_cache=False)
        self.refresh_status()

    @staticmethod
    def _api_fields(embedding: bool) -> tuple[tuple[str, str, str], ...]:
        if embedding:
            return (
                ("embedding_url", "SILICONFLOW_BASE", "Embedding URL"),
                ("embedding_key", "SILICONFLOW_KEY", "Embedding Key"),
            )
        return (("base_url", "BASE_URL", "BASE_URL"), ("api_key", "API_KEY", "API_KEY"))

    @staticmethod
    def _api_value_label(key: str, value: str) -> str:
        if "KEY" in key:
            return "已填写" if value else "(空；优先 config 再 .env)"
        return value or "(空；优先 config 再 .env)"

    @staticmethod
    def _api_prompt(label: str) -> str:
        return f"请输入 {label}（回车保持当前值）:"

    def _begin_api_input(self, *, embedding: bool = False) -> None:
        from redlotus.config.app_config import get_env
        from redlotus.cli.render import print_panel

        fields = self._api_fields(embedding)
        lines = []
        for _, key, label in fields:
            cur = (get_env(key, warn=False) or "").strip()
            lines.append(f"当前 {label}: {self._api_value_label(key, cur)}")
        print_panel(
            "\n".join(lines),
            title="/api embedding" if embedding else "/api",
        )
        self._api_state = {"fields": fields, "index": 0, "values": {}}
        self._set_api_prompt(self._api_prompt(fields[0][2]))

    def _handle_api_input(self, value: str) -> bool:
        if self._api_state is None:
            return False
        fields = self._api_state["fields"]
        index = int(self._api_state["index"])
        arg, _, _ = fields[index]
        self._api_state["values"][arg] = value or None
        index += 1
        if index < len(fields):
            self._api_state["index"] = index
            self._set_api_prompt(self._api_prompt(fields[index][2]))
            return True

        from redlotus.config import app_config
        from redlotus.cli.render import print_success

        app_config.set_api(**self._api_state["values"])
        self._api_state = None
        self._restore_api_prompt()
        print_success("已更新 API 配置并写入 config.json。")
        return True

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        value = event.value.strip()
        event.input.value = ""
        if self._api_state is not None:
            self._handle_api_input(value)
            return
        if not value:
            return
        if self._ask_future is not None and not self._ask_future.done():
            self._write_user_reply(value)
            self._ask_future.set_result(value)
            return
        parts = value.split()
        if parts and parts[0].lower() == "/panel":
            include_all = any(part.lower() == "--all" for part in parts[1:])
            await self.open_panel(include_all=include_all)
            return
        if self._panel_mode:
            self._exit_panel()
        if parts and parts[0].lower() == "/api":
            if len(parts) == 1:
                self._begin_api_input()
            elif len(parts) == 2 and parts[1].lower() == "embedding":
                self._begin_api_input(embedding=True)
            else:
                from redlotus.cli.render import print_error
                print_error("用法: /api [embedding]")
            return
        if self._active_line_handlers > 0 and self._should_echo_as_user_input(value):
            from redlotus.cli.render import print_warning
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
            action = await self.system.process_cli_line(
                value,
                self.state,
                wait_for_turn=False,
                goal_mode=self._run_mode == TuiRunMode.GOAL,
            )
            if action == "break":
                self.exit()
        finally:
            self._active_line_handlers = max(0, self._active_line_handlers - 1)
            self.refresh_status()

    async def action_stop_or_quit(self) -> None:
        ask_cancelled = self._cancel_pending_ask()
        if self.system.has_current_turn:
            msg = await self.system.cancel_current_turn()
            from redlotus.cli.render import print_warning
            print_warning(msg)
        elif not ask_cancelled:
            self.exit()


async def run_textual_tui(system: Any, *, stop_event: asyncio.Event | None = None) -> None:
    app = RedLotusTui(system, stop_event=stop_event)
    try:
        await app.run_async()
    finally:
        app._stop_panel_timer()
        app._cancel_pending_ask()
        set_output_sink(None)
        system.set_ask_user_handler(None)
        try:
            system.review_store.deactivate()
        except Exception:
            pass
        await system.shutdown()
