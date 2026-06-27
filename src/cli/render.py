"""Rich 终端输出封装。"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import Any

from infra import logger
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from cli.output import (
    append_model_stream_delta,
    begin_model_stream,
    clear_model_stream,
    emit_renderable,
    emit_rule,
    status_message,
    supports_model_stream,
)
from cli.diff_view import compute_line_diff, diff_stats, format_diff_text, render_diff

if sys.platform == "win32":
    for _s in (sys.stdout, sys.stderr):
        if hasattr(_s, "reconfigure"):
            try:
                _s.reconfigure(encoding="utf-8")
            except Exception:
                pass

class OutputConsoleProxy:
    def print(self, *objects: Any, **_: Any) -> None:
        for obj in objects:
            emit_renderable(obj)


console = OutputConsoleProxy()


def print_error(message: str) -> None:
    emit_renderable(Text(f"Error: {message}", style="bold red"))


def print_warning(message: str) -> None:
    emit_renderable(Text(f"Warning: {message}", style="yellow"))


def print_success(message: str) -> None:
    emit_renderable(Text(message, style="green"))


def print_markdown(text: str) -> None:
    body = (text or "").strip()
    if not body:
        return
    emit_renderable(Markdown(body))


def print_panel(content: str, *, title: str = "") -> None:
    emit_renderable(Panel(content, title=title or None))


def print_markdown_panel(text: str, *, title: str = "") -> None:
    body = (text or "").strip()
    if not body:
        return
    emit_renderable(Panel(Markdown(body), title=title or None, border_style="cyan"))


def show_file_diff(old: str, new: str, *, path: str) -> tuple[int, int, int]:
    """在 CLI 打印 old→new 的彩色 diff（增绿/删红/改蓝）；无变化则不输出。返回 (增, 删, 改)。"""
    lines = compute_line_diff(old, new)
    stats = diff_stats(lines)
    if stats == (0, 0, 0):
        return stats
    emit_renderable(render_diff(lines, path=path, stats=stats))
    logger.info_file_only("[diff] %s\n%s", path, format_diff_text(lines, path=path, stats=stats))
    return stats


def show_model_output(text: str, *, title: str = "模型", markdown: bool = True) -> None:
    """Rich 渲染模型输出；原文仅写入日志文件。纯文本汇总请设 markdown=False 以保留换行。"""
    body = (text or "").strip()
    if not body:
        return
    content: str | Markdown = Markdown(body) if markdown else body
    emit_renderable(Panel(content, title=title, border_style="cyan"))
    logger.info_file_only("[模型]\n%s", body)


def finish_model_stream(text: str, *, title: str = "模型", markdown: bool = True) -> None:
    clear_model_stream()
    show_model_output(text, title=title, markdown=markdown)


def _text_from_stream_event(event: Any) -> str:
    event_kind = getattr(event, "event_kind", "")
    if event_kind == "part_start":
        part = getattr(event, "part", None)
        if getattr(part, "part_kind", None) == "text":
            return getattr(part, "content", "") or ""
    if event_kind == "part_delta":
        delta = getattr(event, "delta", None)
        if getattr(delta, "part_delta_kind", None) == "text":
            return getattr(delta, "content_delta", "") or ""
    return ""


class TextEventStreamHandler:
    def __init__(self, *, title: str) -> None:
        self.title = title
        self._chunks: list[str] = []
        self._started = False

    async def __call__(self, _run_ctx: Any, event_stream: Any) -> None:
        try:
            async for event in event_stream:
                text = _text_from_stream_event(event)
                if not text:
                    continue
                if not self._started:
                    begin_model_stream(f"{self.title} 正在回复")
                    self._started = True
                self._chunks.append(text)
                append_model_stream_delta(text)
        except BaseException:
            if self._started:
                clear_model_stream()
                self._started = False
            raise


@contextmanager
def model_generating_indicator():
    """模型回复生成中的 spinner 提示。"""
    with status_message("模型回复生成中..."):
        yield


def print_phase(title: str) -> None:
    emit_rule(f"[dim]{title}[/dim]")
    logger.info_file_only(title)


def print_repl_welcome() -> None:
    print_panel(
        "输入 /help 查看斜杠命令；@文件路径 引用文本\n"
        "新任务 或 /clear 开启新对话 · /exit 或 quit 退出\n"
        "任务执行中按 Ctrl+C 中断",
        title="RedLotus CLI",
    )


async def consume_stream_markdown(
    stream: Any,
    history: Any,
    *,
    title: str = "模型",
) -> str:
    """
    流式接收模型输出：TUI 下走实时预览（边收边显），完成后一次性 Markdown 渲染；
    Legacy/非 TTY 下仅显示 spinner，完成后一次性渲染。
    不使用 Live（避免 Windows 下光标控制序列乱码）。
    """
    chunks: list[str] = []
    if supports_model_stream():
        begin_model_stream(title)
        try:
            async for chunk in stream.stream_text(delta=True):
                if chunk:
                    chunks.append(chunk)
                    append_model_stream_delta(chunk)
        except BaseException:
            clear_model_stream()
            raise
        final_text = "".join(chunks)
        if final_text.strip():
            finish_model_stream(final_text, title=title)
        else:
            clear_model_stream()
    else:
        with model_generating_indicator():
            async for chunk in stream.stream_text(delta=True):
                if chunk:
                    chunks.append(chunk)
        final_text = "".join(chunks)
        if final_text.strip():
            show_model_output(final_text, title=title)
    history.update(stream)
    return final_text
