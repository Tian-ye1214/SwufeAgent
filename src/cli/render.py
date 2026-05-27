"""Rich 终端输出封装。"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import Any

import logger
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from cli.output import emit_renderable, emit_rule, status_message

_IS_WINDOWS = sys.platform == "win32"
if _IS_WINDOWS:
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

    def rule(self, title: str = "", **_: Any) -> None:
        emit_rule(title)

    def status(self, message: str, **_: Any):
        return status_message(message)


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


def show_model_output(text: str, *, title: str = "模型", markdown: bool = True) -> None:
    """Rich 渲染模型输出；原文仅写入日志文件。纯文本汇总请设 markdown=False 以保留换行。"""
    body = (text or "").strip()
    if not body:
        return
    content: str | Markdown = Markdown(body) if markdown else body
    emit_renderable(Panel(content, title=title, border_style="cyan"))
    logger.info_file_only("[模型]\n%s", body)


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
        "新任务 或 /clear 清除上下文 · /exit 或 quit 退出\n"
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
    流式接收模型输出：生成中显示 spinner，完成后一次性 Markdown 渲染。
    不使用 Live（避免 Windows 下光标控制序列乱码）。
    """
    chunks: list[str] = []
    with model_generating_indicator():
        async for chunk in stream.stream_text(delta=True):
            if chunk:
                chunks.append(chunk)
    final_text = "".join(chunks)
    if final_text.strip():
        show_model_output(final_text, title=title)
    history.update(stream)
    return final_text
