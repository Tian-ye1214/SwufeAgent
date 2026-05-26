"""Rich 终端输出封装。"""

from __future__ import annotations

import sys
from typing import Any

import logger
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

_IS_WINDOWS = sys.platform == "win32"
if _IS_WINDOWS:
    for _s in (sys.stdout, sys.stderr):
        if hasattr(_s, "reconfigure"):
            try:
                _s.reconfigure(encoding="utf-8")
            except Exception:
                pass

console = Console(highlight=False, legacy_windows=_IS_WINDOWS)


def print_error(message: str) -> None:
    console.print(Text(f"Error: {message}", style="bold red"))


def print_warning(message: str) -> None:
    console.print(Text(f"Warning: {message}", style="yellow"))


def print_success(message: str) -> None:
    console.print(Text(message, style="green"))


def print_markdown(text: str) -> None:
    body = (text or "").strip()
    if not body:
        return
    console.print(Markdown(body))


def print_panel(content: str, *, title: str = "") -> None:
    console.print(Panel(content, title=title or None))


def print_markdown_panel(text: str, *, title: str = "") -> None:
    body = (text or "").strip()
    if not body:
        return
    console.print(Panel(Markdown(body), title=title or None, border_style="cyan"))


def show_model_output(text: str, *, title: str = "模型") -> None:
    """Rich 渲染模型 Markdown 回复；原文仅写入日志文件。"""
    body = (text or "").strip()
    if not body:
        return
    console.print(Panel(Markdown(body), title=title, border_style="cyan"))
    logger.info_file_only("[模型]\n%s", body)


def print_phase(title: str) -> None:
    console.rule(f"[dim]{title}[/dim]")
    logger.info_file_only(title)


def print_repl_welcome() -> None:
    print_panel(
        "输入 /help 查看斜杠命令；@文件路径 引用文本\n"
        "新任务 或 /clear 清除上下文 · /exit 或 quit 退出 · /stop 中断当前任务",
        title="RedLotus CLI",
    )


def print_startup_banner() -> None:
    """Rich 版启动 Logo。"""
    logo_lines = [
        "██████╗ ███████╗██████╗ ██╗      ██████╗ ████████╗██╗   ██╗███████╗",
        "██╔══██╗██╔════╝██╔══██╗██║     ██╔═══██╗╚══██╔══╝██║   ██║██╔════╝",
        "██████╔╝█████╗  ██║  ██║██║     ██║   ██║   ██║   ██║   ██║███████╗",
        "██╔══██╗██╔══╝  ██║  ██║██║     ██║   ██║   ██║   ██║   ██║╚════██║",
        "██║  ██║███████╗██████╔╝███████╗╚██████╔╝   ██║   ╚██████╔╝███████║",
        "╚═╝  ╚═╝╚══════╝╚═════╝ ╚══════╝ ╚═════╝    ╚═╝    ╚═════╝ ╚══════╝",
    ]
    max_width = max(len(line) for line in logo_lines)
    start = (255, 80, 60)
    end = (140, 50, 130)

    lines: list[Text] = []
    for row in logo_lines:
        t = Text()
        for col, ch in enumerate(row):
            if ch == " ":
                t.append(" ")
                continue
            ratio = col / (max_width - 1) if max_width > 1 else 0
            r = int(start[0] + (end[0] - start[0]) * ratio)
            g = int(start[1] + (end[1] - start[1]) * ratio)
            b = int(start[2] + (end[2] - start[2]) * ratio)
            t.append(ch, style=f"rgb({r},{g},{b})")
        lines.append(t)

    tagline = Text("❦ ────  红莲极意  ·  RedLotus Agent  ──── ❦", style="rgb(255,165,90)")
    console.print()
    console.print(Group(*lines))
    console.print(tagline, justify="center")
    console.print()


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
    with console.status("[dim]模型回复生成中…[/dim]", spinner="dots"):
        async for chunk in stream.stream_text(delta=True):
            if chunk:
                chunks.append(chunk)
    final_text = "".join(chunks)
    if final_text.strip():
        show_model_output(final_text, title=title)
    history.update(stream)
    return final_text


consume_stream_text = consume_stream_markdown
