from __future__ import annotations

import sys
from typing import Any

from tools.Memory import ChatHistory, UserMessage


def print_startup_logo() -> None:
    stdout_encoding = (sys.stdout.encoding or "").lower()
    unicode_safe = "utf" in stdout_encoding

    def color_text(r: int, g: int, b: int, text: str) -> str:
        return f"\033[38;2;{r};{g};{b}m{text}\033[0m"

    logo_lines = [
        "██████╗ ███████╗██████╗ ██╗      ██████╗ ████████╗██╗   ██╗███████╗",
        "██╔══██╗██╔════╝██╔══██╗██║     ██╔═══██╗╚══██╔══╝██║   ██║██╔════╝",
        "██████╔╝█████╗  ██║  ██║██║     ██║   ██║   ██║   ██║   ██║███████╗",
        "██╔══██╗██╔══╝  ██║  ██║██║     ██║   ██║   ██║   ██║   ██║╚════██║",
        "██║  ██║███████╗██████╔╝███████╗╚██████╔╝   ██║   ╚██████╔╝███████║",
        "╚═╝  ╚═╝╚══════╝╚═════╝ ╚══════╝ ╚═════╝    ╚═╝    ╚═════╝ ╚══════╝",
    ]
    shadow_char = "░" if unicode_safe else "."
    shadow_offset_row = 1
    shadow_offset_col = 2
    shadow_color = (70, 40, 60)
    start_color = (255, 80, 60)
    end_color = (140, 50, 130)

    max_width = max(len(line) for line in logo_lines)
    total_rows = len(logo_lines)
    canvas_rows = total_rows + shadow_offset_row
    canvas_cols = max_width + shadow_offset_col
    canvas = [[(" ", None) for _ in range(canvas_cols)] for _ in range(canvas_rows)]

    for row_idx, line in enumerate(logo_lines):
        for col_idx, ch in enumerate(line):
            if ch == " ":
                continue
            ratio = col_idx / (max_width - 1) if max_width > 1 else 0
            r = int(start_color[0] + (end_color[0] - start_color[0]) * ratio)
            g = int(start_color[1] + (end_color[1] - start_color[1]) * ratio)
            b = int(start_color[2] + (end_color[2] - start_color[2]) * ratio)
            canvas[row_idx][col_idx] = (ch, (r, g, b))

    for row_idx in range(total_rows):
        for col_idx in range(max_width):
            original_char = (
                logo_lines[row_idx][col_idx] if col_idx < len(logo_lines[row_idx]) else " "
            )
            if original_char == " ":
                continue
            shadow_row = row_idx + shadow_offset_row
            shadow_col = col_idx + shadow_offset_col
            if (
                0 <= shadow_row < canvas_rows
                and 0 <= shadow_col < canvas_cols
                and canvas[shadow_row][shadow_col][0] == " "
            ):
                canvas[shadow_row][shadow_col] = (shadow_char, shadow_color)

    print()
    for row in canvas:
        line_text = ""
        for ch, color in row:
            if color is None:
                line_text += ch
            else:
                line_text += color_text(color[0], color[1], color[2], ch)
        print(line_text)

    if unicode_safe:
        tagline = "❦ ────  红莲极意  ·  RedLotus Agent  ──── ❦"
    else:
        tagline = "<>----  RedLotus Agent  ----<>"
    pad = max(0, (max_width + shadow_offset_col - len(tagline)) // 2)
    print(" " * pad + color_text(255, 165, 90, tagline))
    print()


def format_user_log_text(message: UserMessage) -> str:
    """供任务 .log 落盘：用户可见文本 + 附件说明。"""
    t = (message.text or "").strip()
    n = len(message.attachments or [])
    if n and t:
        return f"{t}\n（含 {n} 个多媒体附件）"
    if n:
        return f"（仅 {n} 个多媒体附件，无文本）"
    return t or "（空文本）"


async def consume_stream_text_to_stdout(stream: Any, history: ChatHistory) -> str:
    """边打印 Manager 摘要流边拼接全文，结束时写入 history。"""
    collected: list[str] = []
    render_buf = ""
    async for chunk in stream.stream_text(delta=True):
        if not chunk:
            continue
        collected.append(chunk)
        render_buf += chunk
        while "\n" in render_buf:
            line, render_buf = render_buf.split("\n", 1)
            print(line)
        if len(render_buf) >= 240:
            print(render_buf, end="", flush=True)
            render_buf = ""
    if render_buf:
        print(render_buf, end="", flush=True)
    print(flush=True)
    final_text = "".join(collected)
    history.update(stream)
    return final_text
