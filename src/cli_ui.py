"""CLI 启动 Logo 与日志文本格式化。"""

from __future__ import annotations

import sys

from cli.output import emit_text
from tools.memory import UserMessage


def print_startup_logo() -> None:
    stdout_encoding = (getattr(sys.stdout, "encoding", None) or "utf-8").lower()
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

    emit_text("")
    for row in canvas:
        line_text = ""
        for ch, color in row:
            if color is None:
                line_text += ch
            else:
                line_text += color_text(color[0], color[1], color[2], ch)
        emit_text(line_text)

    if unicode_safe:
        tagline = "❦ ────  红莲极意  ·  RedLotus Agent  ──── ❦"
    else:
        tagline = "<>----  RedLotus Agent  ----<>"
    pad = max(0, (max_width + shadow_offset_col - len(tagline)) // 2)
    emit_text(" " * pad + color_text(255, 165, 90, tagline))
    emit_text("")


def format_user_log_text(message: UserMessage) -> str:
    """供任务 .log 落盘：用户可见文本 + 附件说明。"""
    t = (message.text or "").strip()
    n = len(message.attachments or [])
    if n and t:
        return f"{t}\n（含 {n} 个多媒体附件）"
    if n:
        return f"（仅 {n} 个多媒体附件，无文本）"
    return t or "（空文本）"
