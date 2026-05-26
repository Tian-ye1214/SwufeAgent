"""CLI 展示层兼容入口（Rich + 历史落盘格式）。"""

from __future__ import annotations

from typing import Any

from cli.render import consume_stream_markdown, print_startup_banner
from tools.memory import UserMessage

# 向后兼容：agent_app 仍从此处导入
print_startup_logo = print_startup_banner
consume_stream_text_to_stdout = consume_stream_markdown


def format_user_log_text(message: UserMessage) -> str:
    """供任务 .log 落盘：用户可见文本 + 附件说明。"""
    t = (message.text or "").strip()
    n = len(message.attachments or [])
    if n and t:
        return f"{t}\n（含 {n} 个多媒体附件）"
    if n:
        return f"（仅 {n} 个多媒体附件，无文本）"
    return t or "（空文本）"
