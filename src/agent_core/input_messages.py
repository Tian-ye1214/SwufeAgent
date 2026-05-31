from __future__ import annotations

import mimetypes
import re
from dataclasses import dataclass, field

from pydantic_ai import BinaryContent

from path_sandbox import resolve_readable_path, runtime_repo_root, work_database_root

_WORK_DATABASE_ROOT = work_database_root()


@dataclass
class UserMessage:
    """User prompt text plus optional pydantic-AI multimodal content."""

    _MEDIA_EXT_PATTERN = re.compile(
        r"[a-zA-Z0-9_\-./\\:]+\.(?:jpg|jpeg|png|gif|webp|bmp|mp4|avi|mov|mkv|webm)",
        re.IGNORECASE,
    )

    text: str
    attachments: list = field(default_factory=list)

    def to_prompt(self):
        """Return the shape accepted by ``Agent.run``."""
        if not self.attachments:
            return self.text
        return [self.text, *self.attachments]

    def _read_file_to_binary(self, path: str) -> BinaryContent | None:
        try:
            resolved = resolve_readable_path(
                path, work_base=_WORK_DATABASE_ROOT, repo_root=runtime_repo_root()
            )
        except ValueError:
            return None
        if not resolved.is_file():
            return None
        mime, _ = mimetypes.guess_type(str(resolved))
        if mime is None:
            mime = "image/png"
        return BinaryContent(data=resolved.read_bytes(), media_type=mime)


def user_message_from_text(message: str | UserMessage) -> UserMessage:
    if isinstance(message, UserMessage):
        return message
    return UserMessage(text=str(message))


def user_message_from_cli_input(raw_input: str) -> UserMessage:
    attachments: list = []
    text = raw_input
    um = UserMessage(text="", attachments=[])

    for match in UserMessage._MEDIA_EXT_PATTERN.finditer(raw_input):
        path = match.group()
        bc = um._read_file_to_binary(path)
        if bc:
            attachments.append(bc)
            text = text.replace(path, "")

    text = text.strip()
    if not text and attachments:
        text = "请分析这些内容。"
    return UserMessage(text=text, attachments=attachments)
