from __future__ import annotations

import mimetypes
import re
from dataclasses import dataclass, field, replace
from typing import Any

from pydantic_ai import BinaryContent, ImageUrl
from pydantic_ai.messages import BaseToolReturnPart, ModelRequest, UserPromptPart

from redlotus.infra.path_sandbox import resolve_readable_path, work_database_root

_WORK_DATABASE_ROOT = work_database_root()
_IMAGE_OMITTED = "[image omitted: model does not support image input]"


def _is_image_content(value: Any) -> bool:
    if isinstance(value, ImageUrl):
        return True
    if isinstance(value, BinaryContent):
        media_type = str(value.media_type or "").split(";", 1)[0].lower()
        return media_type.startswith("image/")
    return False


def _filter_content_for_input_modalities(content: Any, allowed_modalities: set[str]) -> tuple[Any, bool]:
    if "image" in allowed_modalities:
        return content, False
    if _is_image_content(content):
        return _IMAGE_OMITTED, True
    if isinstance(content, (list, tuple)):
        changed = False
        out: list[Any] = []
        for item in content:
            filtered, item_changed = _filter_content_for_input_modalities(
                item, allowed_modalities
            )
            out.append(filtered)
            changed = changed or item_changed
        return (out, True) if changed else (content, False)
    return content, False


def filter_messages_for_input_modalities(messages: list, allowed_modalities: set[str]) -> list:
    allowed = {str(item).lower() for item in allowed_modalities}
    out: list[Any] = []
    changed = False
    for message in messages:
        if not isinstance(message, ModelRequest):
            out.append(message)
            continue
        parts = []
        message_changed = False
        for part in message.parts:
            if isinstance(part, (UserPromptPart, BaseToolReturnPart)):
                filtered, part_changed = _filter_content_for_input_modalities(part.content, allowed)
                parts.append(replace(part, content=filtered) if part_changed else part)
                message_changed = message_changed or part_changed
            else:
                parts.append(part)
        out.append(replace(message, parts=parts) if message_changed else message)
        changed = changed or message_changed
    return out if changed else list(messages)


@dataclass
class UserMessage:
    """User prompt text plus optional pydantic-AI multimodal content."""

    _MEDIA_EXT_PATTERN = re.compile(
        r"[a-zA-Z0-9_\-./\\:]+\.(?:jpg|jpeg|png|gif|webp|bmp|mp4|avi|mov|mkv|webm)",
        re.IGNORECASE,
    )

    text: str
    attachments: list = field(default_factory=list)

    def to_prompt(self, *, include_attachments: bool = True):
        """Return the shape accepted by ``Agent.run``."""
        if not include_attachments:
            return self.text or (_IMAGE_OMITTED if self.attachments else "")
        if not self.attachments:
            return self.text
        return [self.text, *self.attachments]

    def _read_file_to_binary(self, path: str) -> BinaryContent | None:
        try:
            resolved = resolve_readable_path(path, work_base=_WORK_DATABASE_ROOT)
        except ValueError:
            return None
        if not resolved.is_file():
            return None
        mime, _ = mimetypes.guess_type(str(resolved))
        if mime is None:
            mime = "image/png"
        return BinaryContent(data=resolved.read_bytes(), media_type=mime)


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
