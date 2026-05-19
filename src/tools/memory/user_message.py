from __future__ import annotations

import mimetypes
import re
from dataclasses import dataclass, field
from pydantic_ai import BinaryContent
from path_sandbox import resolve_readable_path, runtime_repo_root

_WORK_DATABASE_ROOT = runtime_repo_root() / "WorkDatabase"


@dataclass
class UserMessage:
    """用户消息：文本 + 可选的多模态附件（图片/视频）。"""

    _MEDIA_EXT_PATTERN = re.compile(
        r'[a-zA-Z0-9_\-./\\:]+\.(?:jpg|jpeg|png|gif|webp|bmp|mp4|avi|mov|mkv|webm)',
        re.IGNORECASE,
    )

    text: str
    attachments: list = field(default_factory=list)

    def to_prompt(self):
        """转换为 pydantic-ai agent.run() 可接受的 prompt 格式。"""
        if not self.attachments:
            return self.text
        return [self.text, *self.attachments]

    def _read_file_to_binary(self, path: str) -> BinaryContent | None:
        """读取本地文件，返回 BinaryContent（pydantic-ai 会自动 base64 编码）。"""
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
        data = resolved.read_bytes()
        return BinaryContent(data=data, media_type=mime)


def user_message_from_text(message: str) -> UserMessage:
    if isinstance(message, UserMessage):
        return message
    return UserMessage(text=str(message))


def user_message_from_cli_input(raw_input: str) -> UserMessage:
    """解析命令行输入：用正则提取图片/视频文件路径，读取字节作为附件。"""
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
