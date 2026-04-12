from dataclasses import dataclass, field
import os
import re
import mimetypes
from pydantic_ai import BinaryContent

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

    def build_prompt(self, template: str):
        """用 template 替换文本部分，保留附件。"""
        if not self.attachments:
            return template
        return [template, *self.attachments]

    @staticmethod
    def _read_file_to_binary(path: str) -> BinaryContent | None:
        """读取本地文件，返回 BinaryContent（pydantic-ai 会自动 base64 编码）。"""
        if not os.path.isfile(path):
            return None
        mime, _ = mimetypes.guess_type(path)
        if mime is None:
            mime = 'image/png'
        with open(path, 'rb') as f:
            data = f.read()
        return BinaryContent(data=data, media_type=mime)

    @staticmethod
    def from_text(text: str) -> "UserMessage":
        return UserMessage(text=text)

    @classmethod
    def from_cli_input(cls, raw_input: str) -> "UserMessage":
        """解析命令行输入：用正则提取图片/视频文件路径，读取字节作为附件。"""
        attachments = []
        text = raw_input

        for match in cls._MEDIA_EXT_PATTERN.finditer(raw_input):
            path = match.group()
            bc = cls._read_file_to_binary(path)
            if bc:
                attachments.append(bc)
                text = text.replace(path, '')

        text = text.strip()
        if not text and attachments:
            text = "请分析这些内容。"
        return UserMessage(text=text, attachments=attachments)


class ChatHistory:
    """通用对话历史管理器，可被任意 Agent 组件复用。

    封装了 pydantic-ai 中反复出现的 message_history 读写模式：
        result = agent.run(prompt, message_history=history.messages)
        history.update(result)
    """

    __slots__ = ("_messages",)

    def __init__(self):
        self._messages: list = []

    def update(self, result) -> None:
        """从 RunResult / StreamedRunResult 提取完整消息列表并保存。"""
        self._messages = list(result.all_messages())

    def reset(self) -> None:
        self._messages = []

    @property
    def messages(self) -> list:
        """传入 agent.run(message_history=...) 的只读引用。"""
        return self._messages

    def __len__(self) -> int:
        return len(self._messages)

    def __bool__(self) -> bool:
        return bool(self._messages)
