from __future__ import annotations

from dataclasses import dataclass, field
import os
import re
import mimetypes
from typing import TYPE_CHECKING, Any

from pydantic_ai import BinaryContent
from pydantic_ai.messages import ModelMessagesTypeAdapter

if TYPE_CHECKING:
    from RAG.RAG import RAG as RAGEngine

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

    @classmethod
    def from_model_messages_json(cls, data: list[Any]) -> ChatHistory:
        """由 conversation_log 保存的 model_messages 字段恢复可继续对话的历史。"""
        h = cls()
        h._messages = list(ModelMessagesTypeAdapter.validate_python(data))
        return h

    @property
    def messages(self) -> list:
        """传入 agent.run(message_history=...) 的只读引用。"""
        return self._messages

    def __len__(self) -> int:
        return len(self._messages)

    def __bool__(self) -> bool:
        return bool(self._messages)


class Retriever:
    """RAG 入口：在送入 Agent 前对用户文本做向量检索与上下文拼接。"""

    def __init__(self, history: ChatHistory, rag: RAGEngine | None = None):
        self._history = history
        self._rag = rag

    def _get_rag(self) -> RAGEngine:
        if self._rag is None:
            from RAG.RAG import RAG as RAGEngineCls

            self._rag = RAGEngineCls()
        return self._rag

    async def run(self, prompt: UserMessage) -> UserMessage:
        query = (prompt.text or "").strip()
        if not query:
            return prompt

        rag = self._get_rag()
        await rag.connect()
        hits = await rag.retrieve(query)
        if not hits:
            return prompt

        ctx = "\n\n".join(f"[{i + 1}] {h.get('text', '')}" for i, h in enumerate(hits))
        augmented = (
            "以下为检索到的参考资料（按相关性排序），请结合其作答；若无相关可忽略。\n\n"
            f"{ctx}\n\n---\n用户问题：\n{query}"
        )
        return UserMessage(text=augmented, attachments=list(prompt.attachments))
