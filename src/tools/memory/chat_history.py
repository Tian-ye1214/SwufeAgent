from __future__ import annotations


class ChatHistory:
    __slots__ = (
        "_messages",
        "_compress_summary_state",
        "_compress_weak_streak",
    )

    def __init__(self):
        self._messages: list = []
        self._compress_summary_state: str | None = None
        self._compress_weak_streak: int = 0

    def update(self, result) -> None:
        """从 RunResult / StreamedRunResult 提取完整消息列表并保存。"""
        self._messages = list(result.all_messages())

    def reset(self) -> None:
        self._messages = []
        self._compress_summary_state = None
        self._compress_weak_streak = 0

    def set_messages(self, messages: list) -> None:
        """直接替换消息列表（供上下文压缩等使用）。"""
        self._messages = list(messages)

    @property
    def compress_summary_state(self) -> str | None:
        """上一轮压缩模型产出的 Markdown 摘要文本，供下次压缩合并。"""
        return self._compress_summary_state

    @compress_summary_state.setter
    def compress_summary_state(self, value: str | None) -> None:
        self._compress_summary_state = value

    @property
    def messages(self) -> list:
        """传入 agent.run(message_history=...) 的只读引用。"""
        return self._messages

    def __len__(self) -> int:
        return len(self._messages)

    def __bool__(self) -> bool:
        return bool(self._messages)
