import re
import asyncio
import inspect
from functools import partial

from ncatbot.core import BotClient, BaseMessageEvent, GroupMessageEvent, MetaEvent
from ncatbot.plugin_system import on_message
from ncatbot.utils import config
from pydantic_ai import BinaryContent

import logger
from app_config import get_env
from base import BotBase
from qq_media_helpers import extract_media
from tools.ExtractFileContent import extract_text_from_pdf_bytes

_u = get_env("QQBOT_ID", warn=False)
config.set_bot_uin(_u if _u else None)

_FILE_ALLOW_EXT = frozenset(
    {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".mp4", ".mov", ".mkv", ".webm",
     ".pdf", ".txt", ".md", ".docx", ".xlsx", ".csv", ".json"}
)


class QQBot(BotBase):
    _ENV_AGENT_TIMEOUT = "QQ_AGENT_TIMEOUT_S"
    _ENV_SEND_TIMEOUT = "QQ_SEND_REPLY_TIMEOUT_S"

    def __init__(self):
        super().__init__()
        self._bot_client = BotClient()
        self._bot_client.add_shutdown_handler(self._on_shutdown)

    @property
    def platform_tag(self) -> str:
        return "QQ"

    @property
    def session_prefix(self) -> str:
        return "qq_"

    def clean_text(self, raw: str) -> str:
        return re.sub(r"\[CQ:[^\]]+\]", "", raw or "").strip()

    def _is_at_me(self, event: BaseMessageEvent) -> bool:
        if not isinstance(event, GroupMessageEvent):
            return True
        msg = getattr(event, "message", None)
        return msg is not None and msg.is_user_at(config.bt_uin)

    def _session_id(self, event: BaseMessageEvent) -> str:
        if isinstance(event, GroupMessageEvent):
            return f"group_{event.group_id}"
        return f"private_{event.user_id}"

    async def _reply_event(self, event: BaseMessageEvent, text: str) -> None:
        sig = inspect.signature(event.reply)
        await (event.reply(text=text, at=False) if "at" in sig.parameters else event.reply(text=text))

    async def _extract_attachments(self, event: BaseMessageEvent) -> list:
        return await extract_media(self._bot_client.api, event, _FILE_ALLOW_EXT)

    def _is_pdf_binary(self, att: BinaryContent) -> bool:
        mt = (att.media_type or "").split(";")[0].strip().lower()
        if mt in ("application/pdf", "application/x-pdf"):
            return True
        data = att.data or b""
        return len(data) >= 4 and data[:4] == b"%PDF"

    async def _inline_pdf_attachments(self, user_text: str, attachments: list) -> tuple[str, list]:
        """将 PDF 的 BinaryContent 解析为正文，其余附件原样保留。"""
        out: list = []
        blocks: list[str] = []
        for att in attachments:
            if not isinstance(att, BinaryContent) or not self._is_pdf_binary(att):
                out.append(att)
                continue
            extracted = await asyncio.to_thread(extract_text_from_pdf_bytes, att.data)
            if extracted:
                blocks.append(f"【PDF 附件】\n\n{extracted}")
            else:
                logger.warning("[QQ] PDF 解析失败，未注入文本")
                blocks.append("（PDF 附件无法解析为文本，请尝试发送截图或纯文本。）")
        text = user_text
        if blocks:
            body = "\n\n".join(blocks)
            text = f"{text}\n\n{body}".strip() if text else body
        return text, out

    async def _on_shutdown(self, _: MetaEvent) -> None:
        await self.release_all_resources_async()

    async def _handle_message(self, event: BaseMessageEvent) -> None:
        raw_text = (event.raw_message or "").strip()
        if not raw_text or (isinstance(event, GroupMessageEvent) and not self._is_at_me(event)):
            return
        session_id = self._session_id(event)
        user_text = self.clean_text(raw_text)
        attachments = await self._extract_attachments(event)
        user_text, attachments = await self._inline_pdf_attachments(user_text, attachments)
        await self.dispatch_message(
            session_id,
            user_text,
            attachments,
            partial(self._reply_event, event),
            asyncio.get_running_loop(),
        )

    def _register_handlers(self) -> None:
        on_message(self._handle_message)

    def run(self, *, debug: bool = True, remote_mode: bool = True,
            enable_webui_interaction: bool = False, **kwargs):
        self._released = False
        self._register_handlers()
        try:
            self._bot_client.run_frontend(
                debug=debug,
                remote_mode=remote_mode,
                enable_webui_interaction=enable_webui_interaction,
                **kwargs,
            )
        finally:
            self.release_all_resources()


if __name__ == "__main__":
    QQBot().run()
