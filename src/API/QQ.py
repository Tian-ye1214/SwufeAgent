import inspect
import re
from functools import partial

from ncatbot.core import BaseMessageEvent, BotClient, GroupMessageEvent, MetaEvent
from ncatbot.plugin_system import on_message
from ncatbot.utils import config

from infra import logger
from config.app_config import get_env
from agent_core.input_messages import UserMessage
from base import BotBase
from qq_media_helpers import extract_media

class QQBot(BotBase):
    _ENV_AGENT_TIMEOUT = "QQ_AGENT_TIMEOUT_S"
    _ENV_SEND_TIMEOUT = "QQ_SEND_REPLY_TIMEOUT_S"
    _FILE_ALLOW_EXT = frozenset({
        ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".mp4", ".mov", ".mkv", ".webm",
        ".pdf", ".txt", ".md", ".docx", ".xlsx", ".csv", ".json",
    })

    def __init__(self):
        super().__init__()
        config.set_bot_uin(get_env("QQBOT_ID", warn=False) or None)
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
        await (
            event.reply(text=text, at=False)
            if "at" in sig.parameters
            else event.reply(text=text)
        )

    async def _extract_attachments(self, event: BaseMessageEvent) -> list:
        return await extract_media(self._bot_client.api, event, self._FILE_ALLOW_EXT)

    async def _on_shutdown(self, _: MetaEvent) -> None:
        await self.release_all_resources_async()

    async def _handle_message(self, event: BaseMessageEvent) -> None:
        raw_text = (event.raw_message or "").strip()
        if not raw_text or (isinstance(event, GroupMessageEvent) and not self._is_at_me(event)):
            return
        session_id = self._session_id(event)
        user_text = self.clean_text(raw_text)
        attachments = await self._extract_attachments(event)
        user_text, attachments = await self._partition_pdf_attachments(user_text, attachments)
        await self.dispatch_user_message(
            session_id,
            UserMessage(text=user_text, attachments=attachments),
            partial(self._reply_event, event),
        )

    def _register_handlers(self) -> None:
        on_message(self._handle_message)

    def run(
        self,
        *,
        debug: bool = True,
        remote_mode: bool = True,
        enable_webui_interaction: bool = False,
        **kwargs,
    ):
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
