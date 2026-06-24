import asyncio
import base64
from functools import partial

from pydantic_ai import BinaryContent, ImageUrl
from wechatbot import WeChatBot

from infra import logger
from agent_core.input_messages import UserMessage
from base import BotBase

class WeChatAgentBot(BotBase):
    _ENV_AGENT_TIMEOUT = "WECHAT_AGENT_TIMEOUT_S"
    _ENV_SEND_TIMEOUT = "WECHAT_SEND_REPLY_TIMEOUT_S"
    _MIME_MAP = {
        "image": "image/jpeg",
        "voice": "audio/mpeg",
        "video": "video/mp4",
        "file": "application/octet-stream",
    }

    @property
    def platform_tag(self) -> str:
        return "WeChat"

    @property
    def session_prefix(self) -> str:
        return "wx_"

    async def _build_user_message(self, bot: WeChatBot, msg) -> UserMessage:
        """Build a UserMessage from text plus downloaded media bytes."""
        text = self.clean_text(msg.text or "")
        attachments: list = []
        try:
            media = await bot.download(msg)
        except Exception as e:
            logger.warning(f"[WeChat] 下载媒体失败: {e}")
            media = None
        if media is not None and getattr(media, "data", None):
            mime = self.guess_download_mime(
                filename=getattr(media, "file_name", None) or "",
                media_type_key=(getattr(media, "type", None) or "").lower(),
            )
            filename = getattr(media, "file_name", None) or ""
            mtype = (getattr(media, "type", None) or "").lower()
            if mtype == "image":
                b64 = base64.standard_b64encode(media.data).decode("ascii")
                attachments.append(ImageUrl(url=f"data:{mime};base64,{b64}"))
            else:
                text, consumed = await self._inline_pdf_bytes(
                    text,
                    media.data,
                    media_type=mime,
                    filename=filename or None,
                )
                if not consumed:
                    attachments.append(BinaryContent(data=media.data, media_type=mime))
        return UserMessage(text=text, attachments=attachments)

    async def _handle_message(self, bot: WeChatBot, msg) -> None:
        if not msg.user_id:
            return
        session_id = f"{self.session_prefix}{msg.user_id}"
        user_message = await self._build_user_message(bot, msg)
        await self.dispatch_user_message(
            session_id,
            user_message,
            partial(bot.reply, msg),
        )

    async def _async_main(self) -> None:
        self._released = False
        kwargs: dict = {
            "on_qr_url": lambda url: logger.info(f"[WeChat] 请扫码登录: {url}"),
            "on_scanned": lambda: logger.info("[WeChat] 已扫码，确认登录中..."),
            "on_expired": lambda: logger.warning("[WeChat] 登录二维码已过期"),
            "on_error": lambda err: logger.error(f"[WeChat] SDK 错误: {err}"),
        }

        bot = WeChatBot(**kwargs)
        await bot.login()
        bot.on_message(partial(self._handle_message, bot))
        try:
            await bot.start()
        finally:
            await self.release_all_resources_async()
            try:
                bot.stop()
            except Exception as e:
                logger.debug("[WeChat] bot.stop 失败: %s", e)

    def run(self) -> None:
        asyncio.run(self._async_main())


if __name__ == "__main__":
    WeChatAgentBot().run()
