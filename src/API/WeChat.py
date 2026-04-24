import asyncio
import base64
import mimetypes
from functools import partial

from pydantic_ai import BinaryContent, ImageUrl
from wechatbot import WeChatBot

import logger
from base import BotBase
from tools.ExtractFileContent import extract_text_from_pdf_bytes
from tools.Memory import UserMessage

_MIME_MAP = {
    "image": "image/jpeg",
    "voice": "audio/mpeg",
    "video": "video/mp4",
    "file": "application/octet-stream",
}


class WeChatAgentBot(BotBase):
    _ENV_AGENT_TIMEOUT = "WECHAT_AGENT_TIMEOUT_S"
    _ENV_SEND_TIMEOUT = "WECHAT_SEND_REPLY_TIMEOUT_S"

    @property
    def platform_tag(self) -> str:
        return "WeChat"

    @property
    def session_prefix(self) -> str:
        return "wx_"

    def _mime_for_downloaded(self, media) -> str:
        if media.file_name:
            mime, _ = mimetypes.guess_type(media.file_name)
            if mime:
                return mime
        return _MIME_MAP.get((media.type or "").lower(), "application/octet-stream")

    def _is_pdf_media(self, media, mime: str) -> bool:
        if mime in ("application/pdf", "application/x-pdf"):
            return True
        fn = (getattr(media, "file_name", None) or "").lower()
        return fn.endswith(".pdf")

    async def _build_user_message(self, bot: WeChatBot, msg) -> UserMessage:
        """文本 + 下载得到的媒体字节 → UserMessage（后续 to_prompt 即模型输入）。"""
        text = (msg.text or "").strip()
        attachments: list = []
        try:
            media = await bot.download(msg)
        except Exception as e:
            logger.warning(f"[WeChat] 下载媒体失败: {e}")
            media = None
        if media is not None and getattr(media, "data", None):
            mime = self._mime_for_downloaded(media)
            mtype = (getattr(media, "type", None) or "").lower()
            if mtype == "image":
                b64 = base64.standard_b64encode(media.data).decode("ascii")
                attachments.append(ImageUrl(url=f"data:{mime};base64,{b64}"))
            elif self._is_pdf_media(media, mime):
                extracted = await asyncio.to_thread(extract_text_from_pdf_bytes, media.data)
                label = (
                    f"【PDF 附件：{media.file_name}】\n\n"
                    if getattr(media, "file_name", None)
                    else "【PDF 附件】\n\n"
                )
                if extracted:
                    body = f"{label}{extracted}"
                    text = f"{text}\n\n{body}".strip() if text else body
                else:
                    logger.warning("[WeChat] PDF 解析失败，未注入文本")
                    hint = "（PDF 附件无法解析为文本，请尝试发送截图或纯文本。）"
                    text = f"{text}\n\n{hint}".strip() if text else hint
            else:
                attachments.append(BinaryContent(data=media.data, media_type=mime))
        return UserMessage(text=text, attachments=attachments)

    async def _handle_message(self, bot: WeChatBot, msg) -> None:
        if not msg.user_id:
            return
        session_id = f"{self.session_prefix}{msg.user_id}"
        user_message = await self._build_user_message(bot, msg)
        await self.dispatch_message(
            session_id,
            self.clean_text(user_message.text),
            user_message.attachments,
            partial(bot.reply, msg),
            asyncio.get_running_loop(),
        )

    async def _async_main(self) -> None:
        self._released = False
        kwargs: dict = {
            "on_qr_url": lambda url: logger.info(f"[WeChat] 请扫码登录: {url}"),
            "on_scanned": lambda: logger.info("[WeChat] 已扫码，确认登录中…"),
            "on_expired": lambda: logger.warning("[WeChat] 登录二维码已过期"),
            "on_error": lambda err: logger.error(f"[WeChat] SDK 错误: {err}"),
        }

        bot = WeChatBot(**kwargs)
        await bot.login()
        bot.on_message(partial(self._handle_message, bot))
        try:
            await bot.start()
        finally:
            self.release_all_resources()
            try:
                bot.stop()
            except Exception:
                pass

    def run(self) -> None:
        asyncio.run(self._async_main())


if __name__ == "__main__":
    WeChatAgentBot().run()
