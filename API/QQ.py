import sys
import os
import re
import asyncio
import mimetypes
import threading
from concurrent.futures import ThreadPoolExecutor

import httpx
from ncatbot.core import BotClient, BaseMessageEvent, GroupMessageEvent
from ncatbot.plugin_system import on_message
from ncatbot.utils import config
from pydantic_ai import BinaryContent

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

from main import AgentSystem
from tools.memory import ChatHistory, UserMessage
from tools.ManagementTools import task_manager
from tools.BasicTools import set_task_directory, reset_task_directory, set_ask_user_handler
import logger

BOT_QQ = os.environ.get("QQBOT_ID", None)
config.set_bot_uin(os.environ.get("QQBOT_ID", None))


class QQBot:
    _MAX_LENGTH = 1800

    def __init__(self):
        self.__sessions: dict[str, ChatHistory] = {}
        self.__is_first: dict[str, bool] = {}
        self.__agent_lock = threading.Lock()
        self.__executor = ThreadPoolExecutor(max_workers=1)
        self.__pending_questions: dict[str, dict] = {}
        self.__ctx = threading.local()
        self.__bot = BotClient()
        self.__agent_system = AgentSystem()

        set_ask_user_handler(self._qq_ask_user)

    def run(self, *, debug: bool = True, remote_mode: bool = True,
            enable_webui_interaction: bool = False, **kwargs):
        """启动 QQ Bot，注册消息处理器并连接 NapCat 服务。"""
        self._register_handlers()
        self.__bot.run_frontend(
            debug=debug,
            remote_mode=remote_mode,
            enable_webui_interaction=enable_webui_interaction,
            **kwargs,
        )

    def _qq_ask_user(self, question: str) -> str:
        """
        Agent 调用 ask_user 时执行：
        1. 通过 QQ 把问题发给用户
        2. 阻塞等待用户回复（最长 120 秒）
        3. 返回用户的回答
        """
        session_id: str | None = getattr(self.__ctx, "session_id", None)
        send_func = getattr(self.__ctx, "send_func", None)
        loop: asyncio.AbstractEventLoop | None = getattr(self.__ctx, "loop", None)

        if not session_id or not send_func or not loop:
            return input(f"[ask_user] {question}\n回复: ").strip()

        event = threading.Event()
        self.__pending_questions[session_id] = {"event": event, "answer": None}

        future = asyncio.run_coroutine_threadsafe(
            send_func(f"🤔 {question}"), loop
        )
        try:
            future.result(timeout=10)
        except Exception as e:
            logger.error(f"[QQ ask_user] 发送问题失败: {e}")

        got_reply = event.wait(timeout=120)
        pending = self.__pending_questions.pop(session_id, {})

        if got_reply and pending.get("answer"):
            return pending["answer"]
        return "(用户未在 120 秒内回复，已超时)"

    def _reset_session(self, session_id: str):
        self.__sessions.pop(session_id, None)
        self.__is_first.pop(session_id, None)
        self.__pending_questions.pop(session_id, None)
        self.__agent_system.reset_manager_history()
        logger.info(f"[QQ] 会话 {session_id} 已重置")

    def _try_answer_pending(self, session_id: str, user_text: str) -> bool:
        """若 Agent 正在等待该会话的用户回复，则将本条消息作为答案。"""
        pending = self.__pending_questions.get(session_id)
        if pending and not pending["event"].is_set():
            pending["answer"] = user_text
            pending["event"].set()
            return True
        return False

    def _call_agent(
        self,
        session_id: str,
        message: UserMessage,
        send_func,
        loop: asyncio.AbstractEventLoop,
    ) -> str:
        """在独立线程中同步调用 Agent，持有全局锁保证串行。"""
        with self.__agent_lock:
            self.__ctx.session_id = session_id
            self.__ctx.send_func = send_func
            self.__ctx.loop = loop

            history = self.__sessions.setdefault(session_id, ChatHistory())

            logger.setup_session_logger(session_id)
            if not self.__is_first.get(session_id, False):
                set_task_directory(f"QQ_{session_id[:12]}")
                self.__is_first[session_id] = True

            try:
                _, output = self.__agent_system.run_agent_system(message, history)
            except Exception as e:
                logger.error(f"[QQ] Agent 调用异常: {e}")
                output = f"抱歉，处理您的请求时出现了错误：{e}"
            finally:
                self.__ctx.session_id = None
                self.__ctx.send_func = None
                self.__ctx.loop = None

        if len(output) > self._MAX_LENGTH:
            output = output[:self._MAX_LENGTH] + "\n\n…（内容过长，已截断）"
        return output

    async def _async_call_agent(
        self,
        session_id: str,
        message: UserMessage,
        send_func,
        loop: asyncio.AbstractEventLoop,
    ) -> str:
        return await loop.run_in_executor(
            self.__executor, self._call_agent, session_id, message, send_func, loop
        )

    @staticmethod
    def _download_to_binary(url: str) -> BinaryContent | None:
        """下载 URL 资源，返回 BinaryContent（base64 由 pydantic-ai 自动处理）。"""
        try:
            resp = httpx.get(url, timeout=30, follow_redirects=True)
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "").split(";")[0].strip()
            if not content_type:
                mime, _ = mimetypes.guess_type(url)
                content_type = mime or "image/png"
            return BinaryContent(data=resp.content, media_type=content_type)
        except Exception as e:
            logger.warning(f"[QQ] 下载媒体失败 {url[:80]}: {e}")
            return None

    @staticmethod
    def _extract_media(event: BaseMessageEvent) -> list:
        """从 QQ 消息 segments 中提取图片/视频，下载后转为 BinaryContent。"""
        urls: list[str] = []
        msg = getattr(event, "message", None)
        if msg and hasattr(msg, "__iter__"):
            for seg in msg:
                if isinstance(seg, dict):
                    seg_type = seg.get("type", "")
                    seg_data = seg.get("data", {})
                else:
                    seg_type = getattr(seg, "type", "")
                    seg_data = getattr(seg, "data", {})
                    if not isinstance(seg_data, dict):
                        seg_data = vars(seg_data) if hasattr(seg_data, "__dict__") else {}

                if seg_type in ("image", "video"):
                    url = seg_data.get("url") or seg_data.get("file")
                    if url and url.startswith("http"):
                        urls.append(url)

        if not urls:
            raw = getattr(event, "raw_message", "") or ""
            for m in re.finditer(r"\[CQ:(?:image|video),[^\]]*url=([^\],]+)", raw):
                urls.append(m.group(1))

        attachments = []
        for url in urls:
            bc = QQBot._download_to_binary(url)
            if bc:
                attachments.append(bc)
        return attachments

    @staticmethod
    def _clean_text(raw_message: str) -> str:
        """从 raw_message 中移除 CQ 码，提取纯文本。"""
        return re.sub(r"\[CQ:[^\]]+\]", "", raw_message).strip()

    @staticmethod
    def _is_at_me(event: BaseMessageEvent) -> bool:
        """群聊里是否 @ 了机器人（或 @ 全体）。私聊视为 True。"""
        if not isinstance(event, GroupMessageEvent):
            return True
        msg = getattr(event, "message", None)
        if msg is None:
            return False
        return msg.is_user_at(config.bt_uin)

    def _register_handlers(self):
        """注册 ncatbot 消息处理器，仅在 run() 时调用一次。"""
        bot = self

        @on_message
        async def handle_message(event: BaseMessageEvent):
            raw_text = (event.raw_message or "").strip()
            if not raw_text:
                return

            if isinstance(event, GroupMessageEvent) and not QQBot._is_at_me(event):
                return

            if isinstance(event, GroupMessageEvent):
                session_id = f"group_{event.group_id}"
            else:
                session_id = f"private_{event.user_id}"

            user_text = QQBot._clean_text(raw_text)
            attachments = QQBot._extract_media(event)

            loop = asyncio.get_event_loop()
            media_hint = f" (+{len(attachments)}个附件)" if attachments else ""
            logger.info(f"[NapCat] {session_id} 消息: {user_text[:60]}{media_hint}")

            if bot._try_answer_pending(session_id, user_text):
                return

            if user_text in ("新任务", "/reset", "/新任务"):
                bot._reset_session(session_id)
                task_manager.reset()
                reset_task_directory()
                await event.reply(text="已开始新对话，上下文已清除。", at=False)
                return

            if not user_text and not attachments:
                return

            if not user_text and attachments:
                user_text = "请分析这些内容。"

            message = UserMessage(text=user_text, attachments=attachments)

            async def send_func(text: str):
                await event.reply(text=text, at=False)

            reply = await bot._async_call_agent(
                session_id, message, send_func, loop
            )
            await event.reply(text=reply)


if __name__ == "__main__":
    qq_bot = QQBot()
    qq_bot.run()
