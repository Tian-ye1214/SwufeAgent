import sys
import os
import re
import asyncio
import contextvars
import inspect
import threading
import concurrent.futures
from functools import partial
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from ncatbot.core import BotClient, BaseMessageEvent, GroupMessageEvent, MetaEvent
from ncatbot.plugin_system import on_message
from ncatbot.utils import config

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_API_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in (_ROOT, _API_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(_ROOT)

from main import AgentSystem
from tools.memory import ChatHistory, UserMessage
from qq_media_helpers import extract_media
import logger

config.set_bot_uin(os.environ.get("QQBOT_ID", None))


@dataclass(frozen=True)
class _QueuedAgentTurn:
    user_message: UserMessage
    send_to_qq: Callable[..., Awaitable[Any]]
    loop: asyncio.AbstractEventLoop


class QQBot:
    _MAX_LENGTH = 1800
    _AGENT_RUN_TIMEOUT_S = float(os.environ.get("QQ_AGENT_TIMEOUT_S", "900"))
    _SEND_REPLY_TIMEOUT_S = float(os.environ.get("QQ_SEND_REPLY_TIMEOUT_S", "120"))
    _DEFAULT_MEDIA_PROMPT = "请分析这些内容。"
    _SESSION_END_TASK_PHRASES = frozenset({"结束任务", "/结束任务", "结束当前任务", "/结束当前任务"})
    _FILE_ALLOW_EXT = frozenset(
        {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".mp4", ".mov", ".mkv", ".webm",
         ".pdf", ".txt", ".md", ".docx", ".xlsx", ".csv", ".json"}
    )
    _AGENT_CTX: contextvars.ContextVar[
        tuple[str, int, Callable[..., Awaitable[Any]], asyncio.AbstractEventLoop] | None
    ] = contextvars.ContextVar("qq_agent_ctx", default=None)

    def __init__(self):
        self.__sessions: dict[str, ChatHistory] = {}
        self.__is_first: dict[str, bool] = {}
        self.__agent_systems: dict[str, AgentSystem] = {}
        self.__session_message_queues: dict[str, asyncio.Queue[_QueuedAgentTurn]] = {}
        self.__session_consumer_tasks: dict[str, asyncio.Task] = {}
        self.__session_task_generation: dict[str, int] = {}
        self.__pending_questions: dict[str, dict] = {}
        self.__bot = BotClient()
        self.__session_last_attachments: dict[str, list] = {}
        self.__released = False
        self._agent_executor: concurrent.futures.ThreadPoolExecutor | None = None

        async def _on_shutdown(_: MetaEvent) -> None:
            self.release_all_resources()

        self.__bot.add_shutdown_handler(_on_shutdown)

    def release_all_resources(self) -> None:
        if self.__released:
            return
        self.__released = True
        n_agents = len(self.__agent_systems)
        for pending in list(self.__pending_questions.values()):
            ev = pending.get("event")
            if ev and not ev.is_set():
                pending["answer"] = None
                ev.set()
        self.__pending_questions.clear()
        self.__sessions.clear()
        self.__is_first.clear()
        self.__session_last_attachments.clear()
        self.__agent_systems.clear()
        for t in list(self.__session_consumer_tasks.values()):
            if t and not t.done():
                t.cancel()
        self.__session_consumer_tasks.clear()
        self.__session_message_queues.clear()
        self.__session_task_generation.clear()
        ex, self._agent_executor = self._agent_executor, None
        if ex is not None:
            try:
                ex.shutdown(wait=False, cancel_futures=True)
            except Exception as e:
                logger.warning(f"[QQ] 关闭 Agent 线程池: {e}")
        logger.info(f"[QQ] 已释放全部会话与 Agent 资源（共 {n_agents} 个 Agent 实例）")

    def _agent_for_session(self, session_id: str) -> AgentSystem:
        if session_id not in self.__agent_systems:
            s = AgentSystem()
            s.set_ask_user_handler(self._qq_ask_user)
            self.__agent_systems[session_id] = s
        return self.__agent_systems[session_id]

    def _get_session_queue(self, session_id: str) -> asyncio.Queue[_QueuedAgentTurn]:
        if session_id not in self.__session_message_queues:
            self.__session_message_queues[session_id] = asyncio.Queue()
        return self.__session_message_queues[session_id]

    def _bump_session_generation(self, session_id: str) -> int:
        g = self.__session_task_generation.get(session_id, 0) + 1
        self.__session_task_generation[session_id] = g
        return g

    def _current_session_generation(self, session_id: str) -> int:
        return self.__session_task_generation.get(session_id, 0)

    def _drain_queue_nowait(self, session_id: str) -> list[_QueuedAgentTurn]:
        q = self.__session_message_queues.get(session_id)
        if not q:
            return []
        out: list[_QueuedAgentTurn] = []
        while True:
            try:
                item = q.get_nowait()
            except asyncio.QueueEmpty:
                break
            out.append(item)
            q.task_done()
        return out

    @staticmethod
    def _drain_queue_object_discard(q: asyncio.Queue) -> None:
        while True:
            try:
                q.get_nowait()
            except asyncio.QueueEmpty:
                break
            q.task_done()

    def _ensure_session_consumer(self, session_id: str) -> None:
        t = self.__session_consumer_tasks.get(session_id)
        if t is not None:
            if not t.done():
                return
            exc = t.exception()
            if exc is not None:
                logger.error(f"[QQ] {session_id} 队列消费者异常退出，将重启: {exc}")
        self.__session_consumer_tasks[session_id] = asyncio.create_task(self._session_consumer_loop(session_id), name=f"qq-queue-{session_id}")

    @staticmethod
    async def _reply_event(event: BaseMessageEvent, text: str) -> None:
        sig = inspect.signature(event.reply)
        await (event.reply(text=text, at=False) if "at" in sig.parameters else event.reply(text=text))

    @staticmethod
    def _session_id_from_event(event: BaseMessageEvent) -> str:
        return f"group_{event.group_id}" if isinstance(event, GroupMessageEvent) else f"private_{event.user_id}"

    @staticmethod
    def _merge_queued_turns(batch: list[_QueuedAgentTurn]) -> _QueuedAgentTurn:
        if len(batch) == 1:
            return batch[0]
        last, lines, atts = batch[-1], [], []
        for t in batch:
            tx = (t.user_message.text or "").strip()
            if tx:
                lines.append(tx)
            if t.user_message.attachments:
                atts.extend(t.user_message.attachments)
        cmb = (
            ("用户连续发来多条消息，请一并回复：\n\n" + "\n".join(f"{i + 1}. {ln}" for i, ln in enumerate(lines)))
            if lines
            else ("用户连续发来多条带附件的消息，请一并分析并回复。" if atts else "用户连续发来多条消息，请一并处理。")
        )
        return _QueuedAgentTurn(UserMessage(text=cmb, attachments=atts), last.send_to_qq, last.loop)

    async def _session_consumer_loop(self, session_id: str) -> None:
        while True:
            try:
                queue = self._get_session_queue(session_id)
                try:
                    first = await queue.get()
                except asyncio.CancelledError:
                    break
                batch = [first]
                while True:
                    try:
                        batch.append(queue.get_nowait())
                    except asyncio.QueueEmpty:
                        break
                run_gen = self._current_session_generation(session_id)
                try:
                    merged = self._merge_queued_turns(batch)
                    if len(batch) > 1:
                        logger.info(f"[QQ] {session_id} 合并 {len(batch)} 条入队消息，一次调用 Agent")
                    ex = self._agent_executor
                    if ex is None:
                        raise RuntimeError("Agent 线程池未初始化（应先调用 QQBot.run）")
                    run_coro = merged.loop.run_in_executor(ex, partial(self._call_agent, session_id, merged.user_message, merged.send_to_qq, merged.loop))
                    try:
                        reply = await asyncio.wait_for(run_coro, timeout=QQBot._AGENT_RUN_TIMEOUT_S)
                    except asyncio.TimeoutError:
                        self._bump_session_generation(session_id)
                        logger.error(f"[QQ] {session_id} Agent 单次运行超时（{QQBot._AGENT_RUN_TIMEOUT_S:.0f}s），已作废本轮")
                        reply = None
                    if run_gen != self._current_session_generation(session_id):
                        logger.info(f"[QQ] {session_id} 本轮 Agent 已过期（用户已结束任务或重置会话），不发送回复")
                    elif reply is not None:
                        try:
                            await asyncio.wait_for(merged.send_to_qq(reply), timeout=QQBot._SEND_REPLY_TIMEOUT_S)
                        except asyncio.TimeoutError:
                            logger.warning(f"[QQ] 发送回复超时（{QQBot._SEND_REPLY_TIMEOUT_S:.0f}s），已放弃，继续消费队列")
                        except Exception as e:
                            logger.warning(f"[QQ] 发送消息失败: {e}")
                except Exception as e:
                    logger.error(f"[QQ] 会话队列处理异常 {session_id}: {e}")
                finally:
                    for _ in batch:
                        queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[QQ] {session_id} 队列消费单轮未捕获异常，继续下一轮: {e}", exc_info=True)

    @staticmethod
    async def _send_text(send_to_qq: Callable[..., Awaitable[Any]], text: str) -> None:
        try:
            await asyncio.wait_for(send_to_qq(text), timeout=QQBot._SEND_REPLY_TIMEOUT_S)
        except asyncio.TimeoutError:
            logger.warning(f"[QQ] 发送回复超时（{QQBot._SEND_REPLY_TIMEOUT_S:.0f}s），已放弃，继续消费队列")
        except Exception as e:
            logger.warning(f"[QQ] 发送消息失败: {e}")

    def run(self, *, debug: bool = True, remote_mode: bool = True, enable_webui_interaction: bool = False, **kwargs):
        self.__released = False
        self._agent_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max(2, int(os.environ.get("QQ_AGENT_THREAD_WORKERS", "4"))), thread_name_prefix="qq-agent"
        )
        self._register_handlers()
        try:
            self.__bot.run_frontend(debug=debug, remote_mode=remote_mode, enable_webui_interaction=enable_webui_interaction, **kwargs)
        finally:
            self.release_all_resources()

    def _qq_ask_user(self, question: str, timeout: int | None = None) -> str | None:
        actual_timeout = timeout if timeout is not None else 120
        ctx = QQBot._AGENT_CTX.get()
        if not ctx:
            return input(f"[ask_user] {question}\n回复: ").strip()
        session_id, run_gen, send_func, loop = ctx
        if run_gen != self._current_session_generation(session_id):
            return None
        event = threading.Event()
        self.__pending_questions[session_id] = {"event": event, "answer": None}
        try:
            fut = asyncio.run_coroutine_threadsafe(send_func(f"🤔 {question}"), loop)
            try:
                fut.result(timeout=10)
            except Exception as e:
                logger.error(f"[QQ ask_user] 发送问题失败: {e}")
            got_reply = event.wait(timeout=actual_timeout)
            pending = self.__pending_questions.pop(session_id, {})
            return pending.get("answer") if got_reply and pending.get("answer") else None
        finally:
            self.__pending_questions.pop(session_id, None)

    def _reset_session(self, session_id: str) -> None:
        self._bump_session_generation(session_id)
        for d in (self.__sessions, self.__is_first, self.__pending_questions, self.__session_last_attachments, self.__agent_systems):
            d.pop(session_id, None)
        q = self.__session_message_queues.get(session_id)
        if q is not None:
            self._drain_queue_object_discard(q)
        t = self.__session_consumer_tasks.pop(session_id, None)
        if t is not None and not t.done():
            t.cancel()
        logger.info(f"[QQ] 会话 {session_id} 已重置")

    async def _end_task_and_consume_queue(self, session_id: str, send_to_qq: Callable[..., Awaitable[Any]]) -> None:
        drained = self._drain_queue_nowait(session_id)
        self._reset_session(session_id)
        if not drained:
            await self._send_text(send_to_qq, "✓ 已结束当前任务，上下文已清空。队列中没有待处理消息。")
            return
        merged = self._merge_queued_turns(drained)
        logger.info(f"[QQ] {session_id} 结束任务：已合并 {len(drained)} 条队列消息，立即开启新任务")
        await self._get_session_queue(session_id).put(merged)
        self._ensure_session_consumer(session_id)

    def _call_agent(
        self, session_id: str, message: UserMessage, send_to_qq: Callable[..., Awaitable[Any]], loop: asyncio.AbstractEventLoop,
    ) -> str:
        agent_system = self._agent_for_session(session_id)
        run_gen = self._current_session_generation(session_id)
        QQBot._AGENT_CTX.set((session_id, run_gen, send_to_qq, loop))
        history = self.__sessions.setdefault(session_id, ChatHistory())
        logger.setup_session_logger(session_id)
        if not self.__is_first.get(session_id, False):
            agent_system.set_task_directory(f"QQ_{session_id[:12]}")
            self.__is_first[session_id] = True

        def _qq_notify(text: str):
            if run_gen != self._current_session_generation(session_id):
                return
            fut = asyncio.run_coroutine_threadsafe(send_to_qq(text), loop)
            try:
                fut.result(timeout=12)
            except Exception:
                pass

        logger.set_user_notify_callback(_qq_notify)
        try:
            _, output = agent_system.run_agent_system(message, history)
        except Exception as e:
            logger.error(f"[QQ] Agent 调用异常: {e}")
            output = f"抱歉，处理您的请求时出现了错误：{e}"
        finally:
            logger.set_user_notify_callback(None)
            QQBot._AGENT_CTX.set(None)
        if len(output) > self._MAX_LENGTH:
            output = output[: self._MAX_LENGTH] + "\n\n…（内容过长，已截断）"
        return output

    async def _extract_media(self, event: BaseMessageEvent) -> list:
        return await extract_media(self.__bot.api, event, QQBot._FILE_ALLOW_EXT)

    @staticmethod
    def _clean_text(raw_message: str) -> str:
        return re.sub(r"\[CQ:[^\]]+\]", "", raw_message).strip()

    @staticmethod
    def _is_at_me(event: BaseMessageEvent) -> bool:
        if not isinstance(event, GroupMessageEvent):
            return True
        msg = getattr(event, "message", None)
        return msg is not None and msg.is_user_at(config.bt_uin)

    def _register_handlers(self) -> None:
        bot = self

        @on_message
        async def handle_message(event: BaseMessageEvent):
            raw_text = (event.raw_message or "").strip()
            if not raw_text or (isinstance(event, GroupMessageEvent) and not QQBot._is_at_me(event)):
                return
            session_id = QQBot._session_id_from_event(event)
            user_text = QQBot._clean_text(raw_text)
            extracted = await bot._extract_media(event)
            loop = asyncio.get_running_loop()
            if extracted:
                bot.__session_last_attachments[session_id] = list(extracted)
                attachments = extracted
            elif user_text and bot.__session_last_attachments.get(session_id):
                attachments = list(bot.__session_last_attachments[session_id])
                logger.info(f"[QQ] {session_id} 复用上一则消息中的媒体附件（{len(attachments)} 个）")
            else:
                attachments = []
            send_to_qq = partial(QQBot._reply_event, event)
            logger.info(f"[NapCat] {session_id} 消息: {user_text[:60]}{f' (+{len(attachments)}个附件)' if attachments else ''}")
            pending = bot.__pending_questions.get(session_id)
            if pending and not pending["event"].is_set():
                pending["answer"] = user_text
                pending["event"].set()
                logger.info(f"[QQ] {session_id} 本条消息已作为 ask_user 的回复投递，Agent 将继续处理")
                await bot._send_text(send_to_qq, "✓ 已收到你的回复，正在继续处理…")
                return
            if user_text in ("新任务", "/新任务", "/reset"):
                bot._reset_session(session_id)
                await bot._send_text(send_to_qq, "已开始新对话，上下文已清除。")
                return
            if user_text in QQBot._SESSION_END_TASK_PHRASES:
                await bot._end_task_and_consume_queue(session_id, send_to_qq)
                return
            text = (user_text or "").strip()
            if not text and not attachments:
                return
            if not text and attachments:
                text = bot._DEFAULT_MEDIA_PROMPT
            msg = UserMessage(text=text, attachments=attachments)
            q = bot._get_session_queue(session_id)
            if (n := q.qsize()):
                logger.info(f"[QQ] {session_id} 入队（前方还有 {n} 条待处理）")
            await q.put(_QueuedAgentTurn(msg, send_to_qq, loop))
            bot._ensure_session_consumer(session_id)


if __name__ == "__main__":
    QQBot().run()
