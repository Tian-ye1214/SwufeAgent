import os
import sys
import re
import time
import asyncio
import contextvars
import mimetypes
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Any, Awaitable, Callable

from infra.paths import repo_root

_REPO_ROOT = repo_root()
_SRC = _REPO_ROOT / "src"
_API_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in (str(_SRC), _API_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(_REPO_ROOT)

from config.app_config import get_env, settings
from agent_core.input_messages import UserMessage
from agent_core.system import AgentSystem
from tools.memory import ChatHistory
from tools.ExtractFileContent import document_attachment_text_block, is_supported_attachment
from infra.shared_http import close_all_clients
from infra import logger
from runtime import tool_telemetry


@dataclass(frozen=True)
class QueuedTurn:
    user_message: UserMessage
    send_reply: Callable[..., Awaitable[Any]]
    loop: asyncio.AbstractEventLoop


class BotBase(ABC):
    """所有平台机器人的公共基类。

    子类需实现：
      - platform_tag: str      —— 日志前缀，如 "WeChat" / "QQ"
      - session_prefix: str    —— 会话 ID 前缀，如 "wx_" / "qq_"
    """
    AGENT_RUN_TIMEOUT_S = 900.0
    SEND_REPLY_TIMEOUT_S = 120.0
    REPLY_MAX_CHARS = 4500               # 单条回复字符上限，超长按段落/换行/空格切分后分条发送
    SESSION_IDLE_TTL_S = 3600.0          # 空闲超过此时长的会话将被后台回收；<=0 关闭回收
    SESSION_GC_INTERVAL_S = 300.0        # 空闲会话清扫间隔
    RESET_COMMANDS = frozenset({"新任务", "/新任务", "/reset"})
    END_TASK_COMMANDS = frozenset({"结束任务", "/结束任务", "结束当前任务", "/结束当前任务"})
    _MIME_MAP: dict[str, str] = {}

    # 子类可覆盖（从环境变量读取超时）
    _ENV_AGENT_TIMEOUT: str = ""
    _ENV_SEND_TIMEOUT: str = ""
    _ENV_SESSION_IDLE_TTL: str = ""

    def __init__(self):
        self._sessions: dict[str, ChatHistory] = {}
        self._is_first: dict[str, bool] = {}
        self._agent_systems: dict[str, AgentSystem] = {}
        self._session_queues: dict[str, asyncio.Queue[QueuedTurn]] = {}
        self._consumer_tasks: dict[str, asyncio.Task] = {}
        self._session_generation: dict[str, int] = {}
        self._pending_questions: dict[str, dict] = {}
        self._last_attachments: dict[str, list] = {}
        self._last_active: dict[str, float] = {}
        self._gc_task: asyncio.Task | None = None
        self._released = False
        self._agent_ctx: contextvars.ContextVar = contextvars.ContextVar(
            f"{self.__class__.__name__}_ctx", default=None
        )
        if self._ENV_AGENT_TIMEOUT:
            s = get_env(self._ENV_AGENT_TIMEOUT, default="", warn=False)
            if s.strip():
                self.AGENT_RUN_TIMEOUT_S = float(s)
        if self._ENV_SEND_TIMEOUT:
            s = get_env(self._ENV_SEND_TIMEOUT, default="", warn=False)
            if s.strip():
                self.SEND_REPLY_TIMEOUT_S = float(s)
        if self._ENV_SESSION_IDLE_TTL:
            s = get_env(self._ENV_SESSION_IDLE_TTL, default="", warn=False)
            if s.strip():
                self.SESSION_IDLE_TTL_S = float(s)

    @property
    @abstractmethod
    def platform_tag(self) -> str:
        ...

    @property
    @abstractmethod
    def session_prefix(self) -> str:
        ...

    def _agent_for_session(self, session_id: str) -> AgentSystem:
        if session_id not in self._agent_systems:
            s = AgentSystem()
            s.set_ask_user_handler(self._ask_user)
            self._agent_systems[session_id] = s
        return self._agent_systems[session_id]

    def _queue_maxsize(self) -> int:
        raw = settings().get("bot")
        if not isinstance(raw, dict):
            raise RuntimeError("config.json 中须包含 bot 对象。")
        return int(raw["session_queue_maxsize"])

    def _get_queue(self, session_id: str) -> asyncio.Queue[QueuedTurn]:
        if session_id not in self._session_queues:
            self._session_queues[session_id] = asyncio.Queue(maxsize=self._queue_maxsize())
        return self._session_queues[session_id]

    def _bump_generation(self, session_id: str) -> int:
        g = self._session_generation.get(session_id, 0) + 1
        self._session_generation[session_id] = g
        return g

    def _current_generation(self, session_id: str) -> int:
        return self._session_generation.get(session_id, 0)

    def _drain_queue(self, session_id: str) -> list[QueuedTurn]:
        q = self._session_queues.get(session_id)
        if not q:
            return []
        out: list[QueuedTurn] = []
        while True:
            try:
                item = q.get_nowait()
            except asyncio.QueueEmpty:
                break
            out.append(item)
            q.task_done()
        return out

    def _discard_queue(self, q: asyncio.Queue) -> None:
        while True:
            try:
                q.get_nowait()
            except asyncio.QueueEmpty:
                break
            q.task_done()

    def _ensure_consumer(self, session_id: str) -> None:
        t = self._consumer_tasks.get(session_id)
        if t is not None and not t.done():
            return
        if t is not None:
            exc = t.exception()
            if exc:
                logger.error(f"[{self.platform_tag}] {session_id} 队列消费者异常退出，将重启: {exc}")
        self._consumer_tasks[session_id] = asyncio.create_task(
            self._consumer_loop(session_id),
            name=f"{self.platform_tag.lower()}-queue-{session_id}",
        )

    async def _reset_session(self, session_id: str) -> None:
        self._bump_generation(session_id)
        t = self._consumer_tasks.pop(session_id, None)
        agent_system = self._agent_systems.get(session_id)
        if agent_system is not None:
            await agent_system.end_session_agents(session_id)
            await agent_system.shutdown()
        for d in (
            self._sessions,
            self._is_first,
            self._pending_questions,
            self._last_attachments,
            self._agent_systems,
        ):
            d.pop(session_id, None)
        q = self._session_queues.get(session_id)
        if q is not None:
            self._discard_queue(q)
        if t is not None and not t.done():
            t.cancel()
            if t is not asyncio.current_task():
                await asyncio.gather(t, return_exceptions=True)
        logger.info(f"[{self.platform_tag}] 会话 {session_id} 已重置")

    def _touch(self, session_id: str) -> None:
        self._last_active[session_id] = time.monotonic()

    def _session_idle_ttl(self) -> float:
        """有效空闲 TTL：<=0 表示关闭回收；否则夹紧到大于单回合超时，杜绝误杀运行中的长任务。"""
        ttl = self.SESSION_IDLE_TTL_S
        if ttl <= 0:
            return 0.0
        return max(ttl, self.AGENT_RUN_TIMEOUT_S + 60.0)

    def _ensure_session_gc(self) -> None:
        if self._session_idle_ttl() <= 0:
            return
        if self._gc_task is not None and not self._gc_task.done():
            return
        self._gc_task = asyncio.create_task(
            self._session_gc_loop(), name=f"{self.platform_tag.lower()}-session-gc"
        )

    async def _session_gc_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(self.SESSION_GC_INTERVAL_S)
                await self._evict_idle_sessions()
                logger.prune_old_logs()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[{self.platform_tag}] 空闲会话清扫异常: {e}", exc_info=True)

    async def _evict_idle_sessions(self) -> None:
        ttl = self._session_idle_ttl()
        if ttl <= 0:
            return
        now = time.monotonic()
        stale = [sid for sid, ts in list(self._last_active.items()) if now - ts > ttl]
        for sid in stale:
            q = self._session_queues.get(sid)
            if q is not None and q.qsize() > 0:
                continue
            last = self._last_active.get(sid)
            if last is None or time.monotonic() - last <= ttl:
                continue
            logger.info(
                f"[{self.platform_tag}] 回收空闲会话 {sid}（空闲 {int(time.monotonic() - last)}s）"
            )
            await self._evict_session(sid)

    async def _evict_session(self, session_id: str) -> None:
        await self._reset_session(session_id)
        # _reset_session 内部 await（shutdown 最长 15s）期间可能有新消息重新激活该会话：
        # 此时 dispatch_message 已重建消费者并入队。若仍清空队列字典，会丢消息并留下
        # 一个永远阻塞在空队列上的悬挂消费者。检测到重新激活则放弃本次回收。
        # 注：_reset_session 返回后到此处无 await，检查与下方 pop 对协程而言是原子的。
        if session_id in self._consumer_tasks:
            return
        q = self._session_queues.get(session_id)
        if q is not None and q.qsize() > 0:
            return
        self._session_queues.pop(session_id, None)
        self._session_generation.pop(session_id, None)
        self._last_active.pop(session_id, None)

    def _merge_turns(self, batch: list[QueuedTurn]) -> QueuedTurn:
        if len(batch) == 1:
            return batch[0]
        last, lines, atts = batch[-1], [], []
        for t in batch:
            tx = (t.user_message.text or "").strip()
            if tx:
                lines.append(tx)
            if t.user_message.attachments:
                atts.extend(t.user_message.attachments)
        if lines:
            text = "用户连续发来多条消息，请一并回复：\n\n" + "\n".join(
                f"{i + 1}. {ln}" for i, ln in enumerate(lines)
            )
        elif atts:
            text = "用户连续发来多条带附件的消息，请一并分析并回复。"
        else:
            text = "用户连续发来多条消息，请一并处理。"
        return QueuedTurn(UserMessage(text=text, attachments=atts), last.send_reply, last.loop)

    async def _consumer_loop(self, session_id: str) -> None:
        while True:
            try:
                queue = self._get_queue(session_id)
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
                run_gen = self._current_generation(session_id)
                try:
                    merged = self._merge_turns(batch)
                    if len(batch) > 1:
                        logger.info(f"[{self.platform_tag}] {session_id} 合并 {len(batch)} 条消息，一次调用 Agent")
                    run_coro = self._call_agent_async(
                        session_id, merged.user_message, merged.send_reply, merged.loop
                    )
                    try:
                        reply = await asyncio.wait_for(run_coro, timeout=self.AGENT_RUN_TIMEOUT_S)
                    except asyncio.TimeoutError:
                        still_current = run_gen == self._current_generation(session_id)
                        self._bump_generation(session_id)
                        ag = self._agent_systems.get(session_id)
                        if ag is not None:
                            await ag.registry.cancel_session(session_id)
                        logger.error(
                            f"[{self.platform_tag}] {session_id} Agent 超时（{self.AGENT_RUN_TIMEOUT_S:.0f}s），已作废本轮"
                        )
                        if still_current:
                            mins = max(1, round(self.AGENT_RUN_TIMEOUT_S / 60))
                            await self._safe_send(
                                merged.send_reply,
                                f"⏳ 任务超时（已运行约 {mins} 分钟），本轮已取消，可重试或换个问法。",
                            )
                        reply = None
                    if run_gen != self._current_generation(session_id):
                        logger.info(f"[{self.platform_tag}] {session_id} 本轮 Agent 已过期，不发送回复")
                    elif reply is not None:
                        await self._safe_send(merged.send_reply, reply)
                except Exception as e:
                    logger.error(f"[{self.platform_tag}] 会话队列处理异常 {session_id}: {e}")
                finally:
                    for _ in batch:
                        queue.task_done()
                    self._touch(session_id)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[{self.platform_tag}] {session_id} 队列消费异常，继续下一轮: {e}", exc_info=True)

    def _split_reply(self, text: str) -> list[str]:
        """将超长回复按 段落(\\n\\n) > 换行(\\n) > 空格 边界切成 <=REPLY_MAX_CHARS 的若干段。"""
        limit = self.REPLY_MAX_CHARS
        if len(text) <= limit:
            return [text]
        chunks: list[str] = []
        rest = text
        while len(rest) > limit:
            window = rest[:limit]
            cut = window.rfind("\n\n")
            if cut <= 0:
                cut = window.rfind("\n")
            if cut <= 0:
                cut = window.rfind(" ")
            if cut <= 0:
                cut = limit
            chunks.append(rest[:cut].rstrip())
            rest = rest[cut:].lstrip()
        if rest:
            chunks.append(rest)
        return chunks

    async def _send_one(self, send_reply: Callable[..., Awaitable[Any]], text: str) -> None:
        try:
            await asyncio.wait_for(send_reply(text), timeout=self.SEND_REPLY_TIMEOUT_S)
        except asyncio.TimeoutError:
            logger.warning(f"[{self.platform_tag}] 发送回复超时（{self.SEND_REPLY_TIMEOUT_S:.0f}s），重试一次")
            try:
                await asyncio.wait_for(send_reply(text), timeout=self.SEND_REPLY_TIMEOUT_S)
            except Exception as e:
                logger.warning(f"[{self.platform_tag}] 发送回复重试仍失败: {e}")
        except Exception as e:
            logger.warning(f"[{self.platform_tag}] 发送消息失败: {e}")

    async def _safe_send(self, send_reply: Callable[..., Awaitable[Any]], text: str) -> None:
        for chunk in self._split_reply(text):
            await self._send_one(send_reply, chunk)

    async def _inline_document_bytes(
        self,
        text: str,
        data: bytes,
        *,
        media_type: str = "",
        filename: str = "",
    ) -> tuple[str, bool]:
        if not data or not is_supported_attachment(data, media_type=media_type, filename=filename):
            return text, False
        block = await asyncio.to_thread(
            document_attachment_text_block, data, filename=filename or "", media_type=media_type
        )
        if not block:
            return text, False
        return (f"{text}\n\n{block}".strip() if text else block), True

    def guess_download_mime(
        self,
        *,
        filename: str = "",
        media_type_key: str = "",
    ) -> str:
        if filename:
            mime, _ = mimetypes.guess_type(filename)
            if mime:
                return mime
        return self._MIME_MAP.get((media_type_key or "").lower(), "application/octet-stream")

    async def _partition_document_attachments(
        self,
        user_text: str,
        attachments: list,
        *,
        filename_attr: str = "",
    ) -> tuple[str, list]:
        """Inline supported document BinaryContent as text; keep other attachments unchanged."""
        out: list = []
        text = user_text
        for att in attachments:
            data = getattr(att, "data", b"") or b""
            media_type = getattr(att, "media_type", "") or ""
            filename = getattr(att, filename_attr, "") if filename_attr else ""
            text, consumed = await self._inline_document_bytes(
                text,
                data,
                media_type=media_type,
                filename=filename or None,
            )
            if not consumed:
                out.append(att)
        return text, out

    async def _end_task_and_consume_queue(self, session_id: str, send_reply: Callable[..., Awaitable[Any]]) -> None:
        drained = self._drain_queue(session_id)
        await self._reset_session(session_id)
        if not drained:
            await self._safe_send(send_reply, "✓ 已结束当前任务，上下文已清空。队列中没有待处理消息。")
            return
        merged = self._merge_turns(drained)
        logger.info(f"[{self.platform_tag}] {session_id} 结束任务，合并 {len(drained)} 条队列消息，开启新任务")
        await self._get_queue(session_id).put(merged)
        self._ensure_consumer(session_id)

    def _notify_user(self, text: str, session_id: str, run_gen: int,
                     send_reply: Callable[..., Awaitable[Any]], loop: asyncio.AbstractEventLoop) -> None:
        """向用户推送中间通知；与 Bot 同事件循环时用 create_task，避免阻塞。"""
        if run_gen != self._current_generation(session_id):
            return
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        coro = self._safe_send(send_reply, text)
        if running is loop:
            asyncio.create_task(coro)
            return
        fut = asyncio.run_coroutine_threadsafe(coro, loop)
        try:
            fut.result(timeout=12)
        except Exception:
            pass

    async def _call_agent_async(
        self,
        session_id: str,
        message: UserMessage,
        send_reply: Callable[..., Awaitable[Any]],
        loop: asyncio.AbstractEventLoop,
    ) -> str:
        agent_system = self._agent_for_session(session_id)
        run_gen = self._current_generation(session_id)
        token = self._agent_ctx.set((session_id, run_gen, send_reply, loop))
        try:
            with logger.session_log_context(session_id):
                history = self._sessions.setdefault(session_id, ChatHistory())
                if not self._is_first.get(session_id, False):
                    agent_system.set_task_directory(f"{self.platform_tag}_{session_id[:20]}")
                    self._is_first[session_id] = True

                tool_telemetry.set_user_notify_callback(
                    partial(self._notify_user, session_id=session_id, run_gen=run_gen, send_reply=send_reply, loop=loop)
                )
                await agent_system.bind_session(session_id)
                turn_id = uuid.uuid4().hex[:8]
                try:
                    _, output = await agent_system.run_agent_system(
                        message,
                        history,
                        conversation_log_hint=session_id,
                        conversation_log_extra={
                            "session_id": session_id,
                            "platform": self.platform_tag,
                            "turn_id": turn_id,
                        },
                        turn_id=turn_id,
                    )
                except Exception as e:
                    logger.error(f"[{self.platform_tag}] Agent 调用异常: {e}")
                    output = f"抱歉，处理您的请求时出现了错误：{e}"
        finally:
            tool_telemetry.set_user_notify_callback(None)
            self._agent_ctx.reset(token)
        return output

    async def _ask_user(self, question: str, timeout: int | None = None) -> str | None:
        actual_timeout = timeout if timeout is not None else 120
        ctx = self._agent_ctx.get()
        if not ctx:
            return (
                await asyncio.to_thread(input, f"[ask_user] {question}\n回复: ")
            ).strip()
        session_id, run_gen, send_func, loop = ctx
        if run_gen != self._current_generation(session_id):
            return None
        pending = self._pending_questions.get(session_id)
        pending_fut = pending.get("future") if pending else None
        if pending_fut is not None and not pending_fut.done():
            logger.warning(f"[{self.platform_tag} ask_user] {session_id} already has a pending question")
            return None
        answer_fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending_questions[session_id] = {"future": answer_fut}
        try:
            try:
                await asyncio.wait_for(send_func(f"🤔 {question}"), timeout=15)
            except Exception as e:
                logger.error(f"[{self.platform_tag} ask_user] 发送问题失败: {e}")
            try:
                return await asyncio.wait_for(answer_fut, timeout=actual_timeout)
            except asyncio.TimeoutError:
                return None
        finally:
            self._pending_questions.pop(session_id, None)

    async def dispatch_message(
        self,
        session_id: str,
        user_text: str,
        attachments: list,
        send_reply: Callable[..., Awaitable[Any]],
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        """统一的消息分发逻辑，子类在收到消息后调用此方法。"""
        if not session_id:
            return

        if attachments:
            self._last_attachments[session_id] = list(attachments)
        elif user_text and self._last_attachments.get(session_id):
            attachments = list(self._last_attachments[session_id])
            logger.info(f"[{self.platform_tag}] {session_id} 复用上一则消息中的媒体附件（{len(attachments)} 个）")

        logger.info(
            f"[{self.platform_tag}] {session_id} 消息: {user_text[:60]}"
            f"{f' (+{len(attachments)}个附件)' if attachments else ''}"
        )

        pending = self._pending_questions.get(session_id)
        pf = pending.get("future") if pending else None
        if pf is not None and not pf.done():
            pf.set_result(user_text)
            logger.info(f"[{self.platform_tag}] {session_id} 本条消息已作为 ask_user 的回复投递")
            await self._safe_send(send_reply, "✓ 已收到你的回复，正在继续处理…")
            return

        if user_text in self.RESET_COMMANDS:
            await self._reset_session(session_id)
            await self._safe_send(send_reply, "已开始新对话，上下文已清除。")
            return
        if user_text in self.END_TASK_COMMANDS:
            await self._end_task_and_consume_queue(session_id, send_reply)
            return

        if not user_text and not attachments:
            return
        if not user_text and attachments:
            user_text = ""

        msg = UserMessage(text=user_text, attachments=attachments)
        q = self._get_queue(session_id)
        if q.full():
            logger.warning(
                f"[{self.platform_tag}] {session_id} 队列已满（maxsize={q.maxsize}），拒绝入队"
            )
            await self._safe_send(
                send_reply,
                f"当前会话待处理消息过多（上限 {q.maxsize} 条），请稍后再试或发送「新任务」清空。",
            )
            return
        if (n := q.qsize()):
            logger.info(f"[{self.platform_tag}] {session_id} 入队（前方还有 {n} 条待处理）")
        await q.put(QueuedTurn(msg, send_reply, loop))
        await self._safe_send(send_reply, "✓ 收到，正在处理…")
        self._touch(session_id)
        self._ensure_consumer(session_id)
        self._ensure_session_gc()

    async def dispatch_user_message(
        self,
        session_id: str,
        message: UserMessage,
        send_reply: Callable[..., Awaitable[Any]],
    ) -> None:
        await self.dispatch_message(
            session_id,
            message.text,
            message.attachments,
            send_reply,
            asyncio.get_running_loop(),
        )

    async def release_all_resources_async(self) -> None:
        if self._released:
            return
        self._released = True
        gc_task = self._gc_task
        self._gc_task = None
        if gc_task is not None and not gc_task.done() and gc_task is not asyncio.current_task():
            gc_task.cancel()
            await asyncio.gather(gc_task, return_exceptions=True)
        agents = list(self._agent_systems.values())
        n = len(agents)
        for pending in list(self._pending_questions.values()):
            pf = pending.get("future")
            if pf is not None and not pf.done():
                pf.cancel()
        for ag in agents:
            await ag.shutdown()
        consumer_tasks = [
            t for t in list(self._consumer_tasks.values())
            if t and not t.done() and t is not asyncio.current_task()
        ]
        for t in consumer_tasks:
            t.cancel()
        if consumer_tasks:
            await asyncio.gather(*consumer_tasks, return_exceptions=True)
        for attr in (
            self._pending_questions,
            self._sessions,
            self._is_first,
            self._last_attachments,
            self._last_active,
            self._agent_systems,
        ):
            attr.clear()
        self._consumer_tasks.clear()
        self._session_queues.clear()
        self._session_generation.clear()
        await close_all_clients()
        logger.info(f"[{self.platform_tag}] 已释放全部会话与 Agent 资源（共 {n} 个 Agent 实例）")

    def release_all_resources(self) -> None:
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                raise RuntimeError(
                    "release_all_resources() cannot run inside an active event loop; "
                    "use release_all_resources_async()"
                )
        except RuntimeError as e:
            if "release_all_resources()" in str(e):
                raise
        asyncio.run(self.release_all_resources_async())

    def clean_text(self, raw: str) -> str:
        return re.sub(r"\s+", " ", (raw or "").strip())
