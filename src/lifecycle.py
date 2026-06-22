from __future__ import annotations

import asyncio
import inspect
import time
import uuid
from collections import deque
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeVar

import logger
from app_config import settings
from runtime_state import TRACE_STORE, agent_context, turn_context

T = TypeVar("T")

_invocation_stack: ContextVar[tuple[str, ...]] = ContextVar("lifecycle_invocation_stack", default=())


class AgentInstanceState(Enum):
    IDLE = "idle"
    RUNNING = "running"


class AgentInvocationState(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


def make_agent_id(session_key: str, role: str, suffix: str | None = None) -> str:
    if suffix:
        return f"{session_key}:{role}:{suffix}"
    return f"{session_key}:{role}"


def _invocation_history_limit() -> int:
    lc = settings().get("lifecycle")
    if not isinstance(lc, dict):
        raise KeyError("config.json 缺少 lifecycle 配置块")
    n = lc.get("invocation_history_per_session")
    if not isinstance(n, int) or n < 1:
        raise ValueError("lifecycle.invocation_history_per_session 须为正整数")
    return n


@dataclass
class AgentInstance:
    agent_id: str
    role: str
    session_key: str
    state: AgentInstanceState
    current_invocation_id: str | None = None


@dataclass
class AgentInvocation:
    invocation_id: str
    agent_id: str
    role: str
    session_key: str
    parent_invocation_id: str | None
    turn_id: str | None
    state: AgentInvocationState
    started_at: float
    finished_at: float | None
    task_ref: asyncio.Task[Any] | None
    cancel_requested: asyncio.Event = field(default_factory=asyncio.Event)


@dataclass
class SessionLifecycleView:
    session_key: str
    agents: list[AgentInstance]
    active_invocations: list[AgentInvocation]
    recent_invocations: list[AgentInvocation]


class AgentRegistry:
    def __init__(self) -> None:
        self._agents: dict[str, AgentInstance] = {}
        self._invocations: dict[str, AgentInvocation] = {}
        self._history: dict[str, deque[AgentInvocation]] = {}
        self._lock = asyncio.Lock()

    @staticmethod
    def _prefix_matches(keys, prefix: str) -> list[str]:
        key = prefix.strip()
        if key in keys:
            return [key]
        return [k for k in keys if k.startswith(key)]

    def _history_for(self, session_key: str) -> deque[AgentInvocation]:
        if session_key not in self._history:
            self._history[session_key] = deque(maxlen=_invocation_history_limit())
        return self._history[session_key]

    async def ensure_agent(
        self, session_key: str, role: str, suffix: str | None = None,
    ) -> str:
        agent_id = make_agent_id(session_key, role, suffix)
        async with self._lock:
            if agent_id not in self._agents:
                self._agents[agent_id] = AgentInstance(
                    agent_id=agent_id, role=role, session_key=session_key,
                    state=AgentInstanceState.IDLE,
                )
        return agent_id

    async def list_agents(self, session_key: str | None = None) -> list[AgentInstance]:
        async with self._lock:
            agents = list(self._agents.values())
        if session_key is None:
            return agents
        return [a for a in agents if a.session_key == session_key]

    async def remove_session(self, session_key: str) -> None:
        async with self._lock:
            to_remove = [aid for aid, a in self._agents.items() if a.session_key == session_key]
            for aid in to_remove:
                self._agents.pop(aid, None)
            self._history.pop(session_key, None)

    async def register_invocation(self, inv: AgentInvocation) -> None:
        async with self._lock:
            self._invocations[inv.invocation_id] = inv
            agent = self._agents.get(inv.agent_id)
            if agent is not None:
                agent.state = AgentInstanceState.RUNNING
                agent.current_invocation_id = inv.invocation_id

    def _coerce_terminal_state(self, inv: AgentInvocation) -> None:
        if inv.state not in (AgentInvocationState.RUNNING, AgentInvocationState.PENDING):
            return
        inv.state = AgentInvocationState.CANCELLED
        inv.finished_at = inv.finished_at or time.monotonic()

    async def finish_invocation(self, invocation_id: str) -> None:
        async with self._lock:
            inv = self._invocations.pop(invocation_id, None)
            if inv is None:
                return
            self._coerce_terminal_state(inv)
            self._history_for(inv.session_key).append(inv)
            agent = self._agents.get(inv.agent_id)
            if agent is not None and agent.current_invocation_id == invocation_id:
                agent.current_invocation_id = None
                agent.state = AgentInstanceState.IDLE
                if agent.role == "worker":
                    self._agents.pop(inv.agent_id, None)

    async def resolve_active_invocation_id(self, id_or_prefix: str) -> str | None:
        async with self._lock:
            matches = self._prefix_matches(self._invocations, id_or_prefix)
        if len(matches) == 1:
            return matches[0]
        return None

    async def count_active_invocation_prefix_matches(self, prefix: str) -> int:
        async with self._lock:
            return len(self._prefix_matches(self._invocations, prefix))

    async def find_recent_invocation_by_prefix(
        self, session_key: str, prefix: str
    ) -> AgentInvocation | None:
        async with self._lock:
            history = list(self._history.get(session_key, ()))
        matched_ids = set(
            self._prefix_matches((i.invocation_id for i in history), prefix)
        )
        matches = [i for i in history if i.invocation_id in matched_ids]
        if len(matches) == 1:
            return matches[0]
        return None

    async def list_active_invocations(self, session_key: str | None = None) -> list[AgentInvocation]:
        async with self._lock:
            invs = list(self._invocations.values())
        if session_key is None:
            return invs
        return [i for i in invs if i.session_key == session_key]

    async def list_recent_invocations(self, session_key: str) -> list[AgentInvocation]:
        async with self._lock:
            return list(self._history.get(session_key, ()))

    async def get_session_view(self, session_key: str) -> SessionLifecycleView:
        agents = await self.list_agents(session_key)
        active = await self.list_active_invocations(session_key)
        recent = await self.list_recent_invocations(session_key)
        return SessionLifecycleView(
            session_key=session_key, agents=agents,
            active_invocations=active, recent_invocations=recent,
        )

    async def cancel(self, invocation_id: str) -> bool:
        async with self._lock:
            matches = self._prefix_matches(self._invocations, invocation_id)
            if len(matches) != 1:
                return False
            inv = self._invocations[matches[0]]
        inv.cancel_requested.set()
        t = inv.task_ref
        if t is not None and not t.done():
            t.cancel()
        return True

    async def cancel_agent(self, agent_id: str) -> int:
        async with self._lock:
            inv = next(
                (i for i in self._invocations.values() if i.agent_id == agent_id), None,
            )
        if inv is None:
            return 0
        if await self.cancel(inv.invocation_id):
            return 1
        return 0

    async def cancel_turn(self, turn_id: str) -> int:
        async with self._lock:
            ids = [
                i.invocation_id
                for i in self._invocations.values()
                if i.turn_id == turn_id
            ]
        n = 0
        for iid in ids:
            if await self.cancel(iid):
                n += 1
        if n:
            logger.info("[lifecycle] cancel_turn turn=%s count=%s", turn_id, n)
        return n

    async def cancel_all(self) -> int:
        async with self._lock:
            ids = list(self._invocations.keys())
        n = 0
        for iid in ids:
            if await self.cancel(iid):
                n += 1
        if ids:
            logger.info("[lifecycle] cancel_all requested for %s invocation(s)", len(ids))
        return n

    async def cancel_session(self, session_key: str) -> int:
        active = await self.list_active_invocations(session_key)
        n = 0
        for inv in active:
            if await self.cancel(inv.invocation_id):
                n += 1
        return n


class LifecycleHooks:
    def __init__(self) -> None:
        self._on_start: list[Callable[[AgentInvocation], Any]] = []
        self._on_finish: list[Callable[[AgentInvocation], Any]] = []
        self._on_error: list[Callable[[AgentInvocation, BaseException], Any]] = []
        self._on_cancel: list[Callable[[AgentInvocation], Any]] = []

    def add_on_start(self, fn: Callable[[AgentInvocation], Any]) -> None:
        self._on_start.append(fn)

    def add_on_finish(self, fn: Callable[[AgentInvocation], Any]) -> None:
        self._on_finish.append(fn)

    def add_on_error(self, fn: Callable[[AgentInvocation, BaseException], Any]) -> None:
        self._on_error.append(fn)

    def add_on_cancel(self, fn: Callable[[AgentInvocation], Any]) -> None:
        self._on_cancel.append(fn)

    async def _dispatch(self, callbacks: list[Callable[..., Any]], *args: Any) -> None:
        for fn in callbacks:
            try:
                r = fn(*args)
                if inspect.isawaitable(r):
                    await r
            except Exception as e:
                logger.warning("lifecycle hook %s failed: %s", getattr(fn, "__name__", fn), e)

    async def dispatch_start(self, inv: AgentInvocation) -> None:
        await self._dispatch(self._on_start, inv)

    async def dispatch_finish(self, inv: AgentInvocation) -> None:
        await self._dispatch(self._on_finish, inv)

    async def dispatch_error(self, inv: AgentInvocation, exc: BaseException) -> None:
        await self._dispatch(self._on_error, inv, exc)

    async def dispatch_cancel(self, inv: AgentInvocation) -> None:
        await self._dispatch(self._on_cancel, inv)


def _role_from_agent_id(agent_id: str) -> str:
    parts = agent_id.split(":")
    if len(parts) >= 2:
        return parts[1]
    return "unknown"


def register_default_lifecycle_logging(hooks: LifecycleHooks) -> None:
    def _on_start(inv: AgentInvocation) -> None:
        parent = (inv.parent_invocation_id or "-")[:8]
        logger.debug(
            "[lifecycle] agent=%s id=%s inv=%s parent=%s turn=%s state=%s",
            inv.role, inv.agent_id, inv.invocation_id[:8],
            parent, inv.turn_id, inv.state.value,
        )

    def _on_finish(inv: AgentInvocation) -> None:
        logger.debug(
            "[lifecycle] agent=%s id=%s inv=%s state=%s",
            inv.role, inv.agent_id, inv.invocation_id[:8], inv.state.value,
        )

    def _on_error(inv: AgentInvocation, exc: BaseException) -> None:
        logger.warning(
            "[lifecycle] error agent=%s id=%s inv=%s: %s",
            inv.role, inv.agent_id, inv.invocation_id[:8], exc,
        )

    def _on_cancel(inv: AgentInvocation) -> None:
        logger.debug(
            "[lifecycle] cancel agent=%s id=%s inv=%s",
            inv.role, inv.agent_id, inv.invocation_id[:8],
        )

    hooks.add_on_start(_on_start)
    hooks.add_on_finish(_on_finish)
    hooks.add_on_error(_on_error)
    hooks.add_on_cancel(_on_cancel)


async def run_coroutine_with_lifecycle(
    *,
    factory: Callable[[], Awaitable[T]],
    agent_id: str,
    registry: AgentRegistry,
    hooks: LifecycleHooks,
    turn_id: str | None,
    parent_invocation_id: str | None = None,
) -> T:
    stack = _invocation_stack.get()
    parent = parent_invocation_id if parent_invocation_id is not None else (stack[-1] if stack else None)
    invocation_id = str(uuid.uuid4())
    role = _role_from_agent_id(agent_id)
    session_key = agent_id.split(":", 1)[0]
    now = time.monotonic()
    inv = AgentInvocation(
        invocation_id=invocation_id, agent_id=agent_id, role=role,
        session_key=session_key, parent_invocation_id=parent,
        turn_id=turn_id, state=AgentInvocationState.PENDING,
        started_at=now, finished_at=None, task_ref=None,
    )
    token = _invocation_stack.set(stack + (invocation_id,))
    await registry.register_invocation(inv)

    inner: asyncio.Task | None = None
    wait_cancel: asyncio.Task | None = None

    try:
        await hooks.dispatch_start(inv)
        inv.state = AgentInvocationState.RUNNING
        TRACE_STORE.record(
            turn_id, "invocation_start",
            invocation_id=invocation_id, agent_id=agent_id, role=role, parent=parent or "",
        )

        async def _run_in_turn_context() -> T:
            with turn_context(turn_id), agent_context(agent_id):
                return await factory()

        inner = asyncio.create_task(_run_in_turn_context())
        inv.task_ref = inner
        wait_cancel = asyncio.create_task(inv.cancel_requested.wait())
        done, _pending = await asyncio.wait(
            {inner, wait_cancel}, return_when=asyncio.FIRST_COMPLETED,
        )

        if wait_cancel in done:
            inner.cancel()
            try:
                await inner
            except asyncio.CancelledError:
                pass
            inv.state = AgentInvocationState.CANCELLED
            inv.finished_at = time.monotonic()
            TRACE_STORE.record(
                turn_id, "invocation_cancelled",
                invocation_id=invocation_id, agent_id=agent_id, role=role,
            )
            await hooks.dispatch_cancel(inv)
            raise asyncio.CancelledError()

        wait_cancel.cancel()
        try:
            await wait_cancel
        except asyncio.CancelledError:
            pass

        try:
            out: T = inner.result()
        except BaseException as e:
            inv.finished_at = time.monotonic()
            inv.state = AgentInvocationState.FAILED
            TRACE_STORE.record(
                turn_id, "invocation_failed",
                invocation_id=invocation_id, agent_id=agent_id, role=role, error=str(e),
            )
            await hooks.dispatch_error(inv, e)
            raise

        inv.finished_at = time.monotonic()
        inv.state = AgentInvocationState.COMPLETED
        TRACE_STORE.record(
            turn_id, "invocation_completed",
            invocation_id=invocation_id, agent_id=agent_id, role=role,
        )
        await hooks.dispatch_finish(inv)
        return out

    except asyncio.CancelledError:
        if inner is not None and not inner.done():
            inner.cancel()
        if wait_cancel is not None and not wait_cancel.done():
            wait_cancel.cancel()
        pending_cleanup = [t for t in (inner, wait_cancel) if t is not None]
        if pending_cleanup:
            await asyncio.gather(*pending_cleanup, return_exceptions=True)

        if inv.state in (AgentInvocationState.RUNNING, AgentInvocationState.PENDING):
            inv.state = AgentInvocationState.CANCELLED
            inv.finished_at = time.monotonic()
            TRACE_STORE.record(
                turn_id, "invocation_cancelled",
                invocation_id=invocation_id, agent_id=agent_id, role=role,
            )
            await hooks.dispatch_cancel(inv)
        raise
    finally:
        _invocation_stack.reset(token)
        await registry.finish_invocation(invocation_id)


async def run_agent_with_lifecycle(
    *,
    agent: Any,
    prompt: Any,
    agent_id: str,
    registry: AgentRegistry,
    hooks: LifecycleHooks,
    turn_id: str | None,
    message_history: Any,
    usage_limits: Any,
    parent_invocation_id: str | None = None,
    event_stream_handler: Callable[[Any, Any], Awaitable[Any]] | None = None,
) -> Any:
    async def _factory() -> Any:
        return await agent.run(
            prompt,
            message_history=message_history,
            usage_limits=usage_limits,
            event_stream_handler=event_stream_handler,
        )
    return await run_coroutine_with_lifecycle(
        factory=_factory, agent_id=agent_id, registry=registry,
        hooks=hooks, turn_id=turn_id,
        parent_invocation_id=parent_invocation_id,
    )


async def run_agent_stream_with_lifecycle(
    *,
    agent: Any,
    agent_id: str,
    registry: AgentRegistry,
    hooks: LifecycleHooks,
    turn_id: str | None,
    message_history: Any,
    usage_limits: Any,
    prompt: Any,
    consumer: Callable[[Any], Awaitable[T]],
    parent_invocation_id: str | None = None,
) -> T:
    async def _factory() -> T:
        async with agent.run_stream(
            prompt,
            message_history=message_history,
            usage_limits=usage_limits,
        ) as stream:
            return await consumer(stream)
    return await run_coroutine_with_lifecycle(
        factory=_factory, agent_id=agent_id, registry=registry,
        hooks=hooks, turn_id=turn_id,
        parent_invocation_id=parent_invocation_id,
    )
