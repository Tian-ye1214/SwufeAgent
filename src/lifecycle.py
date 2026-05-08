from __future__ import annotations

import asyncio
import inspect
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeVar

import logger

T = TypeVar("T")


class AgentRunState(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AgentRun:
    run_id: str
    role: str
    parent_run_id: str | None
    turn_id: str | None
    state: AgentRunState
    started_at: float
    finished_at: float | None
    error: str | None
    task_ref: asyncio.Task[Any] | None
    cancel_requested: asyncio.Event = field(default_factory=asyncio.Event)


class AgentRegistry:
    def __init__(self) -> None:
        self._runs: dict[str, AgentRun] = {}
        self._lock = asyncio.Lock()

    async def register(self, run: AgentRun) -> None:
        async with self._lock:
            self._runs[run.run_id] = run

    async def unregister(self, run_id: str) -> None:
        async with self._lock:
            self._runs.pop(run_id, None)

    async def get(self, run_id: str) -> AgentRun | None:
        async with self._lock:
            return self._runs.get(run_id)

    async def list_runs(self) -> list[AgentRun]:
        async with self._lock:
            return list(self._runs.values())

    async def cancel(self, run_id: str) -> bool:
        async with self._lock:
            r = self._runs.get(run_id)
        if r is None:
            return False
        r.cancel_requested.set()
        t = r.task_ref
        if t is not None and not t.done():
            t.cancel()
        return True

    async def cancel_all(self) -> int:
        async with self._lock:
            ids = list(self._runs.keys())
        n = 0
        for rid in ids:
            if await self.cancel(rid):
                n += 1
        if ids:
            logger.info("[lifecycle] cancel_all requested for %s run(s)", len(ids))
        return n


class LifecycleHooks:
    def __init__(self) -> None:
        self._on_start: list[Callable[[AgentRun], Any]] = []
        self._on_finish: list[Callable[[AgentRun], Any]] = []
        self._on_error: list[Callable[[AgentRun, BaseException], Any]] = []
        self._on_cancel: list[Callable[[AgentRun], Any]] = []

    def add_on_start(self, fn: Callable[[AgentRun], Any]) -> None:
        self._on_start.append(fn)

    def add_on_finish(self, fn: Callable[[AgentRun], Any]) -> None:
        self._on_finish.append(fn)

    def add_on_error(self, fn: Callable[[AgentRun, BaseException], Any]) -> None:
        self._on_error.append(fn)

    def add_on_cancel(self, fn: Callable[[AgentRun], Any]) -> None:
        self._on_cancel.append(fn)

    async def _dispatch(self, callbacks: list[Callable[..., Any]], *args: Any) -> None:
        for fn in callbacks:
            try:
                r = fn(*args)
                if inspect.isawaitable(r):
                    await r
            except Exception as e:
                logger.warning("lifecycle hook %s failed: %s", getattr(fn, "__name__", fn), e)

    async def dispatch_start(self, run: AgentRun) -> None:
        await self._dispatch(self._on_start, run)

    async def dispatch_finish(self, run: AgentRun) -> None:
        await self._dispatch(self._on_finish, run)

    async def dispatch_error(self, run: AgentRun, exc: BaseException) -> None:
        await self._dispatch(self._on_error, run, exc)

    async def dispatch_cancel(self, run: AgentRun) -> None:
        await self._dispatch(self._on_cancel, run)


def register_default_lifecycle_logging(hooks: LifecycleHooks) -> None:
    def _on_start(run: AgentRun) -> None:
        logger.debug(
            "[lifecycle] start run_id=%s role=%s parent=%s turn=%s",
            run.run_id,
            run.role,
            run.parent_run_id,
            run.turn_id,
        )

    def _on_finish(run: AgentRun) -> None:
        logger.debug(
            "[lifecycle] finish run_id=%s role=%s state=%s",
            run.run_id,
            run.role,
            run.state.value,
        )

    def _on_error(run: AgentRun, exc: BaseException) -> None:
        logger.warning(
            "[lifecycle] error run_id=%s role=%s: %s",
            run.run_id,
            run.role,
            exc,
        )

    def _on_cancel(run: AgentRun) -> None:
        logger.debug("[lifecycle] cancel run_id=%s role=%s", run.run_id, run.role)

    hooks.add_on_start(_on_start)
    hooks.add_on_finish(_on_finish)
    hooks.add_on_error(_on_error)
    hooks.add_on_cancel(_on_cancel)


async def run_coroutine_with_lifecycle(
    *,
    factory: Callable[[], Awaitable[T]],
    role: str,
    registry: AgentRegistry,
    hooks: LifecycleHooks,
    parent_run_id: str | None,
    turn_id: str | None,
) -> T:
    run_id = str(uuid.uuid4())
    now = time.monotonic()
    run = AgentRun(
        run_id=run_id,
        role=role,
        parent_run_id=parent_run_id,
        turn_id=turn_id,
        state=AgentRunState.PENDING,
        started_at=now,
        finished_at=None,
        error=None,
        task_ref=None,
    )
    await registry.register(run)
    try:
        await hooks.dispatch_start(run)
        run.state = AgentRunState.RUNNING
        inner = asyncio.create_task(factory())
        run.task_ref = inner
        wait_cancel = asyncio.create_task(run.cancel_requested.wait())
        done, _pending = await asyncio.wait(
            {inner, wait_cancel},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if wait_cancel in done:
            inner.cancel()
            try:
                await inner
            except asyncio.CancelledError:
                pass
            run.state = AgentRunState.CANCELLED
            run.finished_at = time.monotonic()
            await hooks.dispatch_cancel(run)
            raise asyncio.CancelledError()
        wait_cancel.cancel()
        try:
            await wait_cancel
        except asyncio.CancelledError:
            pass
        try:
            out: T = inner.result()
        except BaseException as e:
            run.finished_at = time.monotonic()
            run.error = str(e)
            run.state = AgentRunState.FAILED
            await hooks.dispatch_error(run, e)
            raise
        run.finished_at = time.monotonic()
        run.state = AgentRunState.COMPLETED
        await hooks.dispatch_finish(run)
        return out
    finally:
        await registry.unregister(run_id)


async def run_agent_with_lifecycle(
    *,
    agent: Any,
    prompt: Any,
    role: str,
    registry: AgentRegistry,
    hooks: LifecycleHooks,
    parent_run_id: str | None,
    turn_id: str | None,
    message_history: Any,
    usage_limits: Any,
) -> Any:
    async def _factory() -> Any:
        return await agent.run(
            prompt,
            message_history=message_history,
            usage_limits=usage_limits,
        )

    return await run_coroutine_with_lifecycle(
        factory=_factory,
        role=role,
        registry=registry,
        hooks=hooks,
        parent_run_id=parent_run_id,
        turn_id=turn_id,
    )


async def run_agent_stream_with_lifecycle(
    *,
    agent: Any,
    role: str,
    registry: AgentRegistry,
    hooks: LifecycleHooks,
    parent_run_id: str | None,
    turn_id: str | None,
    message_history: Any,
    usage_limits: Any,
    prompt: Any,
    consumer: Callable[[Any], Awaitable[T]],
) -> T:
    async def _factory() -> T:
        async with agent.run_stream(
            prompt,
            message_history=message_history,
            usage_limits=usage_limits,
        ) as stream:
            return await consumer(stream)

    return await run_coroutine_with_lifecycle(
        factory=_factory,
        role=role,
        registry=registry,
        hooks=hooks,
        parent_run_id=parent_run_id,
        turn_id=turn_id,
    )
