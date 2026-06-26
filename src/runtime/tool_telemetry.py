from __future__ import annotations

import functools
import inspect
import time
from contextvars import ContextVar
from typing import Any, Callable

from infra import logger
from runtime.runtime_state import TRACE_STORE, AgentRunPolicy, current_short_agent_id, current_turn_id

_notify_callback: ContextVar[Callable[[str], None] | None] = ContextVar("user_notify_callback", default=None)


def set_user_notify_callback(fn: Callable[[str], None] | None) -> None:
    _notify_callback.set(fn)


def wrap_tools_for_user_notify(
    tools: list[Any], *, policy: AgentRunPolicy | None = None
) -> list[Any]:
    """工厂：给每个可调用工具套壳——调用时发 🔧 通知 + 记一条 TRACE 事件。"""
    if not tools:
        return tools
    return [_wrap(t, policy) if callable(t) and not inspect.isclass(t) else t for t in tools]


def _notify(name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
    """把一次调用以 "🔧 名字 [agent] · 参数" 推给用户；无回调则降级为 debug。"""
    def brief(v: Any) -> str:
        text = repr(v)
        return text if len(text) <= 80 else f"{text[:79]}…"

    parts = [f"🔧 {name}"]
    if agent_id := current_short_agent_id():
        parts.append(f"[{agent_id}]")
    if kwargs:
        parts.append(", ".join(f"{k}={brief(v)}" for k, v in list(kwargs.items())[:5]))
    elif args:
        parts.append(", ".join(brief(v) for v in args[:3]))
    line = " · ".join(parts)

    callback = _notify_callback.get()
    if callback:
        logger.info(line)
        try:
            callback(line)
        except Exception:
            logger.debug("user_notify_callback 执行失败", exc_info=True)
    else:
        logger.debug(line)


def _record(
    name: str, t0: float, success: bool, result: Any = None, error: BaseException | None = None
) -> None:
    """把一次工具调用的结果写入 TRACE_STORE。"""
    text = result if isinstance(result, str) else repr(result)
    TRACE_STORE.record(
        current_turn_id(),
        "tool_call",
        tool_name=name,
        agent_id=current_short_agent_id() or "",
        success=success,
        elapsed_ms=int((time.monotonic() - t0) * 1000),
        output_chars=len(text or ""),
        error=f"{type(error).__name__}: {error}" if error else "",
    )


def _run_wrapped(
    fn: Callable[..., Any],
    policy: AgentRunPolicy | None,
    name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    t0 = time.monotonic()
    _notify(name, args, kwargs)
    try:
        result = fn(*args, **kwargs)
    except Exception as e:
        _record(name, t0, False, error=e)
        raise
    if policy is not None and isinstance(result, str):
        result = policy.truncate_text(result)
    _record(name, t0, True, result=result)
    return result


def _wrap(fn: Callable[..., Any], policy: AgentRunPolicy | None) -> Callable[..., Any]:
    """给单个工具套壳：调用前发通知，调用后记事件；区分协程与普通函数。"""
    if getattr(fn, "_notify_tool_wrapped", False):
        return fn
    name = getattr(fn, "__name__", "tool")

    if inspect.iscoroutinefunction(fn):
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            t0 = time.monotonic()
            _notify(name, args, kwargs)
            try:
                result = await fn(*args, **kwargs)
            except Exception as e:
                _record(name, t0, False, error=e)
                raise
            if policy is not None and isinstance(result, str):
                result = policy.truncate_text(result)
            _record(name, t0, True, result=result)
            return result
    else:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return _run_wrapped(fn, policy, name, args, kwargs)

    wrapper._notify_tool_wrapped = True  # type: ignore[attr-defined]
    return wrapper
