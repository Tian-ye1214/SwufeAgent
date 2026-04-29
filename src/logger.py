from __future__ import annotations

import functools
import inspect
import logging
import sys
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


class LoggerManager:
    """进程内单例：控制台 + 可选文件；业务侧用模块级 `logger.info` 等。"""

    def __init__(self, log_dir: Path | str = "./logs", logger_name: str = "NanoClaw") -> None:
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._logger_name = logger_name
        self._logger: logging.Logger | None = None
        self._current_log_file: Path | None = None
        self._session_files: dict[str, Path] = {}
        self._notify_callback: ContextVar[Callable[[str], None] | None] = ContextVar(
            "user_notify_callback", default=None
        )

    @property
    def log_dir(self) -> Path:
        return self._log_dir

    def get_logger(self) -> logging.Logger:
        if self._logger is None:
            logger = logging.getLogger(self._logger_name)
            logger.setLevel(logging.DEBUG)
            logger.propagate = False
            logger.handlers.clear()
            logger.addHandler(self._build_console_handler())
            self._logger = logger
        return self._logger

    def setup_task_logger(self, task_name: str = "task") -> logging.Logger:
        safe_name = self._sanitize_name(task_name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self._log_dir / f"{safe_name}_{timestamp}.log"
        return self._switch_to_file(log_file, announce=True)

    def setup_session_logger(self, session_name: str) -> logging.Logger:
        safe_name = self._sanitize_name(session_name)
        log_file = self._session_files.setdefault(safe_name, self._log_dir / f"{safe_name}.log")
        return self._switch_to_file(log_file, announce=False)

    def set_user_notify_callback(self, fn: Callable[[str], None] | None) -> None:
        self._notify_callback.set(fn)

    def wrap_tools_for_user_notify(self, tools: list[Any]) -> list[Any]:
        if not tools:
            return tools
        wrapped: list[Any] = []
        for tool in tools:
            if tool is not None and callable(tool) and not inspect.isclass(tool):
                wrapped.append(self._wrap_tool(tool))
            else:
                wrapped.append(tool)
        return wrapped

    def _build_console_handler(self) -> logging.Handler:
        handler = logging.StreamHandler(stream=sys.stderr)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", "%H:%M:%S"))
        return handler

    def _build_file_handler(self, log_file: Path) -> logging.Handler:
        handler = logging.FileHandler(log_file, encoding="utf-8")
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", "%Y-%m-%d %H:%M:%S")
        )
        return handler

    def _sanitize_name(self, name: str) -> str:
        cleaned = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in name)
        return cleaned[:50] or "task"

    def _switch_to_file(self, log_file: Path, announce: bool) -> logging.Logger:
        logger = self.get_logger()
        if self._current_log_file == log_file:
            return logger
        logger.handlers.clear()
        logger.addHandler(self._build_console_handler())
        logger.addHandler(self._build_file_handler(log_file))
        self._current_log_file = log_file
        if announce:
            logger.info("日志文件已创建: %s", log_file)
        return logger

    def _safe_repr(self, value: Any, max_len: int = 120) -> str:
        text = repr(value)
        return text if len(text) <= max_len else f"{text[: max_len - 1]}…"

    def _emit_tool_notify(self, tool_name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        pieces = [f"🔧 {tool_name}"]
        if kwargs:
            summary = ", ".join(f"{k}={self._safe_repr(v, 80)}" for k, v in list(kwargs.items())[:5])
            pieces.append(summary)
        elif args:
            summary = ", ".join(self._safe_repr(v, 80) for v in list(args)[:3])
            pieces.append(summary)
        line = " · ".join(pieces)
        callback = self._notify_callback.get()
        log = self.get_logger()
        if callback:
            log.info(line)
            try:
                callback(line)
            except Exception:
                log.debug("user_notify_callback 执行失败", exc_info=True)
        else:
            log.debug(line)

    def _wrap_tool(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        if getattr(fn, "_notify_tool_wrapped", False):
            return fn
        tool_name = getattr(fn, "__name__", "tool")

        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                self._emit_tool_notify(tool_name, args, kwargs)
                return await fn(*args, **kwargs)

            setattr(async_wrapper, "_notify_tool_wrapped", True)
            return async_wrapper

        @functools.wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            self._emit_tool_notify(tool_name, args, kwargs)
            return fn(*args, **kwargs)

        setattr(sync_wrapper, "_notify_tool_wrapped", True)
        return sync_wrapper


_MANAGER = LoggerManager()
LOG_DIR = _MANAGER.log_dir

_root_logger = _MANAGER.get_logger()
debug = _root_logger.debug
info = _root_logger.info
warning = _root_logger.warning
error = _root_logger.error

setup_task_logger = _MANAGER.setup_task_logger
setup_session_logger = _MANAGER.setup_session_logger
set_user_notify_callback = _MANAGER.set_user_notify_callback
wrap_tools_for_user_notify = _MANAGER.wrap_tools_for_user_notify

__all__ = [
    "LOG_DIR",
    "debug",
    "info",
    "warning",
    "error",
    "setup_task_logger",
    "setup_session_logger",
    "set_user_notify_callback",
    "wrap_tools_for_user_notify",
]