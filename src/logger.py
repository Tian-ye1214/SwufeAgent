from __future__ import annotations

import functools
import inspect
import logging
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator

from runtime_state import AgentRunPolicy, TRACE_STORE, current_short_agent_id, current_turn_id

_stm_ingest_console_quiet: ContextVar[int] = ContextVar("stm_ingest_console_quiet", default=0)
_LOG_LEVEL_STYLES = {
    logging.DEBUG: "dim cyan",
    logging.INFO: "green",
    logging.WARNING: "yellow",
    logging.ERROR: "bold red",
    logging.CRITICAL: "bold white on red",
}


class _StmIngestConsoleQuietFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if _stm_ingest_console_quiet.get() <= 0:
            return True
        return record.levelno >= logging.WARNING


class _OutputSinkHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            from rich.text import Text

            from cli.output import emit_renderable

            style = _LOG_LEVEL_STYLES.get(record.levelno, "")
            emit_renderable(Text(self.format(record), style=style))
        except Exception:
            self.handleError(record)


class _SessionFileHandler(logging.Handler):
    def __init__(self, manager: "LoggerManager") -> None:
        super().__init__(logging.DEBUG)
        self._manager = manager

    def emit(self, record: logging.LogRecord) -> None:
        session_name = self._manager._session_log_name.get()
        if not session_name:
            return
        try:
            path = self._manager._session_files.setdefault(
                session_name,
                self._manager._log_dir / f"{session_name}.log",
            )
            line = self.format(record)
            path.parent.mkdir(parents=True, exist_ok=True)
            with self._manager._session_write_lock:
                with path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
        except Exception:
            self.handleError(record)


class LoggerManager:
    """进程内单例：控制台 + 可选文件；业务侧用模块级 `logger.info` 等。"""

    def __init__(self, log_dir: Path | str = "./logs", logger_name: str = "NanoClaw") -> None:
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._logger_name = logger_name
        self._logger: logging.Logger | None = None
        self._current_log_file: Path | None = None
        self._session_files: dict[str, Path] = {}
        self._session_log_name: ContextVar[str | None] = ContextVar(
            f"{logger_name}_session_log_name", default=None
        )
        self._session_write_lock = threading.Lock()
        self._notify_callback: ContextVar[Callable[[str], None] | None] = ContextVar(
            "user_notify_callback", default=None
        )

    @property
    def log_dir(self) -> Path:
        return self._log_dir

    @staticmethod
    def _release_handlers(logger: logging.Logger) -> None:
        for h in list(logger.handlers):
            logger.removeHandler(h)
            h.close()

    def get_logger(self) -> logging.Logger:
        if self._logger is None:
            logger = logging.getLogger(self._logger_name)
            logger.setLevel(logging.DEBUG)
            logger.propagate = False
            self._release_handlers(logger)
            logger.addHandler(self._build_console_handler())
            logger.addHandler(self._build_session_file_handler())
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

    @contextmanager
    def session_log_context(self, session_name: str) -> Iterator[None]:
        self.get_logger()
        token = self._session_log_name.set(self._sanitize_name(session_name))
        try:
            yield
        finally:
            self._session_log_name.reset(token)

    def set_user_notify_callback(self, fn: Callable[[str], None] | None) -> None:
        self._notify_callback.set(fn)

    def wrap_tools_for_user_notify(
        self,
        tools: list[Any],
        *,
        policy: AgentRunPolicy | None = None,
    ) -> list[Any]:
        if not tools:
            return tools
        wrapped: list[Any] = []
        for tool in tools:
            if tool is not None and callable(tool) and not inspect.isclass(tool):
                wrapped.append(self._wrap_tool(tool, policy=policy))
            else:
                wrapped.append(tool)
        return wrapped

    def _build_console_handler(self) -> logging.Handler:
        handler = _OutputSinkHandler()
        handler.setLevel(logging.DEBUG)
        handler.addFilter(_StmIngestConsoleQuietFilter())
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", "%H:%M:%S")
        )
        return handler

    @contextmanager
    def stm_ingest_console_quiet(self) -> Iterator[None]:
        """短期记忆后台入库时压制控制台 INFO/DEBUG，文件日志不受影响。"""
        token = _stm_ingest_console_quiet.set(_stm_ingest_console_quiet.get() + 1)
        try:
            yield
        finally:
            _stm_ingest_console_quiet.reset(token)

    def _build_file_handler(self, log_file: Path) -> logging.Handler:
        handler = logging.FileHandler(log_file, encoding="utf-8")
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", "%Y-%m-%d %H:%M:%S")
        )
        return handler

    def _build_session_file_handler(self) -> logging.Handler:
        handler = _SessionFileHandler(self)
        handler.addFilter(_StmIngestConsoleQuietFilter())
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
        self._release_handlers(logger)
        logger.addHandler(self._build_console_handler())
        logger.addHandler(self._build_session_file_handler())
        logger.addHandler(self._build_file_handler(log_file))
        self._current_log_file = log_file
        if announce:
            logger.info("日志文件已创建: %s", log_file)
        return logger

    def info_file_only(self, msg: str, *args: Any) -> None:
        """仅写入文件 Handler，避免与 Rich 控制台输出重复。"""
        log = self.get_logger()
        formatted = msg % args if args else msg
        record = log.makeRecord(log.name, logging.INFO, "(cli)", 0, formatted, (), None)
        for h in log.handlers:
            if isinstance(h, logging.FileHandler):
                h.handle(record)

    def _safe_repr(self, value: Any, max_len: int = 120) -> str:
        text = repr(value)
        return text if len(text) <= max_len else f"{text[: max_len - 1]}…"

    def _emit_tool_notify(self, tool_name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        pieces = [f"🔧 {tool_name}"]
        agent_id = current_short_agent_id()
        if agent_id:
            pieces.append(f"[{agent_id}]")
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

    def _normalize_tool_result(self, value: Any, policy: AgentRunPolicy | None) -> Any:
        if isinstance(value, str) and policy is not None:
            return policy.truncate_text(value)
        return value

    def _record_tool_event(
        self,
        tool_name: str,
        started: float,
        success: bool,
        result: Any = None,
        error: BaseException | None = None,
    ) -> None:
        text = result if isinstance(result, str) else repr(result)
        TRACE_STORE.record(
            current_turn_id(),
            "tool_call",
            tool_name=tool_name,
            agent_id=current_short_agent_id(),
            success=success,
            elapsed_ms=int((time.monotonic() - started) * 1000),
            output_chars=len(text or ""),
            error=f"{type(error).__name__}: {error}" if error else "",
        )

    def _wrap_tool(
        self,
        fn: Callable[..., Any],
        *,
        policy: AgentRunPolicy | None = None,
    ) -> Callable[..., Any]:
        if getattr(fn, "_notify_tool_wrapped", False):
            return fn
        tool_name = getattr(fn, "__name__", "tool")

        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                started = time.monotonic()
                self._emit_tool_notify(tool_name, args, kwargs)
                try:
                    result = await fn(*args, **kwargs)
                except Exception as e:
                    self._record_tool_event(tool_name, started, False, error=e)
                    raise
                result = self._normalize_tool_result(result, policy)
                self._record_tool_event(tool_name, started, True, result=result)
                return result

            setattr(async_wrapper, "_notify_tool_wrapped", True)
            return async_wrapper

        @functools.wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            started = time.monotonic()
            self._emit_tool_notify(tool_name, args, kwargs)
            try:
                result = fn(*args, **kwargs)
            except Exception as e:
                self._record_tool_event(tool_name, started, False, error=e)
                raise
            result = self._normalize_tool_result(result, policy)
            self._record_tool_event(tool_name, started, True, result=result)
            return result

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
session_log_context = _MANAGER.session_log_context
set_user_notify_callback = _MANAGER.set_user_notify_callback
wrap_tools_for_user_notify = _MANAGER.wrap_tools_for_user_notify
info_file_only = _MANAGER.info_file_only
stm_ingest_console_quiet = _MANAGER.stm_ingest_console_quiet

__all__ = [
    "LOG_DIR",
    "debug",
    "info",
    "info_file_only",
    "warning",
    "error",
    "setup_task_logger",
    "setup_session_logger",
    "session_log_context",
    "set_user_notify_callback",
    "wrap_tools_for_user_notify",
    "stm_ingest_console_quiet",
]
