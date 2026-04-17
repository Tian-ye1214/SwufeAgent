from __future__ import annotations

import functools
import inspect
import logging
import os
import sys
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, List, Optional

os.environ.setdefault("PYTHONUNBUFFERED", "1")

class _ImmediateStreamHandler(logging.StreamHandler):
    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self.flush()


class _ColorFormatter(logging.Formatter):
    _COLORS = {
        logging.DEBUG: "\033[36m",
        logging.INFO: "\033[0m",
        logging.WARNING: "\033[33m",
        logging.ERROR: "\033[31m",
        logging.CRITICAL: "\033[91m",
    }
    _RESET = "\033[0m"

    def __init__(self, fmt: str | None = None, datefmt: str | None = None) -> None:
        super().__init__(fmt, datefmt)
        if sys.platform == "win32":
            try:
                import ctypes

                k = ctypes.windll.kernel32
                k.SetConsoleMode(k.GetStdHandle(-11), 7)
                k.SetConsoleMode(k.GetStdHandle(-12), 7)
            except Exception:
                pass

    def format(self, record: logging.LogRecord) -> str:
        color = self._COLORS.get(record.levelno, self._RESET)
        return f"{color}{super().format(record)}{self._RESET}"


class AppLogger:
    _LOGGER_NAME = "NanoClaw"
    _FMT_CONSOLE = "%(asctime)s | %(levelname)-8s | %(message)s"
    _FMT_FILE = "%(asctime)s | %(levelname)-8s | %(message)s"
    _MAX_NOTIFY_LEN = 400
    _LARGE_KW_KEYS = frozenset(
        {
            "content",
            "tasks_json",
            "text",
            "body",
            "html",
            "message",
            "prompt",
            "user_input",
            "task_description",
            "question",
        }
    )

    def __init__(self, log_dir: Path | str | None = None) -> None:
        self._log_dir = Path(log_dir) if log_dir else Path("./logs")
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._py_logger: logging.Logger | None = None
        self._current_log_file: Path | None = None
        self._session_files: dict[str, Path] = {}
        self._notify_cv: ContextVar[Optional[Callable[[str], None]]] = ContextVar(
            "user_notify_callback", default=None
        )

    @property
    def log_dir(self) -> Path:
        return self._log_dir

    def debug(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        self.get_logger().debug(msg, *args, **kwargs)

    def info(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        self.get_logger().info(msg, *args, **kwargs)

    def warning(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        self.get_logger().warning(msg, *args, **kwargs)

    def error(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        self.get_logger().error(msg, *args, **kwargs)

    def get_logger(self) -> logging.Logger:
        """返回已配置好的标准库 `Logger`（懒初始化，仅控制台）。"""
        if self._py_logger is None:
            lg = logging.getLogger(self._LOGGER_NAME)
            lg.setLevel(logging.DEBUG)
            lg.propagate = False
            if not lg.handlers:
                lg.addHandler(self._make_console_handler())
            self._py_logger = lg
        return self._py_logger

    def set_user_notify_callback(self, fn: Optional[Callable[[str], None]]) -> None:
        """注册或清除（``None``）进度回调；回调异常会被吞掉，不中断主流程。"""
        self._notify_cv.set(fn)

    def setup_task_logger(self, task_name: str = "task") -> logging.Logger:
        """按任务名创建带时间戳的日志文件，并重置为「控制台 + 该文件」。"""
        safe = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in task_name)[:50]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self._log_dir / f"{safe}_{ts}.log"
        self._rebuild_handlers(path)
        self.get_logger().info("日志文件已创建: %s", path)
        return self.get_logger()

    def setup_session_logger(self, session_name: str) -> logging.Logger:
        """同一会话名复用同一日志文件（不含时间戳）。"""
        safe = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in session_name)[:50]
        if safe not in self._session_files:
            self._session_files[safe] = self._log_dir / f"{safe}.log"
        path = self._session_files[safe]
        if self._current_log_file == path and self._py_logger is not None:
            return self.get_logger()
        self._rebuild_handlers(path)
        return self.get_logger()

    def wrap_tools_for_user_notify(self, tools: List[Any]) -> List[Any]:
        """包装可调用工具：每次调用前记录「工具名 + 参数摘要」。"""
        if not tools:
            return tools
        out: List[Any] = []
        for t in tools:
            if t is not None and callable(t) and not inspect.isclass(t):
                out.append(self._wrap_tool(t))
            else:
                out.append(t)
        return out

    def _make_console_handler(self) -> logging.Handler:
        h = _ImmediateStreamHandler(sys.stderr)
        h.setLevel(logging.DEBUG)
        h.setFormatter(_ColorFormatter(self._FMT_CONSOLE, datefmt="%H:%M:%S"))
        return h

    def _make_file_handler(self, log_filepath: Path) -> logging.Handler:
        h = logging.FileHandler(log_filepath, encoding="utf-8")
        h.setLevel(logging.DEBUG)
        h.setFormatter(logging.Formatter(self._FMT_FILE, datefmt="%Y-%m-%d %H:%M:%S"))
        return h

    def _rebuild_handlers(self, log_filepath: Path) -> None:
        lg = logging.getLogger(self._LOGGER_NAME)
        lg.setLevel(logging.DEBUG)
        lg.propagate = False
        lg.handlers.clear()
        lg.addHandler(self._make_console_handler())
        lg.addHandler(self._make_file_handler(log_filepath))
        self._py_logger = lg
        self._current_log_file = log_filepath

    @staticmethod
    def _run_notify(fn: Optional[Callable[[str], None]], text: str) -> None:
        if not fn:
            return
        try:
            fn(text)
        except Exception:
            pass

    @staticmethod
    def _truncate_repr(v: Any, max_len: int = 96) -> str:
        s = repr(v)
        if len(s) > max_len:
            return s[: max_len - 1] + "…"
        return s

    def _format_kwargs_for_notify(self, kwargs: dict) -> str:
        parts: list[str] = []
        for k, v in list(kwargs.items())[:10]:
            if k in self._LARGE_KW_KEYS and isinstance(v, str) and len(v) > 120:
                sv = f"<{len(v)} chars>"
            else:
                sv = self._truncate_repr(v, 100)
            parts.append(f"{k}={sv}")
        return ", ".join(parts)

    @staticmethod
    def _is_likely_bound_self(x: Any) -> bool:
        n = getattr(x.__class__, "__name__", "")
        return n in ("BasicToolkit", "TaskManager", "SkillsToolkit", "AgentSystem")

    def _format_tool_args_for_notify(self, args: tuple, kwargs: dict) -> str:
        if kwargs:
            return self._format_kwargs_for_notify(kwargs)
        if not args:
            return ""
        al = list(args)
        if al and self._is_likely_bound_self(al[0]):
            al = al[1:]
        if not al:
            return ""
        if len(al) == 1:
            return self._truncate_repr(al[0], 120)
        return f"({len(al)} args) " + ", ".join(self._truncate_repr(a, 60) for a in al[:4])

    def _emit_tool_line(self, tool_name: str, detail: str) -> None:
        detail = (detail or "").strip()
        line = f"\U0001f527 {tool_name}"
        if detail:
            if len(detail) > 280:
                detail = detail[:277] + "…"
            line += f" · {detail}"
        if len(line) > self._MAX_NOTIFY_LEN:
            line = line[: self._MAX_NOTIFY_LEN - 1] + "…"
        cb = self._notify_cv.get()
        log = self.get_logger()
        if cb:
            log.info(line)
            self._run_notify(cb, line)
        else:
            log.debug(line)

    def _wrap_tool(self, fn: Callable) -> Callable:
        if getattr(fn, "_notify_tool_wrapped", False):
            return fn
        name = getattr(fn, "__name__", "tool")

        def emit(args: tuple, kwargs: dict) -> None:
            self._emit_tool_line(name, self._format_tool_args_for_notify(args, kwargs))

        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any):
                emit(args, kwargs)
                return await fn(*args, **kwargs)

            async_wrapper._notify_tool_wrapped = True  # type: ignore[attr-defined]
            return async_wrapper

        @functools.wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any):
            emit(args, kwargs)
            return fn(*args, **kwargs)

        sync_wrapper._notify_tool_wrapped = True  # type: ignore[attr-defined]
        return sync_wrapper


_default = AppLogger()
LOG_DIR = _default.log_dir


def get_logger() -> logging.Logger:
    return _default.get_logger()


def set_user_notify_callback(fn: Optional[Callable[[str], None]]) -> None:
    _default.set_user_notify_callback(fn)


def setup_task_logger(task_name: str = "task") -> logging.Logger:
    return _default.setup_task_logger(task_name)


def setup_session_logger(session_name: str) -> logging.Logger:
    return _default.setup_session_logger(session_name)


def wrap_tools_for_user_notify(tools: List[Any]) -> List[Any]:
    return _default.wrap_tools_for_user_notify(tools)


def debug(msg: Any, *args: Any, **kwargs: Any) -> None:
    _default.debug(msg, *args, **kwargs)


def info(msg: Any, *args: Any, **kwargs: Any) -> None:
    _default.info(msg, *args, **kwargs)


def warning(msg: Any, *args: Any, **kwargs: Any) -> None:
    _default.warning(msg, *args, **kwargs)


def error(msg: Any, *args: Any, **kwargs: Any) -> None:
    _default.error(msg, *args, **kwargs)