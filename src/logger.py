import logging
import sys
import os
import contextvars
from datetime import datetime
from pathlib import Path
import functools
import inspect
from typing import Any, Callable, List, Optional


os.environ["PYTHONUNBUFFERED"] = "1"
LOG_DIR = Path("./logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

_logger: logging.Logger | None = None
_current_log_file: Path | None = None

_user_notify_cv: contextvars.ContextVar[Optional[Callable[[str], None]]] = contextvars.ContextVar(
    "user_notify_callback", default=None
)
_MAX_NOTIFY_LEN = 400

_FMT_CONSOLE = "%(asctime)s | %(levelname)-8s | %(message)s"
_FMT_FILE = "%(asctime)s | %(levelname)-8s | %(message)s"


def set_user_notify_callback(fn: Optional[Callable[[str], None]]) -> None:
    """注册或清除（传 None）用户可见进度回调，如 QQ 发消息。"""
    _user_notify_cv.set(fn)


def _run_user_notify(fn: Optional[Callable[[str], None]], text: str) -> None:
    """第三方回调失败时不应打断 Agent；仅此一处吞异常。"""
    if not fn:
        return
    try:
        fn(text)
    except Exception:
        pass


def notify_user(msg: str) -> None:
    """INFO 日志；若已注册回调则同步推送（如 QQ）。"""
    if not (msg and str(msg).strip()):
        return
    text = str(msg).strip()
    if len(text) > _MAX_NOTIFY_LEN:
        text = text[: _MAX_NOTIFY_LEN - 1] + "…"
    get_logger().info(text)
    _run_user_notify(_user_notify_cv.get(), text)


# 大字段参数名：摘要里用长度代替正文，避免刷屏与超长 QQ 消息
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


def _truncate_repr(v: Any, max_len: int = 96) -> str:
    s = repr(v)
    if len(s) > max_len:
        return s[: max_len - 1] + "…"
    return s


def _format_kwargs_for_notify(kwargs: dict) -> str:
    parts: list[str] = []
    for k, v in list(kwargs.items())[:10]:
        if k in _LARGE_KW_KEYS and isinstance(v, str) and len(v) > 120:
            sv = f"<{len(v)} chars>"
        else:
            sv = _truncate_repr(v, 100)
        parts.append(f"{k}={sv}")
    return ", ".join(parts)


def _is_likely_tool_self(x: Any) -> bool:
    n = getattr(x.__class__, "__name__", "")
    return n in ("BasicToolkit", "TaskManager", "SkillsToolkit", "AgentSystem")


def _format_tool_args_for_notify(args: tuple, kwargs: dict) -> str:
    if kwargs:
        return _format_kwargs_for_notify(kwargs)
    if not args:
        return ""
    args = list(args)
    if args and _is_likely_tool_self(args[0]):
        args = args[1:]
    if not args:
        return ""
    if len(args) == 1:
        return _truncate_repr(args[0], 120)
    return f"({len(args)} args) " + ", ".join(_truncate_repr(a, 60) for a in args[:4])


def notify_tool_usage(tool_name: str, detail: str = "") -> None:
    """有 QQ 回调时 INFO+推送；否则 DEBUG，避免 CLI 刷屏。"""
    detail = (detail or "").strip()
    line = f"🔧 {tool_name}"
    if detail:
        if len(detail) > 280:
            detail = detail[:277] + "…"
        line += f" · {detail}"
    if len(line) > _MAX_NOTIFY_LEN:
        line = line[: _MAX_NOTIFY_LEN - 1] + "…"
    fn = _user_notify_cv.get()
    log = get_logger()
    if fn:
        log.info(line)
        _run_user_notify(fn, line)
    else:
        log.debug(line)


def _wrap_one_tool_for_notify(fn: Callable) -> Callable:
    if getattr(fn, "_notify_tool_wrapped", False):
        return fn
    name = getattr(fn, "__name__", "tool")

    def _emit(args: tuple, kwargs: dict) -> None:
        detail = _format_tool_args_for_notify(args, kwargs)
        notify_tool_usage(name, detail)

    if inspect.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def async_wrapper(*args: Any, **kwargs: Any):
            _emit(args, kwargs)
            return await fn(*args, **kwargs)

        async_wrapper._notify_tool_wrapped = True  # type: ignore[attr-defined]
        return async_wrapper

    @functools.wraps(fn)
    def sync_wrapper(*args: Any, **kwargs: Any):
        _emit(args, kwargs)
        return fn(*args, **kwargs)

    sync_wrapper._notify_tool_wrapped = True  # type: ignore[attr-defined]
    return sync_wrapper


def wrap_tools_for_user_notify(tools: List[Any]) -> List[Any]:
    """包装 Agent 工具列表，调用时 notify_tool_usage。"""
    if not tools:
        return tools
    out: List[Any] = []
    for t in tools:
        if t is not None and callable(t) and not inspect.isclass(t):
            out.append(_wrap_one_tool_for_notify(t))
        else:
            out.append(t)
    return out


class ImmediateStreamHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()


class ColorFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: "\033[36m",
        logging.INFO: "\033[0m",
        logging.WARNING: "\033[33m",
        logging.ERROR: "\033[31m",
        logging.CRITICAL: "\033[91m",
    }
    RESET = "\033[0m"

    def __init__(self, fmt=None, datefmt=None):
        super().__init__(fmt, datefmt)
        if sys.platform == "win32":
            try:
                import ctypes

                k = ctypes.windll.kernel32
                k.SetConsoleMode(k.GetStdHandle(-11), 7)
                k.SetConsoleMode(k.GetStdHandle(-12), 7)
            except Exception:
                pass

    def format(self, record):
        color = self.COLORS.get(record.levelno, self.RESET)
        message = super().format(record)
        return f"{color}{message}{self.RESET}"


def _make_console_handler() -> logging.Handler:
    h = ImmediateStreamHandler(sys.stderr)
    h.setLevel(logging.DEBUG)
    h.setFormatter(ColorFormatter(_FMT_CONSOLE, datefmt="%H:%M:%S"))
    return h


def _make_file_handler(log_filepath: Path) -> logging.Handler:
    h = logging.FileHandler(log_filepath, encoding="utf-8")
    h.setLevel(logging.DEBUG)
    h.setFormatter(logging.Formatter(_FMT_FILE, datefmt="%Y-%m-%d %H:%M:%S"))
    return h


def get_logger() -> logging.Logger:
    """全局 AgentDemo logger（懒初始化控制台 handler）。"""
    global _logger
    if _logger is None:
        _logger = logging.getLogger("AgentDemo")
        _logger.setLevel(logging.DEBUG)
        _logger.propagate = False
        if not _logger.handlers:
            _logger.addHandler(_make_console_handler())
    return _logger


def setup_task_logger(task_name: str = "task") -> logging.Logger:
    """按任务名创建带时间戳的日志文件，并重置控制台与文件 handler。"""
    global _logger, _current_log_file

    safe = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in task_name)[:50]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filepath = LOG_DIR / f"{safe}_{ts}.log"

    _logger = logging.getLogger("AgentDemo")
    _logger.setLevel(logging.DEBUG)
    _logger.propagate = False
    _logger.handlers.clear()
    _logger.addHandler(_make_console_handler())
    _logger.addHandler(_make_file_handler(log_filepath))

    _current_log_file = log_filepath
    _logger.info(f"日志文件已创建: {log_filepath}")
    return _logger


_session_log_files: dict[str, Path] = {}


def setup_session_logger(session_name: str) -> logging.Logger:
    """同一会话名复用同一日志文件（不含时间戳）。"""
    global _logger, _current_log_file

    safe_name = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in session_name)[:50]
    if safe_name not in _session_log_files:
        _session_log_files[safe_name] = LOG_DIR / f"{safe_name}.log"
    log_filepath = _session_log_files[safe_name]

    if _current_log_file == log_filepath and _logger is not None:
        return _logger

    _logger = logging.getLogger("AgentDemo")
    _logger.setLevel(logging.DEBUG)
    _logger.propagate = False
    for h in _logger.handlers[:]:
        if isinstance(h, logging.FileHandler):
            h.close()
        _logger.removeHandler(h)
    _logger.addHandler(_make_console_handler())
    _logger.addHandler(_make_file_handler(log_filepath))

    _current_log_file = log_filepath
    return _logger


def debug(msg, *args, **kwargs):
    get_logger().debug(msg, *args, **kwargs)


def info(msg, *args, **kwargs):
    get_logger().info(msg, *args, **kwargs)


def warning(msg, *args, **kwargs):
    get_logger().warning(msg, *args, **kwargs)


def error(msg, *args, **kwargs):
    get_logger().error(msg, *args, **kwargs)

