from __future__ import annotations

import sys
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger as _lg

from persist_utils import safe_name

CONSOLE_FMT = "{time:HH:mm:ss} | {level: <8} | {message}"
FILE_FMT = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}"
_WARNING_NO = 30
_STYLES = {
    "DEBUG": "dim cyan",
    "INFO": "green",
    "WARNING": "yellow",
    "ERROR": "bold red",
    "CRITICAL": "bold white on red",
}
_stm_quiet: ContextVar[int] = ContextVar("stm_quiet", default=0)
LOG_DIR: Path
_task_sink_id: int | None = None


@dataclass(frozen=True)
class LoggingConfig:
    """日志装配参数。default() 把目录锚定到项目根（开发）或 exe 目录（frozen）。"""

    log_dir: Path
    level: str = "DEBUG"
    console_fmt: str = CONSOLE_FMT
    file_fmt: str = FILE_FMT

    @classmethod
    def default(cls) -> "LoggingConfig":
        base = (
            Path(sys.executable).resolve().parent
            if getattr(sys, "frozen", False)
            else Path(__file__).resolve().parent.parent  # src/logger.py → 项目根
        )
        return cls(log_dir=base / "logs")


def configure(cfg: LoggingConfig) -> None:
    """工厂：清掉 loguru 默认 stderr，挂上控制台 sink + 每会话文件 sink。"""
    global LOG_DIR, _task_sink_id
    LOG_DIR = cfg.log_dir
    _task_sink_id = None
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    _lg.remove()
    _lg.add(_console_sink, level=cfg.level, format=cfg.console_fmt, filter=_console_filter)
    _lg.add(_session_sink, level=cfg.level, format=cfg.file_fmt, filter=_session_filter)



def _console_filter(record: dict) -> bool:
    if record["extra"].get("file_only"):
        return False
    if _stm_quiet.get() > 0 and record["level"].no < _WARNING_NO:
        return False
    return True


def _console_sink(message: Any) -> None:
    # 延迟导入：控制台日志走 CLI/TUI 的 Rich sink，而非 stderr。
    from rich.text import Text

    from cli.output import emit_renderable

    level = message.record["level"].name
    emit_renderable(Text(str(message).rstrip("\n"), style=_STYLES.get(level, "")))


def _session_filter(record: dict) -> bool:
    return bool(record["extra"].get("session"))


def _session_sink(message: Any) -> None:
    path = LOG_DIR / f"{message.record['extra']['session']}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(str(message))



def _emit(level: str, msg: object, args: tuple[Any, ...], *, exc_info: bool = False, file_only: bool = False) -> None:
    # loguru 用 {}-style；这里沿用项目的 %-style 先自行格式化，再把成品串原样交给 loguru
    # （不传 args 时 loguru 不会再 .format()，故含 { } / JSON 的文本也安全）。
    text = (str(msg) % args) if args else str(msg)
    target = _lg.bind(file_only=True) if file_only else _lg
    target.opt(exception=exc_info).log(level, text)


def debug(msg: object, *args: Any, exc_info: bool = False) -> None:
    _emit("DEBUG", msg, args, exc_info=exc_info)


def info(msg: object, *args: Any, exc_info: bool = False) -> None:
    _emit("INFO", msg, args, exc_info=exc_info)


def warning(msg: object, *args: Any, exc_info: bool = False) -> None:
    _emit("WARNING", msg, args, exc_info=exc_info)


def error(msg: object, *args: Any, exc_info: bool = False) -> None:
    _emit("ERROR", msg, args, exc_info=exc_info)


def info_file_only(msg: object, *args: Any) -> None:
    """只写文件、不上控制台（避免与 Rich 输出重复）。"""
    _emit("INFO", msg, args, file_only=True)


def setup_task_logger(task_name: str = "task") -> None:
    """为本次任务追加一个文件 sink（重复调用会替换上一个）。"""
    global _task_sink_id
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = LOG_DIR / f"{safe_name(task_name, max_len=50, fallback='task')}_{ts}.log"
    if _task_sink_id is not None:
        _lg.remove(_task_sink_id)
    _task_sink_id = _lg.add(path, level="DEBUG", format=FILE_FMT, encoding="utf-8")
    info("日志文件已创建: %s", path)


@contextmanager
def session_log_context(session_name: str):
    """把本上下文内的日志额外落到 {session}.log。"""
    with _lg.contextualize(session=safe_name(session_name, max_len=50, fallback="task")):
        yield


@contextmanager
def stm_ingest_console_quiet():
    """短期记忆后台入库时压制控制台 INFO/DEBUG，文件日志不受影响。"""
    token = _stm_quiet.set(_stm_quiet.get() + 1)
    try:
        yield
    finally:
        _stm_quiet.reset(token)


configure(LoggingConfig.default())

__all__ = [
    "LOG_DIR",
    "LoggingConfig",
    "configure",
    "debug",
    "info",
    "warning",
    "error",
    "info_file_only",
    "setup_task_logger",
    "session_log_context",
    "stm_ingest_console_quiet",
]
