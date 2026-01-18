import logging
import sys
import os
from datetime import datetime
from pathlib import Path


os.environ['PYTHONUNBUFFERED'] = '1'
LOG_DIR = Path("./logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

_logger = None
_current_log_file = None


class ImmediateStreamHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()


class ColorFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: "\033[36m",     # 青色
        logging.INFO: "\033[0m",       # 白色(默认)
        logging.WARNING: "\033[33m",   # 黄色
        logging.ERROR: "\033[31m",     # 红色
        logging.CRITICAL: "\033[91m",  # 亮红色
    }
    RESET = "\033[0m"
    
    def __init__(self, fmt=None, datefmt=None):
        super().__init__(fmt, datefmt)
        if sys.platform == "win32":
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
                kernel32.SetConsoleMode(kernel32.GetStdHandle(-12), 7)  # stderr
            except Exception:
                pass
    
    def format(self, record):
        color = self.COLORS.get(record.levelno, self.RESET)
        message = super().format(record)
        return f"{color}{message}{self.RESET}"


def get_logger() -> logging.Logger:
    """获取全局logger实例"""
    global _logger
    if _logger is None:
        _logger = logging.getLogger("AgentDemo")
        _logger.setLevel(logging.DEBUG)
        _logger.propagate = False

        if not _logger.handlers:
            console_handler = ImmediateStreamHandler(sys.stderr)
            console_handler.setLevel(logging.DEBUG)
            console_format = ColorFormatter(
                '%(asctime)s | %(levelname)-8s | %(message)s',
                datefmt='%H:%M:%S'
            )
            console_handler.setFormatter(console_format)
            _logger.addHandler(console_handler)
    
    return _logger


def setup_task_logger(task_name: str = "task") -> logging.Logger:
    global _logger, _current_log_file

    safe_task_name = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in task_name)
    safe_task_name = safe_task_name[:50]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{safe_task_name}_{timestamp}.log"
    log_filepath = LOG_DIR / log_filename

    _logger = logging.getLogger("AgentDemo")
    _logger.setLevel(logging.DEBUG)
    _logger.propagate = False

    _logger.handlers.clear()

    console_handler = ImmediateStreamHandler(sys.stderr)
    console_handler.setLevel(logging.DEBUG)
    console_format = ColorFormatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    _logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)
    _logger.addHandler(file_handler)
    
    _current_log_file = log_filepath
    _logger.info(f"日志文件已创建: {log_filepath}")
    
    return _logger


def get_current_log_file() -> str:
    """获取当前日志文件路径"""
    global _current_log_file
    return str(_current_log_file) if _current_log_file else None


def close_logger():
    """关闭当前logger的所有handler"""
    global _logger
    if _logger:
        for handler in _logger.handlers[:]:
            handler.close()
            _logger.removeHandler(handler)


def debug(msg, *args, **kwargs):
    get_logger().debug(msg, *args, **kwargs)

def info(msg, *args, **kwargs):
    get_logger().info(msg, *args, **kwargs)

def warning(msg, *args, **kwargs):
    get_logger().warning(msg, *args, **kwargs)

def error(msg, *args, **kwargs):
    get_logger().error(msg, *args, **kwargs)

def critical(msg, *args, **kwargs):
    get_logger().critical(msg, *args, **kwargs)
