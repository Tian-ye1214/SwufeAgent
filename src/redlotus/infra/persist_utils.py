from __future__ import annotations

import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

from filelock import FileLock


@contextmanager
def file_lock(path: Path, *, timeout: float = 30.0) -> Iterator[None]:
    """跨进程文件锁；锁文件与目标同目录。"""
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(lock_path), timeout=timeout):
        yield


def atomic_write_text(path: Path, content: str, *, encoding: str = "utf-8") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding=encoding)
    os.replace(tmp, path)


def atomic_write_json(path: Path, data: Any, *, indent: int = 2) -> None:
    atomic_write_text(
        path,
        json.dumps(data, ensure_ascii=False, indent=indent) + "\n",
    )


def save_locked_json(path: Path, data: Any) -> None:
    """在跨进程文件锁下原子写 JSON。"""
    with file_lock(path):
        atomic_write_json(path, data)


def load_json_state(path: Path, version: int) -> dict[str, Any]:
    """读取 {"version", "sources": {...}} 形态的 JSON 状态文件；任何异常都退回默认值。"""
    default: dict[str, Any] = {"version": version, "sources": {}}
    if not path.is_file():
        return default
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return default
    if not isinstance(data, dict):
        return default
    if not isinstance(data.get("sources"), dict):
        data["sources"] = {}
    return data


def safe_name(text: str, *, extra: str = "_-", max_len: int = 50, fallback: str = "default") -> str:
    """把任意字符串清洗成文件名/键安全形式：非字母数字且不在 extra 内的字符替换为 _。"""
    cleaned = "".join(c if c.isalnum() or c in extra else "_" for c in text)
    return cleaned[:max_len] or fallback


def safe_segment(s: str, max_len: int = 80) -> str:
    s = (s or "").strip().replace("\n", " ")
    return safe_name(s, extra="_-.", max_len=max_len, fallback="default")


def iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def update_locked_json(
    path: Path, version: int, mutate: Callable[[dict], None]
) -> dict[str, Any]:
    """跨进程文件锁下的原子读-改-写：在同一把锁内 load→mutate→write，
    关闭"读不加锁、只锁写"导致的丢更新。mutate 就地修改 state。"""
    path = Path(path)
    with file_lock(path):
        state: dict[str, Any] = {"version": version}
        if path.is_file():
            try:
                with path.open("r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    state = loaded
            except Exception:
                pass
        state.setdefault("version", version)
        mutate(state)
        atomic_write_json(path, state)
    return state