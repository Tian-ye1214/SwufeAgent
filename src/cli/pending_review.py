from __future__ import annotations

import difflib
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


def _opcodes(baseline: str, current: str):
    a = baseline.splitlines(keepends=True)
    b = current.splitlines(keepends=True)
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    return a, b, sm.get_opcodes()


@dataclass(frozen=True)
class Hunk:
    """一处连续改动（baseline→current 中的一个非 equal 区块）。"""

    index: int
    old_start: int
    old_lines: list[str]
    new_start: int
    new_lines: list[str]

    @property
    def location(self) -> str:
        return f"L{self.new_start}" if self.new_lines else f"L{self.old_start}"


def compute_hunks(baseline: str, current: str) -> list[Hunk]:
    """把 baseline→current 的差异切成逐块 Hunk 列表（equal 区块跳过）。"""
    a, b, ops = _opcodes(baseline, current)
    hunks: list[Hunk] = []
    for tag, i1, i2, j1, j2 in ops:
        if tag == "equal":
            continue
        hunks.append(Hunk(len(hunks), i1 + 1, a[i1:i2], j1 + 1, b[j1:j2]))
    return hunks


def reconstruct(baseline: str, current: str, rejected: set[int]) -> str:
    """按逐块决定重建文件内容：rejected 的块取 baseline 侧，其余取 current 侧。

    rejected 为空 → 完全等于 current；rejected 含全部块 → 完全等于 baseline。
    """
    a, b, ops = _opcodes(baseline, current)
    out: list[str] = []
    idx = 0
    for tag, i1, i2, j1, j2 in ops:
        if tag == "equal":
            out.extend(b[j1:j2])
            continue
        out.extend(a[i1:i2] if idx in rejected else b[j1:j2])
        idx += 1
    return "".join(out)


@dataclass
class ReviewEntry:
    path: Path
    name: str
    baseline: str
    snapshot: str


class PendingReviewStore:
    """跨线程共享的待审查暂存区。复用 toolkit 的 file_lock，避免与 agent 写盘竞争。"""

    def __init__(self, file_lock: threading.Lock) -> None:
        self._lock = file_lock
        self._entries: dict[str, ReviewEntry] = {}
        self._on_change: Callable[[], None] | None = None

    def activate(self, on_change: Callable[[], None]) -> None:
        self._on_change = on_change

    def deactivate(self) -> None:
        self._on_change = None
        with self._lock:
            self._entries.clear()

    def register(self, path: Path, *, name: str, baseline: str, snapshot: str) -> None:
        if self._on_change is None or baseline == snapshot:
            return
        key = str(path)
        with self._lock:
            existing = self._entries.get(key)
            base = existing.baseline if existing is not None else baseline
            self._entries[key] = ReviewEntry(Path(path), name, base, snapshot)
        self._notify()

    def entries(self) -> list[ReviewEntry]:
        with self._lock:
            return list(self._entries.values())

    def get(self, key: str) -> ReviewEntry | None:
        with self._lock:
            return self._entries.get(key)

    def apply(self, key: str, rejected: set[int]) -> None:
        """按当前 rejected 重写文件（幂等）。撤销=该块回退到基线。"""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return
            new_text = reconstruct(entry.baseline, entry.snapshot, rejected)
            try:
                current = entry.path.read_text(encoding="utf-8") if entry.path.exists() else ""
            except Exception:
                current = ""
            if new_text != current:
                entry.path.write_text(new_text, encoding="utf-8")

    def finish(self, key: str) -> None:
        with self._lock:
            self._entries.pop(key, None)
        self._notify()

    def _notify(self) -> None:
        cb = self._on_change
        if cb is not None:
            cb()
