from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import Literal

from rich.panel import Panel
from rich.text import Text

DiffKind = Literal["add", "del", "mod", "ctx", "gap"]

_STYLE: dict[str, str] = {
    "add": "green",
    "del": "red",
    "mod": "blue",
    "ctx": "dim",
    "gap": "dim italic",
}
_SIGN: dict[str, str] = {"add": "+", "del": "-", "mod": "~", "ctx": " ", "gap": " "}

_MAX_RENDER_LINES = 300


@dataclass(frozen=True)
class DiffLine:
    kind: DiffKind
    old_no: int | None
    new_no: int | None
    text: str


def compute_line_diff(old: str, new: str, *, context: int = 3) -> list[DiffLine]:
    """按行对比 old→new，返回带类别和行号的 DiffLine 列表；长未改段折叠为 gap。"""
    old_lines = old.splitlines()
    new_lines = new.splitlines()
    sm = difflib.SequenceMatcher(a=old_lines, b=new_lines, autojunk=False)
    ops = sm.get_opcodes()
    out: list[DiffLine] = []
    for idx, (tag, i1, i2, j1, j2) in enumerate(ops):
        if tag == "equal":
            out += _equal_block(old_lines, i1, i2, j1, context=context, first=idx == 0, last=idx == len(ops) - 1)
        elif tag == "insert":
            out += [DiffLine("add", None, j1 + k + 1, line) for k, line in enumerate(new_lines[j1:j2])]
        elif tag == "delete":
            out += [DiffLine("del", i1 + k + 1, None, line) for k, line in enumerate(old_lines[i1:i2])]
        elif tag == "replace":
            out += [DiffLine("mod", i1 + k + 1, None, line) for k, line in enumerate(old_lines[i1:i2])]
            out += [DiffLine("mod", None, j1 + k + 1, line) for k, line in enumerate(new_lines[j1:j2])]
    return out


def _equal_block(old_lines, i1, i2, j1, *, context, first, last) -> list[DiffLine]:
    n = i2 - i1

    def ctx(off: int) -> DiffLine:
        return DiffLine("ctx", i1 + off + 1, j1 + off + 1, old_lines[i1 + off])

    top = 0 if first else context
    bot = 0 if last else context
    if top + bot >= n:
        return [ctx(k) for k in range(n)]
    hidden = n - top - bot
    return (
        [ctx(k) for k in range(top)]
        + [DiffLine("gap", None, None, f"⋯ {hidden} 行未改动 ⋯")]
        + [ctx(k) for k in range(n - bot, n)]
    )


def diff_stats(lines: list[DiffLine]) -> tuple[int, int, int]:
    """统计 (新增, 删除, 改动) 行数；改动按新侧计。"""
    add = sum(1 for ln in lines if ln.kind == "add")
    dele = sum(1 for ln in lines if ln.kind == "del")
    mod = sum(1 for ln in lines if ln.kind == "mod" and ln.new_no is not None)
    return add, dele, mod


def _gutter_width(lines: list[DiffLine]) -> int:
    nums = [n for ln in lines for n in (ln.old_no, ln.new_no) if n is not None]
    return max((len(str(n)) for n in nums), default=1)


def _line_text(ln: DiffLine, width: int) -> str:
    sign = _SIGN[ln.kind]
    if ln.kind == "gap":
        return f"{sign} {'':>{width}}  {ln.text}"
    no = ln.new_no if ln.new_no is not None else ln.old_no
    return f"{sign} {no:>{width}}  {ln.text}"


def render_diff(lines: list[DiffLine], *, path: str, stats: tuple[int, int, int]) -> Panel:
    """渲染成带行号、彩色、可折叠、超长截断的 rich Panel。"""
    add, dele, mod = stats
    width = _gutter_width(lines)
    body = Text()
    for idx, ln in enumerate(lines[:_MAX_RENDER_LINES]):
        if idx:
            body.append("\n")
        body.append(_line_text(ln, width), style=_STYLE[ln.kind])
    if len(lines) > _MAX_RENDER_LINES:
        body.append(f"\n… 还有 {len(lines) - _MAX_RENDER_LINES} 行（已截断）", style="dim italic")
    title = Text.assemble(
        (f"{path}  ", "bold"),
        (f"+{add} ", "green"),
        (f"-{dele} ", "red"),
        (f"~{mod}", "blue"),
    )
    return Panel(body, title=title, title_align="left", border_style="cyan")


def format_diff_text(lines: list[DiffLine], *, path: str, stats: tuple[int, int, int]) -> str:
    """无样式纯文本版（写日志 / bot 用）。"""
    add, dele, mod = stats
    width = _gutter_width(lines)
    return "\n".join([f"{path}  +{add} -{dele} ~{mod}", *(_line_text(ln, width) for ln in lines)])
