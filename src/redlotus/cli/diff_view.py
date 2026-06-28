from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from enum import StrEnum

from rich.panel import Panel
from rich.text import Text


class DiffKind(StrEnum):
    ADD = "add"
    DEL = "del"
    MOD = "mod"
    CTX = "ctx"
    GAP = "gap"


@dataclass(frozen=True)
class DiffStyle:
    max_lines: int = 300
    colors: dict[DiffKind, str] = field(
        default_factory=lambda: {
            DiffKind.ADD: "green",
            DiffKind.DEL: "red",
            DiffKind.MOD: "blue",
            DiffKind.CTX: "dim",
            DiffKind.GAP: "dim italic",
        }
    )
    signs: dict[DiffKind, str] = field(
        default_factory=lambda: {
            DiffKind.ADD: "+",
            DiffKind.DEL: "-",
            DiffKind.MOD: "~",
            DiffKind.CTX: " ",
            DiffKind.GAP: " ",
        }
    )


DEFAULT_STYLE = DiffStyle()


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
            out += [DiffLine(DiffKind.ADD, None, j1 + k + 1, line) for k, line in enumerate(new_lines[j1:j2])]
        elif tag == "delete":
            out += [DiffLine(DiffKind.DEL, i1 + k + 1, None, line) for k, line in enumerate(old_lines[i1:i2])]
        elif tag == "replace":
            out += [DiffLine(DiffKind.MOD, i1 + k + 1, None, line) for k, line in enumerate(old_lines[i1:i2])]
            out += [DiffLine(DiffKind.MOD, None, j1 + k + 1, line) for k, line in enumerate(new_lines[j1:j2])]
    return out


def _equal_block(old_lines, i1, i2, j1, *, context, first, last) -> list[DiffLine]:
    n = i2 - i1

    def ctx(off: int) -> DiffLine:
        return DiffLine(DiffKind.CTX, i1 + off + 1, j1 + off + 1, old_lines[i1 + off])

    top = 0 if first else context
    bot = 0 if last else context
    if top + bot >= n:
        return [ctx(k) for k in range(n)]
    hidden = n - top - bot
    return (
        [ctx(k) for k in range(top)]
        + [DiffLine(DiffKind.GAP, None, None, f"⋯ {hidden} 行未改动 ⋯")]
        + [ctx(k) for k in range(n - bot, n)]
    )


def diff_stats(lines: list[DiffLine]) -> tuple[int, int, int]:
    """统计 (新增, 删除, 改动) 行数；改动按新侧计。"""
    add = sum(1 for ln in lines if ln.kind is DiffKind.ADD)
    dele = sum(1 for ln in lines if ln.kind is DiffKind.DEL)
    mod = sum(1 for ln in lines if ln.kind is DiffKind.MOD and ln.new_no is not None)
    return add, dele, mod


def _gutter_width(lines: list[DiffLine]) -> int:
    nums = [n for ln in lines for n in (ln.old_no, ln.new_no) if n is not None]
    return max((len(str(n)) for n in nums), default=1)


def _line_text(ln: DiffLine, width: int, signs: dict[DiffKind, str]) -> str:
    sign = signs[ln.kind]
    if ln.kind is DiffKind.GAP:
        return f"{sign} {'':>{width}}  {ln.text}"
    no = ln.new_no if ln.new_no is not None else ln.old_no
    return f"{sign} {no:>{width}}  {ln.text}"


def render_diff(
    lines: list[DiffLine], *, path: str, stats: tuple[int, int, int], style: DiffStyle = DEFAULT_STYLE
) -> Panel:
    """渲染成带行号、彩色、可折叠、超长截断的 rich Panel。"""
    add, dele, mod = stats
    width = _gutter_width(lines)
    body = Text()
    for idx, ln in enumerate(lines[: style.max_lines]):
        if idx:
            body.append("\n")
        body.append(_line_text(ln, width, style.signs), style=style.colors[ln.kind])
    if len(lines) > style.max_lines:
        body.append(f"\n… 还有 {len(lines) - style.max_lines} 行（已截断）", style=style.colors[DiffKind.GAP])
    title = Text.assemble(
        (f"{path}  ", "bold"),
        (f"+{add} ", style.colors[DiffKind.ADD]),
        (f"-{dele} ", style.colors[DiffKind.DEL]),
        (f"~{mod}", style.colors[DiffKind.MOD]),
    )
    return Panel(body, title=title, title_align="left", border_style="cyan")


def format_diff_text(
    lines: list[DiffLine], *, path: str, stats: tuple[int, int, int], style: DiffStyle = DEFAULT_STYLE
) -> str:
    """无样式纯文本版（写日志 / bot 用）。"""
    add, dele, mod = stats
    width = _gutter_width(lines)
    return "\n".join([f"{path}  +{add} -{dele} ~{mod}", *(_line_text(ln, width, style.signs) for ln in lines)])
