from __future__ import annotations

from collections.abc import Awaitable, Callable

from redlotus.cli.render import print_error, print_panel
from redlotus.workspace.workspace import WorkspaceSnapshot


ReadLineFn = Callable[[], Awaitable[str | None]]


def format_snapshot_choices(snapshots: list[WorkspaceSnapshot]) -> str:
    lines = ["选择要加载的对话快照（新 → 旧）：", ""]
    for index, snapshot in enumerate(snapshots, 1):
        lines.append(f"  {index}. {snapshot.label}")
    lines.extend(["", "输入序号加载，留空取消。"])
    return "\n".join(lines)


async def legacy_pick_snapshot(
    snapshots: list[WorkspaceSnapshot],
    read_line: ReadLineFn,
) -> WorkspaceSnapshot | None:
    if not snapshots:
        return None
    if len(snapshots) == 1:
        return snapshots[0]
    print_panel(format_snapshot_choices(snapshots), title="加载对话")
    while True:
        raw = await read_line()
        if raw is None:
            return None
        text = raw.strip()
        if not text:
            return None
        if not text.isdigit():
            print_error("请输入有效序号。")
            continue
        index = int(text)
        if index < 1 or index > len(snapshots):
            print_error(f"序号超出范围（1-{len(snapshots)}）。")
            continue
        return snapshots[index - 1]
