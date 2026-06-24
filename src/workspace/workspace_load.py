from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from cli.render import print_error, print_success, print_warning
from tools.conversation_log import read_saved_model_messages_file
from tools.memory import ChatHistory
from workspace.workspace import (
    WorkspaceSnapshot,
    list_workspace_snapshots,
    set_workspace,
)


PickSnapshotFn = Callable[[list[WorkspaceSnapshot]], Awaitable[WorkspaceSnapshot | None]]


async def load_snapshot_into_session(
    snapshot: WorkspaceSnapshot,
    *,
    coordinator_history: ChatHistory,
    manager_history: ChatHistory,
    reset_cli_session_for_load: Callable[[], Awaitable[None]],
    bind_loaded_snapshot_for_save: Callable[[str, Path, dict], None],
) -> None:
    messages, meta = await asyncio.to_thread(read_saved_model_messages_file, snapshot.path)
    agent = (meta.get("agent") or snapshot.agent or "").strip().lower()
    if agent not in ("coordinator", "manager"):
        raise ValueError(
            f"该快照的 agent={meta.get('agent')!r}，CLI 仅支持 coordinator 或 manager。"
        )
    await reset_cli_session_for_load()
    if agent == "coordinator":
        coordinator_history.set_messages(messages)
        manager_history.reset()
        bind_loaded_snapshot_for_save("coordinator", snapshot.path, meta)
        print_success(
            f"已加载 Coordinator 对话（{len(messages)} 条模型消息），任务与 Manager 上下文已清空。"
        )
        return
    manager_history.set_messages(messages)
    coordinator_history.reset()
    bind_loaded_snapshot_for_save("manager", snapshot.path, meta)
    print_success(
        f"已加载 Manager 对话（{len(messages)} 条模型消息），任务与 Coordinator 上下文已清空。"
    )


async def resolve_snapshot_choice(
    snapshots: list[WorkspaceSnapshot],
    *,
    pick_snapshot: PickSnapshotFn | None,
    force_picker: bool = False,
) -> WorkspaceSnapshot | None:
    if not snapshots:
        return None
    if len(snapshots) == 1 and not force_picker:
        return snapshots[0]
    if pick_snapshot is None:
        print_error("当前环境未提供快照选择器，无法选择要加载的对话。")
        return None
    return await pick_snapshot(snapshots)


async def enter_workspace(
    *,
    workspace_path: Path | str | None = None,
    coordinator_history: ChatHistory | None,
    manager_history: ChatHistory | None,
    reset_cli_session_for_load: Callable[[], Awaitable[None]],
    bind_loaded_snapshot_for_save: Callable[[str, Path, dict], None],
    pick_snapshot: PickSnapshotFn | None = None,
    force_picker: bool = False,
    announce_empty: bool = False,
) -> bool | None:
    """进入工作区：刷新根目录并按 0/1/多 策略加载 coordinator/manager 快照。"""
    if coordinator_history is None or manager_history is None:
        return None
    if workspace_path is not None:
        set_workspace(workspace_path)
    else:
        set_workspace(Path.cwd())

    snapshots = await asyncio.to_thread(list_workspace_snapshots)
    if not snapshots:
        if announce_empty:
            print_warning("当前工作区没有可加载的对话快照。")
        return None

    chosen = await resolve_snapshot_choice(
        snapshots,
        pick_snapshot=pick_snapshot,
        force_picker=force_picker,
    )
    if chosen is None:
        if force_picker:
            print_warning("未选择对话快照。")
        return None

    try:
        await load_snapshot_into_session(
            chosen,
            coordinator_history=coordinator_history,
            manager_history=manager_history,
            reset_cli_session_for_load=reset_cli_session_for_load,
            bind_loaded_snapshot_for_save=bind_loaded_snapshot_for_save,
        )
    except Exception as e:
        print_error(f"加载失败: {e}")
        return None
    return True
