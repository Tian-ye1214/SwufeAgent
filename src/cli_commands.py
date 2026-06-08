"""交互式 CLI 斜杠命令解析（/help、/agent、/api、/load）。"""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from app_config import CONFIG_FILE, get_agent_roles, get_env, get_model_and_params, set_api, set_model_name
from lifecycle import AgentInvocationState
from runtime_state import TRACE_STORE
from ModelGateway.ModelChecker import (
    compress_history_async,
    context_usage_breakdown,
    prewarm_effective_max_contexts_by_role_async,
)
from prompt import get_skills_as_in_system_prompt
from skills.SkillsManager import SkillsManager
from tools.memory import ChatHistory
from tools.conversation_log import read_saved_model_messages_file
from cli.render import console, print_error, print_markdown, print_markdown_panel, print_panel, print_success, print_warning


def _out(text: str = "") -> None:
    console.print(text)


def print_cli_help() -> None:
    print_markdown(
        """
## 斜杠命令

| 命令 | 说明 |
|------|------|
| `/help` | 显示本帮助 |
| `/exit` / `/quit` | 退出程序 |
| `/clear` | 清空任务与对话上下文 |
| `/pwd` | 查看当前工作目录 |
| `/cd <path>` | 切换工作目录 |
| `/config` | 查看配置摘要 |
| `/context` | 查看上下文 token 用量分解 |
| `/skills` | 查看已加载 Skills |
| `/agent` | 查看或切换模型：`/agent <role> <模型名>` |
| `/api` | 修改 BASE_URL 与 API_KEY |
| `/compress` | 压缩 Manager / Coordinator 上下文 |
| `/status` | Agent 生命周期与 invocation |
| `/cancel` | 中止 invocation |
| `/stop` | 中断当前用户回合 |
| `/load <path>` | 从落盘文件恢复对话 |
| `/trace` / `/tasks` | 追踪与任务状态 |

## 其他

- **新任务** — 清空上下文（同 `/clear`）
- **quit / exit** — 退出
- 输入中可用 `@文件路径` 引用文本文件（Tab 补全）
- 可包含图片/视频路径作为多媒体附件
"""
    )


def print_agent_models() -> None:
    labels = {"manager": "Manager（任务规划）", "worker": "Worker（子任务执行）", "coordinator": "Coordinator（入口协调）"}
    roles = get_agent_roles()
    lines = ["当前模型配置（config.json）", ""]
    for role in roles:
        name, p = get_model_and_params(role)
        lines.append(f"• {labels.get(role, role)}")
        lines.append(f"  模型名: {name}")
        th_s = f"  reasoning→thinking: {p['thinking']}"
        lines.append(f"  temperature: {p['temperature']}  max_tokens: {p['max_tokens']}{th_s}")
        lines.append("")
    print_panel("\n".join(lines), title="Agent 模型")


def interactive_set_api() -> None:
    cur_b = (get_env("BASE_URL", warn=False) or "").strip()
    cur_k = (get_env("API_KEY", warn=False) or "").strip()
    _out(f"\n当前 BASE_URL: {cur_b or '(空；优先 .env 再 config)'}")
    _out(f"当前 API_KEY: {'已填写' if cur_k else '(空；优先 .env 再 config)'}\n")
    nb = input("请输入 BASE_URL（回车保持当前值）: ").strip()
    nk = input("请输入 API_KEY（回车保持当前值）: ").strip()
    new_base = nb if nb else cur_b
    new_key = nk if nk else cur_k
    set_api(new_base, new_key)
    _out("已更新 API 配置并写入 config.json。\n")


def print_loaded_skills(skills_manager: SkillsManager) -> None:
    """输出与模型系统提示中 Skills 区块相同的内容。"""
    block = get_skills_as_in_system_prompt(skills_manager)
    print_markdown_panel(
        block if block.strip() else "(无：layout 与 summary 均为空)",
        title="当前 Skills",
    )


def print_config_summary() -> None:
    base = (get_env("BASE_URL", warn=False) or "").strip()
    key_set = bool((get_env("API_KEY", warn=False) or "").strip())
    lines = [
        f"配置文件: {CONFIG_FILE}",
        f"BASE_URL: {base or '(空)'}",
        f"API_KEY: {'已填写' if key_set else '(空)'}",
        f"工作目录: {Path.cwd()}",
    ]
    print_panel("\n".join(lines), title="配置摘要")


def _k_tokens(n: int) -> str:
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


async def _print_context_usage(
    system: Any,
    coordinator_history: ChatHistory | None,
    manager_history: ChatHistory | None,
) -> None:
    skills_manager = system._skills_manager
    memory_injection = system._injection_for_session()
    roles = [
        ("coordinator", "Coordinator", coordinator_history),
        ("manager", "Manager", manager_history),
    ]
    lines: list[str] = []
    for role, label, history in roles:
        messages = list(history.messages) if history is not None else []
        bd = await asyncio.to_thread(
            context_usage_breakdown,
            role,
            messages,
            skills_manager=skills_manager,
            memory_injection=memory_injection,
        )
        lines.append(
            f"{label}: {bd['percent']:.0f}%  "
            f"({_k_tokens(bd['total'])}/{_k_tokens(bd['max'])} tok)"
        )
        lines.append(
            f"   系统提示 {_k_tokens(bd['system'])}"
            f"（含记忆注入 {_k_tokens(bd['memory'])}） · 历史 {_k_tokens(bd['history'])}"
        )
        lines.append(f"   自动压缩阈值 {_k_tokens(bd['threshold'])} tok")
        lines.append("")
    print_panel("\n".join(lines).rstrip(), title="上下文用量")


async def _print_lifecycle_status(system: Any) -> None:
    lines: list[str] = []
    sk = getattr(system, "session_key", None)
    if not sk:
        agents = await system.registry.list_agents()
        if not agents:
            print_panel("（尚未绑定会话，无 Agent 实例）", title="Agent 生命周期")
            return
        lines.append("全部会话")
        for a in agents:
            lines.append(f"  [{a.state.value}] {a.agent_id}")
        active = await system.registry.list_active_invocations()
        if active:
            lines.append("\n活跃 invocation:")
            now = time.monotonic()
            for inv in active:
                elapsed = now - inv.started_at
                parent = (inv.parent_invocation_id or "-")[:8]
                lines.append(
                    f"  inv={inv.invocation_id[:8]} agent={inv.role} "
                    f"parent={parent} turn={inv.turn_id} state={inv.state.value} {elapsed:.1f}s"
                )
        print_panel("\n".join(lines), title="Agent 生命周期")
        return

    view = await system.registry.get_session_view(sk)
    lines.append(f"Session: {sk}")
    if not view.agents:
        lines.append("  （无 Agent 实例）")
    for a in view.agents:
        suffix = a.agent_id.split(":", 2)[-1] if a.agent_id.count(":") >= 2 else a.role
        label = a.role if suffix == a.role else f"{a.role}:{suffix}"
        lines.append(f"  [{a.state.value}] {label}  id={a.agent_id}")

    if view.active_invocations:
        lines.append("\n活跃 invocation:")
        now = time.monotonic()
        for inv in view.active_invocations:
            elapsed = now - inv.started_at
            parent = (inv.parent_invocation_id or "-")[:8]
            lines.append(
                f"  inv={inv.invocation_id[:8]} agent={inv.agent_id} "
                f"parent={parent} turn={inv.turn_id} state={inv.state.value} {elapsed:.1f}s"
            )

    completed = [i for i in view.recent_invocations if i.state == AgentInvocationState.COMPLETED]
    failed = [
        i
        for i in view.recent_invocations
        if i.state in (AgentInvocationState.FAILED, AgentInvocationState.CANCELLED)
    ]
    if completed or failed:
        lines.append(
            f"\n近期历史: {len(completed)} 已完成, {len(failed)} 失败/取消 "
            f"（活跃 {len(view.active_invocations)}）"
        )
        for inv in list(view.recent_invocations)[-5:]:
            parent = (inv.parent_invocation_id or "-")[:8]
            state = inv.state.value
            if inv.state in (AgentInvocationState.RUNNING, AgentInvocationState.PENDING):
                state = f"{state}(已结束)"
            lines.append(
                f"  inv={inv.invocation_id[:8]} {inv.role} parent={parent} "
                f"turn={inv.turn_id} state={state}"
            )
    elif not view.active_invocations:
        lines.append("\n（无活跃 invocation）")
    print_panel("\n".join(lines), title="Agent 生命周期")


async def handle_slash_command(
    raw: str,
    skills_manager: SkillsManager,
    *,
    coordinator_history: ChatHistory | None,
    manager_history: ChatHistory | None,
    reset_cli_session_for_load: Callable[[], Awaitable[None]],
    bind_loaded_snapshot_for_save: Callable[[str, Path, dict], None],
    system: Any,
) -> tuple[bool, bool | None]:
    """
    处理以 / 开头的输入行。
    返回 (True, None) 表示已消费该输入，不应作为普通任务发送。
    返回 (True, True) 表示已消费且已将交互态视为「新会话首条」（如 /load 后）。
    """
    parts = raw.strip().split(maxsplit=2)
    cmd = parts[0].lower() if parts else ""

    if cmd == "/help":
        print_cli_help()
        return True, None
    if cmd == "/config":
        print_config_summary()
        return True, None
    if cmd == "/context":
        await _print_context_usage(system, coordinator_history, manager_history)
        return True, None
    if cmd == "/pwd":
        print_success(str(Path.cwd()))
        return True, None
    if cmd == "/cd":
        if len(parts) < 2:
            print_error("用法：/cd <path>")
            return True, None
        target = Path(parts[1].strip()).expanduser()
        if not target.is_absolute():
            target = (Path.cwd() / target).resolve()
        if not target.is_dir():
            print_error(f"目录不存在: {target}")
            return True, None
        os.chdir(target)
        print_success(f"已切换工作目录: {target}")
        return True, None
    if cmd == "/status":
        await _print_lifecycle_status(system)
        return True, None
    if cmd == "/trace":
        if len(parts) < 2:
            print_error("用法: /trace <turn_id>")
            return True, None
        print_markdown(TRACE_STORE.format_turn(parts[1].strip()))
        return True, None
    if cmd == "/tasks":
        print_markdown_panel(system.structured_task_status(), title="任务状态")
        return True, None
    if cmd == "/stop":
        msg = await system.cancel_current_turn()
        _out(msg)
        return True, None
    if cmd == "/cancel":
        if len(parts) < 2:
            print_error("用法: /cancel <invocation_id> 或 /cancel agent <agent_id>")
            return True, None
        if parts[1].lower() == "agent":
            if len(parts) < 3:
                print_error("用法: /cancel agent <agent_id>")
                return True, None
            aid = parts[2].strip()
            n = await system.registry.cancel_agent(aid)
            if n:
                print_success(f"已请求取消 agent_id={aid!r} 的当前 invocation。")
            else:
                print_warning(f"未找到 agent_id={aid!r} 的活跃 invocation。")
            return True, None
        iid = parts[1].strip()
        n_match = await system.registry.count_active_invocation_prefix_matches(iid)
        if n_match > 1:
            print_warning(f"前缀 {iid!r} 匹配到多个活跃 invocation，请使用更长的 id。")
            return True, None
        resolved = await system.registry.resolve_active_invocation_id(iid)
        ok = await system.registry.cancel(iid)
        if ok:
            print_success(f"已请求取消 invocation_id={(resolved or iid)!r}。")
            return True, None
        sk = getattr(system, "session_key", None)
        if sk:
            recent = await system.registry.find_recent_invocation_by_prefix(sk, iid)
            if recent is not None:
                print_warning(
                    f"invocation {iid!r} 已在近期历史中结束"
                    f"（state={recent.state.value}），无法取消。"
                )
                return True, None
        print_warning(f"未找到活跃 invocation_id={iid!r}（支持 UUID 前缀匹配）。")
        return True, None
    if cmd == "/skills":
        print_loaded_skills(skills_manager)
        return True, None
    if cmd == "/agent":
        roles = get_agent_roles()
        role_text = "|".join(roles)
        if len(parts) == 1:
            print_agent_models()
            _out(f"切换模型: /agent <{role_text}> <模型名称>")
            return True, None
        if len(parts) < 3:
            print_error(f"用法: /agent <{role_text}> <模型名称>")
            return True, None
        role = parts[1].lower()
        model_name = parts[2].strip()
        try:
            set_model_name(role, model_name)
            print_success(f"已设置 [{role}] 模型为: {model_name}（已写入 config.json）")
            await prewarm_effective_max_contexts_by_role_async(
                reason=f"切换模型 {role}={model_name!r}"
            )
        except ValueError as e:
            print_error(str(e))
        return True, None
    if cmd == "/api":
        interactive_set_api()
        return True, None
    if cmd == "/load":
        tail = raw.strip()[5:].strip()
        if not tail:
            print_error("用法: /load <*.model_messages.json 路径>")
            return True, None
        raw_path = tail.strip()
        if (raw_path.startswith('"') and raw_path.endswith('"')) or (
            raw_path.startswith("'") and raw_path.endswith("'")
        ):
            raw_path = raw_path[1:-1]
        load_path = Path(raw_path).expanduser()
        if not load_path.is_file():
            print_error(f"找不到文件: {load_path}")
            return True, None
        try:
            messages, meta = read_saved_model_messages_file(load_path)
        except Exception as e:
            print_error(f"加载失败: {e}")
            return True, None
        agent = (meta.get("agent") or "").strip().lower()
        if agent not in ("coordinator", "manager"):
            print_error(
                f"该快照的 agent={meta.get('agent')!r}，CLI 仅支持从 coordinator 或 manager 落盘文件恢复。"
            )
            return True, None
        if coordinator_history is None or manager_history is None:
            print_error("当前环境未绑定历史对象，无法加载。")
            return True, None
        await reset_cli_session_for_load()
        if agent == "coordinator":
            coordinator_history.set_messages(messages)
            manager_history.reset()
            bind_loaded_snapshot_for_save("coordinator", load_path, meta)
            print_success(
                f"已加载 Coordinator 对话（{len(messages)} 条模型消息），任务与 Manager 上下文已清空。"
            )
        else:
            manager_history.set_messages(messages)
            coordinator_history.reset()
            bind_loaded_snapshot_for_save("manager", load_path, meta)
            print_success(
                f"已加载 Manager 对话（{len(messages)} 条模型消息），任务与 Coordinator 上下文已清空。"
            )
        return True, True
    if cmd == "/compress":
        role_histories: list[tuple[str, str, list[ChatHistory]]] = [
            ("manager", "Manager", [manager_history] if manager_history is not None else []),
            ("coordinator", "Coordinator", [coordinator_history] if coordinator_history is not None else []),
        ]
        if not any(histories for _, _, histories in role_histories):
            print_error("当前环境未绑定任何 Agent 历史，无法压缩。")
            return True, None

        lines = ["压缩结果（按角色）"]
        for role, label, histories in role_histories:
            if not histories:
                lines.append(f"  • {label}: 未绑定历史")
                continue

            try:
                for h in histories:
                    msgs = list(h.messages)
                    if not msgs:
                        continue
                    await compress_history_async(h, role=role, force=True)
                    lines.append(f"  • {label}: 已压缩")
            except Exception as e:
                lines.append(f"  • {label}: 压缩失败: {e}")
        print_panel("\n".join(lines), title="上下文压缩")
        return True, None

    print_warning(f"未知命令 {cmd}，输入 /help 查看可用命令")
    return True, None
