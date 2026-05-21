"""交互式 CLI 斜杠命令解析（/help、/agent、/api、/load）。"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from app_config import get_agent_roles, get_env, get_model_and_params, set_api, set_model_name, settings
from lifecycle import AgentInvocationState
from runtime_state import TRACE_STORE
from ModelGateway.ModelChecker import (
    compress_history_async,
    prewarm_effective_max_contexts_by_role_async,
)
from prompt import get_skills_as_in_system_prompt
from skills.SkillsManager import SkillsManager
from tools.memory import ChatHistory
from tools.conversation_log import read_saved_model_messages_file


def print_cli_help() -> None:
    print(
        "\n── HELP ─────────────────────────────────────────\n"
        "/help     显示本帮助\n"
        "/skills   查看当前已加载 Skills（与系统提示中 {skills_layout}+{skills_summary} 一致）\n"
        "/agent    查看 Manager / Worker / Coordinator 的模型名与 temperature、max_tokens；\n"
        "          切换: /agent <manager|worker|coordinator> <模型名称>\n"
        "/api      按提示修改 BASE_URL 与 API_KEY（写入 config.json，回车跳过单项）\n"
        "/compress 主动压缩当前 Manager / Coordinator 对话上下文（Markdown 摘要）\n"
        "/status  树形列出会话内 Agent 实例与 invocation（活跃 + 近期历史）\n"
        "/cancel  中止 invocation；用法: /cancel <invocation_id 或前缀> 或 /cancel agent <agent_id>\n"
        "/stop    中断当前用户回合（取消该 turn 下所有活跃 invocation）\n"
        "/load    从落盘的 *.model_messages.json 恢复对话（与「新任务」同样会清空任务状态）\n"
        "          用法: /load <文件路径>\n"
        "\n── 其他 ─────────────────────────────────────────────\n"
        "新任务     清空任务与对话上下文\n"
        "quit/exit  退出程序\n"
        "输入中可包含图片/视频路径作为附件\n"
        "────────────────────────────────────────────────────\n"
    )


def print_agent_models() -> None:
    labels = {"manager": "Manager（任务规划）", "worker": "Worker（子任务执行）", "coordinator": "Coordinator（入口协调）"}
    roles = get_agent_roles()
    print("\n── 当前模型配置（config.json）──")
    for role in roles:
        name, p = get_model_and_params(role)
        print(f"  • {labels.get(role, role)}")
        print(f"    模型名: {name}")
        th_s = f"  reasoning→thinking: {p['thinking']}"
        print(f"    temperature: {p['temperature']}  max_tokens: {p['max_tokens']}{th_s}")
    print("")


def interactive_set_api() -> None:
    cur_b = (get_env("BASE_URL", warn=False) or "").strip()
    cur_k = (get_env("API_KEY", warn=False) or "").strip()
    print(f"\n当前 BASE_URL: {cur_b or '(空；优先 .env 再 config)'}")
    print(f"当前 API_KEY: {'已填写' if cur_k else '(空；优先 .env 再 config)'}\n")
    nb = input("请输入 BASE_URL（回车保持当前值）: ").strip()
    nk = input("请输入 API_KEY（回车保持当前值）: ").strip()
    new_base = nb if nb else cur_b
    new_key = nk if nk else cur_k
    set_api(new_base, new_key)
    print("已更新 API 配置并写入 config.json。\n")


def print_loaded_skills(skills_manager: SkillsManager) -> None:
    """输出与模型系统提示中 Skills 区块相同的内容。"""
    block = get_skills_as_in_system_prompt(skills_manager)
    print("\n── 当前 Skills（与系统提示注入内容一致）──\n")
    print(block if block.strip() else "(无：layout 与 summary 均为空)\n")


async def _print_lifecycle_status(system: Any) -> None:
    sk = getattr(system, "session_key", None)
    if not sk:
        agents = await system.registry.list_agents()
        if not agents:
            print("\n── Agent 生命周期 ──\n（尚未绑定会话，无 Agent 实例）\n")
            return
        print("\n── Agent 生命周期（全部会话）──")
        for a in agents:
            print(f"  [{a.state.value}] {a.agent_id}")
        active = await system.registry.list_active_invocations()
        if active:
            print("\n  活跃 invocation:")
            now = time.monotonic()
            for inv in active:
                elapsed = now - inv.started_at
                parent = (inv.parent_invocation_id or "-")[:8]
                print(
                    f"    inv={inv.invocation_id[:8]} agent={inv.role} "
                    f"parent={parent} turn={inv.turn_id} state={inv.state.value} {elapsed:.1f}s"
                )
        print("")
        return

    view = await system.registry.get_session_view(sk)
    print(f"\n── Agent 生命周期 ── Session: {sk}")
    if not view.agents:
        print("  （无 Agent 实例）")
    for a in view.agents:
        suffix = a.agent_id.split(":", 2)[-1] if a.agent_id.count(":") >= 2 else a.role
        label = a.role if suffix == a.role else f"{a.role}:{suffix}"
        print(f"  [{a.state.value}] {label}  id={a.agent_id}")

    if view.active_invocations:
        print("\n  活跃 invocation:")
        now = time.monotonic()
        for inv in view.active_invocations:
            elapsed = now - inv.started_at
            parent = (inv.parent_invocation_id or "-")[:8]
            print(
                f"    inv={inv.invocation_id[:8]} agent={inv.agent_id} "
                f"parent={parent} turn={inv.turn_id} state={inv.state.value} {elapsed:.1f}s"
            )

    completed = [
        i
        for i in view.recent_invocations
        if i.state == AgentInvocationState.COMPLETED
    ]
    failed = [
        i
        for i in view.recent_invocations
        if i.state in (AgentInvocationState.FAILED, AgentInvocationState.CANCELLED)
    ]
    if completed or failed:
        print(
            f"\n  近期历史: {len(completed)} 已完成, {len(failed)} 失败/取消 "
            f"（活跃 {len(view.active_invocations)}）"
        )
        for inv in list(view.recent_invocations)[-5:]:
            parent = (inv.parent_invocation_id or "-")[:8]
            state = inv.state.value
            if inv.state in (AgentInvocationState.RUNNING, AgentInvocationState.PENDING):
                state = f"{state}(已结束)"
            print(
                f"    inv={inv.invocation_id[:8]} {inv.role} parent={parent} "
                f"turn={inv.turn_id} state={state}"
            )
    elif not view.active_invocations:
        print("\n  （无活跃 invocation）")
    print("")


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
    if cmd == "/status":
        await _print_lifecycle_status(system)
        return True, None
    if cmd == "/trace":
        if len(parts) < 2:
            print("Usage: /trace <turn_id>\n")
            return True, None
        print(TRACE_STORE.format_turn(parts[1].strip()) + "\n")
        return True, None
    if cmd == "/tasks":
        print(system.structured_task_status() + "\n")
        return True, None
    if cmd == "/stop":
        msg = await system.cancel_current_turn()
        print(f"{msg}\n")
        return True, None
    if cmd == "/cancel":
        if len(parts) < 2:
            print("用法: /cancel <invocation_id> 或 /cancel agent <agent_id>\n")
            return True, None
        if parts[1].lower() == "agent":
            if len(parts) < 3:
                print("用法: /cancel agent <agent_id>\n")
                return True, None
            aid = parts[2].strip()
            n = await system.registry.cancel_agent(aid)
            if n:
                print(f"已请求取消 agent_id={aid!r} 的当前 invocation。\n")
            else:
                print(f"未找到 agent_id={aid!r} 的活跃 invocation。\n")
            return True, None
        iid = parts[1].strip()
        n_match = await system.registry.count_active_invocation_prefix_matches(iid)
        if n_match > 1:
            print(f"前缀 {iid!r} 匹配到多个活跃 invocation，请使用更长的 id。\n")
            return True, None
        resolved = await system.registry.resolve_active_invocation_id(iid)
        ok = await system.registry.cancel(iid)
        if ok:
            print(f"已请求取消 invocation_id={(resolved or iid)!r}。\n")
            return True, None
        sk = getattr(system, "session_key", None)
        if sk:
            recent = await system.registry.find_recent_invocation_by_prefix(sk, iid)
            if recent is not None:
                print(
                    f"invocation {iid!r} 已在近期历史中结束"
                    f"（state={recent.state.value}），无法取消。\n"
                )
                return True, None
        print(f"未找到活跃 invocation_id={iid!r}（支持 UUID 前缀匹配）。\n")
        return True, None
    if cmd == "/skills":
        print_loaded_skills(skills_manager)
        return True, None
    if cmd == "/agent":
        roles = get_agent_roles()
        role_text = "|".join(roles)
        if len(parts) == 1:
            print_agent_models()
            print(f"切换模型: /agent <{role_text}> <模型名称>\n")
            return True, None
        if len(parts) < 3:
            print(f"用法: /agent <{role_text}> <模型名称>\n")
            return True, None
        role = parts[1].lower()
        model_name = parts[2].strip()
        try:
            set_model_name(role, model_name)
            print(f"已设置 [{role}] 模型为: {model_name}（已写入 config.json）\n")
            await prewarm_effective_max_contexts_by_role_async(
                reason=f"切换模型 {role}={model_name!r}"
            )
        except ValueError as e:
            print(f"错误: {e}\n")
        return True, None
    if cmd == "/api":
        interactive_set_api()
        return True, None
    if cmd == "/load":
        tail = raw.strip()[5:].strip()
        if not tail:
            print("用法: /load <*.model_messages.json 路径>（通常为 logs/conversations/.../*.model_messages.json）\n")
            return True, None
        raw_path = tail.strip()
        if (raw_path.startswith('"') and raw_path.endswith('"')) or (
            raw_path.startswith("'") and raw_path.endswith("'")
        ):
            raw_path = raw_path[1:-1]
        load_path = Path(raw_path).expanduser()
        if not load_path.is_file():
            print(f"找不到文件: {load_path}\n")
            return True, None
        try:
            messages, meta = read_saved_model_messages_file(load_path)
        except Exception as e:
            print(f"加载失败: {e}\n")
            return True, None
        agent = (meta.get("agent") or "").strip().lower()
        if agent not in ("coordinator", "manager"):
            print(
                f"该快照的 agent={meta.get('agent')!r}，CLI 仅支持从 coordinator 或 manager 落盘文件恢复。\n"
            )
            return True, None
        if coordinator_history is None or manager_history is None:
            print("当前环境未绑定历史对象，无法加载。\n")
            return True, None
        await reset_cli_session_for_load()
        if agent == "coordinator":
            coordinator_history.set_messages(messages)
            manager_history.reset()
            bind_loaded_snapshot_for_save("coordinator", load_path, meta)
            print(f"已加载 Coordinator 对话（{len(messages)} 条模型消息），任务与 Manager 上下文已清空。\n")
        else:
            manager_history.set_messages(messages)
            coordinator_history.reset()
            bind_loaded_snapshot_for_save("manager", load_path, meta)
            print(f"已加载 Manager 对话（{len(messages)} 条模型消息），任务与 Coordinator 上下文已清空。\n")
        return True, True
    if cmd == "/compress":
        role_histories: list[tuple[str, str, list[ChatHistory]]] = [
            ("manager", "Manager", [manager_history] if manager_history is not None else []),
            ("coordinator", "Coordinator", [coordinator_history] if coordinator_history is not None else []),
        ]
        if not any(histories for _, _, histories in role_histories):
            print("当前环境未绑定任何 Agent 历史，无法压缩。\n")
            return True, None

        print("\n── 压缩结果（按角色）──")
        for role, label, histories in role_histories:
            if not histories:
                print(f"  • {label}: 未绑定历史")
                continue

            try:
                for h in histories:
                    msgs = list(h.messages)
                    if not msgs:
                        continue
                    await compress_history_async(h, role=role, force=True)
            except Exception as e:
                print(f"  • {label}: 压缩失败: {e}")
                continue
        return True, None

    print(f"未知命令 {cmd}，输入 /help 查看可用命令\n")
    return True, None
