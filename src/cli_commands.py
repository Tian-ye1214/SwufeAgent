"""交互式 CLI 斜杠命令解析（/help、/agent、/api、/load）。"""

from __future__ import annotations

import asyncio
import os
import time
from decimal import Decimal
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app_config import CONFIG_FILE, get_agent_roles, get_env, get_model_and_params, set_api, set_model_name
from lifecycle import AgentInvocationState
from runtime_state import TRACE_STORE
from ModelGateway.ModelChecker import (
    _lookup_openrouter_meta,
    compress_history_async,
    context_usage_breakdown,
    lookup_model_context,
    lookup_model_max_output_tokens,
    prewarm_effective_max_contexts_by_role_async,
)
from ModelGateway.usage_accounting import (
    UsageReport,
    model_message_files_for_path,
    session_model_message_files,
    summarize_usage_files,
)
from prompt import get_skills_as_in_system_prompt
from skills.SkillsManager import SkillsManager
from tools.memory import ChatHistory
from tools.conversation_log import drain_pending_saves, read_saved_model_messages_file
from cli.render import console, print_error, print_markdown, print_markdown_panel, print_panel, print_success, print_warning
from cli.panel import build_panel_snapshot, render_panel
import logger


def _out(text: str = "") -> None:
    console.print(text)


def _strip_quotes(text: str) -> str:
    """去除首尾成对的引号（粘贴路径常带引号）。"""
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        return text[1:-1]
    return text


def _format_model_tokens(value: Any) -> str:
    if not isinstance(value, int) or value <= 0:
        return "-"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.0f}k"
    return str(value)


def _openrouter_agent_meta_lines(model_name: str) -> list[str]:
    meta = _lookup_openrouter_meta(model_name)
    if not meta:
        return []

    lines: list[str] = []
    ctx = lookup_model_context(model_name)
    max_out = lookup_model_max_output_tokens(model_name)
    lines.append(
        f"OpenRouter: context {_format_model_tokens(ctx)}  max_output {_format_model_tokens(max_out)}"
    )

    arch = meta.get("architecture")
    if isinstance(arch, dict):
        inputs = arch.get("input_modalities")
        if isinstance(inputs, list) and inputs:
            lines.append(f"modalities: {', '.join(str(x) for x in inputs)}")

    pricing = meta.get("pricing")
    if isinstance(pricing, dict):
        prompt_price = pricing.get("prompt")
        completion_price = pricing.get("completion")
        if prompt_price is not None or completion_price is not None:
            lines.append(f"price: prompt={prompt_price or '-'} completion={completion_price or '-'}")

    return lines


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
| `/panel` | 查看当前工作区运行与历史总览 |
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
        for meta_line in _openrouter_agent_meta_lines(name):
            lines.append(f"  {meta_line}")
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


def _format_decimal(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _format_usd(value: Decimal) -> str:
    return f"${_format_decimal(value)}"


def _format_usage_report(report: UsageReport) -> str:
    lines: list[str] = []
    totals = report.totals
    lines.append("Total")
    lines.append(f"  responses: {totals.responses}")
    if totals.missing_usage_responses:
        lines.append(f"  missing usage responses: {totals.missing_usage_responses}")
    lines.append(
        f"  raw tokens: input={totals.input_tokens} output={totals.output_tokens} "
        f"reasoning={totals.reasoning_tokens}"
    )
    lines.append(
        f"  billable tokens: prompt={totals.prompt_billable_tokens} "
        f"completion={totals.completion_billable_tokens}"
    )

    total_input = Decimal("0")
    total_output = Decimal("0")
    total_cost = Decimal("0")
    unavailable = 0
    for summary in report.by_model.values():
        if summary.price is None:
            unavailable += summary.price_unavailable_responses
            continue
        total_input += summary.price.input_usd
        total_output += summary.price.output_usd
        total_cost += summary.price.total_usd
        unavailable += summary.price_unavailable_responses
    if report.by_model:
        if unavailable:
            lines.append("  estimated cost: unavailable for some responses")
        else:
            lines.append(
                f"  estimated cost: input={_format_usd(total_input)} "
                f"output={_format_usd(total_output)} total={_format_usd(total_cost)}"
            )

    if report.by_model:
        lines.append("")
        lines.append("By model")
        for model_name in sorted(report.by_model):
            summary = report.by_model[model_name]
            t = summary.totals
            lines.append(
                f"  {model_name}: responses={t.responses} "
                f"raw={t.input_tokens}/{t.output_tokens} "
                f"billable={t.prompt_billable_tokens}/{t.completion_billable_tokens}"
            )
            if summary.price is None:
                lines.append("    price: unavailable")
            else:
                p = summary.price
                lines.append(
                    f"    price: total={_format_usd(p.total_usd)} "
                    f"input={_format_usd(p.input_usd)} output={_format_usd(p.output_usd)} "
                    f"source={p.source}"
                )
            if summary.price_unavailable_responses:
                lines.append(
                    f"    price unavailable responses: {summary.price_unavailable_responses}"
                )

    if report.files:
        lines.append("")
        lines.append(f"Files: {len(report.files)}")
        for item in report.files[:10]:
            lines.append(f"  {item.path}")
        if len(report.files) > 10:
            lines.append(f"  ... {len(report.files) - 10} more")
    return "\n".join(lines)


def _conversation_session_key(system: Any) -> str | None:
    logs = getattr(system, "_session_logs", None)
    if logs is not None:
        session_key = getattr(logs, "session_key", None)
        if callable(session_key):
            value = session_key()
            if value:
                return str(value)
    value = getattr(system, "session_key", None)
    if isinstance(value, str) and "/" in value:
        return value
    return None


async def _print_usage_report(raw: str, system: Any) -> None:
    tail = raw.strip()[6:].strip()
    if tail:
        tail = _strip_quotes(tail)
        target = Path(tail).expanduser()
        if not target.is_absolute():
            target = (Path.cwd() / target).resolve()
        files = model_message_files_for_path(target)
        if not files:
            print_error(f"未找到 model_messages 日志: {target}")
            return
    else:
        await drain_pending_saves()
        session_key = _conversation_session_key(system)
        if not session_key:
            print_error("当前会话尚未绑定日志目录；请使用 /usage <path>")
            return
        files = session_model_message_files(logger.LOG_DIR, session_key)
        if not files:
            print_error(f"当前会话没有 model_messages 日志: {session_key}")
            return

    try:
        report = await asyncio.to_thread(summarize_usage_files, files)
    except Exception as e:
        print_error(f"统计 usage 失败: {e}")
        return
    print_panel(_format_usage_report(report), title="Usage")


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
        if bd.get("has_usage"):
            lines.append(f"   latest real input_tokens {_k_tokens(bd['input'])}")
        else:
            lines.append("   no real usage yet; estimates are disabled")
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


SlashHandler = Callable[["SlashCommandContext"], Awaitable[bool | None]]


@dataclass
class SlashCommandContext:
    raw: str
    parts: list[str]
    cmd: str
    skills_manager: SkillsManager
    coordinator_history: ChatHistory | None
    manager_history: ChatHistory | None
    reset_cli_session_for_load: Callable[[], Awaitable[None]]
    bind_loaded_snapshot_for_save: Callable[[str, Path, dict], None]
    system: Any


async def _cmd_help(ctx: SlashCommandContext) -> bool | None:
    print_cli_help()
    return None


async def _cmd_config(ctx: SlashCommandContext) -> bool | None:
    print_config_summary()
    return None


async def _cmd_context(ctx: SlashCommandContext) -> bool | None:
    await _print_context_usage(ctx.system, ctx.coordinator_history, ctx.manager_history)
    return None


async def _cmd_usage(ctx: SlashCommandContext) -> bool | None:
    await _print_usage_report(ctx.raw, ctx.system)
    return None


async def _cmd_panel(ctx: SlashCommandContext) -> bool | None:
    await drain_pending_saves()
    include_all = any(part.strip().lower() == "--all" for part in ctx.parts[1:])
    snapshot = await build_panel_snapshot(
        log_root=logger.LOG_DIR,
        system=ctx.system,
        coordinator_history=ctx.coordinator_history,
        manager_history=ctx.manager_history,
        include_all=include_all,
    )
    console.print(render_panel(snapshot))
    return None


async def _cmd_pwd(ctx: SlashCommandContext) -> bool | None:
    print_success(str(Path.cwd()))
    return None


async def _cmd_cd(ctx: SlashCommandContext) -> bool | None:
    if len(ctx.parts) < 2:
        print_error("用法：/cd <path>")
        return None
    target = Path(ctx.parts[1].strip()).expanduser()
    if not target.is_absolute():
        target = (Path.cwd() / target).resolve()
    if not target.is_dir():
        print_error(f"目录不存在: {target}")
        return None
    os.chdir(target)
    print_success(f"已切换工作目录: {target}")
    return None


async def _cmd_status(ctx: SlashCommandContext) -> bool | None:
    await _print_lifecycle_status(ctx.system)
    return None


async def _cmd_trace(ctx: SlashCommandContext) -> bool | None:
    if len(ctx.parts) < 2:
        print_error("用法: /trace <turn_id>")
        return None
    print_markdown(TRACE_STORE.format_turn(ctx.parts[1].strip()))
    return None


async def _cmd_tasks(ctx: SlashCommandContext) -> bool | None:
    print_markdown_panel(ctx.system.structured_task_status(), title="任务状态")
    return None


async def _cmd_stop(ctx: SlashCommandContext) -> bool | None:
    msg = await ctx.system.cancel_current_turn()
    _out(msg)
    return None


async def _cmd_cancel(ctx: SlashCommandContext) -> bool | None:
    if len(ctx.parts) < 2:
        print_error("用法: /cancel <invocation_id> 或 /cancel agent <agent_id>")
        return None
    if ctx.parts[1].lower() == "agent":
        if len(ctx.parts) < 3:
            print_error("用法: /cancel agent <agent_id>")
            return None
        aid = ctx.parts[2].strip()
        n = await ctx.system.registry.cancel_agent(aid)
        if n:
            print_success(f"已请求取消 agent_id={aid!r} 的当前 invocation。")
        else:
            print_warning(f"未找到 agent_id={aid!r} 的活跃 invocation。")
        return None
    iid = ctx.parts[1].strip()
    n_match = await ctx.system.registry.count_active_invocation_prefix_matches(iid)
    if n_match > 1:
        print_warning(f"前缀 {iid!r} 匹配到多个活跃 invocation，请使用更长的 id。")
        return None
    resolved = await ctx.system.registry.resolve_active_invocation_id(iid)
    ok = await ctx.system.registry.cancel(iid)
    if ok:
        print_success(f"已请求取消 invocation_id={(resolved or iid)!r}。")
        return None
    sk = getattr(ctx.system, "session_key", None)
    if sk:
        recent = await ctx.system.registry.find_recent_invocation_by_prefix(sk, iid)
        if recent is not None:
            print_warning(
                f"invocation {iid!r} 已在近期历史中结束"
                f"（state={recent.state.value}），无法取消。"
            )
            return None
    print_warning(f"未找到活跃 invocation_id={iid!r}（支持 UUID 前缀匹配）。")
    return None


async def _cmd_skills(ctx: SlashCommandContext) -> bool | None:
    print_loaded_skills(ctx.skills_manager)
    return None


async def _cmd_agent(ctx: SlashCommandContext) -> bool | None:
    roles = get_agent_roles()
    role_text = "|".join(roles)
    if len(ctx.parts) == 1:
        print_agent_models()
        _out(f"切换模型: /agent <{role_text}> <模型名称>")
        return None
    if len(ctx.parts) < 3:
        print_error(f"用法: /agent <{role_text}> <模型名称>")
        return None
    role = ctx.parts[1].lower()
    model_name = ctx.parts[2].strip()
    try:
        set_model_name(role, model_name)
        print_success(f"已设置 [{role}] 模型为: {model_name}（已写入 config.json）")
        await prewarm_effective_max_contexts_by_role_async(
            reason=f"切换模型 {role}={model_name!r}"
        )
    except ValueError as e:
        print_error(str(e))
    return None


async def _cmd_api(ctx: SlashCommandContext) -> bool | None:
    interactive_set_api()
    return None


async def _cmd_load(ctx: SlashCommandContext) -> bool | None:
    tail = ctx.raw.strip()[5:].strip()
    if not tail:
        print_error("用法: /load <*.model_messages.json 路径>")
        return None
    raw_path = _strip_quotes(tail)
    load_path = Path(raw_path).expanduser()
    if not load_path.is_file():
        print_error(f"找不到文件: {load_path}")
        return None
    try:
        messages, meta = read_saved_model_messages_file(load_path)
    except Exception as e:
        print_error(f"加载失败: {e}")
        return None
    agent = (meta.get("agent") or "").strip().lower()
    if agent not in ("coordinator", "manager"):
        print_error(
            f"该快照的 agent={meta.get('agent')!r}，CLI 仅支持从 coordinator 或 manager 落盘文件恢复。"
        )
        return None
    if ctx.coordinator_history is None or ctx.manager_history is None:
        print_error("当前环境未绑定历史对象，无法加载。")
        return None
    await ctx.reset_cli_session_for_load()
    if agent == "coordinator":
        ctx.coordinator_history.set_messages(messages)
        ctx.manager_history.reset()
        ctx.bind_loaded_snapshot_for_save("coordinator", load_path, meta)
        print_success(
            f"已加载 Coordinator 对话（{len(messages)} 条模型消息），任务与 Manager 上下文已清空。"
        )
    else:
        ctx.manager_history.set_messages(messages)
        ctx.coordinator_history.reset()
        ctx.bind_loaded_snapshot_for_save("manager", load_path, meta)
        print_success(
            f"已加载 Manager 对话（{len(messages)} 条模型消息），任务与 Coordinator 上下文已清空。"
        )
    return True


async def _cmd_compress(ctx: SlashCommandContext) -> bool | None:
    task_state_getter = getattr(ctx.system, "structured_task_status", None)
    task_state = task_state_getter() if callable(task_state_getter) else None
    role_histories: list[tuple[str, str, list[ChatHistory]]] = [
        ("manager", "Manager", [ctx.manager_history] if ctx.manager_history is not None else []),
        ("coordinator", "Coordinator", [ctx.coordinator_history] if ctx.coordinator_history is not None else []),
    ]
    if not any(histories for _, _, histories in role_histories):
        print_error("当前环境未绑定任何 Agent 历史，无法压缩。")
        return None

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
                await compress_history_async(
                    h,
                    role=role,
                    force=True,
                    task_state=task_state,
                )
                lines.append(f"  • {label}: 已压缩")
        except Exception as e:
            lines.append(f"  • {label}: 压缩失败: {e}")
    print_panel("\n".join(lines), title="上下文压缩")
    return None


SLASH_COMMAND_HANDLERS: dict[str, SlashHandler] = {
    "/help": _cmd_help,
    "/config": _cmd_config,
    "/context": _cmd_context,
    "/usage": _cmd_usage,
    "/panel": _cmd_panel,
    "/pwd": _cmd_pwd,
    "/cd": _cmd_cd,
    "/status": _cmd_status,
    "/trace": _cmd_trace,
    "/tasks": _cmd_tasks,
    "/stop": _cmd_stop,
    "/cancel": _cmd_cancel,
    "/skills": _cmd_skills,
    "/agent": _cmd_agent,
    "/api": _cmd_api,
    "/load": _cmd_load,
    "/compress": _cmd_compress,
}


async def handle_slash_command(
    raw: str,
    skills_manager: SkillsManager,
    *,
    coordinator_history: ChatHistory | None,
    manager_history: ChatHistory | None,
    reset_cli_session_for_load: Callable[[], Awaitable[None]],
    bind_loaded_snapshot_for_save: Callable[[str, Path, dict], None],
    system: Any,
) -> bool | None:
    """
    处理以 / 开头的输入行。
    返回 None 表示已消费该输入，不应作为普通任务发送，且不改变 is_first_input。
    返回 True 表示已消费且已将交互态视为「新会话首条」（如 /load 后）。
    """
    parts = raw.strip().split(maxsplit=2)
    cmd = parts[0].lower() if parts else ""

    ctx = SlashCommandContext(
        raw=raw,
        parts=parts,
        cmd=cmd,
        skills_manager=skills_manager,
        coordinator_history=coordinator_history,
        manager_history=manager_history,
        reset_cli_session_for_load=reset_cli_session_for_load,
        bind_loaded_snapshot_for_save=bind_loaded_snapshot_for_save,
        system=system,
    )

    handler = SLASH_COMMAND_HANDLERS.get(cmd)
    if handler is not None:
        return await handler(ctx)

    print_warning(f"未知命令 {cmd}，输入 /help 查看可用命令")
    return None
