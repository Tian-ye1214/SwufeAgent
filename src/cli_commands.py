"""交互式 CLI 斜杠命令解析（/help、/agent、/api）。"""

from __future__ import annotations

from app_config import get_context_config, get_env, get_model_and_params, set_api, set_model_name
from ModelGateway.ModelChecker import (
    compress_history_async,
    estimate_history_tokens_async,
    format_context_usage_line,
    get_effective_max_context_async,
    prewarm_effective_max_contexts_by_role_async,
)
from prompt import get_skills_as_in_system_prompt
from skills.SkillsManager import SkillsManager
from tools.Memory import ChatHistory


def print_cli_help() -> None:
    print(
        "\n── 斜杠命令 ─────────────────────────────────────────\n"
        "/help     显示本帮助\n"
        "/skills   查看当前已加载 Skills（与系统提示中 {skills_layout}+{skills_summary} 一致）\n"
        "/agent    查看 Manager / Worker / Coordinator 的模型名与 temperature、max_tokens；\n"
        "          切换: /agent <manager|worker|coordinator> <模型名称>\n"
        "/api      按提示修改 BASE_URL 与 API_KEY（写入 config.json，回车跳过单项）\n"
        "/compress 主动压缩当前 Coordinator 对话上下文（Markdown 摘要）\n"
        "\n── 其他 ─────────────────────────────────────────────\n"
        "新任务     清空任务与对话上下文\n"
        "quit/exit  退出程序\n"
        "输入中可包含图片/视频路径作为附件\n"
        "────────────────────────────────────────────────────\n"
    )


def print_agent_models() -> None:
    labels = {
        "manager": "Manager（任务规划）",
        "worker": "Worker（子任务执行）",
        "coordinator": "Coordinator（入口协调）",
    }
    print("\n── 当前模型配置（config.json）──")
    for role in ("manager", "worker", "coordinator"):
        name, p = get_model_and_params(role)
        print(f"  • {labels[role]}")
        print(f"    模型名: {name}")
        th = p.get("thinking")
        th_s = f"  reasoning→thinking: {th}" if th is not None else ""
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


async def handle_slash_command(
    raw: str,
    skills_manager: SkillsManager,
    *,
    coordinator_history: ChatHistory | None = None,
) -> bool:
    """
    处理以 / 开头的输入行。
    返回 True 表示已消费该输入，不应作为普通任务发送。
    """
    parts = raw.strip().split(maxsplit=2)
    cmd = parts[0].lower() if parts else ""

    if cmd == "/help":
        print_cli_help()
        return True
    if cmd == "/skills":
        print_loaded_skills(skills_manager)
        return True
    if cmd == "/agent":
        if len(parts) == 1:
            print_agent_models()
            print("切换模型: /agent <manager|worker|coordinator> <模型名称>\n")
            return True
        if len(parts) < 3:
            print("用法: /agent <manager|worker|coordinator> <模型名称>\n")
            return True
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
        return True
    if cmd == "/api":
        interactive_set_api()
        return True
    if cmd == "/compress":
        if coordinator_history is None:
            print("当前环境未绑定 Coordinator 历史，无法压缩。\n")
            return True
        msgs = list(coordinator_history.messages)
        if not msgs:
            print("对话历史为空，无需压缩。\n")
            return True
        ctx = get_context_config("coordinator")
        cpt = float(ctx["token_estimate_fallback_chars_per_token"])
        max_t = await get_effective_max_context_async(role="coordinator")
        before = await estimate_history_tokens_async(
            msgs, chars_per_token=cpt, role="coordinator"
        )
        try:
            did = await compress_history_async(
                coordinator_history, role="coordinator", force=True
            )
        except Exception as e:
            print(f"压缩失败: {e}\n")
            return True
        after = await estimate_history_tokens_async(
            list(coordinator_history.messages),
            chars_per_token=cpt,
            role="coordinator",
        )
        if did:
            print(
                f"已压缩上下文: {format_context_usage_line(before, max_t)} → "
                f"{format_context_usage_line(after, max_t)}\n"
            )
        else:
            print(
                f"未执行压缩（可能历史过短或头尾保护区重叠）。"
                f"当前 {format_context_usage_line(after, max_t)}\n"
            )
        return True

    print(f"未知命令 {cmd}，输入 /help 查看可用命令\n")
    return True
