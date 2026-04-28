"""交互式 CLI 斜杠命令解析（/help、/agent、/api）。"""

from __future__ import annotations

from typing import Iterable

from app_config import get_agent_roles, get_env, get_model_and_params, set_api, set_model_name
from ModelGateway.ModelChecker import (
    compress_history_async,
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
        "/compress 主动压缩当前 Manager / Coordinator 对话上下文（Markdown 摘要）\n"
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


async def handle_slash_command(
    raw: str,
    skills_manager: SkillsManager,
    *,
    coordinator_history: ChatHistory | None = None,
    manager_history: ChatHistory | None = None,
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
        roles = get_agent_roles()
        role_text = "|".join(roles)
        if len(parts) == 1:
            print_agent_models()
            print(f"切换模型: /agent <{role_text}> <模型名称>\n")
            return True
        if len(parts) < 3:
            print(f"用法: /agent <{role_text}> <模型名称>\n")
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
        role_histories: list[tuple[str, str, list[ChatHistory]]] = [
            ("manager", "Manager", [manager_history] if manager_history is not None else []),
            ("coordinator", "Coordinator", [coordinator_history] if coordinator_history is not None else []),
        ]
        if not any(histories for _, _, histories in role_histories):
            print("当前环境未绑定任何 Agent 历史，无法压缩。\n")
            return True

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
        return True

    print(f"未知命令 {cmd}，输入 /help 查看可用命令\n")
    return True
