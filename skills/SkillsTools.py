# -*- coding: utf-8 -*-
import logger
from skills.SkillsManager import get_skills_manager


def list_available_skills() -> str:
    """
    列出所有可用的 Agent Skills。
    
    返回所有已注册 Skills 的名称和描述，帮助了解当前可用的能力扩展。
    在执行复杂任务前，建议先调用此函数查看有哪些 Skills 可以辅助完成任务。
    
    Returns:
        格式化的 Skills 列表，包含每个 Skill 的名称和描述
    """
    logger.debug("(list_available_skills)")
    manager = get_skills_manager()
    metadata_list = manager.get_all_metadata()
    
    if not metadata_list:
        return "当前没有可用的 Skills。可以在 skills/ 目录下创建新的 Skill。"
    
    lines = ["可用的 Agent Skills:", "=" * 50]
    
    for i, metadata in enumerate(metadata_list, 1):
        lines.append(f"\n{i}. {metadata.name}")
        lines.append(f"   描述: {metadata.description}")
        lines.append(f"   路径: {metadata.path}")
    
    lines.append("\n" + "=" * 50)
    lines.append("使用 get_skill_instructions(skill_name) 获取具体 Skill 的详细指令。")
    
    return "\n".join(lines)


def get_skill_instructions(skill_name: str) -> str:
    """
    获取指定 Skill 的详细指令内容。
    
    加载 Skill 的完整指令，包括工作流程、代码示例和最佳实践。
    这是使用 Skill 前的必要步骤，通过阅读指令了解如何正确使用该 Skill。
    
    Parameters:
        skill_name: Skill 名称 (如 "pdf-processing", "web-scraping")
        
    Returns:
        Skill 的完整指令内容，包含使用方法和代码示例
    """
    logger.debug(f"(get_skill_instructions {skill_name})")
    manager = get_skills_manager()
    
    instructions = manager.load_skill_instructions(skill_name)
    
    if instructions is None:
        available = [m.name for m in manager.get_all_metadata()]
        return f"错误: Skill '{skill_name}' 不存在。\n可用的 Skills: {', '.join(available) if available else '无'}"
    
    skill = manager.get_skill(skill_name)
    
    result = [
        f"# Skill: {skill_name}",
        f"描述: {skill.description}",
        "=" * 50,
        "",
        instructions
    ]

    resources = manager.list_skill_resources(skill_name)
    if resources:
        result.append("")
        result.append("-" * 50)
        result.append("可用的额外资源文件:")
        for res in resources:
            result.append(f"  - {res}")
        result.append("使用 load_skill_resource(skill_name, resource_name) 加载资源。")
    
    return "\n".join(result)


def load_skill_resource(skill_name: str, resource_name: str) -> str:
    """
    加载 Skill 的额外资源文件。
    
    某些 Skill 包含额外的资源文件，如详细指南、参考文档、模板等。
    使用此函数按需加载这些资源，避免一次性加载所有内容。
    
    Parameters:
        skill_name: Skill 名称
        resource_name: 资源文件名 (如 "FORMS.md", "scripts/helper.py")
        
    Returns:
        资源文件的内容
    """
    logger.debug(f"(load_skill_resource {skill_name}/{resource_name})")
    manager = get_skills_manager()
    
    content = manager.load_skill_resource(skill_name, resource_name)
    
    if content is None:
        skill = manager.get_skill(skill_name)
        if skill is None:
            return f"错误: Skill '{skill_name}' 不存在"
        
        resources = manager.list_skill_resources(skill_name)
        return f"错误: 资源 '{resource_name}' 不存在。\n可用资源: {', '.join(resources) if resources else '无'}"
    
    return f"# 资源: {skill_name}/{resource_name}\n\n{content}"


def request_skill_usage(skill_name: str, task_description: str) -> str:
    """
    请求使用某个 Skill 来完成任务（需要用户确认）。
    
    在使用 Skill 之前，调用此函数向用户说明将要使用的 Skill 和执行的任务，
    获取用户确认后再继续执行。这是一个安全机制，确保用户了解 Agent 的行为。
    
    Parameters:
        skill_name: 要使用的 Skill 名称
        task_description: 任务描述，说明为什么需要使用此 Skill
        
    Returns:
        用户的确认结果和 Skill 指令（如果用户同意）
    """
    logger.debug(f"(request_skill_usage {skill_name})")
    manager = get_skills_manager()
    
    skill = manager.get_skill(skill_name)
    if skill is None:
        available = [m.name for m in manager.get_all_metadata()]
        return f"错误: Skill '{skill_name}' 不存在。\n可用的 Skills: {', '.join(available) if available else '无'}"

    logger.info("=" * 50)
    logger.info("🔧 Agent Skills 使用请求")
    logger.info("=" * 50)
    logger.info(f"Skill: {skill_name}")
    logger.info(f"描述: {skill.description}")
    logger.info(f"任务: {task_description}")
    logger.info("-" * 50)
    print("\n是否允许使用此 Skill? (y/n): ", end="")
    user_response = input().strip().lower()
    
    if user_response in ['y', 'yes', '是', '确认', '同意']:
        logger.info("✅ 用户已确认，加载 Skill 指令...")
        instructions = manager.load_skill_instructions(skill_name)
        
        result = [
            "用户已确认使用 Skill。",
            "",
            f"# Skill: {skill_name}",
            "=" * 50,
            "",
            instructions,
            "",
            "=" * 50,
            "请按照上述指令完成任务。"
        ]
        return "\n".join(result)
    else:
        logger.info("❌ 用户拒绝使用此 Skill")
        return f"用户拒绝使用 Skill '{skill_name}'。请尝试其他方法完成任务。"


def suggest_skill_for_task(task_description: str) -> str:
    """
    根据任务描述推荐合适的 Skill。
    
    分析任务描述，自动匹配最相关的 Skill。这有助于快速找到
    完成任务所需的能力扩展。
    
    Parameters:
        task_description: 任务描述
        
    Returns:
        推荐的 Skill 信息，如果没有匹配则返回提示
    """
    logger.debug(f"(suggest_skill_for_task)")
    manager = get_skills_manager()
    
    matched_skill = manager.match_skill(task_description)
    
    if matched_skill:
        return (
            f"推荐使用 Skill: {matched_skill.name}\n"
            f"描述: {matched_skill.description}\n\n"
            f"使用 get_skill_instructions('{matched_skill.name}') 获取详细指令，\n"
            f"或使用 request_skill_usage('{matched_skill.name}', '任务描述') 请求使用此 Skill。"
        )
    else:
        available = [m.name for m in manager.get_all_metadata()]
        if available:
            return (
                f"未找到与任务直接匹配的 Skill。\n"
                f"可用的 Skills: {', '.join(available)}\n"
                f"可以使用 list_available_skills() 查看详细信息。"
            )
        else:
            return "当前没有可用的 Skills。请手动完成任务。"


def refresh_skills() -> str:
    """
    刷新 Skills 列表。
    
    重新扫描 skills 目录，发现新添加的 Skills 或更新已修改的 Skills。
    当 skills 目录有变化时调用此函数。
    
    Returns:
        刷新结果信息
    """
    logger.debug("(refresh_skills)")
    manager = get_skills_manager()
    manager.refresh()
    
    metadata_list = manager.get_all_metadata()
    return f"Skills 已刷新。当前共有 {len(metadata_list)} 个 Skills 可用。"


def execute_skill_script(skill_name: str, script_name: str, args: str = "") -> str:
    """
    执行 Skill 中的脚本文件。
    
    某些 Skill 包含可执行的脚本，用于完成特定操作。
    脚本的执行输出会被返回，而脚本代码本身不会进入上下文。
    
    Parameters:
        skill_name: Skill 名称
        script_name: 脚本文件名 (如 "scripts/process.py")
        args: 传递给脚本的参数
        
    Returns:
        脚本执行的输出结果
    """
    logger.debug(f"(execute_skill_script {skill_name}/{script_name} {args})")
    manager = get_skills_manager()
    
    return manager.execute_skill_script(skill_name, script_name, args)


# 导出的工具函数列表
skills_tools = [
    list_available_skills,
    get_skill_instructions,
    load_skill_resource,
    request_skill_usage,
    suggest_skill_for_task,
    refresh_skills,
    execute_skill_script,
]

