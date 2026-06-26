# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from skills.SkillsManager import SkillsManager


class SkillsToolkit:

    def __init__(self, manager: "SkillsManager"):
        self._manager: SkillsManager = manager


    def list_available_skills(self) -> str:
        """
        列出所有可用的 Agent Skills。

        返回所有已注册 Skills 的名称和描述，帮助了解当前可用的能力扩展。
        在执行复杂任务前，建议先调用此函数查看有哪些 Skills 可以辅助完成任务。

        Returns:
            格式化的 Skills 列表，包含每个 Skill 的名称和描述
        """
        metadata_list = self._manager.get_all_metadata()

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

    def get_skill_instructions(self, skill_name: str) -> str:
        """
        获取指定 Skill 的详细指令内容。

        加载 Skill 的完整指令，包括工作流程、代码示例和最佳实践。
        这是使用 Skill 前的必要步骤，通过阅读指令了解如何正确使用该 Skill。

        Parameters:
            skill_name: Skill 名称 (如 "web-scraping")

        Returns:
            Skill 的完整指令内容，包含使用方法和代码示例
        """
        instructions = self._manager.load_skill_instructions(skill_name)

        if instructions is None:
            available = [m.name for m in self._manager.get_all_metadata()]
            return (
                f"错误: Skill '{skill_name}' 不存在。\n"
                f"可用的 Skills: {', '.join(available) if available else '无'}"
            )

        skill = self._manager.get_skill(skill_name)
        result = [
            f"# Skill: {skill_name}",
            f"描述: {skill.description}",
            "=" * 50,
            "",
            instructions,
        ]

        resources = self._manager.list_skill_resources(skill_name)
        if resources:
            result.append("")
            result.append("-" * 50)
            result.append("可用的额外资源文件:")
            for res in resources:
                result.append(f"  - {res}")
            result.append("使用 load_skill_resource(skill_name, resource_name) 加载资源。")

        return "\n".join(result)

    def load_skill_resource(self, skill_name: str, resource_name: str) -> str:
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
        content = self._manager.load_skill_resource(skill_name, resource_name)

        if content is None:
            skill = self._manager.get_skill(skill_name)
            if skill is None:
                return f"错误: Skill '{skill_name}' 不存在"
            resources = self._manager.list_skill_resources(skill_name)
            return (
                f"错误: 资源 '{resource_name}' 不存在。\n"
                f"可用资源: {', '.join(resources) if resources else '无'}"
            )

        return f"# 资源: {skill_name}/{resource_name}\n\n{content}"

    def refresh_skills(self) -> str:
        """
        刷新 Skills 列表。

        重新扫描 skills 目录，发现新添加的 Skills 或更新已修改的 Skills。
        当 skills 目录有变化时调用此函数。

        Returns:
            刷新结果信息
        """
        self._manager.refresh()
        metadata_list = self._manager.get_all_metadata()
        return f"Skills 已刷新。当前共有 {len(metadata_list)} 个 Skills 可用。"

    async def execute_skill_script(
        self, skill_name: str, script_name: str, args: str = "", timeout: float = 300
    ) -> str:
        """
        执行 Skill 中的脚本文件。

        某些 Skill 包含可执行的脚本，用于完成特定操作。
        脚本的执行输出会被返回，而脚本代码本身不会进入上下文。
        .py 脚本使用项目自身的 Python 解释器运行（能直接 import 项目依赖）。

        Parameters:
            skill_name: Skill 名称
            script_name: 脚本文件名 (如 "scripts/process.py")，仅限技能目录内
            args: 传递给脚本的参数
            timeout: 最长执行秒数，默认 300；长回测/数据抓取可调大

        Returns:
            脚本执行的输出结果
        """
        return await self._manager.execute_skill_script(
            skill_name, script_name, args, timeout
        )

    @property
    def tools(self) -> list:
        """返回供 Worker Agent 使用的工具函数列表（绑定方法）"""
        return [
            self.list_available_skills,
            self.get_skill_instructions,
            self.load_skill_resource,
            self.refresh_skills,
            self.execute_skill_script,
        ]
