# -*- coding: utf-8 -*-
import re
import threading
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from infra import logger
from infra.paths import skills_dir as default_skills_dir


@dataclass
class SkillMetadata:
    name: str
    description: str
    path: Path

    def to_summary(self) -> str:
        return f"- **{self.name}**: {self.description}"


@dataclass
class Skill:
    metadata: SkillMetadata
    instructions: str = ""
    resources: Dict[str, str] = field(default_factory=dict)

    @property
    def description(self) -> str:
        return self.metadata.description

    @property
    def path(self) -> Path:
        return self.metadata.path


class SkillsManager:
    """
    Skills 管理器：仅支持 skills/<目录名>/SKILL.md；增删改查与显式 refresh。
    """

    FRONTMATTER_PATTERN = re.compile(
        r"^---\s*\n(.*?)\n---\s*\n",
        re.DOTALL,
    )
    SKILL_FILENAME = "SKILL.md"
    IGNORED_RESOURCE_DIRS = {".git", "__pycache__", ".idea", ".vscode"}

    def __init__(self, skills_dir: str | Path | None = None):
        if skills_dir is None:
            skills_dir = default_skills_dir()
        self.skills_dir = Path(skills_dir)
        self.skills: Dict[str, Skill] = {}

        self._refresh_lock = threading.Lock()

        self._discover_skills()

    def _parse_skill_file(self, skill_path: Path) -> Optional[Tuple[SkillMetadata, str]]:
        try:
            content = skill_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning(f"无法读取 Skill 文件 {skill_path}: {e}")
            return None

        match = self.FRONTMATTER_PATTERN.match(content)
        if not match:
            logger.warning(f"Skill 文件 {skill_path} 缺少 YAML 前置元数据")
            return None

        try:
            frontmatter = yaml.safe_load(match.group(1))
        except yaml.YAMLError as e:
            logger.warning(f"Skill 文件 {skill_path} YAML 解析失败: {e}")
            return None

        if not isinstance(frontmatter, dict):
            logger.warning(f"Skill 文件 {skill_path} YAML 格式无效")
            return None

        name = str(frontmatter.get("name", ""))
        description = str(frontmatter.get("description", ""))

        instructions = content[match.end() :].strip()
        metadata = SkillMetadata(
            name=name,
            description=description,
            path=skill_path.parent,
        )
        return metadata, instructions

    def refresh(self) -> None:
        fresh: Dict[str, Skill] = {}
        with self._refresh_lock:
            self._discover_skills(into=fresh)
            self.skills = fresh

    def _discover_skills(self, into: Dict[str, Skill] | None = None) -> None:
        target = self.skills if into is None else into
        if into is not None:
            into.clear()
        if not self.skills_dir.exists():
            logger.info(f"Skills 目录不存在: {self.skills_dir}")
            self.skills_dir.mkdir(parents=True, exist_ok=True)
            return

        for item in self.skills_dir.iterdir():
            if not item.is_dir():
                continue
            skill_file = item / self.SKILL_FILENAME
            if not skill_file.is_file():
                continue
            result = self._parse_skill_file(skill_file)
            if result:
                metadata, instructions = result
                if not metadata.name:
                    logger.warning(
                        f"Skill {skill_file} 缺少 name，回退使用文件夹名 '{item.name}'"
                    )
                    metadata.name = item.name
                if metadata.name in target:
                    logger.warning(
                        f"Skill 名称冲突: '{metadata.name}' 已存在，"
                        f"目录 '{item.name}' 将覆盖之前的同名技能"
                    )
                target[metadata.name] = Skill(
                    metadata=metadata,
                    instructions=instructions,
                )

    def get_skill(self, name: str) -> Optional[Skill]:
        return self.skills.get(name)

    def get_all_metadata(self) -> List[SkillMetadata]:
        return [self.skills[n].metadata for n in sorted(self.skills)]

    def get_skills_summary(self) -> str:
        if not self.skills:
            return "当前没有可用的 Skills。"

        lines = ["## 可用的 Agent Skills", ""]
        for metadata in self.get_all_metadata():
            lines.append(metadata.to_summary())
        lines.append("")
        lines.append("使用 `get_skill_instructions(skill_name)` 获取具体 Skill 的详细指令。")
        return "\n".join(lines)

    def load_skill_instructions(self, name: str) -> Optional[str]:
        skill = self.skills.get(name)
        if not skill:
            return None
        logger.info(f"加载 Skill 指令: {name}")
        return skill.instructions

    def load_skill_resource(self, skill_name: str, resource_name: str) -> Optional[str]:
        skill = self.skills.get(skill_name)
        if not skill:
            return None

        if resource_name in skill.resources:
            return skill.resources[resource_name]

        resource_path = self._resolve_within(skill.path, resource_name)
        if resource_path is None or not resource_path.exists():
            logger.warning(f"Skill {skill_name} 的资源 {resource_name} 不存在或越界")
            return None

        try:
            content = resource_path.read_text(encoding="utf-8")
            skill.resources[resource_name] = content
            logger.info(f"加载 Skill 资源: {skill_name}/{resource_name}")
            return content
        except Exception as e:
            logger.warning(f"无法读取 Skill 资源 {resource_path}: {e}")
            return None

    def list_skill_resources(self, skill_name: str) -> List[str]:
        skill = self.skills.get(skill_name)
        if not skill:
            return []

        resources = []
        skill_dir = skill.path
        for item in skill_dir.rglob("*"):
            if not item.is_file() or item.name == self.SKILL_FILENAME:
                continue
            rel_path = item.relative_to(skill_dir)
            if any(part in self.IGNORED_RESOURCE_DIRS for part in rel_path.parts):
                continue
            resources.append(str(rel_path))
        return resources

    @staticmethod
    def _resolve_within(skill_dir: Path, relative: str) -> Optional[Path]:
        """把 relative 解析到 skill_dir 内；绝对路径或 .. 逃逸出目录则返回 None。"""
        base = skill_dir.resolve()
        try:
            target = (base / relative).resolve()
        except Exception:
            return None
        return target if target.is_relative_to(base) else None

    async def execute_skill_script(
        self, skill_name: str, script_name: str, args: str = "", timeout: float = 300
    ) -> str:
        import sys
        import subprocess
        from infra.subprocess_runner import run_subprocess

        skill = self.skills.get(skill_name)
        if not skill:
            return f"错误: Skill '{skill_name}' 不存在"

        script_path = self._resolve_within(skill.path, script_name)
        if script_path is None:
            return f"错误: 脚本路径越界: '{script_name}'"
        if not script_path.exists():
            return f"错误: 脚本 '{script_name}' 不存在"

        ext = script_path.suffix.lower()
        py_exec = "python" if getattr(sys, "frozen", False) else sys.executable
        executors = {
            ".py": [py_exec],
            ".sh": ["bash"],
            ".bat": ["cmd", "/c"],
            ".ps1": ["powershell", "-File"],
        }

        if ext not in executors:
            return f"错误: 不支持的脚本类型 '{ext}'"

        cmd = executors[ext] + [str(script_path)]
        if args:
            cmd.extend(args.split())

        try:
            stdout, stderr, return_code = await run_subprocess(
                cmd, shell=False, cwd=str(skill.path), timeout=timeout
            )
            output = stdout + stderr
            return (
                f"返回码: {return_code}\n输出:\n{output}"
                if output
                else f"执行完成，返回码: {return_code}"
            )
        except subprocess.TimeoutExpired:
            return f"错误: 脚本执行超时 ({timeout}秒)"
        except Exception as e:
            return f"执行错误: {e}"
