# -*- coding: utf-8 -*-
import re
import shutil
import threading
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import logger


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
    loaded: bool = False

    @property
    def name(self) -> str:
        return self.metadata.name

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

    def __init__(self, skills_dir: str | Path | None = None):
        if skills_dir is None:
            skills_dir = Path(__file__).resolve().parent.parent / "skills"
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

    def _discover_skills(self, *, quiet: bool = False) -> None:
        if not self.skills_dir.exists():
            if not quiet:
                logger.info(f"Skills 目录不存在: {self.skills_dir}")
            self.skills_dir.mkdir(parents=True, exist_ok=True)
            return

        discovered = 0

        for item in self.skills_dir.iterdir():
            if not item.is_dir():
                continue
            skill_file = item / self.SKILL_FILENAME
            if not skill_file.is_file():
                continue
            result = self._parse_skill_file(skill_file)
            if result:
                metadata, instructions = result
                self.skills[metadata.name] = Skill(
                    metadata=metadata,
                    instructions=instructions,
                    loaded=False,
                )
                discovered += 1
                if not quiet:
                    logger.debug(f"发现 Skill: {metadata.name}")

        if not quiet:
            logger.info(f"共发现 {discovered} 个 Skills")

    def refresh(self, *, quiet: bool = False) -> None:
        with self._refresh_lock:
            self.skills.clear()
            self._discover_skills(quiet=quiet)

    def _skill_markdown_path(self, name: str) -> Optional[Path]:
        skill = self.skills.get(name)
        if not skill:
            return None
        p = skill.path / self.SKILL_FILENAME
        return p if p.is_file() else None

    def create_skill(
        self, name: str, description: str, instructions: str = ""
    ) -> Tuple[bool, str]:
        skill_dir = self.skills_dir / name
        skill_file = skill_dir / self.SKILL_FILENAME

        with self._refresh_lock:
            if skill_dir.exists() or skill_file.exists():
                return False, f"Skill '{name}' 已存在"
            try:
                skill_dir.mkdir(parents=True, exist_ok=False)
                fm = yaml.dump(
                    {"name": name, "description": description},
                    allow_unicode=True,
                    default_flow_style=False,
                    sort_keys=False,
                ).strip()
                body = instructions.strip()
                skill_file.write_text(f"---\n{fm}\n---\n\n{body}\n", encoding="utf-8")
            except OSError as e:
                return False, str(e)
            self.refresh(quiet=True)
        return True, ""

    def update_skill(
        self,
        name: str,
        *,
        description: Optional[str] = None,
        instructions: Optional[str] = None,
    ) -> Tuple[bool, str]:
        with self._refresh_lock:
            if description is None and instructions is None:
                return True, ""

            path = self._skill_markdown_path(name)
            if path is None:
                return False, f"Skill '{name}' 不存在"

            parsed = self._parse_skill_file(path)
            if not parsed:
                return False, f"无法解析 {path}"

            meta, cur_instructions = parsed
            new_desc = description if description is not None else meta.description
            new_instr = instructions if instructions is not None else cur_instructions

            try:
                fm = yaml.dump(
                    {"name": meta.name, "description": new_desc},
                    allow_unicode=True,
                    default_flow_style=False,
                    sort_keys=False,
                ).strip()
                body = new_instr.strip()
                path.write_text(f"---\n{fm}\n---\n\n{body}\n", encoding="utf-8")
            except OSError as e:
                return False, str(e)
            self.refresh(quiet=True)
        return True, ""

    def delete_skill(self, name: str) -> Tuple[bool, str]:
        with self._refresh_lock:
            skill = self.skills.get(name)
            if not skill:
                return False, f"Skill '{name}' 不存在"

            try:
                shutil.rmtree(skill.path)
            except OSError as e:
                return False, str(e)
            self.refresh(quiet=True)
        return True, ""

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
        skill.loaded = True
        logger.info(f"加载 Skill 指令: {name}")
        return skill.instructions

    def load_skill_resource(self, skill_name: str, resource_name: str) -> Optional[str]:
        skill = self.skills.get(skill_name)
        if not skill:
            return None

        if resource_name in skill.resources:
            return skill.resources[resource_name]

        resource_path = skill.path / resource_name
        if not resource_path.exists():
            logger.warning(f"Skill {skill_name} 的资源 {resource_name} 不存在")
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
            if item.is_file() and item.name != self.SKILL_FILENAME:
                rel_path = item.relative_to(skill_dir)
                resources.append(str(rel_path))
        return resources

    def execute_skill_script(self, skill_name: str, script_name: str, args: str = "") -> str:
        import subprocess

        skill = self.skills.get(skill_name)
        if not skill:
            return f"错误: Skill '{skill_name}' 不存在"

        script_path = skill.path / script_name
        if not script_path.exists():
            return f"错误: 脚本 '{script_name}' 不存在"

        ext = script_path.suffix.lower()
        executors = {
            ".py": ["python"],
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
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=60,
                cwd=str(skill.path),
            )
            output = result.stdout + result.stderr
            return (
                f"返回码: {result.returncode}\n输出:\n{output}"
                if output
                else f"执行完成，返回码: {result.returncode}"
            )
        except subprocess.TimeoutExpired:
            return "错误: 脚本执行超时 (60秒)"
        except Exception as e:
            return f"执行错误: {e}"

    def match_skill(self, query: str) -> Optional[Skill]:
        query_lower = query.lower()

        for skill in self.skills.values():
            if skill.name in query_lower:
                return skill

            desc_words = skill.description.lower().split()
            query_words = query_lower.split()
            matches = sum(1 for w in query_words if any(w in dw for dw in desc_words))
            if matches >= 2:
                return skill

        return None
