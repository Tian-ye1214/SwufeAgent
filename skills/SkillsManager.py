# -*- coding: utf-8 -*-
import re
import yaml
from pathlib import Path
from typing import Dict, List, Optional
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
    Skills 管理器
    
    负责发现、解析和管理所有 Skills。实现渐进式加载机制，
    确保只在需要时将相关内容加载到上下文窗口。
    """

    FRONTMATTER_PATTERN = re.compile(
        r'^---\s*\n(.*?)\n---\s*\n',
        re.DOTALL
    )

    NAME_MAX_LENGTH = 64
    NAME_PATTERN = re.compile(r'^[a-z0-9\-]+$')
    RESERVED_WORDS = ['anthropic', 'claude']

    DESCRIPTION_MAX_LENGTH = 1024
    
    def __init__(self, skills_dir: str = None):
        if skills_dir is None:
            project_root = Path(__file__).parent.parent
            skills_dir = project_root / "skills"
        
        self.skills_dir = Path(skills_dir)
        self.skills: Dict[str, Skill] = {}
        self._metadata_cache: Dict[str, SkillMetadata] = {}

        self._discover_skills()
    
    def _validate_name(self, name: str) -> tuple[bool, str]:
        if not name:
            return False, "name 不能为空"
        
        if len(name) > self.NAME_MAX_LENGTH:
            return False, f"name 长度不能超过 {self.NAME_MAX_LENGTH} 字符"
        
        if not self.NAME_PATTERN.match(name):
            return False, "name 只能包含小写字母、数字和连字符"
        
        for word in self.RESERVED_WORDS:
            if word in name:
                return False, f"name 不能包含保留字: {word}"
        
        if '<' in name or '>' in name:
            return False, "name 不能包含 XML 标签"
        
        return True, ""
    
    def _validate_description(self, description: str) -> tuple[bool, str]:
        if not description:
            return False, "description 不能为空"
        
        if len(description) > self.DESCRIPTION_MAX_LENGTH:
            return False, f"description 长度不能超过 {self.DESCRIPTION_MAX_LENGTH} 字符"
        
        if '<' in description or '>' in description:
            return False, "description 不能包含 XML 标签"
        
        return True, ""
    
    def _parse_skill_file(self, skill_path: Path) -> Optional[tuple[SkillMetadata, str]]:
        """
        解析 SKILL.md 文件
        
        Parameters:
            skill_path: SKILL.md 文件路径
            
        Returns:
            (SkillMetadata, instructions) 元组，解析失败返回 None
        """
        try:
            content = skill_path.read_text(encoding='utf-8')
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

        name = frontmatter.get('name', '')
        valid, error = self._validate_name(name)
        if not valid:
            logger.warning(f"Skill {skill_path} name 验证失败: {error}")
            return None

        description = frontmatter.get('description', '')
        valid, error = self._validate_description(description)
        if not valid:
            logger.warning(f"Skill {skill_path} description 验证失败: {error}")
            return None

        instructions = content[match.end():].strip()
        
        metadata = SkillMetadata(
            name=name,
            description=description,
            path=skill_path.parent
        )
        
        return metadata, instructions
    
    def _discover_skills(self) -> None:
        """
        发现 skills 目录下的所有 Skill
        
        扫描 skills_dir 下的所有子目录，查找包含 SKILL.md 的目录。
        只加载 Level 1 (元数据) 到内存中。
        """
        if not self.skills_dir.exists():
            logger.info(f"Skills 目录不存在: {self.skills_dir}")
            self.skills_dir.mkdir(parents=True, exist_ok=True)
            return
        
        discovered = 0

        # 方式1: 扫描子目录中的 SKILL.md
        for item in self.skills_dir.iterdir():
            if item.is_dir():
                skill_file = item / "SKILL.md"
                if skill_file.exists():
                    result = self._parse_skill_file(skill_file)
                    if result:
                        metadata, instructions = result
                        self.skills[metadata.name] = Skill(
                            metadata=metadata,
                            instructions=instructions,
                            loaded=False
                        )
                        self._metadata_cache[metadata.name] = metadata
                        discovered += 1
                        logger.debug(f"发现 Skill: {metadata.name}")

        # 方式2: 扫描根目录下的 *.skill.md 文件
        for skill_file in self.skills_dir.glob("*.skill.md"):
            result = self._parse_skill_file(skill_file)
            if result:
                metadata, instructions = result
                if metadata.name not in self.skills:
                    self.skills[metadata.name] = Skill(
                        metadata=metadata,
                        instructions=instructions,
                        loaded=False
                    )
                    self._metadata_cache[metadata.name] = metadata
                    discovered += 1
                    logger.debug(f"发现 Skill: {metadata.name}")
        
        logger.info(f"共发现 {discovered} 个 Skills")
    
    def refresh(self) -> None:
        self.skills.clear()
        self._metadata_cache.clear()
        self._discover_skills()
    
    def get_all_metadata(self) -> List[SkillMetadata]:
        return list(self._metadata_cache.values())
    
    def get_skills_summary(self) -> str:
        """
        生成 Skills 摘要，用于系统提示
        
        返回所有可用 Skills 的简短描述，供 Agent 了解有哪些能力可用。
        """
        if not self.skills:
            return "当前没有可用的 Skills。"
        
        lines = ["## 可用的 Agent Skills", ""]
        for metadata in self._metadata_cache.values():
            lines.append(metadata.to_summary())
        lines.append("")
        lines.append("使用 `get_skill_instructions(skill_name)` 获取具体 Skill 的详细指令。")
        
        return "\n".join(lines)
    
    def get_skill(self, name: str) -> Optional[Skill]:
        """获取指定名称的 Skill"""
        return self.skills.get(name)
    
    def load_skill_instructions(self, name: str) -> Optional[str]:
        """
        加载 Skill 的指令内容 (Level 2)
        
        Parameters:
            name: Skill 名称
            
        Returns:
            Skill 的完整指令内容，不存在返回 None
        """
        skill = self.skills.get(name)
        if not skill:
            return None
        
        skill.loaded = True
        logger.info(f"加载 Skill 指令: {name}")
        
        return skill.instructions
    
    def load_skill_resource(self, skill_name: str, resource_name: str) -> Optional[str]:
        """
        加载 Skill 的额外资源文件 (Level 3)
        
        Parameters:
            skill_name: Skill 名称
            resource_name: 资源文件名 (如 "FORMS.md", "scripts/helper.py")
            
        Returns:
            资源文件内容，不存在返回 None
        """
        skill = self.skills.get(skill_name)
        if not skill:
            return None
        
        # 检查缓存
        if resource_name in skill.resources:
            return skill.resources[resource_name]
        
        # 从文件系统加载
        resource_path = skill.path / resource_name
        if not resource_path.exists():
            logger.warning(f"Skill {skill_name} 的资源 {resource_name} 不存在")
            return None
        
        try:
            content = resource_path.read_text(encoding='utf-8')
            skill.resources[resource_name] = content
            logger.info(f"加载 Skill 资源: {skill_name}/{resource_name}")
            return content
        except Exception as e:
            logger.warning(f"无法读取 Skill 资源 {resource_path}: {e}")
            return None
    
    def list_skill_resources(self, skill_name: str) -> List[str]:
        """
        列出 Skill 目录下的所有资源文件
        
        Parameters:
            skill_name: Skill 名称
            
        Returns:
            资源文件路径列表
        """
        skill = self.skills.get(skill_name)
        if not skill:
            return []
        
        resources = []
        skill_dir = skill.path
        
        for item in skill_dir.rglob("*"):
            if item.is_file() and item.name != "SKILL.md":
                rel_path = item.relative_to(skill_dir)
                resources.append(str(rel_path))
        
        return resources
    
    def execute_skill_script(self, skill_name: str, script_name: str, args: str = "") -> str:
        """
        执行 Skill 中的脚本 (Level 3)
        
        Parameters:
            skill_name: Skill 名称
            script_name: 脚本文件名 (如 "scripts/process.py")
            args: 传递给脚本的参数
            
        Returns:
            脚本执行输出
        """
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
                cwd=str(skill.path)
            )
            output = result.stdout + result.stderr
            return f"返回码: {result.returncode}\n输出:\n{output}" if output else f"执行完成，返回码: {result.returncode}"
        except subprocess.TimeoutExpired:
            return "错误: 脚本执行超时 (60秒)"
        except Exception as e:
            return f"执行错误: {e}"
    
    def match_skill(self, query: str) -> Optional[Skill]:
        """
        根据用户查询匹配最相关的 Skill
        
        简单的关键词匹配，实际应用中可以使用更复杂的语义匹配。
        
        Parameters:
            query: 用户查询
            
        Returns:
            最匹配的 Skill，没有匹配返回 None
        """
        query_lower = query.lower()
        
        for skill in self.skills.values():
            # 检查 name 或 description 中是否包含查询关键词
            if skill.name in query_lower:
                return skill
            
            # 简单的关键词匹配
            desc_words = skill.description.lower().split()
            query_words = query_lower.split()
            
            # 计算匹配度
            matches = sum(1 for w in query_words if any(w in dw for dw in desc_words))
            if matches >= 2:  # 至少匹配2个关键词
                return skill
        
        return None


# 全局 Skills 管理器实例
_skills_manager: Optional[SkillsManager] = None


def get_skills_manager() -> SkillsManager:
    """获取全局 Skills 管理器实例"""
    global _skills_manager
    if _skills_manager is None:
        _skills_manager = SkillsManager()
    return _skills_manager


def reset_skills_manager() -> None:
    """重置全局 Skills 管理器"""
    global _skills_manager
    _skills_manager = None

