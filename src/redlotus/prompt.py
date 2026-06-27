from __future__ import annotations

import datetime
import os
import platform
from typing import TYPE_CHECKING

from redlotus.infra.paths import prompts_dir

if TYPE_CHECKING:
    from redlotus.skills.SkillsManager import SkillsManager


PROMPTS_DIR = str(prompts_dir())


def get_skills_summary(skills_manager: SkillsManager) -> str:
    return skills_manager.get_skills_summary()


def get_system_info() -> dict[str, object]:
    return {
        "os": platform.system(),
        "os_release": platform.release(),
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
        "cpu_cores": os.cpu_count(),
    }


def format_system_info() -> str:
    info = get_system_info()
    return "\n".join(
        [
            "## System Environment",
            "",
            f"- **Operating System**: {info['os']} {info['os_release']} ({info['architecture']})",
            f"- **Python Version**: {info['python_version']}",
            f"- **CPU Cores**: {info['cpu_cores']}",
            "",
        ]
    )


system_info = format_system_info()


def format_prompt_current_time() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def load_prompt(filename: str) -> str:
    filepath = os.path.join(PROMPTS_DIR, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def get_skills_layout_text(skills_manager: SkillsManager) -> str:
    root = skills_manager.skills_dir.resolve()
    skills_root_path = f"Local absolute path: `{root}`"
    return load_prompt("skills_layout.md").format(skills_root_path=skills_root_path)


def format_long_term_memory_for_prompt(memory_injection: str) -> str:
    text = (memory_injection or "").strip()
    if not text:
        return ""
    return f"## Persistent memory (SOUL & USER)\n\n{text}\n"


def get_skills_as_in_system_prompt(skills_manager: SkillsManager) -> str:
    layout = get_skills_layout_text(skills_manager).rstrip()
    summary = get_skills_summary(skills_manager).rstrip()
    if not layout:
        return summary
    if not summary:
        return layout
    return f"{layout}\n\n{summary}"


def get_common_conduct() -> str:
    return load_prompt("common_conduct.md")


def _build_role_prompt(
    template_name: str,
    skills_manager: SkillsManager,
    memory_injection: str = "",
    *,
    include_system_info: bool = False,
) -> str:
    template = load_prompt(template_name)
    fields = {
        "current_time": format_prompt_current_time(),
        "skills_layout": get_skills_layout_text(skills_manager),
        "skills_summary": get_skills_summary(skills_manager),
        "long_term_memory": format_long_term_memory_for_prompt(memory_injection),
        "common_conduct": get_common_conduct(),
    }
    if include_system_info:
        fields["system_info"] = system_info
    return template.format(**fields)


def get_manager_system_prompt(
    skills_manager: SkillsManager,
    memory_injection: str = "",
) -> str:
    return _build_role_prompt(
        "manager_system.md", skills_manager, memory_injection, include_system_info=True
    )


def get_worker_system_prompt(
    skills_manager: SkillsManager,
    memory_injection: str = "",
) -> str:
    return _build_role_prompt("worker_system.md", skills_manager, memory_injection)


def get_coordinator_system_prompt(
    skills_manager: SkillsManager,
    memory_injection: str = "",
) -> str:
    return _build_role_prompt("coordinator_system.md", skills_manager, memory_injection)
