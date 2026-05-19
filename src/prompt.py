from __future__ import annotations

import datetime
import platform
import os
import subprocess
import shutil
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from skills.SkillsManager import SkillsManager


def get_skills_summary(skills_manager: SkillsManager) -> str:
    """获取 Skills 摘要，用于系统提示"""
    try:
        return skills_manager.get_skills_summary()
    except Exception:
        return ""


def get_system_info():
    """获取当前系统环境信息"""
    info = {}
    info['os'] = platform.system()
    info['os_version'] = platform.version()
    info['os_release'] = platform.release()
    info['architecture'] = platform.machine()

    info['python_version'] = platform.python_version()

    info['cpu'] = platform.processor() or "Unknown"
    info['cpu_cores'] = os.cpu_count()

    try:
        import psutil
        mem = psutil.virtual_memory()
        info['memory_total'] = f"{mem.total / (1024**3):.1f} GB"
        info['memory_available'] = f"{mem.available / (1024**3):.1f} GB"
    except ImportError:
        info['memory_total'] = "Unknown"
        info['memory_available'] = "Unknown"

    info['gpu'] = detect_gpu()
    info['available_tools'] = detect_available_tools()
    
    return info


def detect_gpu():
    """检测 GPU 信息"""
    gpu_info = {"has_gpu": False, "gpus": []}
    
    system = platform.system()

    subprocess_kwargs = {
        "capture_output": True,
        "text": True,
        "timeout": 10
    }
    if system == "Windows":
        subprocess_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            **subprocess_kwargs
        )
        
        if result.returncode == 0 and result.stdout.strip():
            gpu_info["has_gpu"] = True
            for line in result.stdout.strip().split('\n'):
                parts = line.split(', ')
                if len(parts) >= 2:
                    gpu_info["gpus"].append({
                        "name": parts[0].strip(),
                        "memory": f"{int(float(parts[1].strip()))} MB"
                    })
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass

    if not gpu_info["has_gpu"]:
        if system == "Windows":
            try:
                result = subprocess.run(
                    ["wmic", "path", "win32_VideoController", "get", "name"],
                    **subprocess_kwargs
                )
                if result.returncode == 0:
                    lines = [l.strip() for l in result.stdout.split('\n') if l.strip() and l.strip() != "Name"]
                    for gpu_name in lines:
                        if gpu_name:
                            gpu_info["gpus"].append({"name": gpu_name, "memory": "Unknown"})
                    if gpu_info["gpus"]:
                        gpu_info["has_gpu"] = True
            except Exception:
                pass

        elif system == "Linux":
            try:
                result = subprocess.run(
                    ["lspci"], capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if 'VGA' in line or '3D' in line or 'Display' in line:
                            gpu_info["gpus"].append({"name": line.split(': ')[-1] if ': ' in line else line, "memory": "Unknown"})
                    if gpu_info["gpus"]:
                        gpu_info["has_gpu"] = True
            except Exception:
                pass
        
        elif system == "Darwin":  # macOS
            try:
                result = subprocess.run(
                    ["system_profiler", "SPDisplaysDataType"],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if 'Chipset Model' in line:
                            gpu_name = line.split(':')[-1].strip()
                            gpu_info["gpus"].append({"name": gpu_name, "memory": "Unknown"})
                    if gpu_info["gpus"]:
                        gpu_info["has_gpu"] = True
            except Exception:
                pass
    
    return gpu_info


def detect_available_tools():
    """检测系统中可用的常用工具"""
    tools = {}
    common_tools = ['git', 'node', 'npm', 'python', 'pip', 'docker', 'ffmpeg', 'curl', 'wget']
    for tool in common_tools:
        tools[tool] = shutil.which(tool) is not None
    
    return tools


def format_system_info():
    """格式化系统信息为字符串"""
    info = get_system_info()
    
    lines = [
        "## System Environment",
        "",
        f"- **Operating System**: {info['os']} {info['os_release']} ({info['architecture']})",
        f"- **Python Version**: {info['python_version']}",
        f"- **CPU**: {info['cpu']} ({info['cpu_cores']} cores)",
        f"- **Memory**: {info['memory_total']} (Available: {info['memory_available']})",
    ]

    gpu = info['gpu']
    if gpu['has_gpu'] and gpu['gpus']:
        gpu_list = ", ".join([f"{g['name']} ({g['memory']})" for g in gpu['gpus']])
        lines.append(f"- **GPU**: {gpu_list}")
    else:
        lines.append("- **GPU**: No dedicated GPU detected")

    available = [tool for tool, exists in info['available_tools'].items() if exists]
    if available:
        lines.append(f"- **Available Tools**: {', '.join(available)}")
    
    lines.append("")
    return "\n".join(lines)


system_info = format_system_info()
PROMPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")


def format_prompt_current_time() -> str:
    """系统提示中使用的当前本地时间（构建提示时取一次）。"""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def load_prompt(filename: str) -> str:
    """从 prompts 目录加载 markdown 格式的 prompt 模板"""
    filepath = os.path.join(PROMPTS_DIR, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def get_skills_layout_text(skills_manager: SkillsManager) -> str:
    """从 prompts/skills_layout.md 加载 Skills 目录说明，并填入本机 skills 路径。"""
    try:
        root = skills_manager.skills_dir.resolve()
        skills_root_path = f"本机绝对路径：`{root}`"
    except Exception:
        skills_root_path = "相对项目根目录：`skills/`"
    return load_prompt("skills_layout.md").format(skills_root_path=skills_root_path)


def format_long_term_memory_for_prompt(memory_injection: str) -> str:
    """将 LongTermMemory.get_injection() 的文本格式化为可嵌入各 Agent system prompt 的区块。"""
    t = (memory_injection or "").strip()
    if not t:
        return ""
    return f"## Persistent memory (SOUL & USER)\n\n{t}\n"


def get_skills_as_in_system_prompt(skills_manager: SkillsManager) -> str:
    layout = get_skills_layout_text(skills_manager).rstrip()
    summary = get_skills_summary(skills_manager).rstrip()
    if not layout and not summary:
        return ""
    if not layout:
        return summary
    if not summary:
        return layout
    return f"{layout}\n\n{summary}"


def get_common_conduct() -> str:
    """三角色共用的行为约束块（从 prompts/common_conduct.md 加载）。"""
    return load_prompt("common_conduct.md")


def get_manager_system_prompt(
    skills_manager: SkillsManager,
    memory_injection: str = "",
) -> str:
    """构建 Manager Agent 的系统提示（含当前 skills 摘要，随热加载更新）。"""
    template = load_prompt("manager_system.md")
    return template.format(
        current_time=format_prompt_current_time(),
        system_info=system_info,
        skills_layout=get_skills_layout_text(skills_manager),
        skills_summary=get_skills_summary(skills_manager),
        long_term_memory=format_long_term_memory_for_prompt(memory_injection),
        common_conduct=get_common_conduct(),
    )


def get_worker_system_prompt(
    skills_manager: SkillsManager,
    memory_injection: str = "",
) -> str:
    """构建 Worker Agent 的系统提示（含当前 skills 摘要，随热加载更新）。"""
    template = load_prompt("worker_system.md")
    return template.format(
        current_time=format_prompt_current_time(),
        skills_layout=get_skills_layout_text(skills_manager),
        skills_summary=get_skills_summary(skills_manager),
        long_term_memory=format_long_term_memory_for_prompt(memory_injection),
        common_conduct=get_common_conduct(),
    )


def get_coordinator_system_prompt(
    skills_manager: SkillsManager,
    memory_injection: str = "",
) -> str:
    """构建 Coordinator 的系统提示（含 Skills 摘要，可直接解决简单任务或派发）。"""
    template = load_prompt("coordinator_system.md")
    return template.format(
        current_time=format_prompt_current_time(),
        skills_layout=get_skills_layout_text(skills_manager),
        skills_summary=get_skills_summary(skills_manager),
        long_term_memory=format_long_term_memory_for_prompt(memory_injection),
        common_conduct=get_common_conduct(),
    )
