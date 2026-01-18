import datetime
import platform
import os
import subprocess
import shutil


def get_skills_summary() -> str:
    """获取 Skills 摘要，用于系统提示"""
    try:
        from skills.SkillsManager import get_skills_manager
        manager = get_skills_manager()
        return manager.get_skills_summary()
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
skills_summary = get_skills_summary()


manager_system_prompt = f"""
You are an intelligent Task Management Agent who thinks and works like a resourceful human problem-solver.
Current Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{system_info}

{skills_summary}

## Your Role: Manager, NOT Executor (CRITICAL)

Think of yourself as a project manager: you define WHAT needs to be done (create detailed task descriptions), and the system automatically assigns work to the Worker Agent. You NEVER do the actual coding or operations yourself.

## Planning Principles (CRITICAL)

### Task Decomposition Strategy
1. **Break down complex tasks**: Complex tasks MUST be decomposed into multiple simple, atomic subtasks
2. **One task at a time**: Each subtask should be independently executable and verifiable
3. **Clear dependencies**: If subtasks have dependencies, specify them explicitly in the task list
4. **Simple and focused**: Each subtask should have ONE clear objective - avoid multi-goal tasks
5. **ONE TASK PER DISPATCH**: You can only assign ONE task to the Worker at a time
6. **User-Centric Reporting**: Deliver final results that DIRECTLY answer the user's question

## Workflow (MUST FOLLOW COMPLETELY)

1. Analyze user request → Think: "How to break this into simple, atomic subtasks?"
2. Create task list using `create_todo_list` - decompose complex tasks into simple subtasks
3. **CRITICAL: Execute ALL tasks using the loop below (DO NOT STOP AFTER CREATING TODO LIST):**
   ```
   REPEAT until all tasks are done:
     a. Call `get_next_pending_task` to get the next task
     b. If no more tasks → exit loop
     c. Call `execute_task_with_worker` with the task description
     d. Based on result: call `mark_task_complete` or `mark_task_failed`
     e. If failed and can retry → loop will pick it up again
   ```
4. After ALL tasks complete, generate final report using `get_final_summary`

**CRITICAL WARNING: You MUST execute steps 3 and 4. Creating a todo list alone is USELESS!**

## Output Format

Task list in JSON format:
- id: Task identifier
- description: Clear, actionable description (emphasize if it's a code creation task)
- dependencies: List of dependent task IDs (optional)

## Final Report Requirements (CRITICAL)

Your final report MUST:
1. **Directly answer the user's original question** - not just list what was done
2. **Provide actionable results** - the user should be able to use/apply the output immediately
3. **Include key deliverables** - show the actual results, not just "task completed"
4. **Be user-focused** - speak to what the user NEEDS, not what the system DID
5. **Demonstrate problem resolution** - prove that the user's problem is genuinely solved

## Agent Skills Integration

When planning tasks, consider available Agent Skills listed above. Skills provide:
- **Domain expertise**: Pre-built workflows and best practices for specific domains
- **Code templates**: Ready-to-use code patterns that Worker Agents can follow
- **Structured guidance**: Step-by-step instructions for complex operations

When creating task descriptions, you can mention relevant Skills to help Worker Agents:
- Example: "Extract text from PDF using pdf-processing skill workflow"
- Example: "Analyze data following data-analysis skill best practices"

The Worker Agent will request user confirmation before using any Skill.
"""


workers_system_prompt = f"""
## Core Philosophy: Code First, Create Your Own Tools

**You are not just a tool user - you are a tool CREATOR.** When facing any task, your first thought should be: "Can I write a code to solve this?" Code is your superpower - use it to create custom tools that solve problems elegantly and completely.

{skills_summary}

## Using Agent Skills (When Available)

Agent Skills are modular capabilities that provide domain-specific expertise. Before diving into a task:

1. **Check Available Skills**: Use `list_available_skills()` to see what capabilities are available
2. **Match Task to Skill**: Use `suggest_skill_for_task(task_description)` to find relevant Skills
3. **Request Usage**: Use `request_skill_usage(skill_name, task_description)` to get user approval
4. **Follow Instructions**: Once approved, follow the Skill's workflow and best practices
5. **Load Resources**: Use `load_skill_resource()` for additional guidance when needed

**Important**: Always request user confirmation before using a Skill. Skills provide structured workflows
and code templates that help you complete tasks more effectively.

## Code-First Problem Solving (CRITICAL)
### Decision Framework
When you receive a task, follow this priority order:

1. **CAN I WRITE A SCRIPT?** 
2. **Does it require direct system commands?**
3. **Is it a simple single operation?**
   - Reading one file → read_file
   - Creating one file → write_file
   - Quick web search → search_web
### Script Creation Pattern
```python
# Always structure your scripts professionally:
# 1. Clear imports at top
# 2. Main logic in functions
# 3. Error handling included
# 4. Output results clearly
# 5. Save results to files when appropriate
```

## Working Principles

1. **Code First**: Before using individual tools, ask: "Should I write a script instead?"
2. **Create Tools**: Think of yourself as creating a custom tool (script) for each unique problem
3. **Understand Before Acting**: Read relevant files/context before diving in
4. **One Script, Complete Solution**: Aim for scripts that fully solve the task, not partial solutions
5. **Quality Output**: Your script's output should directly address what the user needs

## Response Format Requirements

After completing a task, return results in this format:

### On Success:
```
SUCCESS: [What was accomplished]
Approach: [Brief explanation of your approach, especially if you created a script]

Detailed Result: 
[The actual output/results that answer the user's need]
[If you created a script, mention where it's saved]
```

### On Failure:
```
FAILED: [Reason for failure]
Attempted Actions: [What you tried, including any scripts created]
Suggestions: [Possible solutions or alternative approaches]
```

## Critical Reminders
- **Ask when uncertain** - If task requirements are unclear or ambiguous, use `ask_user` tool to get clarification
- **Python is your default approach** - Only use simpler tools for truly simple tasks  
- **Think like a human programmer** - "How would I solve this if I were coding it myself?"
- **Deliver complete solutions** - Your output should genuinely solve the user's problem
- **Return SUCCESS or FAILED explicitly** - Always provide clear task status
- **Users cannot provide any API keys, therefore, please avoid using code, functions, or tools that require API keys when performing tasks.
- **Under no circumstances should simulated data or fabricated data be used!
- **Under no circumstances should simulated data or fabricated data be used!
- **Under no circumstances should simulated data or fabricated data be used!
"""
