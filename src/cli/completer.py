"""Tab 补全：/命令、/agent 参数、@文件路径。"""

from __future__ import annotations

from pathlib import Path

from prompt_toolkit.completion import Completer, Completion

from app_config import get_agent_roles
from path_sandbox import runtime_repo_root

COMMANDS: tuple[str, ...] = (
    "/help",
    "/exit",
    "/quit",
    "/clear",
    "/status",
    "/config",
    "/usage",
    "/pwd",
    "/cd",
    "/skills",
    "/agent",
    "/api",
    "/compress",
    "/cancel",
    "/stop",
    "/load",
    "/trace",
    "/tasks",
)

_COMPLETION_LIMIT = 50


class AgentCompleter(Completer):
    """根据光标前上下文补全命令、角色名或文件路径。"""

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor

        if text.startswith("/") and " " not in text:
            word = document.get_word_before_cursor(WORD=True)
            for cmd in COMMANDS:
                if cmd.startswith(word):
                    yield Completion(cmd, start_position=-len(word), display_meta="command")
            return

        if text.startswith("/agent "):
            parts = text.split()
            prefix = parts[-1] if len(parts) > 1 else ""
            if len(parts) == 2:
                for role in get_agent_roles():
                    if role.startswith(prefix):
                        yield Completion(role, start_position=-len(prefix), display_meta="role")
            return

        if text.startswith("/cd ") or text.startswith("/load "):
            prefix = text.split(" ", 1)[1] if " " in text else ""
            yield from _iter_file_completions(prefix, at_mode=False)
            return

        at_index = text.rfind("@")
        if at_index != -1:
            fragment = text[at_index + 1 :]
            if " " in fragment:
                return
            yield from _iter_file_completions(fragment, at_mode=True)


def _resolve_parent(fragment: str) -> tuple[Path, str]:
    path = Path(fragment.replace("\\", "/")).expanduser()
    parent = path.parent if str(path.parent) not in (".", "") else Path(".")
    prefix = path.name

    if parent != Path("."):
        for root in (Path.cwd(), runtime_repo_root()):
            candidate = (root / parent).resolve()
            if candidate.is_dir():
                return candidate, prefix
        if not parent.is_absolute():
            resolved = (Path.cwd() / parent).resolve()
            if resolved.is_dir():
                return resolved, prefix
    cwd = Path.cwd()
    if cwd.is_dir():
        return cwd, prefix
    return parent, prefix


def _iter_file_completions(fragment: str, *, at_mode: bool):
    parent, prefix = _resolve_parent(fragment)
    if not parent.exists():
        return

    try:
        children = sorted(parent.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except (PermissionError, OSError):
        return

    count = 0
    for child in children:
        if not child.name.startswith(prefix):
            continue
        try:
            candidate = child.relative_to(Path.cwd()).as_posix()
        except ValueError:
            candidate = child.as_posix()
        if child.is_dir():
            candidate += "/"
        display = ("@" if at_mode else "") + candidate
        yield Completion(
            candidate,
            start_position=-len(fragment),
            display=display,
            display_meta="dir" if child.is_dir() else "file",
        )
        count += 1
        if count >= _COMPLETION_LIMIT:
            break
