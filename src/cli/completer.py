"""Tab 补全：/命令、/agent 参数、@文件路径。"""

from __future__ import annotations

from pathlib import Path

from prompt_toolkit.completion import Completer, Completion

from cli.completion import (
    completion_for_input,
    iter_agent_role_completions,
    iter_command_completions,
    iter_effort_value_completions,
    iter_literal_choice_completions,
)
from infra.path_sandbox import runtime_repo_root

_COMPLETION_LIMIT = 50


class AgentCompleter(Completer):
    """根据光标前上下文补全命令、角色名或文件路径。"""

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        ctx = completion_for_input(text)
        if ctx is None:
            return

        if ctx.kind == "command":
            word = document.get_word_before_cursor(WORD=True)
            for cmd in iter_command_completions(word):
                yield Completion(cmd, start_position=-len(word), display_meta="command")
            return

        if ctx.kind == "agent_role":
            for role in iter_agent_role_completions(ctx.prefix):
                yield Completion(role, start_position=-len(ctx.prefix), display_meta="role")
            return

        if ctx.kind == "effort_value":
            for value in iter_effort_value_completions(ctx.prefix):
                yield Completion(value, start_position=-len(ctx.prefix), display_meta="effort")
            return

        if ctx.kind == "literal_choice":
            for choice in iter_literal_choice_completions(ctx.prefix, ctx.choices):
                yield Completion(choice, start_position=-len(ctx.prefix), display_meta="option")
            return

        if ctx.kind == "file_path":
            yield from _iter_file_completions(ctx.prefix, at_mode=ctx.at_mode)


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
