"""Shared slash-command and @file-path completion logic for CLI and TUI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app_config import get_agent_roles

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

CompletionKind = Literal["command", "agent_role", "file_path"]


@dataclass(frozen=True)
class InputCompletion:
    """Describes what to complete for a given input prefix."""

    kind: CompletionKind
    prefix: str
    at_mode: bool = False


def completion_for_input(text: str) -> InputCompletion | None:
    """Return completion context for *text*, or None if no completion applies."""
    if text.startswith("/") and " " not in text:
        return InputCompletion(kind="command", prefix=text)

    if text.startswith("/agent "):
        parts = text.split()
        prefix = parts[-1] if len(parts) > 1 else ""
        if len(parts) == 2:
            return InputCompletion(kind="agent_role", prefix=prefix)
        return None

    if text.startswith("/cd ") or text.startswith("/load "):
        prefix = text.split(" ", 1)[1] if " " in text else ""
        return InputCompletion(kind="file_path", prefix=prefix)

    at_index = text.rfind("@")
    if at_index != -1:
        fragment = text[at_index + 1 :]
        if " " in fragment:
            return None
        return InputCompletion(kind="file_path", prefix=fragment, at_mode=True)

    return None


def iter_command_completions(prefix: str):
    """Yield command strings matching *prefix*."""
    for cmd in COMMANDS:
        if cmd.startswith(prefix):
            yield cmd


def iter_agent_role_completions(prefix: str):
    """Yield agent role names matching *prefix*."""
    for role in get_agent_roles():
        if role.startswith(prefix):
            yield role
