"""Shared slash-command and @file-path completion logic for CLI and TUI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from redlotus.config.app_config import get_agent_roles

COMMANDS: tuple[str, ...] = (
    "/help",
    "/exit",
    "/quit",
    "/clear",
    "/status",
    "/config",
    "/usage",
    "/panel",
    "/LTM",
    "/STM",
    "/pwd",
    "/cd",
    "/skills",
    "/agent",
    "/effort",
    "/api",
    "/compress",
    "/cancel",
    "/stop",
    "/load",
    "/trace",
    "/tasks",
)

EFFORT_VALUES: tuple[str, ...] = ("off", "minimal", "low", "medium", "high", "xhigh", "max")

CompletionKind = Literal["command", "agent_role", "effort_value", "literal_choice", "file_path"]

_SUBCOMMAND_CHOICES: dict[str, tuple[str, ...]] = {
    "/ltm": ("show", "clear"),
    "/stm": ("show", "clear"),
    "/cancel": ("agent",),
}


@dataclass(frozen=True)
class InputCompletion:
    """Describes what to complete for a given input prefix."""

    kind: CompletionKind
    prefix: str
    at_mode: bool = False
    choices: tuple[str, ...] = ()


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

    if text.startswith("/effort "):
        parts = text.split()
        prefix = parts[-1] if len(parts) > 1 else ""
        if len(parts) == 2:
            return InputCompletion(kind="agent_role", prefix=prefix)
        if len(parts) == 3:
            return InputCompletion(kind="effort_value", prefix=prefix)
        return None

    if text.startswith("/cd "):
        prefix = text.split(" ", 1)[1] if " " in text else ""
        return InputCompletion(kind="file_path", prefix=prefix)

    for cmd, choices in _SUBCOMMAND_CHOICES.items():
        if text.lower().startswith(cmd + " "):
            parts = text.split()
            prefix = parts[-1] if len(parts) > 1 else ""
            if len(parts) == 2:
                return InputCompletion(kind="literal_choice", prefix=prefix, choices=choices)
            return None

    at_index = text.rfind("@")
    if at_index != -1:
        fragment = text[at_index + 1 :]
        if " " in fragment:
            return None
        return InputCompletion(kind="file_path", prefix=fragment, at_mode=True)

    return None


def iter_command_completions(prefix: str):
    """Yield command strings matching *prefix*."""
    folded = prefix.lower()
    for cmd in COMMANDS:
        if cmd.lower().startswith(folded):
            yield cmd


def iter_agent_role_completions(prefix: str):
    """Yield agent role names matching *prefix*."""
    for role in get_agent_roles():
        if role.startswith(prefix):
            yield role


def iter_effort_value_completions(prefix: str):
    """Yield thinking on/off + effort-level values matching *prefix*."""
    folded = prefix.lower()
    for value in EFFORT_VALUES:
        if value.startswith(folded):
            yield value


def iter_literal_choice_completions(prefix: str, choices: tuple[str, ...]):
    """Yield fixed subcommand choices (e.g. show/clear) matching *prefix*."""
    folded = prefix.lower()
    for choice in choices:
        if choice.startswith(folded):
            yield choice
