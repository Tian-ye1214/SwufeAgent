from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pydantic_ai import Agent, FunctionToolset
from pydantic_ai.capabilities import Capability

from runtime import tool_telemetry
from config.app_config import get_agent_run_policy
from ModelGateway.model_factory import create_model


_WORKER_DEFERRED_CAPABILITIES: dict[str, tuple[str, str]] = {
    "file_mutation": (
        "worker_file_mutation",
        "Use for writing, editing, appending files, or creating directories.",
    ),
    "execution": (
        "worker_execution",
        "Use for running shell commands or executing files.",
    ),
    "browser": (
        "worker_browser",
        "Use for browser navigation, screenshots, page interaction, and browser inspection.",
    ),
    "media": (
        "worker_media",
        "Use for reading images or extracting text from documents and attachments.",
    ),
    "memory": (
        "worker_memory",
        "Use for querying short-term memory or maintaining long-term memory.",
    ),
    "skills": (
        "worker_skills",
        "Use for listing, loading, refreshing, or executing Agent Skills.",
    ),
}


def create_function_toolset(
    tools: list,
    *,
    toolset_id: str = "default",
    instructions: str | None = None,
    defer_loading: bool = False,
) -> FunctionToolset:
    wrapped_tools = tool_telemetry.wrap_tools_for_user_notify(
        list(tools), policy=get_agent_run_policy()
    )
    return FunctionToolset(
        wrapped_tools,
        id=toolset_id,
        instructions=instructions,
        defer_loading=defer_loading,
    )


def create_deferred_tool_capability(
    capability_id: str,
    description: str,
    tools: Sequence[Any],
) -> Capability:
    toolset = create_function_toolset(
        list(tools),
        toolset_id=capability_id,
        instructions=description,
        defer_loading=True,
    )
    return Capability(
        id=capability_id,
        description=description,
        toolsets=[toolset],
        defer_loading=True,
    )


def create_worker_toolsets_and_capabilities(
    tool_groups: Mapping[str, Sequence[Any]],
    *,
    core_extra_tools: Sequence[Any] = (),
) -> tuple[list[FunctionToolset], list[Capability]]:
    core_tools = [
        *list(tool_groups.get("core", ())),
        *list(core_extra_tools),
    ]
    toolsets = [
        create_function_toolset(
            core_tools,
            toolset_id="worker_core",
            instructions="Always-available Worker tools for reading, searching, user input, and coordination.",
        )
    ] if core_tools else []

    capabilities: list[Capability] = []
    for group_id, (capability_id, description) in _WORKER_DEFERRED_CAPABILITIES.items():
        tools_for_group = list(tool_groups.get(group_id, ()))
        if not tools_for_group:
            continue
        capabilities.append(
            create_deferred_tool_capability(
                capability_id,
                description,
                tools_for_group,
            )
        )
    return toolsets, capabilities


def create_agent(
    model_name: Any,
    parameter: dict | None,
    instructions: str | None = None,
    *,
    toolsets: list | None = None,
    capabilities: list | None = None,
):
    if parameter is None:
        parameter = {
            "temperature": 1.0,
            "max_tokens": 32768,
            "reasoning_effort": False,
            "thinking": "disabled",
        }

    model = create_model(model_name, parameter) if isinstance(model_name, str) else model_name

    return Agent(
        model,
        toolsets=list(toolsets) if toolsets is not None else None,
        capabilities=capabilities,
        instructions=instructions or "",
    )
