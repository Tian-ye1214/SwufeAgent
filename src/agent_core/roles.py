from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from typing import Any

from ModelGateway.agent_factory import create_agent, create_function_toolset
from app_config import get_model_and_params
from prompt import get_coordinator_system_prompt, get_manager_system_prompt
from skills.SkillsManager import SkillsManager


def _toolset(toolset_id: str, tools: Sequence[Any], *, instructions: str | None = None):
    if not tools:
        return None
    return create_function_toolset(
        list(tools),
        toolset_id=toolset_id,
        instructions=instructions,
    )


async def _create_role_agent(
    role: str,
    skills_manager: SkillsManager,
    memory_injection: str,
    toolset_specs: list[tuple[str, Sequence[Any], str | None]],
    *,
    prompt_fn: Callable[[SkillsManager, str], str],
):
    model_name, model_params = get_model_and_params(role)
    instructions = await asyncio.to_thread(prompt_fn, skills_manager, memory_injection)
    toolsets = [
        ts
        for ts in (
            _toolset(toolset_id, tools, instructions=instructions_text)
            for toolset_id, tools, instructions_text in toolset_specs
        )
        if ts is not None
    ]
    return create_agent(
        model_name,
        model_params,
        instructions=instructions,
        toolsets=toolsets,
    )


async def create_manager_agent(
    skills_manager: SkillsManager,
    memory_injection: str,
    manager_tools: Sequence[Any],
):
    return await _create_role_agent(
        "manager",
        skills_manager,
        memory_injection,
        [
            (
                "manager_planning",
                manager_tools,
                "Tools for planning, task list management, and asking the user.",
            )
        ],
        prompt_fn=get_manager_system_prompt,
    )


async def create_coordinator_agent(
    skills_manager: SkillsManager,
    memory_injection: str,
    routing_tools: Sequence[Any],
    worker_tools: Sequence[Any],
):
    return await _create_role_agent(
        "coordinator",
        skills_manager,
        memory_injection,
        [
            (
                "coordinator_routing",
                routing_tools,
                "Tools for delegating work to manager or worker agents.",
            ),
            (
                "coordinator_worker_tools",
                worker_tools,
                "Direct tools available when the coordinator can answer or act without delegation.",
            ),
        ],
        prompt_fn=get_coordinator_system_prompt,
    )
