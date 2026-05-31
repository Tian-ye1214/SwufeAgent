from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from ModelGateway.agent_factory import create_agent, create_function_toolset
from app_config import get_model_and_params
from prompt import get_coordinator_system_prompt, get_manager_system_prompt, get_worker_system_prompt
from skills.SkillsManager import SkillsManager


def _toolset(toolset_id: str, tools: Sequence[Any], *, instructions: str | None = None):
    if not tools:
        return None
    return create_function_toolset(
        list(tools),
        toolset_id=toolset_id,
        instructions=instructions,
    )


async def create_manager_agent(
    skills_manager: SkillsManager,
    memory_injection: str,
    manager_tools: Sequence[Any],
):
    model_name, model_params = get_model_and_params("manager")
    instructions = await asyncio.to_thread(
        get_manager_system_prompt, skills_manager, memory_injection
    )
    toolsets = [
        ts
        for ts in [
            _toolset(
                "manager_planning",
                manager_tools,
                instructions="Tools for planning, task list management, and asking the user.",
            )
        ]
        if ts is not None
    ]
    return create_agent(
        model_name,
        model_params,
        instructions=instructions,
        toolsets=toolsets,
    )


async def create_coordinator_agent(
    skills_manager: SkillsManager,
    memory_injection: str,
    routing_tools: Sequence[Any],
    worker_tools: Sequence[Any],
):
    model_name, model_params = get_model_and_params("coordinator")
    instructions = await asyncio.to_thread(
        get_coordinator_system_prompt, skills_manager, memory_injection
    )
    toolsets = [
        ts
        for ts in [
            _toolset(
                "coordinator_routing",
                routing_tools,
                instructions="Tools for delegating work to manager or worker agents.",
            ),
            _toolset(
                "coordinator_worker_tools",
                worker_tools,
                instructions="Direct tools available when the coordinator can answer or act without delegation.",
            ),
        ]
        if ts is not None
    ]
    return create_agent(
        model_name,
        model_params,
        instructions=instructions,
        toolsets=toolsets,
    )


async def create_worker_agent(
    skills_manager: SkillsManager,
    memory_injection: str,
    worker_tools: Sequence[Any],
    *,
    parallel_addon: str = "",
):
    model_name, model_params = get_model_and_params("worker")

    def _instructions() -> str:
        return get_worker_system_prompt(skills_manager, memory_injection) + parallel_addon

    instructions = await asyncio.to_thread(_instructions)
    toolsets = [
        ts
        for ts in [
            _toolset(
                "worker_tools",
                worker_tools,
                instructions="Tools for file work, web/search/browser actions, skills, and memory lookup.",
            )
        ]
        if ts is not None
    ]
    return create_agent(
        model_name,
        model_params,
        instructions=instructions,
        toolsets=toolsets,
    )
