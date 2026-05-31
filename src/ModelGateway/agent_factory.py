from __future__ import annotations

from typing import Any

from pydantic_ai import Agent, FunctionToolset

import logger
from app_config import get_agent_run_policy
from ModelGateway.model_factory import create_model


def create_function_toolset(
    tools: list,
    *,
    toolset_id: str = "default",
    instructions: str | None = None,
) -> FunctionToolset:
    wrapped_tools = logger.wrap_tools_for_user_notify(
        list(tools), policy=get_agent_run_policy()
    )
    return FunctionToolset(
        wrapped_tools,
        id=toolset_id,
        instructions=instructions,
    )


def create_agent(
    model_name: Any,
    parameter: dict | None,
    tools: list | None = None,
    instructions: str | None = None,
    *,
    toolsets: list | None = None,
    system_prompt: str | None = None,
):
    if parameter is None:
        parameter = {
            "temperature": 1.0,
            "max_tokens": 32768,
            "reasoning_effort": False,
            "thinking": "disabled",
        }

    model = create_model(model_name, parameter) if isinstance(model_name, str) else model_name
    if instructions is None:
        instructions = system_prompt or ""

    agent_toolsets = list(toolsets) if toolsets is not None else None
    if agent_toolsets is None and tools:
        agent_toolsets = [create_function_toolset(list(tools), toolset_id="default")]

    return Agent(
        model,
        toolsets=agent_toolsets,
        instructions=instructions,
    )
