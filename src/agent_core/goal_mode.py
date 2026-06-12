from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from agent_core.input_messages import UserMessage
from prompt import load_prompt


class GoalSignal(str, Enum):
    CONTINUE = "CONTINUE"
    DONE = "DONE"


GOAL_MARKER_RE = re.compile(
    r"<!--\s*REDLOTUS_GOAL\s*:\s*(CONTINUE|DONE)\s*-->",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class GoalParseResult:
    signal: GoalSignal
    cleaned_text: str
    marker_count: int

    @property
    def missing_marker(self) -> bool:
        return self.marker_count == 0


@dataclass(frozen=True)
class GoalLoopResult:
    iterations: int
    output: str


def parse_goal_output(text: str) -> GoalParseResult:
    """Strip goal-mode sentinel markers and return the effective signal.

    If no marker is present, goal mode treats the turn as CONTINUE so the next
    prompt can remind the model to add an explicit status marker.
    """
    body = text or ""
    matches = list(GOAL_MARKER_RE.finditer(body))
    cleaned = GOAL_MARKER_RE.sub("", body).strip()
    if not matches:
        return GoalParseResult(GoalSignal.CONTINUE, cleaned, 0)
    signal = GoalSignal(matches[-1].group(1).upper())
    return GoalParseResult(signal, cleaned, len(matches))


def build_goal_iteration_prompt(
    *,
    original_goal: str,
    iteration: int,
    previous_output: str = "",
    user_updates: Sequence[str] = (),
    missing_marker_reminder: bool = False,
) -> str:
    updates = "\n".join(
        f"{index}. {item}" for index, item in enumerate(user_updates, 1)
    )
    template = load_prompt("goal_iteration.md")
    return template.format(
        original_goal=original_goal.strip(),
        iteration=iteration,
        previous_output=previous_output.strip(),
        user_updates=updates,
        missing_marker_reminder=str(bool(missing_marker_reminder)).lower(),
    )


async def run_goal_loop(
    system: Any,
    *,
    message: UserMessage,
    history: Any,
    turn_id: str | None,
    conversation_log_hint: str,
    take_queued_inputs: Callable[[], list[str]],
    set_iteration: Callable[[int], None] | None = None,
) -> GoalLoopResult:
    original_goal = message.text or ""
    previous_output = ""
    pending_updates: list[str] = []
    missing_marker = False
    iteration = 0

    while True:
        iteration += 1
        if set_iteration is not None:
            set_iteration(iteration)

        pending_updates.extend(take_queued_inputs())
        prompt_text = build_goal_iteration_prompt(
            original_goal=original_goal,
            iteration=iteration,
            previous_output=previous_output,
            user_updates=pending_updates,
            missing_marker_reminder=missing_marker,
        )
        pending_updates = []

        parse_result: GoalParseResult | None = None

        def output_transform(raw_output: str) -> str:
            nonlocal parse_result
            parse_result = parse_goal_output(raw_output)
            return parse_result.cleaned_text

        prompt_message = UserMessage(
            text=prompt_text,
            attachments=message.attachments if iteration == 1 else [],
        )
        _history, output = await system.run_agent_system(
            prompt_message,
            history,
            conversation_log_hint=conversation_log_hint,
            conversation_log_extra={
                "turn_id": turn_id,
                "goal_mode": True,
                "goal_iteration": iteration,
            },
            turn_id=turn_id,
            output_transform=output_transform,
        )

        parsed = parse_result or parse_goal_output(output)
        previous_output = output
        missing_marker = parsed.missing_marker

        pending_updates.extend(take_queued_inputs())
        if parsed.signal == GoalSignal.DONE and not pending_updates:
            return GoalLoopResult(iterations=iteration, output=output)
