from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import logger
from persist_utils import safe_name
from ModelGateway.ModelChecker import (
    estimate_history_tokens_async,
    get_effective_max_context_async,
    prewarm_effective_max_contexts_by_role_async,
)
from agent_core.input_messages import user_message_from_cli_input
from app_config import get_context_config
from cli.file_ref import augment_text_with_file_refs, load_file_refs
from cli.output import ContextUsageItem, clear_context_usage, set_context_usage
from cli.render import print_repl_welcome, print_success, print_warning
from cli.repl import InteractiveRepl
from cli_commands import handle_slash_command
from cli_ui import print_startup_logo
from tools.memory import ChatHistory

if TYPE_CHECKING:
    from agent_core.system import AgentSystem


@dataclass
class CliSessionState:
    history: ChatHistory
    is_first_input: bool = True


@dataclass
class QueuedCliInput:
    raw_input: str
    state: CliSessionState


class AgentCliController:
    """CLI/TUI orchestration for AgentSystem."""

    EXIT_COMMANDS = {"/exit", "/quit", "exit", "quit", "退出"}
    RESET_COMMANDS = {"/clear", "新任务"}  # 备用：当前未在本类内引用
    BUSY_SAFE_COMMANDS = {"/stop", "/status", "/cancel", "/help", "/trace", "/tasks", "/pwd", "/config", "/context", "/skills"}

    def __init__(self, system: "AgentSystem") -> None:
        self.system = system
        self._queued_inputs: list[QueuedCliInput] = []
        self._queue_drain_task: asyncio.Task[None] | None = None
        self._queue_draining = False

    @property
    def queued_input_count(self) -> int:
        return len(self._queued_inputs)

    @property
    def has_queued_input(self) -> bool:
        return bool(self._queued_inputs)

    def new_session_state(self) -> CliSessionState:
        return CliSessionState(history=ChatHistory())

    async def reset_session(self, history: ChatHistory) -> None:
        system = self.system
        if system._session_key:
            await system.end_session_agents(system._session_key)
        system._task_manager.reset()
        system._toolkit.reset_task_directory()
        system.reset_manager_history()
        system._session_logs.reset()
        system._memory.reset_injection_snapshot()
        self._queued_inputs.clear()
        history.reset()
        clear_context_usage()

    async def prepare_session(self) -> None:
        print_startup_logo()
        print_repl_welcome()
        await prewarm_effective_max_contexts_by_role_async(reason="program startup")
        self.system._context_prewarmed = True

    def schedule_queue_drain(self) -> None:
        if self.system._current_turn is not None or not self._queued_inputs:
            return
        if self._queue_drain_task is not None and not self._queue_drain_task.done():
            return
        self._queue_drain_task = asyncio.create_task(self._drain_input_queue())

    def _queue_input(self, raw_input: str, state: CliSessionState) -> None:
        self._queued_inputs.append(QueuedCliInput(raw_input=raw_input, state=state))
        print_success(
            f"输入已入队（{len(self._queued_inputs)} 项待处理）。将在当前回合结束后运行。"
        )

    @staticmethod
    def _merge_queued_inputs(raw_inputs: list[str]) -> str:
        if len(raw_inputs) == 1:
            return raw_inputs[0]
        lines = [
            "User sent multiple inputs while the previous response was running. "
            "Process them in order as one request:",
            "",
        ]
        lines.extend(f"{index}. {text}" for index, text in enumerate(raw_inputs, 1))
        return "\n".join(lines)

    async def _drain_input_queue(self) -> None:
        if self._queue_draining or self.system._current_turn is not None:
            return
        if not self._queued_inputs:
            return

        self._queue_draining = True
        try:
            batch = list(self._queued_inputs)
            self._queued_inputs.clear()
            raw_input = self._merge_queued_inputs([item.raw_input for item in batch])
            print_success(f"Processing {len(batch)} queued input(s).")
            await self._start_user_turn_from_raw_input(
                raw_input,
                batch[0].state,
                wait_for_turn=False,
            )
        finally:
            self._queue_draining = False

    async def _publish_context_usage(self, history: ChatHistory) -> None:
        system = self.system
        has_any_history = bool(history.messages) or bool(system._manager_history.messages)
        if not has_any_history:
            clear_context_usage()
            return

        items: list[ContextUsageItem] = []
        roles_and_histories = [
            ("manager", "Manager", [system._manager_history]),
            ("coordinator", "Coordinator", [history]),
        ]
        for role, label, histories in roles_and_histories:
            ctx_cfg = get_context_config(role)
            chars_per_token = float(ctx_cfg["token_estimate_fallback_chars_per_token"])
            max_tokens = int(await get_effective_max_context_async(role=role))
            used_tokens = 0
            for role_history in histories:
                if role_history.messages:
                    used_tokens += int(
                        await estimate_history_tokens_async(
                            role_history.messages,
                            chars_per_token=chars_per_token,
                            role=role,
                        )
                    )
            percent = 0.0 if max_tokens <= 0 else min(100.0, used_tokens * 100.0 / max_tokens)
            items.append(ContextUsageItem(label, used_tokens, max_tokens, percent))
        set_context_usage(items)

    async def _handle_slash_command(self, raw_input: str, state: CliSessionState) -> str:
        system = self.system
        if system._current_turn is not None:
            command = raw_input.split()[0].lower()
            if command not in self.BUSY_SAFE_COMMANDS:
                print_warning(
                    "A turn is currently running. Use /stop first or wait for it to finish."
                )
                return "continue"

        await system._sync_skills_for_user_turn()
        _consumed, first_override = await handle_slash_command(
            raw_input,
            system._skills_manager,
            coordinator_history=state.history,
            manager_history=system._manager_history,
            reset_cli_session_for_load=lambda: self.reset_session(state.history),
            bind_loaded_snapshot_for_save=lambda agent, path, meta: (
                system._session_logs.bind_loaded_snapshot(agent, path, meta)
            ),
            system=system,
        )
        if first_override is not None:
            state.is_first_input = first_override
        return "continue"

    async def _start_user_turn_from_raw_input(
        self,
        raw_input: str,
        state: CliSessionState,
        *,
        wait_for_turn: bool,
    ) -> str:
        system = self.system
        history = state.history
        await self._publish_context_usage(history)

        if system._current_turn is not None:
            self._queue_input(raw_input, state)
            return "continue"

        file_refs = await asyncio.to_thread(load_file_refs, raw_input)
        for ref in file_refs:
            if not ref.ok:
                print_warning(f"@{ref.path}: {ref.error}")
            elif ref.truncated:
                print_success(f"@{ref.path} (truncated)")
            else:
                print_success(f"@{ref.path}")

        augmented = augment_text_with_file_refs(raw_input, file_refs)
        message = await asyncio.to_thread(user_message_from_cli_input, augmented)
        if message.attachments:
            logger.info("Detected %s media attachment(s)", len(message.attachments))

        if state.is_first_input:
            task_name = (message.text or "task")[:30].replace(" ", "_")
            logger.setup_task_logger(task_name)
            system._toolkit.set_task_directory(task_name)
            safe_session_key = safe_name(task_name, max_len=50, fallback="task")
            await system.bind_session(safe_session_key)
            state.is_first_input = False

        turn_task = system._start_user_turn(message, history)
        if turn_task is None:
            self._queue_input(raw_input, state)
            return "continue"

        if wait_for_turn:
            try:
                await turn_task
            except KeyboardInterrupt:
                print_warning(await system.cancel_current_turn())
            except asyncio.CancelledError:
                pass
        return "continue"

    async def process_line(
        self,
        raw_input: str,
        state: CliSessionState,
        *,
        wait_for_turn: bool,
    ) -> str:
        raw_input = raw_input.strip()
        if not raw_input:
            return "continue"

        command = raw_input.lower()
        if command in self.EXIT_COMMANDS:
            if self.system._current_turn is not None:
                print_warning("A turn is running. Use /stop before exiting.")
                return "continue"
            print_success("Bye.")
            return "break"

        if command == "/clear" or "新任务" in raw_input:
            if self.system._current_turn is not None:
                print_warning("A turn is running. Use /stop before clearing the session.")
                return "continue"
            await self.reset_session(state.history)
            state.is_first_input = True
            return "continue"

        if raw_input.startswith("/"):
            await self._publish_context_usage(state.history)
            return await self._handle_slash_command(raw_input, state)

        if self.system._current_turn is not None or self._queue_draining:
            self._queue_input(raw_input, state)
            return "continue"

        return await self._start_user_turn_from_raw_input(
            raw_input,
            state,
            wait_for_turn=wait_for_turn,
        )

    async def run_interactive(self, *, stop_event: asyncio.Event | None = None) -> None:
        if os.environ.get("REDLOTUS_LEGACY_CLI", "").strip() not in (
            "1",
            "true",
            "TRUE",
            "yes",
        ):
            from cli.tui import run_textual_tui

            await run_textual_tui(self.system, stop_event=stop_event)
            return

        await self.prepare_session()
        state = self.new_session_state()

        async def on_cli_keyboard_interrupt() -> None:
            if self.system._current_turn is not None:
                print_warning(await self.system.cancel_current_turn())
                return
            raise KeyboardInterrupt

        repl = InteractiveRepl(on_interrupt_during_handler=on_cli_keyboard_interrupt)

        async def process_one_line(raw_input: str) -> str:
            return await self.process_line(raw_input, state, wait_for_turn=True)

        try:
            await repl.run(process_one_line, stop_event=stop_event)
        finally:
            await self.system.shutdown()
