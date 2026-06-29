from redlotus.config import app_config

app_config.load_config()

from redlotus.prompt import load_prompt
from redlotus.tools.BasicTools import BasicToolkit
from redlotus.tools.ManagementTools import TaskManager
from redlotus.tools.WorkerOrchestrator import WorkerOrchestrator
from redlotus.tools.memory import (
    ChatHistory,
)
from redlotus.tools.memory.chat_history import messages_safe_for_new_prompt
from redlotus.agent_core.input_messages import (
    UserMessage,
    filter_messages_for_input_modalities,
)

from redlotus.tools.conversation_log import SessionConversationLogs, drain_pending_saves
from redlotus.ModelGateway.ModelChecker import (
    prewarm_effective_max_contexts_by_role_async,
    maybe_auto_compress_async,
)
from redlotus.config.app_config import get_agent_usage_limits, role_supports_input_modality
from redlotus.cli.output import supports_model_stream
from redlotus.cli.render import (
    TextEventStreamHandler,
    consume_stream_markdown,
    clear_model_stream,
    finish_model_stream,
    model_generating_indicator,
    print_phase,
    print_warning,
    show_model_output,
)
from redlotus.cli.cli_ui import format_user_log_text
from redlotus.infra import logger
import traceback
import time
import asyncio
import uuid
from collections.abc import Callable
from dataclasses import replace
from typing import Any, Coroutine, Tuple

from pydantic_ai.exceptions import ModelHTTPError

from redlotus.skills.SkillsManager import SkillsManager
from redlotus.agent_core.goal_mode import run_goal_loop

from redlotus.runtime.lifecycle import (
    AgentRegistry,
    LifecycleHooks,
    register_default_lifecycle_logging,
    run_agent_iter_with_lifecycle,
    run_agent_stream_with_lifecycle,
    run_agent_with_lifecycle,
)
from redlotus.agent_core.cli_controller import AgentCliController, CliSessionState
from redlotus.agent_core.memory_runtime import MemoryRuntime
from redlotus.agent_core.roles import create_coordinator_agent, create_manager_agent


def _make_coordinator_stream_handler() -> TextEventStreamHandler | None:
    if not supports_model_stream():
        return None
    return TextEventStreamHandler(title="Coordinator")


def _role_input_modalities(role: str) -> set[str]:
    return {"text", "image"} if role_supports_input_modality(role, "image") else {"text"}


def _history_for_role(messages: list, role: str) -> list:
    return filter_messages_for_input_modalities(
        messages_safe_for_new_prompt(messages),
        _role_input_modalities(role),
    )


def _prompt_for_role(text: str, attachments: list, role: str):
    return [text, *attachments] if attachments and "image" in _role_input_modalities(role) else text


def _messages_with_replaced_output(result: Any, output: str) -> list[Any]:
    """Return result messages with the last assistant text part replaced."""
    messages = list(result.all_messages())
    for message_index in range(len(messages) - 1, -1, -1):
        message = messages[message_index]
        parts = getattr(message, "parts", None)
        if not parts:
            continue
        new_parts = list(parts)
        for part_index in range(len(new_parts) - 1, -1, -1):
            part = new_parts[part_index]
            if getattr(part, "part_kind", None) != "text":
                continue
            new_parts[part_index] = replace(part, content=output)
            messages[message_index] = replace(message, parts=new_parts)
            return messages
    return messages


def _coordinator_tool_report(primary: str, fallback: str) -> str:
    """Prefer primary tool output; fall back when delegation produced no summary text."""
    text = (primary or "").strip()
    if text:
        return text
    return (fallback or "").strip()


async def _maybe_auto_compress(
    history: ChatHistory,
    *,
    role: str,
    task_state: str | None,
    log_suffix: str = "",
) -> None:
    try:
        if await maybe_auto_compress_async(
            history,
            role=role,
            task_state=task_state,
        ):
            suffix = f"（{log_suffix}）" if log_suffix else ""
            logger.info("已自动压缩 %s 上下文%s（达到配置阈值）", role.capitalize(), suffix)
    except Exception as ce:
        logger.warning("%s 自动压缩失败: %s", role.capitalize(), ce)


class AgentSystem:
    """Agent 任务协调系统，管理 Manager/Coordinator 的对话历史与执行流程。"""
    def __init__(self):
        self._registry = AgentRegistry()
        self._hooks = LifecycleHooks()
        register_default_lifecycle_logging(self._hooks)
        self._background_tasks: set[asyncio.Task] = set()
        self._shutdown_done = False
        self._skills_manager = SkillsManager()
        self._manager_history = ChatHistory()
        self._current_attachments: list = []
        self._memory = MemoryRuntime()
        self._short_term_memory = self._memory.short_term
        self._toolkit = BasicToolkit(
            self._skills_manager,
            extra_worker_tools=self._memory.worker_tools,
        )
        self._task_manager = TaskManager()
        self._orchestrator = WorkerOrchestrator(
            self._toolkit,
            self._task_manager,
            memory_injection_getter=self._injection_for_session,
            registry=self._registry,
            hooks=self._hooks,
            short_term_memory=self._short_term_memory,
        )
        self._session_logs = SessionConversationLogs(
            on_activate=self._orchestrator.set_conversation_session,
            on_reset=self._orchestrator.clear_conversation_session,
        )
        self._context_prewarmed = False
        self._current_turn: dict[str, Any] | None = None
        self._cli_turn_id: str | None = None
        self._session_key: str | None = None
        self._cli_controller = AgentCliController(self)

    async def bind_session(self, session_key: str) -> None:
        self._session_key = session_key
        await self._registry.ensure_agent(session_key, "coordinator")
        await self._registry.ensure_agent(session_key, "manager")
        self._orchestrator.set_session_key(session_key)

    async def end_session_agents(self, session_key: str) -> None:
        await self._registry.cancel_session(session_key)
        await self._registry.remove_session(session_key)
        if self._session_key == session_key:
            self._session_key = None
            self._orchestrator.set_session_key(None)

    def _require_session_key(self) -> str:
        if not self._session_key:
            raise RuntimeError("会话未绑定，请先调用 bind_session")
        return self._session_key

    @property
    def registry(self) -> AgentRegistry:
        return self._registry

    @property
    def session_key(self) -> str | None:
        return self._session_key

    @property
    def has_current_turn(self) -> bool:
        return self._current_turn is not None

    @property
    def has_current_goal_turn(self) -> bool:
        return bool(self._current_turn and self._current_turn.get("mode") == "goal")

    @property
    def current_goal_iteration(self) -> int:
        if not self.has_current_goal_turn:
            return 0
        return int(self._current_turn.get("goal_iteration") or 0)

    def new_cli_session_state(self) -> CliSessionState:
        return self._cli_controller.new_session_state()

    async def _agent_id(self, role: str, suffix: str | None = None) -> str:
        sk = self._require_session_key()
        return await self._registry.ensure_agent(sk, role, suffix)

    def _spawn_background(self, coro: Coroutine[Any, Any, Any]) -> None:
        t = asyncio.create_task(coro)
        self._background_tasks.add(t)
        t.add_done_callback(self._background_tasks.discard)

    async def shutdown(self) -> None:
        if self._shutdown_done:
            return
        self._shutdown_done = True
        if self._session_key:
            await self.end_session_agents(self._session_key)
        await self._registry.cancel_all()
        pending = list(self._background_tasks)
        if pending:
            logger.info("[lifecycle] draining %s background task(s)", len(pending))
            try:
                await asyncio.wait_for(
                    asyncio.gather(*pending, return_exceptions=True), timeout=15.0
                )
            except asyncio.TimeoutError:
                logger.warning("[lifecycle] 后台任务超时未结束，取消之")
                for t in pending:
                    t.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
        await drain_pending_saves()
        await self._memory.close()
        self._toolkit.close()
        logger.info("[lifecycle] shutdown complete")

    def _on_turn_task_done(self, done_task: asyncio.Task) -> None:
        ct = self._current_turn
        if ct is not None and ct.get("task") is done_task:
            self._current_turn = None
        self._cli_controller.schedule_queue_drain()

    async def cancel_current_turn(self) -> str:
        ct = self._current_turn
        if ct is None:
            return "当前没有正在执行的用户任务。"
        t: asyncio.Task = ct["task"]
        turn_id = ct.get("turn_id")
        if turn_id:
            await self._registry.cancel_turn(turn_id)
        else:
            await self._registry.cancel_all()
        if not t.done():
            t.cancel()
        try:
            await asyncio.wait_for(asyncio.shield(t), timeout=5.0)
        except asyncio.TimeoutError:
            if not t.done():
                self._background_tasks.add(t)
                t.add_done_callback(self._background_tasks.discard)
                logger.warning("[lifecycle] 任务取消超时，转交后台收尾（门控保持关闭直至其结束）")
                return "已请求停止；任务仍在后台收尾，结束前暂不接受新任务。"
        except asyncio.CancelledError:
            pass
        self._current_turn = None
        return "已请求停止当前任务（本回合未完整写入对话历史）。"

    def _start_user_turn(self, message: UserMessage, history: ChatHistory) -> asyncio.Task[None] | None:
        if self._current_turn is not None:
            return None
        turn_id = uuid.uuid4().hex[:8]
        task = asyncio.create_task(self._run_user_turn(turn_id, message, history))
        self._current_turn = {
            "turn_id": turn_id,
            "task": task,
            "text": (message.text or "")[:200],
            "mode": "single",
        }
        task.add_done_callback(self._on_turn_task_done)
        return task

    def _start_goal_turn(
        self,
        message: UserMessage,
        history: ChatHistory,
        *,
        conversation_log_hint: str,
        take_queued_inputs: Callable[[], list[str]],
    ) -> asyncio.Task[None] | None:
        if self._current_turn is not None:
            return None
        turn_id = uuid.uuid4().hex[:8]
        task = asyncio.create_task(
            self._run_goal_turn(
                turn_id,
                message,
                history,
                conversation_log_hint=conversation_log_hint,
                take_queued_inputs=take_queued_inputs,
            )
        )
        self._current_turn = {
            "turn_id": turn_id,
            "task": task,
            "text": (message.text or ""),
            "mode": "goal",
            "goal_iteration": 0,
        }
        task.add_done_callback(self._on_turn_task_done)
        return task

    def _handle_turn_error(self, e: Exception) -> None:
        if isinstance(e, ModelHTTPError):
            body = e.body or {}
            code = body.get("code", "") if isinstance(body, dict) else ""
            if code == "data_inspection_failed":
                print_warning(
                    "模型内容安全审查拦截：您的输入或上下文中包含被判定为不当的内容。"
                    "请尝试换一种表达方式，或 /clear 清空上下文后重试。"
                )
            else:
                print_warning(f"模型请求错误 (HTTP {e.status_code}): {e}")
                logger.error("详细信息:\n%s", traceback.format_exc())
            return
        print_warning(f"未预期的系统错误: {e}")
        logger.error("详细信息:\n%s", traceback.format_exc())

    async def _run_user_turn(self, turn_id: str, message: UserMessage, history: ChatHistory) -> None:
        try:
            await self.run_agent_system(
                message,
                history,
                conversation_log_hint=(message.text or "")[:40],
                conversation_log_extra={"turn_id": turn_id},
                turn_id=turn_id,
            )
        except asyncio.CancelledError:
            logger.info("用户回合已取消 turn_id=%s", turn_id)
        except Exception as e:
            self._handle_turn_error(e)

    async def _run_goal_turn(
        self,
        turn_id: str,
        message: UserMessage,
        history: ChatHistory,
        *,
        conversation_log_hint: str,
        take_queued_inputs: Callable[[], list[str]],
    ) -> None:
        def set_iteration(iteration: int) -> None:
            ct = self._current_turn
            if ct is not None and ct.get("turn_id") == turn_id:
                ct["goal_iteration"] = iteration

        try:
            await run_goal_loop(
                self,
                message=message,
                history=history,
                turn_id=turn_id,
                conversation_log_hint=conversation_log_hint,
                take_queued_inputs=take_queued_inputs,
                set_iteration=set_iteration,
            )
        except asyncio.CancelledError:
            logger.info("目标模式回合已取消 turn_id=%s", turn_id)
        except Exception as e:
            self._handle_turn_error(e)

    async def _save_manager_result(
        self,
        *,
        phase: str,
        turn_id: str,
        result=None,
        log_suffix: str = "",
    ) -> None:
        if result is not None:
            self._manager_history.update(result)
        self._session_logs.for_agent("manager").save(
            self._manager_history.messages,
            extra={"kind": "manager", "phase": phase, "turn_id": turn_id},
        )
        await _maybe_auto_compress(
            self._manager_history,
            role="manager",
            task_state=self.structured_task_status(),
            log_suffix=log_suffix,
        )

    def _injection_for_session(self) -> str:
        return self._memory.injection_for_session()

    async def ask_user(self, question: str) -> str:
        return await self._toolkit.ask_user(question)

    async def wait_for_memory_quiescent(self, timeout: float = 15.0) -> bool:
        deadline = time.monotonic() + timeout

        def remaining() -> float:
            return max(0.0, deadline - time.monotonic())

        current = self._current_turn
        if current is not None:
            task = current.get("task")
            if task is not None and not task.done():
                left = remaining()
                if left <= 0:
                    return False
                try:
                    await asyncio.wait_for(asyncio.shield(task), timeout=left)
                except asyncio.TimeoutError:
                    return False
                except asyncio.CancelledError:
                    pass

        pending = [t for t in self._background_tasks if not t.done()]
        if pending:
            left = remaining()
            if left <= 0:
                return False
            try:
                await asyncio.wait_for(
                    asyncio.gather(*pending, return_exceptions=True),
                    timeout=left,
                )
            except asyncio.TimeoutError:
                return False

        left = remaining()
        if left <= 0:
            return False
        if not await drain_pending_saves(timeout=left):
            return False

        left = remaining()
        if left <= 0:
            return False
        return await self._memory.short_term.wait_idle(timeout=left)

    async def long_term_memory_snapshot(self) -> dict[str, dict[str, Any]]:
        return await self._memory.long_term_snapshot()

    async def short_term_memory_snapshot(self) -> dict[str, Any]:
        return await self._memory.short_term_snapshot()

    async def clear_long_term_memory(self) -> None:
        await self._memory.clear_long_term()

    async def clear_short_term_memory(self) -> None:
        await self._memory.clear_short_term()

    async def _sync_skills_for_user_turn(self) -> None:
        """每次用户输入：在同一实例上重新扫描 skills（静默），避免磁盘 I/O 阻塞事件循环。"""
        await asyncio.to_thread(self._skills_manager.refresh)

    def set_ask_user_handler(self, handler):
        self._toolkit.set_ask_user_handler(handler)

    @property
    def review_store(self):
        return self._toolkit.review_store

    def set_task_directory(self, task_name: str):
        self._toolkit.set_task_directory(task_name)

    def _fallback_task_name(self, user_text: str) -> str:
        """LLM 标题生成失败时的降级方案：截断用户输入。"""
        task_name = (user_text or "task")[:30].replace(" ", "_")
        return task_name.strip("_") or "task"

    async def generate_task_title(self, user_text: str) -> str:
        """用 LLM（worker 模型，thinking=disabled）生成简洁的任务目录标题。"""
        try:
            from redlotus.config.app_config import get_model_and_params, get_agent_usage_limits
            from redlotus.ModelGateway.agent_factory import create_agent

            name, params = get_model_and_params("worker")
            params = dict(params)
            params["thinking"] = "disabled"
            params["reasoning_effort"] = False
            params["max_tokens"] = 64

            agent = create_agent(name, params, instructions="用一小段话总结以下任务的标题，直接输出标题不要加任何前缀、标点符号或解释")
            result = await agent.run(user_text, usage_limits=get_agent_usage_limits())

            title = (result.output or "").strip()
            if not title:
                return self._fallback_task_name(user_text)

            title = (
                title.replace("\n", " ")
                .replace("/", "_")
                .replace("\\", "_")
                .replace("\r", "")
            )
            title = title.strip("。，. ,\"'\"\"' '''《》「」【】[]()（）")
            return title or self._fallback_task_name(user_text)
        except Exception as e:
            logger.warning("LLM 标题生成失败，使用 fallback: %s", e)
            return self._fallback_task_name(user_text)

    def reset_manager_history(self):
        self._manager_history.reset()

    def structured_task_status(self) -> str:
        return self._task_manager.structured_status()

    async def _after_coordinator_turn(self, history: ChatHistory) -> None:
        """Schedule memory consolidation after a Coordinator turn."""
        self._memory.schedule_after_coordinator_turn(
            history, self._session_logs, self._spawn_background
        )

    async def execute_task_with_manager(
        self, user_input: str, continue_from_previous: bool = False
    ) -> str:
        """
        Execute complex, multi-step tasks using Manager Agent with intelligent task orchestration.

        Description:
            This function handles sophisticated tasks that require planning, coordination, and
            multi-step execution. The Manager Agent analyzes the user's request, breaks it down
            into a structured task list (Todo List), and orchestrates the execution of each subtask
            using Worker Agents. It provides comprehensive project management capabilities including
            automatic retry on failure, progress tracking, and final report generation. Supports
            iterative refinement based on user feedback.

        Parameters:
            user_input (str):
                The user's request or requirement description. Should be comprehensive enough
                for the Manager Agent to understand the full scope of work needed.
                For new tasks: Complete description of what needs to be accomplished
                For continued tasks: Additional requirements or feedback on previous results

            continue_from_previous (bool, optional):
                Indicates whether this is a continuation of a previous task execution.
                - False (default): Start a new task with fresh task list
                - True: Continue from previous execution, preserving completed tasks and
                        adding new tasks based on user feedback

        Returns:
            str: A comprehensive response to the user containing:
                - Direct answer to the user's original question
                - Key information extracted from task execution results
                - Summary of what was accomplished
                - Explanation of any failures (if applicable)

                The response is conversational and user-focused, avoiding technical
                implementation details like "task completed" or "file created" unless
                directly relevant to the user's question.
        """
        self._session_logs.ensure(user_input or "task")
        logger.info("[用户]\n%s", user_input)
        tid = self._cli_turn_id

        manager_tools = [
            self._task_manager.create_todo_list,
            self._task_manager.get_todo_list,
            self._toolkit.ask_user,
        ]
        mem_inj = self._injection_for_session()
        manager_agent = await create_manager_agent(
            self._skills_manager,
            mem_inj,
            manager_tools,
        )
        attachments = self._current_attachments

        if not continue_from_previous:
            print_phase("第一阶段: Manager 规划任务列表")
            tmpl = await asyncio.to_thread(load_prompt, "manager_planning_new.md")
            planning_text = tmpl.format(user_input=user_input)
        else:
            print_phase("第一阶段: 基于用户反馈调整任务")
            current_todo = self._task_manager.get_todo_list()
            tmpl = await asyncio.to_thread(load_prompt, "manager_planning_continue.md")
            planning_text = tmpl.format(
                user_input=user_input,
                current_todo=current_todo
            )

        planning_prompt = _prompt_for_role(planning_text, attachments, "manager")
        manager_aid = await self._agent_id("manager")
        manager_log = self._session_logs.for_agent("manager")
        planning_extra = {"kind": "manager", "phase": "planning", "turn_id": tid}

        async def _save_planning_node(run: Any) -> None:
            self._manager_history.set_messages(list(run.all_messages()))
            manager_log.save(self._manager_history.messages, extra=planning_extra)

        with model_generating_indicator():
            result = await run_agent_iter_with_lifecycle(
                agent=manager_agent,
                prompt=planning_prompt,
                agent_id=manager_aid,
                registry=self._registry,
                hooks=self._hooks,
                turn_id=tid,
                message_history=_history_for_role(self._manager_history.messages, "manager"),
                usage_limits=get_agent_usage_limits(),
                on_node=_save_planning_node,
            )
        await self._save_manager_result(phase="planning", turn_id=tid, result=result)
        show_model_output(result.output, title="Manager 规划")

        print_phase("第二阶段: 多Worker并行执行任务")

        final_summary = await self._orchestrator.execute_all_tasks_parallel(
            user_input, attachments=attachments, turn_id=tid
        )
        show_model_output(final_summary, title="任务汇总", markdown=False)

        print_phase("第三阶段: 生成最终报告")

        summary_tmpl = await asyncio.to_thread(load_prompt, "manager_summary.md")
        summary_text = summary_tmpl.format(
            user_input=user_input,
            final_summary=final_summary
        )
        summary_prompt = _prompt_for_role(summary_text, attachments, "manager")
        try:
            final_text = await run_agent_stream_with_lifecycle(
                agent=manager_agent,
                agent_id=manager_aid,
                registry=self._registry,
                hooks=self._hooks,
                turn_id=tid,
                message_history=_history_for_role(self._manager_history.messages, "manager"),
                usage_limits=get_agent_usage_limits(),
                prompt=summary_prompt,
                consumer=lambda s: consume_stream_markdown(
                    s, self._manager_history, title="最终报告"
                ),
            )
            await self._save_manager_result(phase="summary", turn_id=tid, log_suffix="summary 后")
            return _coordinator_tool_report(final_text, final_summary)
        except Exception as e:
            logger.warning(f"流式输出回退到普通模式: {e}")
            try:
                final_result = await run_agent_with_lifecycle(
                    agent=manager_agent,
                    prompt=summary_prompt,
                    agent_id=manager_aid,
                    registry=self._registry,
                    hooks=self._hooks,
                    turn_id=tid,
                    message_history=_history_for_role(self._manager_history.messages, "manager"),
                    usage_limits=get_agent_usage_limits(),
                )
                await self._save_manager_result(phase="summary", turn_id=tid, result=final_result, log_suffix="summary 后")
                show_model_output(final_result.output, title="最终报告")
                return _coordinator_tool_report(final_result.output, final_summary)
            except Exception:
                show_model_output(final_summary, title="任务汇总", markdown=False)
                return final_summary

    async def _execute_task_with_worker(
        self, task_description: str, user_goal: str = "", retry_info: str = ""
    ) -> Tuple[bool, str]:
        """
        Execute a single simple task using Worker Agent.

        Description:
            This function creates and runs a Worker Agent to execute straightforward, well-defined tasks
            that don't require complex planning or multi-step coordination. The Worker Agent has access
            to basic operational tools (search, file operations, web browsing, etc.) and executes tasks
            independently. It includes retry mechanism support and automatic success/failure detection
            based on the agent's output format.

        Parameters:
            task_description (str):
                A clear and specific description of the task to be executed. Should contain enough
                detail for the Worker Agent to understand and complete the task independently.

            user_goal (str, optional):
                The user's ultimate objective or broader context for this task. This helps the
                Worker Agent understand the bigger picture and make better decisions.
                Default: "" (empty string)

            retry_info (str, optional):
                Detailed information about previous failed attempts to execute this task. Used when
                retrying a task after failure, providing context about what went wrong and helping
                the agent avoid repeating the same mistakes.
                Default: "" (empty string)

        Returns:
            Tuple[bool, str]: A tuple containing:
                - success (bool): Whether the task completed successfully
                - result (str): The output message from the Worker Agent
        """
        return await self._orchestrator.execute_task_with_worker(
            task_description,
            user_goal,
            retry_info,
            attachments=self._current_attachments,
            turn_id=self._cli_turn_id,
        )

    async def run_agent_system(
        self,
        message: "UserMessage",
        history: "ChatHistory",
        *,
        conversation_log_hint: str = "",
        conversation_log_extra: dict | None = None,
        turn_id: str | None = None,
        output_transform: Callable[[str], str] | None = None,
    ) -> tuple["ChatHistory", str]:
        """
        任务协调系统入口，负责判断任务复杂度并调用相应的执行器。

        Parameters:
            message: 用户输入（UserMessage 实例）
            history: 对话历史（ChatHistory 实例）

        Returns:
            tuple[ChatHistory, str]: (更新后的对话历史, Agent 输出)
        """
        self._current_attachments = message.attachments

        prev_cli_turn = self._cli_turn_id
        self._cli_turn_id = turn_id
        try:
            return await self._run_agent_system_impl(
                message,
                history,
                conversation_log_hint=conversation_log_hint,
                conversation_log_extra=conversation_log_extra,
                turn_id=turn_id,
                output_transform=output_transform,
            )
        finally:
            self._cli_turn_id = prev_cli_turn

    async def _run_agent_system_impl(
        self,
        message: UserMessage,
        history: "ChatHistory",
        *,
        conversation_log_hint: str,
        conversation_log_extra: dict | None,
        turn_id: str | None,
        output_transform: Callable[[str], str] | None,
    ) -> tuple["ChatHistory", str]:
        if not self._context_prewarmed:
            await prewarm_effective_max_contexts_by_role_async(
                reason="非 CLI 首条任务（预取三角色）"
            )
            self._context_prewarmed = True

        await self._sync_skills_for_user_turn()

        self._session_logs.ensure(conversation_log_hint or message.text or "")
        logger.info_file_only("[用户]\n%s", format_user_log_text(message))

        routing_tools = [
            self.execute_task_with_manager,
            self._execute_task_with_worker,
        ]
        mem_inj = self._injection_for_session()
        agent = await create_coordinator_agent(
            self._skills_manager,
            mem_inj,
            routing_tools,
            self._toolkit.worker_tools(
                include_browser=True,
                include_multimodal=role_supports_input_modality("coordinator", "image"),
            ),
        )

        if self._session_key is None:
            sk = conversation_log_hint or (message.text or "")[:40] or "session"
            await self.bind_session(sk)

        start_time = time.time()
        coord_aid = await self._agent_id("coordinator")
        stream_handler = None if output_transform is not None else _make_coordinator_stream_handler()
        coord_log = self._session_logs.for_agent("coordinator")
        extra: dict = {"kind": "coordinator"}
        if conversation_log_extra:
            extra.update(conversation_log_extra)

        async def _save_coordinator_node(run: Any) -> None:
            history.set_messages(list(run.all_messages()))
            coord_log.save(history.messages, extra=extra)

        try:
            result = await run_agent_iter_with_lifecycle(
                agent=agent,
                prompt=message.to_prompt(
                    include_attachments=role_supports_input_modality("coordinator", "image")
                ),
                agent_id=coord_aid,
                registry=self._registry,
                hooks=self._hooks,
                turn_id=turn_id,
                message_history=_history_for_role(history.messages, "coordinator"),
                usage_limits=get_agent_usage_limits(),
                event_stream_handler=stream_handler,
                on_node=_save_coordinator_node,
            )
        except BaseException:
            clear_model_stream()
            raise
        raw_output = str(result.output or "")
        output = output_transform(raw_output) if output_transform is not None else raw_output
        if stream_handler is not None:
            finish_model_stream(output, title="Coordinator")
        else:
            show_model_output(output, title="Coordinator")
        if output != raw_output:
            history.set_messages(_messages_with_replaced_output(result, output))
        else:
            history.update(result)
        elapsed = time.time() - start_time

        logger.debug("run_agent_system 完成，耗时 %.2f 秒", elapsed)
        coord_log.save(history.messages, extra=extra)
        await _maybe_auto_compress(
            history,
            role="coordinator",
            task_state=self.structured_task_status(),
        )
        await self._after_coordinator_turn(history)
        return history, output

    async def prepare_cli_session(self) -> None:
        await self._cli_controller.prepare_session()

    async def process_cli_line(
        self,
        raw_input: str,
        state: CliSessionState,
        *,
        wait_for_turn: bool,
        goal_mode: bool = False,
    ) -> str:
        return await self._cli_controller.process_line(
            raw_input, state, wait_for_turn=wait_for_turn, goal_mode=goal_mode
        )

    async def run_interactive(self, *, stop_event: asyncio.Event | None = None) -> None:
        """Run the interactive CLI/TUI."""
        await self._cli_controller.run_interactive(stop_event=stop_event)
