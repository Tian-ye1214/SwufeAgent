import app_config

app_config.load_config()

from prompt import get_manager_system_prompt, get_coordinator_system_prompt, load_prompt
from tools.BasicTools import BasicToolkit
from tools.ManagementTools import TaskManager
from tools.WorkerOrchestrator import WorkerOrchestrator
from tools.memory import (
    ChatHistory,
    LongTermMemory,
    ShortTermMemory,
    UserMessage,
    user_message_from_cli_input,
    user_message_from_text,
)

from tools.conversation_log import SessionConversationLogs
from ModelGateway.BasicFunction import create_agent
from ModelGateway.ModelChecker import (
    estimate_history_tokens_async,
    format_context_usage_line,
    get_effective_max_context_async,
    prewarm_effective_max_contexts_by_role_async,
    maybe_auto_compress_async,
)
from app_config import get_agent_usage_limits, get_context_config, get_model_and_params, settings
from cli_commands import handle_slash_command
from cli.repl import InteractiveRepl
from cli.file_ref import augment_text_with_file_refs, load_file_refs
from cli.render import (
    consume_stream_markdown,
    model_generating_indicator,
    print_panel,
    print_phase,
    print_repl_welcome,
    print_success,
    print_warning,
    show_model_output,
)
from cli_ui import format_user_log_text, print_startup_logo
import logger
import traceback
import time
import asyncio
import signal
import uuid
from typing import Any, Coroutine, Tuple

from pydantic_ai.exceptions import ModelHTTPError

from skills.SkillsManager import SkillsManager
from RAG.embedding_function import close_http_client

from lifecycle import (
    AgentRegistry,
    LifecycleHooks,
    register_default_lifecycle_logging,
    run_agent_stream_with_lifecycle,
    run_agent_with_lifecycle,
)


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
        self._long_term_memory = LongTermMemory()
        self._long_term_memory.refresh_from_disk_sync()
        self._short_term_memory = ShortTermMemory(
            settings()["short_term_memory"],
            log_root=logger.LOG_DIR,
        )
        self._toolkit = BasicToolkit(
            self._skills_manager,
            extra_worker_tools=[
                self._short_term_memory.query_short_term_memory,
                self._long_term_memory.add,
                self._long_term_memory.remove,
                self._long_term_memory.list_memory,
            ],
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
        self._memory_injection_snapshot: str | None = None
        self._current_turn: dict[str, Any] | None = None
        self._cli_turn_id: str | None = None
        self._session_key: str | None = None

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
            await asyncio.gather(*pending, return_exceptions=True)
        await self._short_term_memory.drain()
        await self._short_term_memory.close()
        await close_http_client()
        self._toolkit.close()
        logger.info("[lifecycle] shutdown complete")

    def _on_turn_task_done(self, done_task: asyncio.Task) -> None:
        ct = self._current_turn
        if ct is not None and ct.get("task") is done_task:
            self._current_turn = None

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
            await asyncio.wait_for(t, timeout=5.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass
        self._current_turn = None
        return "已请求停止当前任务（本回合未完整写入对话历史）。"

    def _start_user_turn(self, message: UserMessage, history: ChatHistory) -> asyncio.Task[None] | None:
        if self._current_turn is not None:
            return None
        turn_id = uuid.uuid4().hex[:8]
        task = asyncio.create_task(self._run_user_turn(turn_id, message, history))
        self._current_turn = {"turn_id": turn_id, "task": task, "text": (message.text or "")[:200]}
        task.add_done_callback(self._on_turn_task_done)
        return task

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
        except ModelHTTPError as e:
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
                self._task_manager.reset()
        except Exception as e:
            print_warning(f"未预期的系统错误: {e}")
            logger.error("详细信息:\n%s", traceback.format_exc())
            self._task_manager.reset()

    def _injection_for_session(self) -> str:
        if self._memory_injection_snapshot is None:
            self._memory_injection_snapshot = self._long_term_memory.get_injection()
        return self._memory_injection_snapshot

    async def _sync_skills_for_user_turn(self) -> None:
        """每次用户输入：在同一实例上重新扫描 skills（静默），避免磁盘 I/O 阻塞事件循环。"""
        await asyncio.to_thread(self._skills_manager.refresh)

    def set_ask_user_handler(self, handler):
        self._toolkit.set_ask_user_handler(handler)

    def set_task_directory(self, task_name: str):
        self._toolkit.set_task_directory(task_name)

    def reset_manager_history(self):
        self._manager_history.reset()

    def structured_task_status(self) -> str:
        return self._task_manager.structured_status()

    async def reset_cli_interactive_session(self, history: ChatHistory) -> None:
        """与输入「新任务」相同：清空任务、工作目录绑定、双方历史与会话落盘状态。"""
        if self._session_key:
            await self.end_session_agents(self._session_key)
        self._task_manager.reset()
        self._toolkit.reset_task_directory()
        self.reset_manager_history()
        self._session_logs.reset()
        self._memory_injection_snapshot = None
        history.reset()

    async def _after_coordinator_turn(self, history: ChatHistory) -> None:
        """每轮 Coordinator 结束后：后台长期记忆合并与短期记忆向量入库。"""
        msgs = list(history.messages)
        self._spawn_background(self._long_term_memory.consolidate_from_messages(msgs, silent=True))
        self._spawn_background(
            self._long_term_memory.consolidate_from_logs(logger.LOG_DIR, silent=True)
        )
        coord_log = self._session_logs.for_agent("coordinator")
        mp = coord_log.model_messages_path()
        sk = self._session_logs.session_key()
        if mp is not None and sk is not None:
            root = logger.LOG_DIR.resolve()
            try:
                log_key = mp.resolve().relative_to(root).as_posix()
            except ValueError:
                log_key = mp.resolve().as_posix()
            self._short_term_memory.schedule_ingest_after_turn(
                msgs, log_key, "coordinator", sk
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
        m_name, m_params = get_model_and_params("manager")
        mem_inj = self._injection_for_session()
        mgr_prompt = await asyncio.to_thread(
            get_manager_system_prompt, self._skills_manager, mem_inj
        )
        manager_agent = create_agent(m_name, m_params, manager_tools, mgr_prompt)
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

        planning_prompt = [planning_text, *attachments] if attachments else planning_text
        manager_aid = await self._agent_id("manager")
        with model_generating_indicator():
            result = await run_agent_with_lifecycle(
                agent=manager_agent,
                prompt=planning_prompt,
                agent_id=manager_aid,
                registry=self._registry,
                hooks=self._hooks,
                turn_id=tid,
                message_history=self._manager_history.messages,
                usage_limits=get_agent_usage_limits(),
            )
        self._manager_history.update(result)
        self._session_logs.for_agent("manager").save(
            self._manager_history.messages,
            extra={"kind": "manager", "phase": "planning", "turn_id": tid},
        )
        try:
            if await maybe_auto_compress_async(self._manager_history, role="manager"):
                logger.info("已自动压缩 Manager 上下文（达到配置阈值）")
        except Exception as ce:
            logger.warning("Manager 自动压缩失败: %s", ce)
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
        summary_prompt = [summary_text, *attachments] if attachments else summary_text
        try:
            final_text = await run_agent_stream_with_lifecycle(
                agent=manager_agent,
                agent_id=manager_aid,
                registry=self._registry,
                hooks=self._hooks,
                turn_id=tid,
                message_history=self._manager_history.messages,
                usage_limits=get_agent_usage_limits(),
                prompt=summary_prompt,
                consumer=lambda s: consume_stream_markdown(
                    s, self._manager_history, title="最终报告"
                ),
            )
            self._session_logs.for_agent("manager").save(
                self._manager_history.messages,
                extra={"kind": "manager", "phase": "summary", "turn_id": tid},
            )
            try:
                if await maybe_auto_compress_async(self._manager_history, role="manager"):
                    logger.info("已自动压缩 Manager 上下文（summary 后，达到配置阈值）")
            except Exception as ce:
                logger.warning("Manager 自动压缩失败: %s", ce)
            return final_text
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
                    message_history=self._manager_history.messages,
                    usage_limits=get_agent_usage_limits(),
                )
                self._manager_history.update(final_result)
                self._session_logs.for_agent("manager").save(
                    self._manager_history.messages,
                    extra={"kind": "manager", "phase": "summary", "turn_id": tid},
                )
                try:
                    if await maybe_auto_compress_async(self._manager_history, role="manager"):
                        logger.info("已自动压缩 Manager 上下文（summary 后，达到配置阈值）")
                except Exception as ce:
                    logger.warning("Manager 自动压缩失败: %s", ce)
                show_model_output(final_result.output, title="最终报告")
                return final_result.output
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
        message: "str | UserMessage",
        history: "ChatHistory | None" = None,
        *,
        conversation_log_hint: str = "",
        conversation_log_extra: dict | None = None,
        turn_id: str | None = None,
    ) -> tuple["ChatHistory", str]:
        """
        任务协调系统入口，负责判断任务复杂度并调用相应的执行器。

        Parameters:
            message: 用户输入（str 或 UserMessage 实例，str 会自动转换）
            history: 对话历史（ChatHistory 实例，传 None 则自动创建）

        Returns:
            tuple[ChatHistory, str]: (更新后的对话历史, Agent 输出)
        """
        if isinstance(message, str):
            message = user_message_from_text(message)
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
            )
        finally:
            self._cli_turn_id = prev_cli_turn

    async def _run_agent_system_impl(
        self,
        message: UserMessage,
        history: "ChatHistory | None",
        *,
        conversation_log_hint: str,
        conversation_log_extra: dict | None,
        turn_id: str | None,
    ) -> tuple["ChatHistory", str]:
        if not self._context_prewarmed:
            await prewarm_effective_max_contexts_by_role_async(
                reason="非 CLI 首条任务（预取三角色）"
            )
            self._context_prewarmed = True

        await self._sync_skills_for_user_turn()

        if history is None:
            history = ChatHistory()

        self._session_logs.ensure(conversation_log_hint or message.text or "")
        logger.info_file_only("[用户]\n%s", format_user_log_text(message))

        c_name, c_params = get_model_and_params("coordinator")
        coordinator_tools = [
            self.execute_task_with_manager,
            self._execute_task_with_worker,
            *self._toolkit.workers_tools,
        ]
        mem_inj = self._injection_for_session()
        coord_prompt = await asyncio.to_thread(
            get_coordinator_system_prompt, self._skills_manager, mem_inj
        )
        agent = create_agent(
            c_name,
            c_params,
            coordinator_tools,
            coord_prompt,
        )

        if self._session_key is None:
            sk = conversation_log_hint or (message.text or "")[:40] or "session"
            await self.bind_session(sk)

        start_time = time.time()
        coord_aid = await self._agent_id("coordinator")
        # Coordinator 必须走 agent.run()：run_stream/stream_text 在首个文本输出后
        # 即结束图执行，不会继续调用 execute_task_with_manager 等工具（pydantic-ai 文档说明）。
        result = await run_agent_with_lifecycle(
            agent=agent,
            prompt=message.to_prompt(),
            agent_id=coord_aid,
            registry=self._registry,
            hooks=self._hooks,
            turn_id=turn_id,
            message_history=history.messages,
            usage_limits=get_agent_usage_limits(),
        )
        output = result.output
        show_model_output(output, title="Coordinator")
        history.update(result)
        elapsed = time.time() - start_time

        logger.debug("run_agent_system 完成，耗时 %.2f 秒", elapsed)
        extra: dict = {"kind": "coordinator"}
        if conversation_log_extra:
            extra.update(conversation_log_extra)
        self._session_logs.for_agent("coordinator").save(history.messages, extra=extra)
        try:
            if await maybe_auto_compress_async(history, role="coordinator"):
                logger.info("已自动压缩 Coordinator 上下文（达到配置阈值）")
        except Exception as ce:
            logger.warning("Coordinator 自动压缩失败: %s", ce)
        await self._after_coordinator_turn(history)
        return history, output

    async def run_interactive(self, *, stop_event: asyncio.Event | None = None):
        """交互式命令行运行入口。支持在输入中包含图片/视频文件路径以发送多模态内容。"""
        print_startup_logo()
        print_repl_welcome()
        await prewarm_effective_max_contexts_by_role_async(reason="程序启动")
        self._context_prewarmed = True

        is_first_input = True
        history = ChatHistory()

        async def on_cli_keyboard_interrupt() -> None:
            if self._current_turn is not None:
                print_warning(await self.cancel_current_turn())
                return
            raise KeyboardInterrupt

        repl = InteractiveRepl(on_interrupt_during_handler=on_cli_keyboard_interrupt)

        async def process_one_line(raw_input: str) -> str:
            nonlocal history, is_first_input
            raw_input = raw_input.strip()
            if not raw_input:
                return "continue"

            cmd_lower = raw_input.lower()
            if cmd_lower in ("/exit", "/quit"):
                if self._current_turn is not None:
                    print_warning("任务执行中，请按 Ctrl+C 中断后再退出。")
                    return "continue"
                print_success("再见！")
                return "break"

            if cmd_lower == "/clear" or "新任务" in raw_input:
                if self._current_turn is not None:
                    print_warning("任务执行中，请按 Ctrl+C 中断后再执行「新任务」或 /clear。")
                    return "continue"
                await self.reset_cli_interactive_session(history)
                is_first_input = True
                return "continue"

            has_any_history = bool(history.messages) or bool(self._manager_history.messages)
            if has_any_history:
                usage_lines = []
                roles_and_histories = [
                    ("manager", "Manager", [self._manager_history]),
                    ("coordinator", "Coordinator", [history]),
                ]
                for role, label, histories in roles_and_histories:
                    ctx_cfg = get_context_config(role)
                    cpt = float(ctx_cfg["token_estimate_fallback_chars_per_token"])
                    max_tok = await get_effective_max_context_async(role=role)
                    used_tok = 0
                    for h in histories:
                        if h.messages:
                            used_tok += await estimate_history_tokens_async(
                                h.messages, chars_per_token=cpt, role=role
                            )
                    usage_lines.append(f"{label}: {format_context_usage_line(used_tok, max_tok)}")
                print_panel("\n".join(usage_lines), title="上下文占用")

            if raw_input.startswith("/"):
                if self._current_turn is not None:
                    cmd = raw_input.split()[0].lower()
                    busy_ok = {
                        "/stop", "/status", "/cancel", "/help", "/trace", "/tasks",
                        "/pwd", "/config", "/skills",
                    }
                    if cmd not in busy_ok:
                        print_warning(
                            "当前有任务正在执行，此命令暂不可用。请先 /stop 或等待完成。"
                        )
                        return "continue"
                await self._sync_skills_for_user_turn()
                _consumed, first_override = await handle_slash_command(
                    raw_input,
                    self._skills_manager,
                    coordinator_history=history,
                    manager_history=self._manager_history,
                    reset_cli_session_for_load=lambda: self.reset_cli_interactive_session(
                        history
                    ),
                    bind_loaded_snapshot_for_save=lambda agent, path, meta: self._session_logs.bind_loaded_snapshot(
                        agent, path, meta
                    ),
                    system=self,
                )
                if first_override is not None:
                    is_first_input = first_override
                return "continue"

            if raw_input.lower() in ["quit", "exit", "退出"]:
                if self._current_turn is not None:
                    print_warning("任务执行中，请按 Ctrl+C 中断后再退出。")
                    return "continue"
                print_success("再见！")
                return "break"

            if self._current_turn is not None:
                print_warning(
                    "当前有任务正在执行。请等待完成，或按 Ctrl+C 中断。"
                )
                return "continue"

            file_refs = await asyncio.to_thread(load_file_refs, raw_input)
            for ref in file_refs:
                if not ref.ok:
                    print_warning(f"@{ref.path}: {ref.error}")
                elif ref.truncated:
                    print_success(f"📄 @{ref.path}（已截断，超出字符限制）")
                else:
                    print_success(f"📄 @{ref.path}")

            augmented = augment_text_with_file_refs(raw_input, file_refs)
            message = await asyncio.to_thread(user_message_from_cli_input, augmented)
            if message.attachments:
                logger.info(f"📎 已识别 {len(message.attachments)} 个多媒体附件")

            if is_first_input:
                task_name = message.text[:30].replace(" ", "_")
                logger.setup_task_logger(task_name)
                self._toolkit.set_task_directory(task_name)
                safe_sk = "".join(
                    c if c.isalnum() or c in ("_", "-") else "_" for c in task_name
                )[:50] or "task"
                await self.bind_session(safe_sk)
                is_first_input = False

            turn_task = self._start_user_turn(message, history)
            if turn_task is None:
                print_warning("无法启动任务（已有运行中的回合）。")
                return "continue"
            try:
                await turn_task
            except KeyboardInterrupt:
                msg = await self.cancel_current_turn()
                print_warning(msg)
            except asyncio.CancelledError:
                pass
            return "continue"

        try:
            await repl.run(process_one_line, stop_event=stop_event)
        finally:
            await self.shutdown()


def main() -> None:
    """CLI 入口：实例化 Agent 并进入交互循环（供仓库根 `main.py` 或 `python -m agent_app` 调用）。"""
    system = AgentSystem()

    async def _run() -> None:
        loop = asyncio.get_running_loop()
        stop_event = asyncio.Event()

        def _request_stop() -> None:
            stop_event.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, _request_stop)
            except (NotImplementedError, ValueError):
                signal.signal(sig, lambda *_a, _sig=sig: _request_stop())

        await system.run_interactive(stop_event=stop_event)

    asyncio.run(_run())


if __name__ == "__main__":
    main()
