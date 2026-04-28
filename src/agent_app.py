import app_config

app_config.load_config()

from prompt import get_manager_system_prompt, get_coordinator_system_prompt, load_prompt
from tools.BasicTools import BasicToolkit
from tools.ManagementTools import TaskManager
from tools.WorkerOrchestrator import WorkerOrchestrator
from tools.Memory import (
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
from app_config import get_context_config, get_model_and_params
from cli_commands import handle_slash_command
import logger
import traceback
import time
import asyncio
from typing import Tuple

from pydantic_ai.exceptions import ModelHTTPError

from skills.SkillsManager import SkillsManager


def _format_user_log_text(message: UserMessage) -> str:
    """供任务 .log 落盘：用户可见文本 + 附件说明。"""
    t = (message.text or "").strip()
    n = len(message.attachments or [])
    if n and t:
        return f"{t}\n（含 {n} 个多媒体附件）"
    if n:
        return f"（仅 {n} 个多媒体附件，无文本）"
    return t or "（空文本）"


class AgentSystem:
    """Agent 任务协调系统，管理 Manager/Coordinator 的对话历史与执行流程。"""

    def __init__(self):
        self._skills_manager = SkillsManager()
        self._manager_history = ChatHistory()
        self._current_attachments: list = []
        self._long_term_memory = LongTermMemory()
        self._long_term_memory.refresh_from_disk_sync()
        self._short_term_memory = ShortTermMemory(log_root=logger.LOG_DIR)
        self._toolkit = BasicToolkit(
            self._skills_manager,
            extra_worker_tools=[
                self._short_term_memory.recall,
                self._long_term_memory.add,
                self._long_term_memory.remove,
                self._long_term_memory.list_memory,
            ],
        )
        self._task_manager = TaskManager()
        self._orchestrator = WorkerOrchestrator(
            self._toolkit,
            self._task_manager,
            memory_injection_getter=lambda: self._long_term_memory.get_injection(),
        )
        self._session_logs = SessionConversationLogs(
            on_activate=self._orchestrator.set_conversation_session,
            on_reset=self._orchestrator.clear_conversation_session,
        )
        self._context_prewarmed = False

    async def _sync_skills_for_user_turn(self) -> None:
        """每次用户输入：在同一实例上重新扫描 skills（静默），避免磁盘 I/O 阻塞事件循环。"""
        await asyncio.to_thread(self._skills_manager.refresh)

    def set_ask_user_handler(self, handler):
        self._toolkit.set_ask_user_handler(handler)

    def set_task_directory(self, task_name: str):
        self._toolkit.set_task_directory(task_name)

    def reset_task_directory(self):
        self._toolkit.reset_task_directory()

    def reset_manager_history(self):
        self._manager_history.reset()

    async def _after_coordinator_turn(self, history: ChatHistory) -> None:
        """每轮 Coordinator 结束后：后台静默合并长期记忆（不阻塞当前回复）。"""
        msgs = list(history.messages)
        asyncio.create_task(self._long_term_memory.consolidate_from_messages(msgs, silent=True))
        asyncio.create_task(
            self._long_term_memory.consolidate_from_logs(logger.LOG_DIR, silent=True)
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
        logger.log_conversation_user(user_input)

        manager_tools = [
            self._task_manager.create_todo_list,
            self._task_manager.get_todo_list,
            self._toolkit.ask_user,
        ]
        m_name, m_params = get_model_and_params("manager")
        mem_inj = self._long_term_memory.get_injection()
        mgr_prompt = await asyncio.to_thread(
            get_manager_system_prompt, self._skills_manager, mem_inj
        )
        manager_agent = create_agent(m_name, m_params, manager_tools, mgr_prompt)
        attachments = self._current_attachments

        if not continue_from_previous:
            logger.info("第一阶段: Manager 规划任务列表")
            tmpl = await asyncio.to_thread(load_prompt, "manager_planning_new.md")
            planning_text = tmpl.format(user_input=user_input)
        else:
            logger.info("第一阶段: 基于用户反馈调整任务")
            current_todo = self._task_manager.get_todo_list()
            tmpl = await asyncio.to_thread(load_prompt, "manager_planning_continue.md")
            planning_text = tmpl.format(
                user_input=user_input,
                current_todo=current_todo
            )

        planning_prompt = [planning_text, *attachments] if attachments else planning_text
        result = await manager_agent.run(
            planning_prompt, message_history=self._manager_history.messages
        )
        self._manager_history.update(result)
        self._session_logs.for_agent("manager").save(
            self._manager_history.messages, extra={"kind": "manager", "phase": "planning"}
        )
        try:
            if await maybe_auto_compress_async(self._manager_history, role="manager"):
                logger.info("已自动压缩 Manager 上下文（达到配置阈值）")
        except Exception as ce:
            logger.warning("Manager 自动压缩失败: %s", ce)
        logger.log_conversation_model(result.output)

        logger.info("=" * 60)
        logger.info("第二阶段: 多Worker并行执行任务")
        logger.info("=" * 60)

        final_summary = await self._orchestrator.execute_all_tasks_parallel(
            user_input, attachments=attachments
        )
        logger.log_conversation_task_summary(final_summary)

        logger.info("=" * 60)
        logger.info("第三阶段: 生成最终报告")
        logger.info("=" * 60)

        summary_tmpl = await asyncio.to_thread(load_prompt, "manager_summary.md")
        summary_text = summary_tmpl.format(
            user_input=user_input,
            final_summary=final_summary
        )
        summary_prompt = [summary_text, *attachments] if attachments else summary_text
        try:
            async with manager_agent.run_stream(
                summary_prompt, message_history=self._manager_history.messages
            ) as stream:
                collected = []
                async for chunk in stream.stream_text(delta=True):
                    print(chunk, end="", flush=True)
                    collected.append(chunk)
                print(flush=True)
                final_text = "".join(collected)
                self._manager_history.update(stream)
            self._session_logs.for_agent("manager").save(
                self._manager_history.messages, extra={"kind": "manager", "phase": "summary"}
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
                final_result = await manager_agent.run(
                    summary_prompt, message_history=self._manager_history.messages
                )
                self._manager_history.update(final_result)
                self._session_logs.for_agent("manager").save(
                    self._manager_history.messages, extra={"kind": "manager", "phase": "summary"}
                )
                try:
                    if await maybe_auto_compress_async(self._manager_history, role="manager"):
                        logger.info("已自动压缩 Manager 上下文（summary 后，达到配置阈值）")
                except Exception as ce:
                    logger.warning("Manager 自动压缩失败: %s", ce)
                logger.log_conversation_model(final_result.output)
                return final_result.output
            except Exception:
                logger.log_conversation_task_summary(final_summary)
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
            task_description, user_goal, retry_info, attachments=self._current_attachments
        )

    async def run_agent_system(
        self,
        message: "str | UserMessage",
        history: "ChatHistory | None" = None,
        *,
        conversation_log_hint: str = "",
        conversation_log_extra: dict | None = None,
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

        if not self._context_prewarmed:
            await prewarm_effective_max_contexts_by_role_async(
                reason="非 CLI 首条任务（预取三角色）"
            )
            self._context_prewarmed = True

        await self._sync_skills_for_user_turn()

        if history is None:
            history = ChatHistory()

        self._session_logs.ensure(conversation_log_hint or message.text or "")
        logger.log_conversation_user(_format_user_log_text(message))

        c_name, c_params = get_model_and_params("coordinator")
        coordinator_tools = [
            self.execute_task_with_manager,
            self._execute_task_with_worker,
            *self._toolkit.workers_tools,
        ]
        mem_inj = self._long_term_memory.get_injection()
        coord_prompt = await asyncio.to_thread(
            get_coordinator_system_prompt, self._skills_manager, mem_inj
        )
        agent = create_agent(
            c_name,
            c_params,
            coordinator_tools,
            coord_prompt,
        )

        start_time = time.time()
        result = await agent.run(message.to_prompt(), message_history=history.messages)
        elapsed = time.time() - start_time

        logger.info(f"[DEBUG] run_agent_system agent.run() 完成，耗时 {elapsed:.2f} 秒")
        logger.log_conversation_model(result.output)
        history.update(result)
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
        return history, result.output

    async def run_interactive(self):
        """交互式命令行运行入口。支持在输入中包含图片/视频文件路径以发送多模态内容。"""
        logger.info("=" * 60)
        logger.info("输入 /help 查看斜杠命令；输入 '新任务' 清除上下文；quit/exit 退出")
        logger.info("输入中可包含图片/视频文件路径，系统会自动识别为附件")
        logger.info("=" * 60)
        await prewarm_effective_max_contexts_by_role_async(reason="程序启动")
        self._context_prewarmed = True

        is_first_input = True
        history = ChatHistory()

        while True:
            try:
                worker_histories = [
                    t.worker_chat_history for t in self._task_manager.tasks.values()
                ]
                has_any_history = bool(history.messages) or bool(self._manager_history.messages)
                if has_any_history:
                    print("\n── 上下文占用 ──")
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
                        print(f"{label}: {format_context_usage_line(used_tok, max_tok)}")
                raw_input = (await asyncio.to_thread(input, "\n\U0001f4dd 请输入您的任务: ")).strip()
                if not raw_input:
                    continue

                if raw_input.startswith("/"):
                    await self._sync_skills_for_user_turn()
                    worker_histories = [
                        t.worker_chat_history for t in self._task_manager.tasks.values()
                    ]
                    await handle_slash_command(
                        raw_input,
                        self._skills_manager,
                        coordinator_history=history,
                        manager_history=self._manager_history,
                        worker_histories=worker_histories,
                    )
                    continue

                if raw_input.lower() in ["quit", "exit", "退出"]:
                    logger.info("👋 再见！")
                    break

                if "新任务" in raw_input:
                    self._task_manager.reset()
                    self._toolkit.reset_task_directory()
                    self.reset_manager_history()
                    self._session_logs.reset()
                    history.reset()
                    is_first_input = True
                    continue

                message = await asyncio.to_thread(user_message_from_cli_input, raw_input)
                if message.attachments:
                    logger.info(f"📎 已识别 {len(message.attachments)} 个多媒体附件")

                if is_first_input:
                    task_name = message.text[:30].replace(" ", "_")
                    logger.setup_task_logger(task_name)
                    self._toolkit.set_task_directory(task_name)
                    is_first_input = False

                history, _ = await self.run_agent_system(
                    message, history, conversation_log_hint=message.text[:40]
                )

            except KeyboardInterrupt:
                logger.info("\n\n👋 程序已中断，再见！")
                break

            except ModelHTTPError as e:
                body = e.body or {}
                code = body.get("code", "") if isinstance(body, dict) else ""
                if code == "data_inspection_failed":
                    logger.warning(
                        "\n⚠️  模型内容安全审查拦截：您的输入或上下文中包含被判定为不当的内容。\n"
                        '   请尝试换一种表达方式重新输入，或输入"新任务"清空上下文后重试。'
                    )
                else:
                    logger.error(f"\n❌ 模型请求错误 (HTTP {e.status_code}): {e}")
                    logger.error(f"详细信息:\n{traceback.format_exc()}")
                    self._task_manager.reset()

            except Exception as e:
                logger.error(f"\n❌ 未预期的系统错误: {e}")
                logger.error(f"详细信息:\n{traceback.format_exc()}")
                self._task_manager.reset()


def main() -> None:
    """CLI 入口：实例化 Agent 并进入交互循环（供仓库根 `main.py` 或 `python -m agent_app` 调用）。"""
    system = AgentSystem()
    asyncio.run(system.run_interactive())


if __name__ == "__main__":
    main()
