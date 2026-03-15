from prompt import manager_system_prompt
from tools.BasicTools import ask_user, set_task_directory, reset_task_directory
from tools.ManagementTools import manager_tools, task_manager, execute_task_with_worker, execute_all_tasks_parallel
from tools.memory import ChatHistory, UserMessage
from ModelConfig import manager_parameter
from BasicFunction import create_agent
from ModelConfig import MANAGER_MODEL, COORDINATOR_MODEL
import logger
import traceback
import time
import nest_asyncio
from typing import Tuple

nest_asyncio.apply()


class AgentSystem:
    """Agent 任务协调系统，管理 Manager/Coordinator 的对话历史与执行流程。"""

    def __init__(self):
        self._manager_history = ChatHistory()
        self._current_attachments: list = []

    def reset_manager_history(self):
        self._manager_history.reset()

    # ── Manager：复杂任务编排 ─────────────────────────────────────────────────

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
        manager_agent = create_agent(
            MANAGER_MODEL, manager_parameter, manager_tools, manager_system_prompt
        )
        attachments = self._current_attachments

        if not continue_from_previous:
            logger.info("第一阶段: Manager 规划任务列表")
            planning_text = f"""
        Please analyze the following user request and create a detailed task list (Todo List).
        User Request: {user_input}
        
        Use the create_todo_list tool to generate the task list.
        
        IMPORTANT PARALLEL EXECUTION RULES:
        - Tasks WITHOUT dependencies on each other will be executed IN PARALLEL by multiple workers simultaneously
        - Only add dependencies when a task TRULY needs another task's output
        - Maximize parallelism by minimizing unnecessary dependencies
        - Each task description should be sufficiently detailed for the Worker Agent to complete independently
        
        Example of good parallel design:
        - "Search for info about X" and "Search for info about Y" → NO dependencies (parallel)
        - "Write report based on search results" → depends on both search tasks (sequential after them)
    """
        else:
            logger.info("第一阶段: 基于用户反馈调整任务")
            current_todo = task_manager.get_todo_list()
            planning_text = f"""
        The user has provided additional requirements or feedback on the previous results.
        
        Current Task List Status:
        {current_todo}
        
        User's New Requirements/Feedback: {user_input}
        
        Please create an updated task list that addresses the user's new requirements.
        
        IMPORTANT: Tasks without dependencies will be executed IN PARALLEL.
        Only add dependencies when truly needed.
    """

        planning_prompt = [planning_text, *attachments] if attachments else planning_text
        result = await manager_agent.run(
            planning_prompt, message_history=self._manager_history.messages
        )
        self._manager_history.update(result)
        logger.info(result.output)

        logger.info("=" * 60)
        logger.info("第二阶段: 多Worker并行执行任务")
        logger.info("=" * 60)

        final_summary = await execute_all_tasks_parallel(user_input, attachments=attachments)
        logger.info(final_summary)

        logger.info("=" * 60)
        logger.info("第三阶段: 生成最终报告")
        logger.info("=" * 60)

        summary_text = f"""Task execution completed. Please respond directly to the user's original question based on the execution report below.

User's Original Question: {user_input}

Execution Report:
{final_summary}

Important Guidelines:
- Do not report task execution status (e.g., "file created", "task completed successfully")
- Respond directly to the user's question as if you were having a conversation
- Extract key information from the task results in the execution report to answer the user
- If task failures prevent a proper answer, briefly explain why the information could not be obtained

Examples:
- If the user asks "What's the weather like in Wenjiang?", respond with the weather conditions, not "Successfully queried the weather"
- If the user asks "Write me a script", tell them where the script was saved and what its main functions are
"""
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
            logger.info("")
            logger.info("=" * 60)
            logger.info("🎯 最终回复")
            logger.info("=" * 60)
            logger.info(final_text)
            return final_text
        except Exception as e:
            logger.warning(f"流式输出回退到普通模式: {e}")
            try:
                final_result = await manager_agent.run(
                    summary_prompt, message_history=self._manager_history.messages
                )
                self._manager_history.update(final_result)
                return final_result.output
            except Exception:
                return final_summary

    # ── Coordinator：任务入口 ─────────────────────────────────────────────────

    _COORDINATOR_SYSTEM_PROMPT = """
    You are the task coordinating agent.
    1. Determine task complexity:
        - Simple tasks (single, explicit operation): Execute directly using `execute_task_with_worker`
        - Complex tasks (requiring multiple steps or planning): Execute using `execute_task_with_manager`
    2. After task execution, provide clear feedback on the results, then immediately end the current dialogue.
    3. Do not proactively ask the user if they are satisfied; the user will proactively inform you of their next requirements.
    4. If the tool call fails, clearly explain the reason for the failure to the user.
    5. Important: After executing a task using the tool, immediately summarize the results and end the dialogue, awaiting the user's next instruction.
"""

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
        return await execute_task_with_worker(
            task_description, user_goal, retry_info, attachments=self._current_attachments
        )

    def run_agent_system(
        self, message: "str | UserMessage", history: "ChatHistory | None" = None
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
            message = UserMessage.from_text(message)
        self._current_attachments = message.attachments

        if history is None:
            history = ChatHistory()

        agent = create_agent(
            COORDINATOR_MODEL,
            manager_parameter,
            [self.execute_task_with_manager, self._execute_task_with_worker, ask_user],
            self._COORDINATOR_SYSTEM_PROMPT,
        )

        start_time = time.time()
        result = agent.run_sync(message.to_prompt(), message_history=history.messages)
        elapsed = time.time() - start_time

        logger.info(f"[DEBUG] run_agent_system agent.run_sync() 完成，耗时 {elapsed:.2f} 秒")
        logger.info(result.output)
        history.update(result)
        return history, result.output

    # ── 交互式运行 ────────────────────────────────────────────────────────────

    def run_interactive(self):
        """交互式命令行运行入口。支持在输入中包含图片/视频文件路径以发送多模态内容。"""
        log = logger.get_logger()
        log.info("=" * 60)
        log.info("输入 '新任务' 可以清除上下文重新开始")
        log.info("输入 'quit' 或 'exit' 退出程序")
        log.info("输入中可包含图片/视频文件路径，系统会自动识别为附件")
        log.info("=" * 60)

        is_first_input = True
        history = ChatHistory()

        while True:
            try:
                raw_input = input("\n📝 请输入您的任务: ").strip()

                if not raw_input:
                    continue

                if raw_input.lower() in ["quit", "exit", "退出"]:
                    log.info("👋 再见！")
                    break

                if "新任务" in raw_input:
                    task_manager.reset()
                    reset_task_directory()
                    self.reset_manager_history()
                    history.reset()
                    is_first_input = True
                    continue

                message = UserMessage.from_cli_input(raw_input)
                if message.attachments:
                    log.info(f"📎 已识别 {len(message.attachments)} 个多媒体附件")

                if is_first_input:
                    task_name = message.text[:30].replace(" ", "_")
                    logger.setup_task_logger(task_name)
                    set_task_directory(task_name)
                    is_first_input = False

                history, _ = self.run_agent_system(message, history)

            except KeyboardInterrupt:
                log.info("\n\n👋 程序已中断，再见！")
                break

            except Exception as e:
                log.error(f"\n❌ 未预期的系统错误: {e}")
                log.error(f"详细信息:\n{traceback.format_exc()}")
                task_manager.reset()


if __name__ == "__main__":
    system = AgentSystem()
    system.run_interactive()
