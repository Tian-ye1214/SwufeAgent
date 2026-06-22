from __future__ import annotations

from typing import Tuple, TYPE_CHECKING, Any, Callable
import asyncio
import time
import traceback

import logger
from prompt import get_worker_system_prompt
from ModelGateway.agent_factory import create_agent, create_worker_toolsets_and_capabilities
from ModelGateway.ModelChecker import maybe_auto_compress_async
from app_config import get_agent_run_policy, get_agent_usage_limits, get_model_and_params
from lifecycle import AgentRegistry, LifecycleHooks, run_agent_with_lifecycle
from tools.ManagementTools import Task, TaskStatus, TaskManager
from tools.memory import ChatHistory
from tools.conversation_log import ConversationLog
from worker_result import parse_worker_result

if TYPE_CHECKING:
    from tools.BasicTools import BasicToolkit


class WorkerOrchestrator:
    """Worker 执行编排器，负责单任务执行与多任务并行调度。"""

    def __init__(
        self,
        toolkit: BasicToolkit,
        task_manager: TaskManager,
        *,
        memory_injection_getter: Callable[[], str] | None = None,
        registry: AgentRegistry | None = None,
        hooks: LifecycleHooks | None = None,
        short_term_memory: Any = None,
    ):
        self._toolkit = toolkit
        self._task_manager = task_manager
        self._memory_injection_getter = memory_injection_getter or (lambda: "")
        self._registry = registry
        self._hooks = hooks
        self._stm = short_term_memory
        self._conversation_date: str | None = None
        self._conversation_topic: str | None = None
        self._worker_adhoc_seq = 0
        self._session_key: str | None = None

    def set_session_key(self, session_key: str | None) -> None:
        self._session_key = session_key

    def set_conversation_session(self, date: str, topic: str) -> None:
        self._conversation_date = date
        self._conversation_topic = topic

    def clear_conversation_session(self) -> None:
        self._conversation_date = None
        self._conversation_topic = None
        self._worker_adhoc_seq = 0

    def _save_worker_messages(
        self, messages: list[Any], sub_id: str, extra: dict[str, Any] | None = None
    ) -> None:
        if not self._conversation_date or not self._conversation_topic:
            return
        wl = ConversationLog("worker", self._conversation_date, self._conversation_topic, sub_id=sub_id)
        merged: dict[str, Any] = {"kind": "worker"}
        if extra:
            merged.update(extra)
        wl.save(messages, extra=merged)
        if self._stm is not None and self._conversation_date and self._conversation_topic:
            mp = wl.model_messages_path()
            if mp is not None:
                root = logger.LOG_DIR.resolve()
                try:
                    log_key = mp.resolve().relative_to(root).as_posix()
                except ValueError:
                    log_key = mp.resolve().as_posix()
                sk = f"{self._conversation_date}/{self._conversation_topic}"
                self._stm.schedule_ingest_after_turn(
                    messages, log_key, "worker", sk
                )

    async def _worker_agent_id(self, suffix: str) -> str:
        if not self._session_key or self._registry is None:
            raise RuntimeError("Worker 执行前须绑定 session_key")
        return await self._registry.ensure_agent(self._session_key, "worker", suffix)

    async def _run_worker(
        self,
        worker_agent,
        prompt,
        history,
        *,
        agent_id_suffix: str,
        turn_id: str | None,
        save_sub_id: str,
        save_extra: dict[str, Any] | None = None,
        start_log_lines: tuple[str, ...] = (),
        log_done: Callable[[float], None] | None = None,
        compress_warn_label: str = "Worker",
        after_parse: Callable[[Any, str], None] | None = None,
        on_cancelled_raise: bool = False,
        exception_prefix: str = "执行异常",
        error_log_prefix: str | None = None,
        log_exception_type: bool = True,
    ) -> Tuple[bool, str]:
        try:
            for line in start_log_lines:
                logger.info(line)

            start_time = time.time()
            worker_aid = await self._worker_agent_id(agent_id_suffix)
            if self._registry is not None and self._hooks is not None:
                result = await run_agent_with_lifecycle(
                    agent=worker_agent,
                    prompt=prompt,
                    agent_id=worker_aid,
                    registry=self._registry,
                    hooks=self._hooks,
                    turn_id=turn_id,
                    message_history=history.messages,
                    usage_limits=get_agent_usage_limits(),
                )
            else:
                result = await worker_agent.run(
                    prompt,
                    message_history=history.messages,
                    usage_limits=get_agent_usage_limits(),
                )
            elapsed = time.time() - start_time
            if log_done is not None:
                log_done(elapsed)

            history.update(result)
            try:
                await maybe_auto_compress_async(
                    history,
                    role="worker",
                    task_state=self._task_manager.structured_status(),
                )
            except Exception as ce:
                logger.warning("%s 上下文自动压缩失败: %s", compress_warn_label, ce)

            self._save_worker_messages(
                list(history.messages),
                sub_id=save_sub_id,
                extra=save_extra,
            )

            output = result.output
            parsed = parse_worker_result(output)
            if after_parse is not None:
                after_parse(parsed, output)
            return parsed.success, output

        except asyncio.CancelledError:
            if on_cancelled_raise:
                raise
            return False, "已取消（Agent 运行被中止）"

        except Exception as e:
            error_msg = f"{exception_prefix}: {str(e)}"
            if error_log_prefix:
                logger.error(f"{error_log_prefix} {error_msg}")
            else:
                logger.error(f"❌ {error_msg}")
            if log_exception_type:
                logger.error(f"异常类型: {type(e).__name__}")
            logger.error(f"异常详情:\n{traceback.format_exc()}" if log_exception_type else traceback.format_exc())
            return False, error_msg

    def _create_worker_agent(
        self,
        model_name: str,
        model_params: dict[str, Any],
        instructions: str,
        *,
        include_browser: bool,
        core_extra_tools: list[Any] | None = None,
    ):
        toolsets, capabilities = create_worker_toolsets_and_capabilities(
            self._toolkit.worker_tool_groups(include_browser=include_browser),
            core_extra_tools=core_extra_tools or (),
        )
        return create_agent(
            model_name,
            model_params,
            instructions=instructions,
            toolsets=toolsets,
            capabilities=capabilities,
        )

    async def execute_task_with_worker(
        self,
        task_description: str,
        user_goal: str = "",
        retry_info: str = "",
        attachments: list | None = None,
        *,
        turn_id: str | None,
    ) -> Tuple[bool, str]:
        """
        Execute a single simple task using Worker Agent.

        Parameters:
            task_description: Task description for the Worker Agent.
            user_goal: The user's ultimate objective or broader context.
            retry_info: Information about previous failed attempts.
            attachments: Optional multimodal attachments (images/videos) to include in the prompt.

        Returns:
            Tuple[bool, str]: (success, result_message)
        """
        w_name, w_params = get_model_and_params("worker")
        mem = self._memory_injection_getter()
        worker_sysprompt = await asyncio.to_thread(
            get_worker_system_prompt, self._toolkit.skills_manager, mem
        )
        worker_agent = self._create_worker_agent(
            w_name,
            w_params,
            worker_sysprompt,
            include_browser=True,
        )
        prompt_text = (
            f"[User's Ultimate Goal]\n{user_goal}\n\n"
            f"[Current Task]\nPlease execute the following task:\n\n{task_description}"
        )
        if retry_info:
            prompt_text += (
                f"\n\nThis is a retry attempt. Previous failure details:\n{retry_info}\n"
                "Please try an alternative approach to complete the task."
            )

        prompt = [prompt_text, *attachments] if attachments else prompt_text
        adhoc_history = ChatHistory()

        self._worker_adhoc_seq += 1
        adhoc_id = self._worker_adhoc_seq
        start_lines: list[str] = [
            "=" * 50,
            "Working Agent 开始执行任务...",
            f"当前任务: {task_description}",
        ]
        if retry_info:
            start_lines.append(f"重试信息: {retry_info}")
        start_lines.append("=" * 50)

        return await self._run_worker(
            worker_agent,
            prompt,
            adhoc_history,
            agent_id_suffix=f"adhoc-{adhoc_id}",
            turn_id=turn_id,
            save_sub_id=f"adhoc-{adhoc_id}",
            save_extra={"mode": "simple", "turn_id": turn_id},
            start_log_lines=tuple(start_lines),
            log_done=lambda elapsed: logger.info(
                f"[DEBUG] worker_agent.run() 完成，耗时 {elapsed:.2f} 秒"
            ),
            compress_warn_label="Worker",
        )

    async def execute_all_tasks_parallel(
        self,
        user_goal: str,
        max_concurrent: int = 3,
        attachments: list | None = None,
        *,
        turn_id: str | None,
    ) -> str:
        """
        并行执行所有任务。按照依赖关系分波执行，同一波内的任务由多个Worker并行运行。

        Parameters:
            user_goal: 用户的最终目标描述
            max_concurrent: 最大并行Worker数量
            attachments: 可选的多模态附件列表
        """
        max_concurrent = min(max_concurrent, get_agent_run_policy().max_worker_concurrent)
        max_waves = 15
        tm = self._task_manager

        for wave in range(1, max_waves + 1):
            ready_tasks = tm.get_all_ready_tasks()

            if not ready_tasks:
                if tm.tasks and all(t.status == TaskStatus.COMPLETED for t in tm.tasks.values()):
                    logger.info("~~~~~~~~~~~所有任务已完成！~~~~~~~~~")
                elif any(t.status == TaskStatus.FAILED for t in tm.tasks.values()):
                    logger.info("！！！！！！！！部分任务失败，无法继续执行！！！！！！！！！")
                else:
                    logger.info("！！！！！！！没有可执行的任务（可能存在循环依赖）！！！！！！！！")
                break

            logger.info(f"\n{'='*60}")
            logger.info(f"第 {wave} 波并行执行 - 启动 {len(ready_tasks)} 个Worker")
            logger.info(f"{'='*60}")

            for t in ready_tasks:
                t.status = TaskStatus.IN_PROGRESS
                logger.info(f"  📋 Worker-{t.id}: {t.description}")

            sem = asyncio.Semaphore(max_concurrent)

            async def _run_one(task_to_run):
                async with sem:
                    success, output = await self._execute_worker_task(
                        task_to_run,
                        user_goal,
                        attachments=attachments,
                        turn_id=turn_id,
                    )
                    return task_to_run.id, success, output

            results = await asyncio.gather(*[_run_one(t) for t in ready_tasks], return_exceptions=True)

            for i, result in enumerate(results):
                if isinstance(result, asyncio.CancelledError):
                    failed_id = ready_tasks[i].id
                    t = tm.tasks.get(failed_id)
                    if t is not None:
                        t.failure_history.append("已取消（外部取消 Agent 运行）")
                        t.status = TaskStatus.FAILED
                    logger.warning("\n\nWorker-%s 已取消\n\n", failed_id)
                elif isinstance(result, Exception):
                    failed_id = ready_tasks[i].id
                    tm.mark_task_failed(failed_id, f"异常: {result}")
                    logger.error(f"\n\n！！！！！！！！Worker-{failed_id} 异常: {result}！！！！！！！！\n\n")
                else:
                    task_id, success, output = result
                    if success:
                        tm.mark_task_complete(task_id, output)
                        logger.info(f"Worker-{task_id} 完成")
                    else:
                        tm.mark_task_failed(task_id, output)
                        logger.warning(f"Worker-{task_id} 失败")

            logger.info(f"\n{tm.get_todo_list()}")

        return tm.get_final_summary()

    async def _execute_worker_task(
        self,
        task: Task,
        user_goal: str,
        attachments: list | None = None,
        *,
        turn_id: str | None,
    ):
        """执行单个Worker。"""
        worker_id = f"Worker-{task.id}"
        tm = self._task_manager
        full_system_prompt = await asyncio.to_thread(
            get_worker_system_prompt,
            self._toolkit.skills_manager,
            self._memory_injection_getter(),
        )
        w_name, w_params = get_model_and_params("worker")
        worker_agent = self._create_worker_agent(
            w_name,
            w_params,
            full_system_prompt,
            include_browser=False,
        )

        prompt = f"[User's Ultimate Goal]\n{user_goal}\n\n"

        if task.dependencies:
            dep_parts = []
            for dep_id in task.dependencies:
                dep_task = tm.tasks.get(dep_id)
                if dep_task and dep_task.status == TaskStatus.COMPLETED and dep_task.result:
                    dep_parts.append(f"[Task {dep_id}: {dep_task.description}]\n{dep_task.result}")
            if dep_parts:
                prompt += "[Results from Prerequisite Tasks]\n" + "\n---\n".join(dep_parts) + "\n\n"

        prompt += f"[Current Task]\nPlease execute the following task:\n\n{task.description}"

        if task.retry_count > 0:
            prompt += f"\n\nThis is retry attempt {task.retry_count}. Previous failures:\n"
            for i, failure in enumerate(task.failure_history):
                prompt += f"  Attempt {i+1}: {failure}\n"
            prompt += "Please try an alternative approach."

        prompt_input = [prompt, *attachments] if attachments else prompt

        def _after_parse(parsed, _output: str) -> None:
            task.artifacts.extend(parsed.artifacts)
            if parsed.risks:
                task.tool_summaries.append("risks: " + "; ".join(parsed.risks))
            if parsed.needs_user_confirmation:
                task.tool_summaries.append("needs user confirmation")

        return await self._run_worker(
            worker_agent,
            prompt_input,
            task.worker_chat_history,
            agent_id_suffix=f"task-{task.id}",
            turn_id=turn_id,
            save_sub_id=f"task-{task.id}",
            save_extra={"mode": "parallel", "task_id": task.id, "turn_id": turn_id},
            start_log_lines=(
                f"{'='*50}",
                f"[{worker_id}] 开始执行: {task.description}",
                f"{'='*50}",
            ),
            log_done=lambda elapsed: logger.info(f"[{worker_id}] 完成，耗时 {elapsed:.2f}秒"),
            compress_warn_label=f"[{worker_id}] Worker",
            after_parse=_after_parse,
            on_cancelled_raise=True,
            exception_prefix="Worker执行异常",
            error_log_prefix=f"[{worker_id}]",
            log_exception_type=False,
        )
