from __future__ import annotations

from typing import List, Dict, Tuple, TYPE_CHECKING, Any, Callable
import asyncio
import time
import traceback

import logger
from prompt import get_worker_system_prompt, load_prompt
from ModelGateway.agent_factory import create_agent
from ModelGateway.ModelChecker import maybe_auto_compress_async
from app_config import get_agent_run_policy, get_agent_usage_limits, get_model_and_params
from lifecycle import AgentRegistry, LifecycleHooks, run_agent_with_lifecycle
from tools.ManagementTools import Task, TaskStatus, TaskManager
from tools.memory import ChatHistory
from tools.conversation_log import ConversationLog
from worker_result import parse_worker_result

if TYPE_CHECKING:
    from tools.BasicTools import BasicToolkit


class _SharedMessageBoard:
    """Worker间共享的消息板，支持并行Worker之间的实时通讯"""

    def __init__(self):
        self._messages: List[Dict] = []
        self._lock = asyncio.Lock()

    async def post(self, worker_id: str, task_desc: str, message: str, status: str = "completed"):
        async with self._lock:
            self._messages = [
                m for m in self._messages
                if not (m["worker_id"] == worker_id and m["status"] != "completed")
            ]
            self._messages.append({
                "worker_id": worker_id,
                "task": task_desc,
                "message": message,
                "status": status,
                "timestamp": time.strftime("%H:%M:%S"),
            })

    async def get_updates(self, exclude_worker: str = None) -> str:
        async with self._lock:
            msgs = [m for m in self._messages if m["worker_id"] != exclude_worker]

        if not msgs:
            return "No updates from other workers yet."

        lines = ["=== Other Workers' Progress ==="]
        for m in msgs:
            icon = "✅" if m["status"] == "completed" else "🔄"
            lines.append(f"{icon} [{m['worker_id']}] Task: {m['task']}")
            lines.append(f"   Result: {m['message']}")
            lines.append("")
        return "\n".join(lines)


class _BoardWorkerTools:
    """并行 Worker 在消息板上的两个工具，用实例属性替代嵌套闭包。"""
    def __init__(self, board: _SharedMessageBoard, worker_id: str, task_desc: str):
        self._board = board
        self._worker_id = worker_id
        self._task_desc = task_desc

    async def check_other_workers_progress(self) -> str:
        """
        Check the progress and results of other parallel workers.
        Use this when you need to know what other workers have accomplished,
        to avoid duplicate work or to build upon their results.
        """
        return await self._board.get_updates(exclude_worker=self._worker_id)

    async def report_progress(self, message: str) -> str:
        """
        Report your current progress to other workers via the shared message board.
        Use this to share intermediate results or important findings.
        Parameters:
            message: Summary of what you've accomplished so far
        """
        await self._board.post(self._worker_id, self._task_desc, message, status="in_progress")
        return "Progress update posted to the message board."


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
        worker_agent = create_agent(
            w_name,
            w_params,
            self._toolkit.workers_tools,
            worker_sysprompt,
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

        try:
            logger.info("=" * 50)
            logger.info("Working Agent 开始执行任务...")
            logger.info(f"当前任务: {task_description}")
            if retry_info:
                logger.info(f"重试信息: {retry_info}")
            logger.info("=" * 50)

            start_time = time.time()
            self._worker_adhoc_seq += 1
            worker_aid = await self._worker_agent_id(f"adhoc-{self._worker_adhoc_seq}")
            if self._registry is not None and self._hooks is not None:
                result = await run_agent_with_lifecycle(
                    agent=worker_agent,
                    prompt=prompt,
                    agent_id=worker_aid,
                    registry=self._registry,
                    hooks=self._hooks,
                    turn_id=turn_id,
                    message_history=adhoc_history.messages,
                    usage_limits=get_agent_usage_limits(),
                )
            else:
                result = await worker_agent.run(
                    prompt,
                    message_history=adhoc_history.messages,
                    usage_limits=get_agent_usage_limits(),
                )
            elapsed = time.time() - start_time
            logger.info(f"[DEBUG] worker_agent.run() 完成，耗时 {elapsed:.2f} 秒")

            adhoc_history.update(result)
            try:
                await maybe_auto_compress_async(adhoc_history, role="worker")
            except Exception as ce:
                logger.warning("Worker 上下文自动压缩失败: %s", ce)

            self._save_worker_messages(
                list(adhoc_history.messages),
                sub_id=f"adhoc-{self._worker_adhoc_seq}",
                extra={"mode": "simple", "turn_id": turn_id},
            )

            parsed = parse_worker_result(result.output)
            return parsed.success, result.output

        except asyncio.CancelledError:
            return False, "已取消（Agent 运行被中止）"

        except Exception as e:
            error_msg = f"执行异常: {str(e)}"
            logger.error(f"❌ {error_msg}")
            logger.error(f"异常类型: {type(e).__name__}")
            logger.error(f"异常详情:\n{traceback.format_exc()}")
            if e.__cause__:
                logger.error(f"原始异常 (cause): {type(e.__cause__).__name__}: {e.__cause__}")
            if e.__context__ and e.__context__ != e.__cause__:
                logger.error(f"上下文异常 (context): {type(e.__context__).__name__}: {e.__context__}")
            if hasattr(e, 'args') and e.args:
                logger.error(f"异常参数: {e.args}")
            return False, error_msg

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
        Worker之间通过 _SharedMessageBoard 进行实时通讯。

        Parameters:
            user_goal: 用户的最终目标描述
            max_concurrent: 最大并行Worker数量
            attachments: 可选的多模态附件列表
        """
        board = _SharedMessageBoard()
        max_concurrent = min(max_concurrent, get_agent_run_policy().max_worker_concurrent)
        max_waves = 15
        tm = self._task_manager

        for wave in range(1, max_waves + 1):
            ready_tasks = tm.get_all_ready_tasks()

            if not ready_tasks:
                if tm.is_all_completed():
                    logger.info("~~~~~~~~~~~所有任务已完成！~~~~~~~~~")
                elif tm.has_failed_tasks():
                    logger.info("！！！！！！！！部分任务失败，无法继续执行！！！！！！！！！")
                else:
                    logger.info("！！！！！！！没有可执行的任务（可能存在循环依赖）！！！！！！！！")
                break

            logger.info(f"\n{'='*60}")
            logger.info(f"第 {wave} 波并行执行 - 启动 {len(ready_tasks)} 个Worker")
            logger.info(f"{'='*60}")

            for t in ready_tasks:
                tm.mark_task_in_progress(t.id)
                logger.info(f"  📋 Worker-{t.id}: {t.description}")

            sem = asyncio.Semaphore(max_concurrent)

            async def _run_one(task_to_run):
                async with sem:
                    success, output = await self._execute_worker_with_board(
                        task_to_run,
                        board,
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

    def _create_board_tools(
        self, board: _SharedMessageBoard, worker_id: str, task_desc: str
    ) -> list:
        tools = _BoardWorkerTools(board, worker_id, task_desc)
        return [tools.check_other_workers_progress, tools.report_progress]

    async def _execute_worker_with_board(
        self,
        task: Task,
        board: _SharedMessageBoard,
        user_goal: str,
        attachments: list | None = None,
        *,
        turn_id: str | None,
    ):
        """执行单个Worker，支持通过消息板与其他Worker通讯"""
        worker_id = f"Worker-{task.id}"
        tm = self._task_manager

        board_tools = self._create_board_tools(board, worker_id, task.description)
        all_tools = self._toolkit.workers_tools + board_tools

        def _parallel_worker_prompt():
            mem = self._memory_injection_getter()
            return get_worker_system_prompt(self._toolkit.skills_manager, mem) + load_prompt(
                "worker_parallel_addon.md"
            )

        full_system_prompt = await asyncio.to_thread(_parallel_worker_prompt)
        w_name, w_params = get_model_and_params("worker")
        worker_agent = create_agent(w_name, w_params, all_tools, full_system_prompt)

        other_progress = await board.get_updates(exclude_worker=worker_id)

        prompt = f"[User's Ultimate Goal]\n{user_goal}\n\n"

        if task.dependencies:
            dep_parts = []
            for dep_id in task.dependencies:
                dep_task = tm.tasks.get(dep_id)
                if dep_task and dep_task.status == TaskStatus.COMPLETED and dep_task.result:
                    dep_parts.append(f"[Task {dep_id}: {dep_task.description}]\n{dep_task.result}")
            if dep_parts:
                prompt += "[Results from Prerequisite Tasks]\n" + "\n---\n".join(dep_parts) + "\n\n"

        if "No updates" not in other_progress:
            prompt += f"[Other Workers' Current Progress]\n{other_progress}\n\n"

        prompt += f"[Current Task]\nPlease execute the following task:\n\n{task.description}"

        if task.retry_count > 0:
            prompt += f"\n\nThis is retry attempt {task.retry_count}. Previous failures:\n"
            for i, failure in enumerate(task.failure_history):
                prompt += f"  Attempt {i+1}: {failure}\n"
            prompt += "Please try an alternative approach."

        prompt_input = [prompt, *attachments] if attachments else prompt

        try:
            logger.info(f"{'='*50}")
            logger.info(f"[{worker_id}] 开始执行: {task.description}")
            logger.info(f"{'='*50}")

            start_time = time.time()
            worker_aid = await self._worker_agent_id(f"task-{task.id}")
            if self._registry is not None and self._hooks is not None:
                result = await run_agent_with_lifecycle(
                    agent=worker_agent,
                    prompt=prompt_input,
                    agent_id=worker_aid,
                    registry=self._registry,
                    hooks=self._hooks,
                    turn_id=turn_id,
                    message_history=task.worker_chat_history.messages,
                    usage_limits=get_agent_usage_limits(),
                )
            else:
                result = await worker_agent.run(
                    prompt_input,
                    message_history=task.worker_chat_history.messages,
                    usage_limits=get_agent_usage_limits(),
                )
            elapsed = time.time() - start_time

            logger.info(f"[{worker_id}] 完成，耗时 {elapsed:.2f}秒")

            task.worker_chat_history.update(result)
            try:
                await maybe_auto_compress_async(task.worker_chat_history, role="worker")
            except Exception as ce:
                logger.warning("[%s] Worker 上下文自动压缩失败: %s", worker_id, ce)

            self._save_worker_messages(
                list(task.worker_chat_history.messages),
                sub_id=f"task-{task.id}",
                extra={"mode": "parallel", "task_id": task.id, "turn_id": turn_id},
            )

            output = result.output

            parsed = parse_worker_result(output)
            task.artifacts.extend(parsed.artifacts)
            if parsed.risks:
                task.tool_summaries.append("risks: " + "; ".join(parsed.risks))
            if parsed.needs_user_confirmation:
                task.tool_summaries.append("needs user confirmation")

            if not parsed.success:
                await board.post(worker_id, task.description, output, "failed")
                return False, output
            await board.post(worker_id, task.description, output, "completed")
            return True, output

        except asyncio.CancelledError:
            await board.post(worker_id, task.description, "已取消（Agent 运行被中止）", "failed")
            raise

        except Exception as e:
            error_msg = f"Worker执行异常: {str(e)}"
            logger.error(f"[{worker_id}] {error_msg}")
            logger.error(traceback.format_exc())
            return False, error_msg
