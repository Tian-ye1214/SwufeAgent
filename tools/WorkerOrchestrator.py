from __future__ import annotations

from typing import List, Dict, Tuple, TYPE_CHECKING
import asyncio
import time
import traceback

import logger
from prompt import workers_system_prompt, load_prompt
from BasicFunction import create_agent
from ModelConfig import WORKER_MODEL, workers_parameter
from tools.ManagementTools import Task, TaskStatus, TaskManager

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
                "message": message[:2000],
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


class WorkerOrchestrator:
    """Worker 执行编排器，负责单任务执行与多任务并行调度。"""

    def __init__(self, toolkit: BasicToolkit, task_manager: TaskManager):
        self._toolkit = toolkit
        self._task_manager = task_manager

    async def execute_task_with_worker(
        self,
        task_description: str,
        user_goal: str = "",
        retry_info: str = "",
        attachments: list | None = None,
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
        worker_agent = create_agent(
            WORKER_MODEL, workers_parameter, self._toolkit.workers_tools, workers_system_prompt
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

        try:
            logger.info("=" * 50)
            logger.info("Working Agent 开始执行任务...")
            logger.info(f"当前任务: {task_description}")
            if retry_info:
                logger.info(f"重试信息: {retry_info}")
            logger.info("=" * 50)

            start_time = time.time()
            result = await worker_agent.run(prompt)
            elapsed = time.time() - start_time
            logger.info(f"[DEBUG] worker_agent.run() 完成，耗时 {elapsed:.2f} 秒")

            output = result.output
            output_upper = output.upper().strip()
            output_lines = output.strip().split('\n')
            first_line = output_lines[0].upper() if output_lines else ""

            if first_line.startswith("FAILED:") or first_line.startswith("FAILED："):
                return False, output
            elif first_line.startswith("SUCCESS:") or first_line.startswith("SUCCESS："):
                return True, output
            elif output_upper.startswith("ERROR:") or output_upper.startswith("错误:") or "执行异常" in output:
                return False, output
            else:
                return True, output

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
                        task_to_run, board, user_goal, attachments=attachments
                    )
                    return task_to_run.id, success, output

            results = await asyncio.gather(
                *[_run_one(t) for t in ready_tasks],
                return_exceptions=True,
            )

            for i, result in enumerate(results):
                if isinstance(result, Exception):
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

    # ── 内部方法 ─────────────────────────────────────────────────────────────

    @staticmethod
    def _create_board_tools(board: _SharedMessageBoard, worker_id: str, task_desc: str):
        """为Worker创建消息板通讯工具（闭包绑定到具体board和worker）"""

        async def check_other_workers_progress() -> str:
            """
            Check the progress and results of other parallel workers.
            Use this when you need to know what other workers have accomplished,
            to avoid duplicate work or to build upon their results.
            """
            return await board.get_updates(exclude_worker=worker_id)

        async def report_progress(message: str) -> str:
            """
            Report your current progress to other workers via the shared message board.
            Use this to share intermediate results or important findings.
            Parameters:
                message: Summary of what you've accomplished so far
            """
            await board.post(worker_id, task_desc, message, status="in_progress")
            return "Progress update posted to the message board."

        return [check_other_workers_progress, report_progress]

    async def _execute_worker_with_board(
        self,
        task: Task,
        board: _SharedMessageBoard,
        user_goal: str,
        attachments: list | None = None,
    ):
        """执行单个Worker，支持通过消息板与其他Worker通讯"""
        worker_id = f"Worker-{task.id}"
        tm = self._task_manager

        board_tools = self._create_board_tools(board, worker_id, task.description)
        all_tools = self._toolkit.workers_tools + board_tools

        full_system_prompt = workers_system_prompt + load_prompt("worker_parallel_addon.md")
        worker_agent = create_agent(WORKER_MODEL, workers_parameter, all_tools, full_system_prompt)

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
            result = await worker_agent.run(prompt_input)
            elapsed = time.time() - start_time

            logger.info(f"[{worker_id}] 完成，耗时 {elapsed:.2f}秒")

            output = result.output

            output_lines = output.strip().split('\n')
            first_line = output_lines[0].upper() if output_lines else ""
            output_upper = output.upper().strip()

            if first_line.startswith("FAILED:") or first_line.startswith("FAILED："):
                await board.post(worker_id, task.description, output[:500], "failed")
                return False, output
            elif output_upper.startswith("ERROR:") or output_upper.startswith("错误:") or "执行异常" in output:
                await board.post(worker_id, task.description, output[:500], "failed")
                return False, output
            else:
                await board.post(worker_id, task.description, output[:500], "completed")
                return True, output

        except Exception as e:
            error_msg = f"Worker执行异常: {str(e)}"
            logger.error(f"[{worker_id}] {error_msg}")
            logger.error(traceback.format_exc())
            return False, error_msg
