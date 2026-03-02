from typing import List, Dict, Optional
from dataclasses import dataclass, field
from enum import Enum
import json_repair as json
import asyncio
import logger
from tools.BasicTools import ask_user
from typing import Tuple
from tools.BasicTools import workers_tools
from ModelConfig import workers_parameter
from prompt import workers_system_prompt
import time
from BasicFunction import create_agent
from ModelConfig import WORKER_MODEL
import traceback


class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Task:
    """Task data structure"""
    id: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    result: str = ""
    retry_count: int = 0
    max_retries: int = 3
    dependencies: List[str] = field(default_factory=list)
    failure_history: List[str] = field(default_factory=list)


class TaskManager:
    """Task Manager - Manages the Todo List"""
    
    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.task_order: List[str] = []
    
    def reset(self):
        """Reset task manager state, clear all tasks"""
        self.tasks.clear()
        self.task_order.clear()
        logger.debug("(task_manager reset)")
    
    def create_todo_list(self, tasks_json: str) -> str:
        """
        Create a task list from JSON.
        Parameters:
            tasks_json: JSON format task list, format:
                [{"id": "1", "description": "Task description", "dependencies": ["dependent task id"]}]
        """
        logger.debug("(create_todo_list)")
        try:
            tasks_data = json.loads(tasks_json)
            self.tasks.clear()
            self.task_order.clear()
            
            for task_data in tasks_data:
                task_id = str(task_data.get("id", len(self.tasks) + 1))
                task = Task(
                    id=task_id,
                    description=task_data.get("description", ""),
                    dependencies=task_data.get("dependencies", [])
                )
                self.tasks[task_id] = task
                self.task_order.append(task_id)
            
            return self._format_todo_list()
        except json.JSONDecodeError as e:
            return f"Error: JSON parsing failed - {e}"
        except Exception as e:
            return f"Error: Failed to create task list - {e}"
    
    def _format_todo_list(self) -> str:
        """Format and output the Todo List"""
        if not self.tasks:
            return "Task list is empty"
        
        lines = ["Task List (Todo List)", "=" * 40]
        for task_id in self.task_order:
            task = self.tasks[task_id]
            status_icon = {
                TaskStatus.PENDING: "⬜",
                TaskStatus.IN_PROGRESS: "🔄",
                TaskStatus.COMPLETED: "✅",
                TaskStatus.FAILED: "❌"
            }.get(task.status, "⬜")
            
            line = f"{status_icon} [{task.id}] {task.description}"
            if task.dependencies:
                line += f" (Dependencies: {', '.join(task.dependencies)})"
            if task.retry_count > 0:
                line += f" [Retry: {task.retry_count}/{task.max_retries}]"
            lines.append(line)

        completed = sum(1 for t in self.tasks.values() if t.status == TaskStatus.COMPLETED)
        total = len(self.tasks)
        lines.append("=" * 40)
        lines.append(f"Progress: {completed}/{total} ({completed/total*100:.1f}%)" if total > 0 else "Progress: 0/0")
        todo_list = "\n".join(lines)
        
        return todo_list
    
    def get_next_task(self) -> Optional[Task]:
        """Get the next executable task"""
        for task_id in self.task_order:
            task = self.tasks[task_id]
            if task.status == TaskStatus.PENDING:
                deps_satisfied = True
                for dep_id in task.dependencies:
                    if dep_id not in self.tasks:
                        logger.warning(f"Warning: Dependency task '{dep_id}' does not exist, ignoring this dependency")
                        continue
                    if self.tasks[dep_id].status != TaskStatus.COMPLETED:
                        deps_satisfied = False
                        break
                
                if deps_satisfied:
                    return task
        return None
    
    def get_all_ready_tasks(self) -> List[Task]:
        """获取所有可以并行执行的任务（依赖已满足且状态为PENDING）"""
        ready = []
        for task_id in self.task_order:
            task = self.tasks[task_id]
            if task.status != TaskStatus.PENDING:
                continue
            deps_satisfied = True
            for dep_id in task.dependencies:
                if dep_id not in self.tasks:
                    continue
                if self.tasks[dep_id].status != TaskStatus.COMPLETED:
                    deps_satisfied = False
                    break
            if deps_satisfied:
                ready.append(task)
        return ready
    
    def mark_task_in_progress(self, task_id: str) -> str:
        """Mark a task as in progress"""
        if task_id not in self.tasks:
            return f"Error: Task {task_id} does not exist"
        self.tasks[task_id].status = TaskStatus.IN_PROGRESS
        return f"Task {task_id} has started execution"
    
    def mark_task_complete(self, task_id: str, result: str = "") -> str:
        """
        Mark a task as completed.
        Parameters:
            task_id: Task ID
            result: Task execution result
        """
        logger.debug(f"(mark_task_complete {task_id})")
        if task_id not in self.tasks:
            return f"Error: Task {task_id} does not exist"
        
        task = self.tasks[task_id]
        task.status = TaskStatus.COMPLETED
        task.result = result
        
        return f"Task [{task_id}] completed\n{self._format_todo_list()}"
    
    def mark_task_failed(self, task_id: str, reason: str) -> str:
        """
        Record task failure and increment retry count.
        Parameters:
            task_id: Task ID
            reason: Failure reason
        """
        logger.debug(f"(mark_task_failed {task_id})")
        if task_id not in self.tasks:
            return f"Error: Task {task_id} does not exist"
        
        task = self.tasks[task_id]
        task.failure_history.append(reason)
        task.retry_count += 1
        
        if task.retry_count >= task.max_retries:
            task.status = TaskStatus.FAILED
            return f"Task [{task_id}] has reached maximum retry attempts ({task.max_retries})\nFailure history:\n" + \
                   "\n".join([f"  Attempt {i+1}: {r}" for i, r in enumerate(task.failure_history)])
        else:
            task.status = TaskStatus.PENDING
            return f"Task [{task_id}] execution failed, preparing retry attempt {task.retry_count + 1}\n" + \
                   f"Failure reason: {reason}\n" + \
                   f"Remaining retries: {task.max_retries - task.retry_count}"
    
    def can_retry(self, task_id: str) -> bool:
        """Check if a task can still be retried"""
        if task_id not in self.tasks:
            return False
        task = self.tasks[task_id]
        return task.retry_count < task.max_retries
    
    def get_task_status(self, task_id: str) -> str:
        """Get task status"""
        if task_id not in self.tasks:
            return f"Error: Task {task_id} does not exist"
        task = self.tasks[task_id]
        return f"Task [{task_id}]: {task.status.value}\nDescription: {task.description}\nResult: {task.result or 'None'}"
    
    def get_todo_list(self) -> str:
        """Get current Todo List status"""
        logger.debug("(get_todo_list)")
        return self._format_todo_list()
    
    def is_all_completed(self) -> bool:
        """Check if all tasks are completed"""
        return all(
            task.status == TaskStatus.COMPLETED 
            for task in self.tasks.values()
        )
    
    def has_failed_tasks(self) -> bool:
        """Check if there are any failed tasks"""
        return any(
            task.status == TaskStatus.FAILED 
            for task in self.tasks.values()
        )
    
    def get_final_summary(self) -> str:
        """
        Generate the final task execution summary report.
        """
        logger.debug("(get_final_summary)")
        lines = [
            "=" * 50,
            "📊 Task Execution Summary Report",
            "=" * 50,
            ""
        ]
        
        completed_tasks = []
        failed_tasks = []
        
        for task_id in self.task_order:
            task = self.tasks[task_id]
            if task.status == TaskStatus.COMPLETED:
                completed_tasks.append(task)
            elif task.status == TaskStatus.FAILED:
                failed_tasks.append(task)

        lines.append(f"✅ Completed Tasks: {len(completed_tasks)}/{len(self.tasks)}")
        lines.append("-" * 40)
        for task in completed_tasks:
            lines.append(f"  [{task.id}] {task.description}")
            if task.result:
                result_lines = task.result.split('\n')
                for rl in result_lines[:5]:
                    lines.append(f"      → {rl}")
                if len(result_lines) > 5:
                    lines.append(f"      ... ({len(result_lines) - 5} more lines)")

        if failed_tasks:
            lines.append("")
            lines.append(f"❌ Failed Tasks: {len(failed_tasks)}")
            lines.append("-" * 40)
            for task in failed_tasks:
                lines.append(f"  [{task.id}] {task.description}")
                lines.append(f"      Retry count: {task.retry_count}")
                if task.failure_history:
                    lines.append(f"      Last failure reason: {task.failure_history[-1]}")
        
        lines.append("")
        lines.append("=" * 50)

        if self.is_all_completed():
            lines.append("All tasks completed successfully!")
        elif self.has_failed_tasks():
            lines.append("⚠️ Some tasks failed. Please review the failure reasons.")
        else:
            lines.append("Tasks in progress...")
        
        return "\n".join(lines)


task_manager = TaskManager()


def create_todo_list(tasks_json: str) -> str:
    """
    Create a task list (Todo List).
    Parameters:
        tasks_json: JSON format task list, format:
            [{"id": "1", "description": "Task description", "dependencies": []}]
    Example:
        create_todo_list('[{"id": "1", "description": "Search for relevant information"}, {"id": "2", "description": "Download files", "dependencies": ["1"]}]')
    """
    return task_manager.create_todo_list(tasks_json)


def get_todo_list() -> str:
    """
    Get the current task list status.
    """
    return task_manager.get_todo_list()


def mark_task_complete(task_id: str, result: str) -> str:
    """
    Mark a task as completed.
    Parameters:
        task_id: Task ID
        result: Task execution result description
    """
    return task_manager.mark_task_complete(task_id, result)


def mark_task_failed(task_id: str, reason: str) -> str:
    """
    Mark a task as failed and record the reason. Automatically increments retry count.
    Parameters:
        task_id: Task ID
        reason: Failure reason
    """
    return task_manager.mark_task_failed(task_id, reason)


def get_final_summary() -> str:
    """
    Get the final task execution summary report.
    Call this after all tasks have been executed.
    """
    return task_manager.get_final_summary()


def get_next_pending_task() -> str:
    """
    Get the next pending task.
    Automatically considers task dependencies.
    """
    logger.debug("(get_next_pending_task)")
    task = task_manager.get_next_task()
    if task:
        task_manager.mark_task_in_progress(task.id)
        return f"Next Task:\nID: {task.id}\nDescription: {task.description}\n" + \
               (f"Current retry count: {task.retry_count}/{task.max_retries}" if task.retry_count > 0 else "")
    else:
        if task_manager.is_all_completed():
            return "All tasks completed!"
        elif task_manager.has_failed_tasks():
            return "Some tasks could not be completed. Please review failure details."
        else:
            return "No executable tasks at the moment (may be waiting for dependent tasks to complete)"


def check_task_can_retry(task_id: str) -> str:
    """
    Check if a task can still be retried.
    Parameters:
        task_id: Task ID
    """
    can_retry = task_manager.can_retry(task_id)
    task = task_manager.tasks.get(task_id)
    if task:
        return f"Task [{task_id}] {'can be retried' if can_retry else 'has reached maximum retry attempts'}\n" + \
               f"Current retry count: {task.retry_count}/{task.max_retries}"
    return f"Error: Task {task_id} does not exist"


async def execute_task_with_worker(task_description: str,
                                   user_goal: str = "",
                                   retry_info: str = "", ) -> Tuple[bool, str]:
    """
    Execute a single simple task using Worker Agen.

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
        Tuple[bool, str]: A tuple containing three elements:
            - success (bool):
                Indicates whether the task was completed successfully.
                True: Task completed successfully
                False: Task failed or encountered an error

            - result (str):
                The output message from the Worker Agent. Contains either:
                - On success: A detailed description of what was accomplished and the results
                - On failure: An explanation of what went wrong and why the task couldn't be completed
    """
    worker_agent = create_agent(WORKER_MODEL, workers_parameter, workers_tools, workers_system_prompt)
    prompt = f"[User's Ultimate Goal]\n{user_goal}\n\n[Current Task]\nPlease execute the following task:\n\n{task_description}"
    if retry_info:
        prompt += f"\n\nThis is a retry attempt. Previous failure details:\n{retry_info}\nPlease try an alternative approach to complete the task."

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
        # history = list(result.all_messages())

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


class SharedMessageBoard:
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
                "message": message[:500],
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


def _create_board_tools(board: SharedMessageBoard, worker_id: str, task_desc: str):
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


async def _execute_worker_with_board(task: Task, board: SharedMessageBoard, user_goal: str):
    """执行单个Worker，支持通过消息板与其他Worker通讯"""
    worker_id = f"Worker-{task.id}"

    board_tools = _create_board_tools(board, worker_id, task.description)
    all_tools = workers_tools + board_tools

    parallel_addon = """

## Parallel Worker Communication

You are one of several workers executing tasks IN PARALLEL. You have special communication tools:
- `check_other_workers_progress()`: See what other workers have done or are doing
- `report_progress(message)`: Share your progress with other workers

Use these tools when:
- Your task might relate to other workers' output
- You've completed a significant milestone worth sharing
- You want to check if another worker has already done something relevant
"""
    full_system_prompt = workers_system_prompt + parallel_addon
    worker_agent = create_agent(WORKER_MODEL, workers_parameter, all_tools, full_system_prompt)

    other_progress = await board.get_updates(exclude_worker=worker_id)

    prompt = f"[User's Ultimate Goal]\n{user_goal}\n\n"
    if "No updates" not in other_progress:
        prompt += f"[Other Workers' Current Progress]\n{other_progress}\n\n"

    prompt += f"[Current Task]\nPlease execute the following task:\n\n{task.description}"

    if task.retry_count > 0:
        prompt += f"\n\nThis is retry attempt {task.retry_count}. Previous failures:\n"
        for i, failure in enumerate(task.failure_history):
            prompt += f"  Attempt {i+1}: {failure}\n"
        prompt += "Please try an alternative approach."

    try:
        logger.info(f"{'='*50}")
        logger.info(f"[{worker_id}] 开始执行: {task.description}")
        logger.info(f"{'='*50}")

        start_time = time.time()
        result = await worker_agent.run(prompt)
        elapsed = time.time() - start_time

        logger.info(f"[{worker_id}] 完成，耗时 {elapsed:.2f}秒")

        output = result.output
        await board.post(worker_id, task.description, output[:500], "completed")

        output_lines = output.strip().split('\n')
        first_line = output_lines[0].upper() if output_lines else ""
        output_upper = output.upper().strip()

        if first_line.startswith("FAILED:") or first_line.startswith("FAILED："):
            return False, output
        elif output_upper.startswith("ERROR:") or output_upper.startswith("错误:") or "执行异常" in output:
            return False, output
        else:
            return True, output

    except Exception as e:
        error_msg = f"Worker执行异常: {str(e)}"
        logger.error(f"[{worker_id}] {error_msg}")
        logger.error(traceback.format_exc())
        return False, error_msg


async def execute_all_tasks_parallel(user_goal: str, max_concurrent: int = 3) -> str:
    """
    并行执行所有任务。按照依赖关系分波执行，同一波内的任务由多个Worker并行运行。
    Worker之间通过SharedMessageBoard进行实时通讯。

    Parameters:
        user_goal: 用户的最终目标描述
        max_concurrent: 最大并行Worker数量
    """
    board = SharedMessageBoard()
    max_waves = 15

    for wave in range(1, max_waves + 1):
        ready_tasks = task_manager.get_all_ready_tasks()

        if not ready_tasks:
            if task_manager.is_all_completed():
                logger.info("~~~~~~~~~~~所有任务已完成！~~~~~~~~~")
            elif task_manager.has_failed_tasks():
                logger.info("！！！！！！！！部分任务失败，无法继续执行！！！！！！！！！")
            else:
                logger.info("！！！！！！！没有可执行的任务（可能存在循环依赖）！！！！！！！！")
            break

        logger.info(f"\n{'='*60}")
        logger.info(f"第 {wave} 波并行执行 - 启动 {len(ready_tasks)} 个Worker")
        logger.info(f"{'='*60}")

        for t in ready_tasks:
            task_manager.mark_task_in_progress(t.id)
            logger.info(f"  📋 Worker-{t.id}: {t.description}")

        sem = asyncio.Semaphore(max_concurrent)

        async def _run_one(task_to_run):
            async with sem:
                success, output = await _execute_worker_with_board(task_to_run, board, user_goal)
                return task_to_run.id, success, output

        results = await asyncio.gather(
            *[_run_one(t) for t in ready_tasks],
            return_exceptions=True,
        )

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failed_id = ready_tasks[i].id
                task_manager.mark_task_failed(failed_id, f"异常: {result}")
                logger.error(f"\n\n！！！！！！！！Worker-{failed_id} 异常: {result}！！！！！！！！\n\n")
            else:
                task_id, success, output = result
                if success:
                    task_manager.mark_task_complete(task_id, output)
                    logger.info(f"Worker-{task_id} 完成")
                else:
                    task_manager.mark_task_failed(task_id, output)
                    logger.warning(f"Worker-{task_id} 失败")

        logger.info(f"\n{task_manager.get_todo_list()}")

    return task_manager.get_final_summary()


manager_tools = [
    create_todo_list,
    get_todo_list,
    ask_user,
]
