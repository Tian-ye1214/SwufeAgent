from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json_repair as json
from redlotus.infra import logger

from redlotus.tools.memory import ChatHistory


class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PENDING_CONFIRMATION = "pending_confirmation"


@dataclass
class Task:
    """Task data structure"""
    id: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    result: str = ""
    retry_count: int = 0
    max_retries: int = 3
    dependencies: list[str] = field(default_factory=list)
    failure_history: list[str] = field(default_factory=list)
    worker_chat_history: ChatHistory = field(default_factory=ChatHistory)
    artifacts: list[str] = field(default_factory=list)
    tool_summaries: list[str] = field(default_factory=list)


class TaskManager:
    """Task Manager - Manages the Todo List"""
    
    def __init__(self):
        self.tasks: dict[str, Task] = {}
        self.task_order: list[str] = []
    
    def reset(self):
        """Reset task manager state, clear all tasks"""
        self.tasks.clear()
        self.task_order.clear()
        logger.debug("(task_manager reset)")
    
    def create_todo_list(self, tasks_json: str) -> str:
        """
        Create a task list from JSON.
        Parameters:
            tasks_json: JSON array of id/description/dependencies objects.
        """
        logger.debug("(create_todo_list)")
        try:
            tasks_data = json.loads(tasks_json)
            if not isinstance(tasks_data, list):
                return f"Error: Expected a JSON array, got {type(tasks_data).__name__}"
            validation_error = self._validate_tasks_data(tasks_data)
            if validation_error:
                return f"Error: {validation_error}"
            self.tasks.clear()
            self.task_order.clear()
            
            for task_data in tasks_data:
                task_id = str(task_data.get("id", len(self.tasks) + 1)).strip()
                task = Task(
                    id=task_id,
                    description=task_data.get("description", ""),
                    dependencies=[str(d).strip() for d in task_data.get("dependencies", [])]
                )
                self.tasks[task_id] = task
                self.task_order.append(task_id)
            
            return self._format_todo_list()
        except Exception as e:
            return f"Error: Failed to create task list - {e}"

    def _validate_tasks_data(self, tasks_data: list) -> str:
        seen: set[str] = set()
        deps_by_id: dict[str, list[str]] = {}
        for i, task_data in enumerate(tasks_data):
            if not isinstance(task_data, dict):
                return f"task #{i + 1} must be an object"
            task_id = str(task_data.get("id", i + 1)).strip()
            if not task_id:
                return f"task #{i + 1} id must not be empty"
            if task_id in seen:
                return f"duplicate task id: {task_id}"
            seen.add(task_id)
            desc = str(task_data.get("description", "")).strip()
            if not desc:
                return f"task {task_id} description must not be empty"
            raw_deps = task_data.get("dependencies", [])
            if not isinstance(raw_deps, list):
                return f"task {task_id} dependencies must be an array"
            deps = [str(d).strip() for d in raw_deps]
            if any(not d for d in deps):
                return f"task {task_id} dependencies must not contain empty ids"
            deps_by_id[task_id] = deps

        for task_id, deps in deps_by_id.items():
            for dep_id in deps:
                if dep_id not in seen:
                    return f"task {task_id} has unknown dependency: {dep_id}"

        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(task_id: str, path: list[str]) -> str | None:
            if task_id in visited:
                return None
            if task_id in visiting:
                cycle = " -> ".join(path + [task_id])
                return f"dependency cycle detected: {cycle}"
            visiting.add(task_id)
            for dep_id in deps_by_id.get(task_id, []):
                err = visit(dep_id, path + [task_id])
                if err:
                    return err
            visiting.remove(task_id)
            visited.add(task_id)
            return None

        for task_id in deps_by_id:
            err = visit(task_id, [])
            if err:
                return err
        return ""
    
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
                TaskStatus.FAILED: "❌",
                TaskStatus.PENDING_CONFIRMATION: "⏸"
            }[task.status]
            
            line = f"{status_icon} [{task.id}] {task.description}"
            if task.dependencies:
                line += f" (Dependencies: {', '.join(task.dependencies)})"
            if task.retry_count > 0:
                line += f" [Retry: {task.retry_count}/{task.max_retries}]"
            lines.append(line)

        completed = sum(1 for t in self.tasks.values() if t.status == TaskStatus.COMPLETED)
        total = len(self.tasks)
        lines.append("=" * 40)
        lines.append(f"Progress: {completed}/{total} ({completed/total*100:.1f}%)")
        todo_list = "\n".join(lines)
        
        return todo_list
    
    def get_all_ready_tasks(self) -> list[Task]:
        """获取所有可以并行执行的任务（依赖已满足且状态为PENDING）"""
        return [
            self.tasks[task_id]
            for task_id in self.task_order
            if self.tasks[task_id].status == TaskStatus.PENDING
            and all(
                self.tasks[dep_id].status == TaskStatus.COMPLETED
                for dep_id in self.tasks[task_id].dependencies
            )
        ]
    
    def mark_task_complete(self, task_id: str, result: str = "") -> str:
        """
        Mark a task as completed.
        Parameters:
            task_id: Task ID
            result: Task result
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
        Record a task failure and schedule retry while attempts remain.
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
        
        if task.retry_count > task.max_retries:
            task.status = TaskStatus.FAILED
            return f"Task [{task_id}] has reached maximum retry attempts ({task.max_retries})\nFailure history:\n" + \
                   "\n".join([f"  Attempt {i+1}: {r}" for i, r in enumerate(task.failure_history)])
        else:
            task.status = TaskStatus.PENDING
            return f"Task [{task_id}] execution failed, preparing retry attempt {task.retry_count + 1}\n" + \
                   f"Failure reason: {reason}\n" + \
                   f"Remaining retries: {task.max_retries - task.retry_count}"
    
    def get_todo_list(self) -> str:
        """Get current Todo List status. """
        logger.debug("(get_todo_list)")
        return self._format_todo_list()

    def structured_status(self) -> str:
        if not self.tasks:
            return "Task Status\n(no tasks)"
        lines = ["Task Status"]
        for task_id in self.task_order:
            task = self.tasks[task_id]
            blocked = self._blocked_reason(task)
            lines.append(
                f"[{task.status.value}] {task.id}: {task.description}"
                f" deps={task.dependencies or []} retries={task.retry_count}/{task.max_retries}"
            )
            if blocked:
                lines.append(f"  blocked_reason: {blocked}")
            if task.failure_history:
                lines.append(f"  last_failure: {task.failure_history[-1]}")
            if task.artifacts:
                lines.append(f"  artifacts: {', '.join(task.artifacts)}")
            if task.tool_summaries:
                lines.append(f"  tools: {'; '.join(task.tool_summaries[-3:])}")
        return "\n".join(lines)

    def _blocked_reason(self, task: Task) -> str:
        if task.status != TaskStatus.PENDING:
            return ""
        missing = [d for d in task.dependencies if d not in self.tasks]
        if missing:
            return f"unknown dependencies: {', '.join(missing)}"
        incomplete = [
            d
            for d in task.dependencies
            if self.tasks[d].status != TaskStatus.COMPLETED
        ]
        if incomplete:
            return f"waiting for dependencies: {', '.join(incomplete)}"
        return ""
    
    def get_final_summary(self) -> str:
        """Generate the final task execution summary report."""
        logger.debug("(get_final_summary)")
        if not self.tasks:
            return "Manager did not create executable tasks."
        lines = [
            "=" * 50,
            "📊 Task Execution Summary Report",
            "=" * 50,
            ""
        ]
        
        ordered_tasks = [self.tasks[task_id] for task_id in self.task_order]
        completed_tasks = [t for t in ordered_tasks if t.status == TaskStatus.COMPLETED]
        failed_tasks = [t for t in ordered_tasks if t.status == TaskStatus.FAILED]

        lines.append(f"✅ Completed Tasks: {len(completed_tasks)}/{len(self.tasks)}")
        lines.append("-" * 40)
        for task in completed_tasks:
            lines.append(f"  [{task.id}] {task.description}")
            if task.result:
                for rl in task.result.split('\n'):
                    lines.append(f"      → {rl}")

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

        if len(completed_tasks) == len(self.tasks):
            lines.append("All tasks completed successfully!")
        elif failed_tasks:
            lines.append("⚠️ Some tasks failed. Please review the failure reasons.")
        else:
            lines.append("Tasks in progress...")
        
        return "\n".join(lines)
