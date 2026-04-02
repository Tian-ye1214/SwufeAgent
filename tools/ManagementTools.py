from __future__ import annotations

from typing import List, Dict, Optional
from dataclasses import dataclass, field
from enum import Enum
import json_repair as json
import logger


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
            if not isinstance(tasks_data, list):
                return f"Error: Expected a JSON array, got {type(tasks_data).__name__}"
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
        
        if task.retry_count > task.max_retries:
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
        return task.retry_count <= task.max_retries
    
    def get_todo_list(self) -> str:
        """Get current Todo List status. """
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
        """Generate the final task execution summary report."""
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
