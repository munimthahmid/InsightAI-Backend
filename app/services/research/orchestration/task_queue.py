"""
Task queue for managing asynchronous research operations.
Implements prioritized task scheduling with dependency tracking.
"""

import asyncio
from typing import Dict, List, Any, Optional, Callable, Awaitable, Set
import time
import uuid
import json
from loguru import logger

from app.services.research.orchestration.schemas import (
    TaskSchema,
    TaskStatus,
    TaskDependency,
)


class TaskQueue:
    """
    Asynchronous task queue for research operations.
    Manages task lifecycles, priorities, and dependencies.
    """

    def __init__(self):
        """Initialize the task queue."""
        self.tasks: Dict[str, TaskSchema] = {}
        self.completed_tasks: Set[str] = set()
        self.agent_handlers: Dict[
            str, List[Callable[[TaskSchema], Awaitable[Dict[str, Any]]]]
        ] = {}
        self.task_event = asyncio.Event()
        self._running = False
        self._background_task = None

        logger.info("Task queue initialized")

    async def start(self) -> None:
        """Start the task queue processor."""
        if self._running:
            return

        self._running = True
        self._background_task = asyncio.create_task(self._process_tasks())
        logger.info("Task queue processor started")

    async def stop(self) -> None:
        """Stop the task queue processor."""
        if not self._running:
            return

        self._running = False
        if self._background_task:
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass
            self._background_task = None
        logger.info("Task queue processor stopped")

    async def submit_task(
        self,
        task_type: str,
        params: Dict[str, Any],
        submitter_id: Optional[str] = None,
        priority: int = 5,
        dependencies: Optional[List[TaskDependency]] = None,
        tags: Optional[List[str]] = None,
    ) -> str:
        """
        Submit a new task to the queue.

        Args:
            task_type: Type of task to execute
            params: Parameters for the task
            submitter_id: ID of the agent submitting the task
            priority: Priority level (1-10, higher is more important)
            dependencies: List of task dependencies
            tags: List of tags for categorization

        Returns:
            Task ID of the submitted task
        """
        task = TaskSchema(
            task_type=task_type,
            params=params,
            submitter_id=submitter_id,
            priority=priority,
            dependencies=dependencies or [],
            tags=tags or [],
        )

        self.tasks[task.task_id] = task
        logger.info(
            f"Task submitted: ID={task.task_id}, Type={task_type}, Priority={priority}"
        )

        # Signal that a new task is available
        self.task_event.set()

        return task.task_id

    async def get_task(self, task_id: str) -> Optional[TaskSchema]:
        """
        Get a task by ID.

        Args:
            task_id: ID of the task to retrieve

        Returns:
            The task or None if not found
        """
        return self.tasks.get(task_id)

    async def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a task if it hasn't started execution.

        Args:
            task_id: ID of the task to cancel

        Returns:
            True if successfully canceled, False otherwise
        """
        task = self.tasks.get(task_id)
        if not task:
            return False

        if task.status in [TaskStatus.PENDING, TaskStatus.ASSIGNED]:
            task.update_status(TaskStatus.CANCELED)
            logger.info(f"Task canceled: ID={task_id}")
            return True

        logger.warning(f"Cannot cancel task ID={task_id}, status={task.status}")
        return False

    async def mark_completed(self, task_id: str, result: Dict[str, Any]) -> bool:
        """
        Mark a task as completed with results.

        Args:
            task_id: ID of the task to mark as completed
            result: Result data from the task

        Returns:
            True if successfully updated, False otherwise
        """
        task = self.tasks.get(task_id)
        if not task:
            return False

        # Ensure result is JSON serializable
        try:
            # Test JSON serialization - this will fail if there are non-serializable objects
            json.dumps(result)
        except Exception as e:
            logger.error(f"Error serializing task result: {str(e)}")
            # Create a simplified result that's definitely serializable
            result = {
                "success": False,
                "error": f"Task result could not be serialized: {str(e)}",
                "partial_result": (
                    str(result)[:1000] + "..."
                    if len(str(result)) > 1000
                    else str(result)
                ),
            }

        task.set_result(result)
        self.completed_tasks.add(task_id)

        # Signal that task status has changed, potentially unblocking dependencies
        self.task_event.set()

        logger.info(f"Task completed: ID={task_id}")
        return True

    async def mark_failed(self, task_id: str, error: str) -> bool:
        """
        Mark a task as failed with error details.

        Args:
            task_id: ID of the task to mark as failed
            error: Error message

        Returns:
            True if successfully updated, False otherwise
        """
        task = self.tasks.get(task_id)
        if not task:
            return False

        task.set_error(error)

        # If task has hit max attempts, consider it done (but failed)
        if task.status == TaskStatus.FAILED:
            self.completed_tasks.add(task_id)

        # Signal that task status has changed
        self.task_event.set()

        logger.warning(
            f"Task failed: ID={task_id}, Error={error}, Attempts={task.attempts}/{task.max_attempts}"
        )
        return True

    def register_handler(
        self, task_type: str, handler: Callable[[TaskSchema], Awaitable[Dict[str, Any]]]
    ) -> None:
        """
        Register a handler function for a specific task type.

        Args:
            task_type: The type of task this handler can process
            handler: Async function that processes tasks of this type
        """
        if task_type not in self.agent_handlers:
            self.agent_handlers[task_type] = []

        self.agent_handlers[task_type].append(handler)
        logger.info(f"Handler registered for task type: {task_type}")

    async def _process_tasks(self) -> None:
        """Process tasks in the queue continuously."""
        while self._running:
            # Clear the event flag
            self.task_event.clear()

            # Get all executable tasks (not blocked, pending, with registered handlers)
            executable_tasks = self._get_executable_tasks()

            if executable_tasks:
                # Process each executable task
                for task in executable_tasks:
                    # Only process if we have handlers for this task type
                    if task.task_type in self.agent_handlers:
                        # Update status to in_progress
                        task.update_status(TaskStatus.IN_PROGRESS)

                        # Create a task to execute this asynchronously
                        asyncio.create_task(self._execute_task(task))

            # Wait for new tasks or status changes
            await self.task_event.wait()

    def _get_executable_tasks(self) -> List[TaskSchema]:
        """
        Get all tasks that can be executed based on status and dependencies.

        Returns:
            List of executable tasks, sorted by priority
        """
        executable = []

        for task in self.tasks.values():
            # Check if task is pending and not blocked by dependencies
            if (
                task.status == TaskStatus.PENDING
                and not task.is_blocked(list(self.completed_tasks))
                and task.task_type in self.agent_handlers
            ):
                executable.append(task)

        # Sort by priority (higher first) and creation time (older first)
        return sorted(executable, key=lambda t: (-t.priority, t.created_at))

    async def _execute_task(self, task: TaskSchema) -> None:
        """
        Execute a task using the appropriate handler.

        Args:
            task: The task to execute
        """
        handlers = self.agent_handlers.get(task.task_type, [])

        if not handlers:
            await self.mark_failed(
                task.task_id, f"No handlers registered for task type {task.task_type}"
            )
            return

        # For now, just use the first handler (could implement more sophisticated selection)
        handler = handlers[0]

        try:
            # Execute the handler
            result = await handler(task)

            # Mark task as completed with result
            await self.mark_completed(task.task_id, result)

        except Exception as e:
            # Mark task as failed
            await self.mark_failed(task.task_id, str(e))
            logger.exception(f"Error executing task {task.task_id}: {str(e)}")

    def _make_serializable(self, obj: Any) -> Any:
        """
        Convert a complex object to a JSON-serializable form.

        Args:
            obj: Object to make serializable

        Returns:
            JSON-serializable version of the object
        """
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
            return self._make_serializable(obj.to_dict())
        elif hasattr(obj, "__dict__"):
            # For objects with __dict__, convert to dictionary
            return self._make_serializable(obj.__dict__)
        else:
            # Try to return the object directly, or convert to string if that fails
            try:
                # Test JSON serialization
                json.dumps(obj)
                return obj
            except (TypeError, OverflowError):
                # If it can't be serialized, convert to string
                return str(obj)
