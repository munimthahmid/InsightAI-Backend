"""
Base agent class that all specialized research agents inherit from.
Defines common interfaces and functionality for all agents.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import uuid
import json
from loguru import logger

from app.services.research.orchestration.context_manager import ResearchContextManager
from app.services.research.orchestration.task_queue import TaskQueue


class BaseAgent(ABC):
    """
    Abstract base class for all research agents in the multi-agent system.
    Provides common functionality and interfaces that all agents must implement.
    """

    def __init__(
        self,
        agent_id: Optional[str] = None,
        agent_type: str = "base",
        context_manager: Optional[ResearchContextManager] = None,
        task_queue: Optional[TaskQueue] = None,
    ):
        """
        Initialize the base agent with core components.

        Args:
            agent_id: Unique identifier for this agent instance
            agent_type: Type of agent (e.g., 'controller', 'acquisition')
            context_manager: Shared research context manager
            task_queue: Task queue for asynchronous operations
        """
        self.agent_id = agent_id or str(uuid.uuid4())
        self.agent_type = agent_type
        self.context_manager = context_manager
        self.task_queue = task_queue

        logger.info(f"Initialized {agent_type} agent with ID: {self.agent_id}")

    @abstractmethod
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a task assigned to this agent.

        Args:
            task: The task to execute, with all necessary parameters

        Returns:
            Dictionary with the task result
        """
        pass

    async def log_activity(
        self, activity_type: str, details: Dict[str, Any] = None
    ) -> None:
        """
        Log agent activity to the context manager for tracking.

        Args:
            activity_type: Description of the activity
            details: Additional details about the activity
        """
        if self.context_manager:
            if details is None:
                details = {}
            await self.context_manager.log_agent_activity(
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                activity=activity_type,
                details=details,
            )
            logger.debug(
                f"Agent activity logged: {activity_type} by {self.agent_type} agent {self.agent_id}"
            )
        else:
            logger.debug(
                f"No context manager available to log: {activity_type} by {self.agent_type} agent {self.agent_id}"
            )

    async def get_context(self, key: str) -> Any:
        """
        Get a value from the shared context.

        Args:
            key: The key to retrieve

        Returns:
            The value associated with the key, or None if not found
        """
        if self.context_manager:
            logger.debug(f"Getting context key '{key}' for agent {self.agent_id}")
            return await self.context_manager.get(key)
        else:
            logger.debug(f"No context manager available for context retrieval: {key}")
            return None

    async def set_context(self, key: str, value: Any) -> None:
        """
        Set a value in the shared context.

        Args:
            key: The key to set
            value: The value to store
        """
        if self.context_manager:
            logger.debug(f"Setting context key '{key}' for agent {self.agent_id}")
            # Ensure the value is serializable
            try:
                serializable_value = self._make_serializable(value)
                await self.context_manager.set(key, serializable_value)
            except Exception as e:
                logger.error(f"Error making value serializable for key {key}: {str(e)}")
                await self.context_manager.add_error(
                    f"Failed to store non-serializable data for key {key}: {str(e)}",
                    self.agent_type,
                )
        else:
            logger.debug(f"No context manager available for context setting: {key}")

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

    async def submit_task(
        self,
        task_type: str,
        params: Dict[str, Any],
        priority: int = 5,
        dependencies: Optional[List[Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> str:
        """
        Submit a task to the task queue for execution by any available agent.

        Args:
            task_type: Type of task to execute
            params: Parameters for the task
            priority: Priority level (1-10, higher is more important)
            dependencies: List of task dependencies
            tags: List of tags for categorization

        Returns:
            Task ID of the submitted task
        """
        if self.task_queue:
            return await self.task_queue.submit_task(
                task_type=task_type,
                params=params,
                submitter_id=self.agent_id,
                priority=priority,
                dependencies=dependencies,
                tags=tags,
            )
        logger.warning(
            f"Agent {self.agent_id} attempted to submit task but has no task queue"
        )
        return ""
