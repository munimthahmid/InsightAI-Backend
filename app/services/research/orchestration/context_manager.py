"""
Context manager for shared state in the multi-agent research system.
Provides a way for agents to share data and track research progress.
"""

from typing import Dict, List, Any, Optional, Union
import time
import json
import asyncio
from loguru import logger

from app.services.research.orchestration.schemas import AgentAction


class ResearchContextManager:
    """
    Manages shared context for a research session across multiple agents.
    Provides thread-safe access to shared state and activity tracking.
    """

    def __init__(self, research_id: str):
        """
        Initialize a new context manager for a research session.

        Args:
            research_id: The ID of the research session
        """
        self.research_id = research_id
        self.context: Dict[str, Any] = {
            "research_id": research_id,
            "start_time": time.time(),
            "status": "initialized",
            "progress": 0.0,
            "errors": [],
            "warnings": [],
            "agent_activities": [],
        }
        self._lock = asyncio.Lock()
        logger.info(f"Research context initialized for research_id: {research_id}")

    async def get(self, key: str, default: Any = None) -> Any:
        """
        Get a value from the shared context.

        Args:
            key: The key to retrieve
            default: Default value if key doesn't exist

        Returns:
            The value associated with the key, or default if not found
        """
        async with self._lock:
            if key in self.context:
                return self.context[key]
            return default

    async def set(self, key: str, value: Any) -> None:
        """
        Set a value in the shared context.

        Args:
            key: The key to set
            value: The value to store
        """
        async with self._lock:
            self.context[key] = value
            logger.debug(f"Context updated: {key} for research {self.research_id}")

    async def update(self, data: Dict[str, Any]) -> None:
        """
        Update multiple values in the context at once.

        Args:
            data: Dictionary of key-value pairs to update
        """
        async with self._lock:
            self.context.update(data)
            logger.debug(f"Context bulk update for research {self.research_id}")

    async def delete(self, key: str) -> bool:
        """
        Delete a key from the context.

        Args:
            key: The key to remove

        Returns:
            True if the key was removed, False if it didn't exist
        """
        async with self._lock:
            if key in self.context:
                del self.context[key]
                logger.debug(
                    f"Context key deleted: {key} for research {self.research_id}"
                )
                return True
            return False

    async def log_agent_activity(
        self, agent_id: str, agent_type: str, activity: str, details: Dict[str, Any]
    ) -> None:
        """
        Log an activity performed by an agent.

        Args:
            agent_id: ID of the agent
            agent_type: Type of the agent
            activity: Description of the activity
            details: Additional details
        """
        action = AgentAction(
            agent_id=agent_id,
            agent_type=agent_type,
            action_type=activity,
            details=details,
            related_task_id=details.get("task_id"),
        )

        async with self._lock:
            if "agent_activities" not in self.context:
                self.context["agent_activities"] = []

            self.context["agent_activities"].append(action.dict())
            logger.debug(
                f"Agent activity logged: {activity} by {agent_type} agent {agent_id}"
            )

    async def add_error(self, error: str, source: Optional[str] = None) -> None:
        """
        Add an error to the context.

        Args:
            error: Error message
            source: Source of the error (optional)
        """
        error_entry = {
            "message": error,
            "timestamp": time.time(),
            "source": source or "unknown",
        }

        async with self._lock:
            self.context["errors"].append(error_entry)
            logger.error(f"Error in research {self.research_id}: {error}")

    async def add_warning(self, warning: str, source: Optional[str] = None) -> None:
        """
        Add a warning to the context.

        Args:
            warning: Warning message
            source: Source of the warning (optional)
        """
        warning_entry = {
            "message": warning,
            "timestamp": time.time(),
            "source": source or "unknown",
        }

        async with self._lock:
            self.context["warnings"].append(warning_entry)
            logger.warning(f"Warning in research {self.research_id}: {warning}")

    async def update_status(self, status: str) -> None:
        """
        Update the overall research status.

        Args:
            status: New status value
        """
        async with self._lock:
            self.context["status"] = status
            self.context["last_updated"] = time.time()
            logger.info(f"Research status updated to '{status}' for {self.research_id}")

    async def update_progress(self, progress: float) -> None:
        """
        Update the research progress (0-100).

        Args:
            progress: Progress percentage (0-100)
        """
        # Ensure progress is between 0 and 100
        progress = max(0, min(100, progress))

        async with self._lock:
            self.context["progress"] = progress
            self.context["last_updated"] = time.time()
            logger.debug(
                f"Research progress updated to {progress}% for {self.research_id}"
            )

    async def get_full_context(self) -> Dict[str, Any]:
        """
        Get the full context dictionary.

        Returns:
            Complete context dictionary
        """
        async with self._lock:
            # Create a deep copy to avoid concurrent modification issues
            return json.loads(json.dumps(self.context))

    async def get_agent_activities(
        self, agent_id: Optional[str] = None, agent_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get filtered agent activities.

        Args:
            agent_id: Filter by agent ID (optional)
            agent_type: Filter by agent type (optional)

        Returns:
            List of agent activities matching the filters
        """
        async with self._lock:
            activities = self.context.get("agent_activities", [])

            # Apply filters if specified
            if agent_id:
                activities = [a for a in activities if a.get("agent_id") == agent_id]
            if agent_type:
                activities = [
                    a for a in activities if a.get("agent_type") == agent_type
                ]

            return activities

    async def finalize(self, status: str = "completed") -> Dict[str, Any]:
        """
        Finalize the research context and mark as completed.

        Args:
            status: Final status (completed, failed, etc.)

        Returns:
            The final context dictionary
        """
        async with self._lock:
            self.context["status"] = status
            self.context["end_time"] = time.time()
            self.context["duration"] = (
                self.context["end_time"] - self.context["start_time"]
            )

            if status == "completed":
                self.context["progress"] = 100.0

            logger.info(
                f"Research context finalized with status '{status}' for {self.research_id}"
            )
            return await self.get_full_context()
