"""
JSON schemas and data structures for inter-agent communication and task management.
"""

from enum import Enum
from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field
import uuid
import time


class TaskStatus(str, Enum):
    """Status values for research tasks."""

    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


class TaskDependency(BaseModel):
    """Model representing a dependency between tasks."""

    task_id: str = Field(..., description="ID of the dependent task")
    dependency_type: str = Field(
        "blocking", description="Type of dependency (blocking or non-blocking)"
    )


class TaskSchema(BaseModel):
    """Schema for representing tasks in the multi-agent orchestration system."""

    task_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="Unique task identifier"
    )
    task_type: str = Field(
        ..., description="Type of task (e.g., 'data_collection', 'analysis')"
    )
    params: Dict[str, Any] = Field(default_factory=dict, description="Task parameters")
    status: TaskStatus = Field(
        default=TaskStatus.PENDING, description="Current status of the task"
    )
    priority: int = Field(
        default=5, description="Priority level (1-10, higher is more important)"
    )
    created_at: float = Field(
        default_factory=time.time, description="Task creation timestamp"
    )
    updated_at: float = Field(
        default_factory=time.time, description="Task last update timestamp"
    )
    assigned_to: Optional[str] = Field(
        None, description="ID of agent assigned to this task"
    )
    submitter_id: Optional[str] = Field(
        None, description="ID of agent that submitted this task"
    )
    result: Optional[Dict[str, Any]] = Field(None, description="Task execution result")
    error: Optional[str] = Field(None, description="Error message if task failed")
    dependencies: List[TaskDependency] = Field(
        default_factory=list, description="Tasks that must complete before this one"
    )
    attempts: int = Field(default=0, description="Number of execution attempts")
    max_attempts: int = Field(
        default=3, description="Maximum number of execution attempts"
    )
    estimated_duration: Optional[float] = Field(
        None, description="Estimated duration in seconds"
    )
    started_at: Optional[float] = Field(
        None, description="Timestamp when task execution started"
    )
    completed_at: Optional[float] = Field(
        None, description="Timestamp when task execution completed"
    )
    tags: List[str] = Field(
        default_factory=list, description="Tags for task categorization"
    )

    class Config:
        """Pydantic model configuration."""

        json_schema_extra = {
            "example": {
                "task_id": "550e8400-e29b-41d4-a716-446655440000",
                "task_type": "data_collection",
                "params": {"query": "quantum computing", "sources": ["arxiv", "web"]},
                "status": "pending",
                "priority": 5,
                "created_at": 1616361600.0,
                "updated_at": 1616361600.0,
                "assigned_to": None,
                "submitter_id": "agent-123",
                "dependencies": [],
                "attempts": 0,
                "max_attempts": 3,
                "tags": ["research", "initial"],
            }
        }

    def is_blocked(self, completed_tasks: List[str]) -> bool:
        """
        Check if this task is blocked by any dependencies.

        Args:
            completed_tasks: List of completed task IDs

        Returns:
            True if task is blocked, False otherwise
        """
        for dependency in self.dependencies:
            if (
                dependency.dependency_type == "blocking"
                and dependency.task_id not in completed_tasks
            ):
                return True
        return False

    def update_status(self, new_status: TaskStatus) -> None:
        """
        Update the task status and timestamp.

        Args:
            new_status: New status to set
        """
        self.status = new_status
        self.updated_at = time.time()

        if new_status == TaskStatus.IN_PROGRESS and not self.started_at:
            self.started_at = time.time()
        elif new_status in [
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELED,
        ]:
            self.completed_at = time.time()

    def assign_to_agent(self, agent_id: str) -> None:
        """
        Assign this task to an agent.

        Args:
            agent_id: ID of the agent to assign
        """
        self.assigned_to = agent_id
        self.update_status(TaskStatus.ASSIGNED)

    def set_result(self, result: Dict[str, Any]) -> None:
        """
        Set the task result and mark as completed.

        Args:
            result: Task execution result
        """
        self.result = result
        self.update_status(TaskStatus.COMPLETED)

    def set_error(self, error: str) -> None:
        """
        Set error message and mark task as failed.

        Args:
            error: Error message
        """
        self.error = error
        self.attempts += 1

        if self.attempts >= self.max_attempts:
            self.update_status(TaskStatus.FAILED)
        else:
            # Reset to pending for retry
            self.update_status(TaskStatus.PENDING)


class AgentAction(BaseModel):
    """Model for tracking agent actions in the system."""

    agent_id: str = Field(..., description="ID of the agent performing the action")
    agent_type: str = Field(
        ..., description="Type of the agent (e.g., 'controller', 'acquisition')"
    )
    action_type: str = Field(..., description="Type of action performed")
    timestamp: float = Field(
        default_factory=time.time, description="When the action occurred"
    )
    details: Dict[str, Any] = Field(
        default_factory=dict, description="Additional details about the action"
    )
    related_task_id: Optional[str] = Field(
        None, description="ID of related task if applicable"
    )
