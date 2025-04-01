"""
Critique agent stub implementation.
"""

from typing import Dict, Any
from loguru import logger

from app.services.research.agents.base_agent import BaseAgent
from app.services.research.orchestration.schemas import TaskSchema


class CritiqueAgent(BaseAgent):
    """Stub implementation of the critique agent."""

    def __init__(self, context_manager=None, task_queue=None, agent_id=None):
        super().__init__(
            agent_id=agent_id,
            agent_type="critique",
            context_manager=context_manager,
            task_queue=task_queue,
        )

        if task_queue:
            task_queue.register_handler("critique_task", self.execute_task)

        logger.info(f"CritiqueAgent (stub) initialized with ID: {self.agent_id}")

    async def execute_task(self, task: TaskSchema) -> Dict[str, Any]:
        """Stub implementation."""
        logger.warning("Using stub implementation of CritiqueAgent")

        query = task.params.get("query", "Unknown query")

        return {
            "success": True,
            "critique": "This is a placeholder critique.",
            "message": "Critique generated using stub implementation",
        }
