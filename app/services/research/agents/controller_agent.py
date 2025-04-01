"""
Controller agent that orchestrates the research workflow across specialized agents.
"""

import random
import asyncio
from typing import Dict, List, Any, Optional, Set
import uuid
import time
import math
from loguru import logger

from app.services.research.agents.base_agent import BaseAgent
from app.services.research.orchestration.task_queue import TaskQueue
from app.services.research.orchestration.context_manager import ResearchContextManager
from app.services.research.orchestration.schemas import (
    TaskDependency,
    TaskSchema,
    TaskStatus,
)


class ControllerAgent(BaseAgent):
    """
    Orchestrates the research process by delegating to specialized agents.
    Implements task planning, dependency management, and workflow coordination.
    """

    def __init__(
        self,
        context_manager: ResearchContextManager,
        task_queue: TaskQueue,
        agent_id: Optional[str] = None,
    ):
        """
        Initialize the controller agent.

        Args:
            context_manager: Shared research context manager
            task_queue: Task queue for asynchronous operations
            agent_id: Optional unique identifier
        """
        super().__init__(
            agent_id=agent_id,
            agent_type="controller",
            context_manager=context_manager,
            task_queue=task_queue,
        )

        # Track agent performance for Thompson sampling
        self.agent_performance = {
            "acquisition": {"success": 1, "failure": 1, "avg_time": 10.0},
            "analysis": {"success": 1, "failure": 1, "avg_time": 15.0},
            "synthesis": {"success": 1, "failure": 1, "avg_time": 20.0},
            "critique": {"success": 1, "failure": 1, "avg_time": 5.0},
        }

        # Register this agent's task handler
        if task_queue:
            task_queue.register_handler("research_orchestration", self.execute_task)

        logger.info(f"ControllerAgent initialized with ID: {self.agent_id}")

    async def execute_task(self, task: TaskSchema) -> Dict[str, Any]:
        """
        Execute a research orchestration task.

        Args:
            task: The task to execute

        Returns:
            Dictionary with task results
        """
        task_type = task.task_type
        params = task.params

        if task_type != "research_orchestration":
            raise ValueError(f"ControllerAgent cannot handle task type: {task_type}")

        # Log the task start
        await self.log_activity(
            "start_research_orchestration",
            {"task_id": task.task_id, "query": params.get("query", "")},
        )

        try:
            # Update research status
            await self.context_manager.update_status("planning")

            # Create the research plan
            research_plan = await self._create_research_plan(params)

            # Store the plan in the context
            await self.set_context("research_plan", research_plan)

            # Create and submit tasks based on the plan
            submitted_tasks = await self._submit_planned_tasks(research_plan)

            # Wait for all planned tasks to complete
            results = await self._monitor_tasks(submitted_tasks)

            # Finalize research results
            final_result = await self._finalize_research(results)

            # Log success
            await self.log_activity(
                "complete_research_orchestration",
                {"task_id": task.task_id, "success": True},
            )

            return final_result

        except Exception as e:
            # Log failure
            await self.log_activity(
                "failed_research_orchestration",
                {"task_id": task.task_id, "error": str(e)},
            )

            # Update context with error
            await self.context_manager.add_error(str(e), "controller_agent")
            await self.context_manager.update_status("failed")

            # Re-raise for task queue error handling
            raise

    async def _create_research_plan(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a plan for the research process.

        Args:
            params: Research parameters including query, sources, etc.

        Returns:
            Research plan with tasks and dependencies
        """
        query = params.get("query", "")
        sources = params.get("sources", [])
        template_id = params.get("template_id")

        await self.log_activity(
            "create_research_plan",
            {"query": query, "sources": sources, "template_id": template_id},
        )

        # A simple sequential plan for now:
        # 1. Data acquisition
        # 2. Analysis
        # 3. Synthesis
        # 4. Critique (optional)
        plan = {
            "query": query,
            "sources": sources,
            "template_id": template_id,
            "phases": [
                {
                    "name": "data_acquisition",
                    "agent_type": "acquisition",
                    "description": "Gather information from specified sources",
                    "estimated_duration": 60,  # in seconds
                    "priority": 10,
                    "dependencies": [],
                    "params": {
                        "query": query,
                        "sources": sources,
                    },
                },
                {
                    "name": "data_analysis",
                    "agent_type": "analysis",
                    "description": "Analyze collected information",
                    "estimated_duration": 45,
                    "priority": 8,
                    "dependencies": ["data_acquisition"],
                    "params": {
                        "query": query,
                    },
                },
                {
                    "name": "report_synthesis",
                    "agent_type": "synthesis",
                    "description": "Generate research report",
                    "estimated_duration": 30,
                    "priority": 6,
                    "dependencies": ["data_analysis"],
                    "params": {
                        "query": query,
                        "template_id": template_id,
                    },
                },
                {
                    "name": "quality_critique",
                    "agent_type": "critique",
                    "description": "Validate research findings and improve quality",
                    "estimated_duration": 20,
                    "priority": 4,
                    "dependencies": ["report_synthesis"],
                    "params": {
                        "query": query,
                    },
                },
            ],
            "created_at": time.time(),
        }

        logger.info(f"Created research plan for query: {query}")
        return plan

    async def _submit_planned_tasks(self, plan: Dict[str, Any]) -> Dict[str, str]:
        """
        Submit tasks based on the research plan.

        Args:
            plan: The research plan

        Returns:
            Dictionary mapping phase names to task IDs
        """
        phase_to_task_id = {}
        phases = plan.get("phases", [])

        # First pass: Submit all tasks
        for phase in phases:
            # For now, we're simulating the specialized agent tasks
            # using the existing research functions

            # Eventually each phase would be handled by a specialized agent
            task_type = f"{phase['agent_type']}_task"
            priority = phase.get("priority", 5)

            # Collect dependencies
            dependencies = []
            for dep_phase_name in phase.get("dependencies", []):
                if dep_phase_name in phase_to_task_id:
                    dependencies.append(
                        TaskDependency(
                            task_id=phase_to_task_id[dep_phase_name],
                            dependency_type="blocking",
                        )
                    )

            # Submit the task
            task_id = await self.submit_task(
                task_type=task_type,
                params=phase.get("params", {}),
                priority=priority,
                dependencies=dependencies,
                tags=[phase["name"], phase["agent_type"]],
            )

            # Store the mapping
            phase_to_task_id[phase["name"]] = task_id

            await self.log_activity(
                "submit_task",
                {
                    "task_id": task_id,
                    "phase": phase["name"],
                    "agent_type": phase["agent_type"],
                },
            )

        return phase_to_task_id

    async def _monitor_tasks(self, task_ids: Dict[str, str]) -> Dict[str, Any]:
        """
        Monitor tasks until completion.

        Args:
            task_ids: Dictionary mapping phase names to task IDs

        Returns:
            Dictionary with results from all tasks
        """
        if not self.task_queue:
            raise ValueError("Task queue not available")

        results = {}
        pending_tasks = set(task_ids.values())

        # Update progress as tasks complete
        total_tasks = len(pending_tasks)
        completed_tasks = 0

        # Simple polling approach for now
        while pending_tasks:
            for phase_name, task_id in task_ids.items():
                if task_id not in pending_tasks:
                    continue

                task = await self.task_queue.get_task(task_id)
                if not task:
                    # Task might not be available yet, continue
                    await asyncio.sleep(0.1)
                    continue

                if task.status == TaskStatus.COMPLETED:
                    # Store the result
                    results[phase_name] = task.result
                    pending_tasks.remove(task_id)
                    completed_tasks += 1

                    # Update progress
                    progress = (completed_tasks / total_tasks) * 100
                    await self.context_manager.update_progress(progress)

                    # Log success and update agent performance metrics
                    await self._update_agent_performance(
                        task.task_type.split("_")[
                            0
                        ],  # Extract agent type from task_type
                        True,
                        (
                            task.completed_at - task.started_at
                            if task.completed_at and task.started_at
                            else 30
                        ),
                    )

                elif task.status == TaskStatus.FAILED:
                    # Log failure and update agent performance metrics
                    await self._update_agent_performance(
                        task.task_type.split("_")[0],
                        False,
                        (
                            task.completed_at - task.started_at
                            if task.completed_at and task.started_at
                            else 30
                        ),
                    )

                    # For now, we'll treat failure as terminal
                    # In a more robust implementation, we might retry with a different agent
                    error_msg = f"Task {task_id} failed: {task.error}"
                    await self.context_manager.add_error(error_msg, phase_name)
                    raise ValueError(error_msg)

            # Sleep briefly before checking again
            await asyncio.sleep(0.5)

        return results

    async def _finalize_research(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Finalize research results by combining outputs from all phases.

        Args:
            results: Dictionary mapping phase names to task results

        Returns:
            Final research result
        """
        # Get synthesis result as the base
        synthesis_result = results.get("report_synthesis", {})
        critique_result = results.get("quality_critique", {})
        acquisition_result = results.get("data_acquisition", {})
        analysis_result = results.get("data_analysis", {})

        # Get research ID from context
        research_id = await self.get_context("research_id")
        query = await self.get_context("query")
        vector_namespace = await self.get_context("vector_namespace")

        # Combine results from all phases
        final_result = {
            "research_id": research_id,
            "query": query,
            "timestamp": time.time(),
            "report": synthesis_result.get("report", "No report generated"),
            "critique": critique_result.get("critique", ""),
            "sources_used": acquisition_result.get("sources_used", []),
            "result_count": acquisition_result.get("processed_docs", 0),
            "namespace": vector_namespace,
        }

        # Ensure sources is always a dictionary
        sources = synthesis_result.get("sources", {})
        if not isinstance(sources, dict):
            # Convert sources to dictionary if it's a list
            if isinstance(sources, list):
                sources_dict = {}
                for source in sources:
                    sources_dict[source] = 1  # Default count
                final_result["sources"] = sources_dict
            else:
                # Fallback with empty dict if it's neither dict nor list
                final_result["sources"] = acquisition_result.get("sources", {})
                if not isinstance(final_result["sources"], dict):
                    final_result["sources"] = {}
        else:
            final_result["sources"] = sources

        # Add sources_dict for backward compatibility
        final_result["sources_dict"] = final_result["sources"]

        # Add entire context for debugging and tracking
        final_result["context"] = await self.context_manager.get_full_context()

        # Add success flag
        final_result["success"] = "error" not in final_result[
            "context"
        ] and synthesis_result.get("success", False)

        # Log the finalization
        await self.log_activity(
            "finalize_research",
            {
                "research_id": research_id,
                "success": final_result["success"],
                "result_count": final_result["result_count"],
            },
        )

        # Update research status
        await self.context_manager.update_status("completed")

        return final_result

    async def _update_agent_performance(
        self, agent_type: str, success: bool, duration: float
    ) -> None:
        """
        Update agent performance metrics for Thompson sampling.

        Args:
            agent_type: Type of agent
            success: Whether the task was successful
            duration: Task duration in seconds
        """
        if agent_type not in self.agent_performance:
            return

        metrics = self.agent_performance[agent_type]

        # Update success/failure counts
        if success:
            metrics["success"] += 1
        else:
            metrics["failure"] += 1

        # Update average time (simple moving average)
        alpha = 0.1  # Weight for new observations
        metrics["avg_time"] = (1 - alpha) * metrics["avg_time"] + alpha * duration

    def _sample_agent_success_rate(self, agent_type: str) -> float:
        """
        Sample from Beta distribution for Thompson sampling.

        Args:
            agent_type: Type of agent

        Returns:
            Sampled success probability
        """
        if agent_type not in self.agent_performance:
            return 0.5

        metrics = self.agent_performance[agent_type]
        alpha = metrics["success"]
        beta = metrics["failure"]

        # Sample from Beta distribution
        return random.betavariate(alpha, beta)
