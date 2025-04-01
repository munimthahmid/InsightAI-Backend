"""
Focused report generation functionality for the research agent.
Enables generating detailed reports on specific aspects of previous research.
"""

from typing import Dict, List, Any, Optional
import time
import uuid
from loguru import logger


class FocusedReportGenerator:
    """Handles generation of focused reports based on existing research."""

    def __init__(
        self,
        vector_storage=None,
        template_manager=None,
        report_generator=None,
        history_manager=None,
    ):
        """
        Initialize focused report generator with required components.

        Args:
            vector_storage: Vector storage instance
            template_manager: Template manager instance
            report_generator: Report generator instance
            history_manager: History manager instance
        """
        self.vector_storage = vector_storage
        self.template_manager = template_manager
        self.report_generator = report_generator
        self.history_manager = history_manager

    async def generate_focused_report(
        self, research_id: str, focus_area: str, template_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a focused report on a specific aspect of previous research.

        Args:
            research_id: The ID of the existing research
            focus_area: The specific area to focus on
            template_id: Optional template ID for the focused report

        Returns:
            Dictionary with the focused report
        """
        # Get the original research
        research = await self.history_manager.get_research_by_id(research_id)
        if not research:
            return {"error": f"Research with ID {research_id} not found"}

        try:
            # Create a more focused query combining the original and focus area
            original_query = research.get("query", "")
            focused_query = f"{original_query} focusing on {focus_area}"

            # Create a unique namespace for this focused report
            focused_namespace = str(uuid.uuid4())
            logger.info(
                f"Using namespace {focused_namespace} for focused report on {focus_area}"
            )

            # Get relevant documents from the vector store using the focused query
            relevant_docs = self.vector_storage.query(
                query_text=focused_query,
                top_k=15,  # Fewer results for a more focused report
                namespace=focused_namespace,
            )

            # Generate report
            report = ""
            if template_id:
                template = self.template_manager.get_template(template_id)
                if template:
                    report = await self.report_generator.generate_template_report(
                        query=focused_query,
                        relevant_docs=relevant_docs,
                        template=template,
                    )
                else:
                    report = await self.report_generator.generate_standard_report(
                        query=focused_query, relevant_docs=relevant_docs
                    )
            else:
                report = await self.report_generator.generate_standard_report(
                    query=focused_query, relevant_docs=relevant_docs
                )

            # Enhance report with citations
            enhanced_report = await self.report_generator.enhance_report_with_citations(
                report=report,
                evidence_chunks=relevant_docs["matches"],
                sources_dict=research.get("raw_data", {}),
            )

            # Create result
            result = {
                "research_id": f"{research_id}_focus_{int(time.time())}",
                "original_research_id": research_id,
                "query": focused_query,
                "focus_area": focus_area,
                "namespace": focused_namespace,
                "report": enhanced_report,
                "timestamp": time.time(),
                "sources": {},  # Add empty sources dict for validation
            }

            # Save to history
            await self.history_manager.save_research(result)

            return result

        except Exception as e:
            error_msg = f"Error generating focused report: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}
