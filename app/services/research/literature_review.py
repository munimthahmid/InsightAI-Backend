"""
Literature review generation functionality for the research agent.
Enables creating formal academic literature reviews based on previous research.
"""

from typing import Dict, Any, Optional
import time
from loguru import logger


class LiteratureReviewGenerator:
    """Handles generation of formal literature reviews from research data."""

    def __init__(self, report_generator=None, history_manager=None):
        """
        Initialize literature review generator with required components.

        Args:
            report_generator: Report generator instance
            history_manager: History manager instance
        """
        self.report_generator = report_generator
        self.history_manager = history_manager

    async def generate_literature_review(
        self,
        research_id: str,
        format_type: str = "APA",
        section_format: str = "thematic",
        max_length: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate a formal literature review based on previous research.

        Args:
            research_id: The ID of the existing research
            format_type: Citation format (APA, MLA, Chicago, IEEE)
            section_format: Organization method (chronological, thematic, methodological)
            max_length: Maximum length of the review

        Returns:
            Dictionary with the literature review
        """
        # Get the original research
        research = await self.history_manager.get_research_by_id(research_id)
        if not research:
            return {"error": f"Research with ID {research_id} not found"}

        try:
            # Generate literature review
            literature_review = await self.report_generator.generate_literature_review(
                research_data=research,
                format_type=format_type,
                section_format=section_format,
                max_length=max_length,
            )

            # Create result
            result = {
                "research_id": f"{research_id}_lit_review_{int(time.time())}",
                "original_research_id": research_id,
                "query": research.get("query", "Unknown Topic"),
                "review_type": "literature_review",
                "format_type": format_type,
                "section_format": section_format,
                "report": literature_review,
                "timestamp": time.time(),
                "sources": {},  # Add empty sources dict for validation
            }

            # Save to history
            await self.history_manager.save_research(result)

            return result

        except Exception as e:
            error_msg = f"Error generating literature review: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}
