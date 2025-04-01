"""
ResearchAgent orchestrates the research process across multiple data sources.
This is the main entry point for the research functionality.
"""

from typing import Dict, List, Any, Optional
from loguru import logger

from app.services.data_sources.manager import DataSourceManager
from app.services.vector_db.storage import VectorStorage
from app.services.vector_db.processors import DocumentProcessor
from app.services.templates.manager import TemplateManager
from app.services.research.report import ReportGenerator
from app.services.research.history import ResearchHistoryManager

# Import refactored modules
from app.services.research.core import ResearchAgent as CoreResearchAgent
from app.services.research.focused_reports import FocusedReportGenerator
from app.services.research.literature_review import LiteratureReviewGenerator


class ResearchAgent:
    """
    Primary agent for conducting comprehensive research across multiple sources.
    Orchestrates data collection, processing, storage, and report generation.
    """

    def __init__(self, use_multi_agent: bool = True):
        """
        Initialize the research agent with required components.

        Args:
            use_multi_agent: Whether to use the experimental multi-agent system
        """
        # Initialize core components
        self.data_source_manager = DataSourceManager()
        self.vector_storage = VectorStorage()
        self.document_processor = DocumentProcessor()
        self.template_manager = TemplateManager()
        self.report_generator = ReportGenerator()
        self.history_manager = ResearchHistoryManager()

        # Initialize specialized handlers
        self.core_agent = CoreResearchAgent(use_multi_agent=use_multi_agent)

        self.focused_report_generator = FocusedReportGenerator(
            vector_storage=self.vector_storage,
            template_manager=self.template_manager,
            report_generator=self.report_generator,
            history_manager=self.history_manager,
        )

        self.literature_review_generator = LiteratureReviewGenerator(
            report_generator=self.report_generator, history_manager=self.history_manager
        )

        if use_multi_agent:
            logger.info("ResearchAgent initialized with multi-agent orchestration")
        else:
            logger.info("ResearchAgent initialized with traditional processing")

    async def conduct_research(
        self,
        query: str,
        sources: Optional[List[str]] = None,
        template_id: Optional[str] = None,
        max_results_per_source: int = 5,
        research_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Conduct comprehensive research based on the query.

        Args:
            query: The research query
            sources: List of source types to query (if None, uses all available)
            template_id: Template ID to use for the research (if any)
            max_results_per_source: Maximum results to fetch per source
            research_id: Optional ID for this research (generated if not provided)

        Returns:
            Dictionary with research results including report and metadata
        """
        return await self.core_agent.conduct_research(
            query=query,
            sources=sources,
            template_id=template_id,
            max_results_per_source=max_results_per_source,
            research_id=research_id,
        )

    async def get_research_status(self, research_id: str) -> Dict[str, Any]:
        """
        Get the status of an ongoing or completed research.

        Args:
            research_id: The ID of the research to check

        Returns:
            Status information for the research
        """
        return await self.core_agent.get_research_status(research_id)

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
        return await self.focused_report_generator.generate_focused_report(
            research_id=research_id, focus_area=focus_area, template_id=template_id
        )

    async def get_research_by_id(self, research_id: str) -> Dict[str, Any]:
        """
        Retrieve a completed research by ID.

        Args:
            research_id: The ID of the research to retrieve

        Returns:
            The research result or error message
        """
        return await self.core_agent.get_research_by_id(research_id)

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
        return await self.literature_review_generator.generate_literature_review(
            research_id=research_id,
            format_type=format_type,
            section_format=section_format,
            max_length=max_length,
        )
