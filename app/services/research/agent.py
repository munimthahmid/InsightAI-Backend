"""
ResearchAgent orchestrates the research process across multiple data sources.
"""

import asyncio
from typing import Dict, List, Any, Optional, Union, Tuple
import time
from pydantic import BaseModel
import uuid
from loguru import logger

from app.core.config import settings
from app.services.data_sources.manager import DataSourceManager
from app.services.vector_db.storage import VectorStorage
from app.services.vector_db.processors import DocumentProcessor
from app.services.templates.manager import TemplateManager
from app.services.templates.models import ResearchTemplate
from app.services.research.report import ReportGenerator
from app.services.research.history import ResearchHistoryManager


class ResearchAgent:
    """
    Primary agent for conducting comprehensive research across multiple sources.
    Orchestrates data collection, processing, storage, and report generation.
    """

    def __init__(self):
        """Initialize the research agent with required components."""
        # Initialize core components
        self.data_source_manager = DataSourceManager()
        self.vector_storage = VectorStorage()
        self.document_processor = DocumentProcessor()
        self.template_manager = TemplateManager()
        self.report_generator = ReportGenerator()
        self.history_manager = ResearchHistoryManager()

        # Track ongoing research tasks
        self.ongoing_research = {}

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
        # Generate a research ID if not provided
        if not research_id:
            research_id = str(uuid.uuid4())

        # Generate a unique namespace for this research session
        # This prevents namespace conflicts that can occur with shared namespaces
        vector_namespace = str(uuid.uuid4())

        logger.info(
            f"Starting research for query: {query} (ID: {research_id}, namespace: {vector_namespace})"
        )

        # Track this research
        self.ongoing_research[research_id] = {
            "query": query,
            "status": "started",
            "start_time": time.time(),
            "namespace": vector_namespace,
        }

        try:
            # Step 1: Determine which sources to use
            if not sources:
                # Default to all available sources if none specified
                sources = self.data_source_manager.get_available_sources()
                # Ensure web search is included by default
                if "web" not in sources:
                    sources.append("web")

            # Step 2: Fetch data from all specified sources
            raw_data = await self.data_source_manager.fetch_data_from_sources(
                query=query,
                sources=sources,
                max_results_per_source=max_results_per_source,
            )

            # Update status
            self.ongoing_research[research_id]["status"] = "data_collected"
            self.ongoing_research[research_id]["raw_data_count"] = len(raw_data)

            if not raw_data:
                logger.warning(f"No data collected for query: {query}")
                return self._create_empty_result(
                    research_id, query, "No data found from any source"
                )

            # Step 3: Process documents for vector storage
            processed_docs = []
            for source_type, docs in raw_data.items():
                for doc in docs:
                    # Add source_type to metadata
                    if "metadata" not in doc:
                        doc["metadata"] = {}
                    doc["metadata"]["source_type"] = source_type
                    doc["metadata"][
                        "research_id"
                    ] = research_id  # Tag with research ID for future reference

                    # Ensure URL is preserved in metadata and content
                    if "url" in doc and "url" not in doc["metadata"]:
                        doc["metadata"]["url"] = doc["url"]
                    elif "link" in doc and "url" not in doc["metadata"]:
                        doc["metadata"]["url"] = doc["link"]
                    elif "html_url" in doc and "url" not in doc["metadata"]:
                        doc["metadata"]["url"] = doc["html_url"]

                    # If we have a URL, make sure it's also included in the content
                    if "url" in doc["metadata"] and "page_content" in doc:
                        if "URL:" not in doc["page_content"]:
                            doc["page_content"] = (
                                f"URL: {doc['metadata']['url']}\n\n{doc['page_content']}"
                            )

                    # Log URL status for debugging
                    url = doc.get("url", doc.get("link", doc.get("html_url", None)))
                    if url:
                        logger.info(f"URL found in source {source_type}: {url}")
                    else:
                        logger.warning(f"No URL found in source {source_type}")

                    # Process the document
                    processed_chunks = self.document_processor.process_document(doc)
                    processed_docs.extend(processed_chunks)

            # Update status
            self.ongoing_research[research_id]["status"] = "documents_processed"
            self.ongoing_research[research_id]["processed_docs_count"] = len(
                processed_docs
            )

            # Step 4: Store processed documents in vector database
            document_count = self.vector_storage.process_and_store(
                documents=processed_docs,
                source_type="research",
                namespace=vector_namespace,  # Use the unique namespace created for this research
            )

            # Update status
            self.ongoing_research[research_id]["status"] = "documents_stored"
            logger.info(
                f"Stored {document_count} document chunks in namespace {vector_namespace}"
            )

            # Step 5: Search for relevant documents using the query
            relevant_docs = self.vector_storage.query(
                query_text=query,
                top_k=25,  # Get top 25 results for report generation
                namespace=vector_namespace,  # Use the same namespace we stored with
            )

            logger.info(
                f"Query returned {len(relevant_docs.get('matches', []))} matches from namespace {vector_namespace}"
            )

            # Debug: Log source data to check for URLs
            logger.info("Debugging source data for URLs:")
            for source_type, sources in raw_data.items():
                logger.info(f"Source type: {source_type}")
                for idx, source in enumerate(
                    sources[:3]
                ):  # Log first 3 sources of each type
                    title = source.get("title", "No title")
                    url = source.get("url", "No URL")
                    logger.info(f"  [{idx}] Title: {title} | URL: {url}")

            # Update status
            self.ongoing_research[research_id]["status"] = "relevant_docs_found"

            # Step 6: Generate report based on the template or standard format
            report = ""
            if template_id:
                template = self.template_manager.get_template(template_id)
                if template:
                    report = await self.report_generator.generate_template_report(
                        query=query, relevant_docs=relevant_docs, template=template
                    )
                else:
                    logger.warning(
                        f"Template {template_id} not found, using standard report"
                    )
                    report = await self.report_generator.generate_standard_report(
                        query=query, relevant_docs=relevant_docs
                    )
            else:
                report = await self.report_generator.generate_standard_report(
                    query=query, relevant_docs=relevant_docs
                )

            # Step 7: Enhance report with detailed citations
            enhanced_report = await self.report_generator.enhance_report_with_citations(
                report=report,
                evidence_chunks=relevant_docs["matches"],
                sources_dict=raw_data,
            )

            # Update status
            self.ongoing_research[research_id]["status"] = "report_generated"

            # Step 8: Save the research result
            sources_dict = {}
            for source_type, docs in raw_data.items():
                sources_dict[source_type] = len(docs)

            research_result = {
                "research_id": research_id,
                "query": query,
                "report": enhanced_report,
                "timestamp": time.time(),
                "sources_used": sources,
                "template_id": template_id,
                "result_count": len(processed_docs),
                "namespace": vector_namespace,  # Include namespace for potential future retrieval
                "raw_data": raw_data,  # Include for potential future reference
                "relevant_docs": relevant_docs,  # Include for citations and evidence
                "sources": sources_dict,  # Add sources field with counts
            }

            # Save to history
            await self.history_manager.save_research(research_result)

            # Update status
            self.ongoing_research[research_id]["status"] = "completed"
            self.ongoing_research[research_id]["end_time"] = time.time()

            # Remove from ongoing research after completion
            self.ongoing_research.pop(research_id, None)

            logger.info(f"Research completed for query: {query} (ID: {research_id})")
            return research_result

        except Exception as e:
            error_msg = f"Error during research: {str(e)}"
            logger.error(error_msg)

            # Update status to error
            if research_id in self.ongoing_research:
                self.ongoing_research[research_id]["status"] = "error"
                self.ongoing_research[research_id]["error"] = error_msg
                self.ongoing_research[research_id]["end_time"] = time.time()

            # Return error result
            return self._create_empty_result(research_id, query, error_msg)

    async def get_research_status(self, research_id: str) -> Dict[str, Any]:
        """
        Get the status of an ongoing or completed research.

        Args:
            research_id: The ID of the research to check

        Returns:
            Status information for the research
        """
        # Check if it's an ongoing research
        if research_id in self.ongoing_research:
            return self.ongoing_research[research_id]

        # Check if it's in history
        research = await self.history_manager.get_research_by_id(research_id)
        if research:
            return {
                "research_id": research_id,
                "query": research.get("query", "Unknown"),
                "status": "completed",
                "timestamp": research.get("timestamp", 0),
            }

        # Not found
        return {"research_id": research_id, "status": "not_found"}

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

    async def get_research_by_id(self, research_id: str) -> Dict[str, Any]:
        """
        Retrieve a completed research by ID.

        Args:
            research_id: The ID of the research to retrieve

        Returns:
            The research result or error message
        """
        research = await self.history_manager.get_research_by_id(research_id)
        if research:
            return research
        return {"error": f"Research with ID {research_id} not found"}

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

    def _create_empty_result(
        self, research_id: str, query: str, error_msg: str
    ) -> Dict[str, Any]:
        """Create an empty result with error information."""
        return {
            "research_id": research_id,
            "query": query,
            "error": error_msg,
            "timestamp": time.time(),
            "report": f"Research could not be completed: {error_msg}",
            "sources_used": [],
            "result_count": 0,
            "sources": {},  # Empty dict to match the ResearchResponse model
        }

    def _get_source_counts(self, raw_data: Dict[str, List]) -> Dict[str, int]:
        """
        Extract source counts from raw data.

        Args:
            raw_data: Dictionary with source types as keys and lists of results as values

        Returns:
            Dictionary with source types as keys and count of results as values
        """
        return {source_type: len(items) for source_type, items in raw_data.items()}
