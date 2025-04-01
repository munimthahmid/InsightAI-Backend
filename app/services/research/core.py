"""
Core research agent functionality for orchestrating the research process.
"""

import asyncio
from typing import Dict, List, Any, Optional
import time
import uuid
import json
from loguru import logger

from app.core.config import settings
from app.services.data_sources.manager import DataSourceManager
from app.services.vector_db.storage import VectorStorage
from app.services.vector_db.processors import DocumentProcessor
from app.services.templates.manager import TemplateManager
from app.services.research.report import ReportGenerator
from app.services.research.history import ResearchHistoryManager
from app.services.research.utils import create_empty_result

# Import multi-agent system components (gradually migrating)
from app.services.research.orchestration.task_queue import TaskQueue
from app.services.research.orchestration.context_manager import ResearchContextManager
from app.services.research.orchestration.schemas import TaskStatus
from app.services.research.agents.controller_agent import ControllerAgent
from app.services.research.agents.acquisition_agent import AcquisitionAgent
from app.services.research.agents.analysis_agent import AnalysisAgent
from app.services.research.agents.synthesis_agent import SynthesisAgent
from app.services.research.agents.critique_agent import CritiqueAgent


class ResearchAgent:
    """
    Primary agent for conducting comprehensive research across multiple sources.
    Orchestrates data collection, processing, storage, and report generation.
    """

    def __init__(self, use_multi_agent: bool = True):
        """
        Initialize the research agent with required components.

        Args:
            use_multi_agent: Whether to use the multi-agent system (experimental)
        """
        # Initialize core components
        self.data_source_manager = DataSourceManager()
        self.vector_storage = VectorStorage()
        self.document_processor = DocumentProcessor()
        self.template_manager = TemplateManager()
        self.report_generator = ReportGenerator()
        self.history_manager = ResearchHistoryManager()

        # Track ongoing research tasks
        self.ongoing_research = {}

        # Multi-agent system (experimental)
        self.use_multi_agent = use_multi_agent
        if use_multi_agent:
            logger.info("Initializing multi-agent research system")
            self.task_queue = TaskQueue()
            # Agents will be created per research session
        else:
            logger.info("Using traditional single-agent research system")

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
            # If using multi-agent system, use that implementation
            if self.use_multi_agent:
                return await self._conduct_research_multi_agent(
                    query=query,
                    sources=sources,
                    template_id=template_id,
                    max_results_per_source=max_results_per_source,
                    research_id=research_id,
                    vector_namespace=vector_namespace,
                )

            # Otherwise use the traditional implementation
            return await self._conduct_research_traditional(
                query=query,
                sources=sources,
                template_id=template_id,
                max_results_per_source=max_results_per_source,
                research_id=research_id,
                vector_namespace=vector_namespace,
            )

        except Exception as e:
            error_msg = f"Error during research: {str(e)}"
            logger.error(error_msg)

            # Update status to error
            if research_id in self.ongoing_research:
                self.ongoing_research[research_id]["status"] = "error"
                self.ongoing_research[research_id]["error"] = error_msg
                self.ongoing_research[research_id]["end_time"] = time.time()

            # Return error result
            return create_empty_result(research_id, query, error_msg)

    async def _conduct_research_multi_agent(
        self,
        query: str,
        sources: Optional[List[str]],
        template_id: Optional[str],
        max_results_per_source: int,
        research_id: str,
        vector_namespace: str,
    ) -> Dict[str, Any]:
        """
        Conduct research using the multi-agent architecture.

        Args:
            Same as conduct_research

        Returns:
            Research results
        """
        logger.info(f"Using multi-agent approach for research ID: {research_id}")

        # Create context manager for this research session
        context_manager = ResearchContextManager(research_id)

        # Store initial context
        await context_manager.set("research_id", research_id)
        await context_manager.set("query", query)
        await context_manager.set("vector_namespace", vector_namespace)
        await context_manager.set("template_id", template_id)
        await context_manager.set("sources", sources)

        # Start the task queue if not already running
        if not self.task_queue._running:
            await self.task_queue.start()

        # Create agents for this research session
        controller = ControllerAgent(
            context_manager=context_manager,
            task_queue=self.task_queue,
        )

        acquisition_agent = AcquisitionAgent(
            context_manager=context_manager,
            task_queue=self.task_queue,
            data_source_manager=self.data_source_manager,
            vector_storage=self.vector_storage,
            document_processor=self.document_processor,
        )

        # Create the other specialized agents
        analysis_agent = AnalysisAgent(
            context_manager=context_manager,
            task_queue=self.task_queue,
        )

        synthesis_agent = SynthesisAgent(
            context_manager=context_manager,
            task_queue=self.task_queue,
        )

        critique_agent = CritiqueAgent(
            context_manager=context_manager,
            task_queue=self.task_queue,
        )

        # Submit the top-level research task
        task_id = await self.task_queue.submit_task(
            task_type="research_orchestration",
            params={
                "query": query,
                "sources": sources,
                "template_id": template_id,
                "max_results_per_source": max_results_per_source,
                "research_id": research_id,
                "vector_namespace": vector_namespace,
            },
            priority=10,
        )

        logger.info(f"Submitted research orchestration task: {task_id}")

        # Wait for the task to complete
        while True:
            task = await self.task_queue.get_task(task_id)
            if not task:
                await asyncio.sleep(0.5)
                continue

            if task.status == TaskStatus.COMPLETED:
                # Task completed successfully
                result = task.result
                break
            elif task.status in [TaskStatus.FAILED, TaskStatus.CANCELED]:
                # Task failed
                error = task.error or "Research task failed without error message"
                raise ValueError(error)

            # Still in progress, wait a bit
            await asyncio.sleep(0.5)

        # Make sure result exists and has a minimum structure
        if not result:
            result = {
                "research_id": research_id,
                "query": query,
                "report": "No report was generated by the multi-agent system.",
                "timestamp": time.time(),
                "sources": sources or [],
                "sources_used": sources or [],
                "sources_dict": {},
                "result_count": 0,
                "namespace": vector_namespace,
            }

        # Ensure required fields are present
        if "research_id" not in result:
            result["research_id"] = research_id
        if "query" not in result:
            result["query"] = query
        if "timestamp" not in result:
            result["timestamp"] = time.time()

        # Fix sources field to be a dictionary as required by the API model
        if "sources" in result and not isinstance(result["sources"], dict):
            # Convert sources list to a dictionary if needed
            if isinstance(result["sources"], list):
                sources_dict = {}
                for source in result["sources"]:
                    sources_dict[source] = 1  # Default count
                result["sources"] = sources_dict
            else:
                # Fallback with empty dict if it's neither dict nor list
                result["sources"] = {}

        # Ensure sources_dict exists
        if "sources_dict" not in result:
            result["sources_dict"] = result.get("sources", {})

        # If sources is missing, use sources_dict
        if "sources" not in result:
            result["sources"] = result.get("sources_dict", {})

        if "result_count" not in result:
            result["result_count"] = result.get("processed_docs", 0)
        if "namespace" not in result:
            result["namespace"] = vector_namespace

        # Get the final context and merge with the result
        final_context = await context_manager.get_full_context()
        result["context"] = final_context

        # Ensure result is JSON serializable by converting to dict and back
        try:
            # This will ensure everything is JSON serializable
            result = json.loads(json.dumps(self._make_serializable(result)))
        except Exception as e:
            logger.error(f"Error serializing result: {str(e)}")
            # If serialization fails, create a simplified result
            result = {
                "research_id": research_id,
                "query": query,
                "report": "Error: Could not serialize the full research result.",
                "timestamp": time.time(),
                "sources": sources or [],
                "sources_dict": {},
                "error": f"JSON serialization error: {str(e)}",
            }

        # Save to history
        await self.history_manager.save_research(result)

        # Update status
        self.ongoing_research[research_id]["status"] = "completed"
        self.ongoing_research[research_id]["end_time"] = time.time()

        # Remove from ongoing research
        self.ongoing_research.pop(research_id, None)

        return result

    async def _conduct_research_traditional(
        self,
        query: str,
        sources: Optional[List[str]],
        template_id: Optional[str],
        max_results_per_source: int,
        research_id: str,
        vector_namespace: str,
    ) -> Dict[str, Any]:
        """
        Conduct research using the traditional single-agent approach.
        This is the original implementation, kept for backward compatibility.

        Args:
            Same as conduct_research

        Returns:
            Research results
        """
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
                return create_empty_result(
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
                namespace=vector_namespace,
            )

            # Update status
            self.ongoing_research[research_id]["status"] = "documents_stored"
            logger.info(
                f"Stored {document_count} document chunks in namespace {vector_namespace}"
            )

            # Step 5: Search for relevant documents using the query
            # Use clustering-enhanced query if enough documents
            try:
                if hasattr(self.vector_storage, "query_with_clustering"):
                    logger.info("Using clustering-enhanced retrieval")
                    relevant_docs = self.vector_storage.query_with_clustering(
                        query_text=query,
                        top_k=25,  # Get top 25 results for report generation
                        namespace=vector_namespace,
                        cluster_method="kmeans",
                        num_clusters=5,
                        diversity_weight=0.3,
                    )
                    # Check if clustering worked
                    if "cluster_stats" in relevant_docs:
                        logger.info(
                            f"Successfully clustered into {relevant_docs.get('num_clusters', 0)} clusters"
                        )
                    else:
                        # Fall back to standard query if needed
                        logger.warning(
                            "Clustering failed, falling back to standard retrieval"
                        )
                        relevant_docs = self.vector_storage.query(
                            query_text=query,
                            top_k=25,
                            namespace=vector_namespace,
                        )
                else:
                    logger.info(
                        "Vector storage does not support clustering, using standard retrieval"
                    )
                    relevant_docs = self.vector_storage.query(
                        query_text=query,
                        top_k=25,
                        namespace=vector_namespace,
                    )
            except Exception as e:
                logger.warning(f"Error during cluster-enhanced retrieval: {str(e)}")
                # Fall back to standard query
                relevant_docs = self.vector_storage.query(
                    query_text=query,
                    top_k=25,
                    namespace=vector_namespace,
                )

            logger.info(
                f"Query returned {len(relevant_docs.get('matches', []))} matches from namespace {vector_namespace}"
            )

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
                "namespace": vector_namespace,
                "raw_data": raw_data,
                "relevant_docs": relevant_docs,
                "sources": sources_dict,
            }

            # Add clustering info if available
            if "cluster_stats" in relevant_docs:
                research_result["cluster_stats"] = relevant_docs["cluster_stats"]
                research_result["num_clusters"] = relevant_docs.get("num_clusters", 0)

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
            # Handle errors and add logging if needed
            logger.error(f"Error in traditional research flow: {str(e)}")
            raise  # Re-raise to be caught by the main method

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
