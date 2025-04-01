"""
Acquisition agent responsible for gathering information from various sources.
"""

import asyncio
import json
from typing import Dict, List, Any, Optional
import time
from loguru import logger

from app.services.research.agents.base_agent import BaseAgent
from app.services.data_sources.manager import DataSourceManager
from app.services.vector_db.storage import VectorStorage
from app.services.vector_db.processors import DocumentProcessor
from app.services.research.orchestration.schemas import TaskSchema


class AcquisitionAgent(BaseAgent):
    """
    Specializes in gathering information from various sources.
    Interfaces with the DataSourceManager to collect relevant data.
    """

    def __init__(
        self,
        context_manager=None,
        task_queue=None,
        agent_id=None,
        data_source_manager=None,
        vector_storage=None,
        document_processor=None,
    ):
        """
        Initialize the acquisition agent.

        Args:
            context_manager: Shared research context manager
            task_queue: Task queue for asynchronous operations
            agent_id: Optional unique identifier
            data_source_manager: Manager for data sources
            vector_storage: Vector database interface
            document_processor: Document processor for chunking
        """
        super().__init__(
            agent_id=agent_id,
            agent_type="acquisition",
            context_manager=context_manager,
            task_queue=task_queue,
        )

        # Initialize data components
        self.data_source_manager = data_source_manager or DataSourceManager()
        self.vector_storage = vector_storage or VectorStorage()
        self.document_processor = document_processor or DocumentProcessor()

        # Register this agent's task handler
        if task_queue:
            task_queue.register_handler("acquisition_task", self.execute_task)

        logger.info(f"AcquisitionAgent initialized with ID: {self.agent_id}")

    async def execute_task(self, task: TaskSchema) -> Dict[str, Any]:
        """
        Execute a data acquisition task.

        Args:
            task: The task containing query and source parameters

        Returns:
            Dictionary with collected data and metadata
        """
        params = task.params
        query = params.get("query", "")
        sources = params.get("sources", [])
        max_results_per_source = params.get("max_results_per_source", 5)

        await self.log_activity(
            "start_data_acquisition",
            {
                "task_id": task.task_id,
                "query": query,
                "sources": sources,
                "max_results": max_results_per_source,
            },
        )

        try:
            # Get research ID from context
            research_id = await self.get_context("research_id")

            # Get vector namespace from context, or generate one
            namespace = await self.get_context("vector_namespace")
            if not namespace:
                # Normally this would be created by the controller
                await self.set_context("error", "No vector namespace found in context")
                raise ValueError("No vector namespace found in context")

            # If no sources specified, use all available
            if not sources:
                sources = self.data_source_manager.get_available_sources()
                # Ensure web search is included by default
                if "web" not in sources:
                    sources.append("web")

            # Log the sources being used
            await self.log_activity(
                "using_sources", {"sources": sources, "task_id": task.task_id}
            )

            # Fetch data from all specified sources
            await self.context_manager.update_status("collecting_data")
            raw_data = await self.data_source_manager.fetch_data_from_sources(
                query=query,
                sources=sources,
                max_results_per_source=max_results_per_source,
            )

            # Verify we got data
            if not raw_data:
                await self.context_manager.add_warning(
                    f"No data collected for query: {query}", "acquisition_agent"
                )
                return {
                    "success": False,
                    "error": "No data found from any source",
                    "processed_docs": 0,
                }

            # Process documents for vector storage
            await self.context_manager.update_status("processing_documents")
            processed_docs = []

            for source_type, docs in raw_data.items():
                for doc in docs:
                    # Add source_type to metadata
                    if "metadata" not in doc:
                        doc["metadata"] = {}
                    doc["metadata"]["source_type"] = source_type
                    doc["metadata"]["research_id"] = research_id

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

            # Store processed documents in vector database
            await self.context_manager.update_status("storing_documents")
            document_count = self.vector_storage.process_and_store(
                documents=processed_docs,
                source_type="research",
                namespace=namespace,
            )

            # Store raw data in context for other agents to use
            # Make sure it's JSON serializable first
            try:
                serialized_raw_data = self._make_serializable(raw_data)
                await self.set_context("raw_data", serialized_raw_data)

                serialized_docs = self._make_serializable(processed_docs)
                await self.set_context("processed_docs", serialized_docs)
            except Exception as e:
                logger.error(f"Error serializing data: {str(e)}")
                await self.context_manager.add_error(
                    f"Could not serialize data: {str(e)}", "acquisition_agent"
                )

            # Log success
            sources_dict = {}
            for source_type, docs in raw_data.items():
                sources_dict[source_type] = len(docs)

            await self.log_activity(
                "complete_data_acquisition",
                {
                    "task_id": task.task_id,
                    "success": True,
                    "document_count": document_count,
                    "sources_summary": sources_dict,
                },
            )

            # Return results
            return {
                "success": True,
                "query": query,
                "sources_used": sources,
                "raw_data_count": len(raw_data),
                "processed_docs": len(processed_docs),
                "vector_namespace": namespace,
                "sources": sources_dict,
            }

        except Exception as e:
            # Log failure
            await self.log_activity(
                "failed_data_acquisition",
                {"task_id": task.task_id, "error": str(e)},
            )

            # Add error to context
            await self.context_manager.add_error(str(e), "acquisition_agent")

            # Re-raise the exception for the task queue
            raise

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
