"""
Vector database storage for embedding and retrieving research data.
This file serves as the main interface for vector database operations.
"""

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from pinecone import Pinecone, ServerlessSpec
import requests.exceptions
import urllib3.exceptions
from typing import List, Dict, Any, Optional, Tuple, Union
import uuid
import time
from loguru import logger
import json
import os
import random

from app.core.config import settings
from app.services.vector_db.document_preparation import DocumentPreparation
from app.services.vector_db.vector_operations import VectorOperations


# Define a simple MockVectorStorage for local testing when Pinecone fails
class MockVectorStorage:
    """A simple in-memory vector storage for testing without Pinecone."""

    def __init__(self):
        """Initialize mock storage."""
        self.storage = {}
        self.initialized = True
        logger.warning(
            "Using MockVectorStorage because Pinecone initialization failed."
        )

    def process_and_store(self, documents, source_type, namespace=None):
        """Store documents in mock storage."""
        namespace = namespace or "default"
        if namespace not in self.storage:
            self.storage[namespace] = []

        # Store documents with their content and metadata
        for doc in documents:
            self.storage[namespace].append(
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "source_type": source_type,
                }
            )

        count = len(documents)
        logger.info(f"[MOCK] Stored {count} documents in namespace '{namespace}'")
        return count

    def query(
        self, query_text, top_k=5, namespace=None, filter_dict=None, source_types=None
    ):
        """Query documents from mock storage (very basic matching)."""
        namespace = namespace or "default"
        if namespace not in self.storage:
            logger.warning(f"[MOCK] Namespace '{namespace}' not found in storage")
            return {"matches": []}

        # Very simple "matching" - just return all documents in the namespace
        docs = self.storage.get(namespace, [])
        matches = []

        # Take the top_k docs or all if fewer
        for i, doc in enumerate(docs[:top_k]):
            matches.append(
                {
                    "id": f"mock-{i}",
                    "score": 0.99,  # Fake high similarity score
                    "metadata": doc["metadata"],
                    "values": [],  # No actual vectors
                }
            )

        logger.info(
            f"[MOCK] Returned {len(matches)} matches from namespace '{namespace}'"
        )
        return {"matches": matches}

    def delete_namespace(self, namespace):
        """Delete a namespace from mock storage."""
        if namespace in self.storage:
            del self.storage[namespace]
            logger.info(f"[MOCK] Deleted namespace '{namespace}'")
            return True
        return False


class VectorStorage:
    """Handles vector embeddings and storage with Pinecone."""

    def __init__(self):
        """Initialize the vector storage with OpenAI embeddings and Pinecone."""
        self.use_mock = False
        self.document_preparation = DocumentPreparation()

        # Use a model compatible with the existing Pinecone index dimensions (1536)
        try:
            self.embeddings = OpenAIEmbeddings(
                openai_api_key=settings.OPENAI_API_KEY,
                model="text-embedding-ada-002",  # Use model with 1536 dimensions to match existing index
            )
        except Exception as e:
            logger.error(f"Error initializing OpenAI embeddings: {str(e)}")
            self.initialized = False
            # Fall back to mock storage
            self.mock_storage = MockVectorStorage()
            self.use_mock = True
            return

        # Initialize Pinecone
        if not (settings.PINECONE_API_KEY and settings.PINECONE_ENVIRONMENT):
            logger.warning(
                "Pinecone API key or environment not provided, falling back to mock storage"
            )
            self.initialized = False
            self.mock_storage = MockVectorStorage()
            self.use_mock = True
            return

        try:
            # Initialize Pinecone client with the new API
            logger.info(
                f"Initializing Pinecone with environment: {settings.PINECONE_ENVIRONMENT}"
            )
            self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)

            try:
                # Test connection by listing indexes
                logger.info("Testing Pinecone connection by listing indexes...")
                indexes = self.pc.list_indexes()

                if not indexes:
                    logger.info("No existing indexes found in Pinecone.")
                else:
                    index_names = [index.name for index in indexes]
                    logger.info(f"Found existing indexes: {', '.join(index_names)}")

                # Create index if it doesn't exist
                if not indexes or settings.INDEX_NAME not in [
                    index.name for index in indexes
                ]:
                    logger.info(f"Creating Pinecone index: {settings.INDEX_NAME}")

                    # Create a ServerlessSpec using settings
                    spec = ServerlessSpec(
                        cloud=settings.INDEX_SPEC["cloud"],
                        region=settings.INDEX_SPEC["region"],
                    )

                    self.pc.create_index(
                        name=settings.INDEX_NAME,
                        dimension=settings.DIMENSION,
                        metric="cosine",
                        spec=spec,
                    )
                    # Wait for index to be initialized
                    logger.info(
                        f"Waiting for index {settings.INDEX_NAME} to initialize..."
                    )
                    time.sleep(20)  # Give more time for index to become ready
                    logger.info(f"Index creation wait time completed")

                # Connect to the index
                self.index = self.pc.Index(settings.INDEX_NAME)

                # Initialize vector operations with our components
                self.vector_operations = VectorOperations(
                    embeddings=self.embeddings,
                    index=self.index,
                    document_preparation=self.document_preparation,
                )

                # Verify the index is working by checking stats
                try:
                    stats = self.index.describe_index_stats()
                    logger.info(f"Successfully connected to index. Stats: {stats}")

                    # Log the total vector count
                    vector_count = stats.get("total_vector_count", 0)
                    logger.info(f"Total vectors in index: {vector_count}")

                    # Log namespaces
                    namespaces = stats.get("namespaces", {})
                    if namespaces:
                        logger.info(f"Existing namespaces: {list(namespaces.keys())}")
                    else:
                        logger.info("No existing namespaces found in index")

                    self.initialized = True
                    logger.info("Vector storage initialized successfully")
                except Exception as stats_err:
                    logger.error(f"Error getting index stats: {str(stats_err)}")
                    self.initialized = False
                    self.mock_storage = MockVectorStorage()
                    self.use_mock = True
            except Exception as index_err:
                logger.error(
                    f"Error accessing or creating Pinecone index: {str(index_err)}"
                )
                self.initialized = False
                self.mock_storage = MockVectorStorage()
                self.use_mock = True
        except Exception as e:
            logger.error(f"Error initializing vector storage: {str(e)}")
            self.initialized = False
            self.mock_storage = MockVectorStorage()
            self.use_mock = True

    def _check_initialized(self):
        """Check if the vector storage is initialized."""
        if not self.initialized:
            raise ValueError(
                "Vector storage not initialized. Check Pinecone API key and environment."
            )

    def process_and_store(
        self,
        documents: List[Dict[str, Any]],
        source_type: str,
        namespace: Optional[str] = None,
    ) -> int:
        """
        Process data and store embeddings in Pinecone.

        Args:
            documents: List of document data
            source_type: Type of source
            namespace: Optional namespace for the vectors

        Returns:
            Number of chunks stored in the vector database
        """
        if self.use_mock:
            return self.mock_storage.process_and_store(
                documents, source_type, namespace
            )

        # Prepare documents if they aren't already Document objects
        prepared_docs = []
        for doc in documents:
            if isinstance(doc, Document):
                prepared_docs.append(doc)
            else:
                # If it's a dict with page_content and metadata
                if (
                    isinstance(doc, dict)
                    and "page_content" in doc
                    and "metadata" in doc
                ):
                    prepared_docs.append(
                        Document(
                            page_content=doc["page_content"], metadata=doc["metadata"]
                        )
                    )

        # Use vector operations to process and store
        return self.vector_operations.process_and_store(
            documents=prepared_docs, source_type=source_type, namespace=namespace
        )

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        namespace: Optional[str] = None,
        filter_dict: Optional[Dict[str, Any]] = None,
        source_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Query the vector store with advanced filtering options.

        Args:
            query_text: The query text
            top_k: Number of top results to return
            namespace: Optional namespace for the query
            filter_dict: Optional filter for the query
            source_types: Optional list of source types to filter by

        Returns:
            Query results from Pinecone
        """
        if self.use_mock:
            return self.mock_storage.query(
                query_text, top_k, namespace, filter_dict, source_types
            )

        # Use vector operations to query
        return self.vector_operations.query(
            query_text=query_text,
            top_k=top_k,
            namespace=namespace,
            filter_dict=filter_dict,
            source_types=source_types,
        )

    def delete_namespace(self, namespace: str) -> bool:
        """
        Delete a namespace from Pinecone.

        Args:
            namespace: Namespace to delete

        Returns:
            True if successful, False otherwise
        """
        if self.use_mock:
            return self.mock_storage.delete_namespace(namespace)

        # Use vector operations to delete namespace
        return self.vector_operations.delete_namespace(namespace)
