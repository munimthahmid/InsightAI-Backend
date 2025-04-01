"""
Vector database storage for embedding and retrieving research data.
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
            except Exception as index_err:
                logger.error(
                    f"Error accessing or creating Pinecone index: {str(index_err)}"
                )
                self.initialized = False
        except Exception as e:
            logger.error(f"Error initializing vector storage: {str(e)}")
            self.initialized = False

    def _check_initialized(self):
        """Check if the vector storage is initialized."""
        if not self.initialized:
            raise ValueError(
                "Vector storage not initialized. Check Pinecone API key and environment."
            )

    def _clean_metadata_for_pinecone(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean metadata to ensure it's compatible with Pinecone.

        Args:
            metadata: The metadata to clean

        Returns:
            Cleaned metadata
        """
        # Clean up the metadata by removing values that might cause issues with Pinecone
        cleaned = {}
        for k, v in metadata.items():
            # Skip complex nested objects that aren't needed for search (e.g., owner details)
            if isinstance(v, dict) or isinstance(v, list):
                continue

            # Convert all values to strings for consistency
            if v is not None:
                cleaned[k] = str(v)

        return cleaned

    def process_and_store(
        self,
        documents: List[Document],
        source_type: str,
        namespace: Optional[str] = None,
    ) -> int:
        """
        Process and store documents in the vector database.

        Args:
            documents: List of Document objects to process and store
            source_type: Type of source for metadata
            namespace: Optional namespace for grouping related vectors

        Returns:
            Number of documents processed
        """
        # If using mock storage, delegate to it
        if self.use_mock:
            return self.mock_storage.process_and_store(
                documents, source_type, namespace
            )

        self._check_initialized()

        # Ensure we have a valid namespace
        namespace = namespace or ""

        if not documents:
            logger.warning("No documents provided to process_and_store")
            return 0

        logger.info(f"Generating embeddings for {len(documents)} documents")

        # Pre-emptively ensure the namespace exists
        if namespace:
            namespace_success = self._ensure_namespace_exists(namespace)
            if not namespace_success:
                logger.warning(
                    f"Could not ensure namespace '{namespace}' exists. Will attempt to create it during vector upsert."
                )

        # Generate IDs and embeddings
        ids = []
        embeddings = []
        metadatas = []

        # Batch processing to avoid overwhelming the API
        batch_size = 100
        processed_count = 0

        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]

            # Generate embeddings for content
            texts = [doc.page_content for doc in batch]
            try:
                batch_embeddings = self.embeddings.embed_documents(texts)
                embeddings.extend(batch_embeddings)

                # Generate IDs and clean metadata
                for j, doc in enumerate(batch):
                    doc_id = str(uuid.uuid4())
                    ids.append(doc_id)

                    # Clean metadata for Pinecone
                    metadata = self._clean_metadata_for_pinecone(doc.metadata)
                    metadata["source_type"] = source_type
                    metadata["text"] = doc.page_content[
                        :1000
                    ]  # Store truncated text for retrieval
                    metadatas.append(metadata)

                processed_count += len(batch)
                logger.info(f"Successfully generated {len(batch)} embeddings")
            except Exception as e:
                logger.error(f"Error generating embeddings for batch: {str(e)}")
                # Continue with next batch

        if not embeddings:
            logger.error("Failed to generate any embeddings")
            return 0

        # Store vectors in batches
        upsert_batch_size = 100
        vectors_stored = 0

        # Get initial stats for verification
        try:
            before_stats = self.index.describe_index_stats()
            logger.info(f"Index stats before upsert: {before_stats}")
        except Exception as e:
            logger.warning(f"Could not get index stats before upsert: {str(e)}")
            before_stats = {"total_vector_count": 0, "namespaces": {}}

        # Process in batches
        for i in range(0, len(ids), upsert_batch_size):
            end_idx = min(i + upsert_batch_size, len(ids))
            batch_ids = ids[i:end_idx]
            batch_embeddings = embeddings[i:end_idx]
            batch_metadatas = metadatas[i:end_idx]

            # Prepare vectors for upsert
            batch = []
            for j in range(len(batch_ids)):
                batch.append(
                    {
                        "id": batch_ids[j],
                        "values": batch_embeddings[j],
                        "metadata": batch_metadatas[j],
                    }
                )

            # Upsert to Pinecone
            try:
                logger.info(
                    f"Upserting {len(batch)} vectors to namespace '{namespace}'"
                )
                self.index.upsert(
                    vectors=batch,
                    namespace=namespace,
                )
                vectors_stored += len(batch)
                logger.info(
                    f"Successfully stored batch {i//upsert_batch_size + 1} ({len(batch)} vectors)"
                )
            except Exception as e:
                logger.error(f"Error upserting batch to Pinecone: {str(e)}")

        # Verify the upsert by checking index stats
        try:
            # Allow a delay for Pinecone to process
            logger.info("Waiting for Pinecone to update index stats...")
            time.sleep(3)

            after_stats = self.index.describe_index_stats()

            # Check if the namespace exists in the index stats
            namespaces = after_stats.get("namespaces", {})

            if namespace in namespaces:
                logger.info(
                    f"Verification successful: Namespace '{namespace}' found with {namespaces[namespace].get('vector_count', 0)} vectors"
                )
            else:
                # Namespace still not visible, but vectors might have been added
                namespace_list = list(namespaces.keys())
                logger.warning(
                    f"Verification partial: Namespace '{namespace}' not visible in index stats. Available namespaces: {namespace_list}"
                )

                # Check if total vectors increased
                before_count = before_stats.get("total_vector_count", 0)
                after_count = after_stats.get("total_vector_count", 0)

                if after_count > before_count:
                    logger.info(
                        f"Total vector count increased from {before_count} to {after_count}, vectors were likely added"
                    )
                else:
                    logger.warning(
                        "No increase in total vector count, vectors may not have been added correctly"
                    )

                # One last attempt to ensure namespace exists
                if namespace:
                    logger.info(
                        f"Making final attempt to ensure namespace '{namespace}' is registered..."
                    )
                    self._ensure_namespace_exists(namespace, max_retries=1)
        except Exception as e:
            logger.warning(f"Error verifying upsert: {str(e)}")

        logger.info(
            f"Completed storing {vectors_stored}/{len(ids)} vectors from {source_type}"
        )
        return vectors_stored

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        namespace: Optional[str] = None,
        filter_dict: Optional[Dict[str, Any]] = None,
        source_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Query the vector database for similar documents.

        Args:
            query_text: Text to search for
            top_k: Number of results to return
            namespace: Namespace to search in
            filter_dict: Optional filter for metadata
            source_types: Optional list of source types to filter by

        Returns:
            Dictionary with matches
        """
        # If using mock storage, delegate to it
        if self.use_mock:
            return self.mock_storage.query(
                query_text, top_k, namespace, filter_dict, source_types
            )

        self._check_initialized()

        # Use default namespace if not provided
        namespace = namespace or ""

        logger.info(
            f"Querying vector store with: text='{query_text[:30]}...', top_k={top_k}, namespace='{namespace}'"
        )

        # First ensure the namespace exists before querying
        namespace_verified = False
        if namespace:
            namespace_exists = self._ensure_namespace_exists(namespace, max_retries=2)
            if namespace_exists:
                namespace_verified = True
                logger.info(f"Verified namespace '{namespace}' exists")
            else:
                logger.warning(
                    f"Namespace '{namespace}' could not be verified to exist, but will attempt query anyway"
                )

        # Create embedding for query text
        query_embedding = self.embeddings.embed_query(query_text)

        # Apply source_type filter if provided
        filter_conditions = {}
        if filter_dict:
            filter_conditions.update(filter_dict)
        if source_types:
            filter_conditions["source_type"] = {"$in": source_types}

        # Execute the query with retry logic
        max_retries = 2
        retry_count = 0
        query_results = {"matches": []}

        while retry_count <= max_retries:
            try:
                results = self.index.query(
                    vector=query_embedding,
                    top_k=top_k,
                    namespace=namespace,
                    filter=filter_conditions if filter_conditions else None,
                    include_metadata=True,
                )

                # Process and log results
                matches = results.get("matches", [])
                match_count = len(matches)

                if match_count > 0:
                    logger.info(
                        f"Query returned {match_count} matches in namespace '{namespace}'"
                    )
                    query_results = results
                    break  # Success, exit the retry loop
                else:
                    logger.warning(
                        f"Attempt {retry_count+1}: No matches found for query in namespace '{namespace}'"
                    )

                    # Check if we should retry
                    if retry_count < max_retries:
                        # Get index stats for debugging
                        try:
                            stats = self.index.describe_index_stats()
                            logger.info(f"Index stats: {stats}")

                            # Namespace not found in stats but we should retry
                            if namespace and namespace not in stats.get(
                                "namespaces", {}
                            ):
                                logger.warning(
                                    f"Namespace '{namespace}' not found in index stats, will retry query"
                                )
                                # Wait before retry with exponential backoff
                                wait_time = 2**retry_count
                                logger.info(
                                    f"Waiting {wait_time} seconds before retry..."
                                )
                                time.sleep(wait_time)
                                retry_count += 1
                                continue
                        except Exception as stats_err:
                            logger.error(
                                f"Error fetching index stats: {str(stats_err)}"
                            )

                    # No more retries or namespace exists but no matches
                    query_results = results
                    break

            except Exception as e:
                logger.error(
                    f"Error querying vector store (attempt {retry_count+1}): {str(e)}"
                )

                # Check if we should retry
                if retry_count < max_retries:
                    wait_time = 2**retry_count
                    logger.info(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                    retry_count += 1
                else:
                    break

        # If we still got no results, do a final check on index stats
        if not query_results.get("matches", []):
            try:
                stats = self.index.describe_index_stats()
                total_vectors = stats.get("total_vector_count", 0)
                namespaces = stats.get("namespaces", {})

                if namespace and namespace in namespaces:
                    ns_vectors = namespaces[namespace].get("vector_count", 0)
                    logger.info(
                        f"Namespace '{namespace}' exists with {ns_vectors} vectors, but no matches found"
                    )
                else:
                    logger.warning(
                        f"Final check: Namespace '{namespace}' not found in index stats"
                    )
                    if namespaces:
                        all_ns = list(namespaces.keys())
                        logger.info(f"Available namespaces: {all_ns}")

                        # If we can't find our namespace but vectors were stored, try querying without namespace
                        if len(all_ns) > 0 and not namespace_verified:
                            logger.warning(
                                f"Trying fallback query without namespace specification"
                            )
                            try:
                                fallback_results = self.index.query(
                                    vector=query_embedding,
                                    top_k=top_k,
                                    namespace="",  # Try empty namespace as fallback
                                    filter=(
                                        filter_conditions if filter_conditions else None
                                    ),
                                    include_metadata=True,
                                )

                                fallback_matches = fallback_results.get("matches", [])
                                if fallback_matches:
                                    logger.info(
                                        f"Fallback query returned {len(fallback_matches)} matches"
                                    )
                                    query_results = fallback_results
                            except Exception as fallback_err:
                                logger.error(
                                    f"Error during fallback query: {str(fallback_err)}"
                                )
            except Exception as e:
                logger.error(f"Error during final stats check: {str(e)}")

        return query_results

    def delete_namespace(self, namespace: str) -> bool:
        """
        Delete all vectors in a namespace.

        Args:
            namespace: The namespace to delete

        Returns:
            True if successful, False otherwise
        """
        self._check_initialized()

        try:
            self.index.delete(delete_all=True, namespace=namespace)
            logger.info(f"Deleted all vectors in namespace: {namespace}")
            return True
        except Exception as e:
            logger.error(f"Error deleting namespace {namespace}: {str(e)}")
            return False

    def _ensure_namespace_exists(self, namespace: str, max_retries: int = 3) -> bool:
        """
        Ensure a namespace exists in Pinecone by creating a small test vector if needed.

        Args:
            namespace: The namespace to verify/create
            max_retries: Maximum number of retry attempts

        Returns:
            bool: True if namespace exists or was created, False if failed
        """
        if not namespace:
            logger.warning("Empty namespace provided to _ensure_namespace_exists")
            return False

        # First check if namespace already exists
        try:
            stats = self.index.describe_index_stats()
            namespaces = stats.get("namespaces", {})

            if namespace in namespaces:
                logger.info(
                    f"Namespace '{namespace}' already exists with {namespaces[namespace].get('vector_count', 0)} vectors"
                )
                return True
        except Exception as e:
            logger.warning(f"Error checking namespace existence: {str(e)}")
            # Continue with creation attempt

        # Namespace doesn't exist or couldn't be verified, try to create it with a minimal vector
        retry_delay = 1  # seconds

        for retry in range(max_retries):
            try:
                logger.info(
                    f"Attempt {retry+1}: Creating namespace '{namespace}' with test vector..."
                )

                # Create a test vector with some non-zero values (Pinecone requires at least one non-zero value)
                # Initialize with small random values instead of zeros
                test_values = [0.001] * settings.DIMENSION
                # Add some variation to ensure uniqueness
                for i in range(10):
                    random_idx = random.randint(0, settings.DIMENSION - 1)
                    test_values[random_idx] = 0.1 + (
                        random.random() * 0.9
                    )  # Random value between 0.1 and 1.0

                test_vector = {
                    "id": f"ns-init-{namespace}",
                    "values": test_values,
                    "metadata": {
                        "namespace_init": True,
                        "created_at": time.time(),
                        "source_type": "system",
                    },
                }

                # Upsert test vector to initialize namespace
                self.index.upsert(vectors=[test_vector], namespace=namespace)

                # Wait for namespace to be reflected in stats
                logger.info(
                    f"Waiting {retry_delay} seconds for namespace to be registered..."
                )
                time.sleep(retry_delay)

                # Verify namespace now exists
                stats = self.index.describe_index_stats()
                if namespace in stats.get("namespaces", {}):
                    logger.info(
                        f"Successfully created and verified namespace '{namespace}'"
                    )
                    return True

                logger.warning(
                    f"Namespace '{namespace}' still not found after creation attempt {retry+1}"
                )
                retry_delay *= 2  # Exponential backoff

            except Exception as e:
                logger.warning(
                    f"Error creating namespace '{namespace}' (attempt {retry+1}): {str(e)}"
                )
                time.sleep(retry_delay)
                retry_delay *= 2

        logger.error(
            f"Failed to ensure namespace '{namespace}' exists after {max_retries} attempts"
        )
        return False
