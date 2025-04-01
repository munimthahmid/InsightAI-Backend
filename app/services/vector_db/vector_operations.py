"""
Vector operations for the vector database.
Handles embedding, storing, and querying vector data.
"""

import uuid
import time
from typing import List, Dict, Any, Optional
from loguru import logger
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


class VectorOperations:
    """Handles vector embedding operations and interactions with the vector database."""

    def __init__(self, embeddings=None, index=None, document_preparation=None):
        """
        Initialize vector operations with required components.

        Args:
            embeddings: Language model embeddings instance
            index: Vector database index
            document_preparation: Document preparation service
        """
        self.embeddings = embeddings
        self.index = index
        self.document_preparation = document_preparation
        self.initialized = False if embeddings is None or index is None else True

    def process_and_store(
        self,
        documents: List[Document],
        source_type: str,
        namespace: Optional[str] = None,
    ) -> int:
        """
        Process documents and store embeddings in the vector database.

        Args:
            documents: List of Document objects
            source_type: Type of source (arxiv, news, github, wikipedia, semantic_scholar)
            namespace: Optional namespace for the vectors

        Returns:
            Number of chunks stored in the vector database
        """
        if not self.initialized:
            logger.error("Vector operations not initialized")
            return 0

        if not documents:
            logger.info(f"No documents to process for source: {source_type}")
            return 0

        # Ensure namespace is provided
        if not namespace:
            namespace = str(uuid.uuid4())
            logger.info(f"No namespace provided, generated: {namespace}")

        # Get optimal chunking settings for this source type
        chunk_size, chunk_overlap = (
            self.document_preparation.get_optimal_chunk_settings(source_type)
        )

        # Split text into chunks with source-specific settings
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        splits = text_splitter.split_documents(documents)

        if not splits:
            logger.warning(f"No text chunks generated for source: {source_type}")
            return 0

        logger.info(f"Generated {len(splits)} text chunks for source: {source_type}")

        # Create embeddings for the splits
        embeddings = self.embeddings.embed_documents([s.page_content for s in splits])

        # Prepare data for vector database
        vector_data = []
        for i, (split, embedding) in enumerate(zip(splits, embeddings)):
            # Clean metadata to ensure no null values that would cause Pinecone to reject
            metadata = self.document_preparation.clean_metadata_for_pinecone(
                split.metadata.copy()
            )

            # Add the text to the metadata for retrieval
            metadata["text"] = split.page_content
            # Add chunk identifier
            metadata["chunk_id"] = i
            # Add source type for easier filtering
            metadata["source_type"] = source_type

            # Create a unique ID
            vector_id = f"{source_type}-{namespace}-{i}"

            vector_data.append(
                {"id": vector_id, "values": embedding, "metadata": metadata}
            )

        # Try a two-phase approach - first try the shared namespace for better reuse
        # then the unique namespace if needed
        try:
            # First add to a shared research namespace
            shared_namespace = "shared_research"
            logger.info(
                f"Storing vectors in shared namespace '{shared_namespace}' first"
            )

            # Only store a subset in shared to avoid bloat
            shared_vectors = vector_data[: min(10, len(vector_data))]
            if shared_vectors:
                self.index.upsert(vectors=shared_vectors, namespace=shared_namespace)
                logger.info(
                    f"Successfully stored {len(shared_vectors)} vectors in shared namespace"
                )

            # Wait to ensure shared namespace persists
            time.sleep(2)
        except Exception as e:
            logger.warning(f"Could not store vectors in shared namespace: {str(e)}")

        # Upsert to main unique namespace in batches
        batch_size = 50  # Reduced batch size for better reliability
        try:
            logger.info(
                f"Storing {len(vector_data)} vectors in namespace '{namespace}'"
            )
            for i in range(0, len(vector_data), batch_size):
                batch = vector_data[i : i + batch_size]
                self.index.upsert(vectors=batch, namespace=namespace)
                logger.info(
                    f"Stored batch {i//batch_size + 1}/{(len(vector_data)-1)//batch_size + 1} ({len(batch)} vectors)"
                )

                # Add small delay between batches to help Pinecone keep up
                if i + batch_size < len(vector_data):
                    time.sleep(0.5)

            logger.info(f"Completed storing vectors in namespace '{namespace}'")

            # Give Pinecone time to update its internal state
            logger.info("Waiting for Pinecone to update index stats (5 seconds)...")
            time.sleep(5)
        except Exception as e:
            logger.error(f"Error storing vectors in namespace '{namespace}': {str(e)}")

        # Verify the namespace contains vectors
        try:
            stats = self.index.describe_index_stats()
            namespaces = stats.get("namespaces", {})
            if namespace in namespaces:
                ns_count = namespaces[namespace].get("vector_count", 0)
                logger.info(
                    f"✓ Verified namespace '{namespace}' contains {ns_count} vectors"
                )
            else:
                logger.warning(
                    f"! Verification failed: Namespace '{namespace}' not found in index stats after waiting"
                )
                logger.info(f"Available namespaces: {list(namespaces.keys())}")

                # Check the shared namespace as fallback
                if shared_namespace in namespaces:
                    shared_count = namespaces[shared_namespace].get("vector_count", 0)
                    logger.info(
                        f"✓ But shared namespace '{shared_namespace}' contains {shared_count} vectors"
                    )
        except Exception as e:
            logger.warning(f"Could not verify namespace '{namespace}': {str(e)}")

        # Store the shared namespace for use in queries
        self.last_shared_namespace = shared_namespace

        return len(vector_data)

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
            Query results from vector database
        """
        if not self.initialized:
            logger.error("Vector operations not initialized")
            return {"matches": []}

        # Create the query embedding first to reuse in different attempts
        logger.info(f"Creating embedding for query: '{query_text[:50]}...' (truncated)")
        query_embedding = self.embeddings.embed_query(query_text)

        # Apply source type filters if specified
        if source_types and not filter_dict:
            filter_dict = {"source_type": {"$in": source_types}}
            logger.info(f"Applied source type filter: {source_types}")
        elif source_types and filter_dict:
            # Merge with existing filter
            filter_dict = {
                "$and": [filter_dict, {"source_type": {"$in": source_types}}]
            }
            logger.info(f"Applied combined filter with source types: {source_types}")

        # Multi-stage querying approach - first try specifically provided namespace
        all_results = {"matches": []}
        namespaces_to_try = []

        # Add provided namespace as primary
        if namespace:
            namespaces_to_try.append(namespace)

        # Check index stats to help with query planning
        try:
            stats = self.index.describe_index_stats()
            all_namespaces = list(stats.get("namespaces", {}).keys())

            # Log available namespaces for debugging
            logger.info(f"Available namespaces: {all_namespaces[:5]}... (truncated)")

            # Make sure the shared namespace is included in our list
            if (
                hasattr(self, "last_shared_namespace")
                and self.last_shared_namespace not in namespaces_to_try
            ):
                namespaces_to_try.append(self.last_shared_namespace)

            # Add "shared_research" namespace which is our backup
            if "shared_research" not in namespaces_to_try:
                namespaces_to_try.append("shared_research")

            # If we still don't have a namespace to try, use the first available one
            if not namespaces_to_try and all_namespaces:
                namespaces_to_try.append(all_namespaces[0])
                logger.info(
                    f"No valid namespace found, using first available: {all_namespaces[0]}"
                )
        except Exception as e:
            logger.warning(f"Error checking index stats: {str(e)}")
            # Fallback to using shared_research if no others are available
            if "shared_research" not in namespaces_to_try:
                namespaces_to_try.append("shared_research")

        # Try each namespace in order
        for ns_to_query in namespaces_to_try:
            logger.info(f"Attempting to query namespace: '{ns_to_query}'")

            try:
                # Query Pinecone with the current namespace
                results = self.index.query(
                    vector=query_embedding,
                    top_k=top_k,
                    include_metadata=True,
                    namespace=ns_to_query,
                    filter=filter_dict,
                )

                match_count = len(results.get("matches", []))
                logger.info(
                    f"Retrieved {match_count} results from namespace '{ns_to_query}'"
                )

                # If we found matches, use these results
                if match_count > 0:
                    logger.info(
                        f"✓ Successfully found {match_count} matches in namespace '{ns_to_query}'"
                    )
                    return results

                # If not, try without the filter as a fallback
                if filter_dict:
                    logger.info(
                        f"No results with filter, trying namespace '{ns_to_query}' without filter"
                    )
                    results_no_filter = self.index.query(
                        vector=query_embedding,
                        top_k=top_k,
                        include_metadata=True,
                        namespace=ns_to_query,
                    )

                    match_count_no_filter = len(results_no_filter.get("matches", []))
                    logger.info(
                        f"Retrieved {match_count_no_filter} results without filter"
                    )

                    if match_count_no_filter > 0:
                        logger.info(
                            f"✓ Found {match_count_no_filter} matches without filter"
                        )
                        return results_no_filter

            except Exception as e:
                logger.warning(f"Error querying namespace '{ns_to_query}': {str(e)}")

        # If we get here, no results were found in any namespace
        logger.warning(f"No results found in any namespace for query: {query_text}")
        return {"matches": []}

    def delete_namespace(self, namespace: str) -> bool:
        """
        Delete a namespace from the vector database.

        Args:
            namespace: Namespace to delete

        Returns:
            True if successful, False otherwise
        """
        if not self.initialized:
            logger.error("Vector operations not initialized")
            return False

        try:
            self.index.delete(delete_all=True, namespace=namespace)
            logger.info(f"Deleted namespace: {namespace}")
            return True
        except Exception as e:
            logger.error(f"Error deleting namespace {namespace}: {str(e)}")
            return False
