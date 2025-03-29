from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from pinecone import Pinecone, ServerlessSpec
import requests.exceptions
import urllib3.exceptions
from typing import List, Dict, Any, Optional, Tuple, Union
import uuid
import time
from loguru import logger

from app.core.config import settings


class VectorStorage:
    """Handles vector embeddings and storage with Pinecone."""

    def __init__(self):
        """Initialize the vector storage with OpenAI embeddings and Pinecone."""
        # Use a model compatible with the existing Pinecone index dimensions (1536)
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=settings.OPENAI_API_KEY,
            model="text-embedding-ada-002",  # Use model with 1536 dimensions to match existing index
        )

        # Initialize Pinecone
        if not (settings.PINECONE_API_KEY and settings.PINECONE_ENVIRONMENT):
            logger.warning(
                "Pinecone API key or environment not provided, vector storage will not work"
            )
            self.initialized = False
            return

        try:
            # Initialize Pinecone client with the new API
            self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)

            try:
                # List existing indexes
                indexes = self.pc.list_indexes()
                index_names = [index.name for index in indexes]

                # Create index if it doesn't exist
                if settings.INDEX_NAME not in index_names:
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
                    # Wait for index to be ready
                    time.sleep(1)

                self.index = self.pc.Index(settings.INDEX_NAME)
                self.initialized = True
                logger.info("Vector storage initialized successfully")
            except (
                requests.exceptions.ConnectionError,
                urllib3.exceptions.NewConnectionError,
            ) as conn_err:
                logger.error(f"Connection error to Pinecone: {str(conn_err)}")
                logger.error(
                    "Please check your internet connection or Pinecone service status"
                )
                self.initialized = False
            except Exception as e:
                logger.error(f"Error accessing Pinecone indexes: {str(e)}")
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

    def _prepare_documents(
        self, data: List[Dict[str, Any]], source_type: str
    ) -> List[Document]:
        """
        Convert API data to Document objects based on source type.

        Args:
            data: List of data from API
            source_type: Type of source (arxiv, news, github, wikipedia, semantic_scholar)

        Returns:
            List of Document objects
        """
        documents = []

        if source_type == "arxiv":
            for paper in data:
                # Create a more detailed text representation with additional metadata
                text = (
                    f"Title: {paper.get('title', '')}\n\n"
                    f"Authors: {', '.join(paper.get('authors', []))}\n\n"
                    f"Categories: {', '.join(paper.get('categories', []))}\n\n"
                    f"Summary: {paper.get('summary', '')}"
                )

                # Include DOI and journal reference if available
                if paper.get("doi"):
                    text += f"\n\nDOI: {paper.get('doi')}"
                if paper.get("journal_ref"):
                    text += f"\n\nJournal Reference: {paper.get('journal_ref')}"

                # Create metadata
                metadata = {
                    "source": "arxiv",
                    "title": paper.get("title", ""),
                    "authors": ", ".join(paper.get("authors", [])),
                    "url": paper.get("url", ""),
                    "published": paper.get("published", ""),
                    "categories": ", ".join(paper.get("categories", [])),
                    "doi": paper.get("doi", ""),
                    "journal_ref": paper.get("journal_ref", ""),
                }

                doc = Document(page_content=text, metadata=metadata)
                documents.append(doc)

        elif source_type == "news":
            for article in data:
                # Combine title, description and content
                text = (
                    f"Title: {article.get('title', '')}\n\n"
                    f"Source: {article.get('source', '')}\n\n"
                    f"Published: {article.get('publishedAt', '')}\n\n"
                    f"Description: {article.get('description', '')}\n\n"
                    f"Content: {article.get('content', '')}"
                )

                # Create metadata
                metadata = {
                    "source": "news",
                    "title": article.get("title", ""),
                    "news_source": article.get("source", ""),
                    "url": article.get("url", ""),
                    "published": article.get("publishedAt", ""),
                }

                doc = Document(page_content=text, metadata=metadata)
                documents.append(doc)

        elif source_type == "github":
            for repo in data:
                # Get language with default value to avoid null
                language = repo.get("language") or "Not specified"

                # Create text from repo details with more structured information
                topics = ", ".join(repo.get("topics", []))
                owner_info = f"Owner: {repo.get('owner', {}).get('login', 'Unknown')}"
                created = f"Created: {repo.get('created_at', '')}"
                updated = f"Updated: {repo.get('updated_at', '')}"

                text = (
                    f"Repository: {repo.get('full_name', '')}\n\n"
                    f"Description: {repo.get('description', '')}\n\n"
                    f"Language: {language}\n\n"
                    f"Topics: {topics}\n\n"
                    f"Stars: {repo.get('stargazers_count', 0)}\n\n"
                    f"{owner_info}\n\n{created}\n\n{updated}"
                )

                # Create metadata
                metadata = {
                    "source": "github",
                    "name": repo.get("name", ""),
                    "full_name": repo.get("full_name", ""),
                    "url": repo.get("html_url", ""),
                    "stars": repo.get("stargazers_count", 0),
                    "language": language,
                    "created_at": repo.get("created_at", ""),
                    "updated_at": repo.get("updated_at", ""),
                    "owner": repo.get("owner", {}).get("login", ""),
                }

                doc = Document(page_content=text, metadata=metadata)
                documents.append(doc)

        elif source_type == "wikipedia":
            for page in data:
                # Create more structured text with sections for better context
                title_section = f"Title: {page.get('title', '')}"
                content_section = f"Content: {page.get('content', '')}"

                # Add category information if available
                categories = page.get("categories", [])
                category_section = (
                    "Categories: " + ", ".join(categories) if categories else ""
                )

                # Add related links for additional context
                links = page.get("links", [])
                links_section = (
                    "Related topics: " + ", ".join(links[:10]) if links else ""
                )

                text = f"{title_section}\n\n{category_section}\n\n{links_section}\n\n{content_section}"

                # Create metadata with enriched information
                metadata = {
                    "source": "wikipedia",
                    "title": page.get("title", ""),
                    "url": page.get("url", ""),
                    "page_id": page.get("page_id", ""),
                    "categories": page.get("categories", []),
                    "links": page.get("links", [])[:10],  # Include top 10 links
                }

                doc = Document(page_content=text, metadata=metadata)
                documents.append(doc)

        elif source_type == "semantic_scholar":
            for paper in data:
                # Create a detailed text representation with academic information
                title = f"Title: {paper.get('title', '')}"
                authors = f"Authors: {', '.join(paper.get('authors', []))}"
                year = f"Year: {paper.get('year', '')}" if paper.get("year") else ""
                venue = f"Venue: {paper.get('venue', '')}" if paper.get("venue") else ""

                citation_info = (
                    f"Citations: {paper.get('citation_count', 0)} total, "
                    f"{paper.get('influential_citation_count', 0)} influential"
                )

                fields = (
                    f"Fields of Study: {', '.join(paper.get('fields_of_study', []))}"
                )
                abstract = f"Abstract: {paper.get('abstract', '')}"

                text = f"{title}\n\n{authors}\n\n{year}\n\n{venue}\n\n{citation_info}\n\n{fields}\n\n{abstract}"

                # Create metadata
                metadata = {
                    "source": "semantic_scholar",
                    "title": paper.get("title", ""),
                    "authors": ", ".join(paper.get("authors", [])),
                    "year": paper.get("year", ""),
                    "venue": paper.get("venue", ""),
                    "url": paper.get("url", ""),
                    "citation_count": paper.get("citation_count", 0),
                    "influential_citation_count": paper.get(
                        "influential_citation_count", 0
                    ),
                    "fields_of_study": paper.get("fields_of_study", []),
                }

                doc = Document(page_content=text, metadata=metadata)
                documents.append(doc)

        return documents

    def _get_optimal_chunk_settings(self, source_type: str) -> Tuple[int, int]:
        """
        Get optimal chunking settings based on source type for better retrieval.

        Args:
            source_type: Type of source

        Returns:
            Tuple of (chunk_size, chunk_overlap)
        """
        # Different source types have different optimal chunking strategies
        if source_type == "wikipedia":
            # Wikipedia articles are long and have well-structured sections
            return 1500, 300
        elif source_type in ["arxiv", "semantic_scholar"]:
            # Scientific papers need larger chunks to maintain context
            return 2000, 400
        elif source_type == "news":
            # News articles are shorter and more focused
            return 1000, 200
        elif source_type == "github":
            # GitHub descriptions are typically short
            return 800, 150
        else:
            # Default settings
            return 1000, 200

    def process_and_store(
        self,
        data: List[Dict[str, Any]],
        source_type: str,
        namespace: Optional[str] = None,
    ) -> int:
        """
        Process data and store embeddings in Pinecone.

        Args:
            data: List of data from API
            source_type: Type of source (arxiv, news, github, wikipedia, semantic_scholar)
            namespace: Optional namespace for the vectors

        Returns:
            Number of chunks stored in the vector database
        """
        self._check_initialized()

        if not data:
            logger.info(f"No data to process for source: {source_type}")
            return 0

        # Convert data to documents
        documents = self._prepare_documents(data, source_type)

        # Get optimal chunking settings for this source type
        chunk_size, chunk_overlap = self._get_optimal_chunk_settings(source_type)

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

        # Prepare data for Pinecone
        pinecone_data = []
        for i, (split, embedding) in enumerate(zip(splits, embeddings)):
            # Clean metadata to ensure no null values that would cause Pinecone to reject
            metadata = self._clean_metadata_for_pinecone(split.metadata.copy())

            # Add the text to the metadata for retrieval
            metadata["text"] = split.page_content
            # Add chunk identifier
            metadata["chunk_id"] = i
            # Add source type for easier filtering
            metadata["source_type"] = source_type

            # Create a unique ID
            vector_id = f"{source_type}-{namespace if namespace else uuid.uuid4()}-{i}"

            pinecone_data.append(
                {"id": vector_id, "values": embedding, "metadata": metadata}
            )

        # Upsert to Pinecone in batches
        batch_size = 100
        for i in range(0, len(pinecone_data), batch_size):
            batch = pinecone_data[i : i + batch_size]
            self.index.upsert(vectors=batch, namespace=namespace)

        logger.info(
            f"Stored {len(pinecone_data)} vector embeddings for source: {source_type}"
        )
        return len(pinecone_data)

    def _clean_metadata_for_pinecone(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean metadata to ensure all values are compatible with Pinecone.
        Pinecone accepts strings, numbers, booleans, or lists of strings.

        Args:
            metadata: The metadata dictionary to clean

        Returns:
            Cleaned metadata dictionary
        """
        cleaned = {}
        for key, value in metadata.items():
            if value is None:
                # Replace None values with empty strings
                cleaned[key] = ""
            elif isinstance(value, (str, int, float, bool)):
                # These types are directly supported
                cleaned[key] = value
            elif isinstance(value, list):
                # For lists, ensure all elements are strings
                cleaned[key] = [str(item) if item is not None else "" for item in value]
            else:
                # Convert other types to strings
                cleaned[key] = str(value)

        return cleaned

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
        self._check_initialized()

        # Create the query embedding
        query_embedding = self.embeddings.embed_query(query_text)

        # Apply source type filters if specified
        if source_types and not filter_dict:
            filter_dict = {"source_type": {"$in": source_types}}
        elif source_types and filter_dict:
            # Merge with existing filter
            filter_dict = {
                "$and": [filter_dict, {"source_type": {"$in": source_types}}]
            }

        # Query Pinecone
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            namespace=namespace,
            filter=filter_dict,
        )

        logger.info(
            f"Retrieved {len(results.get('matches', []))} results for query: {query_text}"
        )
        return results

    def delete_namespace(self, namespace: str) -> bool:
        """
        Delete a namespace from Pinecone.

        Args:
            namespace: Namespace to delete

        Returns:
            True if successful, False otherwise
        """
        self._check_initialized()

        try:
            self.index.delete(delete_all=True, namespace=namespace)
            logger.info(f"Deleted namespace: {namespace}")
            return True
        except Exception as e:
            logger.error(f"Error deleting namespace {namespace}: {str(e)}")
            return False
