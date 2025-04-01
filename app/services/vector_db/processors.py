"""
Document processing for vector database storage.
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger
import re


class DocumentProcessor:
    """Processes documents for storage in the vector database."""

    def __init__(self):
        """Initialize the document processor."""
        pass

    def _get_optimal_chunk_settings(self, source_type: str) -> Tuple[int, int]:
        """
        Get optimal chunk size and overlap settings based on source type.

        Args:
            source_type: Type of source (arxiv, news, github, wikipedia, semantic_scholar)

        Returns:
            Tuple of (chunk_size, chunk_overlap)
        """
        # Different content types benefit from different chunking strategies
        if source_type == "arxiv":
            # Academic papers need slightly larger chunks for context
            return (1500, 200)
        elif source_type == "news":
            # News articles can use smaller chunks
            return (1000, 100)
        elif source_type == "github":
            # Repository data is more structured, can use smaller chunks
            return (800, 100)
        elif source_type == "wikipedia":
            # Wikipedia articles benefit from medium-sized chunks
            return (1200, 150)
        elif source_type == "semantic_scholar":
            # Similar to ArXiv, academic content needs larger chunks
            return (1500, 200)
        else:
            # Default settings
            return (1000, 150)

    def prepare_documents(
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
                    f"Categories: {', '.join(categories)}" if categories else ""
                )

                # Combine all sections
                text = f"{title_section}\n\n{content_section}"
                if category_section:
                    text += f"\n\n{category_section}"

                # Create metadata
                metadata = {
                    "source": "wikipedia",
                    "title": page.get("title", ""),
                    "url": page.get("url", ""),
                    "pageid": page.get("pageid", ""),
                    "categories": ", ".join(categories),
                }

                doc = Document(page_content=text, metadata=metadata)
                documents.append(doc)

        elif source_type == "semantic_scholar":
            for paper in data:
                # Create structured text with sections
                title_section = f"Title: {paper.get('title', '')}"

                authors = paper.get("authors", [])
                authors_section = (
                    f"Authors: {', '.join(authors)}" if authors else "Authors: Unknown"
                )

                abstract = paper.get("abstract", "")
                abstract_section = f"Abstract: {abstract}" if abstract else ""

                venue = paper.get("venue", "")
                year = paper.get("year", "")
                publication_info = (
                    f"Publication: {venue} ({year})" if venue and year else ""
                )

                citations = f"Citations: {paper.get('citation_count', 0)}"

                # Combine all sections
                text_parts = [title_section, authors_section]
                if abstract_section:
                    text_parts.append(abstract_section)
                if publication_info:
                    text_parts.append(publication_info)
                text_parts.append(citations)

                text = "\n\n".join(text_parts)

                # Create metadata
                metadata = {
                    "source": "semantic_scholar",
                    "title": paper.get("title", ""),
                    "authors": ", ".join(authors),
                    "url": paper.get("url", ""),
                    "venue": venue,
                    "year": year,
                    "citation_count": paper.get("citation_count", 0),
                    "pdf_url": paper.get("pdf_url", ""),
                }

                doc = Document(page_content=text, metadata=metadata)
                documents.append(doc)

        else:
            logger.warning(f"Unknown source type: {source_type}")

        return documents

    def chunk_documents(
        self, documents: List[Document], source_type: str
    ) -> List[Document]:
        """
        Split documents into chunks appropriate for embedding.

        Args:
            documents: List of Document objects
            source_type: Type of source for optimal chunk settings

        Returns:
            List of chunked Document objects
        """
        if not documents:
            return []

        # Get optimal settings for this source type
        chunk_size, chunk_overlap = self._get_optimal_chunk_settings(source_type)

        # Create text splitter with source-appropriate settings
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

        # Split documents into chunks
        chunked_docs = text_splitter.split_documents(documents)
        logger.debug(
            f"Split {len(documents)} {source_type} documents into {len(chunked_docs)} chunks"
        )

        return chunked_docs

    def process_document(self, document: Dict[str, Any]) -> List[Document]:
        """
        Process a single document for vector storage.

        Args:
            document: Document with content and metadata

        Returns:
            List of processed document chunks
        """
        # Extract source type from metadata
        source_type = document.get("metadata", {}).get("source_type", "unknown")

        # Check if we already have a Document object
        if isinstance(document, Document):
            documents = [document]
        else:
            # Convert to Document object first
            # Check if there's content directly in the document
            metadata = document.get("metadata", {}).copy()

            # Ensure URL is preserved in metadata
            # Try multiple potential locations for URL
            if "url" not in metadata:
                # Check if URL is directly in the document
                if "url" in document:
                    metadata["url"] = document["url"]
                elif "link" in document:
                    metadata["url"] = document["link"]
                elif "html_url" in document:
                    metadata["url"] = document["html_url"]

            # Ensure the title is preserved
            if "title" not in metadata and "title" in document:
                metadata["title"] = document["title"]

            # Include source type in metadata if not already there
            if "source_type" not in metadata and source_type != "unknown":
                metadata["source_type"] = source_type

            # For web search results, ensure URL is properly extracted from content
            if source_type == "web" and "url" not in metadata:
                if "page_content" in document:
                    content = document["page_content"]
                    url_match = re.search(r"URL: (https?://\S+)", content)
                    if url_match:
                        metadata["url"] = url_match.group(1)

            # Log metadata for debugging
            logger.debug(f"Document metadata after processing: {metadata}")

            # Get the content
            if "content" in document:
                text = document["content"]
            elif "page_content" in document:
                text = document["page_content"]

                # For text content, preserve URL in the content if available
                if "url" in metadata and "URL:" not in text:
                    url = metadata["url"]
                    if text:
                        text = f"URL: {url}\n\n{text}"
                    else:
                        text = f"URL: {url}"
            else:
                # Default handling with what we have
                text = str(document)

            documents = [Document(page_content=text, metadata=metadata)]

        # Chunk document
        chunked_docs = self.chunk_documents(documents, source_type)

        # Ensure each chunk preserves the URL
        for chunk in chunked_docs:
            if "url" in document.get("metadata", {}):
                chunk.metadata["url"] = document["metadata"]["url"]
            elif "url" in document:
                chunk.metadata["url"] = document["url"]

        return chunked_docs
