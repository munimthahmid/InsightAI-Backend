"""
Document preparation functionality for vector database storage.
Handles conversion of data from different sources into document objects.
"""

from langchain.docstore.document import Document
from typing import List, Dict, Any, Tuple
from loguru import logger


class DocumentPreparation:
    """Prepares documents for embedding and vector storage."""

    def __init__(self):
        """Initialize the document preparation service."""
        pass

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

    def get_optimal_chunk_settings(self, source_type: str) -> Tuple[int, int]:
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

    def clean_metadata_for_pinecone(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
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
