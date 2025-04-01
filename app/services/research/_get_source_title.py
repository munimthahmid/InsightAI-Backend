"""
Utility function to extract meaningful titles from source chunks
"""

from typing import Dict, Any


def get_source_title(
    chunk: Dict[str, Any], fallback_title: str = "Source document"
) -> str:
    """
    Extract an appropriate title from a chunk based on source type.
    Ensures a meaningful title is always returned instead of "Untitled Source".

    Args:
        chunk: Document chunk with metadata
        fallback_title: Default title to use if no meaningful title can be extracted

    Returns:
        A meaningful title for the source
    """
    metadata = chunk.get("metadata", {})
    source_type = metadata.get("source_type", "unknown")

    # Try to extract meaningful title based on source type
    if source_type == "arxiv":
        title = metadata.get("title")
        if title:
            return title
        return "ArXiv Paper"

    elif source_type == "news":
        title = metadata.get("title")
        source = metadata.get("news_source", "")
        if title:
            if source:
                return f"{title} ({source})"
            return title
        if source:
            return f"News from {source}"
        return "News Article"

    elif source_type == "github":
        repo_name = metadata.get("full_name") or metadata.get("name")
        if repo_name:
            return f"GitHub: {repo_name}"
        return "GitHub Repository"

    elif source_type == "wikipedia":
        title = metadata.get("title")
        if title:
            return f"Wikipedia: {title}"
        return "Wikipedia Article"

    elif source_type == "semantic_scholar":
        title = metadata.get("title")
        if title:
            return title
        return "Research Paper"

    elif source_type == "web":
        title = metadata.get("title")
        url = metadata.get("url", "")
        if title:
            return title
        if url:
            # Extract domain name from URL
            from urllib.parse import urlparse

            domain = urlparse(url).netloc
            if domain:
                return f"Web content from {domain}"
        return "Web Page"

    # Generic sources
    title = metadata.get("title")
    if title:
        return title

    # Last resort fallback
    return fallback_title
