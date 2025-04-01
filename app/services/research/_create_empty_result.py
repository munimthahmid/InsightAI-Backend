"""
Utility function for creating empty research results
"""

from typing import Dict, Any


def create_empty_result(research_id: str, query: str, error_msg: str) -> Dict[str, Any]:
    """
    Create an empty result with error information that matches ResearchResponse structure.

    Args:
        research_id: The ID of the research
        query: The research query
        error_msg: Error message to include

    Returns:
        Dictionary with properly structured research result including required fields
    """
    return {
        "research_id": research_id,
        "query": query,
        "error": error_msg,
        "timestamp": 0,  # Will be set by the agent
        "report": f"Research could not be completed: {error_msg}",
        "sources": {},  # Empty dict instead of empty list to match ResearchResponse
        "sources_used": [],
        "result_count": 0,
    }
