"""
Utility functions for the research agent.
"""

import time
from typing import Dict, Any, List


def create_empty_result(research_id: str, query: str, error_msg: str) -> Dict[str, Any]:
    """
    Create an empty result with error information.

    Args:
        research_id: The ID of the research
        query: The original query
        error_msg: Error message explaining why research failed

    Returns:
        Dictionary with research result structure but empty content
    """
    return {
        "research_id": research_id,
        "query": query,
        "error": error_msg,
        "timestamp": time.time(),
        "report": f"Research could not be completed: {error_msg}",
        "sources_used": [],
        "result_count": 0,
        "sources": {},  # Empty dict to match the ResearchResponse model
    }


def get_source_counts(raw_data: Dict[str, List]) -> Dict[str, int]:
    """
    Extract source counts from raw data.

    Args:
        raw_data: Dictionary with source types as keys and lists of results as values

    Returns:
        Dictionary with source types as keys and count of results as values
    """
    return {source_type: len(items) for source_type, items in raw_data.items()}
