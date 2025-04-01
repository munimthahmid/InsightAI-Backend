"""
Data sources module for fetching research data from various external APIs.
"""

from app.services.data_sources.base import DataSourceBase
from app.services.data_sources.sources import (
    ArxivDataSource,
    NewsDataSource,
    GitHubDataSource,
    WikipediaDataSource,
    SemanticScholarDataSource,
)
from app.services.data_sources.manager import DataSourceManager

__all__ = [
    "DataSourceBase",
    "ArxivDataSource",
    "NewsDataSource",
    "GitHubDataSource",
    "WikipediaDataSource",
    "SemanticScholarDataSource",
    "DataSourceManager",
]
