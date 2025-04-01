"""
Manager for data sources that fetches from multiple APIs.
"""

import httpx
import asyncio
from typing import Dict, List, Any, Optional

from app.core.config import settings
from app.services.data_sources.base import DataSourceBase
from app.services.data_sources.sources import (
    ArxivDataSource,
    NewsDataSource,
    GitHubDataSource,
    WikipediaDataSource,
    SemanticScholarDataSource,
    WebSearchSource,
)
from loguru import logger


class DataSourceManager:
    """Manager for fetching data from various sources."""

    def __init__(self):
        """Initialize the data sources manager with an HTTP client."""
        self.client = httpx.AsyncClient(
            timeout=60.0
        )  # Increased timeout for external APIs

        # Initialize data sources with shared client for efficiency
        self.sources = {
            "arxiv": ArxivDataSource(self.client),
            "news": NewsDataSource(self.client),
            "github": GitHubDataSource(self.client),
            "wikipedia": WikipediaDataSource(self.client),
            "semantic_scholar": SemanticScholarDataSource(self.client),
            "web": WebSearchSource(self.client),
        }

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    async def fetch_source(
        self, source_name: str, query: str, max_results: int = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch data from a specific source.

        Args:
            source_name: Name of the source to fetch from
            query: The search query
            max_results: Maximum number of results to return

        Returns:
            List of dictionaries containing the data
        """
        if source_name not in self.sources:
            logger.error(f"Unknown data source: {source_name}")
            return []

        try:
            return await self.sources[source_name].fetch_data(query, max_results)
        except Exception as e:
            logger.error(f"Error fetching from {source_name}: {str(e)}")
            return []

    async def fetch_all_sources(
        self, query: str, max_results_per_source: int = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Fetch data from all sources in parallel.

        Args:
            query: The search query
            max_results_per_source: Maximum number of results to fetch per source

        Returns:
            Dictionary mapping source names to their results
        """
        if not max_results_per_source:
            max_results_per_source = settings.MAX_RESULTS_PER_SOURCE

        # Set up tasks for each source
        tasks = {
            source_name: asyncio.create_task(
                self.fetch_source(source_name, query, max_results_per_source)
            )
            for source_name in self.sources
        }

        # Wait for all tasks to complete
        done, pending = await asyncio.wait(tasks.values())

        # Get results from each source
        results = {}
        for source_name, task in tasks.items():
            try:
                results[source_name] = task.result()
            except Exception as e:
                logger.error(f"Error getting results from {source_name}: {str(e)}")
                results[source_name] = []

        logger.info(
            f"Fetched data from all sources for query: {query} "
            f"(total items: {sum(len(items) for items in results.values())})"
        )

        return results

    async def fetch_wikipedia_info(
        self, query: str, max_results: int = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch information from Wikipedia.

        Args:
            query: The search query
            max_results: Maximum number of results to return

        Returns:
            List of dictionaries containing the data
        """
        return await self.fetch_source("wikipedia", query, max_results)

    async def fetch_news_articles(
        self, query: str, max_results: int = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch news articles.

        Args:
            query: The search query
            max_results: Maximum number of results to return

        Returns:
            List of dictionaries containing the data
        """
        return await self.fetch_source("news", query, max_results)

    def get_available_sources(self) -> List[str]:
        """
        Get a list of all available data sources.

        Returns:
            List of source names
        """
        return list(self.sources.keys())

    async def fetch_data_from_sources(
        self, query: str, sources: List[str], max_results_per_source: int = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Fetch data from specified sources only.

        Args:
            query: The search query
            sources: List of source names to fetch from
            max_results_per_source: Maximum results per source

        Returns:
            Dictionary mapping source names to their results
        """
        if not max_results_per_source:
            max_results_per_source = settings.MAX_RESULTS_PER_SOURCE

        # Validate sources
        valid_sources = [s for s in sources if s in self.sources]
        if len(valid_sources) < len(sources):
            invalid = set(sources) - set(valid_sources)
            logger.warning(f"Unknown data sources: {', '.join(invalid)}")

        # Set up tasks for each valid source
        tasks = {
            source_name: asyncio.create_task(
                self.fetch_source(source_name, query, max_results_per_source)
            )
            for source_name in valid_sources
        }

        # Wait for all tasks to complete
        done, pending = await asyncio.wait(tasks.values()) if tasks else (set(), set())

        # Get results from each source
        results = {}
        for source_name, task in tasks.items():
            try:
                results[source_name] = task.result()
            except Exception as e:
                logger.error(f"Error getting results from {source_name}: {str(e)}")
                results[source_name] = []

        logger.info(
            f"Fetched data from specified sources for query: {query} "
            f"(total items: {sum(len(items) for items in results.values())})"
        )

        return results

    def _register_sources(self):
        """Register all available data sources."""
        from app.services.data_sources.sources import (
            ArxivDataSource,
            NewsDataSource,
            GitHubDataSource,
            WikipediaDataSource,
            SemanticScholarDataSource,
            WebSearchSource,
        )

        self.register_source("arxiv", ArxivDataSource())
        self.register_source("news", NewsDataSource())
        self.register_source("github", GitHubDataSource())
        self.register_source("wikipedia", WikipediaDataSource())
        self.register_source("semantic_scholar", SemanticScholarDataSource())
        self.register_source("web", WebSearchSource())
