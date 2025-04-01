"""
Base class for all data sources.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import httpx
from loguru import logger


class DataSourceBase(ABC):
    """Abstract base class for all data sources."""

    def __init__(self, client: Optional[httpx.AsyncClient] = None):
        """
        Initialize the data source.

        Args:
            client: Optional HTTP client to use for requests.
                   If not provided, a new client will be created.
        """
        self.client = client or httpx.AsyncClient(timeout=60.0)
        self.source_name = self._get_source_name()

    def _get_source_name(self) -> str:
        """
        Get the name of this data source.
        Default implementation extracts from class name.

        Returns:
            The name of the data source in lowercase
        """
        class_name = self.__class__.__name__
        if class_name.endswith("DataSource"):
            # Convert camel case to snake case and remove "DataSource" suffix
            name = class_name[:-10]  # Remove "DataSource"
            # Insert underscore before capital letters and convert to lowercase
            import re

            name = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
            return name
        return class_name.lower()

    @abstractmethod
    async def fetch_data(
        self, query: str, max_results: int = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch data from the source.

        Args:
            query: The search query
            max_results: Maximum number of results to return

        Returns:
            List of dictionaries containing the retrieved data
        """
        pass

    async def aclose(self):
        """Close the HTTP client."""
        if self.client:
            await self.client.aclose()
