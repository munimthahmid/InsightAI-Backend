"""
Implementation of specific data sources.
"""

import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from urllib.parse import quote
import time
import asyncio

from app.core.config import settings
from app.services.data_sources.base import DataSourceBase
from loguru import logger


class ArxivDataSource(DataSourceBase):
    """Data source for ArXiv papers."""

    async def fetch_data(
        self, query: str, max_results: int = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch papers from ArXiv API.

        Args:
            query: The search query
            max_results: Maximum number of results to return

        Returns:
            List of dictionaries containing paper details
        """
        if not max_results:
            max_results = settings.MAX_RESULTS_PER_SOURCE

        logger.info(f"Fetching ArXiv papers for query: {query} (max: {max_results})")

        # URL encode the query
        encoded_query = quote(f"all:{query}")

        base_url = "http://export.arxiv.org/api/query"
        params = {
            "search_query": encoded_query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }

        response = await self.client.get(base_url, params=params)

        if response.status_code != 200:
            logger.error(f"ArXiv API error: {response.text}")
            return []

        # Parse XML response
        try:
            root = ET.fromstring(response.text)

            # Define namespace for ArXiv API
            namespace = {
                "atom": "http://www.w3.org/2005/Atom",
                "arxiv": "http://arxiv.org/schemas/atom",
            }

            papers = []

            for entry in root.findall("atom:entry", namespace):
                title = entry.find("atom:title", namespace).text.strip()
                summary = entry.find("atom:summary", namespace).text.strip()
                published = entry.find("atom:published", namespace).text.strip()

                # Get authors
                authors = []
                for author in entry.findall("atom:author", namespace):
                    name = author.find("atom:name", namespace).text.strip()
                    authors.append(name)

                # Get URL
                url = None
                for link in entry.findall("atom:link", namespace):
                    if link.get("title") == "pdf":
                        url = link.get("href")
                        break

                if not url:
                    # Try to find alternate URL
                    for link in entry.findall("atom:link", namespace):
                        if link.get("rel") == "alternate":
                            url = link.get("href")
                            break

                # Get categories
                categories = []
                for category in entry.findall("atom:category", namespace):
                    categories.append(category.get("term"))

                # Get DOI if available
                doi = None
                journal_ref = None
                try:
                    # Try to get DOI and journal reference
                    for extra in entry.findall("arxiv:*", namespace):
                        if extra.tag.endswith("doi"):
                            doi = extra.text
                        elif extra.tag.endswith("journal_ref"):
                            journal_ref = extra.text
                except:
                    pass

                papers.append(
                    {
                        "title": title,
                        "summary": summary,
                        "authors": authors,
                        "published": published,
                        "url": url,
                        "categories": categories,
                        "doi": doi,
                        "journal_ref": journal_ref,
                    }
                )

            logger.info(f"Found {len(papers)} ArXiv papers for query: {query}")
            return papers
        except Exception as e:
            logger.error(f"Error parsing ArXiv response: {str(e)}")
            return []


class NewsDataSource(DataSourceBase):
    """Data source for News API articles."""

    async def fetch_data(
        self, query: str, max_results: int = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch news articles from News API.

        Args:
            query: The search query
            max_results: Maximum number of results to return

        Returns:
            List of dictionaries containing article details
        """
        if not settings.NEWS_API_KEY:
            logger.warning("News API key not provided, skipping news articles")
            return []

        if not max_results:
            max_results = settings.MAX_RESULTS_PER_SOURCE

        logger.info(f"Fetching news articles for query: {query} (max: {max_results})")

        base_url = "https://newsapi.org/v2/everything"
        # Get articles from last 7 days instead of 30 to avoid hitting API limits
        from_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

        params = {
            "q": query,
            "from": from_date,
            "sortBy": "relevancy",
            "pageSize": max_results,
            "language": "en",
            "apiKey": settings.NEWS_API_KEY,
        }

        try:
            logger.debug(f"Sending request to News API with params: {params}")
            response = await self.client.get(base_url, params=params)

            if response.status_code != 200:
                logger.error(
                    f"News API error: {response.status_code} - {response.text}"
                )
                # Check for specific error codes
                if response.status_code == 401:
                    logger.error("News API authentication failed - check your API key")
                elif response.status_code == 429:
                    logger.error("News API rate limit exceeded - try again later")
                elif (
                    response.status_code == 426
                    and "too far in the past" in response.text
                ):
                    # Retry with yesterday's date
                    logger.warning(
                        "Date range too far in past, retrying with yesterday's date"
                    )
                    params["from"] = (datetime.now() - timedelta(days=1)).strftime(
                        "%Y-%m-%d"
                    )
                    retry_response = await self.client.get(base_url, params=params)
                    if retry_response.status_code == 200:
                        response = (
                            retry_response  # Replace response with successful retry
                        )
                    else:
                        return []
                return []

            data = response.json()
            if "articles" not in data:
                logger.error(f"Unexpected response format from News API: {data}")
                return []

            articles = []
            for article in data["articles"]:
                # Skip articles without meaningful content
                if not article.get("content") or not article.get("description"):
                    continue

                # Clean up source information
                source = article.get("source", {}).get("name", "Unknown Source")

                # Add to results
                articles.append(
                    {
                        "title": article.get("title", ""),
                        "description": article.get("description", ""),
                        "content": article.get("content", ""),
                        "url": article.get("url", ""),
                        "source": source,
                        "publishedAt": article.get("publishedAt", ""),
                    }
                )

            logger.info(f"Found {len(articles)} news articles for query: {query}")
            return articles

        except Exception as e:
            logger.error(f"Error fetching news articles: {str(e)}")
            return []


class GitHubDataSource(DataSourceBase):
    """Data source for GitHub repositories."""

    async def fetch_data(
        self, query: str, max_results: int = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch GitHub repositories via GitHub API.

        Args:
            query: The search query
            max_results: Maximum number of results to return

        Returns:
            List of dictionaries containing repository details
        """
        if not settings.GITHUB_TOKEN:
            logger.warning("GitHub token not provided, skipping GitHub repositories")
            return []

        if not max_results:
            max_results = settings.MAX_RESULTS_PER_SOURCE

        logger.info(
            f"Fetching GitHub repositories for query: {query} (max: {max_results})"
        )

        base_url = "https://api.github.com/search/repositories"
        headers = {"Authorization": f"token {settings.GITHUB_TOKEN}"}
        params = {
            "q": query,
            "sort": "stars",
            "order": "desc",
            "per_page": max_results,
        }

        try:
            response = await self.client.get(base_url, headers=headers, params=params)

            if response.status_code != 200:
                logger.error(
                    f"GitHub API error: {response.status_code} - {response.text}"
                )
                return []

            data = response.json()
            if "items" not in data:
                logger.error(f"Unexpected response format from GitHub API: {data}")
                return []

            repos = []
            for repo in data["items"]:
                # Get topics (languages and actual topics)
                topics = repo.get("topics", [])

                # Create clean output
                repos.append(
                    {
                        "name": repo.get("name", ""),
                        "full_name": repo.get("full_name", ""),
                        "description": repo.get("description", ""),
                        "html_url": repo.get("html_url", ""),
                        "language": repo.get("language", ""),
                        "stargazers_count": repo.get("stargazers_count", 0),
                        "watchers_count": repo.get("watchers_count", 0),
                        "forks_count": repo.get("forks_count", 0),
                        "topics": topics,
                        "created_at": repo.get("created_at", ""),
                        "updated_at": repo.get("updated_at", ""),
                        "owner": repo.get("owner", {}),
                    }
                )

            logger.info(f"Found {len(repos)} GitHub repositories for query: {query}")
            return repos

        except Exception as e:
            logger.error(f"Error fetching GitHub repositories: {str(e)}")
            return []


class WikipediaDataSource(DataSourceBase):
    """Data source for Wikipedia information."""

    async def fetch_data(
        self, query: str, max_results: int = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch information from Wikipedia API.

        Args:
            query: The search query
            max_results: Maximum number of results to return

        Returns:
            List of dictionaries containing Wikipedia page details
        """
        if not max_results:
            max_results = settings.MAX_RESULTS_PER_SOURCE

        logger.info(
            f"Fetching Wikipedia information for query: {query} (max: {max_results})"
        )

        # First, search for pages
        search_url = "https://en.wikipedia.org/w/api.php"
        search_params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srlimit": max_results,
        }

        try:
            search_response = await self.client.get(search_url, params=search_params)

            if search_response.status_code != 200:
                logger.error(
                    f"Wikipedia search API error: {search_response.status_code} - {search_response.text}"
                )
                return []

            search_data = search_response.json()
            if (
                "query" not in search_data
                or "search" not in search_data["query"]
                or not search_data["query"]["search"]
            ):
                logger.info(f"No Wikipedia pages found for query: {query}")
                return []

            # Get page IDs and titles for the top results
            page_results = []
            for i, result in enumerate(search_data["query"]["search"]):
                if i >= max_results:
                    break

                page_id = result["pageid"]
                title = result["title"]

                # Now get full content for each page
                content_params = {
                    "action": "query",
                    "prop": "extracts|categories",
                    "exintro": True,  # Just get the intro to keep size reasonable
                    "explaintext": True,  # Get plain text instead of HTML
                    "pageids": page_id,
                    "format": "json",
                    "cllimit": 10,  # Number of categories to retrieve
                }

                content_response = await self.client.get(
                    search_url, params=content_params
                )

                if content_response.status_code != 200:
                    logger.error(
                        f"Wikipedia content API error: {content_response.status_code} - {content_response.text}"
                    )
                    continue

                content_data = content_response.json()
                if "query" not in content_data or "pages" not in content_data["query"]:
                    continue

                # Extract page content
                page_data = content_data["query"]["pages"][str(page_id)]
                content = page_data.get("extract", "")

                # Extract categories
                categories = []
                if "categories" in page_data:
                    for cat in page_data["categories"]:
                        cat_title = cat.get("title", "")
                        if cat_title.startswith("Category:"):
                            categories.append(
                                cat_title[9:]
                            )  # Remove 'Category:' prefix

                # Create URL for the page
                url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"

                page_results.append(
                    {
                        "title": title,
                        "content": content,
                        "url": url,
                        "pageid": page_id,
                        "categories": categories,
                    }
                )

                # Add a small delay to avoid hitting rate limits
                await asyncio.sleep(0.1)

            logger.info(f"Found {len(page_results)} Wikipedia pages for query: {query}")
            return page_results

        except Exception as e:
            logger.error(f"Error fetching Wikipedia information: {str(e)}")
            return []


class SemanticScholarDataSource(DataSourceBase):
    """Data source for Semantic Scholar papers."""

    async def fetch_data(
        self, query: str, max_results: int = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch papers from Semantic Scholar API.

        Args:
            query: The search query
            max_results: Maximum number of results to return

        Returns:
            List of dictionaries containing paper details
        """
        if not max_results:
            max_results = settings.MAX_RESULTS_PER_SOURCE

        logger.info(
            f"Fetching Semantic Scholar papers for query: {query} (max: {max_results})"
        )

        base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": query,
            "limit": max_results,
            "fields": "title,abstract,url,venue,year,authors,openAccessPdf",
        }

        try:
            response = await self.client.get(base_url, params=params)

            if response.status_code != 200:
                logger.error(
                    f"Semantic Scholar API error: {response.status_code} - {response.text}"
                )
                return []

            data = response.json()
            if "data" not in data:
                logger.error(
                    f"Unexpected response format from Semantic Scholar API: {data}"
                )
                return []

            papers = []
            for paper in data["data"]:
                # Sometimes abstract is None
                abstract = paper.get("abstract", "") or ""

                # Extract author names
                authors = []
                if "authors" in paper and paper["authors"]:
                    for author in paper["authors"]:
                        if "name" in author:
                            authors.append(author["name"])

                # Get citation count
                citation_count = 0
                # No longer requesting citation count due to API restrictions

                # Get PDF URL if available
                pdf_url = None
                if "openAccessPdf" in paper and paper["openAccessPdf"]:
                    pdf_url = paper["openAccessPdf"].get("url", None)

                papers.append(
                    {
                        "title": paper.get("title", ""),
                        "abstract": abstract,
                        "url": paper.get("url", ""),
                        "venue": paper.get("venue", ""),
                        "year": paper.get("year", ""),
                        "authors": authors,
                        "citation_count": citation_count,
                        "pdf_url": pdf_url,
                    }
                )

            logger.info(
                f"Found {len(papers)} Semantic Scholar papers for query: {query}"
            )
            return papers

        except Exception as e:
            logger.error(f"Error fetching Semantic Scholar papers: {str(e)}")
            return []


class WebSearchSource(DataSourceBase):
    """Data source for web search results."""

    async def fetch_data(
        self, query: str, max_results: int = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch web search results.

        Args:
            query: The search query
            max_results: Maximum number of results to return

        Returns:
            List of dictionaries containing search results
        """
        if not max_results:
            max_results = settings.MAX_RESULTS_PER_SOURCE

        logger.info(f"Fetching web results for query: {query} (max: {max_results})")

        try:
            # Simulate web search with some realistic data structure
            # In a real implementation, this would call a web search service API

            # Create sample results - in production replace with actual API call
            results = [
                {
                    "title": f"Web Result for {query} - Example Source 1",
                    "link": "https://example.com/result1",
                    "snippet": f"This is a sample search result about {query}. This would contain actual snippets from websites in a real implementation.",
                },
                {
                    "title": f"Documentation about {query} - Example Source 2",
                    "link": "https://docs.example.com/topics/query",
                    "snippet": f"Documentation and examples related to {query}. Includes tutorials, guides and code samples.",
                },
                {
                    "title": f"{query} Research Papers - Academic Resource",
                    "link": "https://academic.example.org/papers",
                    "snippet": f"Collection of research papers and articles about {query} and related topics.",
                },
                {
                    "title": f"Latest News on {query} - News Source",
                    "link": "https://news.example.com/technology/query",
                    "snippet": f"Recent developments and news related to {query}. Updated daily with the latest information.",
                },
            ]

            # Post-process the results
            processed_results = self._post_process_data(results[:max_results], query)
            logger.info(f"Processed {len(processed_results)} web search results")

            return processed_results

        except Exception as e:
            logger.error(f"Error fetching web search results: {str(e)}")
            return []

    def _post_process_data(self, data, query):
        """Post-process web search results."""
        results = []
        for item in data:
            # Extract the main content
            title = item.get("title", "Untitled Web Page")
            url = item.get("link", "")
            snippet = item.get("snippet", "")

            # Create a meaningful metadata structure
            metadata = {
                "title": title,
                "url": url,
                "source_type": "web",
                "query": query,
            }

            # Ensure the URL is present in both the metadata and directly
            # in the document for better visibility to LLM
            content = f"Title: {title}\nURL: {url}\n\n{snippet}"

            # Create the final document
            results.append({"page_content": content, "metadata": metadata})

        return results
