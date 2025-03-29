import httpx
import logging
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional
from urllib.parse import quote
import asyncio

from app.core.config import settings
from loguru import logger


class DataSources:
    """Handle fetching data from various APIs."""

    def __init__(self):
        self.client = httpx.AsyncClient(
            timeout=60.0
        )  # Increased timeout for external APIs

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()

    async def fetch_arxiv_papers(
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

    async def fetch_news_articles(
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
                        logger.error(
                            f"News API retry failed: {retry_response.status_code}"
                        )
                        return []
                else:
                    return []

            try:
                data = response.json()
                # Log complete response for debugging
                logger.debug(f"News API response: {data}")

                # Check for API-specific error messages
                if "status" in data and data["status"] != "ok":
                    logger.error(
                        f"News API returned error status: {data.get('status')} - {data.get('message', 'No message')}"
                    )
                    return []

                articles = data.get("articles", [])

                if not articles:
                    logger.warning(f"News API returned 0 articles for query: {query}")
                    # Check if there's a message explaining why
                    if "message" in data:
                        logger.warning(f"News API message: {data['message']}")
                    return []

                # Clean up the articles - some may contain None values
                cleaned_articles = []
                for article in articles:
                    if article.get("title") and article.get("url"):
                        # Ensure all required fields exist
                        article["description"] = article.get("description", "")
                        article["content"] = article.get("content", "")
                        article["publishedAt"] = article.get("publishedAt", "")
                        article["source"] = article.get("source", {}).get(
                            "name", "Unknown Source"
                        )
                        cleaned_articles.append(article)

                logger.info(
                    f"Found {len(cleaned_articles)} news articles for query: {query}"
                )
                return cleaned_articles
            except Exception as e:
                logger.error(f"Error processing news articles: {str(e)}", exc_info=True)
                return []
        except httpx.RequestError as exc:
            logger.error(f"News API request failed: {exc}", exc_info=True)
            return []

    async def fetch_github_repos(
        self, query: str, max_results: int = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch repositories from GitHub API.

        Args:
            query: The search query
            max_results: Maximum number of results to return

        Returns:
            List of dictionaries containing repository details
        """
        if not max_results:
            max_results = settings.MAX_RESULTS_PER_SOURCE

        logger.info(
            f"Fetching GitHub repositories for query: {query} (max: {max_results})"
        )

        base_url = "https://api.github.com/search/repositories"

        params = {"q": query, "sort": "stars", "order": "desc", "per_page": max_results}

        headers = {"Accept": "application/vnd.github.v3+json"}

        if settings.GITHUB_TOKEN:
            headers["Authorization"] = f"token {settings.GITHUB_TOKEN}"

        response = await self.client.get(base_url, params=params, headers=headers)

        if response.status_code != 200:
            logger.error(f"GitHub API error: {response.status_code} - {response.text}")
            return []

        try:
            data = response.json()
            repos = data.get("items", [])

            # Clean and simplify the repository data
            cleaned_repos = []
            for repo in repos:
                cleaned_repo = {
                    "name": repo.get("name", ""),
                    "full_name": repo.get("full_name", ""),
                    "html_url": repo.get("html_url", ""),
                    "description": repo.get("description", ""),
                    "stargazers_count": repo.get("stargazers_count", 0),
                    "language": repo.get("language", ""),
                    "created_at": repo.get("created_at", ""),
                    "updated_at": repo.get("updated_at", ""),
                    "topics": repo.get("topics", []),
                    "owner": {
                        "login": repo.get("owner", {}).get("login", ""),
                        "html_url": repo.get("owner", {}).get("html_url", ""),
                    },
                }
                cleaned_repos.append(cleaned_repo)

            logger.info(
                f"Found {len(cleaned_repos)} GitHub repositories for query: {query}"
            )
            return cleaned_repos
        except Exception as e:
            logger.error(f"Error processing GitHub repositories: {str(e)}")
            return []

    async def fetch_wikipedia_info(
        self, query: str, max_results: int = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch detailed information from Wikipedia API, including full article content
        and related articles for more comprehensive coverage.

        Args:
            query: The search query
            max_results: Maximum number of results to return

        Returns:
            List of dictionaries containing Wikipedia article details
        """
        if not max_results:
            max_results = settings.MAX_RESULTS_PER_SOURCE

        logger.info(
            f"Fetching enhanced Wikipedia information for query: {query} (max: {max_results})"
        )

        base_url = "https://en.wikipedia.org/w/api.php"

        # First search for relevant pages
        search_params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srlimit": max_results,
        }

        try:
            logger.debug(
                f"Sending request to Wikipedia API with params: {search_params}"
            )
            search_response = await self.client.get(base_url, params=search_params)

            if search_response.status_code != 200:
                logger.error(
                    f"Wikipedia search API error: {search_response.status_code} - {search_response.text}"
                )
                return []

            try:
                search_data = search_response.json()
                logger.debug(f"Wikipedia search response: {search_data}")

                # Check for API-specific error messages
                if "error" in search_data:
                    logger.error(
                        f"Wikipedia API returned error: {search_data['error']}"
                    )
                    return []

                search_results = search_data.get("query", {}).get("search", [])

                if not search_results:
                    logger.warning(f"No Wikipedia results found for query: {query}")
                    return []

                # Then fetch content for each page (full content instead of just intro)
                page_titles = [result["title"] for result in search_results]
                page_content = []

                for title in page_titles:
                    # Get full content instead of just the intro (no exintro parameter)
                    content_params = {
                        "action": "query",
                        "prop": "extracts|info|links|categories|images",
                        "inprop": "url",
                        "titles": title,
                        "format": "json",
                        "explaintext": True,
                        "exsectionformat": "plain",
                        "pllimit": 10,  # Get 10 linked pages
                        "cllimit": 10,  # Get 10 categories
                    }

                    logger.debug(
                        f"Fetching enhanced content for Wikipedia page: {title}"
                    )
                    content_response = await self.client.get(
                        base_url, params=content_params
                    )

                    if content_response.status_code != 200:
                        logger.error(
                            f"Wikipedia content API error: {content_response.status_code} - {content_response.text}"
                        )
                        continue

                    content_data = content_response.json()
                    pages = content_data.get("query", {}).get("pages", {})

                    if not pages:
                        logger.warning(f"No pages found for Wikipedia title: {title}")
                        continue

                    for page_id, page_info in pages.items():
                        if page_id == "-1":
                            logger.warning(f"Invalid page ID for title: {title}")
                            continue

                        extract = page_info.get("extract", "")
                        if not extract:
                            logger.warning(
                                f"No content extracted for Wikipedia page: {title}"
                            )
                            continue

                        # Extract related information
                        links = []
                        if "links" in page_info:
                            links = [
                                link.get("title", "") for link in page_info["links"]
                            ]

                        categories = []
                        if "categories" in page_info:
                            categories = [
                                cat.get("title", "").replace("Category:", "")
                                for cat in page_info["categories"]
                            ]

                        images = []
                        if "images" in page_info:
                            images = [
                                img.get("title", "") for img in page_info["images"]
                            ]

                        page_content.append(
                            {
                                "title": page_info.get("title", ""),
                                "content": extract,
                                "url": page_info.get("fullurl", ""),
                                "page_id": page_id,
                                "links": links,
                                "categories": categories,
                                "images": images,
                            }
                        )

                logger.info(
                    f"Found {len(page_content)} Wikipedia articles for query: {query}"
                )
                return page_content
            except Exception as e:
                logger.error(
                    f"Error processing Wikipedia information: {str(e)}", exc_info=True
                )
                return []
        except httpx.RequestError as exc:
            logger.error(f"Wikipedia API request failed: {exc}", exc_info=True)
            return []

    async def fetch_semantic_scholar_papers(
        self, query: str, max_results: int = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch papers from Semantic Scholar API for more comprehensive academic research.

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
            "fields": "title,abstract,url,venue,year,authors,citationCount,influentialCitationCount,references,referenceCount,fieldsOfStudy",
        }

        headers = {}
        if (
            hasattr(settings, "SEMANTIC_SCHOLAR_API_KEY")
            and settings.SEMANTIC_SCHOLAR_API_KEY
        ):
            headers["x-api-key"] = settings.SEMANTIC_SCHOLAR_API_KEY

        try:
            response = await self.client.get(base_url, params=params, headers=headers)

            if response.status_code != 200:
                logger.error(
                    f"Semantic Scholar API error: {response.status_code} - {response.text}"
                )
                return []

            data = response.json()
            papers = data.get("data", [])

            # Clean up and format the papers
            processed_papers = []
            for paper in papers:
                # Skip papers without essential information
                if not paper.get("title") or not paper.get("abstract"):
                    continue

                # Format authors
                authors = []
                if "authors" in paper:
                    authors = [author.get("name", "") for author in paper["authors"]]

                # Get publication details
                year = paper.get("year")
                venue = paper.get("venue")

                # Format fields of study
                fields = paper.get("fieldsOfStudy", [])

                processed_papers.append(
                    {
                        "title": paper.get("title", ""),
                        "abstract": paper.get("abstract", ""),
                        "url": paper.get("url", ""),
                        "authors": authors,
                        "year": year,
                        "venue": venue,
                        "citation_count": paper.get("citationCount", 0),
                        "influential_citation_count": paper.get(
                            "influentialCitationCount", 0
                        ),
                        "reference_count": paper.get("referenceCount", 0),
                        "fields_of_study": fields,
                    }
                )

            logger.info(
                f"Found {len(processed_papers)} Semantic Scholar papers for query: {query}"
            )
            return processed_papers

        except Exception as e:
            logger.error(f"Error fetching from Semantic Scholar: {str(e)}")
            return []

    async def fetch_all_sources(
        self, query: str, max_results_per_source: int = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Fetch data from all sources in parallel.

        Args:
            query: The search query
            max_results_per_source: Maximum number of results to return per source

        Returns:
            Dictionary containing data from all sources
        """
        if not max_results_per_source:
            max_results_per_source = settings.MAX_RESULTS_PER_SOURCE

        # Use gather to fetch from all sources in parallel
        tasks = [
            self.fetch_arxiv_papers(query, max_results_per_source),
            self.fetch_news_articles(query, max_results_per_source),
            self.fetch_github_repos(query, max_results_per_source),
            self.fetch_wikipedia_info(query, max_results_per_source),
            self.fetch_semantic_scholar_papers(query, max_results_per_source),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions
        sources = {}
        source_names = ["arxiv", "news", "github", "wikipedia", "semantic_scholar"]

        for i, result in enumerate(results):
            source_name = source_names[i]

            if isinstance(result, Exception):
                logger.error(f"Error fetching from {source_name}: {str(result)}")
                sources[source_name] = []
            else:
                sources[source_name] = result

        return sources
