"""Search tool for SearXNG integration."""

from dataclasses import dataclass
from typing import Optional

import httpx


@dataclass
class SearchResult:
    """Represents a single search result from SearXNG."""

    title: str
    url: str
    content: str
    engine: str
    category: Optional[str] = None
    score: Optional[float] = None


class SearchTool:
    """Tool for performing web searches using SearXNG API."""

    def __init__(self, searxng_url: str, timeout: float = 30.0):
        """Initialize the search tool.

        Args:
            searxng_url: Base URL of the SearXNG instance.
            timeout: Request timeout in seconds.
        """
        self.searxng_url = searxng_url.rstrip("/")
        self.timeout = timeout

    async def search(
        self,
        query: str,
        categories: Optional[list[str]] = None,
        limit: int = 10,
    ) -> list[SearchResult]:
        """Perform a search query against SearXNG.

        Args:
            query: The search query string.
            categories: List of search categories (e.g., 'general', 'news', 'it').
                       If None, all categories are searched.
            limit: Maximum number of results to return.

        Returns:
            List of search results.

        Raises:
            httpx.RequestError: If the request fails.
            ValueError: If the response is invalid.
        """
        params: dict[str, str | int] = {
            "q": query,
            "format": "json",
            "pageno": 1,
        }

        if categories:
            params["categories"] = ",".join(categories)
        if limit > 0:
            params["num"] = min(limit, 100)  # SearXNG typically caps at 100

        url = f"{self.searxng_url}/search"

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()

        data = response.json()
        return self._parse_results(data.get("results", []))

    def _parse_results(self, results: list[dict]) -> list[SearchResult]:
        """Parse SearXNG JSON response into SearchResult objects.

        Args:
            results: Raw results from SearXNG API.

        Returns:
            List of parsed SearchResult objects.
        """
        parsed = []
        for result in results:
            try:
                parsed.append(
                    SearchResult(
                        title=result.get("title", "Untitled"),
                        url=result.get("url", ""),
                        content=result.get("content", ""),
                        engine=result.get("engine", "unknown"),
                        category=result.get("category"),
                        score=result.get("score"),
                    )
                )
            except Exception:
                # Skip malformed results
                continue
        return parsed

    async def get_engines(self) -> list[str]:
        """Get available search engines from SearXNG.

        Returns:
            List of available engine names.
        """
        url = f"{self.searxng_url}/search?q=test&format=json"

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(url)
            response.raise_for_status()

        data = response.json()
        engines = set()
        for result in data.get("results", []):
            if engine := result.get("engine"):
                engines.add(engine)
        return sorted(engines)
