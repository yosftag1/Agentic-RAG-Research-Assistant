"""Web search tool — DuckDuckGo (free) + optional Tavily.

Wrapped as LangChain tools for use by agents.
"""

from __future__ import annotations

import logging

from langchain_core.tools import tool

from research_assistant.config import get_settings

logger = logging.getLogger(__name__)


def search_web_raw(query: str, max_results: int = 5) -> list[dict]:
    """Search the web using DuckDuckGo and return raw result dictionaries."""
    from duckduckgo_search import DDGS

    logger.info("Web search: '%s'", query)

    results: list[dict] = []
    with DDGS() as ddgs:
        for result in ddgs.text(query, max_results=max_results):
            results.append(result)

    return results


def format_web_results(results: list[dict]) -> str:
    """Format raw web search results into a readable text block."""
    if not results:
        return "No web results found."

    formatted: list[str] = []
    for i, result in enumerate(results, 1):
        formatted.append(
            f"[{i}] {result.get('title', 'No title')}\n"
            f"    URL: {result.get('href', 'N/A')}\n"
            f"    {result.get('body', 'No snippet')}"
        )

    return "\n\n".join(formatted)


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web using DuckDuckGo and return results.

    Args:
        query: Search query string.
        max_results: Maximum number of results to return.

    Returns:
        Formatted search results with titles, snippets, and URLs.
    """
    return format_web_results(search_web_raw(query, max_results))


@tool
def tavily_search(query: str, max_results: int = 5) -> str:
    """Search the web using Tavily API (requires TAVILY_API_KEY).

    Args:
        query: Search query string.
        max_results: Maximum number of results to return.

    Returns:
        Formatted search results with titles, content, and URLs.
    """
    settings = get_settings()
    if not settings.tavily_api_key:
        return "Tavily API key not configured. Use web_search instead."

    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=settings.tavily_api_key)
        response = client.search(query=query, max_results=max_results)

        results = response.get("results", [])
        if not results:
            return "No Tavily results found."

        formatted = []
        for i, r in enumerate(results, 1):
            formatted.append(
                f"[{i}] {r.get('title', 'No title')}\n"
                f"    URL: {r.get('url', 'N/A')}\n"
                f"    {r.get('content', 'No content')}"
            )
        return "\n\n".join(formatted)

    except ImportError:
        return "tavily-python not installed. Run: pip install tavily-python"
