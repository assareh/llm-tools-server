"""Web search tool using Ollama API.

This module provides web search functionality using Ollama's web search API.
Requires an OLLAMA_API_KEY to be configured.

Optional dependencies: requests (usually already installed)
Install with: uv sync --extra websearch
"""

import logging

logger = logging.getLogger(__name__)


def ollama_web_search(query: str, max_results: int = 5, api_key: str = "") -> list[dict[str, str]]:
    """Search the web using Ollama's search API.

    Args:
        query: The search query
        max_results: Maximum number of results (default 5)
        api_key: Ollama API key for authentication

    Returns:
        List of search result dictionaries with 'title', 'url', and 'description' keys

    Raises:
        ValueError: If API key is not provided
        Exception: If API call fails
    """
    if not api_key:
        raise ValueError("OLLAMA_API_KEY not configured")

    logger.info(f"[OLLAMA_SEARCH] Searching with query: {query}")

    try:
        import requests

        # Make API request to Ollama web search endpoint
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        response = requests.post(
            "https://ollama.com/api/web_search",
            headers=headers,
            json={"query": query, "max_results": max_results},
            timeout=30,
        )
        response.raise_for_status()

        data = response.json()
        results = []

        # Parse response
        items = data.get("results", [])

        # Limit to max_results
        for item in items[:max_results]:
            results.append(
                {
                    "title": item.get("title", "No title"),
                    "url": item.get("url", ""),
                    "description": item.get("content", item.get("description", "No description")),
                }
            )

        logger.info(f"[OLLAMA_SEARCH] Found {len(results)} results")
        return results

    except Exception as e:
        logger.error(f"[OLLAMA_SEARCH] Search failed: {e}")
        raise


def web_search(query: str, max_results: int = 5, site: str = "", ollama_api_key: str = "") -> str:
    """Search the web using Ollama API.

    Args:
        query: The search query
        max_results: Maximum number of results (default 5)
        site: Optional site restriction (e.g., 'hashicorp.com')
        ollama_api_key: Ollama API key (required)

    Returns:
        Formatted string with search results
    """
    # Check for API key
    if not ollama_api_key:
        return "Web search requires OLLAMA_API_KEY to be configured in your .env file."

    # Build search query with site restriction if provided
    search_query = f"site:{site} {query}" if site else query

    try:
        logger.info("[WEB_SEARCH] Using Ollama search API")
        results = ollama_web_search(search_query, max_results, ollama_api_key)
    except Exception as e:
        return f"Web search failed: {e}"

    # Format results
    if not results:
        return f"No web results found for query: '{query}'"

    output = [f"Found {len(results)} web result(s):\n"]

    for idx, result in enumerate(results, 1):
        output.append(f"\n{idx}. {result['title']}")
        output.append(f"   URL: {result['url']}")
        output.append(f"   {result['description']}")
        output.append("")

    return "\n".join(output)


__all__ = ["ollama_web_search", "web_search"]
