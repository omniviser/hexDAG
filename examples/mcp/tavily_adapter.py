"""Tavily web search adapter for hexDAG tool integration.

This adapter provides integration with Tavily's AI-powered search API,
enabling web search capabilities for research agents.

Tools are plain functions with type hints and docstrings.
Reference in YAML as: tools: [examples.mcp.tavily_adapter.tavily_search]
"""

from typing import Any

from hexdag.kernel.logging import get_logger

logger = get_logger(__name__)


async def tavily_search(
    query: str,
    search_depth: str = "advanced",
    max_results: int = 5,
    include_answer: bool = True,
    include_raw_content: bool = False,
    include_domains: list[str] | None = None,
    exclude_domains: list[str] | None = None,
) -> dict[str, Any]:
    """Search the web using Tavily API.

    Args
    ----
        query: Search query string
        search_depth: "basic" or "advanced" (default: "advanced")
        max_results: Maximum number of results to return (default: 5)
        include_answer: Whether to include AI-generated answer summary (default: True)
        include_raw_content: Whether to include raw content from pages (default: False)
        include_domains: Optional list of domains to include in search
        exclude_domains: Optional list of domains to exclude from search

    Returns
    -------
        Dictionary containing:
        - answer: AI-generated answer summary (if include_answer=True)
        - results: List of search results with title, url, content, score
        - query: Original search query
        - response_time: Time taken for search

    Examples
    --------
    Basic search::

        result = await tavily_search("latest developments in quantum computing")
        print(result["answer"])
        for item in result["results"]:
            print(f"{item['title']}: {item['url']}")

    Advanced search with domain filtering::

        result = await tavily_search(
            "climate change research",
            search_depth="advanced",
            max_results=10,
            include_domains=["nature.com", "science.org"]
        )
    """
    try:
        # Import tavily-python here to avoid requiring it for non-Tavily workflows
        try:
            from tavily import AsyncTavilyClient
        except ImportError as e:
            raise ImportError(
                "Tavily client not found. Install with: pip install tavily-python"
            ) from e

        import os

        # Get API key from environment
        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            raise ValueError(
                "TAVILY_API_KEY environment variable not set. "
                "Get your API key from https://tavily.com"
            )

        # Initialize Tavily client
        client = AsyncTavilyClient(api_key=api_key)

        # Build search parameters
        search_params: dict[str, Any] = {
            "query": query,
            "search_depth": search_depth,
            "max_results": max_results,
            "include_answer": include_answer,
            "include_raw_content": include_raw_content,
        }

        if include_domains:
            search_params["include_domains"] = include_domains
        if exclude_domains:
            search_params["exclude_domains"] = exclude_domains

        # Execute search
        logger.info(f"Tavily search: {query} (depth={search_depth}, max={max_results})")
        response = await client.search(**search_params)

        # Format response
        result = {
            "query": query,
            "answer": response.get("answer", ""),
            "results": [
                {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "content": item.get("content", ""),
                    "score": item.get("score", 0.0),
                    "raw_content": item.get("raw_content") if include_raw_content else None,
                }
                for item in response.get("results", [])
            ],
            "response_time": response.get("response_time", 0.0),
        }

        logger.info(
            f"Tavily search completed: {len(result['results'])} results in {result['response_time']:.2f}s"
        )
        return result

    except Exception as e:
        logger.error(f"Tavily search error: {e}", exc_info=True)
        return {
            "query": query,
            "answer": "",
            "results": [],
            "error": str(e),
            "response_time": 0.0,
        }


async def tavily_qna_search(query: str) -> str:
    """Quick question-answering search using Tavily's Q&A endpoint.

    This is optimized for direct answers to questions, returning just
    the AI-generated answer without detailed search results.

    Args
    ----
        query: Question to answer

    Returns
    -------
        Direct answer string

    Examples
    --------
    Quick factual question::

        answer = await tavily_qna_search("What is the capital of France?")
        # Returns: "Paris"

    Complex question::

        answer = await tavily_qna_search(
            "What are the main differences between Python and JavaScript?"
        )
    """
    try:
        try:
            from tavily import AsyncTavilyClient
        except ImportError as e:
            raise ImportError(
                "Tavily client not found. Install with: pip install tavily-python"
            ) from e

        import os

        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            raise ValueError("TAVILY_API_KEY environment variable not set")

        client = AsyncTavilyClient(api_key=api_key)

        logger.info(f"Tavily Q&A search: {query}")
        answer = await client.qna_search(query=query)

        logger.info("Tavily Q&A completed")
        return answer or "No answer found"

    except Exception as e:
        logger.error(f"Tavily Q&A error: {e}", exc_info=True)
        return f"Error: {e}"
