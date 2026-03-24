"""LangChain-based web search agent using LangGraph.

This provides a LangGraph workflow that uses LangChain's
SearXNG integration for web searches.
"""

import time
from typing import Any

from langchain_community.tools import SearxSearchTool
from langchain_community.utilities import SearxSearchWrapper
from langgraph.graph import StateGraph, END

from src.agents.state import AgentState, SearchResult


def langchain_search_node(state: AgentState) -> dict[str, Any]:
    """Execute search using LangChain's SearxSearchWrapper.

    Args:
        state: Current agent state.

    Returns:
        Updated state with search results.
    """
    start_time = time.time()

    try:
        wrapper = SearxSearchWrapper(
            searxng_host=getattr(state, "_searxng_url", "http://localhost:8089")
        )
        raw_results = wrapper.results(
            state.refined_query or state.query, limit=state.limit
        )

        # Convert to SearchResult objects
        search_results = [
            SearchResult(
                title=r.get("title", ""),
                url=r.get("link", ""),
                content=r.get("snippet", ""),
                engine=r.get("engine", "searxng"),
                score=r.get("score", 0.0),
            )
            for r in raw_results
        ]

        elapsed_ms = int((time.time() - start_time) * 1000)

        return {
            "search_results": search_results,
            "search_time_ms": elapsed_ms,
            "intermediate_steps": [
                {
                    "step": "langchain_search",
                    "action": "search_completed",
                    "query": state.refined_query or state.query,
                    "results_count": len(search_results),
                    "elapsed_ms": elapsed_ms,
                }
            ],
        }

    except Exception as e:
        elapsed_ms = int((time.time() - start_time) * 1000)

        return {
            "error": f"LangChain search failed: {str(e)}",
            "search_time_ms": elapsed_ms,
            "intermediate_steps": [
                {
                    "step": "langchain_search",
                    "action": "search_failed",
                    "error": str(e),
                    "elapsed_ms": elapsed_ms,
                }
            ],
        }


def langchain_formatter_node(state: AgentState) -> dict[str, Any]:
    """Format search results from LangChain search.

    Args:
        state: Current agent state.

    Returns:
        Updated state with formatted sources.
    """
    start_time = time.time()

    if state.error:
        return {"intermediate_steps": []}

    # Sort by score and limit results
    sorted_results = sorted(
        state.search_results,
        key=lambda r: r.score if r.score is not None else 0,
        reverse=True,
    )
    top_results = sorted_results[: state.limit]

    elapsed_ms = int((time.time() - start_time) * 1000)

    return {
        "sources": top_results,
        "intermediate_steps": [
            {
                "step": "langchain_formatter",
                "action": "results_formatted",
                "total_results": len(state.search_results),
                "formatted_results": len(top_results),
                "elapsed_ms": elapsed_ms,
            }
        ],
    }


def should_summarize(state: AgentState) -> str:
    """Conditional edge to decide whether to summarize.

    Args:
        state: Current agent state.

    Returns:
        Next node to execute.
    """
    if state.error:
        return "end"
    if not state.summarize:
        return "end"
    if not state.search_results:
        return "end"
    return "summarizer"


def create_langchain_web_agent(searxng_url: str):
    """Create a LangGraph agent using LangChain's SearXNG integration.

    Args:
        searxng_url: URL of the SearXNG instance (e.g., "http://localhost:8080")

    Returns:
        Compiled LangGraph agent ready for execution.

    Example:
        ```python
        agent = create_langchain_web_agent("http://localhost:8080")
        result = agent.invoke(AgentState(query="latest AI news"))
        print(result.final_answer)
        ```
    """
    # Create the graph
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("query_processor", query_processor)
    graph.add_node("langchain_search", langchain_search_node)
    graph.add_node("langchain_formatter", langchain_formatter_node)
    graph.add_node("summarizer", summarizer)

    # Define edges
    graph.set_entry_point("query_processor")
    graph.add_edge("query_processor", "langchain_search")
    graph.add_edge("langchain_search", "langchain_formatter")

    # Conditional edge for summarization
    graph.add_conditional_edges(
        "langchain_formatter",
        should_summarize,
        {
            "summarizer": "summarizer",
            "end": END,
        },
    )

    graph.add_edge("summarizer", END)

    return graph.compile()


# Import nodes from the main agent for reuse
from src.agents.nodes import query_processor, summarizer  # noqa: E402


def create_langchain_searx_agent(searxng_url: str):
    """Create a LangChain agent with SearXNG search tool.

    This is a simpler wrapper that returns tools for use with
    other agent frameworks.

    Args:
        searxng_url: URL of the SearXNG instance.

    Returns:
        A dict containing tools and utilities.
    """
    search_tool = SearxSearchTool(searxng_api_url=searxng_url)

    return {
        "tools": [search_tool],
        "search_tool": search_tool,
        "wrapper": SearxSearchWrapper(searxng_host=searxng_url),
    }


async def langchain_search(query: str, searxng_url: str, limit: int = 10) -> list[dict]:
    """Perform a search using LangChain's SearxSearchWrapper.

    Args:
        query: The search query.
        searxng_url: URL of the SearXNG instance.
        limit: Maximum number of results.

    Returns:
        List of search results as dictionaries.
    """
    wrapper = SearxSearchWrapper(searxng_host=searxng_url)
    results = wrapper.results(query, limit=limit)

    return [
        {
            "title": r.get("title", ""),
            "url": r.get("link", ""),
            "content": r.get("snippet", ""),
        }
        for r in results
    ]
