"""Graph node implementations for the web search agent."""

import time
from typing import Any

from src.agents.state import AgentState, SearchResult, add_step
from src.tools.search import SearchTool


async def query_processor(state: AgentState) -> dict[str, Any]:
    """Process and refine the search query.

    This node analyzes the query and may:
    - Expand abbreviations
    - Add relevant keywords
    - Clarify ambiguous terms
    - Determine appropriate search categories

    Args:
        state: Current agent state.

    Returns:
        Updated state with refined query and categories.
    """
    start_time = time.time()

    query = state.query.strip()

    # Simple query refinement logic
    # In production, this could use an LLM for better query understanding
    refined_query = query

    # Determine categories based on query keywords
    categories = state.categories if state.categories else ["general"]

    query_lower = query.lower()
    category_keywords = {
        "news": ["latest", "recent", "breaking", "today", "yesterday", "new"],
        "it": ["programming", "software", "code", "developer", "tech", "computer"],
        "science": ["research", "study", "scientific", "experiment", "discovery"],
    }

    for category, keywords in category_keywords.items():
        if any(kw in query_lower for kw in keywords):
            if category not in categories:
                categories.append(category)

    elapsed_ms = int((time.time() - start_time) * 1000)

    return {
        "refined_query": refined_query,
        "categories": categories,
        "intermediate_steps": [
            {
                "step": "query_processor",
                "action": "query_refined",
                "original_query": query,
                "refined_query": refined_query,
                "categories": categories,
                "elapsed_ms": elapsed_ms,
            }
        ],
    }


async def search_executor(state: AgentState) -> dict[str, Any]:
    """Execute the search using SearXNG.

    This node calls the SearXNG API with the refined query.

    Args:
        state: Current agent state.

    Returns:
        Updated state with search results.
    """
    start_time = time.time()

    search_tool = SearchTool(
        searxng_url=getattr(state, "_searxng_url", "http://192.168.0.100:8089")
    )

    try:
        results = await search_tool.search(
            query=state.refined_query or state.query,
            categories=state.categories,
            limit=state.limit,
        )

        elapsed_ms = int((time.time() - start_time) * 1000)

        return {
            "search_results": results,
            "search_time_ms": elapsed_ms,
            "intermediate_steps": [
                {
                    "step": "search_executor",
                    "action": "search_completed",
                    "query": state.refined_query or state.query,
                    "results_count": len(results),
                    "elapsed_ms": elapsed_ms,
                }
            ],
        }

    except Exception as e:
        elapsed_ms = int((time.time() - start_time) * 1000)

        return {
            "error": f"Search failed: {str(e)}",
            "search_time_ms": elapsed_ms,
            "intermediate_steps": [
                {
                    "step": "search_executor",
                    "action": "search_failed",
                    "error": str(e),
                    "elapsed_ms": elapsed_ms,
                }
            ],
        }


async def results_formatter(state: AgentState) -> dict[str, Any]:
    """Format and aggregate search results.

    This node processes raw results into a readable format.

    Args:
        state: Current agent state.

    Returns:
        Updated state with formatted results.
    """
    start_time = time.time()

    if state.error:
        return {"intermediate_steps": []}

    results = state.search_results

    # Sort by score if available
    sorted_results = sorted(
        results, key=lambda r: r.score if r.score is not None else 0, reverse=True
    )

    # Limit to top results for the answer
    top_results = sorted_results[: state.limit]

    elapsed_ms = int((time.time() - start_time) * 1000)

    return {
        "sources": top_results,
        "intermediate_steps": [
            {
                "step": "results_formatter",
                "action": "results_formatted",
                "total_results": len(results),
                "formatted_results": len(top_results),
                "elapsed_ms": elapsed_ms,
            }
        ],
    }


async def summarizer(state: AgentState) -> dict[str, Any]:
    """Summarize search results into a coherent answer.

    This node creates a summary from the search results.
    In production, this would use an LLM for better summarization.

    Args:
        state: Current agent state.

    Returns:
        Updated state with final answer.
    """
    start_time = time.time()

    if state.error:
        return {"intermediate_steps": []}

    if not state.summarize:
        # Just return raw results as the answer
        answer_parts = []
        for i, result in enumerate(state.sources, 1):
            answer_parts.append(f"{i}. [{result.title}]({result.url})\n   {result.content[:200]}")

        final_answer = "\n\n".join(answer_parts) if answer_parts else "No results found."

        elapsed_ms = int((time.time() - start_time) * 1000)

        return {
            "final_answer": final_answer,
            "intermediate_steps": [
                {
                    "step": "summarizer",
                    "action": "raw_results_returned",
                    "elapsed_ms": elapsed_ms,
                }
            ],
        }

    # Create a summary from the results
    # In production, this would call an LLM
    summary_parts = []

    if not state.sources:
        final_answer = "No search results were found for your query."
    else:
        # Group results by topic/theme
        summary_parts.append(f"Search results for: '{state.refined_query or state.query}'\n")

        # Add top results with context
        for i, result in enumerate(state.sources[:5], 1):
            snippet = result.content[:150] if result.content else "No description available"
            if len(result.content) > 150:
                snippet += "..."

            summary_parts.append(
                f"{i}. **{result.title}**\n"
                f"   Source: {result.engine}\n"
                f"   {snippet}\n"
                f"   URL: {result.url}\n"
            )

        if len(state.sources) > 5:
            summary_parts.append(
                f"\n... and {len(state.sources) - 5} more results available."
            )

        final_answer = "\n\n".join(summary_parts)

    elapsed_ms = int((time.time() - start_time) * 1000)

    return {
        "final_answer": final_answer,
        "intermediate_steps": [
            {
                "step": "summarizer",
                "action": "summary_created",
                "sources_used": len(state.sources),
                "elapsed_ms": elapsed_ms,
            }
        ],
    }
