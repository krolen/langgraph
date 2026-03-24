"""Main LangGraph agent definition for web search."""

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from src.agents.state import AgentState
from src.agents.nodes import (
    query_processor,
    search_executor,
    results_formatter,
    summarizer,
)


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


def create_web_search_agent(searxng_url: str) -> StateGraph:
    """Create the web search agent graph.

    Args:
        searxng_url: URL of the SearXNG instance.

    Returns:
        Compiled LangGraph StateGraph.
    """
    # Create the graph with our state model
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("query_processor", query_processor)
    graph.add_node("search_executor", search_executor)
    graph.add_node("results_formatter", results_formatter)
    graph.add_node("summarizer", summarizer)

    # Define edges
    # Start with query processing
    graph.set_entry_point("query_processor")

    # Query processing -> Search execution
    graph.add_edge("query_processor", "search_executor")

    # Search execution -> Results formatting
    graph.add_edge("search_executor", "results_formatter")

    # Results formatting -> Conditional (summarize or end)
    graph.add_conditional_edges(
        "results_formatter",
        should_summarize,
        {
            "summarizer": "summarizer",
            "end": END,
        },
    )

    # Summarizer -> End
    graph.add_edge("summarizer", END)

    return graph


def create_compiled_agent(searxng_url: str):
    """Create a compiled web search agent ready for execution.

    Args:
        searxng_url: URL of the SearXNG instance.

    Returns:
        Compiled LangGraph application.
    """
    graph = create_web_search_agent(searxng_url)
    return graph.compile()
