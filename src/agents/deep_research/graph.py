from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import partial
from typing import Any, Literal

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, END
from langgraph.runtime import Runtime
from langgraph_sdk.runtime import ServerRuntime

from src.agents.deep_research.context import Context
from src.agents.deep_research.nodes import (
    planner_node,
    search_node,
    crawl_node,
    extraction_node,
    synthesizer_node,
)
from src.agents.deep_research.state import ResearchState
from src.agents.llm_wrapper import create_chat_openai


def _get_field(state: Any, field_name: str, default: Any = None) -> Any:
    """Helper to get field from state whether it is a dict or a dataclass."""
    if isinstance(state, dict):
        return state.get(field_name, default)
    return getattr(state, field_name, default)


def should_continue(state: ResearchState, runtime: Runtime[Context]) -> Literal["planner", "end"]:
    """Conditional edge to decide whether to continue research or end."""
    if _get_field(state, "final_report"):
        return "end"

    # Extract context safely
    ctx = getattr(runtime, "context", None)
    if not ctx and hasattr(runtime, "execution_runtime") and hasattr(runtime.execution_runtime, "context"):
        ctx = runtime.execution_runtime.context

    max_iterations = ctx.max_iterations if ctx else 3

    if _get_field(state, "iteration_count", 0) >= max_iterations:
        return "end"
    return "planner"


def _build_graph(tools_dict: dict[str, BaseTool], ctx: Context) -> Any:
    """Build and compile the graph, injecting tools and the model into nodes."""
    builder = StateGraph(state_schema=ResearchState, context_schema=Context)

    # Initialize the model once for the graph
    model = create_chat_openai(model=ctx.model)

    # Bind tools and model to nodes
    web_search = tools_dict.get("web_search")
    web_crawl_url = tools_dict.get("web_crawl_url")

    # Add nodes
    builder.add_node("planner", partial(planner_node, model=model))
    builder.add_node("search", partial(search_node, model=model, web_search=web_search))
    builder.add_node("crawl", partial(crawl_node, web_crawl_url=web_crawl_url))
    builder.add_node("extract", partial(extraction_node, model=model))
    builder.add_node("synthesizer", partial(synthesizer_node, model=model))

    # Define edges
    builder.set_entry_point("planner")
    builder.add_edge("planner", "search")
    builder.add_edge("search", "crawl")
    builder.add_edge("crawl", "extract")
    builder.add_edge("extract", "synthesizer")

    builder.add_conditional_edges(
        "synthesizer",
        should_continue,
        {
            "planner": "planner",
            "end": END
        }
    )

    return builder.compile(name="Deep Research Agent")


@asynccontextmanager
async def graph(config: dict[str, Any], runtime: ServerRuntime[Context]) -> AsyncIterator[Any]:
    """Factory for the Deep Research Agent with MCP lifecycle management."""
    ert = getattr(runtime, "execution_runtime", None)
    if ert and hasattr(ert, "context") and isinstance(ert.context, Context):
        ctx = ert.context
    else:
        # Fallback for initialization or non-typed context
        raw_ctx = getattr(ert, "context", {}) if ert else {}
        if isinstance(raw_ctx, dict):
            ctx = Context(**raw_ctx)
        else:
            ctx = Context()

    tools_dict: dict[str, BaseTool] = {}

    if ctx.enable_mcp:
        connections = {
            "research-tools": {
                "url": ctx.mcp_server_url,
                "transport": "streamable_http",
            }
        }
        mcp_client = MultiServerMCPClient(connections)
        # As of langchain-mcp-adapters 0.1.0, get_tools() manages sessions per call
        mcp_tools = await mcp_client.get_tools()
        for tool in mcp_tools:
            if tool.name == "web_search":
                tools_dict["web_search"] = tool
            elif tool.name == "web_crawl_url":
                tools_dict["web_crawl_url"] = tool

            tools_dict[tool.name] = tool

    compiled = _build_graph(tools_dict, ctx)

    try:
        yield compiled
    finally:
        # No explicit client cleanup needed for MultiServerMCPClient 0.1.0
        pass


def create_compiled_research_agent():
    # This is for backward compatibility if needed, but the factory is 'graph'
    return None
