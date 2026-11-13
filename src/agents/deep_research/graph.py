from functools import partial
from typing import Any, Literal

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
from src.agents.llm_wrapper import model_repo


def _get_field(state: Any, field_name: str, default: Any = None) -> Any:
    """Helper to get field from state whether it is a dict or a dataclass."""
    if isinstance(state, dict):
        return state.get(field_name, default)
    return getattr(state, field_name, default)


async def should_continue(state: ResearchState, runtime: Runtime[Context]) -> Literal["planner", "end"]:
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


def _build_graph(ctx: Context) -> Any:
    """Build and compile the graph with models for each node."""
    builder = StateGraph(state_schema=ResearchState, context_schema=Context)

    # Initialize models for different roles using the model_repo to avoid redundant creation
    def get_model(attr_name: str) -> Any:
        profile_name = getattr(ctx, attr_name, None) or ctx.default_profile
        return model_repo.get_model_by_profile(profile_name)

    planner_model = get_model("planner_profile")
    search_model = get_model("search_profile")
    extractor_model = get_model("extractor_profile")
    synthesizer_model = get_model("synthesizer_profile")

    # Add nodes
    builder.add_node("planner", partial(planner_node, model=planner_model))
    builder.add_node("search", partial(search_node, model=search_model))
    builder.add_node("crawl", crawl_node)
    builder.add_node("extract", partial(extraction_node, model=extractor_model))
    builder.add_node("synthesizer", partial(synthesizer_node, model=synthesizer_model))

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


def _extract_context(runtime: ServerRuntime[Context]) -> Context:
    """Extract the run context, tolerating untyped or missing context objects."""
    ert = getattr(runtime, "execution_runtime", None)
    ctx = getattr(ert, "context", None) if ert is not None else None
    if isinstance(ctx, Context):
        return ctx
    if isinstance(ctx, dict):
        return Context(**ctx)
    return Context()


def graph(
    config: dict[str, Any],
    runtime: ServerRuntime[Context],
) -> Any:
    """Build the server graph without contacting MCP during inspection."""
    return _build_graph(_extract_context(runtime))


def local_graph(ctx: Context) -> Any:
    """Build the local graph; nodes discover MCP tools when needed."""
    return _build_graph(ctx)
