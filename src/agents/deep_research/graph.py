from typing import Literal

from langgraph.graph import StateGraph, END
from langgraph.runtime import Runtime

from src.agents.deep_research.context import Context
from src.agents.deep_research.nodes import planner_node, search_node, crawl_node, extraction_node, synthesizer_node
from src.agents.deep_research.state import ResearchState


def should_continue(state: ResearchState, runtime: Runtime[Context]) -> Literal["planner", "end"]:
    """
    Conditional edge to decide whether to continue research or end.
    """
    if state.final_report:
        return "end"
    if state.iteration_count >= runtime.context.max_iterations:
        return "end"
    return "planner"

# Define the graph using schemas for Aegra compatibility
builder = StateGraph(ResearchState, context_schema=Context)

# Add nodes
builder.add_node("planner", planner_node)
builder.add_node("search", search_node)
builder.add_node("crawl", crawl_node)
builder.add_node("extract", extraction_node)
builder.add_node("synthesizer", synthesizer_node)

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

graph = builder.compile(name="Deep Research Agent")

def create_compiled_research_agent():
    return graph
