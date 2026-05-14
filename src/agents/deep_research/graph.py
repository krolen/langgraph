from typing import Literal

from langgraph.graph import StateGraph, END

from src.agents.deep_research.nodes import planner_node, executor_node, synthesizer_node
from src.agents.deep_research.state import ResearchState


def should_continue(state: ResearchState) -> Literal["planner", "end"]:
    """
    Conditional edge to decide whether to continue research or end.
    """
    # End if final report is generated
    if state.get("final_report"):
        return "end"

    # End if max iterations reached (max 3)
    if state.get("iteration_count", 0) >= 3:
        return "end"

    # Otherwise, go back to planner to fill remaining gaps
    return "planner"


def create_research_agent() -> StateGraph:
    """
    Constructs the Deep Research LangGraph.
    """
    workflow = StateGraph(ResearchState)

    # Add nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("synthesizer", synthesizer_node)

    # Define edges
    workflow.set_entry_point("planner")

    workflow.add_edge("planner", "executor")
    workflow.add_edge("executor", "synthesizer")

    workflow.add_conditional_edges(
        "synthesizer",
        should_continue,
        {
            "planner": "planner",
            "end": END
        }
    )

    return workflow


def create_compiled_research_agent():
    """
    Creates a compiled version of the research agent.
    """
    return create_research_agent().compile()
