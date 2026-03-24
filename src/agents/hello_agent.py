"""Simple hello world LangGraph agent for AEGRA registration demo."""

from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from sqlalchemy import true


class HelloState(TypedDict):
    """Simple state for the hello world agent."""
    name: str
    greeting: str
    status: str


def hello_node(state: HelloState) -> dict:
    """A simple node that returns a greeting."""
    name = state.get("name", "World")
    return {
        "greeting": f"Hello, {name}!",
        "status": "completed",
    }


def create_hello_agent() -> StateGraph:
    """Create the hello world agent graph.

    Returns:
        Uncompiled LangGraph StateGraph.
    """
    graph = StateGraph(HelloState)

    # Add a single node
    graph.add_node("greeting", hello_node)

    # Set entry point
    graph.set_entry_point("greeting")

    # Edge to end
    graph.add_edge("greeting", END)

    return graph


def create_compiled_hello_agent():
    """Create a compiled hello world agent ready for execution.

    Returns:
        Compiled LangGraph application.
    """
    graph = create_hello_agent()
    return graph.compile()


# Example usage
if __name__ == "__main__":
    import openlit

    openlit.init(
        otlp_endpoint="http://192.168.0.100:4318",  # OpenLIT's OTEL collector
        application_name="my-langgraph-app",
        disable_batch=true
    )
    agent = create_compiled_hello_agent()

    # Run the agent
    result = agent.invoke({"name": "Alice"})
    print(result)
    # Output: {'name': 'Alice', 'greeting': 'Hello, Alice!', 'status': 'completed'}