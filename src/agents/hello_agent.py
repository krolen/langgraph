"""Simple hello world LangGraph agent for AEGRA registration demo."""

from typing import Annotated

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel
import openlit

# LLM configuration
LLM_BASE_URL = "http://192.168.0.100:4444/v1"
LLM_MODEL = "my-vllm/mymodel"
LLM_API_KEY = "sk-bf-db373190-3492-4d62-9ad1-f10200f0928f"

# Initialize the LLM client
llm = ChatOpenAI(
    base_url=LLM_BASE_URL,
    model=LLM_MODEL,
    api_key=LLM_API_KEY,
)


class HelloState(BaseModel):
    """State for the hello world agent with message history."""
    messages: Annotated[list[BaseMessage], add_messages]
    name: str = "World"


def hello_node(state: HelloState) -> dict:
    """Node that generates a greeting and joke using the LLM."""
    name = state.name

    # Prepare messages for the LLM
    messages = [
        SystemMessage(content="You are a friendly assistant. Greet the user and tell them a short, clean joke in one sentence."),
        HumanMessage(content=f"My name is {name}. Please greet me and share a joke."),
    ]

    # Call the LLM
    response = llm.invoke(messages)

    # Return the response as a new message to add to state
    return {"messages": [response]}


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
    openlit.init(
        otlp_endpoint="http://192.168.0.100:4318",  # OpenLIT's OTEL collector
        application_name="my-langgraph-app",
        disable_batch=True
    )
    agent = create_compiled_hello_agent()

    # Run the agent
    result = agent.invoke({"name": "Alice"})
    print(result)