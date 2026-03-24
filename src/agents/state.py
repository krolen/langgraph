"""LangGraph state definition for the web search agent."""

from collections.abc import Sequence
from typing import Annotated

import operator
from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    """Represents a single search result."""

    title: str = Field(..., description="Title of the search result")
    url: str = Field(..., description="URL of the search result")
    content: str = Field(..., description="Content/snippet of the search result")
    engine: str = Field(..., description="Search engine that returned this result")
    category: str | None = Field(None, description="Category of the result")
    score: float | None = Field(None, description="Relevance score")


class AgentState(BaseModel):
    """State model for the web search agent.

    This state is passed between nodes in the LangGraph workflow.
    """

    # Input query
    query: str = Field(..., description="The original search query")

    # Processed/refined query
    refined_query: str = Field("", description="Refined search query after processing")

    # Search options
    categories: list[str] = Field(
        default_factory=list, description="Search categories to use"
    )
    limit: int = Field(default=10, description="Maximum number of results")
    summarize: bool = Field(default=True, description="Whether to summarize results")

    # Search results
    search_results: list[SearchResult] = Field(
        default_factory=list, description="Raw search results from SearXNG"
    )

    # Final output
    final_answer: str = Field("", description="Final summarized answer")
    sources: list[SearchResult] = Field(
        default_factory=list, description="Sources used in the final answer"
    )

    # Metadata
    intermediate_steps: list[dict] = Field(
        default_factory=list, description="Log of intermediate processing steps"
    )
    search_time_ms: int = Field(default=0, description="Total search time in milliseconds")

    # Error handling
    error: str | None = Field(None, description="Error message if any occurred")


def add_step(state: AgentState, step: dict) -> AgentState:
    """Add an intermediate step to the state.

    This is a reducer function for LangGraph's annotated state.
    """
    state.intermediate_steps.append(step)
    return state
