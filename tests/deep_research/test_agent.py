import pytest

from src.agents.deep_research.agent import create_research_agent


@pytest.fixture
def research_agent():
    """Fixture to provide a fresh research agent for each test."""
    return create_research_agent()

@pytest.mark.asyncio
async def test_deep_research_agent_execution(research_agent):
    """
    Test the deep research agent with a sample query to verify
    the end-to-end flow and integration with MCP tools.
    """
    query = "What are the core features of Aegra.dev?"
    
    # Execute the agent
    result = await research_agent.run(query)
    
    # Verify that we got a result
    assert result is not None
    assert "final_report" in result
    
    # Verify the report is not empty
    assert len(result["final_report"]) > 0
    
    # Ensure iteration count is within reasonable bounds
    # (The graph is configured for max 3 iterations)
    assert 0 <= result.get("iteration_count", 0) <= 3