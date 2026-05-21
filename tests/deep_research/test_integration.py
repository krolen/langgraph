import asyncio
import logging
import pytest
import os
from datetime import datetime
from dataclasses import asdict
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage
from src.agents.deep_research.graph import graph as graph_factory
from src.agents.deep_research.context import Context
from src.agents.deep_research.state import ResearchState

# Setup logging to file
os.makedirs("logs", exist_ok=True)
log_file = os.path.join("logs", f"test_integration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

logger = logging.getLogger("test_integration")
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(logging.StreamHandler())

@pytest.mark.asyncio
async def test_deep_research_agent_integration():
    """
    End-to-end integration test for the Deep Research Agent.
    Verifies factory lifecycle, tool injection, and graph execution with mocks.
    """
    logger.info("Starting integration test with mocks...")
    
    # 1. Mock the LLM
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock()
    
    # Mock structured output for Planner
    mock_planner_output = MagicMock()
    mock_planner_output.queries = ["test query 1"]
    
    # Mock structured output for URL Selection
    mock_url_output = MagicMock()
    mock_url_output.selected_urls = ["https://example.com/page1"]
    
    # Mock structured output for Extraction
    mock_extraction_output = MagicMock()
    mock_extraction_output.extracted_info = "Verified: The refactor works perfectly."
    
    # Setup the structured output chain
    def mock_with_structured_output(output_schema):
        m = MagicMock()
        schema_str = str(output_schema)
        if "PlannerOutput" in schema_str:
            m.ainvoke = AsyncMock(return_value=mock_planner_output)
        elif "URLSelectionOutput" in schema_str:
            m.ainvoke = AsyncMock(return_value=mock_url_output)
        elif "ExtractionOutput" in schema_str:
            m.ainvoke = AsyncMock(return_value=mock_extraction_output)
        return m

    mock_llm.with_structured_output = mock_with_structured_output
    
    # Final synthesizer response
    mock_llm.ainvoke.return_value = AIMessage(content="COMPLETE: Yes\nREPORT: The refactor is successful.")

    # 2. Mock MCP Tools
    mock_search_tool = AsyncMock()
    mock_search_tool.name = "web_search"
    mock_search_tool.ainvoke.return_value = [{"type": "text", "text": "Results found: https://example.com/page1"}]
    
    mock_crawl_tool = AsyncMock()
    mock_crawl_tool.name = "web_crawl_url"
    mock_crawl_tool.ainvoke.return_value = [{"type": "text", "text": "Page content: The refactor is successful."}]

    # 3. Create Mock Runtime
    ctx = Context(enable_mcp=True, max_iterations=1)
    
    mock_runtime = MagicMock()
    mock_runtime.context = ctx
    mock_runtime.execution_runtime = MagicMock()
    mock_runtime.execution_runtime.context = ctx
    mock_runtime.user = None

    # 4. Patch and Execute
    with patch("src.agents.deep_research.graph.create_chat_openai", return_value=mock_llm), \
         patch("src.agents.deep_research.graph.MultiServerMCPClient") as mock_mcp_class:
        
        mock_mcp_instance = mock_mcp_class.return_value
        mock_mcp_instance.__aenter__ = AsyncMock(return_value=mock_mcp_instance)
        mock_mcp_instance.__aexit__ = AsyncMock(return_value=None)
        mock_mcp_instance.get_tools = AsyncMock(return_value=[mock_search_tool, mock_crawl_tool])

        async with graph_factory({}, mock_runtime) as agent:
            logger.info("Agent built via factory (mocked).")
            initial_state = asdict(ResearchState(
                query="Verify integration",
                iteration_count=0
            ))
            
            logger.info("Invoking agent...")
            result = await agent.ainvoke(initial_state, config={"configurable": {"runtime": mock_runtime}})
            
            # 5. Assertions
            assert "final_report" in result
            assert "The refactor is successful" in result["final_report"]
            assert result["iteration_count"] == 1
            
            logger.info(f"Test result: {result['final_report']}")
            logger.info("Integration test successful.")
