import asyncio
import logging
import os
from dataclasses import asdict
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from src.agents.deep_research.context import Context
from src.agents.deep_research.graph import graph as graph_factory
from src.agents.deep_research.state import ResearchState

# Setup logging to file and console
os.makedirs("logs", exist_ok=True)
log_file = os.path.join("logs", f"test_live_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

logger = logging.getLogger("test_live")
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)

# Also add handlers to the agent loggers to capture their output in the file
for logger_name in ["src.agents.deep_research.nodes", "src.agents.deep_research.graph"]:
    l = logging.getLogger(logger_name)
    l.addHandler(file_handler)

@pytest.mark.asyncio
async def test_deep_research_agent_live():
    """
    Live integration test for the Deep Research Agent.
    Runs against actual LLM and MCP services.
    """
    logger.info("Starting live verification of Deep Research Agent...")

    # 1. Setup Runtime with real Context
    ctx = Context(
        enable_mcp=True, 
        max_iterations=1,
        model="krolen/mymodel"
    )
    
    mock_runtime = MagicMock()
    mock_runtime.context = ctx
    mock_runtime.execution_runtime = MagicMock()
    mock_runtime.execution_runtime.context = ctx
    mock_runtime.user = None

    # 2. Build and Execute via Factory
    async with graph_factory({}, mock_runtime) as agent:
        logger.info("Agent built successfully via factory. Connecting to real services...")
        
        initial_state = asdict(ResearchState(
            query="Briefly explain what Aegra is.",
            iteration_count=0
        ))
        
        logger.info("Invoking agent for live research...")
        result = await agent.ainvoke(
            initial_state, 
            config={"configurable": {"runtime": mock_runtime}}
        )
        
        logger.info("Live execution finished.")
        
        # 3. Assertions on real output
        assert "final_report" in result
        assert result["final_report"] is not None
        assert len(result["final_report"]) > 10
        
        logger.info(f"LIVE VERIFICATION SUCCESSFUL. Report length: {len(result['final_report'])}")
        logger.info(f"Final Report: {result['final_report']}")

if __name__ == "__main__":
    # Also add stream handler if run directly
    logger.addHandler(logging.StreamHandler())
    for logger_name in ["src.agents.deep_research.nodes", "src.agents.deep_research.graph"]:
        logging.getLogger(logger_name).addHandler(logging.StreamHandler())
    
    asyncio.run(test_deep_research_agent_live())
