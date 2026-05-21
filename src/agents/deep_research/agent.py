from dataclasses import asdict
from typing import Any, Dict

from src.agents.deep_research.context import Context
from src.agents.deep_research.graph import graph as graph_factory
from src.agents.deep_research.state import ResearchState


class DeepResearchAgentRunner:
    """
    Wrapper class for the Deep Research Agent to provide a clean entry point.
    """

    async def run(self, query: str) -> Dict[str, Any]:
        """
        Executes the research process for a given query.
        """
        # Create a mock runtime for local execution
        class MockRuntime:
            def __init__(self, context):
                self.execution_runtime = type('obj', (object,), {'context': context})
                self.user = None

        ctx = Context()
        runtime = MockRuntime(ctx)

        async with graph_factory({}, runtime) as agent:
            # Prepare state as dict for LangGraph
            state_dict = asdict(ResearchState(
                query=query,
                discovered_urls=[],
                knowledge_base={},
                research_plan=[],
                final_report=None,
                iteration_count=0
            ))
            
            result = await agent.ainvoke(state_dict)
            return result

def create_research_agent_runner():
    """
    Factory function to create a DeepResearchAgentRunner instance.
    """
    return DeepResearchAgentRunner()
