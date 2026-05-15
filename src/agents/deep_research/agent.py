from typing import Any, Dict

from dataclasses import asdict
from src.agents.deep_research.graph import graph
from src.agents.deep_research.context import Context
from src.agents.deep_research.state import ResearchState


class DeepResearchAgentRunner:
    """
    Wrapper class for the Deep Research Agent to provide a clean entry point.
    """

    def __init__(self):
        self.agent = graph

    async def run(self, query: str) -> Dict[str, Any]:
        """
        Executes the research process for a given query.
        
        Args:
            query: The research question or goal.
            
        Returns:
            The final state of the agent, including the final_report.
        """
        initial_state: ResearchState = {
            "query": query,
            "discovered_urls": [],
            "knowledge_base": {},
            "research_plan": [],
            "final_report": None,
            "iteration_count": 0
        }

        result = await self.agent.ainvoke(initial_state, config={"configurable": Context().model_dump()})
        return result

def create_research_agent_runner():
    """
    Factory function to create a DeepResearchAgentRunner instance.
    """
    return DeepResearchAgentRunner()
