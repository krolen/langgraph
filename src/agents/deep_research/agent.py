from typing import Any, Dict

from src.agents.deep_research.graph import create_compiled_research_agent
from src.agents.deep_research.state import ResearchState


class DeepResearchAgent:
    """
    Wrapper class for the Deep Research Agent to provide a clean entry point.
    """

    def __init__(self):
        self.agent = create_compiled_research_agent()

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
            "final_report": "",
            "iteration_count": 0
        }

        result = await self.agent.ainvoke(initial_state)
        return result


def create_research_agent():
    """
    Factory function to create a DeepResearchAgent instance.
    """
    return DeepResearchAgent()
