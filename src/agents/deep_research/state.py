import operator
from dataclasses import dataclass, field
from typing import List, Annotated

from src.agents.base import BaseAgentState


@dataclass
class ResearchState(BaseAgentState):
    """State for the Deep Research Agent."""

    # List of URLs discovered during the search process
    discovered_urls: Annotated[List[str], operator.add] = field(default_factory=list)

    # Collection of extracted content and summaries from crawled pages
    # Mapping of URL -> Content/Summary
    knowledge_base: Annotated[dict, operator.ior] = field(default_factory=dict)

    # Current set of questions or information gaps that need to be addressed
    research_plan: List[str] = field(default_factory=list)
