import operator
from dataclasses import dataclass, field
from typing import List, Annotated, Dict


@dataclass
class BaseAgentState:
    """Base state common to all agents."""
    query: str
    final_report: str | None
    iteration_count: int = 0


@dataclass
class ResearchState(BaseAgentState):
    """State for the Deep Research Agent."""

    # List of URLs discovered during the search process
    discovered_urls: Annotated[List[str], operator.add] = field(default_factory=list)

    # Collection of extracted content and summaries from crawled pages
    # Mapping of URL -> Content/Summary
    knowledge_base: Annotated[dict, operator.ior] = field(default_factory=dict)

    # Raw content from crawled pages before extraction
    raw_crawl_results: Dict[str, str] = field(default_factory=dict)

    # URLs selected by the model for crawling
    selected_urls: List[str] = field(default_factory=list)

    # Current set of questions or information gaps that need to be addressed
    research_plan: List[str] = field(default_factory=list)
