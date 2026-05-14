import operator
from typing import TypedDict, List, Annotated


class ResearchState(TypedDict):
    """State for the Deep Research Agent."""

    # The original research goal
    query: str

    # List of URLs discovered during the search process
    discovered_urls: Annotated[List[str], operator.add]

    # Collection of extracted content and summaries from crawled pages
    # Mapping of URL -> Content/Summary
    knowledge_base: Annotated[dict, operator.ior]

    # Current set of questions or information gaps that need to be addressed
    research_plan: List[str]

    # The final synthesized research report
    final_report: str

    # Current iteration count to prevent infinite loops (max 3)
    iteration_count: int
