"""SearXNG search LangGraph agent."""

from datetime import datetime
from typing import Annotated, Optional

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_community.utilities import SearxSearchWrapper
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
import openlit

# LLM configuration
LLM_BASE_URL = "http://192.168.0.100:4444/v1"
LLM_MODEL = "my-vllm/mymodel"
LLM_API_KEY = "sk-bf-db373190-3492-4d62-9ad1-f10200f0928f"

# SearXNG Configuration
SEARXNG_URL = "http://192.168.0.100:8089"

# Initialize the LLM client
llm = ChatOpenAI(
    base_url=LLM_BASE_URL,
    model=LLM_MODEL,
    api_key=LLM_API_KEY,
)

# Recommended SearXNG engines from searxng_recommended_engines.yaml
# Grouped by category for easier selection
AVAILABLE_ENGINES = {
    # Core Search Engines
    "google": "Primary web search with broad coverage",
    "duckduckgo": "Privacy-focused alternative search engine",
    "startpage": "Google results without tracking",
    "brave": "Independent index, privacy-focused",
    "bing": "Microsoft search engine",
    "want": "Privacy-focused search engine",

    # Knowledge & Reference
    "wikipedia": "General encyclopedia",
    "wikidata": "Structured knowledge base",
    "openlibrary": "Books and literature",

    # Academic & Scientific
    "arxiv": "CS, physics, math preprints",
    "google_scholar": "Academic papers",
    "semantic_scholar": "AI-powered research",
    "pubmed": "Medical and life sciences",
    "crossref": "Research publications",
    "openalex": "Open Research Exchange",

    # Programming & Development
    "github": "Code repositories",
    "gitlab": "Alternative code hosting",
    "github_code": "Code-specific GitHub search",
    "gitlab_code": "Code-specific GitLab search",
    "stackexchange": "Stack Overflow and dev Q&A",
    "reddit": "Developer discussions",
    "hackernews": "Tech news and discussions",
    "pypi": "Python packages",
    "npm": "JavaScript packages",
    "crates": "Rust packages",
    "docker_hub": "Container images",
    "huggingface": "ML models and datasets",
    "sourcehut": "Alternative open-source hosting",

    # Documentation & Reference
    "dictzone": "Online dictionary and translator",
    "jisho": "Japanese dictionary",
    "wordnik": "Dictionary and word definitions",

    # News
    "google_news": "News aggregation from Google",
    "reuters": "International news service",

    # Media
    "youtube": "Video tutorials and content",
    "unsplash": "Free stock photos",
    "pexels": "Free stock photos and videos",
    "flickr": "Photo sharing service",
    "imgur": "Image hosting and sharing",

    # Maps
    "openstreetmap": "Maps and locations",

    # Computational
    "wolframalpha_api": "Computational knowledge engine",

    # Translation
    "deepl": "AI-powered translation",
}

# Available categories (derived from engine groups)
AVAILABLE_CATEGORIES = [
    "general",      # Core search engines
    "knowledge",    # Wikipedia, Wikidata, OpenLibrary
    "science",      # Academic engines
    "it",           # Programming and development
    "documentation",  # Dictionaries and references
    "news",         # News sources
    "images",       # Image search
    "videos",       # Video search
    "maps",         # Map services
    "files",        # File downloads
    "music",        # Music services
    "social_media", # Social platforms
    "others",       # Miscellaneous engines
]


class SearXNGSearchState(BaseModel):
    """State for the SearXNG search agent with message history."""
    messages: Annotated[list[BaseMessage], add_messages]
    search_query: str = Field(..., description="Original search query from user")
    rephrased_query: str = Field("", description="LLM-rephrased query optimized for search")
    engines: list[str] = Field(default_factory=list, description="List of search engines to use")
    categories: list[str] = Field(default_factory=list, description="Search categories")
    search_limit: int = Field(default=10, description="Maximum number of results")
    time_range: Optional[str] = Field(None, description="Time range filter: day, month, year")
    safe_search: bool = Field(True, description="Whether to use safe search")
    results: str = Field("", description="Search results as formatted text")
    search_engines_used: list[str] = Field(default_factory=list, description="Engines actually used")


def prepare_search_node(state: SearXNGSearchState) -> dict:
    """Node that prepares the search request with rephrased query.

    Uses LLM to rephrase the search query for better results in specific categories.
    """
    # Get current date at request time for context-aware query rephrasing
    current_date = datetime.now().strftime("%Y-%m-%d")

    # Create system prompt for query rephrasing
    system_prompt = """You are a search query rephrasing assistant. Your task is to rephrase user search queries to make them more effective for web search engines.

Available search categories: {categories}

Available engines:
{engines_list}

Guidelines for rephrasing:
- Keep the query concise but descriptive
- Remove unnecessary words while preserving intent
- Add relevant context if the query is too vague
- Make it suitable for the specified category (if provided)
- Keep language consistent with the original query
- If a time range is specified (day, week, month, year), include appropriate relative date context in the rephrased query
- Time range only supports relative periods: day (last 24h), week (last 7d), month (~30d), year (~365d)
- If no time range is specified, keep the query general

Return ONLY the rephrased query, nothing else.""".format(
        categories=", ".join(AVAILABLE_CATEGORIES),
        engines_list="\n".join(f"- {name}: {desc}" for name, desc in AVAILABLE_ENGINES.items())
    )

    human_message = HumanMessage(
        content=f"Original query: {state.search_query}\n"
        f"Target categories: {state.categories if state.categories else 'general (default)'}\n"
        f"Target engines: {state.engines if state.engines else 'all available'}\n"
        f"Time range: {state.time_range if state.time_range else 'none (all time, uses default engine behavior)'}\n"
        f"Safe search: {'enabled' if state.safe_search else 'disabled'}\n"
        f"Current date: {current_date}\n\n"
        f"Note: time_range only supports relative periods (day, week, month, year). Absolute dates are not supported."
        f"Please rephrase this query for optimal search results."
    )

    messages = [
        SystemMessage(content=system_prompt),
        human_message,
    ]

    return {
        "messages": messages,
        "engines": state.engines,
        "categories": state.categories,
    }


def rephrase_query_node(state: SearXNGSearchState) -> dict:
    """Node that invokes LLM to rephrase the search query."""
    messages = state.messages
    response = llm.invoke(messages)

    rephrased = response.content.strip()

    return {
        "messages": [response],
        "rephrased_query": rephrased,
    }


def build_search_params_node(state: SearXNGSearchState) -> dict:
    """Node that builds the final search parameters.

    Determines which engines to use based on categories if engines not specified.
    Passes engines/categories directly to wrapper (None means "all").
    """
    # Use engines as-is; empty list means "all engines" (pass None to wrapper via conditional in search_executor)
    engines = state.engines

    # Use categories as-is; empty list defaults to ["general"]
    categories = state.categories if state.categories else ["general"]

    return {
        "engines": engines,
        "categories": categories,
    }


def search_executor_node(state: SearXNGSearchState) -> dict:
    """Node that executes the search against SearXNG.

    Uses LangChain's SearxSearchWrapper for the actual search.
    """
    wrapper = SearxSearchWrapper(searx_host=SEARXNG_URL)

    try:
        # Perform search using LangChain's SearxSearchWrapper
        # time_range: only relative periods supported (day=last 24h, week=last 7d, month=~30d, year=~365d)
        # safesearch: 0=off, 1=moderate, 2=strict
        raw_results = wrapper.results(
            query=state.rephrased_query or state.search_query,
            num_results=min(state.search_limit, 100),
            engines=state.engines if state.engines else None,
            categories=state.categories if state.categories else None,
            time_range=state.time_range,  # day, month, year
            safesearch=1 if state.safe_search else 0,
        )

        # Format results as text
        result_lines = []
        engines_used = set()

        for res in raw_results:
            engine = res.get("engines", "unknown")
            if isinstance(engine, list):
                engine = engine[0] if engine else "unknown"
            engines_used.add(engine)
            title = res.get("title", "Untitled")
            res_url = res.get("link", "")
            content = res.get("snippet", "")[:200]

            result_lines.append(
                f"{len(result_lines) + 1}. **{title}**\n"
                f"   URL: {res_url}\n"
                f"   Source: {engine}\n"
                f"   {content}"
            )

        formatted_results = "\n\n".join(result_lines) if result_lines else "No results found."

        return {
            "results": formatted_results,
            "search_engines_used": list(engines_used),
        }

    except Exception as e:
        return {
            "results": f"Search failed: {str(e)}",
            "search_engines_used": [],
        }


def create_searxng_search_agent() -> StateGraph:
    """Create the SearXNG search agent graph.

    Returns:
        Uncompiled LangGraph StateGraph.
    """
    graph = StateGraph(SearXNGSearchState)

    # Add nodes
    graph.add_node("prepare_search", prepare_search_node)
    graph.add_node("rephrase_query", rephrase_query_node)
    graph.add_node("build_params", build_search_params_node)
    graph.add_node("search_executor", search_executor_node)

    # Set entry point
    graph.set_entry_point("prepare_search")

    # Define edges
    graph.add_edge("prepare_search", "rephrase_query")
    graph.add_edge("rephrase_query", "build_params")
    graph.add_edge("build_params", "search_executor")
    graph.add_edge("search_executor", END)

    return graph


def create_compiled_searxng_search_agent():
    """Create a compiled SearXNG search agent ready for execution.

    Returns:
        Compiled LangGraph application.
    """
    graph = create_searxng_search_agent()
    return graph.compile()


# Example usage
if __name__ == "__main__":
    openlit.init(
        otlp_endpoint="http://192.168.0.100:4318",
        application_name="searxng-search-agent",
        disable_batch=True
    )
    agent = create_compiled_searxng_search_agent()

    # Run the agent with search query and optional parameters
    # Note: time_range only supports relative periods: day (24h), week (7d), month (~30d), year (~365d)
    result = agent.invoke({
        "search_query": "latest developments in AI",
        "engines": ["google", "duckduckgo"],  # Optional: specify engines
        "categories": ["it", "news"],  # Optional: specify categories
        "search_limit": 5,
        "time_range": "year",  # Optional: relative period (day, week, month, year)
        "safe_search": True,  # Optional: safe search (default: True)
    })
    print(result)
