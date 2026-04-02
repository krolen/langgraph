"""
Minimal Deep Research Agent
Stack: llama.cpp + Qwen3.5-27B + LangGraph + SearXNG

Install deps:
    pip install langgraph langchain-openai langchain-community

Run llama.cpp server:
    ./llama-server -m qwen3.5-27b.gguf --port 8080 -c 8192
"""

from typing import TypedDict

from langchain_community.utilities import SearxSearchWrapper
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from src.agents.llm_wrapper import router_llm, router_llm_local

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
from src.agents.config import config

# Initialize SearXNG search wrapper
searxng_host = config.searxng_url
search_tool = SearxSearchWrapper(searx_host=searxng_host)


# ─────────────────────────────────────────────
# State
# ─────────────────────────────────────────────

class ResearchState(TypedDict):
    topic: str                              # original user question
    queries: list[str]                      # search queries to run
    search_results: list[dict]              # raw results from Tavily
    final_report: str                       # synthesized answer


# ─────────────────────────────────────────────
# Nodes
# ─────────────────────────────────────────────

def plan_queries(state: ResearchState) -> dict:
    """Ask the LLM to generate 3 targeted search queries for the topic."""
    prompt = f"""You are a research assistant. Given this topic:
"{state['topic']}"

Generate exactly 3 specific, diverse search queries to research it thoroughly.
Return ONLY a Python list of strings, nothing else.
Example: ["query one", "query two", "query three"]"""

    response = router_llm_local.invoke([HumanMessage(content=prompt)])

    # Safely parse the list from the response
    import ast
    try:
        queries = ast.literal_eval(response.content.strip())
    except Exception:
        # Fallback: just use the topic as a single query
        queries = [state["topic"]]

    print(f"\n📋 Generated queries: {queries}")
    return {"queries": queries}


def run_searches(state: ResearchState) -> dict:
    """Run each query through SearXNG and collect results."""
    all_results = []
    for query in state["queries"]:
        print(f"🔍 Searching: {query}")
        # Use SearXNG search with default parameters (5 results)
        results = search_tool.results(query, num_results=20)
        all_results.extend(results)

    return {"search_results": all_results}


def synthesize(state: ResearchState) -> dict:
    """Feed all search results to the LLM and ask for a final research report."""

    # Format search results into readable context
    context = "\n\n".join(
        f"Source: {r.get('link', 'N/A')}\n{r.get('snippet', '')}"
        for r in state["search_results"]
    )

    prompt = f"""You are a research analyst. Using the sources below, write a thorough and
well-structured research report answering this question:

QUESTION: {state['topic']}

SOURCES:
{context}

Write a clear, detailed report with sections. Cite sources where relevant."""

    print("\n📝 Synthesizing report...")
    response = router_llm_local.invoke([
        SystemMessage(content="You are a thorough research analyst. Always base your answers on the provided sources."),
        HumanMessage(content=prompt)
    ])

    return {"final_report": response.content}


def should_refine(state: ResearchState) -> str:
    """
    Optional reflection step: check if the report is good enough.
    Returns 'done' or 'refine' to control the graph flow.

    For now, always returns 'done'. You can extend this to loop
    back and search more if the LLM flags gaps in its report.
    """
    return "done"


# ─────────────────────────────────────────────
# Graph
# ─────────────────────────────────────────────
#
#   START
#     │
#     ▼
#  plan_queries          ← LLM generates 3 search queries
#     │
#     ▼
#  run_searches          ← Tavily fetches results for each query
#     │
#     ▼
#  synthesize            ← LLM writes final report from all results
#     │
#     ▼
#  should_refine ──────► END   (extend this to loop back for deeper research)
#

builder = StateGraph(ResearchState)

builder.add_node("plan_queries", plan_queries)
builder.add_node("run_searches", run_searches)
builder.add_node("synthesize",   synthesize)

builder.add_edge(START,           "plan_queries")
builder.add_edge("plan_queries",  "run_searches")
builder.add_edge("run_searches",  "synthesize")
builder.add_conditional_edges(
    "synthesize",
    should_refine,
    {"done": END}
    # To add a refinement loop, add: "refine": "plan_queries"
)

graph = builder.compile()


# ─────────────────────────────────────────────
# Run it
# ─────────────────────────────────────────────

if __name__ == "__main__":
    topic = "What are the latest news in Toronto?"

    print(f"\n🚀 Starting research on: {topic}\n{'─'*60}")

    result = graph.invoke({
        "topic": topic,
        "queries": [],
        "search_results": [],
        "final_report": "",
    })

    print(f"\n{'─'*60}")
    print("📄 FINAL REPORT\n")
    print(result["final_report"])