import logging
import re
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field

from src.agents.deep_research.context import Context
from src.agents.deep_research.state import ResearchState
from src.agents.deep_research.tools import web_search, web_crawl_multiple_urls
from src.agents.llm_wrapper import create_chat_openai

logger = logging.getLogger(__name__)

class PlannerOutput(BaseModel):
    """The output of the research planner."""
    queries: List[str] = Field(description="A list of specific search queries to fill information gaps.")

async def planner_node(state: ResearchState, runtime: Runtime[Context]) -> Dict[str, Any]:
    """
    Analyzes the current knowledge base and the original query to identify
    information gaps and generate new search queries.
    """
    query = state.query
    knowledge = state.knowledge_base

    logger.info(f"Planning research for query: {query}")
    
    # Use the model from context
    llm = create_chat_openai(model=runtime.context.model).with_structured_output(PlannerOutput)

    prompt = f"""
    You are a research planner. Your goal is to provide a comprehensive answer to the research query.
    
    Original Query: {query}
    
    Current Knowledge Base:
    {knowledge}
    
    Analyze the current knowledge and identify what is missing to fully answer the query.
    Generate a list of specific search queries to fill these gaps.
    
    If you have enough information, return an empty list [].
    """
    messages = [
        SystemMessage(content=runtime.context.system_prompt_planner),
        HumanMessage(content=prompt)
    ]

    try:
        response = await llm.ainvoke(messages)
        queries = response.queries
    except Exception as e:
        logger.error(f"Failed to get structured output from planner: {e}")
        queries = []

    logger.info(f"Planner generated {len(queries)} queries: {queries}")

    return {
        "research_plan": queries,
        "iteration_count": state.iteration_count + 1
    }

async def executor_node(state: ResearchState) -> Dict[str, Any]:
    """
    Executes the search and crawl tools based on the planner's queries.
    """
    queries = state.research_plan
    if not queries:
        logger.info("No queries to execute.")
        return {}

    logger.info(f"Executing {len(queries)} search queries")
    new_knowledge = {}
    all_new_urls = []

    for q in queries:
        logger.debug(f"Performing search for: {q}")
        search_res = await web_search.ainvoke({"query": q})

        if isinstance(search_res, str) and search_res.startswith("Error"):
            logger.warning(f"Search for '{q}' failed: {search_res}")
            continue

        urls = re.findall(r'https?://[^\s"<>]+', search_res)
        logger.debug(f"Search for '{q}' found {len(urls)} URLs")
        all_new_urls.extend(urls)

    if all_new_urls:
        unique_urls = list(set(all_new_urls))
        logger.info(f"Crawling {len(unique_urls)} unique URLs")
        crawl_res = await web_crawl_multiple_urls.ainvoke({"urls": unique_urls})
        if isinstance(crawl_res, str) and crawl_res.startswith("Error"):
            logger.warning(f"Crawl failed: {crawl_res}")
            new_knowledge["last_crawl"] = crawl_res
        else:
            new_knowledge["last_crawl"] = crawl_res
    else:
        logger.warning("No URLs found to crawl")
        new_knowledge["last_crawl"] = "No URLs found to crawl"

    return {
        "discovered_urls": all_new_urls,
        "knowledge_base": new_knowledge
    }

async def synthesizer_node(state: ResearchState, runtime: Runtime[Context]) -> Dict[str, Any]:
    """
    Processes the knowledge base to determine if the research is complete
    and generates the final report if so.
    """
    query = state.query
    knowledge = state.knowledge_base

    logger.info("Synthesizing results...")
    
    llm = create_chat_openai(model=runtime.context.model)

    prompt = f"""
    You are a research synthesizer. Based on the gathered knowledge, provide a detailed answer to the query.

    Original Query: {query}

    Gathered Knowledge:
    {knowledge}

    If the knowledge is sufficient to answer the query completely, provide the final report.
    If it is NOT sufficient, explain what is still missing and provide a summary of what was learned.

    Format your response as follows:
    COMPLETE: [Yes/No]
    REPORT: [Your detailed report or explanation of missing info along with any partial findings]
    """
    messages = [
        SystemMessage(content=runtime.context.system_prompt_synthesizer),
        HumanMessage(content=prompt)
    ]

    response = await llm.ainvoke(messages)
    content = response.content

    is_complete = "COMPLETE: Yes" in content
    report = content.split("REPORT:")[1].strip() if "REPORT:" in content else content

    if is_complete:
        logger.info("Research complete. Final report generated.")
        return {"final_report": report}
    else:
        logger.info("Research incomplete. Identified gaps.")
        return {
            "research_plan": [f"Fill gaps: {report}"],
            "final_report": report
        }
