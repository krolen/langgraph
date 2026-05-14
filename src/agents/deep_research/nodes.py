import logging
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.deep_research.state import ResearchState
from src.agents.deep_research.tools import web_search, web_crawl_multiple_urls
from src.agents.llm_wrapper import router_llm

logger = logging.getLogger(__name__)


async def planner_node(state: ResearchState) -> Dict[str, Any]:
    """
    Analyzes the current knowledge base and the original query to identify
    information gaps and generate new search queries.
    """
    query = state["query"]
    knowledge = state["knowledge_base"]

    logger.info(f"Planning research for query: {query}")
    logger.debug(f"Current knowledge base: {knowledge}")

    prompt = f"""
    You are a research planner. Your goal is to provide a comprehensive answer to the research query.
    
    Original Query: {query}
    
    Current Knowledge Base:
    {knowledge}
    
    Analyze the current knowledge and identify what is missing to fully answer the query.
    Generate a list of specific search queries to fill these gaps.
    
    Return ONLY a JSON list of strings (search queries). 
    If you have enough information, return an empty list [].
    """

    messages = [
        SystemMessage(content="You are a precise research planner. Output only valid JSON lists of strings."),
        HumanMessage(content=prompt)
    ]

    response = await router_llm.ainvoke(messages)

    try:
        # Simple extraction of JSON list from response
        import json
        # Remove potential markdown code blocks
        content = response.content.replace("```json", "").replace("```", "").strip()
        queries = json.loads(content)
    except Exception as e:
        logger.error(f"Failed to parse planner response: {e}. Content: {response.content}")
        queries = []

    logger.info(f"Planner generated {len(queries)} queries: {queries}")

    return {
        "research_plan": queries,
        "iteration_count": state["iteration_count"] + 1
    }


async def executor_node(state: ResearchState) -> Dict[str, Any]:
    """
    Executes the search and crawl tools based on the planner's queries.
    """
    queries = state["research_plan"]
    if not queries:
        logger.info("No queries to execute.")
        return {}

    logger.info(f"Executing {len(queries)} search queries")
    new_knowledge = {}
    all_new_urls = []

    for q in queries:
        # 1. Search for URLs
        logger.debug(f"Performing search for: {q}")
        search_res = await web_search.ainvoke({"query": q})
        # search_res is a string representation of the MCP result
        # In a real scenario, we'd parse the JSON to get URLs
        # For this implementation, we'll simulate extraction or use the raw string
        # assuming the MCP tool returns a list of results.

        # Simulation of parsing search results for URLs (simplified)
        # Normally we'd use a regex or json.loads if the MCP tool returns JSON
        import re
        urls = re.findall(r'https?://[^\s"<>]+', search_res)
        logger.debug(f"Search for '{q}' found {len(urls)} URLs")
        all_new_urls.extend(urls)

    if all_new_urls:
        # 2. Crawl the discovered URLs
        unique_urls = list(set(all_new_urls))
        logger.info(f"Crawling {len(unique_urls)} unique URLs")
        crawl_res = await web_crawl_multiple_urls.ainvoke({"urls": unique_urls})
        # Store crawl result in knowledge base
        # For simplicity, we map the combined crawl result to the search queries
        new_knowledge["last_crawl"] = crawl_res
        logger.debug(f"Crawl results: {crawl_res[:500]}...")
    else:
        logger.warning("No URLs found to crawl")

    return {
        "discovered_urls": all_new_urls,
        "knowledge_base": new_knowledge
    }


async def synthesizer_node(state: ResearchState) -> Dict[str, Any]:
    """
    Processes the knowledge base to determine if the research is complete
    and generates the final report if so.
    """
    query = state["query"]
    knowledge = state["knowledge_base"]

    logger.info("Synthesizing results...")
    prompt = f"""
    You are a research synthesizer. Based on the gathered knowledge, provide a detailed answer to the query.

    Original Query: {query}

    Gathered Knowledge:
    {knowledge}

    If the knowledge is sufficient to answer the query completely, provide the final report.
    If it is NOT sufficient, explain what is still missing.

    Format your response as follows:
    COMPLETE: [Yes/No]
    REPORT: [Your detailed report or explanation of missing info]
    """

    messages = [
        SystemMessage(content="You are a professional research synthesizer."),
        HumanMessage(content=prompt)
    ]

    response = await router_llm.ainvoke(messages)
    content = response.content

    is_complete = "COMPLETE: Yes" in content
    report = content.split("REPORT:")[1].strip() if "REPORT:" in content else content

    if is_complete:
        logger.info("Research complete. Final report generated.")
        return {"final_report": report}
    else:
        logger.info("Research incomplete. Identified gaps.")
        logger.debug(f"Gaps identified: {report}")
        return {"research_plan": [f"Fill gaps: {report}"]}
