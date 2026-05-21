import asyncio
import logging
import re
from typing import Any, Dict, List, Union

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field

from src.agents.deep_research.context import Context
from src.agents.deep_research.state import ResearchState
from src.agents.deep_research.utils import extract_text_from_mcp_result

logger = logging.getLogger(__name__)


class PlannerOutput(BaseModel):
    """The output of the research planner."""
    queries: List[str] = Field(description="A list of specific search queries to fill information gaps.")


class URLSelectionOutput(BaseModel):
    """The output of the URL selection process."""
    selected_urls: List[str] = Field(
        description="The top 10 most relevant URLs to crawl based on the query and search results.")


class ExtractionOutput(BaseModel):
    """The output of the information extraction process."""
    extracted_info: str = Field(description="The relevant information extracted from the page content.")


def _get_field(state: Any, field_name: str, default: Any = None) -> Any:
    """Helper to get field from state whether it is a dict or a dataclass."""
    if isinstance(state, dict):
        return state.get(field_name, default)
    return getattr(state, field_name, default)


def _get_context(runtime: Union[Runtime[Context], Any]) -> Context:
    """Helper to extract Context from runtime, handling both real and mock/fallback scenarios."""
    if hasattr(runtime, "context") and runtime.context:
        return runtime.context
    # Fallback for mock environments where context might be nested differently
    if hasattr(runtime, "execution_runtime") and hasattr(runtime.execution_runtime, "context"):
        return runtime.execution_runtime.context
    return Context()


async def planner_node(state: ResearchState, runtime: Runtime[Context], model: BaseChatModel) -> Dict[str, Any]:
    """
    Analyzes the current knowledge base and the original query to identify
    information gaps and generate new search queries.
    """
    query = _get_field(state, "query")
    knowledge = _get_field(state, "knowledge_base")
    ctx = _get_context(runtime)

    logger.info(f"Planning research for query: {query}")

    system_prompt_planner = ctx.system_prompt_planner
    llm = model.with_structured_output(PlannerOutput)

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
        SystemMessage(content=system_prompt_planner),
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
        "iteration_count": _get_field(state, "iteration_count", 0) + 1
    }


async def search_node(state: ResearchState, runtime: Runtime[Context], model: BaseChatModel, web_search: BaseTool) -> \
Dict[str, Any]:
    """
    Executes search queries and lets the LLM select the most relevant URLs.
    """
    queries = _get_field(state, "research_plan", [])
    if not queries:
        logger.info("No queries to execute.")
        return {"selected_urls": []}

    if not web_search:
        logger.error("web_search tool not available")
        return {"selected_urls": []}

    logger.info(f"Executing {len(queries)} search queries")
    all_found_urls = []
    search_results_text = []

    for q in queries:
        logger.debug(f"Performing search for: {q}")
        try:
            search_res = await web_search.ainvoke({"query": q})
            search_res_text = extract_text_from_mcp_result(search_res)

            search_results_text.append(f"Query: {q}\nResult: {search_res_text}")
            urls = re.findall(r'https?://[^\s"<>]+', search_res_text)
            all_found_urls.extend(urls)
        except Exception as e:
            logger.warning(f"Search for '{q}' failed: {e}")

    if not all_found_urls:
        logger.warning("No URLs found in search results.")
        return {"selected_urls": []}

    unique_urls = list(set(all_found_urls))

    llm = model.with_structured_output(URLSelectionOutput)

    prompt = f"""
    You are a research assistant. Given the original query and the search results, select the top 10 most relevant URLs to crawl to find the most useful information.
    
    Original Query: {_get_field(state, "query")}
    
    Search Results:
    {chr(10).join(search_results_text)}
    
    Available URLs:
    {chr(10).join(unique_urls)}
    """

    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        selected_urls = response.selected_urls[:10]
    except Exception as e:
        logger.error(f"Failed to select URLs: {e}")
        selected_urls = unique_urls[:10]

    logger.info(f"Selected {len(selected_urls)} URLs for crawling.")
    return {"selected_urls": selected_urls}


async def crawl_node(state: ResearchState, runtime: Runtime[Context], web_crawl_url: BaseTool) -> Dict[str, Any]:
    """
    Crawls the selected URLs and stores raw content.
    """
    selected_urls = _get_field(state, "selected_urls", [])

    if not selected_urls:
        logger.info("No URLs to crawl.")
        return {"raw_crawl_results": {}}

    if not web_crawl_url:
        logger.error("web_crawl_url tool not available")
        return {"raw_crawl_results": {}}

    logger.info(f"Crawling {len(selected_urls)} URLs")

    async def crawl_one(url):
        try:
            res = await web_crawl_url.ainvoke({"url": url})
            res_text = extract_text_from_mcp_result(res)
            return url, res_text
        except Exception as e:
            logger.error(f"Error crawling {url}: {e}")
            return url, f"Error: {e}"

    results = await asyncio.gather(*(crawl_one(url) for url in selected_urls))
    raw_crawl_results = {url: content for url, content in results}

    return {"raw_crawl_results": raw_crawl_results}


async def extraction_node(state: ResearchState, runtime: Runtime[Context], model: BaseChatModel) -> Dict[str, Any]:
    """
    Extracts relevant information from each crawled page.
    """
    raw_results = _get_field(state, "raw_crawl_results", {})

    if not raw_results:
        logger.info("No raw crawl results to extract from.")
        return {"knowledge_base": {}}

    llm = model.with_structured_output(ExtractionOutput)
    knowledge_base = {}

    async def extract_one(url, content):
        prompt = f"""
        You are an expert information extractor. Extract all information from the following page content that is relevant to the research query.
        
        Research Query: {_get_field(state, "query")}
        
        Page URL: {url}
        Page Content:
        {content}
        
        Extract only the facts and details that directly help answer the query. Be concise but comprehensive.
        """
        try:
            res = await llm.ainvoke([HumanMessage(content=prompt)])
            return url, res.extracted_info
        except Exception as e:
            logger.error(f"Extraction failed for {url}: {e}")
            return url, f"Error extracting info: {e}"

    batch_size = 2
    all_results = []
    items = list(raw_results.items())
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_results = await asyncio.gather(*(extract_one(url, content) for url, content in batch))
        all_results.extend(batch_results)

    knowledge_base = {url: info for url, info in all_results}

    logger.info(f"Extracted information from {len(knowledge_base)} pages.")
    return {"knowledge_base": knowledge_base}


async def synthesizer_node(state: ResearchState, runtime: Runtime[Context], model: BaseChatModel) -> Dict[str, Any]:
    """
    Processes the knowledge base to determine if the research is complete
    and generates the final report if so.
    """
    query = _get_field(state, "query")
    knowledge = _get_field(state, "knowledge_base")
    ctx = _get_context(runtime)

    logger.info("Synthesizing results...")

    max_knowledge_chars = 30000
    knowledge_str = str(knowledge)
    if len(knowledge_str) > max_knowledge_chars:
        logger.warning(f"Knowledge base too large ({len(knowledge_str)} chars), truncating to {max_knowledge_chars}")
        knowledge_str = knowledge_str[:max_knowledge_chars] + "... [truncated]"

    system_prompt_synthesizer = ctx.system_prompt_synthesizer

    prompt = f"""
    You are a research synthesizer. Based on the gathered knowledge, provide a detailed answer to the query.

    Original Query: {query}

    Gathered Knowledge:
    {knowledge_str}

    If the knowledge is sufficient to answer the query completely, provide the final report.
    If it is NOT sufficient, explain what is still missing and provide a summary of what was learned.

    Format your response as follows:
    COMPLETE: [Yes/No]
    REPORT: [Your detailed report or explanation of missing info along with any partial findings]
    """
    messages = [
        SystemMessage(content=system_prompt_synthesizer),
        HumanMessage(content=prompt)
    ]

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = await model.ainvoke(messages)
            break
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"Synthesizer failed after {max_retries} attempts: {e}")
                return {
                    "final_report": f"Error during synthesis: {str(e)}",
                    "research_plan": []
                }
            logger.warning(f"Synthesizer attempt {attempt + 1} failed: {e}. Retrying...")
            await asyncio.sleep(1 * (attempt + 1))

    content = response.content

    is_complete = "COMPLETE: Yes" in content
    report = content.split("REPORT:")[1].strip() if "REPORT:" in content else content

    if is_complete:
        logger.info("Research complete. Final report generated.")
        return {"final_report": report}
    else:
        logger.info("Research incomplete. Identified gaps.")
        return {
            "research_plan": [f"Fill gaps identified: {report}"],
            "final_report": report
        }
