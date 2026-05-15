import logging
import re
import asyncio
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
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

class URLSelectionOutput(BaseModel):
    """The output of the URL selection process."""
    selected_urls: List[str] = Field(description="The top 10 most relevant URLs to crawl based on the query and search results.")

class ExtractionOutput(BaseModel):
    """The output of the information extraction process."""
    extracted_info: str = Field(description="The relevant information extracted from the page content.")

async def planner_node(state: ResearchState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Analyzes the current knowledge base and the original query to identify
    information gaps and generate new search queries.
    """
    query = state.query
    knowledge = state.knowledge_base

    logger.info(f"Planning research for query: {query}")
    
    configurable = config.get("configurable", {})
    model = configurable.get("model", "openai/gpt-4o")
    system_prompt_planner = configurable.get("system_prompt_planner", "You are a precise research planner. Output only valid JSON lists of strings.")

    llm = create_chat_openai(model=model).with_structured_output(PlannerOutput)

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
        "iteration_count": state.iteration_count + 1
    }

async def search_node(state: ResearchState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Executes search queries and lets the LLM select the most relevant URLs.
    """
    queries = state.research_plan
    if not queries:
        logger.info("No queries to execute.")
        return {"selected_urls": []}

    logger.info(f"Executing {len(queries)} search queries")
    all_found_urls = []
    search_results_text = []

    for q in queries:
        logger.debug(f"Performing search for: {q}")
        search_res = await web_search.ainvoke({"query": q})
        if isinstance(search_res, str) and search_res.startswith("Error"):
            logger.warning(f"Search for '{q}' failed: {search_res}")
            continue
        
        search_results_text.append(f"Query: {q}\nResult: {search_res}")
        urls = re.findall(r'https?://[^\s"<>]+', search_res)
        all_found_urls.extend(urls)

    if not all_found_urls:
        logger.warning("No URLs found in search results.")
        return {"selected_urls": []}

    # Deduplicate
    unique_urls = list(set(all_found_urls))
    
    # Ask model to choose top 10
    configurable = config.get("configurable", {})
    model_name = configurable.get("model", "openai/gpt-4o")
    llm = create_chat_openai(model=model_name).with_structured_output(URLSelectionOutput)

    prompt = f"""
    You are a research assistant. Given the original query and the search results, select the top 10 most relevant URLs to crawl to find the most useful information.
    
    Original Query: {state.query}
    
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

async def crawl_node(state: ResearchState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Crawls the selected URLs and stores raw content.
    """
    # We need selected_urls from the state or the previous node output
    # Since we are using StateGraph, we need to make sure selected_urls is in state or passed
    # Wait, I didn't add selected_urls to ResearchState. I should.
    # But for now, I'll check if it's in state.
    
    # Let's assume I'll update state.py to include selected_urls.
    selected_urls = getattr(state, "selected_urls", []) 
    # Actually ResearchState is a dataclass, but in LangGraph it's often passed as a dict
    if isinstance(state, dict):
        selected_urls = state.get("selected_urls", [])
    else:
        selected_urls = getattr(state, "selected_urls", [])

    if not selected_urls:
        logger.info("No URLs to crawl.")
        return {"raw_crawl_results": {}}

    logger.info(f"Crawling {len(selected_urls)} URLs")
    
    # Using web_crawl_multiple_urls for efficiency
    try:
        crawl_res = await web_crawl_multiple_urls.ainvoke({"urls": selected_urls})
        # The current web_crawl_multiple_urls returns a string (likely a combined markdown)
        # To get per-URL results, we might need to call web_crawl_url in parallel.
        # Let's check tool definition: web_crawl_multiple_urls(urls: List[str]) -> str
        # If it returns a combined string, we can't easily map it back.
        # Let's switch to parallel individual calls for better control.
        
        from src.agents.deep_research.tools import web_crawl_url
        
        async def crawl_one(url):
            try:
                res = await web_crawl_url.ainvoke({"url": url})
                return url, res
            except Exception as e:
                logger.error(f"Error crawling {url}: {e}")
                return url, f"Error: {e}"

        results = await asyncio.gather(*(crawl_one(url) for url in selected_urls))
        raw_crawl_results = {url: content for url, content in results}
        
    except Exception as e:
        logger.error(f"Crawl process failed: {e}")
        raw_crawl_results = {}

    return {"raw_crawl_results": raw_crawl_results}

async def extraction_node(state: ResearchState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Extracts relevant information from each crawled page.
    """
    raw_results = getattr(state, "raw_crawl_results", {})
    if isinstance(state, dict):
        raw_results = state.get("raw_crawl_results", {})
        
    if not raw_results:
        logger.info("No raw crawl results to extract from.")
        return {"knowledge_base": {}}

    configurable = config.get("configurable", {})
    model_name = configurable.get("model", "openai/gpt-4o")
    llm = create_chat_openai(model=model_name).with_structured_output(ExtractionOutput)

    knowledge_base = {}
    
    # Process pages in parallel
    async def extract_one(url, content):
        prompt = f"""
        You are an expert information extractor. Extract all information from the following page content that is relevant to the research query.
        
        Research Query: {state.query if isinstance(state, dict) else state.query}
        
        Page URL: {url}
        Page Content:
        {content}
        
        Extract only the facts and details that directly help answer the query. Be concise but comprehensive.
        """
        try:
            # ainvoke is wrapped by ConcurrencyLimitedLLM
            res = await llm.ainvoke([HumanMessage(content=prompt)])
            return url, res.extracted_info
        except Exception as e:
            logger.error(f"Extraction failed for {url}: {e}")
            return url, f"Error extracting info: {e}"

    # To avoid overwhelming the provider even with the semaphore, we can process in smaller batches
    # or just rely on the semaphore. Given the 504s and 400s, let's use a smaller batch size.
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

async def synthesizer_node(state: ResearchState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Processes the knowledge base to determine if the research is complete
    and generates the final report if so.
    """
    query = state.query if isinstance(state, dict) else state.query
    knowledge = state.knowledge_base if isinstance(state, dict) else state.knowledge_base

    logger.info("Synthesizing results...")
    
    # Limit knowledge size to avoid 400/504 errors from provider
    # Simple truncation: limit total characters
    max_knowledge_chars = 30000
    knowledge_str = str(knowledge)
    if len(knowledge_str) > max_knowledge_chars:
        logger.warning(f"Knowledge base too large ({len(knowledge_str)} chars), truncating to {max_knowledge_chars}")
        knowledge_str = knowledge_str[:max_knowledge_chars] + "... [truncated]"
    else:
        knowledge_str = knowledge_str

    configurable = config.get("configurable", {})
    model = configurable.get("model", "openai/gpt-4o")
    system_prompt_synthesizer = configurable.get("system_prompt_synthesizer", "You are a professional research synthesizer.")

    llm = create_chat_openai(model=model)

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

    # Simple retry loop for synthesis
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = await llm.ainvoke(messages)
            break
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"Synthesizer failed after {max_retries} attempts: {e}")
                # Fallback: return a report stating the error
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
