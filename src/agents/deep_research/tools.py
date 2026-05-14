import httpx
from typing import List, Annotated
from langchain.tools import tool
from src.agents.config import config

# Note: In a production MCP environment, this would use an MCP Client.
# For this implementation, we assume the MCP server is accessible via an API 
# or provided by the Aegra runtime.

class MCPClient:
    """Client to communicate with the my-nas-mcp server."""
    def __init__(self, base_url: str = None):
        self.base_url = base_url or config.mcp_server_url

    async def call_tool(self, tool_name: str, arguments: dict):
        """Call a specific MCP tool."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/tools/call",
                json={"tool_name": tool_name, "arguments": arguments},
                timeout=30.0
            )
            response.raise_for_status()
            return response.json()

mcp_client = MCPClient()

@tool
async def web_search(query: str) -> str:
    """
    Perform a web search to find relevant URLs and snippets.
    Use this when you need to discover information or find sources.
    """
    try:
        result = await mcp_client.call_tool("web_search", {"query": query})
        return str(result)
    except Exception as e:
        return f"Error performing web search: {str(e)}"

@tool
async def web_crawl_url(url: str) -> str:
    """
    Crawl a specific URL and extract its content as markdown.
    Use this when you have a specific URL and need the full content of the page.
    """
    try:
        result = await mcp_client.call_tool("web_crawl_url", {"url": url})
        return str(result)
    except Exception as e:
        return f"Error crawling URL {url}: {str(e)}"

@tool
async def web_crawl_multiple_urls(urls: List[str]) -> str:
    """
    Crawl multiple URLs in parallel.
    Use this to efficiently gather content from several sources discovered during search.
    """
    try:
        result = await mcp_client.call_tool("web_crawl_multiple_urls", {"urls": urls})
        return str(result)
    except Exception as e:
        return f"Error crawling multiple URLs: {str(e)}"

# List of available tools for the agent
RESEARCH_TOOLS = [web_search, web_crawl_url, web_crawl_multiple_urls]