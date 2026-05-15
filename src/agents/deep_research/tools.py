from contextlib import AsyncExitStack
from typing import List, Optional

from langchain.tools import tool
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from src.agents.config import config


class MCPClient:
    """Client to communicate with the my-nas-mcp server using MCP protocol."""

    def __init__(self, server_url: str = None):
        self.server_url = server_url or config.mcp_server_url
        self.session: Optional[ClientSession] = None
        self.exit_stack: AsyncExitStack = AsyncExitStack()

    async def connect(self):
        """Connect to the MCP server."""
        # Parse the URL to determine transport type
        if self.server_url.startswith("http"):
            # Use streamable HTTP transport for MCP over HTTP
            # streamablehttp_client returns (read_stream, write_stream, get_session_id)
            read_stream, write_stream, _ = await self.exit_stack.enter_async_context(
                streamablehttp_client(self.server_url)
            )
        else:
            # Fallback to stdio for local processes (not expected here)
            raise ValueError(f"Unsupported server URL: {self.server_url}")

        self.session = await self.exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )

        # Initialize the connection
        await self.session.initialize()

    async def call_tool(self, tool_name: str, arguments: dict):
        """Call a specific MCP tool."""
        if not self.session:
            await self.connect()

        result = await self.session.call_tool(tool_name, arguments)
        return result

    async def cleanup(self):
        """Clean up resources."""
        await self.exit_stack.aclose()


# Global MCP client instance
mcp_client = MCPClient()


@tool
async def web_search(query: str) -> str:
    """
    Perform a web search to find relevant URLs and snippets.
    Use this when you need to discover information or find sources.
    """
    try:
        result = await mcp_client.call_tool("web_search", {"query": query})
        # Extract text content from MCP result
        if hasattr(result, 'content'):
            return str(result.content)
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
        # Extract text content from MCP result
        if hasattr(result, 'content'):
            return str(result.content)
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
        # Extract text content from MCP result
        if hasattr(result, 'content'):
            return str(result.content)
        return str(result)
    except Exception as e:
        return f"Error crawling multiple URLs: {str(e)}"


# List of available tools for the agent
RESEARCH_TOOLS = [web_search, web_crawl_url, web_crawl_multiple_urls]
