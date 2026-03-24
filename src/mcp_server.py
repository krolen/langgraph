"""MCP (Model Context Protocol) server for the web search agent.

This exposes the agent as a tool that LLMs can discover and call via MCP.
Compatible with Claude Desktop, Zed, and other MCP clients.

Run with: python -m src.mcp_server
"""

import httpx
from mcp.server.fastmcp import FastMCP

# Create MCP server
mcp = FastMCP("web-search-agent")

# Agent endpoint URLs
SEARCH_AGENT_URL = "http://192.168.0.188:8000"
HELLO_AGENT_URL = "http://192.168.0.188:8001"


@mcp.tool()
def web_search(query: str, limit: int = 10, summarize: bool = True) -> str:
    """Search the web for information using SearXNG.

    Use this tool when you need to find current information, facts,
    news, or answers that require searching the web.

    Args:
        query: The search query (e.g., 'latest AI developments 2024')
        limit: Maximum number of results to return (1-100)
        summarize: Whether to provide a summarized answer

    Returns:
        Search results with sources and optional summary
    """
    try:
        response = httpx.post(
            f"{SEARCH_AGENT_URL}/search",
            json={
                "query": query,
                "options": {
                    "limit": min(limit, 100),
                    "summarize": summarize,
                }
            },
            timeout=30.0,
        )
        response.raise_for_status()
        result = response.json()

        # Format response
        output = [f"**Answer:** {result['answer']}\n"]
        output.append("\n**Sources:**\n")
        for i, source in enumerate(result.get("sources", [])[:5], 1):
            output.append(f"{i}. [{source['title']}]({source['url']})")
            if source.get("snippet"):
                output.append(f"   {source['snippet'][:100]}...")

        return "\n".join(output)

    except httpx.HTTPError as e:
        return f"Search failed: {e}"


@mcp.tool()
def hello_agent(name: str = "World") -> str:
    """Get a greeting from the hello world agent.

    A simple demo agent that returns personalized greetings.

    Args:
        name: Name to greet (default: "World")

    Returns:
        Greeting message
    """
    try:
        response = httpx.post(
            f"{HELLO_AGENT_URL}/invoke",
            json={"name": name},
            timeout=10.0,
        )
        response.raise_for_status()
        result = response.json()
        return result["greeting"]
    except httpx.HTTPError as e:
        return f"Greeting failed: {e}"


if __name__ == "__main__":
    mcp.run()
