"""Access to the NASMCP web tools without holding a run-long session."""

from langchain.mcp import MCPAdapter
from langchain_core.tools import BaseTool


class NASMCPServer:
    WEB_SEARCH = "web_search"
    WEB_CRAWL_URL = "web_crawl_url"

    def __init__(self, url: str) -> None:
        self.url = url

    async def list_tools(self) -> list[BaseTool]:
        return await MCPAdapter(self.url).list_tools()

    async def _tool(self, name: str) -> BaseTool:
        for tool in await self.list_tools():
            if tool.name == name:
                return tool
        raise ValueError(f"NASMCP server is missing required tool: {name}")

    async def search_tool(self) -> BaseTool:
        return await self._tool(self.WEB_SEARCH)

    async def crawl_tool(self) -> BaseTool:
        return await self._tool(self.WEB_CRAWL_URL)
