from typing import Annotated

from langchain_core.tools import tool
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_community.utilities import SearxSearchWrapper
import requests
from bs4 import BeautifulSoup
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import create_async_playwright_browser
import asyncio



class AgentState(TypedDict):
    messages: Annotated[list, add_messages]     # conversation (human/AI messages)
    search_results: Annotated[list, ...]        # list of search result snippets/URLs
    page_content: str                           # text from fetched page
    browser_content: str                        # text from browser-extracted page


async_browser = create_async_playwright_browser()
toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=async_browser)
browser_tools = toolkit.get_tools()
# Choose relevant tools:
navigate = next(t for t in browser_tools if t.name=="navigate_browser")
extract_text = next(t for t in browser_tools if t.name=="extract_text")

@tool
def browser_navigate(cmd: str) -> str:
    """Navigate/browser tool wrapper; 'cmd' can be a URL or command."""
    # Very simplistic usage: interpret cmd as URL for now.
    async def nav():
        page = await async_browser.new_page()
        await page.goto(cmd)
        text = await page.content()
        await page.close()
        return text
    return asyncio.get_event_loop().run_until_complete(nav())

@tool
def extract_page(url: str) -> str:
    """Download and extract visible text from the given URL."""
    res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(res.text, "lxml")
    return soup.get_text()

searx = SearxSearchWrapper(searx_host="http://localhost:8888")
@tool
def web_search(query: str) -> str:
    """Query SearxNG and return text results."""
    return searx.run(query)  # uses SEARXNG_HOST if set【26†L127-L130】.
