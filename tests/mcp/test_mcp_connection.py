from unittest.mock import AsyncMock, patch

import pytest

from src.agents.deep_research.tools import web_search, web_crawl_url, web_crawl_multiple_urls


@pytest.mark.asyncio
async def test_mcp_tools_direct_call():
    """Test MCP tools by mocking the MCP client to verify the tool interface works."""

    # Mock successful MCP responses
    mock_search_result = {
        "content": [
            {
                "type": "text",
                "text": '{"results": [{"url": "https://example.com", "title": "Test", "snippet": "Test snippet"}]}'
            }
        ]
    }

    mock_crawl_result = {
        "content": [
            {
                "type": "text",
                "text": "# Test Page\n\nThis is test content."
            }
        ]
    }

    mock_multi_crawl_result = {
        "content": [
            {
                "type": "text",
                "text": '{"https://example.com": "# Test Page\\n\\nThis is test content."}'
            }
        ]
    }

    # Test web_search
    with patch('src.agents.deep_research.tools.mcp_client.call_tool', new_callable=AsyncMock) as mock_call:
        mock_call.return_value = mock_search_result
        result = await web_search.ainvoke({"query": "test query"})
        assert "test query" in result or "Test" in result or "example.com" in result
        mock_call.assert_called_once_with("web_search", {"query": "test query"})

    # Test web_crawl_url
    with patch('src.agents.deep_research.tools.mcp_client.call_tool', new_callable=AsyncMock) as mock_call:
        mock_call.return_value = mock_crawl_result
        result = await web_crawl_url.ainvoke({"url": "https://example.com"})
        assert "Test Page" in result
        mock_call.assert_called_once_with("web_crawl_url", {"url": "https://example.com"})

    # Test web_crawl_multiple_urls
    with patch('src.agents.deep_research.tools.mcp_client.call_tool', new_callable=AsyncMock) as mock_call:
        mock_call.return_value = mock_multi_crawl_result
        result = await web_crawl_multiple_urls.ainvoke({"urls": ["https://example.com"]})
        assert "Test Page" in result or "example.com" in result
        mock_call.assert_called_once_with("web_crawl_multiple_urls", {"urls": ["https://example.com"]})

@pytest.mark.asyncio
async def test_mcp_tools_error_handling():
    """Test that MCP tools handle errors gracefully."""

    # Test web_search error handling
    with patch('src.agents.deep_research.tools.mcp_client.call_tool', new_callable=AsyncMock) as mock_call:
        mock_call.side_effect = Exception("Connection failed")
        result = await web_search.ainvoke({"query": "test"})
        assert "Error performing web search" in result
        assert "Connection failed" in result

    # Test web_crawl_url error handling
    with patch('src.agents.deep_research.tools.mcp_client.call_tool', new_callable=AsyncMock) as mock_call:
        mock_call.side_effect = Exception("Crawl failed")
        result = await web_crawl_url.ainvoke({"url": "https://example.com"})
        assert "Error crawling URL" in result
        assert "Crawl failed" in result

    # Test web_crawl_multiple_urls error handling
    with patch('src.agents.deep_research.tools.mcp_client.call_tool', new_callable=AsyncMock) as mock_call:
        mock_call.side_effect = Exception("Multi-crawl failed")
        result = await web_crawl_multiple_urls.ainvoke({"urls": ["https://example.com"]})
        assert "Error crawling multiple URLs" in result
        assert "Multi-crawl failed" in result

if __name__ == "__main__":
    pytest.main([__file__, "-v"])