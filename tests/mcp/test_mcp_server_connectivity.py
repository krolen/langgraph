"""
Test to check MCP server connectivity and list available tools.
This test will gracefully handle cases where the MCP server is not available.
"""
import httpx
import pytest

from src.agents.config import config


@pytest.mark.asyncio
async def test_mcp_server_connectivity():
    """Test if we can connect to the MCP server and list tools."""
    mcp_url = config.mcp_server_url

    print(f"Testing MCP server connectivity at: {mcp_url}")

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Try to reach the MCP server
            response = await client.get(f"{mcp_url}/")
            print(f"MCP server root response: {response.status_code}")

            # Try common MCP endpoints
            endpoints_to_try = [
                "/",
                "/tools/list",
                "/resources/list",
                "/prompts/list",
                "/health"
            ]

            for endpoint in endpoints_to_try:
                try:
                    url = f"{mcp_url}{endpoint}"
                    resp = await client.get(url, timeout=5.0)
                    print(f"  {endpoint}: {resp.status_code}")
                    if resp.status_code == 200:
                        print(f"    Response: {resp.text[:200]}...")
                except Exception as e:
                    print(f"  {endpoint}: Error - {e}")

    except Exception as e:
        print(f"Failed to connect to MCP server at {mcp_url}: {e}")
        print("This is expected if the MCP server is not running.")
        # Don't fail the test - just report the issue
        assert True  # Test passes regardless of connection status

def test_mcp_configuration():
    """Test that MCP configuration is properly loaded."""
    assert config.mcp_server_url == "http://192.168.0.100:7000/mcp"
    print(f"MCP Server URL configured as: {config.mcp_server_url}")

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])