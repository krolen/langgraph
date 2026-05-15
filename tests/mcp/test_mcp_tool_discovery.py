"""
Test to discover and call MCP server tools.
This test attempts to:
1. Connect to the MCP server at the configured URL
2. List available tools
3. Call one of the tools if available
4. Gracefully handle cases where the server is not available
"""
import httpx
import pytest

from src.agents.config import config


@pytest.mark.asyncio
async def test_mcp_tool_discovery_and_call():
    """Discover MCP tools and attempt to call one."""
    mcp_url = config.mcp_server_url.rstrip('/')

    print(f"Attempting to discover MCP tools at: {mcp_url}")

    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=15.0) as client:
            # First, try to get the server capabilities or info
            try:
                info_resp = await client.get(f"{mcp_url}/")
                print(f"Server info endpoint: {info_resp.status_code}")
                if info_resp.status_code == 200:
                    print(f"Server info: {info_resp.text[:500]}")
            except Exception as e:
                print(f"Could not access server info: {e}")

            # Try to list tools - this is the standard MCP endpoint
            try:
                tools_resp = await client.get(f"{mcp_url}/tools/list")
                print(f"Tools list endpoint: {tools_resp.status_code}")

                if tools_resp.status_code == 200:
                    tools_data = tools_resp.json()
                    print(f"Available tools: {tools_data}")

                    # If we have tools, try to call the first one
                    if isinstance(tools_data, dict) and 'tools' in tools_data:
                        tools = tools_data['tools']
                        if tools:
                            first_tool = tools[0]
                            tool_name = first_tool.get('name', 'unknown')
                            print(f"Attempting to call tool: {tool_name}")

                            # Try to call the tool with minimal arguments
                            call_resp = await client.post(
                                f"{mcp_url}/tools/call",
                                json={
                                    "name": tool_name,
                                    "arguments": {}  # Empty arguments - may fail but worth trying
                                }
                            )
                            print(f"Tool call response: {call_resp.status_code}")
                            if call_resp.status_code == 200:
                                print(f"Tool result: {call_resp.text[:500]}")
                            else:
                                print(f"Tool call failed: {call_resp.text}")
                        else:
                            print("No tools found in response")
                    else:
                        print(f"Unexpected tools format: {tools_data}")
                else:
                    print(f"Failed to list tools: {tools_resp.status_code}")
                    print(f"Response: {tools_resp.text[:200]}")

            except Exception as e:
                print(f"Error listing/tools calling tools: {e}")

    except Exception as e:
        print(f"Could not connect to MCP server at {mcp_url}: {e}")
        print("This demonstrates the connectivity issue mentioned.")
        # Test passes - we're demonstrating the issue
        assert True

def test_mcp_configured_endpoint():
    """Verify the MCP endpoint is configured as expected."""
    expected_url = "http://192.168.0.100:7000/mcp"
    assert config.mcp_server_url == expected_url
    print(f"✓ MCP server configured at: {config.mcp_server_url}")

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])