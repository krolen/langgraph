# How to Tell Your LLM About Your Agent

This guide covers different ways to expose your LangGraph agent to LLMs.

## Overview

```
┌─────────────┐         ┌──────────────┐         ┌─────────────┐
│    LLM      │────────▶│  Tool/Agent  │────────▶│ Your Agent  │
│ (Claude/GPT)│  Tool    │  Discovery   │  HTTP   │  :8000      │
└─────────────┐ Calling  └──────────────┘ Request └─────────────┘
              │
              └─────────────────────────────────────────────────┘
                            Returns Result
```

## Option 1: MCP Server (Recommended for Claude Desktop)

MCP (Model Context Protocol) is the standard way to expose tools to LLMs.

### 1. Install MCP dependencies

```bash
pip install mcp
```

### 2. Run the MCP server

```bash
python -m src.mcp_server
```

### 3. Configure Claude Desktop

Edit `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "web-search-agent": {
      "command": "python",
      "args": ["-m", "src.mcp_server", "/absolute/path/to/your/project"],
      "cwd": "/absolute/path/to/your/project"
    }
  }
}
```

### 4. Use in Claude

Claude will now see your tools automatically:
- `web_search(query, limit, summarize)` - Search the web
- `hello_agent(name)` - Get a greeting

---

## Option 2: Function Calling (OpenAI/Claude API)

Define your agent as a function/tool schema.

### Tool Definition

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information using SearXNG. "
                          "Use for facts, news, or anything requiring web lookup.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (e.g., 'latest AI news')"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results (1-100)",
                        "default": 10
                    }
                },
                "required": ["query"]
            }
        }
    }
]
```

### OpenAI Example

```python
from openai import OpenAI
import requests
import json

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": "What's the weather in London today?"}
    ],
    tools=tools,
    tool_choice="auto"
)

# Handle tool call
if response.choices[0].message.tool_calls:
    tool_call = response.choices[0].message.tool_calls[0]
    args = json.loads(tool_call.function.arguments)

    # Call your agent
    result = requests.post(
        "http://192.168.0.188:8000/search",
        json={"query": args["query"], "options": {"limit": args.get("limit", 10)}}
    ).json()

    # Send result back to LLM
    final = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": "What's the weather in London?"},
            response.choices[0].message,
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result)
            }
        ]
    )
    print(final.choices[0].message.content)
```

### Anthropic/Claude API Example

```python
import anthropic
import requests
import json

client = anthropic.Anthropic()

message = client.messages.create(
    model="claude-3-5-sonnet-latest",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Find info about LangGraph"}],
    tools=[{
        "name": "web_search",
        "description": "Search the web for information",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "default": 10}
            },
            "required": ["query"]
        }
    }]
)

# Handle tool use
if message.stop_reason == "tool_use":
    tool_use = message.content[0]
    args = tool_use.input

    # Call your agent
    result = requests.post(
        "http://192.168.0.188:8000/search",
        json={"query": args["query"]}
    ).json()

    # Continue conversation with result
    final = client.messages.create(
        model="claude-3-5-sonnet-latest",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": "Find info about LangGraph"},
            message,
            {
                "role": "assistant",
                "content": [{
                    "type": "tool_use",
                    "id": tool_use.id,
                    "name": "web_search",
                    "input": args
                }]
            },
            {
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": json.dumps(result)
                }]
            }
        ]
    )
    print(final.content[0].text)
```

---

## Option 3: LangChain Tools

If you're using LangChain, create a tool wrapper:

```python
from langchain.tools import tool
import requests
import json

@tool
def web_search(query: str, limit: int = 10) -> str:
    """Search the web for information.

    Args:
        query: Search query
        limit: Max results (1-100)
    """
    response = requests.post(
        "http://192.168.0.188:8000/search",
        json={"query": query, "options": {"limit": limit}}
    )
    result = response.json()
    return result["answer"]

# Use with LangChain agent
from langchain.agents import initialize_agent, AgentType
from langchain_community.llms import Ollama  # or your LLM

llm = Ollama(model="llama3.2")
agent = initialize_agent(
    tools=[web_search],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

agent.run("What is LangGraph?")
```

---

## Option 4: AEGRA Assistant

Since your agent is registered with AEGRA, LLMs can discover it through AEGRA's API:

```python
import requests

# List available assistants (agents)
response = requests.get("http://192.168.0.100:2026/assistants")
assistants = response.json()["assistants"]

for a in assistants:
    print(f"- {a['name']}: {a['description']}")
    # Check metadata for capabilities
    if "endpoint" in a.get("metadata", {}):
        print(f"  Endpoint: {a['metadata']['endpoint']}")
```

---

## Quick Test

Test your agent directly:

```bash
# Test web search agent
curl -X POST http://192.168.0.188:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "what is LangGraph", "limit": 5}'

# Test hello agent
curl -X POST http://192.168.0.188:8001/invoke \
  -H "Content-Type: application/json" \
  -d '{"name": "Alice"}'
```

## Summary

| Method | Best For | Setup Complexity |
|--------|----------|------------------|
| MCP Server | Claude Desktop, local LLMs | Low |
| Function Calling | OpenAI/Claude API integration | Medium |
| LangChain Tools | LangChain-based apps | Low |
| AEGRA | Enterprise orchestration | Medium |

**Recommendation**: Start with MCP Server for local development, then add function calling for API-based LLMs.
