# How to Register a LangGraph Agent with AEGRA

This guide explains how to create a simple LangGraph agent and register it with a remote AEGRA orchestration platform.

## Overview

AEGRA is a LangGraph orchestration platform that manages agents through:
- **Graphs**: The actual LangGraph agent definitions (stored in `aegra.json`)
- **Assistants**: Runtime instances of graphs that can be invoked
- **Threads**: Conversation sessions for agent execution
- **Runs**: Individual agent executions on threads

```
┌─────────────────────────────────────────────────────────────┐
│  Your Machine (192.168.0.188)                              │
│                                                             │
│  1. Define your LangGraph agent                             │
│  2. Deploy graph to AEGRA (add to aegra.json)               │
│  3. Create assistant via API                                │
│  4. Create threads and run the agent                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                             │
                             │ HTTP/REST
                             │
                    ┌────────▼────────┐
                    │    AEGRA        │
                    │  :2026          │
                    │                 │
                    │ • aegra.json    │
                    │ • Assistants    │
                    │ • Threads/Runs  │
                    └─────────────────┘
```

## Step 1: Create a Simple Agent

Create a basic LangGraph agent (`src/agent/hello_agent.py`):

```python
from typing import TypedDict
from langgraph.graph import StateGraph, END

class HelloState(TypedDict):
    name: str
    greeting: str
    status: str

def hello_node(state: HelloState) -> dict:
    name = state.get("name", "World")
    return {"greeting": f"Hello, {name}!", "status": "completed"}

def create_hello_agent() -> StateGraph:
    graph = StateGraph(HelloState)
    graph.add_node("greeting", hello_node)
    graph.set_entry_point("greeting")
    graph.add_edge("greeting", END)
    return graph

def create_compiled_hello_agent():
    return create_hello_agent().compile()
```

## Step 2: Deploy the Graph to AEGRA

AEGRA requires graphs to be registered in its `aegra.json` configuration file.

### Option A: Add to AEGRA's aegra.json (Recommended)

On the AEGRA host (192.168.0.100), edit `aegra.json`:

```json
{
  "graphs": {
    "agent": { ... },
    "hello-world-agent": {
      "import_path": "your_module.hello_agent:create_hello_agent",
      "state_schema": "your_module.hello_agent:HelloState"
    }
  }
}
```

Then restart AEGRA to load the new graph.

### Option B: Use an Existing Graph

If you can't modify AEGRA, use one of the existing graphs:
- `agent`
- `agent_hitl`
- `subgraph_agent`
- `subgraph_hitl_agent`
- `factory`

## Step 3: Create an Assistant

Once the graph is registered in AEGRA, create an assistant:

```python
from src.agent.register_with_aegra import AegraRegistrar

registrar = AegraRegistrar(
    aegra_url="http://192.168.0.100:2026",
    graph_id="hello-world-agent",  # Must exist in aegra.json
    assistant_name="hello-world-agent",
    assistant_description="Simple hello world LangGraph agent",
    endpoint_url="http://192.168.0.188:8000",
)

# Register the assistant
result = registrar.register()
print(f"Assistant ID: {registrar.assistant_id}")
```

### Assistant Creation Payload

```json
{
  "graph_id": "hello-world-agent",
  "name": "hello-world-agent",
  "description": "Simple hello world LangGraph agent",
  "config": {},
  "context": {},
  "metadata": {
    "version": "0.1.0",
    "framework": "langgraph",
    "endpoint": "http://192.168.0.188:8000"
  }
}
```

## Step 4: Create Threads and Runs

Once registered, you can create threads and run the agent:

```python
# Create a thread (conversation session)
thread = registrar.create_thread(initial_state={"name": "Alice"})
thread_id = thread["thread_id"]

# Create a run (execute the agent)
run = registrar.create_run(thread_id, {"name": "Alice"})
run_id = run["run_id"]

# Wait for completion
result = registrar.get_run_result(thread_id, run_id)
```

## Available AEGRA API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/assistants` | POST | Create an assistant |
| `/assistants` | GET | List all assistants |
| `/assistants/{id}` | GET | Get assistant details |
| `/assistants/{id}` | DELETE | Delete an assistant |
| `/threads` | POST | Create a thread |
| `/threads/{id}/state` | GET | Get thread state |
| `/threads/{id}/runs` | POST | Create a run |
| `/threads/{id}/runs/{run_id}` | GET | Get run status |

## Quick Test

Test the hello agent directly (without AEGRA):

```bash
# Start the server
APP_PORT=8001 python -m src.hello_main

# Test the invoke endpoint
curl -X POST http://localhost:8001/invoke \
  -H "Content-Type: application/json" \
  -d '{"name": "Alice"}'

# Response:
# {"greeting": "Hello, Alice!", "status": "completed", "execution_time_ms": 1}
```

## Current AEGRA Setup

- **AEGRA URL**: `http://192.168.0.100:2026`
- **Your Machine**: `http://192.168.0.188`
- **Available Graphs**: `agent`, `agent_hitl`, `subgraph_agent`, `subgraph_hitl_agent`, `factory`

To add a new graph, you need to modify AEGRA's `aegra.json` file on the AEGRA host.

## Troubleshooting

### "Graph not found in aegra.json"

This means the graph_id you're using doesn't exist in AEGRA's configuration. Either:
1. Use an existing graph_id from the available list
2. Add your graph to AEGRA's `aegra.json` and restart AEGRA

### "404 Not Found" on registration

Check that AEGRA is running:
```bash
curl http://192.168.0.100:2026/health
```

### Agent not reachable from AEGRA

Ensure your agent server uses the correct IP:
```python
endpoint_url="http://192.168.0.188:8000"  # NOT localhost
```