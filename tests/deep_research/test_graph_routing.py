from typing import Any
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from langgraph.runtime import Runtime
from langchain_core.messages import AIMessage

from src.agents.deep_research.context import Context
from src.agents.deep_research.graph import graph, should_continue
from src.agents.deep_research.nodes import synthesizer_node
from src.agents.deep_research.state import ResearchState


@pytest.mark.asyncio
async def test_incomplete_research_loops_until_iteration_limit() -> Any:
    ctx = Context(max_iterations=2)
    runtime = Runtime(context=ctx)
    model = AsyncMock()
    model.ainvoke.return_value = AIMessage(content="COMPLETE: No\nREPORT: Need another source.")

    first = await synthesizer_node(ResearchState(query="test", iteration_count=1, knowledge_base={"https://source.test": "partial evidence"}), runtime, model)
    assert first["final_report"] is None
    assert await should_continue(ResearchState(query="test", iteration_count=1, **first), runtime) == "planner"

    last = await synthesizer_node(ResearchState(query="test", iteration_count=2, knowledge_base={"https://source.test": "partial evidence"}), runtime, model)
    assert last["final_report"].startswith("Need another source.")
    assert "https://source.test" in last["final_report"]
    assert await should_continue(ResearchState(query="test", iteration_count=2, **last), runtime) == "end"


@pytest.mark.asyncio
async def test_graph_inspection_skips_mcp_connection() -> Any:
    runtime: Any = SimpleNamespace(execution_runtime=None)
    compiled = object()
    with patch("src.agents.deep_research.graph._build_graph", return_value=compiled) as build:
        assert graph({}, runtime) is compiled
    assert isinstance(build.call_args.args[0], Context)
