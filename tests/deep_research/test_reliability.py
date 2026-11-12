from typing import Any
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from langgraph.runtime import Runtime
from src.agents.deep_research.state import ResearchState

from src.agents.deep_research.context import Context
from src.agents.deep_research.nodes import crawl_node, extraction_node, search_node, format_knowledge
from src.agents.llm_wrapper import ConcurrencyLimitedLLM, ModelRepository


@pytest.mark.asyncio
async def test_structured_output_shares_concurrency_limit() -> Any:
    active = peak = 0

    class Model:
        def with_structured_output(self, schema: Any) -> Any:
            return self

        async def ainvoke(self, value: Any) -> Any:
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.01)
            active -= 1
            return value

    wrapped = ConcurrencyLimitedLLM(Model(), asyncio.Semaphore(2))
    structured = wrapped.with_structured_output(object)
    await asyncio.gather(*(structured.ainvoke(i) for i in range(8)), wrapped.ainvoke(9))
    assert peak == 2


@pytest.mark.asyncio
async def test_url_selection_only_accepts_discovered_urls() -> Any:
    search = AsyncMock()
    search.ainvoke.return_value = {"results": [{"url": "https://example.com"}]}
    model = Mock()
    model.with_structured_output.return_value.ainvoke = AsyncMock(return_value=SimpleNamespace(
        selected_urls=["http://invented.test", "https://example.com", "https://example.com"]
    ))
    result = await search_node(ResearchState(query="test", research_plan=["test"]), Runtime(context=Context()), model, search)
    assert result["selected_urls"] == ["https://example.com"]


@pytest.mark.asyncio
async def test_failed_crawls_and_extractions_do_not_become_evidence() -> Any:
    crawl = AsyncMock()
    crawl.ainvoke.side_effect = RuntimeError("unavailable")
    crawled = await crawl_node(ResearchState(query="test", selected_urls=["https://example.com"]), Runtime(context=Context()), crawl)
    assert crawled["raw_crawl_results"] == {}
    assert crawled["errors"]
    model = Mock()
    model.with_structured_output.return_value.ainvoke = AsyncMock(side_effect=RuntimeError("bad output"))
    extracted = await extraction_node(
        ResearchState(query="test", raw_crawl_results={"https://example.com": "content"}),
        Runtime(context=Context()), model,
    )
    assert extracted["knowledge_base"] == {}
    assert extracted["errors"]


def test_evidence_budget_keeps_recent_sources() -> Any:
    formatted = format_knowledge({"https://first": "A" * 40000, "https://last": "recent finding"}, 1000)
    assert "recent finding" in formatted
    assert len(formatted) <= 1000


def test_model_repository_uses_profile_credentials(monkeypatch: Any) -> Any:
    from src.agents import config, llm_wrapper
    monkeypatch.setitem(config.MODEL_PROFILES, "test", {"model": "configured", "base_url": "http://example/v1", "api_key": "test-key"})
    factory = Mock()
    monkeypatch.setattr(llm_wrapper, "create_chat_openai", factory)
    repo = ModelRepository()
    repo.get_model_by_profile("test")
    factory.assert_called_once_with(model="configured", base_url="http://example/v1", api_key="test-key", default_headers=None)


@pytest.mark.asyncio
async def test_planner_receives_synthesis_gaps() -> None:
    from src.agents.deep_research.nodes import planner_node
    model = Mock()
    invoke = AsyncMock(return_value=SimpleNamespace(queries=["follow up"]))
    model.with_structured_output.return_value.ainvoke = invoke
    await planner_node(
        ResearchState(query="test", research_plan=["Missing battery lifespan"]),
        Runtime(context=Context()), model,
    )
    assert "Missing battery lifespan" in invoke.call_args.args[0][1].content


@pytest.mark.asyncio
async def test_extraction_caps_page_content() -> None:
    model = Mock()
    invoke = AsyncMock(return_value=SimpleNamespace(extracted_info="fact"))
    model.with_structured_output.return_value.ainvoke = invoke
    await extraction_node(
        ResearchState(query="test", raw_crawl_results={"https://source": "A" * 20 + "excluded tail"}),
        Runtime(context=Context(max_page_chars=20)), model,
    )
    assert "excluded tail" not in invoke.call_args.args[0][0].content


@pytest.mark.asyncio
async def test_empty_evidence_does_not_generate_claims() -> None:
    from src.agents.deep_research.nodes import synthesizer_node
    model = AsyncMock()
    result = await synthesizer_node(ResearchState(query="test"), Runtime(context=Context()), model)
    assert "No verified evidence" in result["final_report"]
    model.ainvoke.assert_not_called()


@pytest.mark.asyncio
async def test_local_entry_point_passes_context(monkeypatch: Any) -> None:
    from contextlib import asynccontextmanager
    from src import main
    agent = AsyncMock()
    agent.ainvoke.return_value = {"final_report": "test report"}
    ctx = Context(max_iterations=4)

    @asynccontextmanager
    async def factory(received: Context) -> Any:
        assert received is ctx
        yield agent

    monkeypatch.setattr(main, "local_graph", factory)
    assert await main.research("question", ctx) == {"final_report": "test report"}
    assert agent.ainvoke.call_args.kwargs["context"] is ctx
    assert agent.ainvoke.call_args.kwargs["config"]["recursion_limit"] > ctx.max_iterations * 5
