# LangGraph Research Agent

A local research workflow using configured OpenAI-compatible LLM and MCP services.

## Run locally

```bash
uv sync
cp .env.example .env
# Edit .env with your LLM model, credentials and MCP endpoint.
uv run python -m src.main "What changed in battery recycling technology?"
```

Use `--max-iterations 3` to bound research, `--profile local` to select a model profile, and `--json` to include evidence and tool errors. The process runs locally; the configured services must be reachable. There is currently no HTTP server or Docker setup.

`langgraph.json` also identifies the `deep_research` graph factory for a compatible LangGraph server. Deployment and AEGRA registration are outside the current scope.

## Configuration

See `.env.example`. `LLM_BASE_URL`, `LLM_MODEL`, and `LLM_API_KEY` configure the local profile. `LLM_BASE_URL_LOCAL` optionally overrides its endpoint. The `fabric` and `openrouter` profiles use routing headers and their respective model names. `MCP_SERVER_URL` must expose `web_search` and `web_crawl_url`.

## Workflow

`planner → search → crawl → extract → synthesizer`

Incomplete research returns to the planner with identified gaps. At the iteration limit, the latest partial report is returned. Model calls share a concurrency limit, including calls with structured output. URL selection is restricted to search results. Failed crawls and extractions are recorded as errors rather than evidence.

Extraction limits page text to the configured character budget. Synthesis shares its evidence budget across sources and requests inline citations; a source list is appended to every evidence-based report. These limits can omit relevant material; citations and generated claims still need review. The optional embedding analyzer loads its model only when used and is not part of the active graph.

## Development

```bash
uv sync --extra dev
uv run pytest tests/
uv run mypy .
# Explicitly opt into requests to real LLM and MCP services:
RUN_LIVE_TESTS=1 uv run pytest tests/deep_research/test_live.py
```

The ordinary suite uses mocks and skips live service checks.
