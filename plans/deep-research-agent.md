# Deep Research Agent: Current Design

## Objective
Provide a locally runnable LangGraph research agent that uses MCP-provided web search and page crawling to research a query and produce a synthesized report. AEGRA deployment and registration are out of scope.

## Architecture

### State (`ResearchState`)
The graph maintains:
- `query`: The original research goal.
- `discovered_urls`: Reserved for discovered URLs; currently unused.
- `knowledge_base`: A URL-keyed collection of extracted information, merged across graph updates.
- `raw_crawl_results`: The latest URL-to-page-content crawl results.
- `research_plan`: The current set of questions or gaps that need to be addressed.
- `final_report`: The synthesized final answer.
- `iteration_count`: A guard against unbounded research; its default limit is three iterations.

### MCP Tool Integration
The graph factory connects to the configured MCP endpoint when `enable_mcp` is true and exposes these discovered tools to nodes:
- `web_search`: Discover search-result URLs for each planner query.
- `web_crawl_url`: Fetch selected pages concurrently.

### Graph Workflow
`planner → search → crawl → extract → synthesizer`

The synthesizer conditionally routes to the planner or `END`. Role-specific LLM profiles are supplied through `Context`, and `ModelRepository` caches the underlying clients while limiting concurrent async requests per model.

## Current Implementation Notes

- The mock integration test covers factory setup, MCP tool discovery, and one complete graph pass.
- The live test is separate and connects to configured LLM and MCP services.
- An incomplete synthesizer response loops back to the planner until `max_iterations` is reached, then returns a partial report.
- Run locally with `uv run python -m src.main "question"`. The graph registry points to the deep research factory; no HTTP API is implemented.

## Configuration
Environment-specific values are read from `.env` by `src/agents/config.py`, and per-run behavior is represented by `Context`.
- **Key Variables**:
    - `MCP_SERVER_URL`: MCP endpoint that supplies search and crawl tools.
    - `LLM_BASE_URL` & `LLM_MODEL`: Connection details for the local LLM.
