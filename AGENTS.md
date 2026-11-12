# Agent Instructions: LangGraph Web Search Agent

## Current Project Scope
- **Goal**: Build a LangGraph research agent that runs locally.
- **Out of scope for now**: Do not deploy, sync, or register anything with AEGRA. In particular, do not run `bin/deploy.py`, use the AEGRA mount, update `aegra.json` for deployment, or create/update AEGRA registration manifests unless the user explicitly changes this scope.

## Local-First Verification
- Prefer locally available evidence before online research: inspect installed Python packages, their bundled documentation and metadata, repository lockfiles, and local command help.
- For Python dependency versions and compatibility, use `uv` first (for example, `uv pip index versions <package>` and `uv lock --upgrade --dry-run`). Use `pip` only when `uv` is unavailable or cannot provide the required information.
- Do not install, upgrade, lock, or sync dependencies merely to inspect them. Use read-only or dry-run commands unless the user asks for a dependency change.
- Use online sources only when the required information is not available locally or requires current registry/release data.

## Developer Commands
- **Install**: `uv pip install -e .`
- **Test**: `pytest tests/` (async tests enabled via `pytest-asyncio`)
- **Typecheck**: `mypy .`
<!-- AEGRA deployment is intentionally disabled for the current local-development scope.
- Deployment: `python bin/deploy.py <agent_dir> [--register]`
-->

## Architecture & Conventions
- **Core Agents**: Located in `src/agents/`. 
- **Primary Agent**: `deep_research` (defined in `src/agents/deep_research/graph.py`).
- **Graph Registry**: `langgraph.json` maps agent names to their compiled graph factories.
- **LLM Management**: Use `src/agents/llm_wrapper.py` (`model_repo`) to retrieve models by profile.
- **Web Search**: Always use NASMCP tools (`NASMCP_web_search`, `NASMCP_web_crawl_url`) for all web searching tasks.
- **Environment**: Configuration is loaded from `.env` (local agent, LLM, and SearXNG settings).

## Quirks & Gotchas
- **Server Start**: The `README.md` command `uv run uvicorn src.main:app` is currently broken as `src/main.py` is missing from the repository.
- **Runtime Context**: The `graph` factory in `src/agents/deep_research/graph.py` expects a `ServerRuntime` with a `context` object.
