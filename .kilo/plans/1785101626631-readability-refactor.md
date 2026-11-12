# Deep Research Agent — Readability Refactor

## Goal
Improve code readability and maintainability of the `src/agents/deep_research/` package by splitting overloaded modules, eliminating duplication, and extracting embedded content.

## Decisions

| Decision | Choice |
|---|---|
| Scope | Deep refactor: split nodes.py, extract prompts, consolidate models, dedup helpers, expose public API |
| Prompts | New `prompts.py` module with template functions |
| analyzer.py | **Leave untouched** — may have been used in prior iterations
| Post-edit check | Run IntelliJ MCP lint check after each code change

## Task List

### 1. Move Pydantic output schemas from `nodes.py` → `models.py`
- Move `PlannerOutput`, `URLSelectionOutput`, `ExtractionOutput` into `models.py`
- Update `nodes.py` imports to pull from `models`

### 2. Create `prompts.py` — extract all inline prompt templates
- Create `src/agents/deep_research/prompts.py` with template functions:
  - `build_planner_prompt(query, knowledge, system_prompt)` → list of messages
  - `build_url_selection_prompt(query, search_results_text, unique_urls)` → list of messages
  - `build_extraction_prompt(query, url, content)` → list of messages
  - `build_synthesizer_prompt(query, knowledge_str, system_prompt)` → list of messages
- Each function returns a ready-to-invoke message list (SystemMessage + HumanMessage)
- Update each node to call the corresponding prompt builder

### 3. Deduplicate helpers — single source of truth
- Keep `_get_field` only in `graph.py` (needed for `should_continue` conditional edge which may receive dict states from tests)
- Remove `_get_field` from `nodes.py` — replace all calls with direct attribute access on `ResearchState` (e.g. `state.query` instead of `_get_field(state, "query")`)
- Fix tests: replace `asdict(ResearchState(...))` with `ResearchState(...)` so nodes receive proper typed state
- Remove `_get_context` from `nodes.py` — use `runtime.context` directly

### 4. Split `nodes.py` into per-node files
- `planner.py` — `planner_node` + imports from `prompts`, `models`
- `search.py` — `search_node` + imports from `prompts`, `models`
- `crawl.py` — `crawl_node`
- `extraction.py` — `extraction_node` + imports from `prompts`, `models`
- `synthesizer.py` — `synthesizer_node` + imports from `prompts`, `models`
- Keep a thin `nodes.py` that re-exports all node functions for import compatibility

### 5. Update `graph.py`
- Import nodes from the new modules (via `nodes.py` re-exports or direct)
- Remove the now-obsolete `_get_field` usage

### 6. Populate `__init__.py`
- Export public API: `ResearchState`, `Context`, `graph`, and the compiled graph factory

### 7. Update tests
- Ensure imports still resolve (test_integration.py, test_live.py import from `nodes` and `graph`)
- Run `pytest tests/` to verify no regressions
- Run `mypy .` to verify type correctness

### 8. Post-refactor lint check
- After each file change, invoke IntelliJ MCP for error/warning inspection
- Fix any auto-fixable issues; note unfixable issues

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Breaking test imports | `nodes.py` re-exports all functions, preserving import paths |
| `asdict(ResearchState)` in tests breaks with typed access | Replace `asdict(...)` with `ResearchState(...)` so nodes receive typed state |
| `_get_field` removal breaks dict-state paths | Keep `_get_field` in `graph.py` for `should_continue`. Fix tests to pass `ResearchState(...)` directly instead of `asdict(...)`, so nodes receive typed state. |

## Validation
1. `pytest tests/deep_research/test_integration.py` — passes with mocked LLM/tools
2. `mypy .` — no type errors
3. Module structure is flat and each file under 100 lines