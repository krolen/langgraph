# Deep Research Agent Plan

## Objective
Create a research agent that leverages the `my-nas-mcp` server's web search and page crawling capabilities to perform iterative, deep research on a given query.

## Architecture

### 1. State Definition (`ResearchState`)
The agent will maintain a state that tracks the progress of the research:
- `query`: The original research goal.
- `discovered_urls`: A list of URLs found during search.
- `knowledge_base`: A collection of extracted content and summaries from crawled pages.
- `research_plan`: The current set of questions or gaps that need to be addressed.
- `final_report`: The synthesized final answer.
- `iteration_count`: To track iterations and prevent infinite loops (max 3).

### 2. Tool Integration
The agent will use wrappers for the following MCP tools:
- `web_search`: To discover relevant URLs based on a query.
- `web_crawl_url`: To extract markdown/HTML content from a specific URL.
- `web_crawl_multiple_urls`: To efficiently gather content from multiple sources.

### 3. Graph Workflow
The LangGraph will implement the following loop:
1. **Planner Node**: Analyzes the current `knowledge_base` and `query` to determine what information is missing. It updates the `research_plan` and generates search queries.
2. **Executor Node**: Executes the search and crawl tools based on the planner's output.
3. **Synthesizer Node**: Processes the new information, updates the `knowledge_base`, and determines if the research is complete.
4. **Conditional Edge**: 
    - If research is incomplete and `iteration_count` < 3 $\rightarrow$ Loop back to **Planner**.
    - If research is complete or limit reached $\rightarrow$ Proceed to **Final Report**.
5. **Reporter Node**: Generates a comprehensive final research report based on the accumulated knowledge.

## Implementation Steps

- [ ] **Define `ResearchState`**: Add the state model to `src/agents/state.py`.
- [ ] **Implement MCP Tool Wrappers**: Create `src/tools/mcp_tools.py` to interface with the MCP server.
- [ ] **Develop Research Nodes**:
    - `planner_node`: Logic for gap analysis and query generation.
    - `executor_node`: Logic for tool invocation.
    - `synthesizer_node`: Logic for information aggregation.
- [ ] **Construct the LangGraph**: Define the graph structure and conditional edges.
- [ ] **Create the Agent Entry Point**: Implement `create_research_agent()` in `src/agents/research_agent.py`.
- [ ] **Local Testing**: Create a local test suite to verify tool integration, loop restrictions (max 3), and synthesis quality.
- [ ] **Aegra Integration Testing**: Verify the agent's behavior when registered and run via the Aegra infrastructure.

## Configuration
To ensure flexibility and security, all environment-specific variables will be managed via a `.env` file and a centralized configuration module (`src/agents/config.py`).
- **Key Variables**:
    - `MCP_SERVER_URL`: The endpoint for the `my-nas-mcp` server.
    - `LLM_BASE_URL` & `LLM_MODEL`: Connection details for the local LLM.
    - `AEGRA_URL`: Endpoint for AEGRA registration.

## Integration with Aegra
The agent will be designed to be compatible with the Aegra infrastructure, ensuring it can be registered and self-hosted.
