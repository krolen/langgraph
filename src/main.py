"""Application entry point for the web search agent."""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.agent.config import config, AgentConfig
from src.agent.web_search_agent import create_compiled_agent
from src.agent.state import AgentState, SearchResult
from src.tools.search import SearchTool

# Configure logging
logging.basicConfig(
    level=logging.INFO if not config.debug else logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Request/Response models
class SearchRequest(BaseModel):
    """Request model for search queries."""

    query: str = Field(..., min_length=1, max_length=500, description="Search query")
    options: SearchOptions | None = Field(default=None, description="Search options")


class SearchOptions(BaseModel):
    """Options for search queries."""

    categories: list[str] = Field(
        default_factory=list, description="Search categories (e.g., 'general', 'news')"
    )
    limit: int = Field(default=10, ge=1, le=100, description="Maximum results")
    summarize: bool = Field(default=True, description="Whether to summarize results")


class SearchResultResponse(BaseModel):
    """Response model for search results."""

    answer: str = Field(..., description="Final answer or summary")
    sources: list[dict[str, Any]] = Field(default_factory=list, description="Source details")
    metadata: SearchMetadata = Field(..., description="Search metadata")


class SearchMetadata(BaseModel):
    """Metadata about the search operation."""

    search_time_ms: int = Field(..., description="Total search time in milliseconds")
    results_count: int = Field(..., description="Number of results returned")
    query: str = Field(..., description="Query that was searched")
    error: str | None = Field(default=None, description="Error message if any")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(default="healthy")
    agent: str = Field(default=config.agent_name)
    version: str = Field(default=config.agent_version)
    searxng_available: bool = Field(default=False)
    timestamp: float = Field(default_factory=time.time)


# Global agent instance
_agent = None
_search_tool = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global _agent, _search_tool

    # Startup
    logger.info(f"Starting {config.agent_name} v{config.agent_version}")
    logger.info(f"SearXNG URL: {config.searxng_url}")

    # Initialize search tool
    _search_tool = SearchTool(searxng_url=config.searxng_url, timeout=config.search_timeout)

    # Initialize agent
    _agent = create_compiled_agent(searxng_url=config.searxng_url)

    logger.info("Agent initialized successfully")
    yield

    # Shutdown
    logger.info("Shutting down agent")


# Create FastAPI application
app = FastAPI(
    title=config.agent_name,
    description=config.agent_description,
    version=config.agent_version,
    lifespan=lifespan,
)

# Configure CORS for local network access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, summary="Health check")
async def health_check():
    """Check the health of the agent and dependencies."""
    response = HealthResponse()

    # Check SearXNG availability
    try:
        if _search_tool:
            async with asyncio.timeout(5.0):
                await _search_tool.search("test", limit=1)
            response.searxng_available = True
    except Exception as e:
        logger.warning(f"SearXNG health check failed: {e}")
        response.searxng_available = False

    return response


@app.post("/search", response_model=SearchResultResponse, summary="Perform web search")
async def search(request: SearchRequest, req: Request):
    """Perform a web search using SearXNG.

    This endpoint accepts a search query and optional parameters,
    executes the search through SearXNG, and returns formatted results.
    """
    start_time = time.time()

    # Prepare search options
    options = request.options or SearchOptions()

    # Create initial state
    initial_state = AgentState(
        query=request.query,
        categories=options.categories or config.default_categories,
        limit=options.limit,
        summarize=options.summarize,
        _searxng_url=config.searxng_url,  # type: ignore
    )

    try:
        # Invoke the agent
        result = await _agent.ainvoke(initial_state)

        # Calculate total time
        total_time_ms = int((time.time() - start_time) * 1000)

        # Build response
        sources_data = []
        for source in result.sources:
            sources_data.append(
                {
                    "title": source.title,
                    "url": source.url,
                    "snippet": source.content[:200] if source.content else "",
                    "engine": source.engine,
                }
            )

        return SearchResultResponse(
            answer=result.final_answer,
            sources=sources_data,
            metadata=SearchMetadata(
                search_time_ms=total_time_ms,
                results_count=len(result.sources),
                query=request.query,
                error=result.error,
            ),
        )

    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post("/stream/search", summary="Perform web search with streaming")
async def stream_search(request: SearchRequest):
    """Perform a web search with streaming output.

    This endpoint streams intermediate results as they become available.
    """
    from fastapi.responses import StreamingResponse

    options = request.options or SearchOptions()

    initial_state = AgentState(
        query=request.query,
        categories=options.categories or config.default_categories,
        limit=options.limit,
        summarize=options.summarize,
        _searxng_url=config.searxng_url,  # type: ignore
    )

    async def stream_results():
        try:
            async for event in _agent.astream(initial_state, stream_mode="values"):
                yield f"data: {event.json()}\n\n"
        except Exception as e:
            logger.error(f"Stream search failed: {e}")
            yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        stream_results(),
        media_type="text/event-stream",
    )


@app.get("/", summary="Root endpoint")
async def root():
    """Root endpoint with API information."""
    return {
        "name": config.agent_name,
        "version": config.agent_version,
        "description": config.agent_description,
        "endpoints": {
            "health": "GET /health",
            "search": "POST /search",
            "stream_search": "POST /stream/search",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host=config.app_host,
        port=config.app_port,
        reload=config.debug,
    )
