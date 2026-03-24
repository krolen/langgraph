"""FastAPI application exposing the hello world agent with AEGRA registration.

This is a complete example showing:
1. Creating a simple LangGraph agent
2. Exposing it via FastAPI endpoints
3. Registering it with a remote AEGRA instance
4. Sending periodic heartbeats to keep the registration alive
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.agents.hello_agent import create_compiled_hello_agent
from src.agents.register_with_aegra import AegraRegistrar, AegraRegistrationError
from src.agents.config import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Request/Response models
class HelloRequest(BaseModel):
    """Request model for hello endpoint."""

    name: str = Field(default="World", min_length=1, max_length=100)


class HelloResponse(BaseModel):
    """Response model for hello endpoint."""

    greeting: str = Field(..., description="The greeting message")
    status: str = Field(..., description="Execution status")
    execution_time_ms: int = Field(..., description="Time taken to execute")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(default="healthy")
    agent: str = Field(default="hello-world-agent")
    version: str = Field(default="0.1.0")
    aegra_registered: bool = Field(default=False)
    timestamp: float = Field(default_factory=time.time)


# Global instances
_agent = None
_aegra_registrar = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager with AEGRA registration."""
    global _agent, _aegra_registrar

    # Startup
    logger.info("Starting hello-world-agent v0.1.0")

    # Create the agent
    _agent = create_compiled_hello_agent()
    logger.info("Agent created successfully")

    # Register with AEGRA
    # Note: graph_id must exist in AEGRA's aegra.json
    # Use 'agent' for now, or add 'hello-world-agent' to aegra.json on the AEGRA host
    _aegra_registrar = AegraRegistrar(
        aegra_url=config.aegra_url,
        graph_id="agent",  # Must match a graph in AEGRA's aegra.json
        assistant_name="hello-world-agent",
        assistant_version="0.1.0",
        assistant_description="Simple hello world LangGraph agent",
        endpoint_url=f"http://192.168.0.188:{config.app_port}",
        api_key=config.aegra_api_key,
    )

    try:
        result = _aegra_registrar.register()
        logger.info(f"Registered with AEGRA: {result}")
    except AegraRegistrationError as e:
        logger.warning(f"Could not register with AEGRA: {e}")
        logger.info(f"AEGRA URL: {config.aegra_url}")

    # Start heartbeat task
    heartbeat_task = asyncio.create_task(_heartbeat_loop())

    yield

    # Shutdown
    logger.info("Shutting down hello-world-agent")
    heartbeat_task.cancel()

    try:
        _aegra_registrar.unregister()
        logger.info("Unregistered from AEGRA")
    except AegraRegistrationError:
        pass


# Create FastAPI application
app = FastAPI(
    title="Hello World Agent",
    description="Simple LangGraph agent for AEGRA demonstration",
    version="0.1.0",
    lifespan=lifespan,
)


async def _heartbeat_loop():
    """Send periodic heartbeats to AEGRA (check assistant status)."""
    while True:
        try:
            await asyncio.sleep(60)  # Every 60 seconds
            if _aegra_registrar:
                # Check if assistant still exists
                _aegra_registrar.get_assistant()
                logger.debug("Heartbeat check completed")
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.warning(f"Heartbeat check failed: {e}")


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check the health of the agent."""
    return HealthResponse(
        status="healthy",
        aegra_registered=_aegra_registrar is not None,
    )


@app.post("/invoke", response_model=HelloResponse, tags=["Agent"])
async def invoke_agent(request: HelloRequest):
    """Invoke the hello world agent.

    This is the main endpoint that AEGRA will use to execute the agent.
    """
    start_time = time.time()

    try:
        result = _agent.invoke({"name": request.name})
        execution_time_ms = int((time.time() - start_time) * 1000)

        return HelloResponse(
            greeting=result["greeting"],
            status=result["status"],
            execution_time_ms=execution_time_ms,
        )

    except Exception as e:
        logger.error(f"Agent execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/stream", tags=["Agent"])
async def stream_agent(request: HelloRequest):
    """Stream the hello world agent execution.

    Returns Server-Sent Events (SSE) with intermediate states.
    """
    from fastapi.responses import StreamingResponse

    async def stream_results():
        try:
            async for event in _agent.astream({"name": request.name}, stream_mode="values"):
                yield f"data: {event.json()}\n\n"
        except Exception as e:
            yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(stream_results(), media_type="text/event-stream")


@app.get("/", tags=["Info"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "hello-world-agent",
        "version": "0.1.0",
        "description": "Simple LangGraph agent for AEGRA demonstration",
        "endpoints": {
            "health": "GET /health",
            "invoke": "POST /invoke",
            "stream": "POST /stream",
        },
        "aegra": {
            "url": config.aegra_url,
            "registered": _aegra_registrar is not None,
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.hello_main:app",
        host=config.app_host,
        port=config.app_port,
        reload=config.debug,
    )