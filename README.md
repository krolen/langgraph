# LangGraph Web Search Agent

A LangGraph-based web search agent that provides privacy-focused web search capabilities to local LLMs using SearXNG.

## Features

- Privacy-focused web searches via SearXNG
- LangGraph agent workflow with query processing, search, formatting, and summarization
- FastAPI server with streaming support
- Designed for deployment on AEGRA framework

## Quick Start

```bash
# Install dependencies
uv pip install -e .

# Copy environment file
cp .env.example .env

# Start the server
uv run uvicorn src.main:app --host 0.0.0.0 --port 8000
```

## API Endpoints

- `GET /health` - Health check
- `POST /search` - Perform web search
- `POST /stream/search` - Streaming search results

## Configuration

Set environment variables in `.env` (copy from `.env.example`):

```
# LLM Configuration (REQUIRED)
LLM_BASE_URL=http://192.168.0.100:4444/v1
LLM_MODEL=my-vllm/mymodel
LLM_API_KEY=your_actual_api_key_here  # Get from your LLM provider

# SearXNG Configuration
SEARXNG_URL=http://192.168.0.100:8089

# AEGRA Configuration
AEGRA_URL=http://192.168.0.100:2026
AEGRA_API_KEY=your_aegra_api_key_here  # Optional

# Application Configuration
APP_HOST=0.0.0.0
APP_PORT=8000
DEBUG=false

# Optional: PostgreSQL for checkpoint persistence (via AEGRA)
POSTGRES_URL=postgresql://user:password@localhost:5432/langgraph
```

## Docker

```bash
docker-compose up
```
