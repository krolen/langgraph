# Deployment Guide for LangGraph Web Search Agent

## Overview

This is a LangGraph-based web search agent that provides privacy-focused web search capabilities using SearXNG. It runs as a FastAPI server with streaming support.

## Prerequisites

1. **Python 3.12+** - The application requires Python 3.12 or higher
2. **SearXNG Instance** - A running SearXNG server for privacy-focused searches
3. **uv package manager** (recommended) or pip for dependency installation

## Deployment Options

### Option 1: Local Development (Recommended for Testing)

#### Step 1: Install Dependencies

```bash
# Using uv (recommended)
uv pip install -e .

# Or using pip
pip install -e .
```

#### Step 2: Configure Environment

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your configuration
nano .env  # or use your preferred editor
```

**Required Environment Variables:**
- `SEARXNG_URL` - URL of your SearXNG instance (e.g., `http://192.168.0.100:8089`)
- `APP_HOST` - Host to bind the server (default: `0.0.0.0`)
- `APP_PORT` - Port to run the server (default: `8000`)

**Optional Environment Variables:**
- `AEGRA_URL` - URL for AEGRA framework integration
- `DEBUG` - Enable debug mode (`true`/`false`)
- `POSTGRES_URL` - PostgreSQL connection for checkpoint persistence

#### Step 3: Start the Server

```bash
# Using uv
uv run uvicorn src.main:app --host 0.0.0.0 --port 8000

# Or using python directly
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

#### Step 4: Verify the Server

```bash
# Health check
curl http://localhost:8000/health

# Test search
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "test search"}'
```

---

### Option 2: Docker Deployment

#### Step 1: Update docker-compose.yml

Edit the environment variables in [`docker-compose.yml`](docker-compose.yml:13) to match your SearXNG URL:

```yaml
environment:
  - SEARXNG_URL=http://your-searxng-url:8089
  - APP_HOST=0.0.0.0
  - APP_PORT=8000
  - DEBUG=false
```

#### Step 2: Build and Run

```bash
# Build and start containers
docker-compose up --build

# Or run in detached mode
docker-compose up -d --build
```

#### Step 3: View Logs

```bash
# View container logs
docker-compose logs -f web-search-agent
```

#### Step 4: Stop the Container

```bash
docker-compose down
```

---

### Option 3: Production Deployment with Docker

For production, consider these enhancements:

1. **Use a reverse proxy** (nginx, traefik) for SSL termination
2. **Set up proper logging** and monitoring
3. **Configure resource limits** in docker-compose.yml
4. **Use environment-specific configurations**

Example production docker-compose with resource limits:

```yaml
services:
  web-search-agent:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: web-search-agent-prod
    ports:
      - "8000:8000"
    environment:
      - SEARXNG_URL=http://searxng:8080
      - APP_HOST=0.0.0.0
      - APP_PORT=8000
      - DEBUG=false
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '1'
          memory: 512M
        reservations:
          cpus: '0.5'
          memory: 256M
    networks:
      - search-network

networks:
  search-network:
    driver: bridge
```

---

## Setting Up SearXNG

If you don't have a SearXNG instance, you can run one locally:

### Option A: Docker (Recommended)

```bash
docker run -d \
  --name searxng \
  -p 8080:8080 \
  -v $(pwd)/searxng_settings.yml:/etc/searxng/settings.yml:ro \
  --restart=always \
  docker.io/searxng/searxng
```

### Option B: Using the included settings

The project includes [`searxng_settings.yml`](searxng_settings.yml:1) which can be mounted to a SearXNG container.

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check endpoint |
| `/search` | POST | Perform web search (non-streaming) |
| `/stream/search` | POST | Streaming search results |

### Example API Usage

**Search Request:**
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "latest AI developments",
    "options": {
      "categories": ["general", "news"],
      "limit": 10,
      "summarize": true
    }
  }'
```

**Streaming Search:**
```bash
curl -X POST http://localhost:8000/stream/search \
  -H "Content-Type: application/json" \
  -d '{"query": "latest AI developments"}'
```

---

## Troubleshooting

### Common Issues

1. **Connection refused to SearXNG**
   - Verify SearXNG is running: `curl http://your-searxng-url:8089`
   - Check `SEARXNG_URL` in your `.env` file

2. **Port already in use**
   - Change `APP_PORT` in `.env` or docker-compose.yml

3. **Dependency installation fails**
   - Ensure Python 3.12+ is installed: `python --version`
   - Try upgrading pip: `pip install --upgrade pip`

4. **Docker build fails**
   - Ensure Docker and docker-compose are installed
   - Try: `docker-compose build --no-cache`

### Debug Mode

Enable debug mode by setting `DEBUG=true` in your environment variables to get more detailed logging.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Server                           │
│  (src/main.py - Entry point with /search, /stream/search)  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  LangGraph Agent                            │
│  (src/agent/web_search_agent.py - Agent workflow)           │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐            │
│  │   Query    │→ │  Search    │→ │  Format    │→           │
│  │ Processing │  │  Results   │  │  Results   │            │
│  └────────────┘  └────────────┘  └────────────┘            │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   Search Tool                               │
│  (src/tools/search.py - SearXNG integration)                │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    SearXNG Server                           │
│  (External privacy-focused search engine)                   │
└─────────────────────────────────────────────────────────────┘
```

---

## Next Steps

1. **Set up SearXNG** if you haven't already
2. **Choose deployment method** (local or Docker)
3. **Configure environment variables**
4. **Start the server**
5. **Test the API endpoints**
6. **Integrate with your application** or use directly via API
