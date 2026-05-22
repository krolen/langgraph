"""AEGRA deployment configuration for the web search agent."""

import os

from dotenv import load_dotenv

load_dotenv()

# LLM Configuration
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://192.168.0.100:4444/v1")
LLM_BASE_URL_LOCAL = os.getenv("LLM_BASE_URL_LOCAL", "http://192.168.0.188:8888/v1")

# local quick model
# LLM_MODEL = os.getenv("LLM_MODEL", "krolen/mymodel")
# routing model going through bitfrost
# LLM_API_KEY = os.getenv("LLM_API_KEY", "my-key")

# SearXNG Configuration
SEARXNG_URL = os.getenv("SEARXNG_URL", "http://192.168.0.100:8089")

# MCP Server Configuration
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://192.168.0.100:7000/mcp")

# AEGRA Configuration
AEGRA_URL = os.getenv("AEGRA_URL", "http://192.168.0.100:2026")

# AEGRA API Key (optional)
AEGRA_API_KEY = os.getenv("AEGRA_API_KEY", "my-key")


MODEL_PROFILES = {
    "openrouter": {
        "model": "openrouter",
        "base_url": LLM_BASE_URL,
        "default_headers": {"X-My-Route": "openrouter"},
    },
    "fabric": {
        "model": "fabric",
        "base_url": LLM_BASE_URL,
        "default_headers": {"X-My-Route": "fabric"},
    },
    "local": {
        "model": "krolen/mymodel",
        "base_url": LLM_BASE_URL_LOCAL,
    },
}

class AgentConfig:
    """Configuration container for the web search agent."""

    def __init__(self):
        """Initialize configuration from environment."""
        self.llm_base_url = LLM_BASE_URL
        self.searxng_url = SEARXNG_URL
        self.mcp_server_url = MCP_SERVER_URL
        self.aegra_url = AEGRA_URL
        self.aegra_api_key = AEGRA_API_KEY

# Global configuration instance
config = AgentConfig()
