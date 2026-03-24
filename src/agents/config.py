"""AEGRA deployment configuration for the web search agent."""

import os
from functools import partial

from dotenv import load_dotenv

load_dotenv()

# SearXNG Configuration
SEARXNG_URL = os.getenv("SEARXNG_URL", "http://192.168.0.100:8089")

# AEGRA Configuration
AEGRA_URL = os.getenv("AEGRA_URL", "http://192.168.0.100:2026")

# AEGRA API Key (optional)
AEGRA_API_KEY = os.getenv("AEGRA_API_KEY", None)

# Application Configuration
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", "8000"))
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# PostgreSQL for checkpoint persistence
POSTGRES_URL = os.getenv("POSTGRES_URL")

# Agent metadata
AGENT_NAME = "web-search-agent"
AGENT_VERSION = "0.1.0"
AGENT_DESCRIPTION = "LangGraph-based web search agent using SearXNG for privacy-focused searches"

# Search defaults
DEFAULT_SEARCH_LIMIT = 10
DEFAULT_CATEGORIES = ["general"]
DEFAULT_SUMMARIZE = True

# Timeout configurations
SEARCH_TIMEOUT = 30.0  # seconds
REQUEST_TIMEOUT = 60.0  # seconds

# Rate limiting (requests per minute per IP)
RATE_LIMIT_PER_MINUTE = 60


class AgentConfig:
    """Configuration container for the web search agent."""

    def __init__(self):
        """Initialize configuration from environment."""
        self.searxng_url = SEARXNG_URL
        self.aegra_url = AEGRA_URL
        self.aegra_api_key = AEGRA_API_KEY
        self.app_host = APP_HOST
        self.app_port = APP_PORT
        self.debug = DEBUG
        self.postgres_url = POSTGRES_URL
        self.agent_name = AGENT_NAME
        self.agent_version = AGENT_VERSION
        self.agent_description = AGENT_DESCRIPTION
        self.default_search_limit = DEFAULT_SEARCH_LIMIT
        self.default_categories = DEFAULT_CATEGORIES
        self.default_summarize = DEFAULT_SUMMARIZE
        self.search_timeout = SEARCH_TIMEOUT
        self.request_timeout = REQUEST_TIMEOUT
        self.rate_limit_per_minute = RATE_LIMIT_PER_MINUTE

    def get_checkpoint_config(self) -> dict | None:
        """Get checkpoint configuration for LangGraph persistence.

        Returns:
            Checkpoint configuration dict if PostgreSQL is configured, None otherwise.
        """
        if not self.postgres_url:
            return None

        return {
            "type": "postgres_saver",
            "config": {
                "connection_string": self.postgres_url,
                "checkpoints_table": "langgraph_checkpoints",
            },
        }

    def get_aegra_config(self) -> dict:
        """Get AEGRA-specific configuration.

        Returns:
            AEGRA configuration dictionary.
        """
        return {
            "url": self.aegra_url,
            "agent_name": self.agent_name,
            "version": self.agent_version,
            "description": self.agent_description,
        }


# Global configuration instance
config = AgentConfig()
