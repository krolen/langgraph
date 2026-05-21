"""AEGRA deployment configuration for the web search agent."""

import os

from dotenv import load_dotenv

load_dotenv()

# LLM Configuration
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://192.168.0.100:4444/v1")
# local quick model
LLM_MODEL = os.getenv("LLM_MODEL", "krolen/mymodel")
# routing model going through bitfrost
LLM_API_KEY = os.getenv("LLM_API_KEY", "my-key")

# SearXNG Configuration
SEARXNG_URL = os.getenv("SEARXNG_URL", "http://192.168.0.100:8089")

# MCP Server Configuration
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://192.168.0.100:7000/mcp")

# AEGRA Configuration
AEGRA_URL = os.getenv("AEGRA_URL", "http://192.168.0.100:2026")

# AEGRA API Key (optional)
AEGRA_API_KEY = os.getenv("AEGRA_API_KEY", "my-key")


class AgentConfig:
    """Configuration container for the web search agent."""

    def __init__(self):
        """Initialize configuration from environment."""
        self.llm_base_url = LLM_BASE_URL
        self.llm_model = LLM_MODEL
        self.llm_api_key = LLM_API_KEY
        self.searxng_url = SEARXNG_URL
        self.mcp_server_url = MCP_SERVER_URL
        self.aegra_url = AEGRA_URL
        self.aegra_api_key = AEGRA_API_KEY

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
