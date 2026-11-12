"""Configurable parameters for the Deep Research Agent."""
from dataclasses import dataclass

from ..config import config


@dataclass(kw_only=True)
class Context:
    """The context for the research agent."""

    default_profile: str = "local"
    planner_profile: str | None = None
    extractor_profile: str | None = None
    synthesizer_profile: str | None = None
    search_profile: str | None = None
    max_iterations: int = 3
    system_prompt_planner: str = "You are a precise research planner. Output only valid JSON lists of strings."
    system_prompt_synthesizer: str = "You are a professional research synthesizer."
    enable_mcp: bool = True
    mcp_server_url: str = config.mcp_server_url

    max_page_chars: int = 20000
    max_knowledge_chars: int = 30000

    def __post_init__(self) -> None:
        if self.max_iterations < 1 or self.max_iterations > 20:
            raise ValueError("max_iterations must be between 1 and 20")
        if self.max_page_chars < 1 or self.max_knowledge_chars < 1:
            raise ValueError("Content limits must be positive")
