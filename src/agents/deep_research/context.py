"""Configurable parameters for the Deep Research Agent."""
from dataclasses import dataclass


@dataclass(kw_only=True)
class Context:
    """The context for the research agent."""

    model: str = "krolen/mymodel"
    max_iterations: int = 3
    system_prompt_planner: str = "You are a precise research planner. Output only valid JSON lists of strings."
    system_prompt_synthesizer: str = "You are a professional research synthesizer."
    enable_mcp: bool = True
    mcp_server_url: str = "http://192.168.0.100:7000/mcp"
