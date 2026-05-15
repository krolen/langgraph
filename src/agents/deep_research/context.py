"""Configurable parameters for the Deep Research Agent."""
import os
from dataclasses import dataclass, field
from typing import Annotated


@dataclass(kw_only=True)
class Context:
    """The context for the research agent."""
    
    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="openai/gpt-4o",
        metadata={"description": "The LLM model to use for planning and synthesis."}
    )
    
    max_iterations: int = field(
        default=3,
        metadata={"description": "Maximum number of research iterations before finishing."}
    )
    
    system_prompt_planner: str = field(
        default="You are a precise research planner. Output only valid JSON lists of strings.",
        metadata={"description": "System prompt for the planner node."}
    )
    
    system_prompt_synthesizer: str = field(
        default="You are a professional research synthesizer.",
        metadata={"description": "System prompt for the synthesizer node."}
    )

    def __post_init__(self) -> None:
        # Allow environment variables to override defaults
        self.model = os.environ.get("RESEARCH_MODEL", self.model)
        self.max_iterations = int(os.environ.get("RESEARCH_MAX_ITERATIONS", str(self.max_iterations)))
