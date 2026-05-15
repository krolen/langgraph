"""Configurable parameters for the Deep Research Agent."""
import os
from typing import Annotated
from pydantic import BaseModel, Field

class Context(BaseModel):
    """The context for the research agent."""
    
    model: str = Field(
        default="krolen/mymodel",
        description="The LLM model to use for planning and synthesis."
    )
    
    max_iterations: int = Field(
        default=3,
        description="Maximum number of research iterations before finishing."
    )
    
    system_prompt_planner: str = Field(
        default="You are a precise research planner. Output only valid JSON lists of strings.",
        description="System prompt for the planner node."
    )
    
    system_prompt_synthesizer: str = Field(
        default="You are a professional research synthesizer.",
        description="System prompt for the synthesizer node."
    )

    def __init__(self, **data):
        super().__init__(**data)
        # Allow environment variables to override defaults
        self.model = os.environ.get("RESEARCH_MODEL", self.model)
        self.max_iterations = int(os.environ.get("RESEARCH_MAX_ITERATIONS", str(self.max_iterations)))
