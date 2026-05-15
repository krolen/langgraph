from dataclasses import dataclass
from typing import Optional

@dataclass
class BaseAgentState:
    """Base state common to all agents."""
    query: str
    final_report: Optional[str] = None
    iteration_count: int = 0
