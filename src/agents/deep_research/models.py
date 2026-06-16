from typing import Optional, List
from pydantic import BaseModel

class SearchResult(BaseModel):
    title: str = ""
    url: str = ""
    content: str = ""
    score: float = 0.01
    engine: str = "unknown"

class WebSearchResponse(BaseModel):
    query: str = ""
    result_count: int = 0
    results: List[SearchResult] = []
    search_time: Optional[float] = None
    engines_used: List[str] = []
    fallback: Optional[str] = None
    error: Optional[str] = None
    instruction: Optional[str] = None
