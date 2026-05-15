"""LLM Wrapper for creating ChatOpenAI instances with custom headers."""

from langchain_openai import ChatOpenAI
import asyncio
from typing import Any, Optional
from collections import defaultdict

from src.agents.config import config


class ModelConcurrencyManager:
    """Manages semaphores for different models to limit parallel requests."""
    def __init__(self):
        self._semaphores = {}

    def get_semaphore(self, model: str) -> asyncio.Semaphore:
        if model not in self._semaphores:
            # Default concurrency limit is 5, but 'krolen/mymodel' is limited to 2
            limit = 2 if "mymodel" in model else 5
            self._semaphores[model] = asyncio.Semaphore(limit)
        return self._semaphores[model]

concurrency_manager = ModelConcurrencyManager()

class ConcurrencyLimitedLLM:
    """Wrapper for LLM to limit the number of concurrent requests."""
    def __init__(self, llm: Any, semaphore: asyncio.Semaphore):
        self.llm = llm
        self.semaphore = semaphore

    async def ainvoke(self, *args, **kwargs):
        async with self.semaphore:
            return await self.llm.ainvoke(*args, **kwargs)

    def invoke(self, *args, **kwargs):
        # Note: sync invoke does not use the async semaphore.
        # If sync concurrency limit is needed, a threading.Semaphore would be required.
        return self.llm.invoke(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.llm, name)


def create_chat_openai(
        base_url: str = None,
        model: str = None,
        api_key: str = None,
        default_headers: dict = None,
        **kwargs
) -> Any:
    """
    Create a ChatOpenAI instance with optional custom headers and concurrency limiting.
    """
    model_name = model or config.llm_model
    llm = ChatOpenAI(
        base_url=base_url or config.llm_base_url,
        model=model_name,
        api_key=api_key or config.llm_api_key,
        default_headers=default_headers or {},
        **kwargs
    )
    
    # Wrap with concurrency limiter
    semaphore = concurrency_manager.get_semaphore(model_name)
    return ConcurrencyLimitedLLM(llm, semaphore)


# Router LLM instance - for routing decisions, sent to specific backend via bifrost
router_llm = create_chat_openai(
    default_headers={"X-My-Route": "router"}
)

# Local LLM instance - for local processing, may go to different backend
router_llm_local = create_chat_openai(
    default_headers={"X-My-Route": "local-fallback"}
)

# Local LLM instance - for local processing, may go to different backend
llm_local = create_chat_openai(
    base_url="http://192.168.0.188:8888/v1",
    model="krolen/mymodel",
    api_key="my-key"
)
