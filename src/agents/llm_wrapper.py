"""LLM Wrapper for creating ChatOpenAI instances with custom headers."""
import asyncio
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from .config import config

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


class ModelRepository:
    """Repository to cache and reuse LLM instances to avoid redundant creation."""

    def __init__(self):
        self._models = {}

    def get_model(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        default_headers: dict | None = None,
    ) -> Any:
        """
        Retrieve a cached model instance or create a new one if it doesn't exist.
        """
        # Use a tuple of the key parameters as the cache key
        # Convert default_headers to a frozenset of items to make it hashable
        headers_key = frozenset(default_headers.items()) if default_headers else None
        key = (model, base_url, api_key, headers_key)

        if key not in self._models:
            self._models[key] = create_chat_openai(
                model=model,
                base_url=base_url,
                api_key=api_key,
                default_headers=default_headers
            )
        return self._models[key]

    def get_model_by_profile(self, profile_name: str) -> Any:
        """
        Retrieve a model instance based on a predefined profile name.
        """
        from src.agents.config import MODEL_PROFILES
        profile = MODEL_PROFILES.get(profile_name)
        if not profile:
            raise ValueError(f"Model profile '{profile_name}' not found in config.")
        
        return self.get_model(
            model=profile.get("model"),
            base_url=profile.get("base_url"),
            api_key=profile.get("api_key"),
            default_headers=profile.get("default_headers")
        )


model_repo = ModelRepository()


class ConcurrencyLimitedLLM:
    """Wrapper for LLM to limit the number of concurrent requests."""

    def __init__(self, llm: BaseChatModel, semaphore: asyncio.Semaphore):
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
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        default_headers: dict | None = None,
        **kwargs
) -> Any:
    """
    Create a ChatOpenAI instance with optional custom headers and concurrency limiting.
    """
    model_name = model or config.llm_model
    llm = ChatOpenAI(
        base_url=base_url or config.llm_base_url,
        model=model_name,
        api_key=api_key or "my-key",
        default_headers=default_headers or {},
        **kwargs
    )

    # Wrap with concurrency limiter
    semaphore = concurrency_manager.get_semaphore(model_name)
    return ConcurrencyLimitedLLM(llm, semaphore)


