"""LLM Wrapper for creating ChatOpenAI instances with custom headers."""
import asyncio
from typing import Any
from pydantic import SecretStr

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from .config import config

class ModelConcurrencyManager:
    """Manages semaphores for different models to limit parallel requests."""

    def __init__(self) -> None:
        self._semaphores: dict[tuple[asyncio.AbstractEventLoop, str], asyncio.Semaphore] = {}

    def get_semaphore(self, model: str) -> asyncio.Semaphore:
        key = (asyncio.get_running_loop(), model)
        if key not in self._semaphores:
            # Default concurrency limit is 5, but 'krolen/mymodel' is limited to 2
            limit = 2 if "mymodel" in model else 5
            self._semaphores[key] = asyncio.Semaphore(limit)
        return self._semaphores[key]


concurrency_manager = ModelConcurrencyManager()


class ModelRepository:
    """Repository to cache and reuse LLM instances to avoid redundant creation."""

    def __init__(self) -> None:
        self._models: dict[tuple, Any] = {}

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
            model=profile["model"],
            base_url=profile.get("base_url"),
            api_key=profile.get("api_key"),
            default_headers=profile.get("default_headers")
        )


model_repo = ModelRepository()


class ConcurrencyLimitedLLM:
    """Wrapper for LLM to limit the number of concurrent requests."""

    def __init__(self, llm: Any, semaphore: asyncio.Semaphore | str):
        self.llm = llm
        self.semaphore = semaphore

    async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
        semaphore = concurrency_manager.get_semaphore(self.semaphore) if isinstance(self.semaphore, str) else self.semaphore
        async with semaphore:
            return await self.llm.ainvoke(*args, **kwargs)

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        # Note: sync invoke does not use the async semaphore.
        # If sync concurrency limit is needed, a threading.Semaphore would be required.
        return self.llm.invoke(*args, **kwargs)

    def with_structured_output(self, *args: Any, **kwargs: Any) -> "ConcurrencyLimitedLLM":
        return ConcurrencyLimitedLLM(
            self.llm.with_structured_output(*args, **kwargs), self.semaphore
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self.llm, name)


def create_chat_openai(
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        default_headers: dict | None = None,
        **kwargs: Any
) -> Any:
    """
    Create a ChatOpenAI instance with optional custom headers and concurrency limiting.
    """
    model_name = model or config.llm_model
    llm = ChatOpenAI(
        base_url=base_url or config.llm_base_url,
        model=model_name,
        api_key=SecretStr(api_key or config.llm_api_key),
        default_headers=default_headers or {},
        **kwargs
    )

    # Wrap with concurrency limiter
    semaphore = model_name
    return ConcurrencyLimitedLLM(llm, semaphore)


