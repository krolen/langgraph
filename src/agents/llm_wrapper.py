"""LLM Wrapper for creating ChatOpenAI instances with custom headers."""

from langchain_openai import ChatOpenAI

from src.agents.config import config


def create_chat_openai(
        base_url: str = None,
        model: str = None,
        api_key: str = None,
        default_headers: dict = None,
        **kwargs
) -> ChatOpenAI:
    """
    Create a ChatOpenAI instance with optional custom headers.

    Args:
        base_url: Base URL for the API (defaults to config.llm_base_url)
        model: Model name (defaults to config.llm_model)
        api_key: API key (defaults to config.llm_api_key)
        default_headers: Dictionary of headers to send with each request
        **kwargs: Additional arguments to pass to ChatOpenAI

    Returns:
        Configured ChatOpenAI instance
    """
    return ChatOpenAI(
        base_url=base_url or config.llm_base_url,
        model=model or config.llm_model,
        api_key=api_key or config.llm_api_key,
        default_headers=default_headers or {},
        **kwargs
    )


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
