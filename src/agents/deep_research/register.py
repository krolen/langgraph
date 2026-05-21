import os

from src.agents.register_with_aegra import AegraRegistrar

from src.agents.config import config


def register_deep_research_agent():
    """
    Register the Deep Research Agent with the AEGRA platform.
    """
    api_key = os.getenv("AEGRA_API_KEY", "my-key")

    registrar = AegraRegistrar(
        aegra_url=config.aegra_url,
        graph_id="deep-research-agent",
        assistant_name="Deep Research Agent",
        assistant_version="0.1.0",
        assistant_description="An advanced research agent that iteratively searches and crawls the web to provide comprehensive reports.",
        api_key=api_key,
    )

    try:
        print("=== Registering Deep Research Agent with AEGRA ===")
        result = registrar.register()
        print(f"Registration successful: {result}")
        print(f"Assistant ID: {registrar.assistant_id}")
    except Exception as e:
        print(f"Registration failed: {e}")


if __name__ == "__main__":
    register_deep_research_agent()
