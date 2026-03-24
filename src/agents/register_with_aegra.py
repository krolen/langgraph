"""Register a LangGraph agent with AEGRA orchestration platform.

AEGRA uses an "assistants" model where:
- An **assistant** represents a registered agent/graph definition
- A **thread** represents a conversation/session with that agent
- A **run** executes the agent on a thread

Registration workflow:
1. Create an assistant (registers the agent definition)
2. Create threads for conversations
3. Create runs to execute the agent
"""

import httpx
from typing import Optional

from src.agents.config import config


class AegraRegistrationError(Exception):
    """Raised when agent registration fails."""
    pass


class AegraRegistrar:
    """Register LangGraph agents with AEGRA orchestration platform.

    AEGRA uses an assistants model:
    - Assistants are registered agent definitions
    - Threads are conversation sessions
    - Runs execute agents on threads

    Attributes:
        aegra_url: Base URL of the AEGRA instance.
        assistant_name: Name of the assistant/agent.
        assistant_version: Version of the agent.
        assistant_description: Description of what the agent does.
        endpoint_url: URL where the agent API is exposed.
        assistant_id: ID returned after registration.
    """

    def __init__(
        self,
        aegra_url: str,
        graph_id: str,
        assistant_name: Optional[str] = None,
        assistant_version: str = "0.1.0",
        assistant_description: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """Initialize the AEGRA registrar.

        Args:
            aegra_url: Base URL of AEGRA (e.g., "http://192.168.0.100:2026")
            graph_id: LangGraph graph ID (required by AEGRA)
            assistant_name: Name for the assistant (defaults to graph_id)
            assistant_version: Agent version string
            assistant_description: Human-readable description
            endpoint_url: URL where the agent is accessible
            api_key: Optional API key for authentication
        """
        self.aegra_url = aegra_url.rstrip("/")
        self.graph_id = graph_id
        self.assistant_name = assistant_name or graph_id
        self.assistant_version = assistant_version
        self.assistant_description = assistant_description or f"Agent for graph '{graph_id}'"
        self.endpoint_url = endpoint_url
        self.api_key = api_key
        self.assistant_id: Optional[str] = None

    def _headers(self) -> dict:
        """Build request headers."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def register(self) -> dict:
        """Register the agent as an assistant with AEGRA.

        Creates a new assistant (agent definition) in AEGRA.

        Returns:
            Registration response containing assistant_id.

        Raises:
            AegraRegistrationError: If registration fails.
        """
        # Build assistant creation payload
        payload = {
            "graph_id": self.graph_id,  # Required field
            "name": self.assistant_name,
            "description": self.assistant_description,
            "config": {},
            "context": {},
            "metadata": {
                "version": self.assistant_version,
                "framework": "langgraph",
            },
        }

        # Add endpoint URL if provided
        if self.endpoint_url:
            payload["metadata"]["endpoint"] = self.endpoint_url

        url = f"{self.aegra_url}/assistants"

        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(url, json=payload, headers=self._headers())

                # Handle 409 Conflict (assistant already exists)
                if response.status_code == 409:
                    error_data = response.json()
                    # Extract assistant_id from error or search for existing
                    self.assistant_id = error_data.get("assistant_id")
                    if self.assistant_id:
                        return {"status": "already_exists", **error_data}
                    # Search for existing assistant by graph_id
                    existing = self.list_assistants()
                    for assistant in existing.get("assistants", []):
                        if assistant.get("graph_id") == self.graph_id:
                            self.assistant_id = assistant["assistant_id"]
                            return {
                                "status": "already_exists",
                                "assistant_id": self.assistant_id,
                                "name": assistant.get("name"),
                            }
                    raise AegraRegistrationError("Assistant exists but could not find assistant_id")

                response.raise_for_status()
                result = response.json()
                # Store the assistant ID for future operations
                self.assistant_id = result.get("assistant_id") or result.get("id")
                return result

        except httpx.HTTPError as e:
            raise AegraRegistrationError(f"Failed to register assistant: {e}")

    def unregister(self) -> dict:
        """Unregister the assistant from AEGRA.

        Returns:
            Unregistration response from AEGRA.

        Raises:
            AegraRegistrationError: If unregistration fails.
        """
        if not self.assistant_id:
            raise AegraRegistrationError("Assistant not registered (no assistant_id)")

        url = f"{self.aegra_url}/assistants/{self.assistant_id}"

        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.delete(url, headers=self._headers())
                response.raise_for_status()
                return response.json()

        except httpx.HTTPError as e:
            raise AegraRegistrationError(f"Failed to unregister assistant: {e}")

    def get_assistant(self) -> dict:
        """Get the assistant details from AEGRA.

        Returns:
            Assistant information from AEGRA.

        Raises:
            AegraRegistrationError: If request fails.
        """
        if not self.assistant_id:
            raise AegraRegistrationError("Assistant not registered (no assistant_id)")

        url = f"{self.aegra_url}/assistants/{self.assistant_id}"

        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.get(url, headers=self._headers())
                response.raise_for_status()
                return response.json()

        except httpx.HTTPError as e:
            raise AegraRegistrationError(f"Failed to get assistant: {e}")

    def create_thread(self, initial_state: Optional[dict] = None) -> dict:
        """Create a new thread (conversation session) for the assistant.

        Args:
            initial_state: Optional initial state for the thread.

        Returns:
            Thread creation response with thread_id.

        Raises:
            AegraRegistrationError: If thread creation fails.
        """
        payload = {"assistant_id": self.assistant_id}
        if initial_state:
            payload["state"] = initial_state

        url = f"{self.aegra_url}/threads"

        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(url, json=payload, headers=self._headers())
                response.raise_for_status()
                return response.json()

        except httpx.HTTPError as e:
            raise AegraRegistrationError(f"Failed to create thread: {e}")

    def create_run(self, thread_id: str, input_data: dict) -> dict:
        """Create a run to execute the agent on a thread.

        Args:
            thread_id: The thread to run the agent on.
            input_data: Input data for the agent execution.

        Returns:
            Run creation response with run_id.

        Raises:
            AegraRegistrationError: If run creation fails.
        """
        payload = {
            "assistant_id": self.assistant_id,
            "input": input_data,
        }

        url = f"{self.aegra_url}/threads/{thread_id}/runs"

        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(url, json=payload, headers=self._headers())
                response.raise_for_status()
                return response.json()

        except httpx.HTTPError as e:
            raise AegraRegistrationError(f"Failed to create run: {e}")

    def list_assistants(self) -> dict:
        """List all assistants registered in AEGRA.

        Returns:
            Dict with 'assistants' key containing list of assistants.

        Raises:
            AegraRegistrationError: If request fails.
        """
        url = f"{self.aegra_url}/assistants"

        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.get(url, headers=self._headers())
                response.raise_for_status()
                return response.json()

        except httpx.HTTPError as e:
            raise AegraRegistrationError(f"Failed to list assistants: {e}")


def register_hello_agent_with_aegra(
    aegra_url: str = config.aegra_url,
    graph_id: str = "hello-world-agent",
    agent_endpoint: str = "http://192.168.0.188:8000",
    api_key: Optional[str] = None,
) -> dict:
    """Register the hello world agent with AEGRA.

    Args:
        aegra_url: AEGRA platform URL.
        graph_id: LangGraph graph ID for the agent.
        agent_endpoint: URL where the agent API is exposed.
        api_key: Optional API key for AEGRA authentication.

    Returns:
        Registration response with assistant_id.
    """
    registrar = AegraRegistrar(
        aegra_url=aegra_url,
        graph_id=graph_id,
        assistant_name="hello-world-agent",
        assistant_version="0.1.0",
        assistant_description="Simple hello world LangGraph agent for demonstration",
        endpoint_url=agent_endpoint,
        api_key=api_key,
    )

    return registrar.register()


# Example usage
if __name__ == "__main__":
    import os

    api_key = os.getenv("AEGRA_API_KEY")

    registrar = AegraRegistrar(
        aegra_url=config.aegra_url,
        graph_id="hello-world-agent",
        assistant_name="hello-world-agent",
        assistant_version="0.1.0",
        assistant_description="Simple hello world LangGraph agent",
        endpoint_url="http://192.168.0.188:8001",
        api_key=api_key,
    )

    try:
        # List existing assistants
        print("=== Existing Assistants ===")
        assistants = registrar.list_assistants()
        for a in assistants.get("assistants", []):
            print(f"  - {a['name']} ({a['assistant_id']})")

        # Register the assistant
        print("\n=== Registering ===")
        result = registrar.register()
        print(f"Registered assistant: {result}")
        print(f"Assistant ID: {registrar.assistant_id}")

    except AegraRegistrationError as e:
        print(f"Registration failed: {e}")