from typing import Any


def extract_text_from_mcp_result(result: Any) -> str:
    """Helper to extract and join text from MCP content blocks."""
    if isinstance(result, list):
        texts = []
        for block in result:
            if isinstance(block, dict) and block.get("type") == "text":
                texts.append(block.get("text", ""))
            elif hasattr(block, "text"):
                texts.append(getattr(block, "text", ""))
            else:
                texts.append(str(block))
        return "\n".join(texts)
    return str(result)
