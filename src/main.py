"""Local command-line entry point: python -m src.main 'research question'."""
import argparse
import asyncio
import json
from typing import Any, cast

from src.agents.deep_research.context import Context
from src.agents.deep_research.graph import local_graph


async def research(query: str, ctx: Context) -> dict:
    agent = local_graph(ctx)
    return cast(dict[str, Any], await agent.ainvoke(
        {"query": query}, context=ctx,
        config={"recursion_limit": ctx.max_iterations * 5 + 5},
    ))


def main() -> None:
    parser = argparse.ArgumentParser(description="Research a question using configured LLM and MCP services.")
    parser.add_argument("query")
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--profile", default="local")
    parser.add_argument("--json", action="store_true", help="Include evidence and errors in JSON output")
    args = parser.parse_args()
    try:
        ctx = Context(max_iterations=args.max_iterations, default_profile=args.profile, extractor_profile=None)
        result = asyncio.run(research(args.query, ctx))
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Research failed: {exc}\n")
    print(json.dumps(result, indent=2, ensure_ascii=False) if args.json else result.get("final_report", "No report produced."))


if __name__ == "__main__":
    main()
