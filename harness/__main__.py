"""Command-line entry point."""

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

from .agent import Agent, AgentConfig


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the coding agent in a workspace.")
    parser.add_argument("prompt", help="Task for the coding agent")
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--base-url", default=os.getenv("LLM_BASE_URL"))
    parser.add_argument("--model", default=os.getenv("LLM_MODEL"))
    parser.add_argument("--context-window", type=int, default=8192)
    parser.add_argument("--max-output-tokens", type=int, default=1024)
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument("--recent-turns", type=int, default=3)
    parser.add_argument("--cache-key", default=os.getenv("LLM_CACHE_KEY"))
    parser.add_argument("--output-token-field", choices=("max_tokens", "max_completion_tokens"),
                        default=os.getenv("LLM_OUTPUT_TOKEN_FIELD", "max_tokens"))
    parser.add_argument("--allow-shell", action="store_true", help="Enable Bash in a trusted workspace")
    parser.add_argument("--json", action="store_true", help="Print result, token usage and compaction events")
    args = parser.parse_args()
    if not args.base_url or not args.model:
        parser.error("set LLM_BASE_URL and LLM_MODEL or supply their command-line flags")

    try:
        config = AgentConfig(
            base_url=args.base_url, model=args.model,
            api_key=os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY", ""),
            context_window=args.context_window, max_output_tokens=args.max_output_tokens,
            max_turns=args.max_turns, recent_turns=args.recent_turns,
            cache_key=args.cache_key, output_token_field=args.output_token_field,
            allow_shell=args.allow_shell,
        )
        result = Agent(config, workspace=args.workspace).run(args.prompt)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if args.json:
        print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    else:
        if result.content:
            print(result.content)
        if result.status != "completed":
            print(f"Agent stopped: {result.status}" + (f" ({result.error})" if result.error else ""))
    return 0 if result.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
