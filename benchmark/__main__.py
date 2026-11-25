"""Run the same task with an append-only baseline and the improved agent."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from harness.agent import Agent, AgentConfig
from harness.provider import ChatClient

from .baseline import run_baseline
from .fixture import FixtureProvider
from .task import BATCH_COUNT, PROMPT, prepare, verify


def _run_case(
    name: str, config: AgentConfig, *, enhanced: bool, fixture_enabled: bool,
) -> dict:
    fixture = FixtureProvider(config.context_window) if fixture_enabled else None
    with tempfile.TemporaryDirectory(prefix=f"loola-{name}-") as temporary:
        workspace = Path(temporary)
        raw_hashes = prepare(workspace)
        if enhanced:
            client = ChatClient(
                config.base_url, config.model, config.api_key,
                max_output_tokens=config.max_output_tokens,
                output_token_field=config.output_token_field, opener=fixture,
            )
            result = Agent(config, workspace=workspace, client=client).run(PROMPT)
            report = {
                "status": result.status,
                "metrics": result.metrics.to_dict(),
                "compactions": result.compactions,
            }
            if result.error is not None:
                report["error"] = result.error
        else:
            report = run_baseline(
                PROMPT, workspace, base_url=config.base_url, model=config.model,
                api_key=config.api_key, max_output_tokens=config.max_output_tokens,
                max_turns=config.max_turns, output_token_field=config.output_token_field,
                opener=fixture,
            )
        acceptance = verify(workspace, raw_hashes)
    report.update(
        context_window=config.context_window,
        successful=report["status"] == "completed" and acceptance["passed"],
        acceptance=acceptance,
    )
    metrics = report["metrics"]
    if metrics["usage_responses"] == metrics["responses"] and metrics["responses"]:
        report["total_tokens"] = metrics["prompt_tokens"] + metrics["completion_tokens"]
    if (
        metrics["prompt_usage_responses"] == metrics["cache_usage_responses"] == metrics["responses"]
        and metrics["responses"]
    ):
        report["uncached_prompt_tokens"] = metrics["prompt_tokens"] - metrics["cached_prompt_tokens"]
    if fixture is not None:
        report["fixture"] = fixture.telemetry()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="use a configured chat/completions provider")
    parser.add_argument("--context-window", type=int, help="actual live context limit, or simulated limit")
    parser.add_argument("--max-output-tokens", type=int, default=640)
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument("--output", type=Path, help="also write the JSON results to this file")
    args = parser.parse_args()

    if args.live:
        base_url = os.environ.get("LLM_BASE_URL", "")
        model = os.environ.get("LLM_MODEL", "")
        if not base_url or not model or args.context_window is None:
            parser.error("--live requires LLM_BASE_URL, LLM_MODEL and --context-window")
    else:
        base_url, model = "https://fixture.invalid/v1", "deterministic-inventory-fixture"

    context_window = 4096 if args.context_window is None else args.context_window
    shared = {
        "base_url": base_url,
        "model": model,
        "api_key": os.environ.get("LLM_API_KEY", "") if args.live else "",
        "max_output_tokens": args.max_output_tokens,
        "max_turns": args.max_turns,
        "output_token_field": (
            os.environ.get("LLM_OUTPUT_TOKEN_FIELD", "max_tokens") if args.live else "max_tokens"
        ),
    }
    try:
        config = AgentConfig(context_window=context_window, **shared)
    except ValueError as exc:
        parser.error(str(exc))
    cases = {
        "baseline": _run_case("baseline", config, enhanced=False, fixture_enabled=not args.live),
        "enhanced": _run_case("enhanced", config, enhanced=True, fixture_enabled=not args.live),
    }
    result = {
        "mode": "live" if args.live else "deterministic_fixture",
        "token_measurement": (
            "Provider usage on successful responses, with local estimates and coverage counters."
            if args.live else
            "Estimated from UTF-8 JSON size divided by four. Cache reuse is the common prefix "
            "with the previous accepted request in the fixture's serialization."
        ),
        "model": model,
        "task": "inventory_migration",
        "batches": BATCH_COUNT,
        "max_output_tokens": args.max_output_tokens,
        "cases": cases,
    }
    expected = cases["enhanced"]["successful"]
    if not args.live:
        wide = AgentConfig(context_window=max(65_536, context_window), **shared)
        cases["baseline_wide"] = _run_case(
            "baseline-wide", wide, enhanced=False, fixture_enabled=True,
        )
        expected = expected and cases["baseline_wide"]["successful"]
        if expected:
            before = cases["baseline_wide"]["metrics"]["prompt_tokens"]
            after = cases["enhanced"]["metrics"]["prompt_tokens"]
            result["equal_work_prompt_reduction_pct"] = round(100 * (before - after) / before, 2)
        if context_window == 4096 and args.max_output_tokens == 640:
            baseline, enhanced = cases["baseline"], cases["enhanced"]
            expected = (
                expected and baseline["status"] == "context_limit"
                and enhanced["metrics"]["compactions"] > 0
                and enhanced["fixture"]["checkpoint_source_at_completion"] == "compacted_history"
                and enhanced["fixture"]["fixed_head_variants"] == 1
                and enhanced["fixture"]["tool_schema_variants"] == 1
            )
    report = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(report, encoding="utf-8")
    print(report, end="")
    return 0 if expected else 1


if __name__ == "__main__":
    raise SystemExit(main())
