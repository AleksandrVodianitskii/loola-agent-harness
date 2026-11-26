"""Run the same task with an append-only baseline and the improved agent."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import TextIO

from harness.agent import Agent, AgentConfig
from harness.provider import ChatClient

from .baseline import run_baseline
from .fixture import FixtureProvider
from .task import BATCH_COUNT, PROMPT, prepare, verify
from .trace import open_trace, write_event


def _run_case(
    name: str, config: AgentConfig, *, enhanced: bool, fixture_enabled: bool,
    trace: TextIO | None = None,
) -> dict:
    fixture = FixtureProvider(config.context_window) if fixture_enabled else None
    observer = partial(write_event, trace, name) if trace is not None else None
    if observer:
        observer("case_started", {"context_window": config.context_window, "prompt": PROMPT})
    with tempfile.TemporaryDirectory(prefix=f"loola-{name}-") as temporary:
        workspace = Path(temporary)
        raw_hashes = prepare(workspace)
        if enhanced:
            client = ChatClient(
                config.base_url, config.model, config.api_key,
                max_output_tokens=config.max_output_tokens,
                output_token_field=config.output_token_field, opener=fixture,
            )
            result = Agent(config, workspace=workspace, client=client, on_event=observer).run(PROMPT)
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
                opener=fixture, on_event=observer,
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
    if observer:
        observer("case_finished", {key: value for key, value in report.items()
                                   if key not in ("compactions", "fixture")})
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="use a configured chat/completions provider")
    parser.add_argument("--context-window", type=int, help="actual live context limit, or simulated limit")
    parser.add_argument("--max-output-tokens", type=int, default=640)
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument("--output", type=Path, help="also write the JSON results to this file")
    parser.add_argument("--trace", type=Path, help="write a turn-by-turn transcript to a new JSONL file")
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
        trace_context = open_trace(args.trace, args.output)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    with trace_context as trace:
        result, expected = _run_benchmark(config, live=args.live, trace=trace)
    report = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(report, encoding="utf-8")
    print(report, end="")
    return 0 if expected else 1


def _run_benchmark(config: AgentConfig, *, live: bool, trace: TextIO | None) -> tuple[dict, bool]:
    mode = "live" if live else "deterministic_fixture"
    if trace is not None:
        write_event(trace, "benchmark", "run_started", {
            "mode": mode, "model": config.model, "max_output_tokens": config.max_output_tokens,
        })
    cases = {
        "baseline": _run_case("baseline", config, enhanced=False, fixture_enabled=not live, trace=trace),
        "enhanced": _run_case("enhanced", config, enhanced=True, fixture_enabled=not live, trace=trace),
    }
    result = {
        "mode": mode,
        "token_measurement": (
            "Provider usage on successful responses, with local estimates and coverage counters."
            if live else
            "Estimated from UTF-8 JSON size divided by four. Cache reuse is the common prefix "
            "with the previous accepted request in the fixture's serialization."
        ),
        "model": config.model,
        "task": "inventory_migration",
        "batches": BATCH_COUNT,
        "max_output_tokens": config.max_output_tokens,
        "cases": cases,
    }
    expected = cases["enhanced"]["successful"]
    if not live:
        wide = replace(config, context_window=max(65_536, config.context_window))
        cases["baseline_wide"] = _run_case(
            "baseline_wide", wide, enhanced=False, fixture_enabled=True, trace=trace,
        )
        expected = expected and cases["baseline_wide"]["successful"]
        if expected:
            before = cases["baseline_wide"]["metrics"]["prompt_tokens"]
            after = cases["enhanced"]["metrics"]["prompt_tokens"]
            result["equal_work_prompt_reduction_pct"] = round(100 * (before - after) / before, 2)
        if config.context_window == 4096 and config.max_output_tokens == 640:
            baseline, enhanced = cases["baseline"], cases["enhanced"]
            expected = (
                expected and baseline["status"] == "context_limit"
                and enhanced["metrics"]["compactions"] > 0
                and enhanced["fixture"]["checkpoint_source_at_completion"] == "compacted_history"
                and enhanced["fixture"]["fixed_head_variants"] == 1
                and enhanced["fixture"]["tool_schema_variants"] == 1
            )
    if trace is not None:
        write_event(trace, "benchmark", "run_finished", {"passed": expected})
    return result, expected


if __name__ == "__main__":
    raise SystemExit(main())
