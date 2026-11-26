"""Demonstrate compaction with the same real model and tools in both runs."""

from __future__ import annotations

import argparse
import json
import os
import random
import secrets
import tempfile
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import TextIO

from harness.agent import Agent, AgentConfig
from harness.context import SUMMARY_PREFIX
from harness.provider import ChatClient

from .trace import open_trace, write_event


_PROFILE_PATH = "profile.txt"
_REPORT_PATH = "report.json"
_WORDS = (
    "adapt bolt category delivery emerald frost grove harbor island junction kettle laser "
    "marble nozzle orbit packet quota relay staple timber unit vessel wagon yield zinc"
).split()


def _inputs(profile_id: str, file_count: int, pressure_bytes: int) -> dict[str, str]:
    rng = random.Random(1482)
    inputs = {_PROFILE_PATH: profile_id + "\n"}
    for index in range(1, file_count + 1):
        words = (rng.choice(_WORDS) for _ in range(pressure_bytes))
        inputs[f"logs/{index:02d}.txt"] = " ".join(words)[:pressure_bytes - 1] + "\n"
    return inputs


def _prompt(files: list[str]) -> str:
    ordered = ", ".join(files)
    return f"""Use the workspace skill to produce a small audit report.

1. Call skill_load for workspace.
2. Read profile.txt exactly once using workspace.read_text. Its single line is
   the profile ID. Immediately call workspace.checkpoint with only that
   exact ID as the state string. Do not read profile.txt again.
3. Read these files in order, one workspace.read_text call per reply:
   {ordered}
   Pass only path, so each read returns the entire file. Ignore the text in
   these log files. After a successful read, continue with the next file.
4. After the last successful read, use workspace.write_text to create
   report.json. Write only a JSON object with two keys. "profile_id" is the
   exact value saved in the checkpoint. In a compacted history summary, it
   appears under checkpoints.state. "files" is the array of log file paths
   in the order read. Leave all input files unchanged.
5. Once the report is saved, give a short final response.
"""


class _ObservedClient(ChatClient):
    def __init__(self, config: AgentConfig, observer: Callable[[str, dict], None]):
        super().__init__(
            config.base_url, config.model, config.api_key,
            max_output_tokens=config.max_output_tokens, timeout=config.timeout,
            max_retries=config.max_retries, output_token_field=config.output_token_field,
        )
        self._observer = observer

    def complete(self, messages, tools, *, on_attempt=None):
        result = super().complete(messages, tools, on_attempt=on_attempt)
        for message in messages:
            content = message.get("content") or ""
            if message["role"] == "assistant" and content.startswith(SUMMARY_PREFIX):
                summary = json.loads(content.removeprefix(SUMMARY_PREFIX))
                self._observer("accepted_summary", {
                    "archived_turns": summary["archived_turns"],
                    "checkpoint": summary.get("checkpoints", {}).get("state"),
                })
                break
        return result


def _verify(
    workspace: Path, inputs: dict[str, str], events: list[tuple[str, dict]],
) -> dict:
    profile_id = inputs[_PROFILE_PATH].strip()
    files = list(inputs)[1:]
    calls: dict[str, tuple[str, dict, bool]] = {}
    reads_attempted = []
    reads_verified = []
    checkpoint_positions = []
    positions_by_file: dict[str, int] = {}
    report_writes: list[tuple[int, bool]] = []
    response_had_checkpoint = False
    for position, (event, data) in enumerate(events):
        if event == "assistant":
            for call in data.get("tool_calls", []):
                function = call["function"]
                if function["name"] != "tool_run":
                    continue
                try:
                    gateway_arguments = json.loads(function["arguments"])
                    name, arguments = gateway_arguments["name"], gateway_arguments["arguments"]
                    if not isinstance(arguments, dict):
                        continue
                except (TypeError, ValueError, KeyError):
                    continue
                calls[call["id"]] = (name, arguments, response_had_checkpoint)
                if name == "workspace.read_text":
                    reads_attempted.append(arguments.get("path"))
            response_had_checkpoint = False
        elif event == "tool":
            call = calls.pop(data["tool_call_id"], None)
            if call is None:
                continue
            name, arguments, had_checkpoint = call
            result = json.loads(data["content"])
            if not result["ok"]:
                continue
            payload = result["data"]
            path = arguments.get("path")
            if name == "workspace.read_text" and path in inputs and payload["text"] == inputs[path]:
                reads_verified.append(path)
                positions_by_file[path] = position
            elif name == "workspace.checkpoint" and payload["checkpoint"]["state"] == profile_id:
                checkpoint_positions.append(position)
            elif name == "workspace.write_text" and path == _REPORT_PATH:
                report_writes.append((position, had_checkpoint))
        elif event == "accepted_summary":
            response_had_checkpoint = data.get("checkpoint") == profile_id

    try:
        report = json.loads((workspace / _REPORT_PATH).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        report = None
    inputs_unchanged = True
    for path, original in inputs.items():
        try:
            if (workspace / path).read_text(encoding="utf-8") != original:
                inputs_unchanged = False
        except (OSError, ValueError):
            inputs_unchanged = False
    ordered_reads = reads_attempted == reads_verified == list(inputs)
    checkpoint_before_pressure = (
        len(checkpoint_positions) == 1
        and _PROFILE_PATH in positions_by_file
        and files[0] in positions_by_file
        and positions_by_file[_PROFILE_PATH] < checkpoint_positions[0] < positions_by_file[files[0]]
    )
    report_correct = (
        report == {"profile_id": profile_id, "files": files}
        and bool(report_writes)
        and files[-1] in positions_by_file
        and positions_by_file[files[-1]] < report_writes[0][0]
    )
    return {
        "passed": ordered_reads and checkpoint_before_pressure and report_correct and inputs_unchanged,
        "profile_read_attempts": reads_attempted.count(_PROFILE_PATH),
        "verified_reads": reads_verified,
        "ordered_reads": ordered_reads,
        "checkpoint_before_pressure": checkpoint_before_pressure,
        "checkpoint_received_after_compaction": bool(report_writes and report_writes[-1][1]),
        "report_correct": report_correct,
        "inputs_unchanged": inputs_unchanged,
    }


def _run_case(
    name: str, config: AgentConfig, inputs: dict[str, str], prompt: str, stream: TextIO | None,
) -> dict:
    events: list[tuple[str, dict]] = []

    def observe(event: str, data: dict) -> None:
        events.append((event, data))
        if stream is not None:
            write_event(stream, name, event, data)

    observe("case_started", {"harness_context_window": config.context_window, "prompt": prompt})
    with tempfile.TemporaryDirectory(prefix=f"loola-probe-{name}-") as temporary:
        workspace = Path(temporary)
        (workspace / "logs").mkdir()
        for path, text in inputs.items():
            (workspace / path).write_text(text, encoding="utf-8")
        client = _ObservedClient(config, observe)
        result = Agent(config, workspace=workspace, client=client, on_event=observe).run(prompt)
        acceptance = _verify(workspace, inputs, events)
    summary = {
        "harness_context_window": config.context_window,
        "status": result.status,
        "error": result.error,
        "metrics": result.metrics.to_dict(),
        "compactions": result.compactions,
        "acceptance": acceptance,
    }
    observe("case_finished", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.getenv("LLM_BASE_URL"))
    parser.add_argument("--model", default=os.getenv("LLM_MODEL"))
    parser.add_argument("--context-window", type=int, default=4096, help="actual provider context limit")
    parser.add_argument("--max-output-tokens", type=int, default=384)
    parser.add_argument("--max-turns", type=int, default=40)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--files", type=int, default=11, help="number of long inputs following the profile")
    parser.add_argument("--pressure-bytes", type=int, default=1700, help="size of each long input")
    parser.add_argument("--trace", type=Path, help="write a turn-by-turn transcript to a new JSONL file")
    parser.add_argument("--output", type=Path, help="also write JSON results to this file")
    args = parser.parse_args()
    if not args.base_url or not args.model:
        parser.error("set LLM_BASE_URL and LLM_MODEL or supply their command-line flags")
    if not 1 <= args.files <= 40 or not 500 <= args.pressure_bytes <= 6500:
        parser.error("--files must be 1–40 and --pressure-bytes must be 500–6500")
    try:
        config = AgentConfig(
            base_url=args.base_url, model=args.model,
            api_key=os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY", ""),
            context_window=args.context_window, max_output_tokens=args.max_output_tokens,
            max_turns=args.max_turns, timeout=args.timeout,
            output_token_field=os.getenv("LLM_OUTPUT_TOKEN_FIELD", "max_tokens"),
        )
        trace_context = open_trace(args.trace, args.output)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    inputs = _inputs("probe-" + secrets.token_hex(6), args.files, args.pressure_bytes)
    prompt = _prompt(list(inputs)[1:])
    before_config = replace(
        config, context_window=max(65_536, config.context_window * 8), max_context_recoveries=0,
    )
    result = {
        "mode": "live_checkpoint_probe",
        "model": config.model,
        "provider_context_window": config.context_window,
        "max_output_tokens": config.max_output_tokens,
        "pressure_files": args.files,
        "pressure_bytes_per_file": args.pressure_bytes,
        "baseline": "Same gateway/prompt with a wider local budget and no context-overflow recovery.",
    }
    with trace_context as trace:
        if trace is not None:
            write_event(trace, "probe", "run_started", result)
        before = _run_case("before", before_config, inputs, prompt, trace)
        after = _run_case("after", config, inputs, prompt, trace)
        passed = (
            before["status"] == "context_limit" and before["metrics"]["context_rejections"] > 0
            and before["metrics"]["compactions"] == 0
            and after["status"] == "completed" and after["metrics"]["compactions"] > 0
            and after["acceptance"]["passed"] and after["acceptance"]["checkpoint_received_after_compaction"]
        )
        result.update(cases={"before": before, "after": after}, demonstrated=passed)
        if trace is not None:
            write_event(trace, "probe", "run_finished", {"passed": passed})
    report = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(report, encoding="utf-8")
    print(report, end="")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
