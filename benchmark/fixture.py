"""A deterministic chat/completions fixture with an explicit token model."""

from __future__ import annotations

import io
import json
import os
import re
import shlex
import textwrap
from collections import defaultdict
from urllib.error import HTTPError
from urllib.request import Request

from harness.context import SUMMARY_PREFIX, estimate_tokens


def _frame(tools: list[dict], messages: list[dict]) -> bytes:
    """Serialize the tool list first, then frame each message incrementally."""
    lines = [tools, *messages]
    return "".join(
        json.dumps(line, ensure_ascii=False, separators=(",", ":")) + "\n"
        for line in lines
    ).encode("utf-8")


def _normalize(records: list[dict]) -> list[dict]:
    totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for record in records:
        if record["enabled"] is True:
            sku = "-".join(record["sku"].upper().split())
            totals[sku][0] += record["units"]
            totals[sku][1] += record["reserved"]
    return [
        {"sku": sku, "units": units, "reserved": reserved, "available": units - reserved}
        for sku, (units, reserved) in sorted(totals.items())
    ]


def _audit_source(scaffold: str, batch_count: int, threshold: int) -> str:
    return textwrap.dedent("""\
        import argparse
        import json
        from pathlib import Path

        """) + scaffold + f"BATCH_COUNT = {batch_count}\nLOW_STOCK_BELOW = {threshold}\n" + textwrap.dedent("""


        def audit(directory):
            total_available = 0
            low_stock = set()
            for index in range(BATCH_COUNT):
                path = directory / f"{index:02d}.json"
                batch = json.loads(path.read_text(encoding="utf-8"))
                for item in batch["items"]:
                    available = item["available"]
                    total_available += available
                    if available < LOW_STOCK_BELOW:
                        low_stock.add(item["sku"])
            return {
                "migration_id": MIGRATION_ID,
                "schema_version": SCHEMA_VERSION,
                "batch_count": BATCH_COUNT,
                "total_available": total_available,
                "low_stock": sorted(low_stock),
            }


        if __name__ == "__main__":
            parser = argparse.ArgumentParser()
            parser.add_argument("directory", type=Path, nargs="?",
                                default=Path(__file__).parent / "normalized")
            args = parser.parse_args()
            try:
                print(json.dumps(audit(args.directory)))
            except (OSError, ValueError, KeyError) as exc:
                parser.exit(1, f"Audit failed: {exc}\\n")
        """)


def _scaffold(profile: dict) -> str:
    return (
        f"MIGRATION_ID = {profile['migration_id']!r}\n"
        f"SCHEMA_VERSION = {profile['schema_version']!r}\n"
    )


class FixtureProvider:
    """Return protocol JSON from the visible conversation and fixed task logic.

    Only telemetry and synthetic response IDs survive between requests. The
    fixture derives the next action from the original user prompt and the
    latest complete tool exchange. It reads values from actual tool output.
    """

    def __init__(self, context_window: int):
        self.context_window = context_window
        self.accepted_requests = 0
        self.rejected_requests = 0
        self.peak_prompt_tokens = 0
        self.first_rejected_prompt_tokens: int | None = None
        self.checkpoint_source: str | None = None
        self._last_prompt = b""
        self._heads: set[bytes] = set()
        self._tool_schemas: set[bytes] = set()
        self._prefixes_at_rewrites: list[int] = []

    def telemetry(self) -> dict:
        return {
            "accepted_requests": self.accepted_requests,
            "rejected_requests": self.rejected_requests,
            "peak_accepted_prompt_tokens": self.peak_prompt_tokens,
            "first_rejected_prompt_tokens": self.first_rejected_prompt_tokens,
            "fixed_head_variants": len(self._heads),
            "tool_schema_variants": len(self._tool_schemas),
            "estimated_prefix_tokens_at_history_rewrites": self._prefixes_at_rewrites,
            "checkpoint_source_at_completion": self.checkpoint_source,
        }

    def __call__(self, request: Request, timeout: float | None = None) -> io.BytesIO:
        if request.get_method() != "POST" or not request.full_url.endswith("/chat/completions"):
            raise AssertionError("expected POST /chat/completions")
        payload = json.loads(request.data)
        messages, tools = payload["messages"], payload["tools"]
        prompt_tokens = estimate_tokens(messages, tools)
        reserved = payload.get("max_tokens", payload.get("max_completion_tokens", 0))
        if prompt_tokens + reserved > self.context_window:
            self.rejected_requests += 1
            if self.first_rejected_prompt_tokens is None:
                self.first_rejected_prompt_tokens = prompt_tokens
            error = {"error": {
                "message": f"Context window {self.context_window} exceeded by "
                           f"estimated prompt {prompt_tokens} + reserve {reserved}.",
                "type": "invalid_request_error",
                "code": "context_length_exceeded",
            }}
            raise HTTPError(
                request.full_url, 400, "Context too large", None, io.BytesIO(json.dumps(error).encode())
            )

        prompt = _frame(tools, messages)
        self._heads.add(_frame(tools, messages[:2]))
        self._tool_schemas.add(_frame(tools, []))
        cached = len(os.path.commonprefix((self._last_prompt, prompt))) // 4
        if self._last_prompt and not prompt.startswith(self._last_prompt):
            self._prefixes_at_rewrites.append(cached)
        self._last_prompt = prompt
        self.accepted_requests += 1
        self.peak_prompt_tokens = max(self.peak_prompt_tokens, prompt_tokens)
        message = self._reply(messages, tools)
        finish = "tool_calls" if message.get("tool_calls") else "stop"
        completion_tokens = estimate_tokens([message]) - estimate_tokens([])
        if completion_tokens > reserved:
            raise ValueError(f"fixture reserve must be at least {completion_tokens} tokens")
        answer = {
            "id": f"fixture-{self.accepted_requests:04d}",
            "object": "chat.completion",
            "model": payload["model"],
            "choices": [{"index": 0, "finish_reason": finish, "message": message}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "prompt_tokens_details": {"cached_tokens": cached},
            },
        }
        return io.BytesIO(json.dumps(answer).encode())

    def _reply(self, messages: list[dict], tools: list[dict]) -> dict:
        enhanced = any(tool["function"]["name"] == "skill_list" for tool in tools)
        original = messages[1]["content"]
        last_index = int(re.search(r"through raw/(\d+)\.json", original)[1])
        threshold = int(re.search(r"strictly below (\d+)", original)[1])
        if len(messages) == 2:
            if enhanced:
                return self._tool_call("skill_list", {})
            return self._file_action(enhanced, "read", "migration_profile.json")
        assistant, result = messages[-2:]
        if assistant.get("role") != "assistant" or result.get("role") != "tool":
            raise AssertionError("fixture requires the latest complete tool exchange")
        function = assistant["tool_calls"][-1]["function"]
        name, args = function["name"], json.loads(function["arguments"])

        if enhanced:
            response = json.loads(result["content"])
            if not response["ok"]:
                raise AssertionError(f"fixture tool failed: {response}")
            data = response["data"]
            if name == "skill_list":
                workspace = next(skill for skill in data["skills"] if skill["name"] == "workspace")
                if workspace["active"]:
                    raise AssertionError("workspace skill should be inactive before loading")
                return self._tool_call("skill_load", {"skill": "workspace"})
            if name == "skill_load":
                names = {tool["name"] for tool in data["tools"]}
                if not {"workspace.read_text", "workspace.write_text", "workspace.checkpoint"} <= names:
                    raise AssertionError("workspace skill did not return its tool definitions")
                return self._file_action(enhanced, "read", "migration_profile.json")
            if args["name"] == "workspace.checkpoint":
                return self._file_action(enhanced, "read", "raw/00.json")
            path = args["arguments"]["path"]
            read = args["name"] == "workspace.read_text"
            source = data["text"] if read else args["arguments"]["text"]
        else:
            prefix = "Exit code: 0\n"
            if not result["content"].startswith(prefix):
                raise AssertionError(f"baseline tool failed: {result['content']}")
            command = shlex.split(args["command"])
            path, read = command[-1], command[0] == "cat"
            source = result["content"][len(prefix):] if read else command[2]

        if path == "migration_profile.json":
            profile = json.loads(source)
            if enhanced:
                return self._tool_call("tool_run", {
                    "name": "workspace.checkpoint",
                    "arguments": {"state": json.dumps(profile, sort_keys=True)},
                })
            return self._file_action(enhanced, "write", "audit.py", _scaffold(profile))
        if path == "audit.py":
            if read:
                self.checkpoint_source = "workspace_file"
                text = _audit_source(source, last_index + 1, threshold)
                return self._file_action(enhanced, "write", path, text)
            if "def audit(" in source:
                return {
                    "role": "assistant",
                    "content": f"Migrated {last_index + 1} batches and implemented the audit CLI.",
                }
            return self._file_action(enhanced, "read", "raw/00.json")
        batch = int(path.rsplit("/", 1)[-1].removesuffix(".json"))
        if read:
            document = json.loads(source)
            text = json.dumps({
                "batch": document["batch"], "items": _normalize(document["records"])
            }) + "\n"
            return self._file_action(enhanced, "write", f"normalized/{batch:02d}.json", text)
        if batch < last_index:
            return self._file_action(enhanced, "read", f"raw/{batch + 1:02d}.json")
        if enhanced:
            state = self._read_checkpoint(messages)
            return self._file_action(
                enhanced, "write", "audit.py",
                _audit_source(_scaffold(state), last_index + 1, threshold),
            )
        return self._file_action(enhanced, "read", "audit.py")

    def _read_checkpoint(self, messages: list[dict]) -> dict:
        for message in reversed(messages):
            text = message.get("content") or ""
            if message["role"] == "tool":
                checkpoint = json.loads(text).get("data", {}).get("checkpoint")
                if checkpoint:
                    self.checkpoint_source = "recent_history"
                    return json.loads(checkpoint["state"])
            if message["role"] == "assistant" and text.startswith(SUMMARY_PREFIX):
                summary = json.loads(text.removeprefix(SUMMARY_PREFIX))
                self.checkpoint_source = "compacted_history"
                return json.loads(summary["checkpoints"]["state"])
        raise AssertionError("migration profile was lost before final audit")

    def _file_action(self, enhanced: bool, action: str, path: str, text: str = "") -> dict:
        if enhanced:
            params = {"path": path}
            if action == "write":
                params["text"] = text
            return self._tool_call("tool_run", {
                "name": f"workspace.{action}_text",
                "arguments": params,
            })
        command = f"cat {shlex.quote(path)}"
        if action == "write":
            command = f"printf %s {shlex.quote(text)} > {shlex.quote(path)}"
        return self._tool_call("bash", {"command": command})

    def _tool_call(self, name: str, arguments: dict) -> dict:
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": f"call_{self.accepted_requests:04d}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }],
        }
