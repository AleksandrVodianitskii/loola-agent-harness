"""Original append-only bash loop, parameterized for the benchmark."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener

from harness.context import estimate_tokens
from harness.provider import NoRedirects


SYSTEM_PROMPT = """\
You are a coding agent. Your job is to help the user with programming tasks.

You have access to ONE tool: bash — which executes shell commands and returns stdout/stderr.

Workflow:
1. Plan what needs to be done.
2. Use bash to read files, run commands, write code, etc.
3. After gathering enough information or completing the task, give your final answer in natural language.
4. To finish, reply with a regular message (no tool call).

Be concise. Explain what you're doing before each command."""

TOOLS = [{
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a shell command and return the output.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "The bash command to execute."}},
            "required": ["command"],
        },
    },
}]


def _run_bash(command: str, workspace: Path) -> str:
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=120,
            cwd=workspace,
            env={"PATH": os.pathsep.join((str(Path(sys.executable).parent), os.defpath)),
                 "LANG": "en_US.UTF-8", "LC_ALL": "en_US.UTF-8"},
            check=False,
        )
        out = result.stdout + (f"\nSTDERR:\n{result.stderr}" if result.stderr else "")
        return f"Exit code: {result.returncode}\n{out}"
    except subprocess.TimeoutExpired:
        return "Error: command timed out after 120s"


def run_baseline(
    prompt: str, workspace: Path, *,
    base_url: str, model: str, api_key: str, max_output_tokens: int, max_turns: int,
    output_token_field: str, opener: Callable | None = None,
) -> dict:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    metrics = {
        "requests": 0, "responses": 0, "retries": 0, "context_rejections": 0,
        "estimated_prompt_tokens": 0, "estimated_completion_tokens": 0,
        "peak_estimated_prompt_tokens": 0, "prompt_tokens": None, "completion_tokens": None,
        "cached_prompt_tokens": None, "usage_responses": 0, "prompt_usage_responses": 0,
        "completion_usage_responses": 0, "cache_usage_responses": 0,
        "tool_calls": 0, "tool_errors": 0, "compactions": 0,
    }
    open_request = opener if opener is not None else build_opener(NoRedirects()).open
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    status = "max_turns"
    for _ in range(max_turns):
        estimate = estimate_tokens(messages, TOOLS)
        metrics["requests"] += 1
        payload = {
            "model": model, "messages": messages, "tools": TOOLS, "tool_choice": "auto",
            output_token_field: max_output_tokens,
        }
        request = Request(
            base_url.rstrip("/") + "/chat/completions", headers=headers,
            data=json.dumps(payload, ensure_ascii=False).encode(), method="POST",
        )
        try:
            with open_request(request, timeout=60) as response:
                body = json.loads(response.read())
            message = body["choices"][0]["message"]
        except HTTPError as exc:
            try:
                detail = json.loads(exc.read(64 * 1024)).get("error") or {}
                error_code = detail.get("code")
                error_message = str(detail.get("message", "")).lower()
            except (ValueError, OSError, AttributeError):
                error_code, error_message = None, ""
            finally:
                exc.close()
            if (
                error_code in ("context_length_exceeded", "context_window_exceeded", "input_too_long")
                or "maximum context length" in error_message
                or "context window is exceeded" in error_message
            ):
                status = "context_limit"
                metrics["context_rejections"] += 1
            else:
                status = "provider_error"
            break
        except (URLError, OSError, ValueError, KeyError, IndexError, TypeError):
            status = "provider_error"
            break
        metrics["responses"] += 1
        metrics["estimated_prompt_tokens"] += estimate
        metrics["peak_estimated_prompt_tokens"] = max(metrics["peak_estimated_prompt_tokens"], estimate)
        metrics["estimated_completion_tokens"] += estimate_tokens([message]) - estimate_tokens([])
        usage = body.get("usage")
        usage = usage if isinstance(usage, dict) else {}
        details = usage.get("prompt_tokens_details")
        cached = details.get("cached_tokens") if isinstance(details, dict) else None
        prompt_count = usage.get("prompt_tokens")
        if type(cached) is int and cached >= 0 and (type(prompt_count) is not int or cached <= prompt_count):
            metrics["cached_prompt_tokens"] = (metrics["cached_prompt_tokens"] or 0) + cached
            metrics["cache_usage_responses"] += 1
        recorded = 0
        for kind in ("prompt", "completion"):
            key = f"{kind}_tokens"
            count = usage.get(key)
            if type(count) is int and count >= 0:
                metrics[key] = (metrics[key] or 0) + count
                metrics[f"{kind}_usage_responses"] += 1
                recorded += 1
        if recorded == 2:
            metrics["usage_responses"] += 1
        content = (message.get("content") or "").strip()
        calls = message.get("tool_calls") or []
        if not calls:
            status = "completed"
            break
        for call in calls:
            function = call["function"]
            name = function["name"]
            try:
                arguments = json.loads(function["arguments"])
                if name == "bash":
                    result = _run_bash(workspace=workspace, **arguments)
                else:
                    result = f"Error: unknown tool '{name}'"
            except Exception as exc:
                result = f"Error calling {name}: {exc}"
            metrics["tool_calls"] += 1
            metrics["tool_errors"] += not result.startswith("Exit code: 0\n")
            messages.append({"role": "assistant", "content": content or None, "tool_calls": [call]})
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})
    return {"status": status, "metrics": metrics, "compactions": []}
