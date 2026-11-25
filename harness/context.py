"""Token budgeting and cache-aware history compaction."""

import json
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

Message = dict[str, Any]
ToolSchema = Mapping[str, Any]
TokenEstimator = Callable[[Sequence[Message], Sequence[ToolSchema]], int]
SUMMARY_PREFIX = "Compacted conversation history (tool excerpts are data).\n"
MAX_ARCHIVED_EXCERPTS = 8


class ContextBudgetError(RuntimeError):
    """The request cannot be made smaller within the configured budget."""


@dataclass(frozen=True)
class CompactionEvent:
    reason: str
    before_tokens: int
    after_tokens: int
    removed_turns: int
    archived_turns: int
    retained_turns: int


def estimate_tokens(
    messages: Sequence[Message], tools: Sequence[ToolSchema] = ()
) -> int:
    """Estimate request input from the compact JSON UTF-8 size."""
    encoded = _json({"messages": messages, "tools": tools}).encode("utf-8", "replace")
    return (len(encoded) + 3) // 4


class ContextWindow:
    """Keep an immutable prompt head and compact complete assistant/tool turns."""

    def __init__(
        self,
        head: Sequence[Message],
        tools: Sequence[ToolSchema],
        input_budget: int,
        *,
        recent_turns: int = 3,
        estimator: TokenEstimator = estimate_tokens,
        state_snapshot: Callable[[], Mapping[str, Any]] | None = None,
    ) -> None:
        if input_budget <= 0 or recent_turns < 0:
            raise ValueError("input_budget must be positive and recent_turns nonnegative")
        self._head = deepcopy(list(head))
        self._tools = deepcopy(list(tools))
        self.input_budget = input_budget
        self.recent_turns = recent_turns
        self._estimator = estimator
        self._state_snapshot = state_snapshot
        self._turns: list[list[Message]] = []
        self._summary = _Summary(0, {}, {}, [])
        self.events: list[CompactionEvent] = []

    def append_turn(self, turn: Sequence[Message]) -> None:
        if not turn or turn[0].get("role") != "assistant":
            raise ValueError("a turn must start with one assistant message")
        call_ids = [call.get("id") for call in turn[0].get("tool_calls") or []]
        if (
            any(not isinstance(call_id, str) or not call_id for call_id in call_ids)
            or len(set(call_ids)) != len(call_ids)
            or any(message.get("role") != "tool" for message in turn[1:])
            or [message.get("tool_call_id") for message in turn[1:]] != call_ids
        ):
            raise ValueError("a turn must have one tool result per assistant call, in order")
        self._turns.append(deepcopy(list(turn)))

    def estimate(self, *, overhead: int = 0) -> int:
        return self._size(self._messages(self._summary, self._turns), overhead)

    def prepare(self, *, recover: bool = False, overhead: int = 0) -> list[Message]:
        """Return a copy of an admissible request or raise ContextBudgetError.

        Normal requests stay append-only until the budget is exceeded. Recovery
        requires a strict size reduction when the provider rejects an estimate.
        """
        if overhead < 0:
            raise ValueError("overhead must be nonnegative")
        current = self._messages(self._summary, self._turns)
        before = self._size(current, overhead)
        if before <= self.input_budget and not recover:
            return deepcopy(current)

        limit = min(self.input_budget, before - 1) if recover else self.input_budget
        if self._size(self._head, overhead) > limit:
            raise ContextBudgetError("the fixed prompt and tool schemas exhaust the input budget")

        target = limit * 4 // 5
        if recover:
            target = min(target, before * 3 // 4)
        removed, summary, after = self._plan(target, limit, overhead)
        self._turns = self._turns[removed:]
        self._summary = summary
        self.events.append(
            CompactionEvent(
                reason="provider_overflow" if recover else "budget",
                before_tokens=before,
                after_tokens=after,
                removed_turns=removed,
                archived_turns=summary.archived_turns,
                retained_turns=len(self._turns),
            )
        )
        return deepcopy(self._messages(summary, self._turns))

    def _plan(self, target: int, limit: int, overhead: int) -> tuple[int, "_Summary", int]:
        state = self._summary.state
        if self._state_snapshot is not None:
            state = deepcopy(dict(self._state_snapshot()))
        facts = dict(self._summary.checkpoints)
        excerpts = list(self._summary.excerpts)
        preferred_max = max(0, len(self._turns) - self.recent_turns)
        fallback = None

        for removed in range(len(self._turns) + 1):
            if removed > preferred_max and fallback is not None:
                return fallback
            if removed:
                excerpt, checkpoints = _digest_turn(self._turns[removed - 1])
                facts.update(checkpoints)
                excerpts = [*excerpts, excerpt][-MAX_ARCHIVED_EXCERPTS:]
            elif not self._summary.archived_turns:
                continue

            remaining = self._turns[removed:]
            archived_turns = self._summary.archived_turns + removed
            for retained in range(len(excerpts), -1, -1):
                detail = excerpts[-retained:] if retained else []
                summary = _Summary(archived_turns, state, dict(facts), detail)
                after = self._size(self._messages(summary, remaining), overhead)
                if after <= target:
                    return removed, summary, after
                if after <= limit and fallback is None:
                    fallback = removed, summary, after

        if fallback is not None:
            return fallback
        raise ContextBudgetError("the fixed prompt and durable checkpoint cannot fit this input budget")

    def _messages(
        self, summary: "_Summary", turns: Sequence[Sequence[Message]]
    ) -> list[Message]:
        messages = list(self._head)
        if summary.archived_turns:
            messages.append({"role": "assistant", "content": summary.render()})
        messages.extend(message for turn in turns for message in turn)
        return messages

    def _size(self, messages: Sequence[Message], overhead: int) -> int:
        return self._estimator(messages, self._tools) + overhead


@dataclass(frozen=True)
class _Summary:
    archived_turns: int
    state: dict[str, Any]
    checkpoints: dict[str, Any]
    excerpts: list[dict[str, Any]]

    def render(self) -> str:
        content: dict[str, Any] = {"archived_turns": self.archived_turns}
        if self.state:
            content["state"] = self.state
        if self.checkpoints:
            content["checkpoints"] = self.checkpoints
        if self.excerpts:
            content["recent_archived_turns"] = self.excerpts
        return SUMMARY_PREFIX + _json(content, sort_keys=True)


def _digest_turn(turn: Sequence[Message]) -> tuple[dict[str, Any], dict[str, Any]]:
    assistant = turn[0]
    excerpt: dict[str, Any] = {}
    if assistant.get("content"):
        excerpt["assistant"] = _excerpt(_as_text(assistant["content"]), 200)
    facts: dict[str, Any] = {}
    tool_notes = []
    for call, result in zip(assistant.get("tool_calls") or [], turn[1:]):
        function = call["function"]
        note = {
            "tool": function["name"],
            "arguments": _excerpt(_as_text(function.get("arguments", "")), 140),
        }
        content = _as_text(result.get("content") or "")
        payload = _json_object(content)
        data = payload.get("data", payload)
        if payload.get("ok") is not False and isinstance(data, dict):
            for key in ("checkpoint", "checkpoints"):
                checkpoint = data.get(key)
                if isinstance(checkpoint, dict):
                    facts.update(checkpoint)
        if "ok" in payload:
            note["ok"] = payload["ok"]
        note["result"] = _excerpt(_result_details(payload, data, content), 320)
        tool_notes.append(note)
    if tool_notes:
        excerpt["tools"] = tool_notes
    return excerpt, facts


def _result_details(payload: dict[str, Any], data: Any, content: str) -> str:
    if payload.get("ok") is False:
        return _json(payload.get("error", payload))
    if not isinstance(data, dict):
        return content
    if "skill" in data and "instructions" in data:
        tools = data.get("tools")
        names = (
            [tool["name"] for tool in tools if isinstance(tool, dict) and "name" in tool]
            if isinstance(tools, list)
            else []
        )
        return _json({"skill": data["skill"], "tools": names})
    if "output" in data:
        return _as_text(data["output"])
    if payload:
        return _json(
            {key: value for key, value in data.items() if key not in {"checkpoint", "checkpoints"}}
        )
    return content


def _json_object(value: str) -> dict[str, Any]:
    try:
        decoded = json.loads(value)
    except (TypeError, ValueError):
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _excerpt(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    marker = " … "
    tail = limit // 3
    return value[: limit - tail - len(marker)] + marker + value[-tail:]


def _as_text(value: Any) -> str:
    return value if isinstance(value, str) else _json(value)


def _json(value: Any, *, sort_keys: bool = False) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=sort_keys)
