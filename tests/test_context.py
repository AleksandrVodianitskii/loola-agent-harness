import json
import unittest

from harness.context import (
    SUMMARY_PREFIX,
    ContextBudgetError,
    ContextWindow,
    estimate_tokens,
)


HEAD = [
    {"role": "system", "content": "You are a coding agent."},
    {"role": "user", "content": "Build and verify the local ledger."},
]
TOOLS = [{"type": "function", "function": {"name": "fixture"}}]


def tool_turn(index, *results, content=None):
    calls = [
        {
            "id": f"call-{index}-{offset}",
            "type": "function",
            "function": {"name": "fixture", "arguments": '{"action":"collect"}'},
        }
        for offset, _ in enumerate(results)
    ]
    return [
        {"role": "assistant", "content": content, "tool_calls": calls},
        *[
            {"role": "tool", "tool_call_id": call["id"], "content": result}
            for call, result in zip(calls, results)
        ],
    ]


def synopsis(messages):
    content = messages[len(HEAD)]["content"]
    if not content.startswith(SUMMARY_PREFIX):
        raise AssertionError("expected a compaction summary")
    return json.loads(content.removeprefix(SUMMARY_PREFIX))


class ContextWindowTests(unittest.TestCase):
    def test_keeps_head_and_complete_recent_tool_batches(self):
        window = ContextWindow(HEAD, TOOLS, 1320, recent_turns=2)
        first = tool_turn(1, "first:" + "x" * 900, "second:" + "x" * 900)
        second = tool_turn(2, "x" * 1000, "x" * 1000)
        third = tool_turn(3, "x" * 950)
        window.append_turn(first)
        prepared = window.prepare()
        window.append_turn(second)
        self.assertEqual(window.prepare()[: len(prepared)], prepared)

        window.append_turn(third)
        prepared = window.prepare()
        self.assertEqual(prepared[: len(HEAD)], HEAD)
        self.assertEqual(prepared[-len(second + third) :], second + third)
        self.assertEqual(synopsis(prepared)["archived_turns"], 1)
        self.assertEqual(window.events[-1].retained_turns, 2)

        last = tool_turn(4, "read the ledger")
        window.append_turn(last)
        after = window.prepare()
        self.assertEqual(after[: len(prepared)], prepared)
        self.assertEqual(len(window.events), 1)
        after[0]["content"] = "external mutation"
        self.assertEqual(window.prepare()[0], HEAD[0])

    def test_retains_exact_checkpoint_values_across_compactions(self):
        facts = {"phase": "validation", "key": "c8c4 non-ASCII синяя птица"}
        state = {"active_skills": ["ledger"]}
        window = ContextWindow(HEAD, TOOLS, 330, recent_turns=1, state_snapshot=lambda: state)
        window.append_turn(
            tool_turn(
                1,
                json.dumps({"ok": True, "data": {"checkpoint": facts, "output": "z" * 5000}}),
            )
        )
        first = window.prepare()
        self.assertEqual(synopsis(first)["checkpoints"], facts)
        self.assertEqual(synopsis(first)["state"], state)

        state["active_skills"].append("verification")
        self.assertEqual(window.prepare(), first)
        window.append_turn(
            tool_turn(
                2,
                json.dumps(
                    {"ok": True, "data": {"checkpoints": {"phase": "finished"}, "output": "z" * 5000}}
                ),
            )
        )
        final = synopsis(window.prepare())
        self.assertEqual(final["state"], state)
        self.assertEqual(final["checkpoints"], {**facts, "phase": "finished"})
        self.assertEqual(final["archived_turns"], 2)

    def test_overflow_recovery_forces_progress_under_configured_limit(self):
        window = ContextWindow(HEAD, TOOLS, 2400, recent_turns=2)
        for index in range(4):
            window.append_turn(tool_turn(index, "x" * 700))
        current = window.prepare(overhead=41)
        self.assertEqual(window.events, [])
        prepared = window.prepare(recover=True, overhead=41)
        self.assertLess(estimate_tokens(prepared, TOOLS), estimate_tokens(current, TOOLS))
        event = window.events[-1]
        self.assertEqual(event.reason, "provider_overflow")
        self.assertEqual(event.after_tokens, estimate_tokens(prepared, TOOLS) + 41)

    def test_exact_checkpoint_that_exceeds_budget_is_not_silently_lost(self):
        window = ContextWindow(HEAD, TOOLS, 300)
        window.append_turn(
            tool_turn(1, json.dumps({"ok": True, "data": {"checkpoint": {"fact": "x" * 2000}}}))
        )
        with self.assertRaisesRegex(ContextBudgetError, "durable checkpoint"):
            window.prepare()
        self.assertEqual(window.events, [])

    def test_oversized_latest_turn_keeps_excerpt_boundaries_and_tool_error(self):
        window = ContextWindow(HEAD, TOOLS, 600)
        output = "leading value " + "x" * 4000 + " trailing value"
        error = json.dumps({"ok": False, "error": {"code": "invalid_arguments", "message": "Fix scope"}})
        window.append_turn(tool_turn(1, output, error, content="Verify the ledger entries."))
        detail = synopsis(window.prepare())["recent_archived_turns"][0]
        self.assertEqual(detail["assistant"], "Verify the ledger entries.")
        self.assertIn("leading value", detail["tools"][0]["result"])
        self.assertIn("trailing value", detail["tools"][0]["result"])
        self.assertFalse(detail["tools"][1]["ok"])
        self.assertIn("invalid_arguments", detail["tools"][1]["result"])

    def test_rejects_incomplete_or_reordered_tool_batches(self):
        window = ContextWindow(HEAD, TOOLS, 1000)
        assistant, first, second = tool_turn(1, "a", "b")
        for broken in ([assistant, first], [assistant, second, first]):
            with self.subTest(broken=broken):
                with self.assertRaisesRegex(ValueError, "one tool result"):
                    window.append_turn(broken)
        self.assertEqual(window.prepare(), HEAD)

    def test_returns_error_when_fixed_head_exhausts_input_budget(self):
        window = ContextWindow(HEAD, TOOLS, estimate_tokens(HEAD, TOOLS))
        self.assertEqual(window.prepare(), HEAD)
        with self.assertRaisesRegex(ContextBudgetError, "fixed prompt"):
            window.prepare(overhead=1)
        with self.assertRaisesRegex(ContextBudgetError, "fixed prompt"):
            window.prepare(recover=True)

    def test_estimator_counts_utf8_bytes(self):
        chars = "漢字" * 40
        self.assertGreater(
            estimate_tokens([{"role": "user", "content": chars}]),
            estimate_tokens([{"role": "user", "content": "z" * len(chars)}]),
        )


if __name__ == "__main__":
    unittest.main()
