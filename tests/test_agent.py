import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

from harness.agent import Agent, AgentConfig
from harness.provider import ChatClient
from harness.tools import ToolInfrastructureError


def call(call_id, name, arguments):
    return {
        "id": call_id, "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def completion(content=None, calls=None, *, finish=None, usage=None):
    message = {"role": "assistant", "content": content}
    if calls is not None:
        message["tool_calls"] = calls
    return {"choices": [{"message": message, "finish_reason": finish}], "usage": usage}


class ScriptedEndpoint:
    def __init__(self, *steps):
        self.steps = iter(steps)
        self.requests = []

    def __call__(self, request, *, timeout):
        self.requests.append(json.loads(request.data))
        step = next(self.steps)
        if isinstance(step, tuple):
            status, error = step
            raise HTTPError(
                request.full_url, status, "", {}, io.BytesIO(json.dumps({"error": error}).encode()),
            )
        return io.BytesIO(json.dumps(step).encode())


class AgentTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.workspace = Path(directory.name)
        self.config = AgentConfig(
            base_url="http://example.invalid/v1", model="test", context_window=4096,
            max_output_tokens=700, cache_key="test-key",
        )
        self.delays = []

    def run_agent(self, endpoint, on_event=None):
        client = ChatClient(
            self.config.base_url, self.config.model, max_output_tokens=self.config.max_output_tokens,
            cache_key=self.config.cache_key, opener=endpoint, sleep=self.delays.append,
        )
        return Agent(
            self.config, workspace=self.workspace, client=client, on_event=on_event,
        ).run("Create a report.")

    def test_atomic_turn_and_model_correctable_json(self):
        invalid = call("2", "tool_run", None)
        invalid["function"]["arguments"] = '{"name":"a","name":"b"}'
        endpoint = ScriptedEndpoint(
            completion("Starting", [call("1", "skill_load", {"skill": "workspace"}), invalid]),
            completion("Done", usage={
                "prompt_tokens": 123, "completion_tokens": 4,
                "prompt_tokens_details": {"cached_tokens": 80},
            }),
        )
        result = self.run_agent(endpoint)
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.metrics.tool_calls, 2)
        self.assertEqual(result.metrics.tool_errors, 1)
        self.assertEqual(result.metrics.prompt_tokens, 123)
        self.assertEqual(result.metrics.usage_responses, 1)
        self.assertEqual(result.metrics.cached_prompt_tokens, 80)
        first, second = endpoint.requests
        self.assertEqual(first["tools"], second["tools"])
        self.assertEqual(first["prompt_cache_key"], second["prompt_cache_key"])
        self.assertEqual(first["messages"], second["messages"][:2])
        assistant, *results = second["messages"][2:]
        self.assertEqual(assistant["content"], "Starting")
        self.assertEqual(len(assistant["tool_calls"]), 2)
        self.assertEqual([response["tool_call_id"] for response in results], ["1", "2"])
        self.assertEqual(json.loads(results[1]["content"])["error"]["code"], "invalid_arguments")

    def test_overflow_retries_inference_without_repeating_tools(self):
        endpoint = ScriptedEndpoint(
            completion("I will use the workspace. " * 100, [call("1", "skill_load", {"skill": "workspace"})]),
            (400, {"code": "context_length_exceeded"}),
            completion("The workspace is ready."),
        )
        result = self.run_agent(endpoint)
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.metrics.requests, 3)
        self.assertEqual(result.metrics.context_rejections, 1)
        self.assertEqual(result.metrics.tool_calls, 1)
        self.assertEqual(result.compactions[0]["reason"], "provider_overflow")
        self.assertEqual(endpoint.requests[1]["messages"][:2], endpoint.requests[2]["messages"][:2])
        self.assertLess(result.compactions[0]["after_tokens"], result.compactions[0]["before_tokens"])
        self.assertEqual(self.delays, [])

    def test_llama_overflow_trace_records_recovery_before_next_response(self):
        first = completion("I will use the workspace. " * 100, [call("1", "skill_list", {})])
        first["choices"][0]["message"]["reasoning_content"] = "private metadata"
        endpoint = ScriptedEndpoint(
            first, (400, {"type": "exceed_context_size"}), completion("Done."),
        )
        observed = []
        result = self.run_agent(endpoint, lambda kind, payload: observed.append((kind, payload)))
        self.assertEqual(result.status, "completed")
        self.assertEqual([kind for kind, _ in observed], [
            "usage", "assistant", "tool", "context_rejection", "compaction", "usage", "assistant",
        ])
        self.assertNotIn("reasoning_content", observed[1][1])
        self.assertEqual(observed[3][1], {"http_status": 400, "request": 2})
        self.assertEqual(observed[4][1], result.compactions[0])

    def test_bad_batch_is_rejected_before_dispatch(self):
        endpoint = ScriptedEndpoint(completion(calls=[
            call("duplicate", "skill_load", {"skill": "workspace"}),
            call("duplicate", "skill_list", {}),
        ]))
        result = self.run_agent(endpoint)
        self.assertEqual((result.status, result.error), ("provider_error", "invalid_tool_batch"))
        self.assertEqual(result.metrics.tool_calls, 0)

    def test_length_stops_before_executing_partial_calls(self):
        endpoint = ScriptedEndpoint(completion("partial", [call("1", "skill_list", {})], finish="length"))
        result = self.run_agent(endpoint)
        self.assertEqual(result.status, "output_limit")
        self.assertEqual(result.metrics.tool_calls, 0)

    def test_transient_provider_error_retries_but_auth_failure_does_not(self):
        endpoint = ScriptedEndpoint(
            (503, {"message": "unavailable"}),
            (401, {"message": "sensitive provider error details"}),
        )
        result = self.run_agent(endpoint)
        self.assertEqual((result.status, result.error), ("provider_error", "provider_access_denied (HTTP 401)"))
        self.assertEqual((result.metrics.requests, result.metrics.retries), (2, 1))
        self.assertEqual(self.delays, [0.5])
        self.assertEqual(endpoint.requests[0], endpoint.requests[1])
        self.assertIsNone(result.metrics.prompt_tokens)

    def test_infrastructure_failure_stops_once(self):
        endpoint = ScriptedEndpoint(completion(calls=[call("1", "skill_list", {})]))
        with patch("harness.agent.ToolRegistry.dispatch", side_effect=ToolInfrastructureError("private")):
            result = self.run_agent(endpoint)
        self.assertEqual((result.status, result.error), ("tool_error", "tool_infrastructure_failure"))
        self.assertEqual((result.metrics.requests, result.metrics.tool_errors), (1, 1))


if __name__ == "__main__":
    unittest.main()
