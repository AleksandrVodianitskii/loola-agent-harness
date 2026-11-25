import json
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from harness import tools


class ToolRegistryTests(unittest.TestCase):
    def setUp(self):
        workspace = tempfile.TemporaryDirectory()
        self.addCleanup(workspace.cleanup)
        self.workspace = Path(workspace.name).resolve()
        self.registry = tools.ToolRegistry(self.workspace)

    def run_workspace(self, tool, **arguments):
        return self.registry.dispatch("tool_run", {"name": f"workspace.{tool}", "arguments": arguments})

    def test_discovery_and_activation_leave_gateway_schema_identical(self):
        schema = json.dumps(tools.GATEWAY_TOOLS)
        with patch("harness.tools.import_module", wraps=tools.import_module) as load:
            found = self.registry.dispatch("skill_list", {})
            self.assertEqual(found["data"]["skills"][0]["active"], False)
            unloaded = self.run_workspace("read_text", path="test.txt")
            self.assertEqual(unloaded["error"]["code"], "skill_not_loaded")
            load.assert_not_called()

            loaded = self.registry.dispatch("skill_load", {"skill": "workspace"})
            loaded["data"]["tools"][0]["parameters"].clear()
            reloaded = self.registry.dispatch("skill_load", {"skill": "workspace"})
            self.assertTrue(reloaded["data"]["tools"][0]["parameters"])
            self.assertEqual(self.registry.active_skill_names, ("workspace",))
            load.assert_called_once_with("harness._workspace_skill")
        self.assertEqual(schema, json.dumps(tools.GATEWAY_TOOLS))

    def test_bad_arguments_do_not_write_files(self):
        self.registry.dispatch("skill_load", {"skill": "workspace"})
        for args in ({"path": "a.txt"}, {"path": "a.txt", "text": []},
                     {"path": "a.txt", "text": "ok", "unexpected": 3}):
            with self.subTest(args=args):
                result = self.run_workspace("write_text", **args)
                self.assertEqual(result["error"]["code"], "invalid_arguments")
        result = self.run_workspace("read_text", path="missing", start_line=True)
        self.assertEqual(result["error"]["code"], "invalid_arguments")
        self.assertFalse((self.workspace / "a.txt").exists())

    def test_escape_by_path_or_symlink_is_rejected(self):
        self.registry.dispatch("skill_load", {"skill": "workspace"})
        (self.workspace / "escape").symlink_to(self.workspace.parent, target_is_directory=True)
        for path in ("../a.txt", "escape/a.txt", str(self.workspace.parent / "a.txt")):
            result = self.run_workspace("write_text", path=path, text="content")
            self.assertEqual(result["error"]["code"], "path_outside_workspace")

    def test_unicode_pages_can_be_reassembled_exactly(self):
        self.registry.dispatch("skill_load", {"skill": "workspace"})
        text = "😀" * 4500 + " café\r\nfinal line\n"
        self.assertTrue(self.run_workspace("write_text", path="nested/a.txt", text=text)["ok"])
        pages = []
        line, column = 1, 1
        while line is not None:
            result = self.run_workspace("read_text", path="nested/a.txt", start_line=line,
                                        start_column=column, max_lines=1)["data"]
            pages.append(result["text"])
            line, column = result["next_line"], result["next_column"]
        self.assertEqual("".join(pages), text)
        self.assertTrue(all(len(page.encode("utf-8")) <= 8192 for page in pages))

    def test_replace_preserves_mode_and_rejects_ambiguous_edit(self):
        self.registry.dispatch("skill_load", {"skill": "workspace"})
        self.run_workspace("write_text", path="a.txt", text="aaa")
        path = self.workspace / "a.txt"
        path.chmod(0o640)
        ambiguous = self.run_workspace("replace_text", path="a.txt", old="aa", new="b")
        self.assertEqual(ambiguous["error"]["code"], "edit_mismatch")
        self.assertEqual(path.read_text(), "aaa")
        self.assertTrue(self.run_workspace("replace_text", path="a.txt", old="aaa", new="b")["ok"])
        self.assertEqual(path.read_text(), "b")
        self.assertEqual(path.stat().st_mode & 0o777, 0o640)

    def test_infrastructure_failure_does_not_become_a_tool_result(self):
        registry = tools.ToolRegistry(self.workspace, skills=[tools.SkillSpec("broken", "test", "broken_module")])
        with patch("harness.tools.import_module", return_value=SimpleNamespace()):
            with self.assertRaises(tools.ToolInfrastructureError):
                registry.dispatch("skill_load", {"skill": "broken"})
        self.assertEqual(registry.active_skill_names, ())

    def test_checkpoint_requires_bounded_nonempty_state(self):
        self.registry.dispatch("skill_load", {"skill": "workspace"})
        self.assertEqual(self.run_workspace("checkpoint", state="phase 2\n7 files pending"), {
            "ok": True, "data": {"checkpoint": {"state": "phase 2\n7 files pending"}},
        })
        for size in (0, 701):
            invalid = self.run_workspace("checkpoint", state="1" * size)
            self.assertEqual(invalid["error"]["code"], "invalid_arguments")

    @unittest.skipUnless(os.name == "posix" and shutil.which("bash"), "requires Bash")
    def test_shell_is_opt_in_and_output_is_bounded(self):
        self.assertEqual(self.registry.dispatch("skill_load", {"skill": "shell"})["error"]["code"],
                         "unknown_skill")
        registry = tools.ToolRegistry(self.workspace, allow_shell=True)
        self.assertTrue(registry.dispatch("skill_load", {"skill": "shell"})["ok"])
        with patch.dict(os.environ, {"LLM_API_KEY": "should-not-be-inherited"}):
            result = registry.dispatch("tool_run", {
                "name": "shell.bash", "arguments": {
                    "command": "printf '%s' \"${LLM_API_KEY-}\"; printf '%10000s' x; exit 7",
                },
            })
        self.assertEqual(result["data"]["exit_code"], 7)
        self.assertTrue(result["data"]["truncated"])
        self.assertEqual(result["data"]["output"], " " * 8192)

    @unittest.skipUnless(os.name == "posix" and shutil.which("bash"), "requires Bash")
    def test_shell_timeout_reports_observed_output(self):
        registry = tools.ToolRegistry(self.workspace, allow_shell=True)
        registry.dispatch("skill_load", {"skill": "shell"})
        result = registry.dispatch("tool_run", {"name": "shell.bash", "arguments": {
            "command": "printf started; sleep 5", "timeout_seconds": 1,
        }})
        self.assertEqual(result["error"]["code"], "shell_timeout")
        self.assertEqual(result["error"]["output"], "started")
