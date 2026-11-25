"""Opt-in Bash execution for a trusted POSIX workspace."""

from __future__ import annotations

from pathlib import Path
from selectors import DefaultSelector, EVENT_READ
import os
import signal
import subprocess
import time

from .tools import Skill, ToolError, ToolSpec, object_schema


MAX_OUTPUT_BYTES = 8 * 1024


def _kill_group(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def bash(workspace: Path, command: str, timeout_seconds: int = 30) -> dict:
    environment = {name: os.environ[name] for name in ("PATH", "LANG", "LC_ALL", "LC_CTYPE",
                                                       "TMPDIR", "TZ") if name in os.environ}
    try:
        if "\x00" in command or len(command.encode("utf-8")) > 65536:
            raise ValueError("Unsupported command argument")
    except ValueError as exc:
        raise ToolError("invalid_arguments", "Bash expects up to 65,536 UTF-8 bytes without NUL characters.") from exc
    deadline = time.monotonic() + timeout_seconds
    captured = bytearray()
    total = 0
    with subprocess.Popen(["bash", "-c", command], cwd=workspace, env=environment,
                          stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, start_new_session=True) as process:
        try:
            if process.stdout is None:
                raise RuntimeError("Bash output pipe was not created")
            with DefaultSelector() as selector:
                selector.register(process.stdout, EVENT_READ)
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or not selector.select(remaining):
                        raise subprocess.TimeoutExpired(command, timeout_seconds)
                    chunk = os.read(process.stdout.fileno(), 64 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    captured.extend(chunk[:MAX_OUTPUT_BYTES - len(captured)])
            exit_code = process.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired as exc:
            _kill_group(process)
            raise ToolError("shell_timeout", f"Bash exceeded {timeout_seconds} seconds. Inspect side effects before retrying.",
                            output=captured.decode("utf-8", errors="replace"),
                            truncated=total > len(captured)) from exc
        except BaseException:
            _kill_group(process)
            raise
    return {"exit_code": exit_code, "output": captured.decode("utf-8", errors="replace"),
            "truncated": total > len(captured)}


SKILL = Skill(
    "Run Bash from the workspace root. Commands can change state. Check exit_code and inspect "
    "the workspace after a timeout before retrying.",
    (ToolSpec("shell.bash", "Execute Bash and return the exit code and bounded combined output.",
              object_schema({"command": {"type": "string", "minLength": 1, "maxLength": 65536},
                             "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 120}},
                            "command"), bash),),
)
