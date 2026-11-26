"""Write optional benchmark transcripts without provider request metadata."""

from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from typing import TextIO


def open_trace(path: Path | None, output: Path | None):
    if path is None:
        return nullcontext()
    if output is not None and path.resolve() == output.resolve():
        raise ValueError("--trace and --output must be different files")
    return path.open("x", encoding="utf-8", buffering=1)


def write_event(stream: TextIO, case: str, event: str, data: dict) -> None:
    if event == "assistant":
        data = {key: data[key] for key in ("role", "content", "tool_calls") if key in data}
    stream.write(json.dumps({"case": case, "event": event, "data": data}, ensure_ascii=True) + "\n")
