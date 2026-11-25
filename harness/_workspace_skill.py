"""Text operations for the workspace skill."""

from __future__ import annotations

from contextlib import contextmanager
import errno
from heapq import nsmallest
from pathlib import Path
import stat
import tempfile

from .tools import Skill, ToolError, ToolSpec, object_schema


MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_TOOL_READ_BYTES = 8 * 1024
PATH_PARAM = {"type": "string", "minLength": 1, "maxLength": 4096}
TEXT_PARAM = {"type": "string", "maxLength": MAX_FILE_BYTES}


def _path(workspace: Path, path: str) -> Path:
    if "\x00" in path:
        raise ToolError("invalid_arguments", "Paths cannot contain a NUL character.")
    relative = Path(path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ToolError("path_outside_workspace", "Use a path relative to the workspace.")
    try:
        destination = (workspace / relative).resolve()
    except RuntimeError as exc:
        raise ToolError("unsupported_file", "The path contains a symlink loop.") from exc
    if not destination.is_relative_to(workspace):
        raise ToolError("path_outside_workspace", "This path resolves outside the workspace.")
    return destination


@contextmanager
def _file_errors():
    try:
        yield
    except (FileNotFoundError, NotADirectoryError) as exc:
        raise ToolError("file_not_found", "The file or directory does not exist.") from exc
    except IsADirectoryError as exc:
        raise ToolError("unsupported_file", "The operation requires a regular file.") from exc
    except FileExistsError as exc:
        raise ToolError("unsupported_file", "A file occupies a directory needed by the operation.") from exc
    except PermissionError as exc:
        raise ToolError("file_permission", "Access to the requested file or directory is denied.") from exc
    except OSError as exc:
        if exc.errno == errno.ENAMETOOLONG:
            raise ToolError("invalid_arguments", "The path is too long for the filesystem.") from exc
        if exc.errno == errno.ELOOP:
            raise ToolError("unsupported_file", "The path contains a symlink loop.") from exc
        raise


def _read(path: Path) -> str:
    if not stat.S_ISREG(path.stat().st_mode):
        raise ToolError("unsupported_file", "The operation requires a regular file.")
    with path.open("rb") as stream:
        data = stream.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES:
        raise ToolError("read_limit", f"Text files may contain up to {MAX_FILE_BYTES} UTF-8 bytes.")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ToolError("unsupported_file", "The file is not valid UTF-8.") from exc


def _write(path: Path, text: str) -> int:
    try:
        data = text.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ToolError("invalid_arguments", "Text must be valid UTF-8.") from exc
    if len(data) > MAX_FILE_BYTES:
        raise ToolError("write_limit", f"Text files may contain up to {MAX_FILE_BYTES} UTF-8 bytes.")
    mode = path.stat().st_mode if path.exists() else None
    if mode is not None and not stat.S_ISREG(mode):
        raise ToolError("unsupported_file", "The operation requires a regular file.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary_path = Path(stream.name)
            stream.write(data)
        if mode is not None:
            temporary_path.chmod(stat.S_IMODE(mode))
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return len(data)


def list_files(workspace: Path, path: str = ".", after: str = "", limit: int = 100) -> dict:
    with _file_errors():
        directory = _path(workspace, path)
        entries = nsmallest(limit + 1, (entry for entry in directory.iterdir() if entry.name > after),
                            key=lambda entry: entry.name)
        return {"path": directory.relative_to(workspace).as_posix(),
                "entries": [{"name": entry.name,
                             "type": "symlink" if entry.is_symlink()
                             else "directory" if entry.is_dir() else "file"}
                            for entry in entries[:limit]],
                "next_after": entries[limit - 1].name if len(entries) > limit else None}


def read_text(workspace: Path, path: str, start_line: int = 1, max_lines: int = 200,
              start_column: int = 1) -> dict:
    with _file_errors():
        source = _path(workspace, path)
        lines = _read(source).splitlines(keepends=True)
    if start_column > 1 and (start_line > len(lines) or start_column > len(lines[start_line - 1])):
        raise ToolError("invalid_arguments", "start_column is outside the selected line.")

    chunks = []
    available = MAX_TOOL_READ_BYTES
    stop = min(len(lines), start_line + max_lines - 1)
    next_line = stop + 1 if stop < len(lines) else None
    next_column = 1 if next_line is not None else None
    for index in range(start_line - 1, stop):
        column = start_column if index == start_line - 1 else 1
        value = lines[index][column - 1:].encode("utf-8")
        if len(value) > available:
            fragment = value[:available].decode("utf-8", errors="ignore")
            chunks.append(fragment)
            next_line = index + 1
            next_column = column + len(fragment)
            break
        chunks.append(value.decode("utf-8"))
        available -= len(value)
    return {"path": source.relative_to(workspace).as_posix(), "text": "".join(chunks),
            "start_line": start_line, "start_column": start_column,
            "next_line": next_line, "next_column": next_column}


def write_text(workspace: Path, path: str, text: str) -> dict:
    with _file_errors():
        destination = _path(workspace, path)
        written = _write(destination, text)
    return {"path": destination.relative_to(workspace).as_posix(), "bytes_written": written}


def replace_text(workspace: Path, path: str, old: str, new: str) -> dict:
    with _file_errors():
        destination = _path(workspace, path)
        before = _read(destination)
        index = before.find(old)
        if index == -1 or before.find(old, index + 1) != -1:
            raise ToolError("edit_mismatch", "Expected one exact match. Read the file again and narrow the edit.")
        written = _write(destination, before.replace(old, new, 1))
    return {"path": destination.relative_to(workspace).as_posix(), "bytes_written": written}


def checkpoint(_workspace: Path, state: str) -> dict:
    return {"checkpoint": {"state": state}}


SKILL = Skill(
    "Use workspace-relative paths. Read an existing file before editing. Follow next_line and "
    "next_column when reading a long file. Use checkpoint after completing a phase to save concise "
    "evidence, key decisions, and remaining work. Each checkpoint replaces the previous one. "
    "Reopen a skill to recover its tool schemas after compaction.",
    (
        ToolSpec("workspace.list_files", "List one directory with optional paging by entry name.",
                 object_schema({"path": PATH_PARAM, "after": PATH_PARAM,
                                "limit": {"type": "integer", "minimum": 1, "maximum": 200}}), list_files),
        ToolSpec("workspace.read_text", "Read UTF-8 text. Resume with the returned line and column.",
                 object_schema({"path": PATH_PARAM,
                                "start_line": {"type": "integer", "minimum": 1},
                                "start_column": {"type": "integer", "minimum": 1},
                                "max_lines": {"type": "integer", "minimum": 1, "maximum": 400}}, "path"),
                 read_text),
        ToolSpec("workspace.write_text", "Create or atomically replace a UTF-8 file with this text.",
                 object_schema({"path": PATH_PARAM, "text": TEXT_PARAM}, "path", "text"), write_text),
        ToolSpec("workspace.replace_text", "Replace one exact occurrence of old in a UTF-8 file.",
                 object_schema({"path": PATH_PARAM, "old": {**TEXT_PARAM, "minLength": 1},
                                "new": TEXT_PARAM}, "path", "old", "new"), replace_text),
        ToolSpec("workspace.checkpoint", "Save a short progress note for context compaction.",
                 object_schema({"state": {"type": "string", "minLength": 1, "maxLength": 700}}, "state"),
                 checkpoint),
    ),
)
