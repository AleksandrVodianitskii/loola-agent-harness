"""Load skills on demand behind a fixed set of model-facing tools."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from copy import deepcopy
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any


JsonObject = dict[str, Any]


def object_schema(properties: JsonObject, *required: str) -> JsonObject:
    return {"type": "object", "properties": properties, "required": list(required),
            "additionalProperties": False}


def _gateway(name: str, description: str, parameters: JsonObject) -> JsonObject:
    return {"type": "function", "function": {
        "name": name, "description": description, "parameters": parameters,
    }}


GATEWAY_TOOLS = [
    _gateway("skill_list", "Discover available skills by name and purpose.", object_schema({})),
    _gateway("skill_load", "Activate a skill or retrieve its instructions and tool schemas again.",
             object_schema({"skill": {"type": "string", "minLength": 1}}, "skill")),
    _gateway("tool_run", "Invoke a tool from an activated skill using its declared arguments.",
             object_schema({"name": {"type": "string", "minLength": 1},
                            "arguments": {"type": "object", "additionalProperties": True}},
                           "name", "arguments")),
]

_GATEWAY_PARAMETERS = {tool["function"]["name"]: tool["function"]["parameters"]
                       for tool in GATEWAY_TOOLS}


@dataclass(frozen=True)
class SkillSpec:
    name: str
    description: str
    module: str


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: JsonObject
    handler: Callable[..., JsonObject]

    def definition(self) -> JsonObject:
        return {"name": self.name, "description": self.description,
                "parameters": deepcopy(self.parameters)}


@dataclass(frozen=True)
class Skill:
    instructions: str
    tools: tuple[ToolSpec, ...]


class ToolError(Exception):
    """An anticipated condition that the model can act on."""

    def __init__(self, code: str, message: str, **details: Any):
        super().__init__(message)
        self.code = code
        self.details = details


class ToolInfrastructureError(RuntimeError):
    """A tool configuration or implementation fault that terminates the run."""


_WORKSPACE = SkillSpec("workspace", "List, read, write, and edit UTF-8 workspace files.",
                       "harness._workspace_skill")
_SHELL = SkillSpec("shell", "Run Bash commands in a trusted workspace.", "harness._shell_skill")
_JSON_TYPES = {"object": dict, "array": list, "string": str, "integer": int,
               "number": (int, float), "boolean": bool, "null": type(None)}


def _validate(value: Any, schema: JsonObject, location: str = "arguments") -> None:
    kind = schema["type"]
    if not isinstance(value, _JSON_TYPES[kind]) or (
        kind in {"integer", "number"} and isinstance(value, bool)
    ):
        raise ToolError("invalid_arguments", f"{location} must be of type {kind}.")
    if "enum" in schema and value not in schema["enum"]:
        raise ToolError("invalid_arguments", f"{location} is outside the allowed values.")
    if kind == "object":
        properties = schema.get("properties", {})
        for key in schema.get("required", ()):
            if key not in value:
                raise ToolError("invalid_arguments", f"Missing required field {location}.{key}.")
        for key, item in value.items():
            if key in properties:
                _validate(item, properties[key], f"{location}.{key}")
            elif schema.get("additionalProperties") is False:
                raise ToolError("invalid_arguments", f"Unexpected field {location}.{key!s:.100s}.")
    if kind == "array" and "items" in schema:
        for index, item in enumerate(value):
            _validate(item, schema["items"], f"{location}[{index}]")
    if kind == "string":
        if len(value) < schema.get("minLength", 0):
            raise ToolError("invalid_arguments", f"{location} is too short.")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            raise ToolError("invalid_arguments", f"{location} is too long.")
        try:
            value.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ToolError("invalid_arguments", f"{location} must be valid Unicode text.") from exc
    if kind in {"integer", "number"}:
        if "minimum" in schema and value < schema["minimum"]:
            raise ToolError("invalid_arguments", f"{location} must be at least {schema['minimum']}.")
        if "maximum" in schema and value > schema["maximum"]:
            raise ToolError("invalid_arguments", f"{location} must be at most {schema['maximum']}.")


class ToolRegistry:
    def __init__(self, workspace: Path, *, allow_shell: bool = False,
                 skills: Iterable[SkillSpec] = ()):
        self.workspace = Path(workspace).resolve(strict=True)
        if not self.workspace.is_dir():
            raise ValueError("workspace must be a directory")
        self._catalog: dict[str, SkillSpec] = {}
        builtins = (_WORKSPACE, _SHELL) if allow_shell else (_WORKSPACE,)
        for skill in (*builtins, *skills):
            if not skill.name or "." in skill.name or skill.name in self._catalog:
                raise ValueError(f"Invalid or duplicate skill name {skill.name!r}")
            if skill.name == _SHELL.name and not allow_shell:
                raise ValueError("The shell skill requires allow_shell=True")
            self._catalog[skill.name] = skill
        self._skills: dict[str, Skill] = {}
        self._tools: dict[str, ToolSpec] = {}

    @property
    def active_skill_names(self) -> tuple[str, ...]:
        return tuple(self._skills)

    def dispatch(self, name: str, arguments: object) -> JsonObject:
        try:
            if name not in _GATEWAY_PARAMETERS:
                raise ToolError("unknown_tool", "Unknown gateway. Use skill_list, skill_load or tool_run.")
            _validate(arguments, _GATEWAY_PARAMETERS[name])
            if name == "skill_list":
                data = {"skills": [{"name": entry.name, "description": entry.description,
                                    "active": entry.name in self._skills}
                                   for entry in self._catalog.values()]}
            elif name == "skill_load":
                data = self._load(arguments["skill"])
            else:
                data = self._run(arguments["name"], arguments["arguments"])
            return {"ok": True, "data": data}
        except ToolError as exc:
            return {"ok": False, "error": {"code": exc.code, "message": str(exc), **exc.details}}
        except Exception as exc:
            raise ToolInfrastructureError(f"Gateway {name} failed ({type(exc).__name__})") from exc

    def _load(self, name: str) -> JsonObject:
        if name not in self._catalog:
            raise ToolError("unknown_skill", "Skill does not exist. Call skill_list to discover skills.")
        if name not in self._skills:
            skill = import_module(self._catalog[name].module).SKILL
            if not isinstance(skill, Skill):
                raise TypeError("Skill module must expose a Skill instance as SKILL")
            tools = {tool.name: tool for tool in skill.tools}
            if len(tools) != len(skill.tools) or any(
                not tool_name.startswith(f"{name}.") or tool_name == f"{name}."
                for tool_name in tools
            ):
                raise ValueError("Skill tools must have unique names in their skill namespace")
            self._tools.update(tools)
            self._skills[name] = skill
        skill = self._skills[name]
        return {"skill": name, "instructions": skill.instructions,
                "tools": [tool.definition() for tool in skill.tools]}

    def _run(self, name: str, arguments: JsonObject) -> JsonObject:
        skill_name, _, _ = name.partition(".")
        if skill_name in self._catalog and skill_name not in self._skills:
            raise ToolError("skill_not_loaded", f"Call skill_load for {skill_name!r} first.")
        tool = self._tools.get(name)
        if tool is None:
            raise ToolError("unknown_tool", "Unknown skill tool. Call skill_load for its current schema.")
        _validate(arguments, tool.parameters)
        data = tool.handler(self.workspace, **arguments)
        if not isinstance(data, dict):
            raise TypeError("Skill handlers must return a JSON object")
        return data
