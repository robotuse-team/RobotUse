"""Small, runtime-independent contracts for registered tool adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy
import inspect
import re
from typing import Any, Callable, Mapping

from .schema import object_schema, validate_schema, validate_value


class ToolRegistrationError(ValueError):
    """An installed tool package has an incomplete or ambiguous definition."""


DispatchCallback = Callable[[str, Mapping[str, Any]], Any]


@dataclass(frozen=True)
class ToolExecutionContext:
    """Delegate execution to the existing boundary, retaining its state checks."""

    dispatch: DispatchCallback

    def __post_init__(self) -> None:
        if not callable(self.dispatch):
            raise TypeError("tool execution requires a dispatch callback")


ToolHandler = Callable[[ToolExecutionContext, str, Mapping[str, Any]], Any]
_NAME = re.compile(r"[a-z][a-z0-9_]*\Z")
_ARGUMENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


@dataclass(frozen=True)
class ToolSpec:
    """One public contract and its existing execution name.

    ``input_schema`` owns every required and optional field. It may be a
    factory for role/configuration-dependent contracts such as ``finish``.
    The provider and agent boundary both consume this declaration.
    ``visibility`` records whether a tool is used internally by a workflow.
    Role and task permissions remain with the agent's existing tool inventory.
    """

    name: str
    required_arguments: tuple[str, ...]
    handler: ToolHandler
    internal_name: str | None = None
    kind: str = "tool"
    visibility: str = "agent"
    input_schema: Mapping[str, Any] | Callable[..., Mapping[str, Any]] | None = None
    description: str | Callable[..., str] | None = None
    argument_validators: Mapping[str, Callable[[Any, Mapping[str, Any]], Any]] = field(default_factory=dict)

    @property
    def dispatch_name(self) -> str:
        return self.internal_name or self.name

    def schema(self, *, context=None, role=None, tools=()) -> dict[str, Any]:
        declared = self.input_schema
        if callable(declared):
            declared = declared(context=context, role=role, tools=tools)
        if declared is None:
            if self.required_arguments:
                raise ToolRegistrationError(f"{self.name}: missing input_schema")
            declared = object_schema({})
        return deepcopy(dict(declared))

    def function_schema(self, *, context=None, role=None, tools=(), name=None):
        description = self.description
        if callable(description):
            description = description(context=context, role=role, tools=tools)
        public_name = name or self.name
        return {"type": "function", "function": {
            "name": public_name,
            "description": description or f"Typed {public_name}; use opaque current-session references only.",
            "parameters": self.schema(context=context, role=role, tools=tools),
        }}

    def validate_argument(self, key, value, *, context=None):
        properties = self.schema(context=context)["properties"]
        if key not in properties:
            from src.core.contracts import BoundaryError
            raise BoundaryError(f"{self.name}: undeclared argument {key}")
        validator = self.argument_validators.get(key)
        return validator(value, properties[key]) if validator else validate_value(properties[key], value, key)

    def validate(self) -> None:
        for label, value in (("name", self.name), ("internal_name", self.internal_name)):
            if value is None and label == "internal_name":
                continue
            if not isinstance(value, str) or not _NAME.fullmatch(value):
                raise ToolRegistrationError(f"invalid tool {label}: {value!r}")
        if not isinstance(self.required_arguments, tuple):
            raise ToolRegistrationError(f"{self.name}: required_arguments must be a tuple")
        if any(not isinstance(key, str) or not _ARGUMENT.fullmatch(key)
               for key in self.required_arguments):
            raise ToolRegistrationError(f"{self.name}: invalid required argument name")
        if len(set(self.required_arguments)) != len(self.required_arguments):
            raise ToolRegistrationError(f"{self.name}: duplicate required argument")
        if self.kind not in ("tool", "control", "backend"):
            raise ToolRegistrationError(f"{self.name}: invalid tool kind {self.kind!r}")
        if self.visibility not in ("agent", "internal"):
            raise ToolRegistrationError(f"{self.name}: invalid visibility {self.visibility!r}")
        if self.kind == "backend" and self.visibility != "internal":
            raise ToolRegistrationError(f"{self.name}: backend tools must remain internal")
        if not callable(self.handler):
            raise ToolRegistrationError(f"{self.name}: handler must be callable")
        try:
            inspect.signature(self.handler).bind(None, self.dispatch_name, {})
        except (TypeError, ValueError) as exc:
            raise ToolRegistrationError(
                f"{self.name}: handler must accept (context, name, arguments)"
            ) from exc
        try:
            schema = self.schema()
            validate_schema(schema, self.name)
            if schema.get("type") != "object":
                raise ValueError(f"{self.name}: input_schema must describe an object")
            if any(not isinstance(key, str) or not _ARGUMENT.fullmatch(key)
                   for key in schema["properties"]):
                raise ValueError(f"{self.name}: invalid input property name")
            if self.kind != "control" and tuple(schema["required"]) != self.required_arguments:
                raise ValueError(f"{self.name}: input_schema required fields must match required_arguments")
            for key, validator in self.argument_validators.items():
                if key not in schema["properties"] or not callable(validator):
                    raise ValueError(f"{self.name}: invalid argument validator for {key}")
                inspect.signature(validator).bind(None, {})
        except (TypeError, ValueError) as exc:
            raise ToolRegistrationError(str(exc)) from exc


def dispatch_existing(
    context: ToolExecutionContext, name: str, arguments: Mapping[str, Any],
) -> Any:
    """Call the supplied boundary without changing arguments or error handling."""
    return context.dispatch(name, arguments)


def define_tool(name, properties, *, required=None, handler=dispatch_existing, **options):
    """Declare a tool once, deriving its ordered argument names from its schema."""
    schema = object_schema(properties, required)
    return ToolSpec(name, tuple(schema["required"]), handler, input_schema=schema, **options)
