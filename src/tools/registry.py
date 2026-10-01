"""Validated immutable lookup, independent of agent and simulator imports."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from types import MappingProxyType
from typing import Any

from .base_tool import ToolExecutionContext, ToolRegistrationError, ToolSpec


class ToolRegistry:
    """Reject incomplete registrations before a session starts.

    Internal aliases resolve to the same definition as public names, so the
    existing workflow's private execution calls retain their original path.
    """

    def __init__(self, definitions: Iterable[ToolSpec]) -> None:
        by_name: dict[str, ToolSpec] = {}
        lookup: dict[str, ToolSpec] = {}
        aliases: dict[str, str] = {}
        for spec in definitions:
            if not isinstance(spec, ToolSpec):
                raise ToolRegistrationError("every registered tool must be a ToolSpec")
            spec.validate()
            names = tuple(dict.fromkeys((spec.name, spec.dispatch_name)))
            for name in names:
                if name in lookup:
                    raise ToolRegistrationError(f"duplicate tool name or alias: {name}")
            by_name[spec.name] = spec
            for name in names:
                lookup[name] = spec
            if spec.dispatch_name != spec.name:
                aliases[spec.name] = spec.dispatch_name
        if not by_name:
            raise ToolRegistrationError("tool registry cannot be empty")
        self._tools = MappingProxyType(by_name)
        self._lookup = MappingProxyType(lookup)
        self._aliases = MappingProxyType(aliases)

    @property
    def tools(self) -> Mapping[str, ToolSpec]:
        return self._tools

    @property
    def aliases(self) -> Mapping[str, str]:
        return self._aliases

    def __getitem__(self, name: str) -> ToolSpec:
        try:
            return self._lookup[name]
        except KeyError:
            raise ToolRegistrationError(f"unregistered tool: {name}") from None

    def __contains__(self, name: str) -> bool:
        return name in self._lookup

    def __iter__(self) -> Iterator[str]:
        return iter(self._tools)

    def __len__(self) -> int:
        return len(self._tools)

    def require(self, names: Iterable[str]) -> tuple[str, ...]:
        """Validate membership while preserving caller-owned role/task order."""
        result = tuple(names)
        for name in result:
            self[name]
        return result

    def required_arguments(self, name: str) -> tuple[str, ...]:
        return self[name].required_arguments

    def optional_arguments(self, name: str, *, context=None) -> tuple[str, ...]:
        schema = self[name].schema(context=context)
        return tuple(key for key in schema["properties"] if key not in schema["required"])

    def function_schema(self, name: str, *, context=None, role=None, tools=()):
        return self[name].function_schema(name=name, context=context, role=role, tools=tools)

    def validate_argument(self, name, key, value, *, context=None):
        return self[name].validate_argument(key, value, context=context)

    def dispatch(
        self, name: str, arguments: Mapping[str, Any], *, context: ToolExecutionContext,
    ) -> Any:
        """Execute through the existing boundary; role checks stay at its caller."""
        spec = self[name]
        return spec.handler(context, spec.dispatch_name, arguments)
