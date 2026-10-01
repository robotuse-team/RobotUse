"""Read the discovered tool catalog without manually maintaining tool names."""

from .base_tool import ToolSpec
from .discovery import discover_tools
from .registry import ToolRegistry


def list_tools(registry: ToolRegistry | None = None) -> tuple[ToolSpec, ...]:
    """Return public definitions, including marked internal workflow tools."""
    registry = discover_tools() if registry is None else registry
    return tuple(registry.tools.values())
