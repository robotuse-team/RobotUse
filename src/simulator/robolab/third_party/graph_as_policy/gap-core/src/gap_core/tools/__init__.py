"""Flat tool layer — one catalog over every tool source.

The ``@tool`` decorator registers an in-process Python callable as a tool.
Connectors register their ``robot.*`` / ``sim.*`` tools explicitly via
:meth:`ToolRegistry.register_callable` (those prefixes are reserved); skill
bundles register theirs at bundle discovery. Tool names are flat strings —
the catalog the LLM sees carries no dispatch details.

Tools have two scopes:

- ``runtime`` (default): the tool is invocable from a workflow ``type: tool``
  state. The harness shows it to the coordinator so the LLM can pick it.
- ``codegen``: the tool is bound to the codegen LLM via the ``tools=``
  parameter (native tool-use loop). It is NOT in the workflow catalog
  and cannot appear in ``workflow.json``.

Public API:

- :func:`tool` — decorator for Python in-process tools.
- :class:`ToolRegistry` — flat registry spanning all tool sources.
- :class:`ToolDescriptor` — registry entry: name, schema, scope, tags.
"""

from __future__ import annotations

from ._registry import (
    RESERVED_TOOL_PREFIXES,
    Adapter,
    ToolDescriptor,
    ToolRegistry,
    default_tool_registry,
    tool,
)

__all__ = [
    "RESERVED_TOOL_PREFIXES",
    "Adapter",
    "ToolDescriptor",
    "ToolRegistry",
    "default_tool_registry",
    "tool",
]
