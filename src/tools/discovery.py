"""Discover every direct tool package, rejecting silently unregistered folders."""

from __future__ import annotations

import importlib
from pathlib import Path

from .base_tool import ToolRegistrationError, ToolSpec
from .registry import ToolRegistry


def discover_tools(package: str = "src.tools") -> ToolRegistry:
    """Load each direct package's ``tool.py`` in deterministic folder order.

    Only entrypoints are imported. Nested third-party checkouts, adapters and
    workers are never traversed. No central list of tool package names exists.
    """
    importlib.invalidate_caches()
    module = importlib.import_module(package)
    paths = getattr(module, "__path__", None)
    if paths is None:
        raise ToolRegistrationError(f"{package} is not a tool package")
    directories: dict[str, Path] = {}
    for root in paths:
        for child in Path(root).iterdir():
            if (not child.is_dir() or child.name in ("__pycache__", "third_party")
                    or child.name.startswith(".")):
                continue
            if child.name in directories:
                raise ToolRegistrationError(f"duplicate tool package: {child.name}")
            directories[child.name] = child
    definitions = []
    for name, directory in sorted(directories.items()):
        if not name.isidentifier():
            raise ToolRegistrationError(f"invalid tool package name: {name}")
        if not (directory / "__init__.py").is_file():
            raise ToolRegistrationError(f"{directory}: missing __init__.py")
        if not (directory / "tool.py").is_file():
            raise ToolRegistrationError(f"{directory}: missing tool.py registration")
        entrypoint = f"{package}.{name}.tool"
        try:
            entry = importlib.import_module(entrypoint)
        except Exception as exc:
            raise ToolRegistrationError(f"{entrypoint}: tool entrypoint failed to import") from exc
        tools = getattr(entry, "TOOLS", None)
        if not isinstance(tools, tuple) or not tools:
            raise ToolRegistrationError(f"{entrypoint}: TOOLS must be a nonempty tuple")
        registered = {id(spec) for spec in tools}
        omitted = [binding for binding, value in vars(entry).items()
                   if isinstance(value, ToolSpec) and id(value) not in registered]
        if omitted:
            raise ToolRegistrationError(
                f"{entrypoint}: ToolSpec declarations omitted from TOOLS: {', '.join(sorted(omitted))}"
            )
        definitions.extend(tools)
    return ToolRegistry(definitions)
