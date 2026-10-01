"""Public names derived from each tool's own execution declaration."""

from .discovery import discover_tools

TOOL_SCHEMA_ALIASES = dict(discover_tools().aliases)
_PUBLIC_NAMES = {internal: public for public, internal in TOOL_SCHEMA_ALIASES.items()}


def public_tool_name(name):
    return _PUBLIC_NAMES.get(name, name)


def public_tool_text(text):
    for internal, public in _PUBLIC_NAMES.items():
        text = text.replace(internal, public)
    return text
