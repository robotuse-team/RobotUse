"""Dependency-free data models and strict validation primitives.

The models deliberately describe high-level choices only.  They cannot carry
joint targets, trajectories, source code, or arbitrary action payloads.
"""


from __future__ import annotations


from typing import Any, Mapping, TypeAlias


JsonScalar: TypeAlias = str | int | float | bool | None


JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


class ValidationError(ValueError):
    """Raised when untrusted data violates the fixed selection contract."""
