"""Unified I/O schema extraction for tools, skills, and generated scripts.

``@tool``-decorated callables, atomic skills (with ``skill.py:run``), and
composite-bundle scripts (under ``scripts/<name>.py``) follow the same
``run(ctx, ...) -> Output`` contract. :func:`extract_schema` introspects any
such module and produces a :class:`UnitSchema` describing inputs and outputs.
This is the single schema extractor shared by the tool registry, the skills
loader, and the graph validator.
"""

from __future__ import annotations

import inspect
import typing
from dataclasses import dataclass, field
from typing import Any


@dataclass
class FieldInfo:
    """Schema for a single input or output field."""

    name: str
    python_type: Any
    type_str: str
    required: bool
    default: Any = None
    description: str = ""


@dataclass
class UnitSchema:
    """Unified I/O schema for any unit (tool, skill, or LLM-generated script)."""

    name: str
    description: str = ""
    inputs: dict[str, FieldInfo] = field(default_factory=dict)
    outputs: dict[str, FieldInfo] = field(default_factory=dict)


def _type_to_str(hint: Any) -> str:
    """Convert a Python type hint to a human-readable string for docs."""
    if hint is type(None):
        return "None"

    origin = typing.get_origin(hint)

    if origin is list:
        args = typing.get_args(hint)
        if args:
            return f"list[{_type_to_str(args[0])}]"
        return "list"

    import types as _types

    if origin is _types.UnionType or origin is typing.Union:
        args = typing.get_args(hint)
        parts = [_type_to_str(a) for a in args]
        return " | ".join(parts)

    if origin is dict:
        args = typing.get_args(hint)
        if args and len(args) == 2:
            return f"dict[{_type_to_str(args[0])}, {_type_to_str(args[1])}]"
        return "dict"

    if isinstance(hint, type):
        return hint.__name__

    if hint is Any:
        return "Any"

    return str(hint)


def extract_schema(module: Any, meta: Any | None = None) -> UnitSchema:
    """Extract I/O schema from a module with a ``run()`` function.

    Works identically for atomic skills, composite scripts, and Python
    tool plugins.

    Args:
        module: A Python module with a ``run(ctx, ...) -> Output`` function.
            Tools are decorator-backed callables; pass the function directly
            wrapped in a SimpleNamespace if needed.
        meta: Optional metadata object with ``description`` / ``params`` /
            ``outputs`` attributes (the skills loader passes its
            ``SkillMeta``). If ``None``, looks for ``module._meta``, falling
            back to the module docstring for the description.

    Returns:
        A :class:`UnitSchema` with input and output field info.
    """
    run_fn = getattr(module, "run", None)
    if run_fn is None or not callable(run_fn):
        raise ValueError(f"Module {getattr(module, '__name__', module)} has no callable run()")

    sig = inspect.signature(run_fn)
    try:
        hints = typing.get_type_hints(run_fn, globalns=vars(module) if hasattr(module, "__dict__") else None)
    except Exception as e:
        raise ValueError(f"Cannot resolve type hints on run(): {e}") from e

    if meta is None:
        meta = getattr(module, "_meta", None)

    description = getattr(meta, "description", None)
    if description is None:
        description = getattr(module, "__doc__", "") or ""
    params_meta = getattr(meta, "params", None) or {}
    outputs_meta = getattr(meta, "outputs", None) or {}

    inputs: dict[str, FieldInfo] = {}
    for name, param in sig.parameters.items():
        if name in ("ctx", "self"):
            continue
        hint = hints.get(name, Any)
        has_default = param.default is not inspect.Parameter.empty
        default = param.default if has_default else None

        field_description = ""
        if name in params_meta:
            field_description = params_meta[name].description

        inputs[name] = FieldInfo(
            name=name,
            python_type=hint,
            type_str=_type_to_str(hint),
            required=not has_default,
            default=default,
            description=field_description,
        )

    outputs: dict[str, FieldInfo] = {}
    return_hint = hints.get("return")
    if return_hint is not None and return_hint is not type(None):
        if hasattr(return_hint, "__annotations__"):
            try:
                output_hints = typing.get_type_hints(return_hint)
            except Exception:
                output_hints = return_hint.__annotations__
            for fname, ftype in output_hints.items():
                field_description = ""
                if fname in outputs_meta:
                    field_description = outputs_meta[fname]

                outputs[fname] = FieldInfo(
                    name=fname,
                    python_type=ftype,
                    type_str=_type_to_str(ftype),
                    required=True,
                    description=field_description,
                )

    return UnitSchema(
        name=getattr(module, "__name__", "unknown"),
        description=description,
        inputs=inputs,
        outputs=outputs,
    )
