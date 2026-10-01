"""Codegen-time registry assembly.

The codegen pipeline needs two registries per run:

- a :class:`gap.skills.SkillsRegistry` built from the configured
  open-robot-skills checkout, and
- a :class:`gap.tools.ToolRegistry` carrying everything the generated
  graphs may dispatch on: the connector-owned ``robot.*`` / ``sim.*``
  tools, the skill bundles' ``@tool`` functions, and the codegen-scope
  meta-tools.

Connector tools are registered by a live connector at *execution* time —
none exists at codegen time. Rather than hand-maintaining a static table
of their signatures, we build a throwaway :class:`SimConnector` around a
null env (its ``__init__`` and tool registration never touch the env) and
copy the resulting descriptors. That keeps the codegen catalog identical
to the real connector's by construction; a parity test pins the
assumption that construction stays env-free.

Bundle ``@tool`` registrations are drained from the decorator's pending
list exactly once per process (modules are cached in ``sys.modules``), so
we archive every drained entry and re-apply the archive to each fresh
registry — re-registration of the same function is idempotent.
"""

from __future__ import annotations

import ast
import logging
import threading
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from gap_core.tools import ToolDescriptor, ToolRegistry
from gap_core.tools import _registry as _tools_registry_module
from gap_core.tools.schema import FieldInfo, UnitSchema

from gap.skills import SkillsRegistry

from ._meta_tools import register_codegen_meta_tools

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()

#: Archive of every bundle @tool registration ever drained in this
#: process, keyed by tool name. Re-applied to each fresh codegen registry.
_TOOL_ARCHIVE: dict[str, dict[str, Any]] = {}

_CONNECTOR_DESCRIPTORS: dict[str, ToolDescriptor] | None = None


def connector_tool_descriptors() -> dict[str, ToolDescriptor]:
    """Descriptors (name → schema/summary/tags) for the connector-owned
    ``robot.*`` / ``sim.*`` tools, derived from the real connector code.

    The returned descriptors are schema-only for codegen purposes — they
    are NOT dispatchable (the throwaway connector has no env).
    """
    global _CONNECTOR_DESCRIPTORS
    with _LOCK:
        if _CONNECTOR_DESCRIPTORS is None:
            from gap.connector.sim import SimConnector

            conn = SimConnector(None, SimpleNamespace())
            _CONNECTOR_DESCRIPTORS = dict(conn.tool_registry._tools)
        return dict(_CONNECTOR_DESCRIPTORS)


def _drain_pending_with_archive(registry: ToolRegistry) -> None:
    """Drain pending ``@tool`` registrations into *registry*, archiving
    them so later registries in the same process see them too."""
    with _LOCK:
        for entry in _tools_registry_module._PENDING_TOOLS:
            _TOOL_ARCHIVE[entry["name"]] = dict(entry)
        registry.discover_pending()
        for entry in _TOOL_ARCHIVE.values():
            try:
                registry._register_python(
                    entry["name"], entry["summary"],
                    entry["scope"], entry["tags"], entry["fn"],
                )
            except ValueError:
                # A different implementation already owns this name in
                # this registry — first registration wins.
                logger.debug("tool %r already registered; keeping existing", entry["name"])


def build_codegen_tool_registry(
    skills_registry: SkillsRegistry | None = None,
) -> ToolRegistry:
    """Build the flat tool catalog the codegen prompts render from.

    Contains: connector tool descriptors (schema-only), every bundle
    ``@tool`` (imported by *skills_registry* discovery), bundles whose
    serving.protocol is ``stdio-msgpack`` (those declare tools in SKILL.md
    ``gap.tools`` — the @tool decorators only fire inside the bundle's
    own venv when ``gap_tool_server`` boots), and the codegen-scope
    meta-tools.
    """
    registry = ToolRegistry()
    for name, descriptor in connector_tool_descriptors().items():
        registry._tools.setdefault(name, descriptor)
    _drain_pending_with_archive(registry)
    if skills_registry is not None:
        _register_rpc_bundle_tools(registry, skills_registry)
    register_codegen_meta_tools(registry)
    return registry


def _register_rpc_bundle_tools(
    registry: ToolRegistry, skills_registry: SkillsRegistry,
) -> None:
    """Register schema stubs for tools whose @tool definitions live in an
    out-of-process bundle venv (``serving.protocol == 'stdio-msgpack'``).

    The codegen LLM needs to KNOW these tools exist, their summaries, AND
    their real signatures: with name+summary-only stubs the generator
    guessed parameter names (``question`` vs ``prompt``, dropped
    ``world_config``, mis-wired joint states) and the graphs failed at
    runtime. Booting each bundle venv at codegen time is too heavy, so
    the signatures are lifted **statically** from the bundle's Python
    source via :func:`_ast_bundle_schemas` — no imports, so it works from
    the engine venv regardless of the bundle's deps. Dispatch still
    happens through the runtime registry's RpcAdapter once
    :class:`ToolBundleManager` boots the bundle at workflow execution
    time.
    """
    for info in skills_registry.list_skills():
        serving = getattr(info.meta, "serving", None)
        if serving is None or getattr(serving, "protocol", None) != "stdio-msgpack":
            continue
        ast_schemas = _ast_bundle_schemas(info.bundle_dir)
        for tool_name, summary in (info.meta.tools or {}).items():
            if tool_name in registry:
                continue
            try:
                registry.register_rpc(
                    tool_name, client=None, summary=summary,
                    schema=ast_schemas.get(tool_name),
                )
            except ValueError:
                # Tool already registered elsewhere (e.g. drained @tool
                # left over in the archive); first registration wins.
                logger.debug("rpc tool %r already registered; keeping existing",
                             tool_name)


# ---------------------------------------------------------------------------
# Static (AST) signature extraction for out-of-process bundles
# ---------------------------------------------------------------------------

#: Cache: bundle dir -> {tool name -> UnitSchema}. Source files are static
#: for the life of a codegen process.
_AST_SCHEMA_CACHE: dict[Path, dict[str, UnitSchema]] = {}


def _ast_bundle_schemas(bundle_dir: Path) -> dict[str, UnitSchema]:
    """Extract ``{tool name -> UnitSchema}`` from a bundle's source, without
    importing it.

    Parses every top-level ``*.py`` in *bundle_dir* (conventionally the
    ``@tool`` functions live in ``tools.py``), finds functions decorated
    with ``@tool(name=..., summary=...)``, and lifts parameter names /
    annotation strings / defaults. A return annotation naming a TypedDict
    defined in the same file is expanded into per-field outputs — the
    same shape the bundle server reports from its own venv at runtime.
    """
    bundle_dir = Path(bundle_dir)
    with _LOCK:
        cached = _AST_SCHEMA_CACHE.get(bundle_dir)
        if cached is not None:
            return cached

    schemas: dict[str, UnitSchema] = {}
    for py in sorted(bundle_dir.glob("*.py")):
        try:
            tree = ast.parse(py.read_text(), filename=str(py))
        except (OSError, SyntaxError):
            logger.debug("AST parse failed for %s", py, exc_info=True)
            continue
        typed_dicts = _module_typed_dicts(tree)
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            decorated = _tool_decorator_kwargs(node)
            if decorated is None:
                continue
            name = decorated.get("name")
            if not isinstance(name, str):
                continue
            summary = decorated.get("summary")
            schemas[name] = _function_schema(
                node, name,
                summary if isinstance(summary, str) else "",
                typed_dicts,
            )

    with _LOCK:
        _AST_SCHEMA_CACHE[bundle_dir] = schemas
    return schemas


def _tool_decorator_kwargs(
    fn: ast.FunctionDef | ast.AsyncFunctionDef,
) -> dict[str, Any] | None:
    """The keyword literals of an ``@tool(...)`` decorator, or None."""
    for dec in fn.decorator_list:
        if not isinstance(dec, ast.Call):
            continue
        target = dec.func
        dotted = (
            target.attr if isinstance(target, ast.Attribute)
            else target.id if isinstance(target, ast.Name)
            else ""
        )
        if dotted != "tool":
            continue
        kwargs: dict[str, Any] = {}
        for kw in dec.keywords:
            if kw.arg is not None and isinstance(kw.value, ast.Constant):
                kwargs[kw.arg] = kw.value.value
        return kwargs
    return None


def _module_typed_dicts(tree: ast.Module) -> dict[str, dict[str, str]]:
    """``{class name -> {field -> type_str}}`` for module-level TypedDicts."""
    out: dict[str, dict[str, str]] = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        base_names = {
            b.attr if isinstance(b, ast.Attribute)
            else b.id if isinstance(b, ast.Name) else ""
            for b in node.bases
        }
        if "TypedDict" not in base_names and not (base_names & set(out)):
            continue
        fields: dict[str, str] = {}
        for base in node.bases:
            bname = base.attr if isinstance(base, ast.Attribute) else (
                base.id if isinstance(base, ast.Name) else "")
            fields.update(out.get(bname, {}))
        for stmt in node.body:
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                fields[stmt.target.id] = ast.unparse(stmt.annotation)
        out[node.name] = fields
    return out


def _function_schema(
    fn: ast.FunctionDef | ast.AsyncFunctionDef,
    tool_name: str,
    summary: str,
    typed_dicts: dict[str, dict[str, str]],
) -> UnitSchema:
    """Build a UnitSchema from one decorated function's AST."""
    inputs: dict[str, FieldInfo] = {}

    def _add(arg: ast.arg, default: ast.expr | None) -> None:
        if arg.arg in ("self", "ctx"):
            return
        type_str = ast.unparse(arg.annotation) if arg.annotation else "Any"
        has_default = default is not None
        default_value: Any = None
        if has_default and isinstance(default, ast.Constant):
            default_value = default.value
        elif has_default:
            default_value = ast.unparse(default)
        inputs[arg.arg] = FieldInfo(
            name=arg.arg, python_type=None, type_str=type_str,
            required=not has_default, default=default_value,
        )

    args = fn.args
    pos = args.posonlyargs + args.args
    pos_defaults: list[ast.expr | None] = [None] * (
        len(pos) - len(args.defaults)
    ) + list(args.defaults)
    for arg, default in zip(pos, pos_defaults, strict=False):
        _add(arg, default)
    for arg, default in zip(args.kwonlyargs, args.kw_defaults, strict=False):
        _add(arg, default)

    outputs: dict[str, FieldInfo] = {}
    ret = fn.returns
    ret_name = (
        ret.id if isinstance(ret, ast.Name)
        else ret.attr if isinstance(ret, ast.Attribute)
        else None
    )
    if ret_name and ret_name in typed_dicts:
        for fname, ftype in typed_dicts[ret_name].items():
            outputs[fname] = FieldInfo(
                name=fname, python_type=None, type_str=ftype, required=True,
            )
    elif ret is not None and not (
        isinstance(ret, ast.Constant) and ret.value is None
    ):
        outputs["result"] = FieldInfo(
            name="result", python_type=None,
            type_str=ast.unparse(ret), required=True,
        )

    doc = ast.get_docstring(fn) or ""
    description = summary or (doc.splitlines()[0] if doc else "")
    return UnitSchema(
        name=tool_name, description=description,
        inputs=inputs, outputs=outputs,
    )


def load_codegen_registries(
    skills_path: str | Path | Sequence[str | Path],
    *,
    only: list[str] | None = None,
    disable: list[str] | None = None,
) -> tuple[SkillsRegistry, ToolRegistry]:
    """Load the skill registry root(s) and build the matching tool catalog."""
    from gap.skills import load_registry_set

    skills = load_registry_set(skills_path, only=only, disable=disable)
    tools = build_codegen_tool_registry(skills)
    return skills, tools
