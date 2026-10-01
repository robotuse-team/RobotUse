"""Python authoring library for v3 workflow JSON (LangGraph-style).

This module wraps the v3 schema defined in :mod:`gap.runtime.workflow` with
two peer authoring classes:

- :class:`Subgraph` — author one self-contained subgraph standalone; what a
  subagent typically returns.
- :class:`Workflow` — author the top-level coordinator workflow; imports
  Subgraph objects via :meth:`Workflow.add_subgraph`.

The runtime is not modified. ``save()`` calls the existing
``gap.runtime.workflow._parse_workflow`` for syntax validation and
``gap.runtime.validate.validate_workflow`` for structural rules. The runtime
is imported lazily inside those methods so the authoring API stays importable
on its own.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from gap_core.errors import (
    GraphValidationError,
    ValidationIssue,
    WorkflowValidationError,
)

if TYPE_CHECKING:  # pragma: no cover
    from gap.runtime.workflow import ToolCall

# Mirror gap.runtime.workflow.START / END. Defined here as literals so the
# builder imports without the runtime; the runtime parser asserts the same
# values.
START: str = "START"
END: str = "END"

_VALID_NODE_TYPES = frozenset({
    "tool", "script", "router", "subgraph", "end", "noop",
})


def Ref(path: str) -> dict[str, str]:
    """Build a JSON $ref dict.

    ``Ref("observe.cameras")`` → ``{"$ref": "observe.cameras"}``. Use inline
    in any ``inputs={...}`` dict to refer to another node's output, or to a
    subgraph-bound input as ``Ref("in.<name>")``.
    """
    if not isinstance(path, str) or not path:
        raise BuilderError(f"Ref path must be a non-empty string, got {path!r}")
    return {"$ref": path}


class BuilderError(Exception):
    """Raised for authoring-time errors before structural validation."""


# ---------------------------------------------------------------------------
# Shared base for Subgraph and Workflow
# ---------------------------------------------------------------------------


class _Builder:
    _nodes: dict[str, dict[str, Any]]
    _edges: list[list[str]]
    _cond: dict[str, dict[str, Any]]

    def __init__(self) -> None:
        self._nodes = {}
        self._edges = []
        self._cond = {}

    def add_edge(self, src: str, dst: str) -> None:
        """Add a static edge ``src → dst``."""
        if not isinstance(src, str) or not isinstance(dst, str):
            raise BuilderError(
                f"add_edge requires string node names, got src={src!r} dst={dst!r}"
            )
        self._edges.append([src, dst])

    def add_conditional_edges(
        self,
        src: str,
        mapping: dict[str, str],
        *,
        router_field: str | None = None,
    ) -> None:
        """Add a conditional-edge dispatch from ``src``.

        ``router_field`` names the field on ``src``'s output to switch on;
        leave as ``None`` for ``type="router"`` nodes (the routing function
        returns the target name directly).
        """
        if not isinstance(src, str):
            raise BuilderError(f"add_conditional_edges src must be a string, got {src!r}")
        if not isinstance(mapping, dict) or not mapping:
            raise BuilderError(
                f"add_conditional_edges mapping must be a non-empty dict, got {mapping!r}"
            )
        for k, v in mapping.items():
            if not isinstance(k, str) or not isinstance(v, str):
                raise BuilderError(
                    f"add_conditional_edges mapping entries must be str→str, got {k!r}: {v!r}"
                )
        if src in self._cond:
            raise BuilderError(
                f"node {src!r} already has a conditional_edges entry"
            )
        self._cond[src] = {
            "router_field": router_field,
            "mapping": dict(mapping),
        }


# ---------------------------------------------------------------------------
# Subgraph
# ---------------------------------------------------------------------------


_SUBGRAPH_ONLY_NODE_TYPES = frozenset({"tool", "script", "router", "noop"})


_CHECKPOINT_SOFT_CAP = 6


@dataclass
class _CheckpointDef:
    """One ``add_checkpoint()`` call's worth of state.

    Held on the :class:`Subgraph` until ``dump_checkpoints_module`` writes
    the sidecar. Never serialized into ``workflow.json``.
    """

    name: str
    predicate: Callable[..., bool]
    diagnostics: Callable[..., dict] | None
    rationale: str
    validate: bool
    weight: float


class Subgraph(_Builder):
    """Author one self-contained subgraph.

    Inside a subgraph, valid node types are ``tool``, ``script``, ``router``,
    and ``noop`` (the latter is created on your behalf by :meth:`add_exit`).
    ``end``/``subgraph`` nodes only appear at the workflow top level.
    """

    name: str
    _skill: str
    _inputs: dict[str, str]
    _outputs: dict[str, dict[str, str]]
    _exit: dict[str, Any]
    _on_error: str | None
    _checkpoints: list[_CheckpointDef]
    _source_block: str | None

    def __init__(self, *, name: str, skill: str = "generic") -> None:
        super().__init__()
        if not isinstance(name, str) or not name:
            raise BuilderError(f"Subgraph name must be a non-empty string, got {name!r}")
        self.name = name
        self._skill = skill
        self._inputs = {}
        self._outputs = {}
        self._exit = {"router_field": None, "success_values": []}
        self._on_error = None
        self._checkpoints = []
        self._source_block = None

    # -- node primitives ----------------------------------------------------

    def add_node(
        self,
        name: str,
        *,
        type: str,
        tool: str | None = None,
        script: str | None = None,
        inputs: dict[str, Any] | None = None,
        streaming: bool = False,
    ) -> None:
        """Add a node to the subgraph.

        ``type`` must be one of ``"tool"``, ``"script"``, ``"router"``,
        ``"noop"``. ``add_exit`` is the preferred way to create the
        terminal ``noop`` marker that names the subgraph's success outcome.
        """
        if name in self._nodes:
            raise BuilderError(f"node {name!r} already exists in subgraph {self.name!r}")
        if name in (START, END):
            raise BuilderError(f"node name {name!r} is reserved")
        if type not in _SUBGRAPH_ONLY_NODE_TYPES:
            raise BuilderError(
                f"invalid subgraph node type {type!r}; expected one of "
                f"{sorted(_SUBGRAPH_ONLY_NODE_TYPES)}"
            )
        node: dict[str, Any] = {"type": type}
        if inputs:
            node["inputs"] = dict(inputs)
        if streaming:
            if type not in ("tool", "script"):
                raise BuilderError(
                    f"streaming=True is only valid on tool/script nodes, got type={type!r}"
                )
            node["streaming"] = True
        if type == "tool":
            if not tool:
                raise BuilderError(f"node {name!r} (type=tool) requires tool=")
            node["tool"] = tool
        elif type == "script":
            if not script:
                raise BuilderError(f"node {name!r} (type=script) requires script=")
            node["script"] = script
        elif type == "router":
            if not script:
                raise BuilderError(f"node {name!r} (type=router) requires script=")
            node["script"] = script
        self._nodes[name] = node

    # -- subgraph-specific --------------------------------------------------

    def add_input(self, name: str, *, type_name: str) -> None:
        """Declare a cross-subgraph input.

        ``type_name`` is a type-name string from the :data:`gap.schema.TYPE_REGISTRY`
        (e.g. ``"OrientedBoundingBox"``). Reference it inside node ``inputs``
        as ``Ref(f"in.{name}")``.
        """
        if not isinstance(name, str) or not name:
            raise BuilderError(f"input name must be a non-empty string, got {name!r}")
        if name in self._inputs:
            raise BuilderError(f"input {name!r} already declared")
        self._inputs[name] = type_name

    def add_exit(self, name: str) -> None:
        """Declare a success-outcome marker.

        Creates a ``noop`` node named ``name`` and registers it as a
        success value. Used when ``exit.router_field`` is null (the common
        case): the terminal node whose name is reached determines the
        subgraph's exit value. Call once per success outcome.

        For data-dependent exits (a terminal script that returns the exit
        value in a field), use :meth:`set_exit_router` instead.
        """
        if self._exit["router_field"] is not None:
            raise BuilderError(
                "cannot use add_exit() after set_exit_router(); "
                "they are mutually exclusive"
            )
        if name in self._nodes:
            raise BuilderError(f"node {name!r} already exists, cannot use add_exit({name!r})")
        self._nodes[name] = {"type": "noop"}
        if name in self._exit["success_values"]:
            raise BuilderError(f"success value {name!r} already registered")
        self._exit["success_values"].append(name)

    def set_exit_router(
        self,
        *,
        router_field: str,
        success_values: Sequence[str],
    ) -> None:
        """Configure data-dependent exit routing.

        ``router_field`` names the field on the terminal node's output that
        carries the exit value; ``success_values`` enumerates the legal
        success-path values. No ``noop`` markers are created — callers wire
        their own terminal node whose output contains ``router_field``.
        """
        if not isinstance(router_field, str) or not router_field:
            raise BuilderError(f"router_field must be a non-empty string, got {router_field!r}")
        sv = [str(v) for v in success_values]
        if not sv:
            raise BuilderError("success_values must be a non-empty list")
        self._exit = {"router_field": router_field, "success_values": sv}

    def set_outputs(self, **bindings: dict[str, str]) -> None:
        """Bind named subgraph outputs to internal node fields.

        Each value must be a :func:`Ref` dict referring to an internal node
        field. Replaces any previously-set outputs.
        """
        out: dict[str, dict[str, str]] = {}
        for k, v in bindings.items():
            if not (isinstance(v, dict) and "$ref" in v):
                raise BuilderError(
                    f"output {k!r} must be a Ref() dict, got {v!r}"
                )
            out[k] = dict(v)
        self._outputs = out

    def set_on_error(self, value: str | None) -> None:
        """Set the failure-path exit symbol (or unset with ``None``)."""
        if value is not None and not isinstance(value, str):
            raise BuilderError(f"on_error must be a string or None, got {value!r}")
        self._on_error = value

    # -- postcondition checkpoints -----------------------------------------

    def add_checkpoint(
        self,
        name: str,
        predicate: Callable[..., bool],
        *,
        diagnostics: Callable[..., dict] | None = None,
        rationale: str = "",
        validate: bool = True,
        weight: float = 1.0,
    ) -> None:
        """Declare a postcondition checkpoint for this subgraph.

        ``predicate`` is evaluated against a :class:`gap.runtime.verify.World`
        snapshot taken at subgraph exit.

        ``validate=True`` (default) marks this as a *hard* postcondition:
        it is enforced at execution time when the connector exposes ground
        truth (``gap.execute(..., checkpoints="warn"|"raise")``).
        ``validate=False`` checkpoints are *probes* — they surface in
        feedback but never gate anything and are never enforced.

        Checkpoints are not part of ``workflow.json``; they are written to
        ``<workflow_dir>/checkpoints/<sg>.py`` by
        :meth:`dump_checkpoints_module` and loaded by the checkpoint hook.
        """
        if not isinstance(name, str) or not name:
            raise BuilderError(
                f"checkpoint name must be a non-empty string, got {name!r}"
            )
        if not callable(predicate):
            raise BuilderError(
                f"checkpoint {name!r} predicate must be callable, got "
                f"{type(predicate).__name__}"
            )
        if diagnostics is not None and not callable(diagnostics):
            raise BuilderError(
                f"checkpoint {name!r} diagnostics must be callable or None, "
                f"got {type(diagnostics).__name__}"
            )
        if not isinstance(weight, (int, float)) or float(weight) <= 0:
            raise BuilderError(
                f"checkpoint {name!r} weight must be a positive number, "
                f"got {weight!r}"
            )
        for existing in self._checkpoints:
            if existing.name == name:
                raise BuilderError(
                    f"checkpoint {name!r} already declared on subgraph "
                    f"{self.name!r}"
                )
        if len(self._checkpoints) >= _CHECKPOINT_SOFT_CAP:
            warnings.warn(
                f"subgraph {self.name!r} declares "
                f"{len(self._checkpoints) + 1} checkpoints — keep it "
                f"≤ {_CHECKPOINT_SOFT_CAP} so the feedback prompt stays focused",
                UserWarning,
                stacklevel=2,
            )
        self._checkpoints.append(_CheckpointDef(
            name=name,
            predicate=predicate,
            diagnostics=diagnostics,
            rationale=str(rationale),
            validate=bool(validate),
            weight=float(weight),
        ))

    def _set_source_block(self, code: str) -> None:
        """Internal: stash the builder block's source so the sidecar can
        re-exec it. Called by the subgraph-agent runner (``gap.agent``)
        after the sandbox exec succeeds.
        """
        if not isinstance(code, str):
            raise BuilderError(
                f"source block must be a string, got {type(code).__name__}"
            )
        self._source_block = code

    def _render_checkpoints_module(self) -> str | None:
        """Return the sidecar module source as a string, or ``None`` if no
        checkpoints were declared.

        The returned source re-executes the original builder block and
        exposes ``CHECKPOINTS: list[Checkpoint]`` by rebinding each entry
        from the rebuilt ``sg._checkpoints``.
        """
        if not self._checkpoints:
            return None
        if not self._source_block:
            raise BuilderError(
                f"subgraph {self.name!r}: cannot render checkpoints sidecar "
                "without a source block (call _set_source_block first)"
            )
        body = self._source_block.rstrip("\n")
        header = (
            '"""Auto-generated checkpoint sidecar for subgraph '
            f'`{self.name}`.\n'
            "\n"
            "Do not edit. The original builder block is preserved verbatim\n"
            "below so the harness can re-exec it and capture predicate\n"
            "lambdas with their original closure cells.\n"
            '"""\n'
            "\n"
            "from __future__ import annotations\n"
            "\n"
            "import math\n"
            "\n"
            "import numpy as np\n"
            "\n"
            "from gap.builder import Subgraph, Ref, START, END\n"
            "from gap.runtime.verify import Checkpoint as _Checkpoint\n"
            "\n"
            "# --- original builder block ---\n"
        )
        footer = (
            "\n"
            "# --- end original builder block ---\n"
            "\n"
            "CHECKPOINTS: list[_Checkpoint] = [\n"
            "    _Checkpoint(\n"
            "        name=_c.name,\n"
            "        subgraph=sg.name,\n"
            "        predicate=_c.predicate,\n"
            "        diagnostics_fn=_c.diagnostics,\n"
            "        rationale=_c.rationale,\n"
            "        validate=_c.validate,\n"
            "        weight=_c.weight,\n"
            "    )\n"
            "    for _c in sg._checkpoints\n"
            "]\n"
        )
        return header + body + footer

    def dump_checkpoints_module(self, path: str | Path) -> Path | None:
        """Write the checkpoint sidecar to ``path``.

        Returns the written path, or ``None`` if no checkpoints are
        declared. Parent directories are created as needed.
        """
        source = self._render_checkpoints_module()
        if source is None:
            return None
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(source, encoding="utf-8")
        return p

    # -- serialization ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return the v3 subgraph JSON shape (no ``name`` field)."""
        d: dict[str, Any] = {
            "skill": self._skill,
            "inputs": dict(self._inputs),
            "outputs": {k: dict(v) for k, v in self._outputs.items()},
            "nodes": {k: dict(v) for k, v in self._nodes.items()},
            "edges": [list(e) for e in self._edges],
            "conditional_edges": {
                k: {"router_field": v["router_field"], "mapping": dict(v["mapping"])}
                for k, v in self._cond.items()
            },
            "exit": {
                "router_field": self._exit["router_field"],
                "success_values": list(self._exit["success_values"]),
            },
        }
        if self._on_error is not None:
            d["on_error"] = self._on_error
        return d

    def save(self, path: str | Path) -> None:
        """Write the subgraph to JSON with a wrapping ``name`` field.

        Standalone subgraph JSON is non-canonical (the executor consumes
        subgraphs only inside a workflow's ``subgraphs`` map). The wrapper
        carries the name so :meth:`load` can round-trip it.

        Runs S1–S11 by stuffing the subgraph into a stub workflow.
        """
        self._check_via_stub_workflow()
        out = {"name": self.name, **self.to_dict()}
        Path(path).write_text(json.dumps(out, indent=2))

    @classmethod
    def load(cls, path_or_dict: str | Path | dict[str, Any]) -> Subgraph:
        """Load a subgraph from a path or from a parsed dict.

        Accepts both the wrapped form produced by :meth:`save` (``{"name":
        ..., "skill": ..., ...}``) and a bare subgraph dict (caller supplies
        the name via the dict key).
        """
        from gap.runtime.workflow import _parse_subgraph

        if isinstance(path_or_dict, (str, Path)):
            raw = json.loads(Path(path_or_dict).read_text())
        else:
            raw = path_or_dict
        if not isinstance(raw, dict):
            raise BuilderError(f"subgraph payload must be an object, got {type(raw).__name__}")
        if "name" not in raw:
            raise BuilderError(
                "subgraph payload missing 'name' field; "
                "use _subgraph_from_runtime_def for unwrapped subgraph dicts"
            )
        name = raw["name"]
        body = {k: v for k, v in raw.items() if k != "name"}
        # Syntax-check via the runtime parser.
        sg_def = _parse_subgraph(name, body)
        return _subgraph_from_runtime_def(name, sg_def)

    # -- internals ----------------------------------------------------------

    def _check_via_stub_workflow(self) -> None:
        """Run S1–S11 by embedding this subgraph in a stub workflow."""
        from gap.runtime.validate import validate_workflow
        from gap.runtime.workflow import _parse_workflow

        stub = {
            "version": 3,
            "meta": {},
            "nodes": {
                "sg":   {"type": "subgraph", "ref": self.name},
                "done": {"type": "end", "status": "success"},
                "abort": {"type": "end", "status": "failure"},
            },
            "edges": [[START, "sg"]],
            "conditional_edges": _stub_conditional_edges(self),
            "subgraphs": {self.name: self.to_dict()},
        }
        try:
            wf = _parse_workflow(stub, Path("."))
        except WorkflowValidationError as e:
            raise GraphValidationError([_issue("error", f"subgraphs.{self.name}", str(e))]) from e
        issues = [i for i in validate_workflow(wf) if i.severity == "error"]
        # Filter W-rule issues that come from the stub's top-level wiring.
        sg_issues = [i for i in issues if i.node_id.startswith(f"subgraphs.{self.name}")]
        if sg_issues:
            raise GraphValidationError(sg_issues)


def _stub_conditional_edges(sg: Subgraph) -> dict[str, dict[str, Any]]:
    success = sg._exit["success_values"]
    mapping: dict[str, str] = {v: "done" for v in success}
    if sg._on_error is not None:
        mapping[sg._on_error] = "abort"
    if not mapping:
        # Shouldn't happen — S7 catches empty success_values — but degrade
        # gracefully so the user sees the real S7 error instead of a stub
        # KeyError.
        mapping = {"_dummy": "done"}
    return {"sg": {"router_field": sg._exit["router_field"], "mapping": mapping}}


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------


_WORKFLOW_NODE_TYPES = frozenset({"tool", "script", "router", "subgraph", "end", "noop"})


class Workflow(_Builder):
    """Author a top-level workflow.

    Top-level nodes are typically ``subgraph`` (one per subagent's
    Subgraph) and ``end``; ``tool``/``script``/``router``/``noop`` are also
    allowed when the coordinator inlines a step.
    """

    _meta: dict[str, str]
    _subgraphs: dict[str, Subgraph]
    _workflow_dir: Path | None

    def __init__(
        self,
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        super().__init__()
        self._meta = {}
        if name is not None:
            self._meta["name"] = name
        if description is not None:
            self._meta["description"] = description
        self._subgraphs = {}
        self._workflow_dir = None

    # -- node primitives ----------------------------------------------------

    def add_node(
        self,
        name: str,
        *,
        type: str,
        tool: str | None = None,
        script: str | None = None,
        ref: str | None = None,
        status: Literal["success", "failure"] | None = None,
        recovery: Sequence[ToolCall | dict[str, Any]] = (),
        inputs: dict[str, Any] | None = None,
        streaming: bool = False,
    ) -> None:
        if name in self._nodes:
            raise BuilderError(f"node {name!r} already exists in workflow")
        if name in (START, END):
            raise BuilderError(f"node name {name!r} is reserved")
        if type not in _WORKFLOW_NODE_TYPES:
            raise BuilderError(
                f"invalid node type {type!r}; expected one of {sorted(_WORKFLOW_NODE_TYPES)}"
            )
        node: dict[str, Any] = {"type": type}
        if inputs:
            node["inputs"] = dict(inputs)
        if streaming:
            if type not in ("tool", "script"):
                raise BuilderError(
                    f"streaming=True is only valid on tool/script nodes, got type={type!r}"
                )
            node["streaming"] = True
        if type == "tool":
            if not tool:
                raise BuilderError(f"node {name!r} (type=tool) requires tool=")
            node["tool"] = tool
        elif type == "script":
            if not script:
                raise BuilderError(f"node {name!r} (type=script) requires script=")
            node["script"] = script
        elif type == "router":
            if not script:
                raise BuilderError(f"node {name!r} (type=router) requires script=")
            node["script"] = script
        elif type == "subgraph":
            if not ref:
                raise BuilderError(f"node {name!r} (type=subgraph) requires ref=")
            node["ref"] = ref
        elif type == "end":
            if status not in ("success", "failure"):
                raise BuilderError(
                    f"node {name!r} (type=end) requires status='success' or 'failure', got {status!r}"
                )
            node["status"] = status
            if recovery:
                node["recovery"] = [_tool_call_to_dict(rc) for rc in recovery]
        self._nodes[name] = node

    # -- subgraph composition ----------------------------------------------

    def add_subgraph(self, sg: Subgraph) -> None:
        """Register a :class:`Subgraph` under its ``name`` field."""
        if not isinstance(sg, Subgraph):
            raise BuilderError(f"add_subgraph expects a Subgraph, got {type(sg).__name__}")
        if sg.name in self._subgraphs:
            raise BuilderError(f"subgraph {sg.name!r} already registered")
        self._subgraphs[sg.name] = sg

    # -- serialization ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 3,
            "meta": dict(self._meta),
            "nodes": {k: dict(v) for k, v in self._nodes.items()},
            "edges": [list(e) for e in self._edges],
            "conditional_edges": {
                k: {"router_field": v["router_field"], "mapping": dict(v["mapping"])}
                for k, v in self._cond.items()
            },
            "subgraphs": {k: sg.to_dict() for k, sg in self._subgraphs.items()},
        }

    def save(self, path: str | Path, *, validate: bool = True) -> None:
        """Serialize to JSON. Parses (W1–W8 syntax) and structurally
        validates (W/S rules) before writing if ``validate=True``."""
        raw = self.to_dict()
        p = Path(path)
        workflow_dir = self._workflow_dir or p.parent
        if validate:
            from gap.runtime.validate import validate_workflow
            from gap.runtime.workflow import _parse_workflow

            try:
                wf = _parse_workflow(raw, workflow_dir)
            except WorkflowValidationError as e:
                raise GraphValidationError([_issue("error", "workflow", str(e))]) from e
            issues = [i for i in validate_workflow(wf) if i.severity == "error"]
            if issues:
                raise GraphValidationError(issues)
        p.write_text(json.dumps(raw, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> Workflow:
        """Load an existing workflow.json into a mutable Workflow builder.

        Uses the runtime's strict parser, then converts the resulting
        frozen dataclasses back into JSON-shaped dicts hydrated into a new
        builder. Round-trip with :meth:`to_dict` is byte-equivalent for
        any valid workflow.json.
        """
        from gap.runtime.workflow import load_workflow

        wf_def = load_workflow(path)
        return _workflow_from_runtime_def(wf_def)


# ---------------------------------------------------------------------------
# WorkflowSpec — coordinator output (workflow scaffold with stub subgraphs)
# ---------------------------------------------------------------------------


class WorkflowSpec(_Builder):
    """Coordinator-output builder — workflow scaffold with subgraph metadata stubs.

    The coordinator decides the workflow topology (which subgraphs run, in
    what order, with what exit conditions) without filling in each
    subgraph's nodes/edges. The subgraph_agent fills those in afterward.
    A ``WorkflowSpec`` carries the topology plus the subgraph metadata
    stubs the assembler later merges with subgraph_agent outputs.

    Output shape (``to_dict()``) matches the legacy coordinator JSON:

    ``{version, meta, nodes, edges, conditional_edges, subgraphs}``,
    where each entry in ``subgraphs`` is a metadata stub
    ``{skill, inputs, outputs, description, exit, on_error}``.
    """

    _meta: dict[str, str]
    _subgraphs: dict[str, dict[str, Any]]

    def __init__(
        self,
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        super().__init__()
        self._meta = {}
        if name is not None:
            self._meta["name"] = name
        if description is not None:
            self._meta["description"] = description
        self._subgraphs = {}

    def add_subgraph_node(
        self,
        name: str,
        *,
        ref: str,
        inputs: dict[str, Any] | None = None,
    ) -> None:
        """Top-level node of ``type="subgraph"`` referencing a declared subgraph."""
        if name in self._nodes:
            raise BuilderError(f"node {name!r} already exists in workflow spec")
        if name in (START, END):
            raise BuilderError(f"node name {name!r} is reserved")
        if not isinstance(ref, str) or not ref:
            raise BuilderError(f"subgraph node {name!r} requires ref=")
        node: dict[str, Any] = {"type": "subgraph", "ref": ref}
        if inputs:
            node["inputs"] = dict(inputs)
        self._nodes[name] = node

    def add_end(
        self,
        name: str,
        *,
        status: Literal["success", "failure"],
        recovery: Sequence[ToolCall | dict[str, Any]] = (),
    ) -> None:
        """Top-level ``type="end"`` node."""
        if name in self._nodes:
            raise BuilderError(f"node {name!r} already exists in workflow spec")
        if name in (START, END):
            raise BuilderError(f"node name {name!r} is reserved")
        if status not in ("success", "failure"):
            raise BuilderError(
                f"end node {name!r} requires status='success' or 'failure', got {status!r}"
            )
        node: dict[str, Any] = {"type": "end", "status": status}
        if recovery:
            node["recovery"] = [_tool_call_to_dict(rc) for rc in recovery]
        self._nodes[name] = node

    def declare_subgraph(
        self,
        name: str,
        *,
        skill: str,
        description: str,
        exit_success_values: Sequence[str],
        on_error: str,
        inputs: dict[str, str] | None = None,
        outputs: dict[str, str] | None = None,
        stage: str | None = None,
        generated: bool = False,
    ) -> None:
        """Declare a subgraph metadata stub.

        ``inputs`` and ``outputs`` are dicts of ``{name: type_name_str}``
        — type declarations, not data bindings (cross-subgraph data flow
        is established later by name-matching in the assembler).

        ``stage`` is the canonical pick-and-place stage tag
        (``"grasp"`` / ``"transport"`` / ``"place"``) used by the
        mechanical-swap engine. Omit for subgraphs outside the canonical
        taxonomy (e.g. perception staging).

        ``generated`` marks an *invented* skill: ``skill`` names a brand-new
        skill the coordinator is defining (NOT a registered bundle), whose
        ``inputs`` / ``outputs`` / ``exit_success_values`` / ``on_error``
        ARE its contract. The subgraph_agent implements it from scratch by
        composing ``type: tool`` nodes and authoring ``type: script`` nodes
        (no SKILL.md, no canonical scripts). Use only when no existing
        skill fits — see the coordinator prompt's fallback policy.
        """
        if not isinstance(name, str) or not name:
            raise BuilderError(f"subgraph name must be a non-empty string, got {name!r}")
        if name in self._subgraphs:
            raise BuilderError(f"subgraph {name!r} already declared")
        for label, dct in (("inputs", inputs or {}), ("outputs", outputs or {})):
            for k, v in dct.items():
                if not isinstance(v, str):
                    raise BuilderError(
                        f"subgraph {name!r} {label}.{k} must be a type name string, got {v!r}"
                    )
        sv = [str(v) for v in exit_success_values]
        if not sv:
            raise BuilderError(f"subgraph {name!r}.exit_success_values must be non-empty")
        if stage is not None and stage not in ("grasp", "transport", "place"):
            raise BuilderError(
                f"subgraph {name!r}.stage must be one of "
                f"'grasp' / 'transport' / 'place' or None, got {stage!r}"
            )
        entry: dict[str, Any] = {
            "skill": skill,
            "inputs": dict(inputs or {}),
            "outputs": dict(outputs or {}),
            "description": description,
            "exit": {"router_field": None, "success_values": sv},
            "on_error": on_error,
        }
        if stage is not None:
            entry["stage"] = stage
        if generated:
            entry["generated"] = True
        self._subgraphs[name] = entry

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 3,
            "meta": dict(self._meta),
            "nodes": {k: dict(v) for k, v in self._nodes.items()},
            "edges": [list(e) for e in self._edges],
            "conditional_edges": {
                k: {"router_field": v["router_field"], "mapping": dict(v["mapping"])}
                for k, v in self._cond.items()
            },
            "subgraphs": {k: dict(v) for k, v in self._subgraphs.items()},
        }


# ---------------------------------------------------------------------------
# Round-trip helpers: frozen runtime types → mutable builder state
# ---------------------------------------------------------------------------


def _workflow_from_runtime_def(wf_def: Any) -> Workflow:
    wf = Workflow()
    wf._meta = dict(wf_def.meta)
    wf._workflow_dir = wf_def.workflow_dir
    wf._nodes = {n: _node_def_to_dict(nd) for n, nd in wf_def.nodes.items()}
    wf._edges = [[s, d] for s, d in wf_def.edges]
    wf._cond = {
        s: {"router_field": ce.router_field, "mapping": dict(ce.mapping)}
        for s, ce in wf_def.conditional_edges.items()
    }
    wf._subgraphs = {
        sg_name: _subgraph_from_runtime_def(sg_name, sg_def)
        for sg_name, sg_def in wf_def.subgraphs.items()
    }
    return wf


def _subgraph_from_runtime_def(name: str, sg_def: Any) -> Subgraph:
    sg = Subgraph(name=name, skill=sg_def.skill)
    sg._inputs = dict(sg_def.inputs)
    sg._outputs = {k: {"$ref": ref.path} for k, ref in sg_def.outputs.items()}
    sg._nodes = {n: _node_def_to_dict(nd) for n, nd in sg_def.nodes.items()}
    sg._edges = [[s, d] for s, d in sg_def.edges]
    sg._cond = {
        s: {"router_field": ce.router_field, "mapping": dict(ce.mapping)}
        for s, ce in sg_def.conditional_edges.items()
    }
    sg._exit = {
        "router_field": sg_def.exit.router_field,
        "success_values": list(sg_def.exit.success_values),
    }
    sg._on_error = sg_def.on_error
    return sg


def _node_def_to_dict(nd: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"type": nd.type}
    inputs = _resolve_inputs_to_json(nd.inputs)
    if inputs:
        out["inputs"] = inputs
    if nd.streaming:
        out["streaming"] = True
    if nd.script is not None:
        out["script"] = nd.script
    if nd.tool is not None:
        out["tool"] = nd.tool
    if nd.ref is not None:
        out["ref"] = nd.ref
    if nd.status is not None:
        out["status"] = nd.status
    if nd.recovery:
        out["recovery"] = [_tool_call_to_dict(tc) for tc in nd.recovery]
    return out


def _resolve_inputs_to_json(inputs: dict[str, Any]) -> dict[str, Any]:
    return {k: _value_to_json(v) for k, v in inputs.items()}


def _value_to_json(value: Any) -> Any:
    # Local import to avoid a top-level dependency on the runtime.
    from gap.runtime.workflow import Ref as _RuntimeRef
    if isinstance(value, _RuntimeRef):
        return {"$ref": value.path}
    if isinstance(value, list):
        return [_value_to_json(v) for v in value]
    return value


def _tool_call_to_dict(tc: ToolCall | dict[str, Any]) -> dict[str, Any]:
    """Serialize a recovery entry to the v3 ``{"tool", "inputs"}`` shape.

    Accepts a :class:`gap.runtime.workflow.ToolCall` (duck-typed on its
    ``tool``/``inputs`` attributes so the runtime is not a builder import)
    or an already-JSON-shaped dict.
    """
    if isinstance(tc, dict):
        return dict(tc)
    tool = getattr(tc, "tool", None)
    if isinstance(tool, str) and tool:
        return {"tool": tool, "inputs": dict(getattr(tc, "inputs", {}) or {})}
    raise BuilderError(f"recovery entry must be a ToolCall or dict, got {type(tc).__name__}")


def _issue(severity: str, location: str, message: str) -> ValidationIssue:
    return ValidationIssue(severity=severity, node_id=location, field=None, message=message)
