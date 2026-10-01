"""Build WorkflowGraph models from a v3 Workflow dataclass.

Walks the v3 nodes/edges/conditional-edges structure directly.
Produces SubgraphSchema, StateSchema, ControlEdge, and DataEdge lists
that the frontend renders with spatial separation (control = top/bottom,
data = left/right handles).

Port-type information comes from the de-proto'd schema surface:
:mod:`gap.runtime.validate` builds :class:`~gap.runtime.validate.NodeSchema`
rows from script type hints and registered tool ``UnitSchema``s, and the
declared graph type names ("PointCloud", "Se3Pose", ...) resolve through
the :mod:`gap.schema` registry. No proto descriptors are consulted.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .models import (
    ControlEdge,
    DataEdge,
    PortSchema,
    RecoveryAction,
    StateSchema,
    SubgraphSchema,
    WorkflowGraph,
)

logger = logging.getLogger(__name__)


def _short_type(fs: Any) -> str:
    """Short display name for a :class:`gap.runtime.validate.FieldSchema`.

    The proto era mapped FieldDescriptor type codes here; the gap surface
    already carries bare type-name strings ("PointCloud", "str", ...).
    """
    name = fs.type_str or "?"
    if fs.is_repeated:
        return f"[]{name}"
    return name


def build_workflow_graph(
    workflow: Any,
    workflow_dir: Path,
    schemas: dict | None = None,
    tool_registry: Any | None = None,
) -> WorkflowGraph:
    """Build a WorkflowGraph from a v3 Workflow dataclass.

    Args:
        workflow: v3 `Workflow` dataclass (from load_workflow).
        workflow_dir: Directory containing workflow.json and scripts.
        schemas: Optional pre-built {state_id: NodeSchema} from validate.
        tool_registry: Optional :class:`gap.tools.ToolRegistry` for tool
            port schemas (each descriptor's ``schema`` is a ``UnitSchema``).
    """
    from gap.runtime.workflow import END, START, Ref, Workflow

    assert isinstance(workflow, Workflow), f"Expected Workflow, got {type(workflow)}"

    if schemas is None:
        schemas = _build_schemas(workflow, workflow_dir, tool_registry)

    # Build lookup: top-level node name → subgraph def name (for subgraph refs)
    toplevel_to_sg: dict[str, str] = {}
    for node_name, node in workflow.nodes.items():
        if node.type == "subgraph" and node.ref:
            toplevel_to_sg[node_name] = node.ref

    # Compute begin: first START target, mapped through subgraph refs
    begin = ""
    for src, dst in workflow.edges:
        if src == START:
            begin = toplevel_to_sg.get(dst, dst)
            break

    # ── 1. Build subgraph schemas ────────────────────────────────────
    subgraph_list: list[SubgraphSchema] = []

    for sg_name, sg in workflow.subgraphs.items():
        sg_begin = ""
        for src, dst in sg.edges:
            if src == START:
                sg_begin = dst
                break

        subgraph_list.append(SubgraphSchema(
            subgraph_id=sg_name,
            agent=sg.skill,
            inputs=dict(sg.inputs),
            outputs={k: v.path for k, v in sg.outputs.items()},
            begin_state=sg_begin,
        ))

    # End nodes at the top level become pseudo-subgraphs
    for node_name, node in workflow.nodes.items():
        if node.type == "end":
            subgraph_list.append(SubgraphSchema(
                subgraph_id=node_name,
                is_end=True,
                end_status=node.status,
                recovery=[RecoveryAction(tool=a.tool) for a in node.recovery],
            ))

    # ── 2. Build state schemas ───────────────────────────────────────
    state_list: list[StateSchema] = []

    for sg_name, sg in workflow.subgraphs.items():
        for node_name, node in sg.nodes.items():
            state_id = f"{sg_name}.{node_name}"

            if node.type in ("noop", "end"):
                state_list.append(StateSchema(
                    state_id=state_id,
                    name=node_name,
                    subgraph_id=sg_name,
                    state_type="end",
                ))
                continue

            schema = schemas.get(state_id)
            state_list.append(_make_state_schema(
                state_id, node_name, sg_name, node, schema,
            ))

    # ── 3. Build control edges ───────────────────────────────────────
    control_edges: list[ControlEdge] = []

    for sg_name, sg in workflow.subgraphs.items():
        # Direct edges within subgraph (skip START/END virtual nodes)
        for src, dst in sg.edges:
            if src == START or dst == END:
                continue
            control_edges.append(ControlEdge(
                source=f"{sg_name}.{src}",
                target=f"{sg_name}.{dst}",
                edge_type="success",
            ))

        # Conditional edges within subgraph
        for src, cond_edge in sg.conditional_edges.items():
            for label, dst in cond_edge.mapping.items():
                if dst == END:
                    continue
                control_edges.append(ControlEdge(
                    source=f"{sg_name}.{src}",
                    target=f"{sg_name}.{dst}",
                    edge_type="transition",
                    label=label,
                ))

    # Transitions: top-level conditional edges route between subgraphs
    for top_node_name, cond_edge in workflow.conditional_edges.items():
        sg_name = toplevel_to_sg.get(top_node_name)
        if sg_name is None:
            continue
        sg = workflow.subgraphs.get(sg_name)
        if sg is None:
            continue

        for exit_cond, target_top_name in cond_edge.mapping.items():
            target_id = toplevel_to_sg.get(target_top_name, target_top_name)
            source_id = f"{sg_name}.{exit_cond}"

            # If exit_cond is on_error (not an actual node), add a synthetic state
            if exit_cond not in sg.nodes and exit_cond == sg.on_error:
                state_list.append(StateSchema(
                    state_id=source_id,
                    name=exit_cond,
                    subgraph_id=sg_name,
                    state_type="end",
                ))

            control_edges.append(ControlEdge(
                source=source_id,
                target=target_id,
                edge_type="transition",
                label=exit_cond,
            ))

    # ── 4. Build data edges ──────────────────────────────────────────
    data_edges: list[DataEdge] = []

    # Pre-build output name → (subgraph_name, ref_path) map for cross-subgraph
    output_producers: dict[str, tuple[str, str]] = {}
    for sg_name, sg in workflow.subgraphs.items():
        for out_name, out_ref in sg.outputs.items():
            output_producers[out_name] = (sg_name, out_ref.path)

    for sg_name, sg in workflow.subgraphs.items():
        for node_name, node in sg.nodes.items():
            if node.type in ("noop", "end"):
                continue
            target_id = f"{sg_name}.{node_name}"
            for input_name, input_val in node.inputs.items():
                if not isinstance(input_val, Ref):
                    continue
                parts = input_val.parts()
                head = parts[0]
                field = ".".join(parts[1:]) if len(parts) > 1 else ""

                if head == "in":
                    # Cross-subgraph reference via bound inputs
                    in_name = parts[1] if len(parts) > 1 else ""
                    producer = output_producers.get(in_name)
                    if producer:
                        prod_sg, prod_ref = producer
                        prod_parts = prod_ref.split(".")
                        prod_state = prod_parts[0]
                        prod_field = ".".join(prod_parts[1:]) if len(prod_parts) > 1 else ""
                        prod_state_id = f"{prod_sg}.{prod_state}"
                        type_label = ""
                        if prod_state_id in schemas and prod_field:
                            fs = schemas[prod_state_id].outputs.get(prod_field)
                            if fs:
                                type_label = _short_type(fs)
                        data_edges.append(DataEdge(
                            source=prod_state_id,
                            source_field=prod_field,
                            target=target_id,
                            target_field=input_name,
                            ref_path=input_val.path,
                            cross_subgraph=True,
                            type_label=type_label,
                            source_sg_port=in_name,
                            target_sg_port=in_name,
                        ))
                else:
                    # Intra-subgraph reference
                    source_id = f"{sg_name}.{head}"
                    type_label = ""
                    if source_id in schemas and field:
                        fs = schemas[source_id].outputs.get(field)
                        if fs:
                            type_label = _short_type(fs)
                    data_edges.append(DataEdge(
                        source=source_id,
                        source_field=field,
                        target=target_id,
                        target_field=input_name,
                        ref_path=input_val.path,
                        cross_subgraph=False,
                        type_label=type_label,
                    ))

    # ── 5. Ensure all data-edge ports exist on state schemas ────────
    state_idx: dict[str, StateSchema] = {s.state_id: s for s in state_list}
    for de in data_edges:
        src = state_idx.get(de.source)
        if src is not None:
            field = de.source_field or "_out"
            if not any(p.name == field for p in src.output_ports):
                src.output_ports.append(PortSchema(
                    name=field, type_label=de.type_label or "?",
                    is_message=False, is_repeated=False,
                ))
        tgt = state_idx.get(de.target)
        if tgt is not None:
            field = de.target_field or "_in"
            if not any(p.name == field for p in tgt.input_ports):
                tgt.input_ports.append(PortSchema(
                    name=field, type_label=de.type_label or "?",
                    is_message=False, is_repeated=False,
                ))

    for de in data_edges:
        if not de.source_field:
            de.source_field = "_out"
        if not de.target_field:
            de.target_field = "_in"

    return WorkflowGraph(
        meta=dict(workflow.meta),
        begin=begin,
        subgraphs=subgraph_list,
        states=state_list,
        control_edges=control_edges,
        data_edges=data_edges,
    )


def _make_state_schema(
    state_id: str,
    name: str,
    subgraph_id: str,
    node: Any,
    schema: Any | None,
) -> StateSchema:
    """Create a StateSchema from a v3 NodeDef and optional validation schema."""
    from gap.runtime.workflow import Ref

    input_ports: list[PortSchema] = []
    output_ports: list[PortSchema] = []

    if schema:
        used_inputs = set(node.inputs.keys())
        for pname, fs in schema.inputs.items():
            if pname in used_inputs:
                input_ports.append(PortSchema(
                    name=pname,
                    type_label=_short_type(fs),
                    is_message=fs.is_message,
                    is_repeated=fs.is_repeated,
                ))
        for pname, fs in schema.outputs.items():
            output_ports.append(PortSchema(
                name=pname,
                type_label=_short_type(fs),
                is_message=fs.is_message,
                is_repeated=fs.is_repeated,
            ))

    inputs_def: dict = {}
    for k, v in node.inputs.items():
        inputs_def[k] = v.path if isinstance(v, Ref) else v

    return StateSchema(
        state_id=state_id,
        name=name,
        subgraph_id=subgraph_id,
        state_type=node.type,
        script=node.script,
        skill=node.tool,
        input_ports=input_ports,
        output_ports=output_ports,
        inputs_def=inputs_def,
    )


def _build_schemas(
    workflow: Any, workflow_dir: Path, tool_registry: Any | None = None,
) -> dict:
    """Build node schemas using validation introspection.

    Walks v3 subgraphs/nodes directly, producing schemas keyed by
    fully-qualified state ID (e.g. "insert_usb_c_sg.insert_usb_c").
    Script nodes introspect their ``run()`` type hints; tool nodes read
    the registered tool's ``UnitSchema`` when a registry is supplied.
    """
    try:
        from gap.runtime.validate import (
            build_script_node_schema,
            build_tool_node_schema,
        )
    except ImportError:
        logger.warning("Cannot import validate module — no port type info available")
        return {}

    schemas: dict = {}

    for sg_name, sg in workflow.subgraphs.items():
        for node_name, node in sg.nodes.items():
            if node.type in ("noop", "end"):
                continue
            state_id = f"{sg_name}.{node_name}"
            try:
                if node.type == "script":
                    schemas[state_id] = build_script_node_schema(
                        state_id, node, workflow_dir,
                    )
                elif (
                    node.type == "tool"
                    and tool_registry is not None
                    and (node.tool or "") in tool_registry
                ):
                    schemas[state_id] = build_tool_node_schema(
                        state_id, node, tool_registry,
                    )
            except Exception:
                logger.debug("Failed to build schema for %s", state_id, exc_info=True)

    return schemas
