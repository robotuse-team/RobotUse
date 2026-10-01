"""Build normalized view models for the workflow visualizer."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

from .models import (
    DataProvenance,
    ExecutionLane,
    ExecutionStep,
    ExecutionTrace,
    ExecutionTransition,
    NodeTraceData,
    ProvenanceEdgeView,
    TraceEvent,
    VizMeta,
    VizTrial,
    WorkflowGraph,
)


def build_viz_trial(
    workflow: WorkflowGraph,
    nodes: list[NodeTraceData],
    raw_trace: dict[str, Any],
    trial_path: str | None = None,
    total_duration_ms: float = 0.0,
) -> VizTrial:
    """Build the normalized visualization payload for one trial."""
    execution = build_execution_trace(workflow, nodes, raw_trace, total_duration_ms)
    provenance = build_data_provenance(workflow, execution)
    return VizTrial(
        meta=VizMeta(
            total_duration_ms=execution.total_duration_ms or total_duration_ms,
            degraded_replay=execution.degraded,
            trial_path=trial_path,
        ),
        workflow=workflow,
        execution=execution,
        provenance=provenance,
    )


def build_execution_trace(
    workflow: WorkflowGraph,
    nodes: list[NodeTraceData],
    raw_trace: dict[str, Any],
    total_duration_ms: float,
) -> ExecutionTrace:
    """Create the column-per-phase execution model."""
    state_map = {state.state_id: state for state in workflow.states}
    lanes = _build_lanes(workflow, nodes)
    step_sequences = _step_sequences(nodes, raw_trace)

    executed_nodes = [
        node for node in nodes
        if node.started_at > 0 or node.finished_at > 0 or node.status in {"ok", "error", "running"}
    ]

    executed_nodes.sort(
        key=lambda node: (
            step_sequences.get(node.node_id, 10**9),
            node.started_at if node.started_at > 0 else float("inf"),
            node.node_id,
        )
    )

    steps: list[ExecutionStep] = []
    for fallback_index, node in enumerate(executed_nodes):
        state = state_map.get(node.node_id)
        subgraph_id = state.subgraph_id if state else node.node_id.split(".")[0]
        steps.append(
            ExecutionStep(
                id=f"step:{node.node_id}:{step_sequences.get(node.node_id, fallback_index)}",
                state_id=state.state_id if state else node.node_id,
                node_id=node.node_id,
                subgraph_id=subgraph_id,
                lane_id=subgraph_id,
                title=state.name if state else node.node_id.split(".")[-1],
                state_type=state.state_type if state else (node.node_type or "service"),
                status=node.status,
                sequence=step_sequences.get(node.node_id, fallback_index),
                started_at=node.started_at,
                finished_at=node.finished_at or node.started_at,
                duration_ms=node.duration_ms,
                has_inputs=node.has_inputs,
                has_output=node.has_output,
                error_message=node.error_message,
                condition_result=node.condition_result,
                assets=node.assets,
            )
        )

    transitions = _build_execution_transitions(workflow, steps)
    events = _parse_trace_events(raw_trace)

    return ExecutionTrace(
        lanes=lanes,
        steps=steps,
        transitions=transitions,
        events=events,
        total_duration_ms=round(total_duration_ms, 2),
        degraded=not bool(events),
    )


def build_data_provenance(
    workflow: WorkflowGraph,
    execution: ExecutionTrace,
) -> DataProvenance:
    """Create provenance edges scoped to concrete executions when possible."""
    steps_by_state: dict[str, list[ExecutionStep]] = defaultdict(list)
    for step in sorted(execution.steps, key=lambda item: item.sequence):
        steps_by_state[step.state_id].append(step)

    edges: list[ProvenanceEdgeView] = []
    if execution.steps:
        for target_step in sorted(execution.steps, key=lambda item: item.sequence):
            for data_edge in workflow.data_edges:
                if data_edge.target != target_step.state_id:
                    continue
                source_step = _latest_step_before(
                    steps_by_state.get(data_edge.source, []),
                    target_step.sequence,
                )
                edges.append(
                    ProvenanceEdgeView(
                        id=f"{target_step.id}:{data_edge.source}:{data_edge.target_field}",
                        source_state_id=data_edge.source,
                        source_step_id=source_step.id if source_step else None,
                        source_subgraph_id=data_edge.source.split(".")[0],
                        source_port=data_edge.source_field,
                        target_state_id=data_edge.target,
                        target_step_id=target_step.id,
                        target_subgraph_id=target_step.subgraph_id,
                        target_port=data_edge.target_field,
                        ref_path=data_edge.ref_path,
                        type_label=data_edge.type_label,
                        cross_subgraph=data_edge.cross_subgraph,
                    )
                )
    else:
        for data_edge in workflow.data_edges:
            edges.append(
                ProvenanceEdgeView(
                    id=f"static:{data_edge.source}:{data_edge.target}:{data_edge.target_field}",
                    source_state_id=data_edge.source,
                    source_subgraph_id=data_edge.source.split(".")[0],
                    source_port=data_edge.source_field,
                    target_state_id=data_edge.target,
                    target_subgraph_id=data_edge.target.split(".")[0],
                    target_port=data_edge.target_field,
                    ref_path=data_edge.ref_path,
                    type_label=data_edge.type_label,
                    cross_subgraph=data_edge.cross_subgraph,
                )
            )

    return DataProvenance(edges=edges)


def _build_lanes(
    workflow: WorkflowGraph,
    nodes: list[NodeTraceData],
) -> list[ExecutionLane]:
    executed_subgraphs = {
        node.node_id.split(".")[0]
        for node in nodes
        if node.started_at > 0 or node.finished_at > 0 or node.status in {"ok", "error", "running"}
    }
    columns = _build_execution_columns(workflow)
    lanes: list[ExecutionLane] = []
    for subgraph in workflow.subgraphs:
        lanes.append(
            ExecutionLane(
                id=subgraph.subgraph_id,
                title=subgraph.subgraph_id,
                agent=subgraph.agent,
                column=columns.get(subgraph.subgraph_id, len(columns)),
                is_end=subgraph.is_end,
                executed=subgraph.subgraph_id in executed_subgraphs,
            )
        )
    lanes.sort(key=lambda lane: (lane.column, lane.title))
    return lanes


def _build_execution_columns(workflow: WorkflowGraph) -> dict[str, int]:
    """BFS from workflow.begin over cross-subgraph transitions.

    Each subgraph is assigned a column index by first-visit order.
    Priority given to ``on_success`` edges so the happy-path subgraphs
    appear left-to-right. Failure/recovery branches get later columns.
    """
    adjacency: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for edge in workflow.control_edges:
        if edge.edge_type != "transition":
            continue
        source = edge.source.split(".")[0]
        target = edge.target
        priority = 0 if (edge.label or "").lower() in {"success", "ok"} else 1
        adjacency[source].append((target, priority))

    for source in adjacency:
        adjacency[source].sort(key=lambda pair: pair[1])

    order: list[str] = []
    seen: set[str] = set()

    def visit(start: str) -> None:
        if start in seen:
            return
        queue: deque[str] = deque([start])
        seen.add(start)
        while queue:
            node = queue.popleft()
            order.append(node)
            for target, _ in adjacency.get(node, []):
                if target in seen:
                    continue
                seen.add(target)
                queue.append(target)

    if workflow.begin:
        visit(workflow.begin)

    for subgraph in workflow.subgraphs:
        if subgraph.subgraph_id not in seen:
            visit(subgraph.subgraph_id)

    return {name: index for index, name in enumerate(order)}


def _step_sequences(nodes: list[NodeTraceData], raw_trace: dict[str, Any]) -> dict[str, int]:
    start_sequences: dict[str, int] = {}
    for event in _parse_trace_events(raw_trace):
        if event.event_type != "node_started":
            continue
        node_name = event.node_name or event.detail.get("name", "")
        if node_name and node_name not in start_sequences:
            start_sequences[node_name] = event.seq

    if start_sequences:
        return start_sequences

    ordered = sorted(
        nodes,
        key=lambda node: (
            node.started_at if node.started_at > 0 else float("inf"),
            node.finished_at if node.finished_at > 0 else float("inf"),
            node.node_id,
        ),
    )
    return {node.node_id: index for index, node in enumerate(ordered)}


def _build_execution_transitions(
    workflow: WorkflowGraph,
    steps: list[ExecutionStep],
) -> list[ExecutionTransition]:
    if not steps:
        return []

    state_to_steps: dict[str, list[ExecutionStep]] = defaultdict(list)
    subgraph_to_steps: dict[str, list[ExecutionStep]] = defaultdict(list)
    for step in sorted(steps, key=lambda item: item.sequence):
        state_to_steps[step.state_id].append(step)
        subgraph_to_steps[step.subgraph_id].append(step)

    transitions: list[ExecutionTransition] = []
    seen: set[tuple[str, str]] = set()
    for step in sorted(steps, key=lambda item: item.sequence):
        best_target: ExecutionStep | None = None
        best_kind = "sequence"
        best_label = ""

        for edge in workflow.control_edges:
            if edge.source != step.state_id:
                continue
            candidate: ExecutionStep | None = None
            if edge.edge_type == "transition":
                candidate = _first_step_after(subgraph_to_steps.get(edge.target, []), step.sequence)
                label = edge.label
                kind = "subgraph"
            else:
                candidate = _first_step_after(state_to_steps.get(edge.target, []), step.sequence)
                label = edge.edge_type
                kind = "state"
            if candidate is None:
                continue
            if best_target is None or candidate.sequence < best_target.sequence:
                best_target = candidate
                best_kind = kind
                best_label = label

        if best_target is None:
            best_target = _first_global_step_after(steps, step.sequence)
            best_kind = "sequence"
            best_label = ""

        if best_target is None:
            continue

        key = (step.id, best_target.id)
        if key in seen:
            continue
        seen.add(key)
        transitions.append(
            ExecutionTransition(
                id=f"{step.id}->{best_target.id}",
                source_step_id=step.id,
                target_step_id=best_target.id,
                kind=best_kind,
                label=best_label,
            )
        )

    return transitions


def _parse_trace_events(raw_trace: dict[str, Any]) -> list[TraceEvent]:
    events = raw_trace.get("events", [])
    parsed: list[TraceEvent] = []
    for index, item in enumerate(events):
        parsed.append(
            TraceEvent(
                seq=int(item.get("seq", index)),
                event_type=item.get("event_type", ""),
                node_id=item.get("node_id", ""),
                node_name=item.get("node_name", ""),
                timestamp=float(item.get("timestamp", 0.0)),
                status=item.get("status"),
                detail=item.get("detail", {}) or {},
            )
        )
    return parsed


def _latest_step_before(steps: list[ExecutionStep], sequence: int) -> ExecutionStep | None:
    for step in reversed(steps):
        if step.sequence < sequence:
            return step
    return steps[-1] if steps else None


def _first_step_after(steps: list[ExecutionStep], sequence: int) -> ExecutionStep | None:
    for step in steps:
        if step.sequence > sequence:
            return step
    return None


def _first_global_step_after(steps: list[ExecutionStep], sequence: int) -> ExecutionStep | None:
    for step in sorted(steps, key=lambda item: item.sequence):
        if step.sequence > sequence:
            return step
    return None
