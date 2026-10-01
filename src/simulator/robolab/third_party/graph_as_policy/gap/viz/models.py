"""Pydantic models for the visualization API.

These shapes are the JSON contract consumed by the bundled React frontend
(``frontend/src/types/workflow.ts`` / ``viz.ts``) — keep them in sync.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PortSchema(BaseModel):
    name: str
    type_label: str
    is_message: bool
    is_repeated: bool


class RecoveryAction(BaseModel):
    """One best-effort recovery tool call attached to a failure end node."""

    tool: str


class SubgraphSchema(BaseModel):
    subgraph_id: str
    agent: str = ""
    inputs: dict[str, str] = Field(default_factory=dict)
    outputs: dict[str, str] = Field(default_factory=dict)
    begin_state: str = ""
    is_end: bool = False
    end_status: str | None = None
    recovery: list[RecoveryAction] = Field(default_factory=list)


class StateSchema(BaseModel):
    state_id: str
    name: str
    subgraph_id: str
    state_type: str  # "tool" | "script" | "router" | "end" | ...
    service: str | None = None
    method: str | None = None
    script: str | None = None
    skill: str | None = None
    input_ports: list[PortSchema] = Field(default_factory=list)
    output_ports: list[PortSchema] = Field(default_factory=list)
    inputs_def: dict[str, Any] = Field(default_factory=dict)


class ControlEdge(BaseModel):
    source: str
    target: str
    edge_type: str  # "success" | "failure" | "transition"
    label: str = ""


class DataEdge(BaseModel):
    source: str
    source_field: str
    target: str
    target_field: str
    ref_path: str
    cross_subgraph: bool = False
    type_label: str = ""
    source_sg_port: str = ""
    target_sg_port: str = ""


class WorkflowGraph(BaseModel):
    meta: dict[str, Any] = Field(default_factory=dict)
    begin: str
    subgraphs: list[SubgraphSchema] = Field(default_factory=list)
    states: list[StateSchema] = Field(default_factory=list)
    control_edges: list[ControlEdge] = Field(default_factory=list)
    data_edges: list[DataEdge] = Field(default_factory=list)


class NodeTraceData(BaseModel):
    node_id: str
    status: str
    started_at: float = 0.0
    finished_at: float = 0.0
    duration_ms: float = 0.0
    condition_result: dict[str, Any] | None = None
    error_message: str | None = None
    has_inputs: bool = False
    has_output: bool = False
    assets: list[str] = Field(default_factory=list)
    node_type: str = ""
    service: str | None = None
    method: str | None = None
    script: str | None = None


class TrialData(BaseModel):
    workflow: WorkflowGraph
    nodes: list[NodeTraceData] = Field(default_factory=list)
    total_duration_ms: float = 0.0


class TraceEvent(BaseModel):
    seq: int
    event_type: str
    node_id: str = ""
    node_name: str = ""
    timestamp: float = 0.0
    status: str | None = None
    detail: dict[str, Any] = Field(default_factory=dict)


class TraceEdgeRecord(BaseModel):
    from_name: str
    to_name: str
    from_id: str | None = None
    to_id: str | None = None


class ExecutionLane(BaseModel):
    id: str
    title: str
    agent: str = ""
    column: int = 0
    is_end: bool = False
    executed: bool = False


class ExecutionStep(BaseModel):
    id: str
    state_id: str
    node_id: str
    subgraph_id: str
    lane_id: str
    title: str
    state_type: str
    status: str
    sequence: int = 0
    started_at: float = 0.0
    finished_at: float = 0.0
    duration_ms: float = 0.0
    has_inputs: bool = False
    has_output: bool = False
    error_message: str | None = None
    condition_result: dict[str, Any] | None = None
    assets: list[str] = Field(default_factory=list)


class ExecutionTransition(BaseModel):
    id: str
    source_step_id: str
    target_step_id: str
    kind: str  # "state" | "subgraph" | "sequence"
    label: str = ""


class ExecutionTrace(BaseModel):
    lanes: list[ExecutionLane] = Field(default_factory=list)
    steps: list[ExecutionStep] = Field(default_factory=list)
    transitions: list[ExecutionTransition] = Field(default_factory=list)
    events: list[TraceEvent] = Field(default_factory=list)
    total_duration_ms: float = 0.0
    degraded: bool = False


class ProvenanceEdgeView(BaseModel):
    id: str
    source_state_id: str
    source_step_id: str | None = None
    source_subgraph_id: str
    source_port: str
    target_state_id: str
    target_step_id: str | None = None
    target_subgraph_id: str
    target_port: str
    ref_path: str
    type_label: str = ""
    cross_subgraph: bool = False


class DataProvenance(BaseModel):
    edges: list[ProvenanceEdgeView] = Field(default_factory=list)


class VizMeta(BaseModel):
    total_duration_ms: float = 0.0
    degraded_replay: bool = False
    trial_path: str | None = None
    #: True when the trial recorded a scene_log/ directory (TrialLogger),
    #: i.e. the 3D replay view can actually be served for this run.
    has_scene_log: bool = False


class VizTrial(BaseModel):
    meta: VizMeta = Field(default_factory=VizMeta)
    workflow: WorkflowGraph
    execution: ExecutionTrace = Field(default_factory=ExecutionTrace)
    provenance: DataProvenance = Field(default_factory=DataProvenance)
