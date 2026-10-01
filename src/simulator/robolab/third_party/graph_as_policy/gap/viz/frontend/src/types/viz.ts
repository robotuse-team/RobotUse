export type {
  PortSchema,
  SubgraphSchema,
  StateSchema,
  WorkflowGraph,
} from "./workflow";

export interface TraceEvent {
  seq: number;
  event_type: string;
  node_id: string;
  node_name: string;
  timestamp: number;
  status: string | null;
  detail: Record<string, unknown>;
}

export interface ExecutionLane {
  id: string;
  title: string;
  agent: string;
  column: number;
  is_end: boolean;
  executed: boolean;
}

export interface ExecutionStep {
  id: string;
  state_id: string;
  node_id: string;
  subgraph_id: string;
  lane_id: string;
  title: string;
  state_type: string;
  status: string;
  sequence: number;
  started_at: number;
  finished_at: number;
  duration_ms: number;
  has_inputs: boolean;
  has_output: boolean;
  error_message: string | null;
  condition_result: {
    actual?: unknown;
    expected?: unknown;
    met?: boolean;
  } | null;
  assets: string[];
}

export interface ExecutionTransition {
  id: string;
  source_step_id: string;
  target_step_id: string;
  kind: "state" | "subgraph" | "sequence";
  label: string;
}

export interface ExecutionTrace {
  lanes: ExecutionLane[];
  steps: ExecutionStep[];
  transitions: ExecutionTransition[];
  events: TraceEvent[];
  total_duration_ms: number;
  degraded: boolean;
}

export interface ProvenanceEdgeView {
  id: string;
  source_state_id: string;
  source_step_id: string | null;
  source_subgraph_id: string;
  source_port: string;
  target_state_id: string;
  target_step_id: string | null;
  target_subgraph_id: string;
  target_port: string;
  ref_path: string;
  type_label: string;
  cross_subgraph: boolean;
}

export interface DataProvenance {
  edges: ProvenanceEdgeView[];
}

export interface VizMeta {
  total_duration_ms: number;
  degraded_replay: boolean;
  trial_path: string | null;
  /** True when the trial recorded a scene_log/ dir — 3D replay is servable. */
  has_scene_log: boolean;
}

export interface VizTrial {
  meta: VizMeta;
  workflow: import("./workflow").WorkflowGraph;
  execution: ExecutionTrace;
  provenance: DataProvenance;
}

export type SelectionKind = "subgraph" | "state" | "step";

export interface Selection {
  kind: SelectionKind;
  id: string;
}
