export interface PortSchema {
  name: string;
  type_label: string;
  is_message: boolean;
  is_repeated: boolean;
}

export interface RecoveryAction {
  /** Flat tool name to invoke best-effort on a failure end node. */
  tool: string;
}

export interface SubgraphSchema {
  subgraph_id: string;
  agent: string;
  inputs: Record<string, string>;
  outputs: Record<string, string>;
  begin_state: string;
  is_end: boolean;
  end_status: string | null;
  recovery: RecoveryAction[];
}

export interface StateSchema {
  state_id: string;
  name: string;
  subgraph_id: string;
  state_type: "tool" | "script" | "router" | "end" | "service" | "skill" | "parallel";
  service: string | null;
  method: string | null;
  script: string | null;
  /** Flat tool name for `type: tool` states (legacy traces: skill name). */
  skill: string | null;
  input_ports: PortSchema[];
  output_ports: PortSchema[];
  inputs_def: Record<string, unknown>;
}

export interface ControlEdge {
  source: string;
  target: string;
  edge_type: "success" | "failure" | "transition";
  label: string;
}

export interface DataEdge {
  source: string;
  source_field: string;
  target: string;
  target_field: string;
  ref_path: string;
  cross_subgraph: boolean;
  type_label: string;
  source_sg_port: string;
  target_sg_port: string;
}

export interface WorkflowGraph {
  meta: Record<string, unknown>;
  begin: string;
  subgraphs: SubgraphSchema[];
  states: StateSchema[];
  control_edges: ControlEdge[];
  data_edges: DataEdge[];
}

export interface NodeTraceData {
  node_id: string;
  status: "pending" | "running" | "ok" | "error" | "skipped";
  started_at: number;
  finished_at: number;
  duration_ms: number;
  condition_result: {
    actual: unknown;
    expected: unknown;
    met: boolean;
  } | null;
  error_message: string | null;
  has_inputs: boolean;
  has_output: boolean;
  assets: string[];
}

export interface TrialData {
  workflow: WorkflowGraph;
  nodes: NodeTraceData[];
  total_duration_ms: number;
}
