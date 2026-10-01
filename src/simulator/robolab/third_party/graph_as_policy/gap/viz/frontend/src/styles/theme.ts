/** Color palette for the v2 dual-graph visualization. */
export const COLORS = {
  // Canvas
  canvasBg: "#16213e",

  // State node types
  serviceFill: "#1a3a5c",
  serviceBody: "#264d73",
  scriptFill: "#5c1a6e",
  scriptBody: "#7a3d8f",
  skillFill: "#1a5c5c",
  skillBody: "#2d7a7a",

  // Control flow edges
  controlSuccess: "#27ae60",
  controlFailure: "#e74c3c",
  controlTransition: "#a855f7",

  // Data flow edges
  dataMessage: "#e67e22",
  dataScalar: "#5dade2",
  dataRepeated: "#e74c3c",
  dataCross: "#c084fc",

  // End states / subgraphs
  endSuccess: "#22c55e",
  endFailure: "#ef4444",

  // Subgraph containers
  subgraphBg: "rgba(255, 255, 255, 0.03)",
  subgraphBorder: "rgba(255, 255, 255, 0.12)",

  // UI
  nodeBorder: "#3a3a5a",
  titleText: "#ffffff",
  fieldText: "#d0d0d0",
  typeText: "#8899aa",
  statusOk: "#27ae60",
  statusError: "#e74c3c",
  statusSkipped: "#95a5a6",
  statusRunning: "#f39c12",
} as const;

export function getStateColors(stateType: string): { header: string; body: string } {
  if (stateType === "skill") return { header: COLORS.skillFill, body: COLORS.skillBody };
  if (stateType === "script" || stateType === "router") return { header: COLORS.scriptFill, body: COLORS.scriptBody };
  // "tool" (and legacy "service") states share the service palette.
  return { header: COLORS.serviceFill, body: COLORS.serviceBody };
}

export function getControlEdgeColor(edgeType: string): string {
  if (edgeType === "success") return COLORS.controlSuccess;
  if (edgeType === "failure") return COLORS.controlFailure;
  return COLORS.controlTransition;
}

export function getDataEdgeColor(crossSubgraph: boolean, typeLabel: string): string {
  if (crossSubgraph) return COLORS.dataCross;
  if (typeLabel.startsWith("[]")) return COLORS.dataRepeated;
  if (typeLabel && /^[A-Z]/.test(typeLabel)) return COLORS.dataMessage;
  return COLORS.dataScalar;
}

export function getPortColor(port: { is_message: boolean; is_repeated: boolean }): string {
  if (port.is_repeated) return COLORS.dataRepeated;
  if (port.is_message) return COLORS.dataMessage;
  return COLORS.dataScalar;
}

export function getStatusColor(status: string): string {
  switch (status) {
    case "ok": return COLORS.statusOk;
    case "error": return COLORS.statusError;
    case "skipped": return COLORS.statusSkipped;
    case "running": return COLORS.statusRunning;
    default: return COLORS.nodeBorder;
  }
}

const AGENT_COLORS: Record<string, string> = {
  perception: "#2980b9",
  grasp: "#c0392b",
  transport: "#27ae60",
  verify: "#8e44ad",
  generic: "#7f8c8d",
};

export function getAgentColor(agentName: string): string {
  return AGENT_COLORS[agentName] ?? AGENT_COLORS.generic;
}
