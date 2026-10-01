import type { Selection } from "../types/viz";

export const VIZ = {
  bg: "#06121f",
  bgGlow: "#123252",
  panel: "rgba(8, 22, 36, 0.82)",
  panelStrong: "rgba(9, 26, 43, 0.96)",
  border: "rgba(154, 194, 255, 0.18)",
  borderStrong: "rgba(154, 194, 255, 0.32)",
  text: "#f5f7fb",
  textMuted: "#91a3ba",
  textSoft: "#6f8198",
  accent: "#66d9ef",
  accentWarm: "#ffaf69",
  accentHot: "#ff6e6e",
  success: "#59d48a",
  error: "#ff5d73",
  running: "#f2c14e",
  skipped: "#64748b",
  data: "#84a9ff",
  crossData: "#feb47b",
  laneFill: "rgba(20, 39, 61, 0.88)",
  selection: "rgba(102, 217, 239, 0.24)",
  hover: "rgba(132, 169, 255, 0.18)",
} as const;

export function getStatusTone(status: string): string {
  switch (status) {
    case "ok":
      return VIZ.success;
    case "error":
      return VIZ.error;
    case "running":
      return VIZ.running;
    case "skipped":
      return VIZ.skipped;
    default:
      return VIZ.textSoft;
  }
}

export function getStateTypeTone(stateType: string): string {
  switch (stateType) {
    case "service":
      return "#4cb4ff";
    case "script":
      return "#ff9f6e";
    case "skill":
      return "#6be4b8";
    case "parallel":
      return "#d8b4fe";
    case "end":
      return "#f6d365";
    default:
      return VIZ.textSoft;
  }
}

export function getAgentTone(agent: string): string {
  const tones: Record<string, string> = {
    perception: "#5bd4ff",
    grasp: "#ff7a59",
    transport: "#a2ff7b",
    verify: "#f8d66d",
    generic: "#9fb6d9",
  };
  return tones[agent] ?? tones.generic;
}

export function selectionMatches(
  selection: Selection | null,
  kind: Selection["kind"],
  id: string,
): boolean {
  return selection?.kind === kind && selection.id === id;
}
