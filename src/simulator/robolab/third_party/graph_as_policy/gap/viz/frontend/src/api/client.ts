import type { VizTrial } from "../types/viz";

const BASE = "/api";

function trialParam(trialPath?: string): string {
  if (!trialPath) return "";
  return `?trial=${encodeURIComponent(trialPath)}`;
}

async function fetchJson<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);
  return res.json();
}

export async function getTrials(): Promise<string[]> {
  return fetchJson(`${BASE}/trials`);
}

export async function getVizTrial(trialPath?: string): Promise<VizTrial> {
  return fetchJson(`${BASE}/viz/trial${trialParam(trialPath)}`);
}

export async function getNodeInputs(nodeId: string, trialPath?: string): Promise<Record<string, unknown>> {
  return fetchJson(`${BASE}/node/${nodeId}/inputs${trialParam(trialPath)}`);
}

export async function getNodeOutput(nodeId: string, trialPath?: string): Promise<Record<string, unknown>> {
  return fetchJson(`${BASE}/node/${nodeId}/output${trialParam(trialPath)}`);
}

export async function getNodeRequest(nodeId: string, trialPath?: string): Promise<Record<string, unknown>> {
  return fetchJson(`${BASE}/node/${nodeId}/request${trialParam(trialPath)}`);
}

export async function getNodeScript(nodeId: string, trialPath?: string): Promise<string> {
  const res = await fetch(`${BASE}/node/${nodeId}/script${trialParam(trialPath)}`);
  if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);
  return res.text();
}

export async function getNodeAssets(nodeId: string, trialPath?: string): Promise<string[]> {
  return fetchJson(`${BASE}/node/${nodeId}/assets${trialParam(trialPath)}`);
}

export function getAssetUrl(nodeId: string, filename: string, trialPath?: string): string {
  return `${BASE}/node/${nodeId}/asset/${filename}${trialParam(trialPath)}`;
}

// --- Sub-call (skill/script internal service calls) ---

export interface SubCall {
  seq: number;
  dir_name: string;
  /** Flat tool name recorded by the tracer (e.g. "robot.move_to_pose"). */
  tool: string;
  /** Derived display fields: tool name split on its last dot. */
  method: string;
  service: string;
  assets: string[];
}

export async function getNodeSubcalls(nodeId: string, trialPath?: string): Promise<SubCall[]> {
  return fetchJson(`${BASE}/node/${nodeId}/subcalls${trialParam(trialPath)}`);
}

export async function getSubcallRequest(nodeId: string, seq: number, trialPath?: string): Promise<Record<string, unknown>> {
  return fetchJson(`${BASE}/node/${nodeId}/subcall/${seq}/request${trialParam(trialPath)}`);
}

export async function getSubcallResponse(nodeId: string, seq: number, trialPath?: string): Promise<Record<string, unknown>> {
  return fetchJson(`${BASE}/node/${nodeId}/subcall/${seq}/response${trialParam(trialPath)}`);
}

export function getSubcallAssetUrl(nodeId: string, seq: number, filename: string, trialPath?: string): string {
  return `${BASE}/node/${nodeId}/subcall/${seq}/asset/${filename}${trialParam(trialPath)}`;
}

// --- 3D Scene Replay ---

/** FastAPI errors arrive as {"detail": "..."} — surface the human-readable
 * reason instead of the raw status + JSON blob. */
async function errorDetail(res: Response): Promise<string> {
  const text = await res.text();
  try {
    const parsed = JSON.parse(text) as { detail?: unknown };
    if (typeof parsed.detail === "string") return parsed.detail;
  } catch {
    // not JSON — fall through to the raw body
  }
  return `${res.status}: ${text}`;
}

export async function startReplay3d(trialPath?: string): Promise<{ url: string }> {
  const res = await fetch(`${BASE}/trial/replay3d${trialParam(trialPath)}`, { method: "POST" });
  if (!res.ok) throw new Error(await errorDetail(res));
  return res.json();
}

export async function replayNode(nodeId: string, trialPath?: string): Promise<Record<string, unknown>> {
  const res = await fetch(`${BASE}/node/${nodeId}/replay${trialParam(trialPath)}`, { method: "POST" });
  if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);
  return res.json();
}
