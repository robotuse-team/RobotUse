import { useCallback, useEffect, useMemo, useState } from "react";
import {
  getAssetUrl,
  getNodeAssets,
  getNodeInputs,
  getNodeOutput,
  getNodeRequest,
  getNodeScript,
  getNodeSubcalls,
  getSubcallAssetUrl,
  getSubcallRequest,
  getSubcallResponse,
  replayNode,
} from "../api/client";
import type { SubCall } from "../api/client";
import type { ProvenanceEdgeView, Selection, StateSchema, VizTrial } from "../types/viz";
import { getStateTypeTone, getStatusTone } from "../styles/vizTheme";
import { JsonTree } from "./JsonTree";

type DataTab = "summary" | "inputs" | "output" | "request" | "script" | "calls";

function ImageLightbox({ src, alt, onClose }: { src: string; alt: string; onClose: () => void }) {
  return (
    <div className="lightbox-backdrop" onClick={onClose}>
      <img className="lightbox-img" src={src} alt={alt} onClick={(e) => e.stopPropagation()} />
      <button type="button" className="lightbox-close" onClick={onClose}>&times;</button>
    </div>
  );
}

function ClickableImage({ src, alt }: { src: string; alt: string }) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <img src={src} alt={alt} style={{ cursor: "zoom-in" }} onClick={() => setOpen(true)} />
      {open && <ImageLightbox src={src} alt={alt} onClose={() => setOpen(false)} />}
    </>
  );
}

function CollapsibleSection({ label, children }: { label: string; children: React.ReactNode }) {
  const [open, setOpen] = useState(false);
  return (
    <div style={{ marginTop: 8 }}>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        style={{
          display: "flex", alignItems: "center", gap: 6, width: "100%",
          padding: "6px 0", background: "none", border: "none", cursor: "pointer",
          fontSize: 11, fontWeight: 600, opacity: 0.6, color: "var(--text)",
          fontFamily: "inherit", textAlign: "left",
        }}
      >
        <span style={{ fontSize: 10 }}>{open ? "▾" : "▸"}</span>
        {label}
      </button>
      {open && children}
    </div>
  );
}

interface Props {
  viz: VizTrial;
  selected: Selection | null;
  trialPath?: string;
  onSelect: (selection: Selection) => void;
}

export function InspectorPanel({ viz, selected, trialPath, onSelect }: Props) {
  const stateMap = useMemo(
    () => new Map(viz.workflow.states.map((state) => [state.state_id, state])),
    [viz.workflow.states],
  );
  const subgraphMap = useMemo(
    () => new Map(viz.workflow.subgraphs.map((subgraph) => [subgraph.subgraph_id, subgraph])),
    [viz.workflow.subgraphs],
  );
  const stepMap = useMemo(
    () => new Map(viz.execution.steps.map((step) => [step.id, step])),
    [viz.execution.steps],
  );

  const resolved = useMemo(() => {
    if (!selected) return null;

    if (selected.kind === "subgraph") {
      const subgraph = subgraphMap.get(selected.id);
      if (!subgraph) return null;
      return {
        key: `subgraph:${selected.id}`,
        title: subgraph.subgraph_id,
        subtitle: subgraph.is_end ? `end · ${subgraph.end_status}` : subgraph.agent,
        summary: subgraph,
        stateId: undefined as string | undefined,
      };
    }

    if (selected.kind === "state") {
      const state = stateMap.get(selected.id);
      if (!state) return null;
      return {
        key: `state:${selected.id}`,
        title: state.name,
        subtitle: state.state_type,
        state,
        nodeId: state.state_id,
        summary: state,
        stateId: state.state_id,
      };
    }

    if (selected.kind === "step") {
      const step = stepMap.get(selected.id);
      const state = step ? stateMap.get(step.state_id) : null;
      if (!step || !state) return null;
      return {
        key: `step:${selected.id}`,
        title: step.title,
        subtitle: `${step.state_type} · ${step.status}`,
        state,
        step,
        nodeId: step.node_id,
        summary: { ...step, state },
        stateId: state.state_id,
      };
    }

    return null;
  }, [selected, stateMap, stepMap, subgraphMap]);

  const [tab, setTab] = useState<DataTab>("summary");
  const [inputs, setInputs] = useState<Record<string, unknown> | null>(null);
  const [output, setOutput] = useState<Record<string, unknown> | null>(null);
  const [request, setRequest] = useState<Record<string, unknown> | null>(null);
  const [script, setScript] = useState<string | null>(null);
  const [assets, setAssets] = useState<string[]>([]);
  const [subcalls, setSubcalls] = useState<SubCall[] | null>(null);
  const [replayResult, setReplayResult] = useState<Record<string, unknown> | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    setTab("summary");
    setInputs(null);
    setOutput(null);
    setRequest(null);
    setScript(null);
    setAssets([]);
    setSubcalls(null);
    setReplayResult(null);
  }, [resolved?.key]);

  const loadTab = useCallback(async (nextTab: DataTab) => {
    setTab(nextTab);
    if (!resolved?.nodeId || !resolved.state) return;

    setLoading(true);
    try {
      if (nextTab === "inputs" && !inputs) {
        setInputs(await getNodeInputs(resolved.nodeId, trialPath));
      }
      if (nextTab === "output" && !output) {
        setOutput(await getNodeOutput(resolved.nodeId, trialPath));
      }
      if (nextTab === "request" && !request && resolved.state.state_type === "service") {
        setRequest(await getNodeRequest(resolved.nodeId, trialPath));
      }
      if (nextTab === "script" && !script && (resolved.state.script || resolved.state.skill)) {
        setScript(await getNodeScript(resolved.nodeId, trialPath));
      }
      if ((nextTab === "inputs" || nextTab === "output") && assets.length === 0) {
        try {
          setAssets(await getNodeAssets(resolved.nodeId, trialPath));
        } catch {
          setAssets([]);
        }
      }
      if (nextTab === "calls" && !subcalls) {
        try {
          setSubcalls(await getNodeSubcalls(resolved.nodeId, trialPath));
        } catch {
          setSubcalls([]);
        }
      }
    } catch {
      // Missing per-node runtime artifacts are expected for some selections.
    } finally {
      setLoading(false);
    }
  }, [assets.length, inputs, output, request, resolved, script, trialPath]);

  if (!resolved) {
    return (
      <aside className="inspector-panel">
        <div className="inspector-panel__empty">
          <h3>No selection</h3>
          <p>Select a subgraph, state, or executed step to inspect structure, runtime data, and artifacts.</p>
        </div>
      </aside>
    );
  }

  const tabs = buildTabs(resolved.state);
  const imageAssets = assets.filter((asset) => asset.endsWith(".png") || asset.endsWith(".jpg") || asset.endsWith(".jpeg"));

  return (
    <aside className="inspector-panel">
      <div className="inspector-panel__header">
        <div>
          <h2>{resolved.title}</h2>
          <p>{resolved.subtitle}</p>
        </div>
        {resolved.state && (
          <span className="inspector-panel__badge" style={{ color: getStateTypeTone(resolved.state.state_type) }}>
            {resolved.state.state_type}
          </span>
        )}
        {resolved.step && (
          <span className="inspector-panel__badge" style={{ color: getStatusTone(resolved.step.status) }}>
            {resolved.step.status}
          </span>
        )}
      </div>

      {tabs.length > 1 && (
        <div className="inspector-tabs">
          {tabs.map((item) => (
            <button
              key={item}
              type="button"
              className={`inspector-tabs__tab ${tab === item ? "is-active" : ""}`}
              onClick={() => { void loadTab(item); }}
            >
              {item}
            </button>
          ))}
        </div>
      )}

      <div className="inspector-panel__body">
        {loading && <div className="inspector-panel__loading">Loading…</div>}
        {tab === "summary" && (
          <SummaryTab
            resolved={resolved}
            provenance={viz.provenance.edges}
            stateMap={stateMap}
            onSelect={onSelect}
          />
        )}
        {tab === "inputs" && resolved.state && (
          <RuntimeTab
            data={inputs}
            fallback={resolved.state.inputs_def}
            fallbackLabel="Input bindings (from workflow)"
            emptyLabel="No inputs"
            images={imageAssets.filter((asset) => asset.includes("input"))}
            nodeId={resolved.nodeId}
            trialPath={trialPath}
          />
        )}
        {tab === "output" && resolved.state && (
          <RuntimeTab
            data={output}
            fallback={resolved.state.output_ports.reduce<Record<string, string>>((acc, port) => {
              acc[port.name] = port.type_label;
              return acc;
            }, {})}
            fallbackLabel="Output ports (from schema)"
            emptyLabel="No outputs"
            images={imageAssets.filter((asset) => asset.includes("output"))}
            nodeId={resolved.nodeId}
            trialPath={trialPath}
          />
        )}
        {tab === "request" && resolved.state && (
          <div className="runtime-tab">
            {request ? <JsonTree data={request} /> : <div className="empty-state">No request data</div>}
            {resolved.step && resolved.state.state_type === "service" && (
              <div className="runtime-tab__replay">
                <button
                  type="button"
                  className="action-button"
                  onClick={async () => {
                    if (!resolved.nodeId) return;
                    setLoading(true);
                    try {
                      setReplayResult(await replayNode(resolved.nodeId, trialPath));
                    } catch (error) {
                      setReplayResult({ ok: false, error: String(error) });
                    } finally {
                      setLoading(false);
                    }
                  }}
                >
                  Replay request
                </button>
                {replayResult && <JsonTree data={replayResult} />}
              </div>
            )}
          </div>
        )}
        {tab === "script" && resolved.state && (
          <div className="runtime-tab">
            {script ? (
              <pre className="script-block">{script}</pre>
            ) : (
              <div className="empty-state">No source available</div>
            )}
          </div>
        )}
        {tab === "calls" && resolved.state && (
          <SubCallsTab
            subcalls={subcalls}
            nodeId={resolved.nodeId}
            trialPath={trialPath}
          />
        )}
      </div>
    </aside>
  );
}

function buildTabs(state: StateSchema | undefined): DataTab[] {
  if (!state) return ["summary"];
  const tabs: DataTab[] = ["summary", "inputs", "output"];
  if (state.state_type === "service") tabs.push("request");
  if (state.state_type === "script" || state.state_type === "skill") {
    tabs.push("calls");
    tabs.push("script");
  }
  return tabs;
}

function SubCallsTab({
  subcalls,
  nodeId,
  trialPath,
}: {
  subcalls: SubCall[] | null;
  nodeId?: string;
  trialPath?: string;
}) {
  const [expanded, setExpanded] = useState<Record<number, boolean>>({});
  const [callData, setCallData] = useState<Record<number, { request?: Record<string, unknown>; response?: Record<string, unknown> }>>({});

  const toggle = useCallback(async (seq: number) => {
    setExpanded((prev) => ({ ...prev, [seq]: !prev[seq] }));
    if (!callData[seq] && nodeId) {
      try {
        const [req, res] = await Promise.all([
          getSubcallRequest(nodeId, seq, trialPath),
          getSubcallResponse(nodeId, seq, trialPath),
        ]);
        setCallData((prev) => ({ ...prev, [seq]: { request: req, response: res } }));
      } catch {
        setCallData((prev) => ({ ...prev, [seq]: {} }));
      }
    }
  }, [callData, nodeId, trialPath]);

  if (!subcalls || subcalls.length === 0) {
    return <div className="runtime-tab"><div className="empty-state">No service calls recorded</div></div>;
  }

  return (
    <div className="runtime-tab">
      {subcalls.map((sc) => {
        const shortService = sc.service.split(".").pop() ?? sc.service;
        const isOpen = expanded[sc.seq] ?? false;
        const data = callData[sc.seq];
        const imageAssets = sc.assets.filter((a) => a.endsWith(".png") || a.endsWith(".jpg"));

        return (
          <div key={sc.seq} className="subcall-card" style={{ border: "1px solid var(--border)", borderRadius: 6, marginBottom: 8 }}>
            <button
              type="button"
              onClick={() => { void toggle(sc.seq); }}
              style={{
                display: "flex", alignItems: "center", gap: 8, width: "100%",
                padding: "8px 12px", background: "none", border: "none", cursor: "pointer",
                fontFamily: "inherit", fontSize: 13, textAlign: "left",
                color: "var(--text-primary)",
              }}
            >
              <span style={{ fontFamily: "monospace", opacity: 0.5 }}>{sc.seq}</span>
              <strong>{shortService}/{sc.method}</strong>
              <span style={{ marginLeft: "auto", opacity: 0.4 }}>{isOpen ? "▾" : "▸"}</span>
            </button>
            {isOpen && (
              <div style={{ padding: "0 12px 12px" }}>
                {nodeId && imageAssets.length > 0 && (
                  <div className="runtime-tab__images" style={{ marginBottom: 8 }}>
                    {imageAssets.map((asset) => (
                      <figure key={asset}>
                        <ClickableImage src={getSubcallAssetUrl(nodeId, sc.seq, asset, trialPath)} alt={asset} />
                        <figcaption>{asset}</figcaption>
                      </figure>
                    ))}
                  </div>
                )}
                {data?.request && (
                  <CollapsibleSection label="Request">
                    <JsonTree data={data.request} />
                  </CollapsibleSection>
                )}
                {data?.response && (
                  <CollapsibleSection label="Response">
                    <JsonTree data={data.response} />
                  </CollapsibleSection>
                )}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

function SummaryTab({
  resolved,
  provenance,
  stateMap,
  onSelect,
}: {
  resolved: {
    state?: StateSchema;
    step?: {
      duration_ms: number;
      assets: string[];
      sequence: number;
      status: string;
    };
    stateId?: string;
  };
  provenance: ProvenanceEdgeView[];
  stateMap: Map<string, StateSchema>;
  onSelect: (selection: Selection) => void;
}) {
  const inputsFrom = useMemo(() => {
    if (!resolved.stateId) return [];
    return provenance.filter((edge) => edge.target_state_id === resolved.stateId);
  }, [provenance, resolved.stateId]);

  const outputsTo = useMemo(() => {
    if (!resolved.stateId) return [];
    return provenance.filter((edge) => edge.source_state_id === resolved.stateId);
  }, [provenance, resolved.stateId]);

  return (
    <div className="summary-tab">
      {resolved.step && (
        <div className="summary-grid">
          <Metric label="Sequence" value={`#${resolved.step.sequence}`} />
          <Metric label="Duration" value={formatDuration(resolved.step.duration_ms)} />
          <Metric label="Assets" value={String(resolved.step.assets.length)} />
        </div>
      )}
      {resolved.stateId && (inputsFrom.length > 0 || outputsTo.length > 0) && (
        <div className="provenance-inline">
          {inputsFrom.length > 0 && (
            <div className="provenance-inline__section">
              <span className="provenance-inline__label">Inputs sourced from</span>
              {inputsFrom.map((edge) => (
                <ProvenanceRow
                  key={edge.id}
                  edge={edge}
                  stateMap={stateMap}
                  target="source"
                  onSelect={onSelect}
                />
              ))}
            </div>
          )}
          {outputsTo.length > 0 && (
            <div className="provenance-inline__section">
              <span className="provenance-inline__label">Outputs consumed by</span>
              {outputsTo.map((edge) => (
                <ProvenanceRow
                  key={edge.id}
                  edge={edge}
                  stateMap={stateMap}
                  target="target"
                  onSelect={onSelect}
                />
              ))}
            </div>
          )}
        </div>
      )}
      {!resolved.step && (!resolved.stateId || (inputsFrom.length === 0 && outputsTo.length === 0)) && (
        <div className="empty-state">Click <strong>inputs</strong> or <strong>output</strong> above to see runtime data.</div>
      )}
    </div>
  );
}

function ProvenanceRow({
  edge,
  stateMap,
  target,
  onSelect,
}: {
  edge: ProvenanceEdgeView;
  stateMap: Map<string, StateSchema>;
  target: "source" | "target";
  onSelect: (selection: Selection) => void;
}) {
  const otherStateId = target === "source" ? edge.source_state_id : edge.target_state_id;
  const otherState = stateMap.get(otherStateId);
  const otherPort = target === "source" ? edge.source_port : edge.target_port;
  const localPort = target === "source" ? edge.target_port : edge.source_port;
  const otherStepId = target === "source" ? edge.source_step_id : edge.target_step_id;
  return (
    <button
      type="button"
      className="provenance-inline__row"
      onClick={() => {
        if (otherStepId) onSelect({ kind: "step", id: otherStepId });
        else onSelect({ kind: "state", id: otherStateId });
      }}
    >
      <span className="provenance-inline__port">{localPort}</span>
      <span className="provenance-inline__arrow">{target === "source" ? "←" : "→"}</span>
      <span>{otherState?.name ?? otherStateId}</span>
      <span className="provenance-inline__port">.{otherPort}</span>
      <span className="provenance-inline__path">{edge.ref_path}</span>
    </button>
  );
}

function RuntimeTab({
  data,
  fallback,
  fallbackLabel,
  emptyLabel,
  images,
  nodeId,
  trialPath,
}: {
  data: Record<string, unknown> | null;
  fallback: Record<string, unknown>;
  fallbackLabel: string;
  emptyLabel: string;
  images: string[];
  nodeId?: string;
  trialPath?: string;
}) {
  return (
    <div className="runtime-tab">
      {data ? (
        <JsonTree data={data} />
      ) : Object.keys(fallback).length > 0 ? (
        <>
          <div className="runtime-tab__caption">{fallbackLabel}</div>
          <JsonTree data={fallback} />
        </>
      ) : (
        <div className="empty-state">{emptyLabel}</div>
      )}
      {nodeId && images.length > 0 && (
        <div className="runtime-tab__images">
          {images.map((asset) => (
            <figure key={asset}>
              <ClickableImage src={getAssetUrl(nodeId, asset, trialPath)} alt={asset} />
              <figcaption>{asset}</figcaption>
            </figure>
          ))}
        </div>
      )}
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric-card">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function formatDuration(ms: number): string {
  if (!ms) return "–";
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
}
