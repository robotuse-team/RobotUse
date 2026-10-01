import { memo } from "react";
import { Handle, Position, type NodeProps } from "@xyflow/react";
import type { SubgraphSchema } from "../types/workflow";
import { COLORS, getAgentColor } from "../styles/theme";

export interface SubgraphNodeData {
  subgraph: SubgraphSchema;
  expanded: boolean;
  [key: string]: unknown;
}

/** Compact size used by ELK when the subgraph is collapsed. */
export const COMPACT_WIDTH = 300;
export const COMPACT_HEIGHT = 86;
export const END_WIDTH = 220;
export const END_HEIGHT = 68;

export const SubgraphNode = memo(function SubgraphNode({ data }: NodeProps) {
  const { subgraph, expanded } = data as SubgraphNodeData;

  // ── End subgraph (pill) ─────────────────────────────────────
  if (subgraph.is_end) {
    const color = subgraph.end_status === "success" ? COLORS.endSuccess : COLORS.endFailure;
    return (
      <div style={{
        width: END_WIDTH,
        height: END_HEIGHT,
        background: "rgba(24, 42, 68, 0.94)",
        border: `2px dashed ${color}`,
        borderRadius: 28,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        fontFamily: "var(--font-sans)",
      }}>
        <Handle type="target" position={Position.Left} id="ctrl_in" style={invisibleHandle} />
        <div style={{ fontSize: 19, fontWeight: 600, color: COLORS.titleText }}>
          {subgraph.subgraph_id}
        </div>
      </div>
    );
  }

  const agentColor = getAgentColor(subgraph.agent);
  const inputNames = Object.keys(subgraph.inputs);
  const outputNames = Object.keys(subgraph.outputs);

  // ── Expanded subgraph (container with header) ───────────────
  if (expanded) {
    return (
      <div style={{
        width: "100%",
        height: "100%",
        background: "rgba(255,255,255,0.04)",
        border: `1px solid ${COLORS.subgraphBorder}`,
        borderLeft: `5px solid ${agentColor}`,
        borderRadius: 14,
        fontFamily: "var(--font-sans)",
        overflow: "visible",
      }}>
        <Handle type="target" position={Position.Left} id="ctrl_in" style={invisibleHandle} />
        <Handle type="source" position={Position.Right} id="ctrl_out" style={invisibleHandle} />
        {inputNames.map((name, i) => (
          <Handle key={`di_${name}`} type="target" position={Position.Left} id={`data_in_${name}`}
            style={{ ...portDot, top: 60 + i * 22, background: COLORS.dataCross }} />
        ))}
        {outputNames.map((name, i) => (
          <Handle key={`do_${name}`} type="source" position={Position.Right} id={`data_out_${name}`}
            style={{ ...portDot, top: 60 + i * 22, background: COLORS.dataCross }} />
        ))}
        <div style={{
          padding: "12px 18px",
          borderBottom: `1px solid ${COLORS.subgraphBorder}`,
        }}>
          <span style={{ fontSize: 19, fontWeight: 700, color: COLORS.titleText }}>
            {subgraph.subgraph_id}
          </span>
        </div>
      </div>
    );
  }

  // ── Compact subgraph (overview pill — click to expand) ─────────
  return (
    <div style={{
      width: COMPACT_WIDTH,
      height: COMPACT_HEIGHT,
      background: "linear-gradient(135deg, #2a3f66 0%, #1f304f 100%)",
      border: "1px solid rgba(180, 205, 245, 0.34)",
      borderLeft: `5px solid ${agentColor}`,
      borderRadius: 14,
      fontFamily: "var(--font-sans)",
      display: "flex",
      alignItems: "center",
      padding: "0 18px",
      gap: 12,
      cursor: "pointer",
      boxShadow: "0 6px 18px rgba(0, 0, 0, 0.32)",
      overflow: "visible",
      position: "relative",
    }}>
      <Handle type="target" position={Position.Left} id="ctrl_in" style={invisibleHandle} />
      <Handle type="source" position={Position.Right} id="ctrl_out" style={invisibleHandle} />
      {inputNames.map((name, i) => (
        <Handle key={`di_${name}`} type="target" position={Position.Left} id={`data_in_${name}`}
          style={{
            ...portDot,
            top: inputNames.length === 1 ? "50%" : `${25 + i * (50 / Math.max(inputNames.length - 1, 1))}%`,
            background: COLORS.dataCross,
          }} />
      ))}
      {outputNames.map((name, i) => (
        <Handle key={`do_${name}`} type="source" position={Position.Right} id={`data_out_${name}`}
          style={{
            ...portDot,
            top: outputNames.length === 1 ? "50%" : `${25 + i * (50 / Math.max(outputNames.length - 1, 1))}%`,
            background: COLORS.dataCross,
          }} />
      ))}
      <span style={{ flex: 1, fontSize: 20, fontWeight: 700, color: COLORS.titleText }}>
        {subgraph.subgraph_id}
      </span>
      <span style={{
        fontSize: 24, fontWeight: 600, color: agentColor,
        opacity: 0.9, lineHeight: 1,
      }}>
        +
      </span>
    </div>
  );
});

const invisibleHandle: React.CSSProperties = {
  width: 6,
  height: 6,
  background: "transparent",
  border: "none",
  opacity: 0,
};

const portDot: React.CSSProperties = {
  width: 7,
  height: 7,
  borderRadius: 4,
  border: "1px solid rgba(255,255,255,0.3)",
};
