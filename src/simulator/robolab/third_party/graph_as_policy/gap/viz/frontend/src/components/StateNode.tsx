import { memo } from "react";
import { Handle, Position, type NodeProps } from "@xyflow/react";
import type { StateSchema, NodeTraceData, PortSchema } from "../types/workflow";
import { COLORS, getStateColors, getPortColor, getStatusColor } from "../styles/theme";

export interface StateNodeData {
  state: StateSchema;
  trace?: NodeTraceData;
  expanded: boolean;
  [key: string]: unknown;
}

/** Height calculation for ELK layout. */
export function stateNodeHeight(state: StateSchema, expanded: boolean): number {
  if (state.state_type === "end") return 52;
  if (!expanded) return 64;
  const totalPorts = Math.max(state.input_ports.length + state.output_ports.length, 1);
  return 70 + totalPorts * 28;
}

/** Width for ELK layout. */
export function stateNodeWidth(state: StateSchema): number {
  if (state.state_type === "end") return 170;
  return 290;
}

export const StateNode = memo(function StateNode({
  data,
}: NodeProps) {
  const { state, trace, expanded } = data as StateNodeData;

  // End state: small terminal marker
  if (state.state_type === "end") {
    return <EndStateMarker state={state} trace={trace} />;
  }

  const { header, body } = getStateColors(state.state_type);
  const hasTrace = !!trace;
  const statusColor = hasTrace ? getStatusColor(trace.status) : undefined;

  // Collapsed view (default — button-like)
  if (!expanded) {
    return (
      <div style={{ ...cardStyle, background: body, height: 64, width: 290 }}>
        <Handle type="target" position={Position.Top} id="ctrl_in" style={ctrlHandleStyle} />
        <Handle type="source" position={Position.Bottom} id="ctrl_out" style={ctrlHandleStyle} />
        {state.input_ports.map((p, i) => (
          <Handle key={`di_${p.name}`} type="target" position={Position.Left} id={`data_in_${p.name}`}
            style={{ ...invisibleHandle, top: `${((i + 1) / (state.input_ports.length + 1)) * 100}%` }} />
        ))}
        {state.output_ports.map((p, i) => (
          <Handle key={`do_${p.name}`} type="source" position={Position.Right} id={`data_out_${p.name}`}
            style={{ ...invisibleHandle, top: `${((i + 1) / (state.output_ports.length + 1)) * 100}%` }} />
        ))}
        <div style={{ display: "flex", alignItems: "center", padding: "0 18px", height: "100%" }}>
          <span style={{ fontSize: 18, fontWeight: 600, color: COLORS.titleText }}>{state.name}</span>
        </div>
      </div>
    );
  }

  // Expanded view
  const portOffset = 48;
  const portGap = 24;

  return (
    <div style={{ ...cardStyle, background: body, width: 280 }}>
      {/* Control flow handles (top/bottom center) */}
      <Handle type="target" position={Position.Top} id="ctrl_in" style={ctrlHandleStyle} />
      <Handle type="source" position={Position.Bottom} id="ctrl_out" style={ctrlHandleStyle} />

      {/* Header */}
      <div
        style={{
          padding: "6px 10px",
          background: header,
          borderRadius: "6px 6px 0 0",
          display: "flex",
          alignItems: "center",
          gap: 6,
        }}
      >
        {statusColor && <div style={{ width: 8, height: 8, borderRadius: 4, background: statusColor, flexShrink: 0 }} />}
        <span style={{ fontSize: 11, fontWeight: 700, color: COLORS.titleText }}>{state.name}</span>
        <TypeBadge type={state.state_type} />
        {hasTrace && trace.duration_ms > 0 && (
          <span style={{ fontSize: 9, color: COLORS.typeText, marginLeft: "auto" }}>
            {formatDuration(trace.duration_ms)}
          </span>
        )}
      </div>

      {/* Subtitle */}
      <div style={{ padding: "4px 10px 2px", fontSize: 9, color: COLORS.typeText }}>
        {state.state_type === "service" && `${state.service?.split(".").pop()}.${state.method}`}
        {(state.state_type === "script" || state.state_type === "router") && state.script?.split("/").pop()}
        {(state.state_type === "tool" || state.state_type === "skill") && state.skill}
      </div>

      {/* Ports */}
      <div style={{ padding: "4px 0 8px" }}>
        {state.input_ports.map((port, i) => (
          <PortRow key={`in_${port.name}`} port={port} side="left" index={i} offset={portOffset} gap={portGap} />
        ))}
        {state.output_ports.map((port, i) => (
          <PortRow key={`out_${port.name}`} port={port} side="right" index={i + state.input_ports.length} offset={portOffset} gap={portGap} />
        ))}
      </div>

      {/* Data flow handles on left/right */}
      {state.input_ports.map((p, i) => (
        <Handle
          key={`di_${p.name}`}
          type="target"
          position={Position.Left}
          id={`data_in_${p.name}`}
          style={{
            ...dataHandleStyle,
            top: portOffset + i * portGap,
            background: getPortColor(p),
          }}
        />
      ))}
      {state.output_ports.map((p, i) => (
        <Handle
          key={`do_${p.name}`}
          type="source"
          position={Position.Right}
          id={`data_out_${p.name}`}
          style={{
            ...dataHandleStyle,
            top: portOffset + (state.input_ports.length + i) * portGap,
            background: getPortColor(p),
          }}
        />
      ))}
    </div>
  );
});

function EndStateMarker({ state }: { state: StateSchema; trace?: NodeTraceData }) {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        height: 52,
        width: 170,
        background: "rgba(255,255,255,0.06)",
        border: `1px solid ${COLORS.nodeBorder}`,
        borderRadius: 26,
        fontFamily: "var(--font-sans)",
      }}
    >
      <Handle type="target" position={Position.Top} id="ctrl_in" style={ctrlHandleStyle} />
      <Handle type="source" position={Position.Bottom} id="ctrl_out" style={ctrlHandleStyle} />
      <span style={{ fontSize: 16, fontWeight: 600, color: COLORS.titleText }}>{state.name}</span>
    </div>
  );
}

function TypeBadge({ type }: { type: string }) {
  const colors: Record<string, string> = {
    tool: COLORS.serviceFill,
    service: COLORS.serviceFill,
    script: COLORS.scriptFill,
    router: "#5c4a1a",
    skill: COLORS.skillFill,
    parallel: "#5c4a1a",
  };
  return (
    <span
      style={{
        fontSize: 8,
        padding: "1px 5px",
        borderRadius: 3,
        background: colors[type] ?? COLORS.nodeBorder,
        color: "#fff",
        fontWeight: 600,
        letterSpacing: 0.5,
        textTransform: "uppercase",
      }}
    >
      {type}
    </span>
  );
}

function PortRow({
  port,
  side,
  index: _index,
  offset: _offset,
  gap: _gap,
}: {
  port: PortSchema;
  side: "left" | "right";
  index: number;
  offset: number;
  gap: number;
}) {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 4,
        padding: "1px 10px",
        justifyContent: side === "left" ? "flex-start" : "flex-end",
      }}
    >
      {side === "left" && (
        <>
          <span style={{ fontSize: 10, color: COLORS.fieldText }}>{port.name}</span>
          <span style={{ fontSize: 9, color: getPortColor(port) }}>{port.type_label}</span>
        </>
      )}
      {side === "right" && (
        <>
          <span style={{ fontSize: 9, color: getPortColor(port) }}>{port.type_label}</span>
          <span style={{ fontSize: 10, color: COLORS.fieldText }}>{port.name}</span>
        </>
      )}
    </div>
  );
}

function formatDuration(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
}

const cardStyle: React.CSSProperties = {
  borderRadius: 6,
  border: `1px solid ${COLORS.nodeBorder}`,
  fontFamily: "monospace",
  overflow: "visible",
};

const ctrlHandleStyle: React.CSSProperties = {
  width: 6,
  height: 6,
  background: "transparent",
  border: "none",
  opacity: 0,
};

const dataHandleStyle: React.CSSProperties = {
  width: 8,
  height: 8,
  borderRadius: 4,
  border: "1px solid rgba(255,255,255,0.4)",
};

const invisibleHandle: React.CSSProperties = {
  width: 6,
  height: 6,
  background: "transparent",
  border: "none",
};
