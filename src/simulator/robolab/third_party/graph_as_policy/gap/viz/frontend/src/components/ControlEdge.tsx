import { memo } from "react";
import { getSmoothStepPath, type EdgeProps } from "@xyflow/react";
import { getControlEdgeColor } from "../styles/theme";

export interface ControlEdgeData {
  edge_type: "success" | "failure" | "transition";
  label: string;
  [key: string]: unknown;
}

export const ControlEdge = memo(function ControlEdge({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  data,
  selected,
}: EdgeProps) {
  const { edge_type, label } = (data ?? {}) as ControlEdgeData;
  const color = getControlEdgeColor(edge_type);
  const isTransition = edge_type === "transition";
  const isDashed = edge_type === "failure";

  const [path, labelX, labelY] = getSmoothStepPath({
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourcePosition,
    targetPosition,
    borderRadius: 8,
  });

  const markerId = `arrow-${id}`;
  const strokeWidth = isTransition ? 2 : 1.5;

  return (
    <g>
      {/* Arrow marker definition */}
      <defs>
        <marker
          id={markerId}
          markerWidth="8"
          markerHeight="6"
          refX="7"
          refY="3"
          orient="auto"
          markerUnits="strokeWidth"
        >
          <path d="M0,0 L8,3 L0,6" fill={color} />
        </marker>
      </defs>

      {/* Edge line */}
      <path
        d={path}
        fill="none"
        stroke={color}
        strokeWidth={selected ? strokeWidth + 0.5 : strokeWidth}
        strokeOpacity={selected ? 1 : 0.75}
        strokeDasharray={isDashed ? "6 4" : undefined}
        markerEnd={`url(#${markerId})`}
      />

      {/* Label for transition edges */}
      {isTransition && label && (
        <g>
          <rect
            x={labelX - label.length * 4.6 - 8}
            y={labelY - 13}
            width={label.length * 9.2 + 16}
            height={24}
            rx={6}
            fill="#0f172a"
            fillOpacity={0.92}
            stroke={color}
            strokeWidth={1}
            strokeOpacity={0.55}
          />
          <text
            x={labelX}
            y={labelY + 5}
            fill={color}
            fontSize={15}
            fontFamily="var(--font-sans)"
            fontWeight={600}
            textAnchor="middle"
            style={{ pointerEvents: "none" }}
          >
            {label}
          </text>
        </g>
      )}
    </g>
  );
});
