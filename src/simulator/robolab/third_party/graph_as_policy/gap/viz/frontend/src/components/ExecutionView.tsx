import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ExecutionStep, Selection, VizTrial } from "../types/viz";
import { VIZ, getAgentTone, getStatusTone, selectionMatches } from "../styles/vizTheme";

interface Props {
  viz: VizTrial;
  selected: Selection | null;
  hovered: Selection | null;
  onSelect: (selection: Selection) => void;
  onHover: (selection: Selection | null) => void;
}

const COL_WIDTH = 240;
const COL_GAP = 56;
const HEADER_HEIGHT = 76;
const ROW_HEIGHT = 46;
const ROW_GAP = 10;
const PAD = 28;

export function ExecutionView({ viz, selected, hovered, onSelect, onHover }: Props) {
  const lanes = useMemo(
    () => [...viz.execution.lanes].sort((a, b) => a.column - b.column),
    [viz.execution.lanes],
  );

  const stepsByLane = useMemo(() => {
    const grouped = new Map<string, ExecutionStep[]>();
    for (const step of viz.execution.steps) {
      const list = grouped.get(step.lane_id) ?? [];
      list.push(step);
      grouped.set(step.lane_id, list);
    }
    for (const list of grouped.values()) {
      list.sort((a, b) => a.sequence - b.sequence);
    }
    return grouped;
  }, [viz.execution.steps]);

  const columnIndex = useMemo(
    () => new Map(lanes.map((lane, index) => [lane.id, index])),
    [lanes],
  );

  const geometry = useMemo(() => {
    const boxes = new Map<string, { x: number; y: number; width: number; height: number }>();
    let maxRows = 0;
    for (const lane of lanes) {
      const c = columnIndex.get(lane.id) ?? 0;
      const steps = stepsByLane.get(lane.id) ?? [];
      maxRows = Math.max(maxRows, steps.length);
      steps.forEach((step, row) => {
        boxes.set(step.id, {
          x: PAD + c * (COL_WIDTH + COL_GAP),
          y: PAD + HEADER_HEIGHT + row * (ROW_HEIGHT + ROW_GAP),
          width: COL_WIDTH,
          height: ROW_HEIGHT,
        });
      });
    }
    const totalWidth = Math.max(
      PAD * 2,
      PAD * 2 + lanes.length * COL_WIDTH + Math.max(lanes.length - 1, 0) * COL_GAP,
    );
    const totalHeight = Math.max(
      320,
      PAD * 2 + HEADER_HEIGHT + Math.max(maxRows, 1) * (ROW_HEIGHT + ROW_GAP),
    );
    return { boxes, totalWidth, totalHeight };
  }, [columnIndex, lanes, stepsByLane]);

  const hasSteps = viz.execution.steps.length > 0;

  const scrollRef = useRef<HTMLDivElement>(null);
  const dragStart = useRef<{ x: number; y: number; sx: number; sy: number } | null>(null);
  const [panning, setPanning] = useState(false);

  const onPointerDown = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    if (e.button !== 0) return;
    const target = e.target as HTMLElement;
    if (target.closest("button")) return;
    const el = scrollRef.current;
    if (!el) return;
    dragStart.current = {
      x: e.clientX,
      y: e.clientY,
      sx: el.scrollLeft,
      sy: el.scrollTop,
    };
    setPanning(true);
  }, []);

  useEffect(() => {
    if (!panning) return;
    const onMove = (e: PointerEvent) => {
      const el = scrollRef.current;
      const start = dragStart.current;
      if (!el || !start) return;
      el.scrollLeft = start.sx - (e.clientX - start.x);
      el.scrollTop = start.sy - (e.clientY - start.y);
    };
    const stop = () => {
      dragStart.current = null;
      setPanning(false);
    };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", stop);
    window.addEventListener("pointercancel", stop);
    return () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", stop);
      window.removeEventListener("pointercancel", stop);
    };
  }, [panning]);

  return (
    <div className="exec-view">
      {viz.execution.degraded && (
        <div className="view-banner">
          Step ordering is derived from coarse timestamps because this trace has no explicit execution event stream.
        </div>
      )}
      {!hasSteps && (
        <div className="view-banner">
          This trial has no executed steps yet — phase columns are shown empty. Switch to State Machine to inspect program structure.
        </div>
      )}
      <div
        ref={scrollRef}
        className={`exec-scroll ${panning ? "is-panning" : ""}`}
        onPointerDown={onPointerDown}
      >
        <div
          className="exec-canvas"
          style={{ width: `${geometry.totalWidth}px`, height: `${geometry.totalHeight}px` }}
        >
          <svg
            className="exec-canvas__edges"
            width={geometry.totalWidth}
            height={geometry.totalHeight}
          >
            {viz.execution.transitions.map((transition) => {
              const source = geometry.boxes.get(transition.source_step_id);
              const target = geometry.boxes.get(transition.target_step_id);
              if (!source || !target) return null;
              const active =
                selectionMatches(selected, "step", transition.source_step_id)
                || selectionMatches(selected, "step", transition.target_step_id)
                || selectionMatches(hovered, "step", transition.source_step_id)
                || selectionMatches(hovered, "step", transition.target_step_id);

              const isCross = transition.kind === "subgraph";
              const sameColumn = source.x === target.x;
              const startX = sameColumn ? source.x + source.width / 2 : source.x + source.width;
              const startY = sameColumn ? source.y + source.height : source.y + source.height / 2;
              const endX = sameColumn ? target.x + target.width / 2 : target.x;
              const endY = sameColumn ? target.y : target.y + target.height / 2;

              let path: string;
              if (sameColumn) {
                const midY = (startY + endY) / 2;
                path = `M ${startX} ${startY} C ${startX} ${midY}, ${endX} ${midY}, ${endX} ${endY}`;
              } else {
                const bend = Math.max((endX - startX) * 0.45, 40);
                path = `M ${startX} ${startY} C ${startX + bend} ${startY}, ${endX - bend} ${endY}, ${endX} ${endY}`;
              }

              const stroke = active
                ? VIZ.accent
                : isCross
                  ? "rgba(255, 175, 105, 0.6)"
                  : "rgba(102, 171, 255, 0.32)";

              return (
                <path
                  key={transition.id}
                  d={path}
                  fill="none"
                  stroke={stroke}
                  strokeWidth={active ? 2.6 : 1.5}
                  strokeDasharray={transition.kind === "sequence" ? "5 5" : undefined}
                />
              );
            })}
          </svg>

          {lanes.map((lane) => {
            const c = columnIndex.get(lane.id) ?? 0;
            const x = PAD + c * (COL_WIDTH + COL_GAP);
            const laneActive =
              selectionMatches(selected, "subgraph", lane.id)
              || selectionMatches(hovered, "subgraph", lane.id);
            const steps = stepsByLane.get(lane.id) ?? [];
            return (
              <button
                key={lane.id}
                type="button"
                className={`exec-column-header ${lane.executed ? "is-executed" : "is-idle"} ${laneActive ? "is-active" : ""}`}
                style={{
                  left: `${x}px`,
                  top: `${PAD}px`,
                  width: `${COL_WIDTH}px`,
                  height: `${HEADER_HEIGHT - 10}px`,
                }}
                onMouseEnter={() => onHover({ kind: "subgraph", id: lane.id })}
                onMouseLeave={() => onHover(null)}
                onClick={() => onSelect({ kind: "subgraph", id: lane.id })}
              >
                <div className="exec-column-header__title-row">
                  <span className="exec-column-header__index">{c + 1}</span>
                  <span className="exec-column-header__title">{lane.title}</span>
                </div>
                <div className="exec-column-header__meta">
                  <span style={{ color: getAgentTone(lane.agent) }}>
                    {lane.is_end ? "end" : lane.agent || "generic"}
                  </span>
                  <span>{steps.length} step{steps.length === 1 ? "" : "s"}</span>
                </div>
              </button>
            );
          })}

          {viz.execution.steps.map((step) => {
            const box = geometry.boxes.get(step.id);
            if (!box) return null;
            const stateActive = selectionMatches(selected, "state", step.state_id);
            const stepActive = selectionMatches(selected, "step", step.id);
            const stateHover = selectionMatches(hovered, "state", step.state_id);
            const stepHover = selectionMatches(hovered, "step", step.id);
            const active = stateActive || stepActive || stateHover || stepHover;
            return (
              <StepCard
                key={step.id}
                step={step}
                style={{
                  left: `${box.x}px`,
                  top: `${box.y}px`,
                  width: `${box.width}px`,
                  height: `${box.height}px`,
                }}
                isActive={active}
                onSelect={onSelect}
                onHover={onHover}
              />
            );
          })}
        </div>
      </div>
    </div>
  );
}

function StepCard({
  step,
  style,
  isActive,
  onSelect,
  onHover,
}: {
  step: ExecutionStep;
  style: React.CSSProperties;
  isActive: boolean;
  onSelect: (selection: Selection) => void;
  onHover: (selection: Selection | null) => void;
}) {
  return (
    <button
      type="button"
      className={`exec-step ${isActive ? "is-active" : ""}`}
      style={{
        ...style,
        borderColor: isActive ? VIZ.accent : "rgba(170, 205, 250, 0.28)",
      }}
      onMouseEnter={() => onHover({ kind: "step", id: step.id })}
      onMouseLeave={() => onHover(null)}
      onClick={() => onSelect({ kind: "step", id: step.id })}
      title={`${step.title} · ${step.status}`}
    >
      <span className="exec-step__dot" style={{ background: getStatusTone(step.status) }} />
      <span className="exec-step__title">{step.title}</span>
      <span className="exec-step__seq">#{step.sequence}</span>
    </button>
  );
}
