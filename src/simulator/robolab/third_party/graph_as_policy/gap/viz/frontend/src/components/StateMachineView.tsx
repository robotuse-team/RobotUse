import { useCallback, useEffect, useMemo, useState } from "react";
import {
  ReactFlow,
  ReactFlowProvider,
  Background,
  Controls,
  useNodesState,
  useEdgesState,
  type Node,
  type Edge,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import ELK, { type ElkNode, type ElkExtendedEdge } from "elkjs/lib/elk.bundled.js";
import type { Selection, StateSchema, SubgraphSchema, WorkflowGraph } from "../types/viz";
import {
  SubgraphNode,
  type SubgraphNodeData,
  COMPACT_WIDTH,
  COMPACT_HEIGHT,
  END_WIDTH,
  END_HEIGHT,
} from "./SubgraphNode";
import { StateNode, type StateNodeData, stateNodeHeight, stateNodeWidth } from "./StateNode";
import { ControlEdge, type ControlEdgeData } from "./ControlEdge";
import { COLORS } from "../styles/theme";

const nodeTypes = { state: StateNode, subgraph: SubgraphNode };
const edgeTypes = { control: ControlEdge };

const SUBGRAPH_PAD_X = 32;
const SUBGRAPH_PAD_TOP = 54;
const SUBGRAPH_PAD_BOTTOM = 22;

const elk = new ELK();

interface Props {
  workflow: WorkflowGraph;
  selected: Selection | null;
  onSelect: (selection: Selection) => void;
}

async function computeLayout(
  graph: WorkflowGraph,
  expandedSubgraphs: Set<string>,
): Promise<{ nodes: Node[]; edges: Edge[] }> {
  const statesBySubgraph = new Map<string, StateSchema[]>();
  for (const sg of graph.subgraphs) statesBySubgraph.set(sg.subgraph_id, []);
  for (const state of graph.states) {
    const list = statesBySubgraph.get(state.subgraph_id);
    if (list) list.push(state);
  }

  const elkChildren: ElkNode[] = [];
  const elkEdges: ElkExtendedEdge[] = [];

  for (const sg of graph.subgraphs) {
    const sgElkId = `sg_${sg.subgraph_id}`;

    if (sg.is_end) {
      elkChildren.push({ id: sgElkId, width: END_WIDTH, height: END_HEIGHT });
      continue;
    }

    const isExpanded = expandedSubgraphs.has(sg.subgraph_id);
    if (!isExpanded) {
      elkChildren.push({ id: sgElkId, width: COMPACT_WIDTH, height: COMPACT_HEIGHT });
      continue;
    }

    const states = statesBySubgraph.get(sg.subgraph_id) ?? [];
    const childNodes: ElkNode[] = states.map((state) => ({
      id: state.state_id,
      width: stateNodeWidth(state),
      height: stateNodeHeight(state, false),
    }));

    const childEdges: ElkExtendedEdge[] = [];
    for (const ce of graph.control_edges) {
      if (ce.edge_type === "transition") continue;
      const srcInSg = states.find((s) => s.state_id === ce.source);
      const tgtInSg = states.find((s) => s.state_id === ce.target);
      if (srcInSg && tgtInSg) {
        childEdges.push({
          id: `elk_${ce.source}_${ce.target}_${ce.edge_type}`,
          sources: [ce.source],
          targets: [ce.target],
        });
      }
    }

    elkChildren.push({
      id: sgElkId,
      layoutOptions: {
        "elk.algorithm": "layered",
        "elk.direction": "DOWN",
        "elk.spacing.nodeNode": "16",
        "elk.layered.spacing.nodeNodeBetweenLayers": "24",
        "elk.padding": `[top=${SUBGRAPH_PAD_TOP},left=${SUBGRAPH_PAD_X},bottom=${SUBGRAPH_PAD_BOTTOM},right=${SUBGRAPH_PAD_X}]`,
      },
      children: childNodes,
      edges: childEdges,
    });
  }

  for (const ce of graph.control_edges) {
    if (ce.edge_type !== "transition") continue;
    const srcSgName = ce.source.split(".")[0];
    elkEdges.push({
      id: `elk_trans_${ce.source}_${ce.target}`,
      sources: [`sg_${srcSgName}`],
      targets: [`sg_${ce.target}`],
    });
  }

  const elkGraph: ElkNode = {
    id: "root",
    layoutOptions: {
      "elk.algorithm": "force",
      "elk.force.model": "FRUCHTERMAN_REINGOLD",
      "elk.force.iterations": "500",
      "elk.force.repulsivePower": "1",
      "elk.spacing.nodeNode": "90",
      "elk.padding": "[top=32,left=32,bottom=32,right=32]",
    },
    children: elkChildren,
    edges: elkEdges,
  };

  const layoutResult = await elk.layout(elkGraph);

  const rfNodes: Node[] = [];
  const rfEdges: Edge[] = [];

  const sgMap = new Map<string, SubgraphSchema>();
  for (const sg of graph.subgraphs) sgMap.set(sg.subgraph_id, sg);

  for (const elkSg of layoutResult.children ?? []) {
    const sgName = elkSg.id.replace(/^sg_/, "");
    const sg = sgMap.get(sgName);
    if (!sg) continue;

    const isExpanded = !sg.is_end && expandedSubgraphs.has(sgName);

    rfNodes.push({
      id: elkSg.id,
      type: "subgraph",
      position: { x: elkSg.x ?? 0, y: elkSg.y ?? 0 },
      style: isExpanded ? { width: elkSg.width, height: elkSg.height } : undefined,
      data: { subgraph: sg, expanded: isExpanded } satisfies SubgraphNodeData,
      selectable: true,
      draggable: false,
    });

    if (isExpanded) {
      for (const elkState of elkSg.children ?? []) {
        const state = graph.states.find((s) => s.state_id === elkState.id);
        if (!state) continue;
        rfNodes.push({
          id: state.state_id,
          type: "state",
          position: { x: elkState.x ?? 0, y: elkState.y ?? 0 },
          parentId: elkSg.id,
          data: { state, expanded: false } satisfies StateNodeData,
        });
      }
    }
  }

  for (const ce of graph.control_edges) {
    if (ce.edge_type === "transition") {
      const srcSgName = ce.source.split(".")[0];
      const sourceExpanded = expandedSubgraphs.has(srcSgName);
      rfEdges.push({
        id: `ctrl_trans_${ce.source}_${ce.target}`,
        source: sourceExpanded ? ce.source : `sg_${srcSgName}`,
        sourceHandle: "ctrl_out",
        target: `sg_${ce.target}`,
        targetHandle: "ctrl_in",
        type: "control",
        data: { edge_type: ce.edge_type, label: ce.label } satisfies ControlEdgeData,
      });
    } else {
      rfEdges.push({
        id: `ctrl_${ce.source}_${ce.target}_${ce.edge_type}`,
        source: ce.source,
        target: ce.target,
        sourceHandle: "ctrl_out",
        targetHandle: "ctrl_in",
        type: "control",
        data: { edge_type: ce.edge_type, label: ce.label } satisfies ControlEdgeData,
      });
    }
  }

  return { nodes: rfNodes, edges: rfEdges };
}

export function StateMachineView({ workflow, selected, onSelect }: Props) {
  return (
    <ReactFlowProvider>
      <StateMachineViewInner workflow={workflow} selected={selected} onSelect={onSelect} />
    </ReactFlowProvider>
  );
}

function StateMachineViewInner({ workflow, selected, onSelect }: Props) {
  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [allExpanded, setAllExpanded] = useState(false);

  useEffect(() => {
    setExpanded(new Set());
    setAllExpanded(false);
  }, [workflow]);

  const effectiveExpanded = useMemo(() => {
    if (!allExpanded) return expanded;
    return new Set(workflow.subgraphs.filter((sg) => !sg.is_end).map((sg) => sg.subgraph_id));
  }, [allExpanded, expanded, workflow.subgraphs]);

  useEffect(() => {
    let cancelled = false;
    void computeLayout(workflow, effectiveExpanded).then(({ nodes: n, edges: e }) => {
      if (cancelled) return;
      setNodes(n);
      setEdges(e);
    });
    return () => {
      cancelled = true;
    };
  }, [workflow, effectiveExpanded, setNodes, setEdges]);

  const highlightedNodes = useMemo(() => {
    return nodes.map((node) => {
      if (node.type === "subgraph") {
        const active = selected?.kind === "subgraph" && selected.id === node.id.replace(/^sg_/, "");
        return {
          ...node,
          style: active
            ? { ...(node.style ?? {}), outline: `2px solid ${COLORS.controlTransition}`, outlineOffset: 2 }
            : node.style,
        };
      }
      if (node.type === "state") {
        const active = selected?.kind === "state" && selected.id === node.id;
        return {
          ...node,
          style: active
            ? { outline: `2px solid ${COLORS.controlTransition}`, outlineOffset: 2, borderRadius: 10 }
            : undefined,
        };
      }
      return node;
    });
  }, [nodes, selected]);

  const toggleSubgraph = useCallback((sgId: string) => {
    setAllExpanded(false);
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(sgId)) next.delete(sgId);
      else next.add(sgId);
      return next;
    });
  }, []);

  return (
    <div className="statemachine-view">
      <ReactFlow
        nodes={highlightedNodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        fitView
        minZoom={0.1}
        maxZoom={2}
        proOptions={{ hideAttribution: true }}
        onNodeClick={(_, node) => {
          if (node.type === "subgraph") {
            const sgId = node.id.replace(/^sg_/, "");
            const sg = workflow.subgraphs.find((s) => s.subgraph_id === sgId);
            if (sg && !sg.is_end) toggleSubgraph(sgId);
            onSelect({ kind: "subgraph", id: sgId });
          } else if (node.type === "state") {
            onSelect({ kind: "state", id: node.id });
          }
        }}
      >
        <Background color="rgba(160, 195, 245, 0.14)" gap={28} />
        <Controls
          style={{ background: "rgba(22, 38, 62, 0.94)", borderColor: COLORS.nodeBorder }}
          showInteractive={false}
        />
      </ReactFlow>
      <div className="statemachine-toolbar">
        <button
          type="button"
          className="statemachine-toolbar__btn"
          onClick={() => setAllExpanded((v) => !v)}
        >
          {allExpanded ? "Collapse all" : "Expand all"}
        </button>
      </div>
    </div>
  );
}
