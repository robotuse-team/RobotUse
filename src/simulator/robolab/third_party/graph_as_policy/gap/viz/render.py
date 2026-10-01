"""Render a v3 ``workflow.json`` graph to a compact, paper-ready matplotlib figure.

The v3 schema is a hierarchical state machine:

    {
      "meta": {"name": ..., "description": ...},
      "nodes":  {name: {type, tool?|script?|ref?, inputs, status?}},
      "edges":  [["START", n], [a, b], [n, "END"]],
      "conditional_edges": {src: {router_field, mapping: {label: dst}}},
      "subgraphs": {sg: {skill, nodes, edges, conditional_edges, exit, ...}}
    }

Top-level ``subgraph``-type nodes point at a named subgraph via ``ref``. We
render each subgraph as a horizontal *lane* (its nodes laid out left→right by
topological layer); lanes are stacked top→bottom in execution order, and the
macro conditional transitions are drawn as labelled arrows between lanes. Flat
graphs (no subgraphs) render as a single clean layered DAG.

No graphviz dependency — layout uses networkx topological generations plus a
back-edge-tolerant fallback for graphs with loops (e.g. RefineLoop).

Usage::

    from gap.viz import render
    render("path/to/workflow.json", "fig.pdf")
    render(workflow_dict, "fig.png", data_edges=True, legend=False)
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from .common import FAIL_LABELS, load_graph  # noqa: E402

logger = logging.getLogger(__name__)

# ── style ──────────────────────────────────────────────────────────────────
# Muted, print-friendly palette keyed by node type.
PALETTE = {
    "tool": ("#cfe3f7", "#2c6fb0"),     # tool call               (fill, edge)
    "script": ("#d6efd6", "#3f8f47"),   # python script
    "subgraph": ("#e8e0f5", "#7a5bb0"),  # nested subgraph (macro node)
    "router": ("#fde7c4", "#c8841a"),   # dynamic dispatch
    "noop": ("#ececec", "#9a9a9a"),     # marker / exit node
    "end_success": ("#d6efd6", "#2f7d36"),
    "end_failure": ("#f7d6d6", "#b03434"),
}
LANE_FILL = "#fbfbfd"
LANE_EDGE = "#c8c8d0"
CTRL_COLOR = "#5a5a66"
COND_OK_COLOR = "#2e7d32"    # success / happy-path transitions (green)
COND_FAIL_COLOR = "#c0392b"  # failure / abort transitions (red)
DATA_COLOR = "#b8b8c4"
FONT = "DejaVu Sans"

# geometry (data units)
NODE_W, NODE_H = 1.62, 0.54
COL_PITCH = 1.92          # horizontal pitch between layers inside a lane
ROW_PITCH = 0.88          # vertical pitch between branch rows inside a lane
LANE_PAD_X, LANE_PAD_Y = 0.34, 0.30
LANE_GAP = 0.42           # vertical gap between stacked lanes
TERM_GAP = 0.55           # gap before the shared terminal row
LABEL_GUTTER = 0.55       # left margin (macro arrows bulge into it)
SCALE_IN = 0.42           # inches per data unit (figure sizing)


def _node_style(node: dict) -> tuple[str, str]:
    t = node.get("type", "tool")
    if t == "end":
        return PALETTE["end_failure" if node.get("status") == "failure" else "end_success"]
    return PALETTE.get(t, PALETTE["tool"])


def _node_text(name: str, node: dict) -> tuple[str, str]:
    """Return (name, subtitle) for a node — subtitle says what it does."""
    t = node.get("type")
    sub = ""
    if t == "tool" and node.get("tool"):
        sub = node["tool"].split(".")[-1]
    elif t == "script" and node.get("script"):
        sub = Path(node["script"]).stem
    elif t == "subgraph" and node.get("ref"):
        sub = node["ref"]
    elif t == "end":
        sub = node.get("status", "end")
    return name, sub


# Average glyph advance for DejaVu Sans, in em (≈ fraction of the font size).
_CHAR_EM = 0.56


def _fit_text(text: str, max_pts: float, cap: float, floor: float = 3.6) -> tuple[str, float]:
    """Shrink the font (down to *floor*) so *text* fits *max_pts*; clip with an
    ellipsis only if it still doesn't fit at the floor size."""
    if not text:
        return text, cap
    f = min(cap, max_pts / (len(text) * _CHAR_EM))
    if f < floor:
        f = floor
        maxchars = max(3, int(max_pts / (f * _CHAR_EM)))
        if len(text) > maxchars:
            text = text[: maxchars - 1] + "…"
    return text, f


# ── layering ─────────────────────────────────────────────────────────────────
def _layers(node_names: list[str], edges: list[tuple[str, str]], begin: str) -> dict[str, int]:
    """Assign each node a layer index (longest-path), tolerant of cycles."""
    g = nx.DiGraph()
    g.add_nodes_from(node_names)
    g.add_edges_from((u, v) for u, v in edges if u in node_names and v in node_names)
    try:
        return {n: i for i, gen in enumerate(nx.topological_generations(g)) for n in gen}
    except nx.NetworkXUnfeasible:
        # Break cycles: keep only edges that move forward in a BFS ordering.
        seed = begin if begin in g else (node_names[0] if node_names else None)
        order = list(nx.bfs_tree(g, seed)) if seed else list(g.nodes)
        order += [n for n in g.nodes if n not in order]
        rank = {n: i for i, n in enumerate(order)}
        dag = nx.DiGraph()
        dag.add_nodes_from(g.nodes)
        dag.add_edges_from((u, v) for u, v in g.edges if rank[u] < rank[v])
        return {n: i for i, gen in enumerate(nx.topological_generations(dag)) for n in gen}


def _is_hidden(name: str, node: dict) -> bool:
    """Nodes to omit from the figure entirely (e.g. the `go_home` reset state)."""
    if name == "go_home" or name.startswith("go_home"):
        return True
    return (node.get("tool") or "").split(".")[-1] in ("GoHome", "go_home")


def _bypass_block(block: dict) -> None:
    """Drop hidden nodes from a node-block in place, rewiring control flow
    straight through them (preds → succs) so the graph stays connected."""
    nodes = block.get("nodes", {})
    edges = [tuple(e) for e in block.get("edges", [])]
    ce = block.get("conditional_edges", {})
    for h in [n for n, nd in nodes.items() if _is_hidden(n, nd)]:
        fwd = [w for (x, w) in edges if x == h and w != h]
        if h in ce:
            fwd += [d for d in ce[h].get("mapping", {}).values() if d != h]
        fwd_main = fwd[0] if fwd else "END"
        preds = [u for (u, x) in edges if x == h and u != h]
        kept = [(u, x) for (u, x) in edges if u != h and x != h]
        kept += [(u, w) for u in preds for w in fwd]
        edges = list(dict.fromkeys(kept))            # dedupe, keep order
        ce = {
            src: {"router_field": c.get("router_field"),
                  "mapping": {lbl: (fwd_main if dst == h else dst)
                              for lbl, dst in c.get("mapping", {}).items()}}
            for src, c in ce.items() if src != h
        }
        nodes.pop(h, None)
    block["edges"] = [list(e) for e in edges]
    block["conditional_edges"] = ce


def _strip_hidden(wf: dict) -> dict:
    """Return a copy of *wf* with hidden nodes removed from the top level and
    every subgraph, edges rewired around them."""
    import copy
    wf = copy.deepcopy(wf)
    _bypass_block(wf)
    for sg in wf.get("subgraphs", {}).values():
        _bypass_block(sg)
    return wf


def _control_edges(block: dict) -> list[tuple[str, str, str]]:
    """(src, dst, label) control edges within a node block, sans START/END."""
    out = []
    for u, v in block.get("edges", []):
        if u in ("START", "END") or v in ("START", "END"):
            continue
        out.append((u, v, ""))
    for src, cond in block.get("conditional_edges", {}).items():
        for label, dst in cond.get("mapping", {}).items():
            if dst in ("START", "END"):
                continue
            out.append((src, dst, label))
    return out


def _begin_node(block: dict) -> str:
    for u, v in block.get("edges", []):
        if u == "START":
            return v
    return next(iter(block.get("nodes", {})), "")


# ── lane layout ──────────────────────────────────────────────────────────────
class Lane:
    """A laid-out block of nodes (one subgraph, or one bare top-level node)."""

    def __init__(self, lane_id: str, title: str, block: dict, *, framed: bool):
        self.id = lane_id
        self.title = title
        self.framed = framed
        self.is_term = False
        self.nodes = block.get("nodes", {})
        self.ctrl = _control_edges(block)
        self.begin = _begin_node(block)
        self.pos: dict[str, tuple[float, float]] = {}  # node -> (cx, cy) local

        names = list(self.nodes)
        layer = _layers(names, [(u, v) for u, v, _ in self.ctrl], self.begin)
        cols: dict[int, list[str]] = {}
        for n in names:
            cols.setdefault(layer.get(n, 0), []).append(n)

        ncols = (max(cols) + 1) if cols else 1
        max_rows = max((len(v) for v in cols.values()), default=1)
        for c, members in cols.items():
            members.sort()
            span = (len(members) - 1) * ROW_PITCH
            for r, n in enumerate(members):
                cx = c * COL_PITCH
                cy = span / 2 - r * ROW_PITCH
                self.pos[n] = (cx, cy)

        self.w = max(ncols * COL_PITCH - (COL_PITCH - NODE_W), NODE_W) + 2 * LANE_PAD_X
        self.h = max(max_rows * ROW_PITCH - (ROW_PITCH - NODE_H), NODE_H) + 2 * LANE_PAD_Y
        self.x0 = 0.0  # filled in during placement (lane bottom-left)
        self.y0 = 0.0

    def abs_pos(self, name: str) -> tuple[float, float]:
        cx, cy = self.pos[name]
        # local centre origin is the geometric middle of the node cloud
        return (self.x0 + LANE_PAD_X + NODE_W / 2 + cx,
                self.y0 + self.h / 2 + cy)


# ── main render ──────────────────────────────────────────────────────────────
def render(
    graph: dict | str | Path,
    out: str | Path,
    *,
    data_edges: bool = False,
    legend: bool = True,
    title: str | None = None,
    also_png: bool = True,
) -> Path:
    """Render a v3 workflow graph to a PDF (and optionally a sibling PNG).

    Args:
        graph: A workflow dict, or a path to ``workflow.json`` (or a
            directory containing one).
        out: Output figure path; the suffix selects the format. When
            *also_png* is true and *out* is not already a PNG, a sibling
            ``.png`` is written too.
        data_edges: Overlay faint dashed data-flow ($ref) edges.
        legend: Draw the node-type / route-color legend.
        title: Reserved figure-title override (kept for CLI parity).
        also_png: Write a PNG next to the requested output format.

    Returns:
        The primary output path.
    """
    wf = load_graph(graph)
    out_path = Path(out)
    _render(wf, out_path, data_edges=data_edges, legend=legend,
            title=title, also_png=also_png)
    return out_path


def _render(wf: dict, out_path: Path, *, data_edges: bool, legend: bool,
            title: str | None, also_png: bool = True) -> None:
    wf = _strip_hidden(wf)
    top_nodes: dict = wf.get("nodes", {})
    subgraphs: dict = wf.get("subgraphs", {})

    # Order the top-level nodes by execution layer, then build one Lane each.
    top_ctrl = _control_edges(wf)
    top_layer = _layers(list(top_nodes), [(u, v) for u, v, _ in top_ctrl], _begin_node(wf))
    ordered = sorted(top_nodes, key=lambda n: (top_layer.get(n, 0), n))

    lanes: list[Lane] = []
    lane_of_top: dict[str, Lane] = {}
    for name in ordered:
        node = top_nodes[name]
        if node.get("type") == "subgraph" and node.get("ref") in subgraphs:
            ref = node["ref"]
            sg = subgraphs[ref]
            # Short lane label: the top-level node name (e.g. "grasp"). The
            # skill/ref is available in the data but omitted to keep the
            # gutter uncluttered on compact figures.
            lane = Lane(name, name, sg, framed=True)
            lane.is_term = False
        else:
            # Bare top-level node. `end` terminals stay frameless and pack onto
            # the bottom row; a bare action node (e.g. a top-level tool) gets a
            # frame so it reads as a stage instead of floating.
            is_end = node.get("type") == "end"
            lane = Lane(name, "", {"nodes": {name: node}, "edges": [], "conditional_edges": {}},
                        framed=not is_end)
            lane.is_term = is_end
        lanes.append(lane)
        lane_of_top[name] = lane

    # Flow lanes stack vertically; terminal `end` nodes pack onto one shared
    # bottom row (side by side) so they don't each eat a full lane of height.
    flow = [lane for lane in lanes if not lane.is_term]
    terms = [lane for lane in lanes if lane.is_term]

    y_cursor = 0.0
    for lane in flow:
        lane.x0 = LABEL_GUTTER
        lane.y0 = y_cursor - lane.h
        y_cursor = lane.y0 - LANE_GAP

    if terms:
        term_top = y_cursor - (TERM_GAP - LANE_GAP)
        term_h = max(t.h for t in terms)
        tx = LABEL_GUTTER
        for t in terms:
            t.x0, t.y0 = tx, term_top - term_h + (term_h - t.h) / 2
            tx += t.w + 0.5
        y_cursor = term_top - term_h

    # Figure size derived from the true data extent so aspect='equal' adds no
    # letterbox padding. Left margin holds the macro arrows + their labels.
    xmin, xmax = 0.15, LABEL_GUTTER + max((lane.w for lane in flow), default=NODE_W) + 0.25
    ymin, ymax = y_cursor - 0.15, 0.18   # ymax headroom holds the top legend
    fig, ax = plt.subplots(figsize=((xmax - xmin) * SCALE_IN, (ymax - ymin) * SCALE_IN))

    # Lane frames + titles.
    for lane in lanes:
        if lane.framed:
            ax.add_patch(FancyBboxPatch(
                (lane.x0, lane.y0), lane.w, lane.h,
                boxstyle="round,pad=0.02,rounding_size=0.18",
                linewidth=1.0, edgecolor=LANE_EDGE, facecolor=LANE_FILL, zorder=1))
            # Title sits inside the frame's top-left corner (horizontal), so
            # the left gutter stays free for the macro arrows + their labels.
            ax.text(lane.x0 + 0.14, lane.y0 + lane.h - 0.1, lane.title,
                    ha="left", va="top", fontsize=6.8, family=FONT,
                    color="#6a6a74", zorder=6)

    # Intra-lane control + data edges.
    for lane in lanes:
        if data_edges:
            for tgt, node in lane.nodes.items():
                for val in node.get("inputs", {}).values():
                    ref = _ref_path(val)
                    if not ref:
                        continue
                    src = ref.split(".")[0]
                    if src in lane.pos and src != tgt:
                        _arrow(ax, lane.abs_pos(src), lane.abs_pos(tgt),
                               color=DATA_COLOR, ls=(0, (2, 2)), lw=0.7, rad=0.25, z=2)
        for u, v, label in lane.ctrl:
            if u in lane.pos and v in lane.pos:
                _arrow(ax, lane.abs_pos(u), lane.abs_pos(v),
                       color=CTRL_COLOR, lw=1.3, rad=0.0, z=4, label=label)

    # Nodes.
    for lane in lanes:
        for name, node in lane.nodes.items():
            cx, cy = lane.abs_pos(name)
            _draw_node(ax, cx, cy, name, node)

    # Macro transitions between lanes, routed as a tidy spine in the left
    # gutter. Lanes are stacked top→bottom, so each edge is a (mostly)
    # vertical connector from the bottom of the source lane to the top of the
    # destination lane; edges that skip lanes bulge further left into outer
    # rails so they never cross a lane box.
    out_count: dict[str, int] = {}
    labeled: dict[str, int] = {}     # distinct labels already drawn per dest
    seen_label: set = set()          # (dest, label) already shown
    for u, v, label in top_ctrl:
        if u not in lane_of_top or v not in lane_of_top:
            continue
        sl, dl = lane_of_top[u], lane_of_top[v]
        k = out_count.get(u, 0)                    # fan out multiple exits of u
        out_count[u] = k + 1

        dst_node = top_nodes.get(v, {})
        is_fail = (dst_node.get("type") == "end" and dst_node.get("status") == "failure") \
            or label.lower() in FAIL_LABELS
        color = COND_FAIL_COLOR if is_fail else COND_OK_COLOR

        # Many subgraphs can route the same value (e.g. "failed") into one
        # destination (e.g. abort). Draw each (dest, label) once so the
        # terminal isn't buried under a dozen identical labels.
        show = bool(label) and (v, label.lower()) not in seen_label
        j = labeled.get(v, 0)
        if show:
            seen_label.add((v, label.lower()))
            labeled[v] = j + 1

        # Source: the subgraph's last (exit) node, not the frame.
        if sl.framed and sl.pos:
            ex_cx, ex_cy = sl.abs_pos(_exit_node(sl))
            sp = (ex_cx + 0.10 * k, ex_cy - NODE_H / 2 - 0.03)   # exit-node bottom
        else:
            sp = (sl.x0 + 0.24 + 0.12 * k, sl.y0 + 0.04)

        # Destination: terminal box top, else the destination subgraph's first
        # (begin) node top.
        if dl.is_term:
            cx, cy = dl.abs_pos(next(iter(dl.nodes)))
            dst_c, dst_box = (cx, cy), (NODE_W, NODE_H)
            dp = (cx + 0.12 * k, cy + NODE_H / 2 + 0.03)
        elif dl.framed and dl.pos:
            bx, by = dl.abs_pos(dl.begin or next(iter(dl.nodes)))
            dst_c, dst_box = (bx, by), (NODE_W, NODE_H)
            dp = (bx, by + NODE_H / 2 + 0.03)
        else:
            dst_c, dst_box = (dl.x0 + 0.24, dl.y0 + dl.h - 0.1), (0.3, 0.3)
            dp = (dl.x0 + 0.24, dl.y0 + dl.h - 0.04)

        dx, dy = dp[0] - sp[0], dp[1] - sp[1]
        rad = 0.18 + 0.05 * k                       # bow downward into the gutter
        _arrow(ax, sp, dp, color=color, lw=1.4, rad=rad, z=3, shrink=2)
        if show:
            # Overlay the label ON the arc, just before it reaches the
            # destination — a point on the curve in the structured area (above
            # the terminal / entering the next stage), clamped to the
            # destination half so it never drifts to the empty middle.
            mx, my = (sp[0] + dp[0]) / 2, (sp[1] + dp[1]) / 2
            c = (mx - rad * dy, my + rad * dx)       # arc3 control point

            def _bez(t, sp=sp, dp=dp, c=c):
                return ((1 - t) ** 2 * sp[0] + 2 * (1 - t) * t * c[0] + t ** 2 * dp[0],
                        (1 - t) ** 2 * sp[1] + 2 * (1 - t) * t * c[1] + t ** 2 * dp[1])

            lx, ly = _bez(0.7)
            for cand in [0.86 - 0.035 * i for i in range(12)]:
                bx, by = _bez(cand)
                if (abs(bx - dst_c[0]) > dst_box[0] / 2 + 0.12
                        or abs(by - dst_c[1]) > dst_box[1] / 2 + 0.12):
                    # Small stagger by distinct-label index so co-located
                    # labels sit side by side near the destination.
                    lx, ly = _bez(max(cand - 0.05 * k - 0.05 * j, 0.5))
                    break
            ax.text(lx, ly, label, ha="center", va="center", fontsize=5.4,
                    family=FONT, color=color, zorder=7,
                    bbox=dict(boxstyle="round,pad=0.04", fc="white", ec="none", alpha=0.85))

    if legend:
        _legend(ax)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    outs = [out_path]
    if also_png and out_path.suffix.lower() != ".png":
        outs.append(out_path.with_suffix(".png"))
    for p in outs:
        dpi = 220 if p.suffix.lower() == ".png" else None
        fig.savefig(p, bbox_inches="tight", pad_inches=0.01, dpi=dpi)
    plt.close(fig)
    logger.info("wrote %s (%d lanes, %d nodes)",
                ", ".join(str(p) for p in outs),
                len(lanes), sum(len(lane.nodes) for lane in lanes))


def _exit_node(lane: Lane) -> str:
    """Last node of a lane (highest layer) to anchor outgoing macro edges."""
    if not lane.pos:
        return next(iter(lane.nodes), "")
    return max(lane.pos, key=lambda n: lane.pos[n][0])


def _ref_path(val) -> str | None:
    if isinstance(val, dict) and "$ref" in val:
        return val["$ref"]
    return None


def _draw_node(ax, cx, cy, name, node):
    fill, edge = _node_style(node)
    style = "round,pad=0.02,rounding_size=0.12"
    if node.get("type") == "noop":
        w, h = NODE_W * 0.78, NODE_H * 0.78
    else:
        w, h = NODE_W, NODE_H
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h, boxstyle=style,
        linewidth=1.2, edgecolor=edge, facecolor=fill, zorder=5))

    nm, sub = _node_text(name, node)
    # Usable inner width in points, so each line is auto-fit (and only clipped
    # as a last resort) — keeps long subtitles inside the box.
    max_pts = w * SCALE_IN * 72 * 0.86
    if sub:
        nm, f_nm = _fit_text(nm, max_pts, 6.4)
        sub_s, f_sub = _fit_text(f"({sub})", max_pts, 5.4)
        ax.text(cx, cy + 0.105, nm, ha="center", va="center", fontsize=f_nm,
                family=FONT, color="#1c1c22", zorder=6)
        ax.text(cx, cy - 0.105, sub_s, ha="center", va="center", fontsize=f_sub,
                family=FONT, color="#555560", zorder=6)
    else:
        nm, f_nm = _fit_text(nm, max_pts, 6.6)
        ax.text(cx, cy, nm, ha="center", va="center", fontsize=f_nm,
                family=FONT, color="#1c1c22", zorder=6)


def _arrow(ax, p0, p1, *, color, lw, rad, z, ls="-", label="", macro=False,
           label_at="mid", shrink=None, label_dy=0.0):
    sh = shrink if shrink is not None else NODE_H * 11
    arrow = FancyArrowPatch(
        p0, p1, connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-|>", mutation_scale=10, lw=lw, ls=ls,
        color=color, zorder=z, shrinkA=sh, shrinkB=sh,
        capstyle="round", joinstyle="round")
    ax.add_patch(arrow)
    if label:
        if label_at == "start":
            lx, ly = p0[0] - 0.12, p0[1] - 0.18 + label_dy
            ha = "right"
        else:
            lx, ly = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2 + (0.22 if macro else 0.18)
            ha = "center"
        ax.text(lx, ly, label, ha=ha, va="center", fontsize=6,
                family=FONT, color=color, zorder=z + 1,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))


def _legend(ax):
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
    keys = [("tool", "tool call"), ("script", "script"),
            ("subgraph", "subgraph"), ("end_success", "success"),
            ("end_failure", "failure")]
    handles = [mpatches.Patch(facecolor=PALETTE[k][0], edgecolor=PALETTE[k][1], label=lbl)
               for k, lbl in keys]
    handles += [
        Line2D([0], [0], color=COND_OK_COLOR, lw=1.6, label="success route"),
        Line2D([0], [0], color=COND_FAIL_COLOR, lw=1.6, label="failure route"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=6.5, frameon=False,
              ncol=len(handles), handlelength=1.3, columnspacing=1.0,
              bbox_to_anchor=(1.0, 1.0))
