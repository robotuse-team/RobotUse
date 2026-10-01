"""Render a v3 ``workflow.json`` graph as Unicode box-drawing terminal text.

Each top-level subgraph node becomes a boxed *card* on a vertical
control-flow spine; the row under a card shows the spine transition on the
left (``│ done``) and the off-spine transitions right-aligned
(``failed ▶ ✗ abort``). Loops render as ``↺ target`` annotations, end nodes
collapse into a ✓/✗ footer, and branches that leave the spine continue as
their own segments below. Unlike the static figure renderer this is an
honest, unfiltered view — no nodes are hidden.

Pure stdlib, no third-party imports. Colors are raw ANSI SGR codes applied
after layout, so they never affect alignment; output is plain text unless
``color=True``.

Usage::

    from gap.viz import to_text
    print(to_text("path/to/workflow.json"))
    print(to_text(workflow_dict, color=True, width=60))
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .common import FAIL_LABELS, load_graph

__all__ = ["to_text"]

# SGR codes mirroring the figure renderer's palette semantics.
_BOLD = "1"
_DIM = "90"
_RED = "31"      # failure routes / failure ends
_GREEN = "32"    # scripts, success routes
_YELLOW = "33"   # routers
_BLUE = "34"     # tool identifiers
_MAGENTA = "35"  # skill names

_TYPE_SGR = {"tool": _BLUE, "script": _GREEN, "router": _YELLOW}

_DEFAULT_WIDTH = 76
_MIN_WIDTH = 24
_MEASURE = 10**6  # first-pass width: nothing wraps, nothing truncates

# One run of styled text: (text, SGR code or None). Width math always sums
# len(text); ANSI is injected only in the final flatten pass.
Seg = tuple[str, "str | None"]


@dataclass(frozen=True)
class _Edge:
    label: str  # "" for plain (unlabelled) edges
    dst: str


@dataclass(frozen=True)
class _Exit:
    """How a card leaves the spine."""

    kind: str  # "continue" | "loop" | "end" | "none"
    primary: _Edge | None
    off: tuple[_Edge, ...]  # non-primary transitions, original order


@dataclass(frozen=True)
class _Segment:
    intro: str | None  # "(from src ▶ label)" / "(unreachable)"; None = main spine
    cards: tuple[str, ...]
    exits: tuple[_Exit, ...]  # one per card


def to_text(workflow: dict | str | Path, *, color: bool = False,
            width: int | None = None) -> str:
    """Render a v3 workflow as box-drawing terminal text.

    Args:
        workflow: A workflow dict, a ``workflow.json`` path, or a directory
            containing one.
        color: Emit ANSI SGR colors (default: plain text).
        width: Cap on output width in columns (default 76, floor 24).

    Returns:
        The rendering — no trailing newline, no ANSI unless *color*.
    """
    wf = load_graph(workflow)
    segments = _segments(wf)
    cap = _DEFAULT_WIDTH if width is None else max(_MIN_WIDTH, width)
    w = max(_MIN_WIDTH, min(cap, _natural_width(wf, segments)))
    return _flatten(_render_lines(wf, segments, w), color)


# ── graph traversal ──────────────────────────────────────────────────────────
def _is_end(nodes: dict, name: str) -> bool:
    return nodes.get(name, {}).get("type") == "end"


def _is_fail_end(nodes: dict, name: str) -> bool:
    node = nodes.get(name, {})
    return node.get("type") == "end" and node.get("status") == "failure"


def _is_fail(label: str, dst: str, nodes: dict) -> bool:
    return label.lower() in FAIL_LABELS or _is_fail_end(nodes, dst)


def _out_edges(block: dict, src: str) -> list[_Edge]:
    """Transitions out of *src*: plain edges (list order), then conditional
    mapping items (dict order)."""
    out = [_Edge("", v) for u, v in block.get("edges", [])
           if u == src and v not in ("START", "END")]
    cond = block.get("conditional_edges", {}).get(src, {})
    out += [_Edge(label, dst) for label, dst in cond.get("mapping", {}).items()
            if dst not in ("START", "END")]
    return out


def _begin(wf: dict) -> str | None:
    for u, v in wf.get("edges", []):
        if u == "START":
            return v
    for name, node in wf.get("nodes", {}).items():
        if node.get("type") != "end":
            return name
    return None


def _classify_exit(transitions: list[_Edge], nodes: dict,
                   visited: set[str]) -> _Exit:
    """Pick the spine transition: prefer a happy-path edge to a fresh node,
    then a happy-path loop, then a success end, then whatever is left."""
    if not transitions:
        return _Exit("none", None, ())
    happy = [t for t in transitions
             if not _is_end(nodes, t.dst) and t.label.lower() not in FAIL_LABELS]
    primary, kind = None, ""
    for t in happy:
        if t.dst not in visited:
            primary, kind = t, "continue"
            break
    if primary is None and happy:
        primary, kind = happy[0], "loop"
    if primary is None:
        for t in transitions:
            if _is_end(nodes, t.dst) and not _is_fail_end(nodes, t.dst):
                primary, kind = t, "end"
                break
    if primary is None:  # everything is fail-labelled and/or a failure end
        primary = transitions[0]
        if _is_end(nodes, primary.dst):
            kind = "end"
        else:
            kind = "loop" if primary.dst in visited else "continue"
    off = tuple(t for t in transitions if t is not primary)
    return _Exit(kind, primary, off)


def _segments(wf: dict) -> list[_Segment]:
    """Walk the top-level graph into spine segments (end nodes are never
    cards). The main segment starts at START's target; remaining non-end
    nodes become orphan segments introduced by their first feeding edge."""
    nodes = wf.get("nodes", {})
    visited: set[str] = set()
    segments: list[_Segment] = []

    def walk(start: str, intro: str | None) -> None:
        cards: list[str] = []
        exits: list[_Exit] = []
        u = start
        while True:
            cards.append(u)
            visited.add(u)
            ex = _classify_exit(_out_edges(wf, u), nodes, visited)
            exits.append(ex)
            if ex.kind != "continue":
                break
            u = ex.primary.dst
        segments.append(_Segment(intro, tuple(cards), tuple(exits)))

    start = _begin(wf)
    if start is not None and start in nodes and not _is_end(nodes, start):
        walk(start, None)

    while True:
        nxt: tuple[str, str] | None = None
        for name in nodes:
            if name in visited or _is_end(nodes, name):
                continue
            intro = _feeding_edge(wf, name, visited)
            if intro is not None:
                nxt = (name, intro)
                break
        if nxt is None:
            for name in nodes:
                if name not in visited and not _is_end(nodes, name):
                    nxt = (name, "(unreachable)")
                    break
        if nxt is None:
            break
        walk(*nxt)
    return segments


def _feeding_edge(wf: dict, target: str, visited: set[str]) -> str | None:
    """Intro text for the first transition into *target* from a visited node
    (or START), or None."""
    for src in ["START", *(n for n in wf.get("nodes", {}) if n in visited)]:
        for t in _out_edges(wf, src):
            if t.dst == target:
                return f"(from {src} ▶ {t.label})" if t.label else f"(from {src})"
    return None


# ── text helpers ─────────────────────────────────────────────────────────────
def _seg_len(segs: list[Seg]) -> int:
    return sum(len(t) for t, _ in segs)


def _trunc(s: str, n: int) -> str:
    return s if len(s) <= n else s[: max(n - 1, 0)] + "…"


def _trunc_segs(segs: list[Seg], n: int) -> list[Seg]:
    if _seg_len(segs) <= n:
        return segs
    out: list[Seg] = []
    used = 0
    for text, sgr in segs:
        if used + len(text) <= n - 1:
            out.append((text, sgr))
            used += len(text)
        else:
            take = n - 1 - used
            if take > 0:
                out.append((text[:take], sgr))
            out.append(("…", sgr))
            break
    return out


# ── cards ────────────────────────────────────────────────────────────────────
def _card_title(card: str, node: dict, wf: dict) -> tuple[str, str, str]:
    """(name, right-side caption, caption SGR) for a card header."""
    if node.get("type") == "subgraph":
        ref = node.get("ref")
        sg = wf.get("subgraphs", {}).get(ref)
        right = sg.get("skill", "") if sg is not None else (ref or "subgraph")
        return card, right, _MAGENTA
    t = node.get("type", "")
    return card, t, _TYPE_SGR.get(t, _DIM)


def _card_header(name: str, right: str, right_sgr: str, w: int) -> list[Seg]:
    # "┌─ name ──…── right ─┐", dashes padded so the line is exactly w wide.
    dashes = w - (len(name) + len(right) + 8)
    if right and dashes < 1:
        right = _trunc(right, w - len(name) - 9)
        dashes = w - (len(name) + len(right) + 8)
    if not right or dashes < 1:
        name = _trunc(name, w - 6)
        return [("┌─ ", _DIM), (name, _BOLD),
                (" " + "─" * (w - len(name) - 5) + "┐", _DIM)]
    return [("┌─ ", _DIM), (name, _BOLD), (" " + "─" * dashes + " ", _DIM),
            (right, right_sgr), (" ─┐", _DIM)]


def _annot_line(name: str, glyph: str, ident: str, sgr: str,
                inner: int) -> list[Seg]:
    """One body line: ``name ⚙ ident`` (ident truncated to fit)."""
    if len(name) + 4 > inner:
        return [(_trunc(name, inner), None)]
    return [(name, None), (" ", None), (glyph, _DIM), (" ", None),
            (_trunc(ident, inner - len(name) - 3), sgr)]


def _action_line(name: str, node: dict, inner: int, *,
                 full_tool: bool) -> list[Seg]:
    t = node.get("type")
    if t == "tool" and node.get("tool"):
        ident = node["tool"]
        if not full_tool or len(name) + 3 + len(ident) > inner:
            ident = ident.split(".")[-1]
        return _annot_line(name, "⚙", ident, _BLUE, inner)
    if t == "script" and node.get("script"):
        return _annot_line(name, "ƒ", Path(node["script"]).stem, _GREEN, inner)
    if t == "router":
        return _annot_line(name, "⎇", "router", _YELLOW, inner)
    return [(_trunc(name, inner), None)]


def _topo(names: list[str], edges: list[tuple[str, str]]) -> list[str]:
    """Kahn topological order; ties broken by insertion order in *names*;
    cycle leftovers appended in insertion order."""
    index = {n: i for i, n in enumerate(names)}
    adj: dict[str, list[str]] = {n: [] for n in names}
    indeg = {n: 0 for n in names}
    for u, v in dict.fromkeys(edges):  # dedupe, keep order
        if u in index and v in index and u != v:
            adj[u].append(v)
            indeg[v] += 1
    ready = [n for n in names if indeg[n] == 0]
    order: list[str] = []
    while ready:
        ready.sort(key=index.__getitem__)
        n = ready.pop(0)
        order.append(n)
        for m in adj[n]:
            indeg[m] -= 1
            if indeg[m] == 0:
                ready.append(m)
    placed = set(order)
    return order + [n for n in names if n not in placed]


def _control_pairs(block: dict) -> list[tuple[str, str]]:
    pairs = [(u, v) for u, v in block.get("edges", [])
             if u not in ("START", "END") and v not in ("START", "END")]
    for src, cond in block.get("conditional_edges", {}).items():
        pairs += [(src, dst) for dst in cond.get("mapping", {}).values()
                  if dst not in ("START", "END")]
    return pairs


def _chain_lines(order: list[str], direct: set[tuple[str, str]],
                 inner: int) -> list[list[Seg]]:
    """Join topo-ordered nodes with ``─▶`` (direct edge) or ``·`` (parallel),
    greedily wrapped at separators."""
    def nm(n: str) -> str:
        return _trunc(n, max(inner - 5, 3))

    lines: list[list[Seg]] = []
    cur: list[Seg] = [(nm(order[0]), None)]
    cur_len = _seg_len(cur)
    for prev, n in zip(order, order[1:], strict=False):
        arrow = (prev, n) in direct
        sep = " ─▶ " if arrow else " · "
        text = nm(n)
        if cur_len + len(sep) + len(text) <= inner:
            cur += [(sep, _DIM), (text, None)]
            cur_len += len(sep) + len(text)
        else:
            lines.append(cur)
            cur = [("  ", None), ("─▶ " if arrow else "· ", _DIM), (text, None)]
            cur_len = _seg_len(cur)
    lines.append(cur)
    return lines


def _sg_body(sg: dict, inner: int) -> list[list[Seg]]:
    nodes = sg.get("nodes", {})
    visible = [n for n, nd in nodes.items() if nd.get("type") != "noop"]
    pairs = _control_pairs(sg)
    if not visible:
        return []
    if len(visible) == 1:
        lines = [_action_line(visible[0], nodes[visible[0]], inner,
                              full_tool=True)]
    else:
        order = _topo(visible, [(u, v) for u, v in pairs
                                if u in visible and v in visible])
        lines = _chain_lines(order, set(pairs), inner)
    # Conditional branches between visible nodes (mappings to noop markers
    # or END are exit labels and already shown on the macro edge rows).
    for src, cond in sg.get("conditional_edges", {}).items():
        items = [(label, dst) for label, dst in cond.get("mapping", {}).items()
                 if dst in visible]
        if not items:
            continue
        segs: list[Seg] = [("  ↳ ", _DIM), (f"{src}: ", None)]
        for i, (label, dst) in enumerate(items):
            if i:
                segs.append((", ", _DIM))
            segs += [(f"{label} ", None), ("▶ ", _DIM), (dst, None)]
        lines.append(_trunc_segs(segs, inner))
    return lines


def _body_lines(card: str, node: dict, wf: dict, inner: int) -> list[list[Seg]]:
    if node.get("type") == "subgraph":
        sg = wf.get("subgraphs", {}).get(node.get("ref"))
        return _sg_body(sg, inner) if sg is not None else []
    if node.get("type") in ("tool", "script", "router"):
        return [_action_line(card, node, inner, full_tool=False)]
    return []


def _card_lines(card: str, wf: dict, w: int) -> list[list[Seg]]:
    node = wf.get("nodes", {})[card]
    name, right, right_sgr = _card_title(card, node, wf)
    inner = w - 4
    lines = [_card_header(name, right, right_sgr, w)]
    for content in _body_lines(card, node, wf, inner):
        pad = inner - _seg_len(content)
        lines.append([("│ ", _DIM), *content, (" " * pad, None), (" │", _DIM)])
    lines.append([("└" + "─" * (w - 2) + "┘", _DIM)])
    return lines


# ── edge rows ────────────────────────────────────────────────────────────────
def _end_mark(nodes: dict, dst: str) -> str:
    return "✗" if _is_fail_end(nodes, dst) else "✓"


def _left_segs(ex: _Exit, nodes: dict) -> list[Seg]:
    """The spine part of an edge row: ``  │ label`` / `` ↺ dst`` / `` ▶ ✓ dst``."""
    segs: list[Seg] = [("  │", _DIM)]
    t = ex.primary
    if t is None:
        return segs
    color = _RED if _is_fail(t.label, t.dst, nodes) else _GREEN
    if ex.kind == "continue":
        if t.label:
            segs.append((" " + t.label, color))
    elif ex.kind == "loop":
        segs.append(((" " + t.label if t.label else "") + " ↺ " + t.dst, color))
    elif ex.kind == "end":
        if t.label:
            segs.append((" " + t.label, color))
        segs += [(" ", None), ("▶ ", _DIM),
                 (f"{_end_mark(nodes, t.dst)} {t.dst}", color)]
    return segs


def _off_item(t: _Edge, nodes: dict) -> list[Seg]:
    color = _RED if _is_fail(t.label, t.dst, nodes) else _GREEN
    segs: list[Seg] = [(t.label + " ", color)] if t.label else []
    segs.append(("▶ ", _DIM))
    dst = (f"{_end_mark(nodes, t.dst)} {t.dst}" if _is_end(nodes, t.dst)
           else t.dst)
    segs.append((dst, color))
    return segs


def _edge_rows(ex: _Exit, nodes: dict, w: int) -> list[list[Seg]]:
    """Rows between a card and the next element: spine label left, off-spine
    transitions right-aligned (3-space joins), overflow on extra rows."""
    if ex.kind == "none":
        return []
    items = [_off_item(t, nodes) for t in ex.off]
    rows: list[list[Seg]] = []
    cur: list[Seg] = _left_segs(ex, nodes)
    i = 0
    while True:
        lw = _seg_len(cur)
        batch: list[list[Seg]] = []
        bw = 0
        while i < len(items):
            need = _seg_len(items[i]) + (3 if batch else 0)
            if lw + 2 + bw + need > w:
                break
            batch.append(items[i])
            bw += need
            i += 1
        if not batch and i < len(items):  # lone item too wide: truncate
            batch = [_trunc_segs(items[i], max(w - lw - 2, 1))]
            bw = _seg_len(batch[0])
            i += 1
        row = list(cur)
        if batch:
            row.append((" " * (w - lw - bw), None))
            for j, item in enumerate(batch):
                if j:
                    row.append(("   ", None))
                row += item
        rows.append(row)
        if i >= len(items):
            return rows
        cur = [("  │", _DIM)] if ex.kind == "continue" else [("   ", None)]


# ── header / footer / assembly ───────────────────────────────────────────────
def _footer_items(nodes: dict) -> list[Seg]:
    items: list[Seg] = []
    for name, node in nodes.items():
        if node.get("type") != "end":
            continue
        if node.get("status") == "failure":
            tools = [r["tool"].split(".")[-1]
                     for r in node.get("recovery", ()) if r.get("tool")]
            suffix = f", recovery: {', '.join(tools)}" if tools else ""
            items.append((f"✗ {name} (failure{suffix})", _RED))
        else:
            items.append((f"✓ {name} (success)", _GREEN))
    return items


def _footer_lines(nodes: dict, w: int) -> list[list[Seg]]:
    lines: list[list[Seg]] = []
    cur: list[Seg] = []
    for text, sgr in _footer_items(nodes):
        text = _trunc(text, w)
        if cur and _seg_len(cur) + 3 + len(text) > w:
            lines.append(cur)
            cur = []
        if cur:
            cur.append(("   ", None))
        cur.append((text, sgr))
    if cur:
        lines.append(cur)
    return lines


def _natural_width(wf: dict, segments: list[_Segment]) -> int:
    """Width at which nothing wraps or truncates (description excluded — it
    is a paragraph and always truncates to the final width)."""
    nodes = wf.get("nodes", {})
    needs = [len("START"), len(wf.get("meta", {}).get("name", ""))]
    needs += [len(text) for text, _ in _footer_items(nodes)]
    for seg in segments:
        if seg.intro:
            needs.append(len(seg.intro))
        for card, ex in zip(seg.cards, seg.exits, strict=False):
            node = nodes[card]
            name, right, _ = _card_title(card, node, wf)
            needs.append(len(name) + len(right) + 9 if right else len(name) + 6)
            for line in _body_lines(card, node, wf, _MEASURE):
                needs.append(_seg_len(line) + 4)
            if ex.kind != "none":
                lw = _seg_len(_left_segs(ex, nodes))
                if ex.off:
                    items = [_seg_len(_off_item(t, nodes)) for t in ex.off]
                    lw += 2 + sum(items) + 3 * (len(items) - 1)
                needs.append(lw)
    return max(needs)


def _render_lines(wf: dict, segments: list[_Segment], w: int) -> list[list[Seg]]:
    nodes = wf.get("nodes", {})
    meta = wf.get("meta", {})
    lines: list[list[Seg]] = []
    name = (meta.get("name") or "").strip()
    desc = " ".join((meta.get("description") or "").split())
    if name:
        lines.append([(name, _BOLD)])
    if desc:
        lines.append([(_trunc(desc, w), _DIM)])
    if lines:
        lines.append([])
    arrow: list[Seg] = [("  ▼", _DIM)]
    for seg in segments:
        if seg.intro is None:
            lines += [[("START", None)], [("  │", _DIM)], arrow]
        else:
            if lines:
                lines.append([])
            lines += [[(seg.intro, _DIM)], arrow]
        for card, ex in zip(seg.cards, seg.exits, strict=False):
            lines += _card_lines(card, wf, w)
            lines += _edge_rows(ex, nodes, w)
            if ex.kind == "continue":
                lines.append([("  ▼", _DIM)])
    footer = _footer_lines(nodes, w)
    if footer:
        lines.append([])
        lines += footer
    return lines


def _flatten(lines: list[list[Seg]], color: bool) -> str:
    out = []
    for segs in lines:
        segs = list(segs)
        while segs:  # rstrip before colorizing, so layout matches plain mode
            text = segs[-1][0].rstrip()
            if text:
                segs[-1] = (text, segs[-1][1])
                break
            segs.pop()
        if color:
            out.append("".join(t if c is None else f"\x1b[{c}m{t}\x1b[0m"
                               for t, c in segs))
        else:
            out.append("".join(t for t, _ in segs))
    return "\n".join(out)
