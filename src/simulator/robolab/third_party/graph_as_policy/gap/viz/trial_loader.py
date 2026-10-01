"""Load trial data from disk for the web viewer.

All on-disk filenames and keys mirror what :class:`gap.runtime.tracing.DagTrace`
writes:

- ``dag_trace.json`` — nodes (null-able ``service``/``method`` fields),
  edges, events
- ``node_data/<name>/resolved_inputs.json`` / ``output.json``
- ``node_data/<name>/assets/`` — extracted images / masks / clouds
- ``node_data/<name>/calls/NNN_<tool>/`` — sub-calls with ``request.json``,
  ``request.meta.json`` ({"tool": ...}), ``response.json``, ``assets/``
- ``node_data/<name>/stream_reads/NNN_<stream>/`` — stream snapshots
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .graph_builder import build_workflow_graph
from .models import NodeTraceData, TrialData, VizTrial, WorkflowGraph
from .view_builder import build_viz_trial

logger = logging.getLogger(__name__)


#: Trial-dir *content* subtrees — a trial directory never nests inside one
#: of these, so the discovery walk prunes them (node_data alone holds
#: thousands of dirs per run; descending into it made every re-scan take
#: seconds on large output trees).
_NON_TRIAL_SUBTREES = frozenset(
    {"node_data", "assets", "calls", "scene_log", "scripts", "checkpoints",
     "codegen", "__pycache__"}
)


def discover_trials(root: Path) -> list[str]:
    """Recursively find trial or workflow directories.

    A directory qualifies if it contains ``dag_trace.json`` (a completed trial)
    or ``workflow.json`` (a static workflow definition with no execution yet).
    Returns paths relative to *root*, sorted lexicographically. Called per
    ``/api/trials`` request (the frontend polls it), so the walk prunes
    trial-content subtrees rather than visiting every file.
    """
    import os

    with_trace: set[str] = set()
    with_workflow: set[str] = set()
    for dirpath, dirnames, filenames in os.walk(root):
        # Codegen dirs hold source copies of workflow.json, not trials.
        dirnames[:] = sorted(
            d for d in dirnames if d not in _NON_TRIAL_SUBTREES
        )
        rel = os.path.relpath(dirpath, root)
        if "dag_trace.json" in filenames:
            with_trace.add(rel)
        if "workflow.json" in filenames:
            with_workflow.add(rel)
    # Preserve the historical ordering contract: completed trials first,
    # then workflow-only dirs, each block sorted lexicographically.
    trials = sorted(with_trace)
    trials += sorted(with_workflow - with_trace)
    return trials


def load_workflow_graph(workflow_dir: Path, tool_registry: object | None = None) -> WorkflowGraph:
    """Load a workflow graph from a workflow directory."""
    from gap.runtime.workflow import load_workflow
    workflow = load_workflow(workflow_dir / "workflow.json")
    return build_workflow_graph(workflow, workflow_dir, tool_registry=tool_registry)


def load_trial(trial_dir: Path, tool_registry: object | None = None) -> TrialData:
    """Load trial data from a trial directory.

    Accepts any of:
      - dag_trace.json + workflow.json (full trial with per-node I/O)
      - workflow.json only (static workflow; execution trace will be empty)
    """
    trace_path = trial_dir / "dag_trace.json"
    workflow_path = trial_dir / "workflow.json"

    if not trace_path.exists() and not workflow_path.exists():
        raise FileNotFoundError(f"No workflow.json or dag_trace.json in {trial_dir}")

    trace_data = _load_trace_json(trial_dir) if trace_path.exists() else {"nodes": [], "events": []}

    if workflow_path.exists():
        from gap.runtime.workflow import load_workflow
        workflow = load_workflow(workflow_path)
        graph = build_workflow_graph(workflow, trial_dir, tool_registry=tool_registry)
    else:
        graph = WorkflowGraph(
            meta={}, begin="", subgraphs=[], states=[],
            control_edges=[], data_edges=[],
        )

    # Build node trace data
    nodes: list[NodeTraceData] = []
    for n in trace_data.get("nodes", []):
        nodes.append(NodeTraceData(
            node_id=n["name"],
            status=n.get("status", "pending"),
            started_at=n.get("started_at", 0.0),
            finished_at=n.get("finished_at", 0.0),
            duration_ms=n.get("duration_ms", 0.0),
            condition_result=n.get("condition_result"),
            error_message=n.get("error_message"),
            has_inputs=n.get("has_inputs", False),
            has_output=n.get("has_output", False),
            assets=n.get("assets", []),
            node_type=n.get("node_type", ""),
            service=n.get("service"),
            method=n.get("method"),
            script=n.get("script"),
        ))

    # Calculate total duration
    started_times = [n.started_at for n in nodes if n.started_at > 0]
    finished_times = [n.finished_at for n in nodes if n.finished_at > 0]
    total_ms = 0.0
    if started_times and finished_times:
        total_ms = (max(finished_times) - min(started_times)) * 1000.0

    return TrialData(
        workflow=graph,
        nodes=nodes,
        total_duration_ms=round(total_ms, 2),
    )


def load_viz_trial(
    trial_dir: Path,
    tool_registry: object | None = None,
    trial_path: str | None = None,
) -> VizTrial:
    """Load the normalized visualization payload for one trial."""
    trial = load_trial(trial_dir, tool_registry=tool_registry)
    trace_path = trial_dir / "dag_trace.json"
    trace_data = _load_trace_json(trial_dir) if trace_path.exists() else {"nodes": [], "events": []}
    viz = build_viz_trial(
        workflow=trial.workflow,
        nodes=trial.nodes,
        raw_trace=trace_data,
        trial_path=trial_path,
        total_duration_ms=trial.total_duration_ms,
    )
    viz.meta.has_scene_log = (trial_dir / "scene_log").is_dir()
    return viz


def _load_trace_json(trial_dir: Path) -> dict[str, Any]:
    """Read dag_trace.json from a trial directory."""
    trace_path = trial_dir / "dag_trace.json"
    if not trace_path.exists():
        raise FileNotFoundError(f"No dag_trace.json in {trial_dir}")

    with open(trace_path) as f:
        return json.load(f)


def load_node_data(trial_dir: Path, node_id: str, filename: str) -> dict | None:
    """Load a JSON file from a node's data directory."""
    path = trial_dir / "node_data" / node_id / filename
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_node_script(workflow_dir: Path, script_path: str) -> str | None:
    """Load a script's source code."""
    path = workflow_dir / script_path
    if not path.exists():
        return None
    return path.read_text()


def list_node_assets(trial_dir: Path, node_id: str) -> list[str]:
    """List asset files for a node."""
    assets_dir = trial_dir / "node_data" / node_id / "assets"
    if not assets_dir.exists():
        return []
    return sorted(f.name for f in assets_dir.iterdir() if f.is_file())


def get_asset_path(trial_dir: Path, node_id: str, filename: str) -> Path | None:
    """Get the full path to a node asset file."""
    path = trial_dir / "node_data" / node_id / "assets" / filename
    if path.exists():
        return path
    return None


# ---------------------------------------------------------------------------
# Sub-call data (tool calls made within skill/script nodes)
# ---------------------------------------------------------------------------


def _split_tool_name(tool: str) -> tuple[str, str]:
    """Split a flat tool name into (namespace, short) for display.

    "robot.move_to_pose" → ("robot", "move_to_pose"); a dotless name has
    an empty namespace. The frontend shows these in the proto-era
    ``service``/``method`` slots.
    """
    if "." in tool:
        ns, _, short = tool.rpartition(".")
        return ns, short
    return "", tool


def list_node_subcalls(trial_dir: Path, node_id: str) -> list[dict]:
    """List tool sub-calls recorded within a skill/script node.

    Each ``calls/NNN_<tool>/request.meta.json`` carries ``{"tool": ...}``
    (the proto-era ``service``/``method`` keys are derived from the flat
    tool name for frontend display).
    """
    calls_dir = trial_dir / "node_data" / node_id / "calls"
    if not calls_dir.exists():
        return []

    subcalls = []
    for call_dir in sorted(calls_dir.iterdir()):
        if not call_dir.is_dir():
            continue
        meta_path = call_dir / "request.meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            tool = meta.get("tool", "")
        else:
            # Calls without inputs persist no request side — recover the
            # tool name from the directory suffix (NNN_<short_tool>).
            tool = call_dir.name.split("_", 1)[1] if "_" in call_dir.name else ""

        service, method = _split_tool_name(tool)
        assets_dir = call_dir / "assets"
        assets = (
            sorted(f.name for f in assets_dir.iterdir() if f.is_file())
            if assets_dir.exists() else []
        )

        subcalls.append({
            "seq": int(call_dir.name.split("_")[0]),
            "dir_name": call_dir.name,
            "tool": tool,
            "method": method,
            "service": service,
            "assets": assets,
        })

    return subcalls


def load_subcall_data(trial_dir: Path, node_id: str, seq: int, filename: str) -> dict | None:
    """Load a JSON file from a sub-call directory."""
    calls_dir = trial_dir / "node_data" / node_id / "calls"
    if not calls_dir.exists():
        return None

    for call_dir in calls_dir.iterdir():
        if call_dir.is_dir() and call_dir.name.startswith(f"{seq:03d}_"):
            path = call_dir / filename
            if path.exists():
                with open(path) as f:
                    return json.load(f)
    return None


def get_subcall_asset_path(
    trial_dir: Path, node_id: str, seq: int, filename: str,
) -> Path | None:
    """Get the full path to a sub-call asset file."""
    calls_dir = trial_dir / "node_data" / node_id / "calls"
    if not calls_dir.exists():
        return None

    for call_dir in calls_dir.iterdir():
        if call_dir.is_dir() and call_dir.name.startswith(f"{seq:03d}_"):
            path = call_dir / "assets" / filename
            if path.exists():
                return path
    return None
