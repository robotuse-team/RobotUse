"""FastAPI REST API routes for the visualization tool."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, PlainTextResponse

from .models import TrialData, VizTrial, WorkflowGraph

router = APIRouter(prefix="/api")

# Set by the server at startup
_root_dir: Path | None = None
_trial_paths: list[str] = []
_skills_dir: Path | None = None
_tool_registry: Any = None


def configure(
    root_dir: Path | None = None,
    skills: Path | None = None,
) -> None:
    """Configure the API with the root output directory.

    If *skills* is supplied (an open-robot-skills checkout), its bundles register
    their tools so node tooltips show port schemas. Without it, only the
    default ``@tool`` registry is consulted and `robot.*` connector tools
    have no schema info.
    """
    global _root_dir, _trial_paths, _skills_dir, _tool_registry
    _root_dir = root_dir
    _skills_dir = skills
    if root_dir:
        from .trial_loader import discover_trials
        _trial_paths = discover_trials(root_dir)
    # Eagerly create the tool registry for schema introspection
    try:
        from gap_core.tools import default_tool_registry
        _tool_registry = default_tool_registry()
        if skills is not None:
            from gap.skills import load_registry_set
            load_registry_set(skills)
            if hasattr(_tool_registry, "discover_pending"):
                _tool_registry.discover_pending()
    except Exception:
        _tool_registry = None  # tools not available


def _resolve_trial_dir(trial: str | None) -> Path:
    """Resolve a trial query-param to an absolute directory."""
    if _root_dir is None:
        raise HTTPException(404, "No root directory configured")
    if trial is not None:
        d = _root_dir / trial
        if not d.exists():
            raise HTTPException(404, f"Trial not found: {trial}")
        return d
    # No explicit trial — auto-select if only one
    if len(_trial_paths) == 1:
        return _root_dir / _trial_paths[0]
    if len(_trial_paths) == 0:
        raise HTTPException(404, "No trials found in output directory")
    raise HTTPException(400, "Multiple trials available — specify ?trial=<path>")


# ── Discovery ────────────────────────────────────────────────────────

@router.get("/trials")
def get_trials() -> list[str]:
    """Return list of discovered trial paths (relative to root).

    Re-scans the root directory on every call so trials created after
    server startup appear without a restart (the frontend polls this).
    """
    global _trial_paths
    if _root_dir is not None:
        from .trial_loader import discover_trials
        _trial_paths = discover_trials(_root_dir)
    return _trial_paths


# ── Workflow / Trial ─────────────────────────────────────────────────

@router.get("/workflow")
def get_workflow(trial: str | None = Query(None)) -> WorkflowGraph:
    """Return the workflow graph structure."""
    if trial is not None or len(_trial_paths) > 0:
        wdir = _resolve_trial_dir(trial)
    elif _root_dir is not None:
        wdir = _root_dir
    else:
        raise HTTPException(404, "No directory configured")

    from .trial_loader import load_workflow_graph
    try:
        return load_workflow_graph(wdir, tool_registry=_tool_registry)
    except Exception as e:
        raise HTTPException(500, str(e)) from e


@router.get("/trial")
def get_trial(trial: str | None = Query(None)) -> TrialData:
    """Return trial execution data."""
    trial_dir = _resolve_trial_dir(trial)

    from .trial_loader import load_trial
    try:
        return load_trial(trial_dir, tool_registry=_tool_registry)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e
    except Exception as e:
        raise HTTPException(500, str(e)) from e


@router.get("/viz/trial")
def get_viz_trial(trial: str | None = Query(None)) -> VizTrial:
    """Return the normalized visualization payload for one trial."""
    trial_dir = _resolve_trial_dir(trial)

    from .trial_loader import load_viz_trial
    try:
        return load_viz_trial(
            trial_dir,
            tool_registry=_tool_registry,
            trial_path=trial,
        )
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e
    except Exception as e:
        raise HTTPException(500, str(e)) from e


# ── Node data ────────────────────────────────────────────────────────

@router.get("/node/{node_id}/inputs")
def get_node_inputs(node_id: str, trial: str | None = Query(None)) -> dict:
    """Return resolved inputs for a node."""
    trial_dir = _resolve_trial_dir(trial)

    from .trial_loader import load_node_data
    data = load_node_data(trial_dir, node_id, "resolved_inputs.json")
    if data is None:
        raise HTTPException(404, f"No input data for node '{node_id}'")
    return data


@router.get("/node/{node_id}/output")
def get_node_output(node_id: str, trial: str | None = Query(None)) -> Any:
    """Return output for a node."""
    trial_dir = _resolve_trial_dir(trial)

    from .trial_loader import load_node_data
    data = load_node_data(trial_dir, node_id, "output.json")
    if data is None:
        raise HTTPException(404, f"No output data for node '{node_id}'")
    return data


@router.get("/node/{node_id}/request")
def get_node_request(node_id: str, trial: str | None = Query(None)) -> dict:
    """Return the recorded tool request for a node."""
    trial_dir = _resolve_trial_dir(trial)

    from .trial_loader import load_node_data
    data = load_node_data(trial_dir, node_id, "request.json")
    if data is None:
        raise HTTPException(404, f"No request data for node '{node_id}'")
    return data


@router.get("/node/{node_id}/script")
def get_node_script(node_id: str, trial: str | None = Query(None)) -> PlainTextResponse:
    """Return the Python source for a script or router node.

    node_id is fully-qualified: "subgraph_name.node_name".
    """
    if trial is not None or len(_trial_paths) > 0:
        wdir = _resolve_trial_dir(trial)
    elif _root_dir is not None:
        wdir = _root_dir
    else:
        raise HTTPException(404, "No directory configured")

    workflow_path = wdir / "workflow.json"
    if not workflow_path.exists():
        raise HTTPException(404, "No workflow.json found")

    from gap.runtime.workflow import load_workflow
    try:
        workflow = load_workflow(workflow_path)
    except Exception as e:
        raise HTTPException(500, f"Failed to load workflow: {e}") from e

    parts = node_id.split(".", 1)
    if len(parts) != 2:
        raise HTTPException(404, f"Invalid node ID '{node_id}' — expected 'subgraph.node'")

    sg_name, node_name = parts
    sg = workflow.subgraphs.get(sg_name)
    if sg is None:
        raise HTTPException(404, f"Subgraph '{sg_name}' not found")

    node = sg.nodes.get(node_name)
    if node is None:
        raise HTTPException(404, f"Node '{node_id}' not found")

    # Script / router nodes: read source from the workflow directory
    if node.type in ("script", "router"):
        if node.script is None:
            raise HTTPException(404, f"Node '{node_id}' has no script path")
        from .trial_loader import load_node_script
        source = load_node_script(wdir, node.script)
        if source is None:
            raise HTTPException(404, f"Script file not found: {node.script}")
        return PlainTextResponse(source)

    raise HTTPException(404, f"Node '{node_id}' is not a script or router node")


@router.get("/node/{node_id}/assets")
def list_node_assets(node_id: str, trial: str | None = Query(None)) -> list[str]:
    """List available asset files for a node."""
    try:
        trial_dir = _resolve_trial_dir(trial)
    except HTTPException:
        return []

    from .trial_loader import list_node_assets as _list
    return _list(trial_dir, node_id)


@router.get("/node/{node_id}/asset/{filename}")
def get_node_asset(node_id: str, filename: str, trial: str | None = Query(None)) -> FileResponse:
    """Serve an asset file (image, point cloud, etc.)."""
    trial_dir = _resolve_trial_dir(trial)

    from .trial_loader import get_asset_path
    path = get_asset_path(trial_dir, node_id, filename)
    if path is None:
        raise HTTPException(404, f"Asset not found: {filename}")

    return FileResponse(path, media_type=_media_type(path))


def _media_type(path: Path) -> str:
    media_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".json": "application/json",
        ".npy": "application/octet-stream",
        ".npz": "application/octet-stream",
    }
    return media_types.get(path.suffix.lower(), "application/octet-stream")


@router.get("/node/{node_id}/subcalls")
def list_subcalls(node_id: str, trial: str | None = Query(None)) -> list[dict]:
    """List tool sub-calls recorded within a skill/script node."""
    try:
        trial_dir = _resolve_trial_dir(trial)
    except HTTPException:
        return []

    from .trial_loader import list_node_subcalls as _list
    return _list(trial_dir, node_id)


@router.get("/node/{node_id}/subcall/{seq}/request")
def get_subcall_request(node_id: str, seq: int, trial: str | None = Query(None)) -> dict:
    """Get the request JSON for a sub-call."""
    trial_dir = _resolve_trial_dir(trial)

    from .trial_loader import load_subcall_data as _load
    data = _load(trial_dir, node_id, seq, "request.json")
    if data is None:
        raise HTTPException(404, f"Sub-call {seq} request not found for node {node_id}")
    return data


@router.get("/node/{node_id}/subcall/{seq}/response")
def get_subcall_response(node_id: str, seq: int, trial: str | None = Query(None)) -> Any:
    """Get the response JSON for a sub-call."""
    trial_dir = _resolve_trial_dir(trial)

    from .trial_loader import load_subcall_data as _load
    data = _load(trial_dir, node_id, seq, "response.json")
    if data is None:
        raise HTTPException(404, f"Sub-call {seq} response not found for node {node_id}")
    return data


@router.get("/node/{node_id}/subcall/{seq}/asset/{filename:path}")
def get_subcall_asset(
    node_id: str, seq: int, filename: str, trial: str | None = Query(None),
) -> FileResponse:
    """Serve an asset file from a sub-call."""
    trial_dir = _resolve_trial_dir(trial)

    from .trial_loader import get_subcall_asset_path as _get
    path = _get(trial_dir, node_id, seq, filename)
    if path is None:
        raise HTTPException(404, f"Asset not found: {filename}")

    return FileResponse(path, media_type=_media_type(path))


@router.post("/trial/replay3d")
def start_replay3d(trial: str | None = Query(None)):
    """Start a viser 3D replay server for a trial's scene_log data."""
    trial_dir = _resolve_trial_dir(trial)

    if not (trial_dir / "scene_log").is_dir():
        raise HTTPException(
            404,
            "3D replay unavailable for this run: no scene_log/ directory was "
            "recorded. Scene logging is opt-in — attach a "
            "gap.viz.TrialLogger to the run to record robot joints, cameras, "
            "and object poses for replay.",
        )

    try:
        from .replay3d import start_replay_server
        url = start_replay_server(trial_dir, port=8890)
        return {"url": url}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            400, f"3D replay failed to start for this run: {e}"
        ) from e


@router.post("/node/{node_id}/replay")
def replay_node(node_id: str, trial: str | None = Query(None)) -> dict:
    """Re-execute a tool node by reissuing its saved request."""
    trial_dir = _resolve_trial_dir(trial)
    if _tool_registry is None:
        raise HTTPException(400, "No tool registry available — replay disabled")

    from .replay import replay_node as _replay
    try:
        return _replay(trial_dir, node_id, _tool_registry)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e
    except Exception as e:
        raise HTTPException(500, str(e)) from e
