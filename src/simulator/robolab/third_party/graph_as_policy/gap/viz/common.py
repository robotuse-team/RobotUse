"""Shared helpers for the workflow renderers (stdlib only)."""

from __future__ import annotations

import json
from pathlib import Path

# Router values that denote a failure transition (in addition to any edge
# whose destination is an `end` node with status "failure").
FAIL_LABELS = {"failed", "fail", "not_found", "blocked", "aborted", "abort",
               "error", "timeout", "unreachable", "missing", "invalid"}


def load_graph(graph: dict | str | Path) -> dict:
    """Return a workflow dict from a dict, a ``workflow.json`` path, or a
    directory containing one."""
    if isinstance(graph, dict):
        return graph
    path = Path(graph)
    if path.is_dir():
        path = path / "workflow.json"
    return json.loads(path.read_text())
